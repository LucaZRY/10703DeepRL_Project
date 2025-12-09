"""
DAgger with offline PPO + diffusion expert for CarRacing-v2.

Pipeline:
  - Offline PPO expert (ppo.py) produced:
        carracing_ppo_strong_dataset.npz
    containing:
        obs: (N, 4, 96, 96)  float32 in [0,1]
        actions: (N, 3)

  - Diffusion training + generation produced:
        data/generated_carracing/states.npy  (M, 4*96*96) or (M, state_dim)
        data/generated_carracing/actions.npy (M, 3)

  - This script:
        * builds an OfflineExpert from PPO + diffusion data
        * seeds an imitation dataset with offline samples
        * runs DAgger:
            - student policy interacts with env
            - offline expert labels each visited state (nearest neighbor)
            - aggregate into dataset
            - BC train student on aggregated dataset

Run:
    python dagger_carracing_2.py
"""

import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DAgger] Global device: {DEVICE}")

# --------------------------------------------------
# Paths for offline data
# --------------------------------------------------

PPO_NPZ_PATH = "carracing_ppo_strong_dataset.npz"   # from ppo.py
DIFFUSION_DIR = "data/generated_carracing"          # from generate_synthetic_carracing.py


# --------------------------------------------------
# Env and preprocessing (96x96, 4-frame stack)
# --------------------------------------------------

def make_env(render_mode=None):
    """
    Creates CarRacing-v2 env with:
      - grayscale
      - 96x96 resize
      - frame stack (k=4)
    Observation after wrappers is (96,96,4) uint8.
    We convert to (4,96,96) float32 in [0,1] before feeding to the net.
    """
    env = gym.make("CarRacing-v2", continuous=True, render_mode=render_mode)
    env = GrayScaleObservation(env, keep_dim=True)  # (H, W, 1)
    env = ResizeObservation(env, 96)               # (96, 96, 1)
    env = FrameStack(env, num_stack=4)             # (96, 96, 4)
    return env


def preprocess_obs(obs):
    """
    Convert env observation into shape (4, 96, 96) float32 in [0,1].

    Handles:
    - (96, 96, 4)   from FrameStack (H,W,stack)
    - (4, 96, 96)   already channels-first
    - (4, 96, 96,1) or (1,96,96,4) etc. with extra singleton dims
    - (96, 96, 1)   single grayscale frame -> tile to 4
    - (96, 96)      single grayscale frame -> tile to 4
    """
    arr = np.array(obs)

    # Squeeze extra singleton dims like (4,96,96,1) -> (4,96,96), or (1,96,96,4) -> (96,96,4)
    while arr.ndim > 3 and (arr.shape[0] == 1 or arr.shape[-1] == 1):
        if arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        elif arr.shape[0] == 1:
            arr = arr.squeeze(0)
        else:
            break

    if arr.ndim == 3:
        # Case (H, W, 4): last dim is stack
        if arr.shape[-1] == 4:
            arr = arr.astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))  # -> (4,H,W)

        # Case (4, H, W): already correct
        elif arr.shape[0] == 4:
            arr = arr.astype(np.float32)

        # Case (H, W, 1): single grayscale frame
        elif arr.shape[-1] == 1:
            img = arr[..., 0].astype(np.float32) / 255.0
            arr = np.tile(img[None, ...], (4, 1, 1))

        else:
            # Fallback: treat as single frame
            img = arr.astype(np.float32) / 255.0
            arr = np.tile(img[None, ...], (4, 1, 1))

    elif arr.ndim == 2:
        # Single grayscale frame (H, W)
        img = arr.astype(np.float32) / 255.0
        arr = np.tile(img[None, ...], (4, 1, 1))

    else:
        raise ValueError(f"Unexpected obs shape in preprocess_obs: {arr.shape}")

    return arr  # (4,96,96)


# --------------------------------------------------
# Student policy (CNN)
# --------------------------------------------------

class CNNPolicy(nn.Module):
    """
    Simple CNN policy: input (B, 4, 96, 96) -> actions [steer, gas, brake].
    steer ∈ [-1, 1], gas/brake ∈ [0, 1].
    """

    def __init__(self, obs_channels=4, act_dim=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4),  # -> (32, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),            # -> (64, 10, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),            # -> (64, 8, 8)
            nn.ReLU(),
        )

        self.fc_body = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(256, act_dim)

    def forward(self, x):
        # x: (B,4,96,96)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc_body(x)
        raw = self.fc_out(x)  # (B,3)

        steer = torch.tanh(raw[:, 0:1])       # [-1, 1]
        gas   = torch.sigmoid(raw[:, 1:2])    # [0, 1]
        brake = torch.sigmoid(raw[:, 2:3])    # [0, 1]

        return torch.cat([steer, gas, brake], dim=1)


# --------------------------------------------------
# Imitation dataset
# --------------------------------------------------

class ImitationDataset:
    def __init__(self):
        self.obs = []   # list of np.array (4,96,96)
        self.acts = []  # list of np.array (3,)

    def add(self, obs, act):
        """
        obs: raw env obs or (4,96,96) -> standardized to (4,96,96)
        act: np.array([steer, gas, brake])
        """
        o = preprocess_obs(obs)  # robust conversion
        a = np.array(act, dtype=np.float32)
        self.obs.append(o)
        self.acts.append(a)

    def __len__(self):
        return len(self.obs)

    def sample_batch(self, batch_size, device=DEVICE):
        idxs = np.random.randint(0, len(self.obs), size=batch_size)
        obs_batch = np.stack([self.obs[i] for i in idxs], axis=0)   # (B,4,96,96)
        act_batch = np.stack([self.acts[i] for i in idxs], axis=0)  # (B,3)

        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_batch, dtype=torch.float32, device=device)
        return obs_t, act_t


# --------------------------------------------------
# Offline Expert (PPO + diffusion) via nearest neighbor
# --------------------------------------------------

class OfflineExpert:
    """
    Offline expert backed by PPO + diffusion datasets.

    - Loads PPO dataset from carracing_ppo_strong_dataset.npz
    - Optionally loads diffusion synthetic dataset from data/generated_carracing
    - Stores all states as flattened vectors (4*96*96)
    - get_action(obs) -> nearest neighbor action in L2 sense
    """

    def __init__(self,
                 ppo_npz_path: str,
                 diffusion_dir: str | None = None,
                 img_shape=(4, 96, 96)):
        self.img_shape = img_shape
        C, H, W = img_shape
        self.state_dim = C * H * W

        all_states = []
        all_actions = []

        # 1) Load PPO expert dataset
        if not os.path.exists(ppo_npz_path):
            raise FileNotFoundError(f"PPO dataset npz not found at {ppo_npz_path}")
        ppo_data = np.load(ppo_npz_path)
        ppo_obs = ppo_data["obs"]      # (N, 4, 96, 96) float32 in [0,1]
        ppo_act = ppo_data["actions"]  # (N, 3)

        ppo_obs = ppo_obs.astype(np.float32)
        ppo_act = ppo_act.astype(np.float32)

        N = ppo_obs.shape[0]
        ppo_flat = ppo_obs.reshape(N, -1)  # (N, state_dim)

        all_states.append(ppo_flat)
        all_actions.append(ppo_act)
        print(f"[OfflineExpert] Loaded PPO dataset: {ppo_npz_path}, N={N}")

        # 2) Optionally load diffusion synthetic dataset
        if diffusion_dir is not None:
            states_path = os.path.join(diffusion_dir, "states.npy")
            actions_path = os.path.join(diffusion_dir, "actions.npy")
            if os.path.exists(states_path) and os.path.exists(actions_path):
                diff_states = np.load(states_path)
                diff_actions = np.load(actions_path)
                # diff_states: (M, state_dim) or (num_traj, T, state_dim)
                if diff_states.ndim == 3:
                    num_traj, T, state_dim = diff_states.shape
                    diff_states = diff_states.reshape(num_traj * T, state_dim)
                    diff_actions = diff_actions.reshape(num_traj * T, diff_actions.shape[-1])
                elif diff_states.ndim == 2:
                    state_dim = diff_states.shape[-1]
                else:
                    raise ValueError(f"[OfflineExpert] Unexpected diffusion states shape: {diff_states.shape}")

                if state_dim != self.state_dim:
                    raise ValueError(
                        f"[OfflineExpert] Diffusion state_dim {state_dim} != {self.state_dim}. "
                        f"Check image resolution / flattening."
                    )

                # filter zero rows (if any padding)
                row_is_zero = np.all(np.isclose(diff_states, 0.0), axis=1)
                mask = ~row_is_zero

                diff_states = diff_states[mask].astype(np.float32)
                diff_actions = diff_actions[mask].astype(np.float32)

                all_states.append(diff_states)
                all_actions.append(diff_actions)
                print(f"[OfflineExpert] Loaded diffusion dataset: {states_path}, M={diff_states.shape[0]}")
            else:
                print(f"[OfflineExpert] No diffusion data found in {diffusion_dir}, using PPO only.")

        # 3) Concatenate
        self.states = np.concatenate(all_states, axis=0)  # (K, state_dim)
        self.actions = np.concatenate(all_actions, axis=0)  # (K, 3)
        self.num_samples = self.states.shape[0]
        print(f"[OfflineExpert] Total offline samples: {self.num_samples}")

    def get_action(self, obs):
        """
        obs: raw env obs or (4,96,96) array
        Returns nearest-neighbor action from offline dataset.
        """
        obs_proc = preprocess_obs(obs)  # (4,96,96) float32
        s_flat = obs_proc.reshape(-1).astype(np.float32)  # (state_dim,)

        # brute-force nearest neighbor (can be optimized later)
        diffs = self.states - s_flat[None, :]
        dists = np.sum(diffs * diffs, axis=1)
        idx = int(np.argmin(dists))
        return self.actions[idx].copy()


# --------------------------------------------------
# Data collection routines (DAgger)
# --------------------------------------------------

def collect_dagger_data(env,
                        student: CNNPolicy,
                        expert: OfflineExpert,
                        dataset: ImitationDataset,
                        num_episodes: int,
                        device: str = DEVICE):
    """
    DAgger rollout:
      - Student chooses action to step the environment
      - Offline expert labels the same state with its action (via nearest neighbor)
      - Label is stored in dataset
    """
    student.eval()

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            # Student action
            obs_proc = preprocess_obs(obs)
            obs_tensor = torch.tensor(
                obs_proc, dtype=torch.float32, device=device
            ).unsqueeze(0)  # [1,4,96,96]

            with torch.no_grad():
                student_action = student(obs_tensor).cpu().numpy()[0]

            # Offline expert label
            expert_action = expert.get_action(obs_proc)

            # Step environment with student action
            next_obs, reward, terminated, truncated, info = env.step(student_action)
            done = terminated or truncated
            ep_ret += reward

            # Store expert label
            dataset.add(obs_proc, expert_action)

            obs = next_obs

        print(f"[DAgger] Episode {ep+1}/{num_episodes}, return = {ep_ret:.2f}")


# --------------------------------------------------
# BC training + evaluation
# --------------------------------------------------

def bc_train_epoch(student,
                   dataset: ImitationDataset,
                   optimizer,
                   loss_fn,
                   batch_size: int,
                   device: str = DEVICE):
    student.train()
    if len(dataset) == 0:
        return 0.0

    steps = max(1, len(dataset) // batch_size)
    total_loss = 0.0

    for _ in range(steps):
        obs_batch, act_batch = dataset.sample_batch(batch_size, device=device)
        pred = student(obs_batch)
        loss = loss_fn(pred, act_batch)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / steps


def evaluate_policy(env,
                    policy: CNNPolicy,
                    episodes: int,
                    device: str = DEVICE):
    policy.eval()
    returns = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            obs_proc = preprocess_obs(obs)

            obs_tensor = torch.tensor(
                obs_proc, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_tensor).cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward

        returns.append(ep_ret)
    return float(np.mean(returns))


# --------------------------------------------------
# DAgger config + training loop
# --------------------------------------------------

@dataclass
class DAggerConfig:
    num_iterations: int = 5
    dagger_episodes_per_iter: int = 3
    batch_size: int = 64
    bc_epochs_init: int = 5
    bc_epochs_per_iter: int = 3
    seed_max_samples: int = 50000
    eval_episodes: int = 3
    device: str = DEVICE


def seed_dataset_with_offline_data(dataset: ImitationDataset,
                                   expert: OfflineExpert,
                                   max_samples: int = 50000):
    """
    Seed imitation dataset with a subset of offline PPO + diffusion samples.
    """
    N = expert.num_samples
    idxs = np.arange(N)
    if N > max_samples:
        idxs = np.random.choice(N, size=max_samples, replace=False)

    C, H, W = expert.img_shape
    added = 0
    for idx in idxs:
        s_flat = expert.states[idx]          # (state_dim,)
        a = expert.actions[idx]              # (3,)
        s_img = s_flat.reshape(C, H, W)
        dataset.add(s_img, a)
        added += 1

    print(f"[Seed] Added {added} offline samples into imitation dataset.")


def train_dagger(cfg: DAggerConfig):
    device = cfg.device
    print(f"[DAgger] Using device: {device}")

    env = make_env(render_mode=None)

    # Student on GPU/CPU
    student = CNNPolicy().to(device)

    # from train_student import StudentMLP

    # # Load offline-distilled student
    # student = StudentMLP(
    #     state_dim=36864,
    #     act_dim=3,
    #     hidden_dim=256,
    #     num_hidden_layers=2
    # ).to(device)
    #
    # student.load_state_dict(torch.load(
    #     "/path/to/student_baseline_model.pt",
    #     map_location=device
    # ))

    # Offline expert from PPO + diffusion (NumPy only, stays on CPU)
    expert = OfflineExpert(
        ppo_npz_path=PPO_NPZ_PATH,
        diffusion_dir=DIFFUSION_DIR,
        img_shape=(4, 96, 96),
    )

    dataset = ImitationDataset()

    # 0) Seed dataset with offline PPO + diffusion samples
    seed_dataset_with_offline_data(dataset, expert, max_samples=cfg.seed_max_samples)
    print(f"[DAgger] Dataset size after offline seed = {len(dataset)}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    # 1) Warm-start BC training on offline data
    if len(dataset) >= cfg.batch_size:
        print("[DAgger] Warm-start BC training on offline dataset...")
        for epoch in range(cfg.bc_epochs_init):
            avg_loss = bc_train_epoch(
                student, dataset, optimizer, loss_fn,
                cfg.batch_size, device=device
            )
            print(f"[Warm BC] Epoch {epoch+1}/{cfg.bc_epochs_init}, loss = {avg_loss:.4f}")

        avg_return = evaluate_policy(env, student, episodes=cfg.eval_episodes, device=device)
        print(f"[DAgger] After warm-start BC, avg return = {avg_return:.2f}")

    # 2) DAgger iterations: interactive student rollouts + offline expert labels
    for it in range(cfg.num_iterations):
        print(f"\n=== DAgger iteration {it+1}/{cfg.num_iterations} ===")

        # Collect new data under student policy, labeled by offline expert
        collect_dagger_data(
            env=env,
            student=student,
            expert=expert,
            dataset=dataset,
            num_episodes=cfg.dagger_episodes_per_iter,
            device=device,
        )
        print(f"[DAgger] Dataset size after iteration {it+1}: {len(dataset)}")

        # BC on aggregated dataset
        for epoch in range(cfg.bc_epochs_per_iter):
            avg_loss = bc_train_epoch(
                student, dataset, optimizer, loss_fn,
                cfg.batch_size, device=device
            )
            print(
                f"[DAgger BC] Iter {it+1}, epoch {epoch+1}/{cfg.bc_epochs_per_iter}, "
                f"loss = {avg_loss:.4f}"
            )

        avg_return = evaluate_policy(env, student, episodes=cfg.eval_episodes, device=device)
        print(f"[Eval] After DAgger iter {it+1}, avg return = {avg_return:.2f}")

    env.close()
    torch.save(student.state_dict(), "student_dagger_carracing_offline_expert.pt")
    print("\n[DAgger] Training finished. Saved student to student_dagger_carracing_offline_expert.pt")


if __name__ == "__main__":
    cfg = DAggerConfig(
        num_iterations=5,
        dagger_episodes_per_iter=3,
        batch_size=64,
        bc_epochs_init=5,
        bc_epochs_per_iter=3,
        seed_max_samples=50000,
        eval_episodes=3,
        device=DEVICE,  # explicitly pass global device
    )
    train_dagger(cfg)
