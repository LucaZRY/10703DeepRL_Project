"""
dagger_carracing.py

Improved DAgger loop on CarRacing-v2.

- Student: CNN policy mapping stacked frames (4 x 84 x 84) -> [steer, gas, brake]
- Expert: placeholder (currently random). Replace with PPO / diffusion expert.
- Features:
  * Warm-start BC on pure expert rollouts
  * Aggregated dataset across DAgger iterations
  * Correct CarRacing action scaling:
        steer in [-1, 1], gas/brake in [0, 1]

Run:
    conda activate drl-diffdist
    python dagger_carracing.py
"""

import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

from ppo_expert import PPOExpertPolicy


# -----------------------------
#  Env and preprocessing
# -----------------------------

def make_env(render_mode=None):
    """
    Creates CarRacing-v2 env with:
      - grayscale
      - 84x84 resize
      - frame stack (k=4)
    Observation after wrappers is (84, 84, 4) uint8.
    We convert to (4, 84, 84) float32 in [0,1] before feeding to the net.
    """
    env = gym.make("CarRacing-v2", continuous=True, render_mode=render_mode)
    env = GrayScaleObservation(env, keep_dim=True)  # (H, W, 1)
    env = ResizeObservation(env, 84)               # (84, 84, 1)
    env = FrameStack(env, num_stack=4)             # (84, 84, 4)
    return env


def preprocess_obs(obs):
    """
    Convert env observation into shape (4, 84, 84) float32 in [0,1].

    Handles:
    - (84, 84, 4)   from FrameStack (H,W,stack)
    - (4, 84, 84)   already channels-first
    - (4, 84, 84,1) or (1,84,84,4) etc. with extra singleton dims
    - (84, 84, 1)   single grayscale frame -> tile to 4
    - (84, 84)      single grayscale frame -> tile to 4
    """
    arr = np.array(obs)

    # Squeeze extra singleton dims like (4,84,84,1) -> (4,84,84), or (1,84,84,4) -> (84,84,4)
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

    return arr  # (4,84,84)


# -----------------------------
#  Student policy network
# -----------------------------

class CNNPolicy(nn.Module):
    """
    Simple CNN policy: input (B, 4, 84, 84) -> actions [steer, gas, brake].
    steer ∈ [-1, 1], gas/brake ∈ [0, 1].
    """

    def __init__(self, obs_channels=4, act_dim=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4),  # -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),            # -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),            # -> (64, 7, 7)
            nn.ReLU(),
        )

        self.fc_body = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(256, act_dim)

    def forward(self, x):
        # x: (B,4,84,84)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc_body(x)
        raw = self.fc_out(x)  # (B,3)

        steer = torch.tanh(raw[:, 0:1])       # [-1, 1]
        gas   = torch.sigmoid(raw[:, 1:2])    # [0, 1]
        brake = torch.sigmoid(raw[:, 2:3])    # [0, 1]

        return torch.cat([steer, gas, brake], dim=1)


# -----------------------------
#  Expert policy (stub)
# -----------------------------

class ExpertPolicy:
    """
    Placeholder expert: currently random actions.
    Replace `get_action` with PPO/diffusion policy inference.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        # obs is (4,84,84) float32 or raw (84,84,4) from env.
        # For now, just use random actions in env's range.
        # Replace this with:
        #   1) preprocess_obs(obs) to tensor
        #   2) pass through expert model (PPO, diffusion, etc.)
        low  = self.action_space.low
        high = self.action_space.high
        a = np.random.uniform(low, high).astype(np.float32)
        return a


# -----------------------------
#  Imitation dataset
# -----------------------------

class ImitationDataset:
    def __init__(self):
        self.obs = []   # list of np.array (4,84,84)
        self.acts = []  # list of np.array (3,)

    def add(self, obs, act):
        """
        obs: raw env obs (any weird shape) -> we standardize to (4,84,84)
        act: np.array([steer, gas, brake])
        """
        o = preprocess_obs(obs)  # <--- always go through our robust function
        a = np.array(act, dtype=np.float32)
        self.obs.append(o)
        self.acts.append(a)

    def __len__(self):
        return len(self.obs)

    def sample_batch(self, batch_size, device="cpu"):
        idxs = np.random.randint(0, len(self.obs), size=batch_size)
        obs_batch = np.stack([self.obs[i] for i in idxs], axis=0)   # (B,4,84,84)
        act_batch = np.stack([self.acts[i] for i in idxs], axis=0)  # (B,3)

        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_batch, dtype=torch.float32, device=device)
        return obs_t, act_t


# -----------------------------
#  Data collection routines
# -----------------------------

def collect_pure_expert_data(env, expert, dataset, num_episodes=5):
    """
    Collect D0: trajectories where the expert controls the env.
    """
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            # Convert raw obs (84,84,4) to (4,84,84) and call expert.
            expert_action = expert.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(expert_action)
            done = terminated or truncated
            ep_ret += reward

            dataset.add(obs, expert_action)
            obs = next_obs

        print(f"[Expert] Episode {ep+1}/{num_episodes}, return = {ep_ret:.2f}")


def collect_dagger_data(env, student, expert, dataset, num_episodes=5,
                        device="cpu", beta=None):
    """
    DAgger rollout:
      - Student chooses action
      - Expert labels the state with its action
      - Optionally execute a mixture of expert/student actions (beta)
    """
    student.eval()

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            # Preprocess for student
            obs_proc = preprocess_obs(obs)
            obs_tensor = torch.tensor(
                obs_proc, dtype=torch.float32, device=device
            ).unsqueeze(0)  # [1,4,84,84]

            with torch.no_grad():
                student_action = student(obs_tensor).cpu().numpy()[0]

            expert_action = expert.get_action(obs)

            # Execution policy:
            # if beta is not None:
            #     if np.random.rand() < beta:
            #         env_action = expert_action
            #     else:
            #         env_action = student_action
            # else:
            #     env_action = student_action
            env_action = student_action  # pure DAgger execution with student

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            ep_ret += reward

            # Always label with expert action
            dataset.add(obs, expert_action)

            obs = next_obs

        print(f"[DAgger] Episode {ep+1}/{num_episodes}, return = {ep_ret:.2f}")


# -----------------------------
#  Training & evaluation
# -----------------------------

def bc_train_epoch(student, dataset, optimizer, loss_fn, batch_size, device="cpu"):
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


def evaluate_policy(env, policy, episodes=5, device="cpu"):
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


@dataclass
class DAggerConfig:
    num_iterations: int = 5
    expert_init_episodes: int = 5
    dagger_episodes_per_iter: int = 3
    batch_size: int = 64
    bc_epochs_init: int = 5
    bc_epochs_per_iter: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_dagger(cfg: DAggerConfig):
    device = cfg.device
    print(f"Using device: {device}")

    env = make_env(render_mode=None)
    action_space = env.action_space

    student = CNNPolicy().to(device)
    expert = PPOExpertPolicy("ppo_discrete_carracing.pt", device_str=device)
    dataset = ImitationDataset()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    # 1) Pure expert dataset D0 + warm-start BC
    print("Collecting initial expert-only dataset...")
    collect_pure_expert_data(env, expert, dataset, num_episodes=cfg.expert_init_episodes)
    print(f"Initial dataset size: {len(dataset)}")

    if len(dataset) >= cfg.batch_size:
        print("Warm-start BC training...")
        for epoch in range(cfg.bc_epochs_init):
            avg_loss = bc_train_epoch(student, dataset, optimizer, loss_fn,
                                      cfg.batch_size, device=device)
            print(f"[Warm BC] Epoch {epoch+1}/{cfg.bc_epochs_init}, loss = {avg_loss:.4f}")

        avg_return = evaluate_policy(env, student, episodes=3, device=device)
        print(f"After warm-start BC, avg return = {avg_return:.2f}")

    # 2) DAgger iterations
    for it in range(cfg.num_iterations):
        print(f"\n=== DAgger iteration {it+1}/{cfg.num_iterations} ===")

        # Optional beta schedule (uncomment if you want mixed control)
        # beta = max(0.1, 1.0 - it / cfg.num_iterations)
        beta = None

        collect_dagger_data(
            env, student, expert, dataset,
            num_episodes=cfg.dagger_episodes_per_iter,
            device=device,
            beta=beta,
        )
        print(f"Dataset size after iteration {it+1}: {len(dataset)}")

        # BC on aggregated dataset
        for epoch in range(cfg.bc_epochs_per_iter):
            avg_loss = bc_train_epoch(student, dataset, optimizer, loss_fn,
                                      cfg.batch_size, device=device)
            print(f"[DAgger BC] Iter {it+1}, epoch {epoch+1}/{cfg.bc_epochs_per_iter}, "
                  f"loss = {avg_loss:.4f}")

        avg_return = evaluate_policy(env, student, episodes=3, device=device)
        print(f"[Eval] After DAgger iter {it+1}, avg return = {avg_return:.2f}")

    env.close()
    torch.save(student.state_dict(), "student_dagger_carracing.pt")
    print("\nTraining finished. Saved student to student_dagger_carracing.pt")


if __name__ == "__main__":
    cfg = DAggerConfig(
        num_iterations=5,
        expert_init_episodes=5,
        dagger_episodes_per_iter=3,
        batch_size=64,
        bc_epochs_init=5,
        bc_epochs_per_iter=3,
    )
    train_dagger(cfg)
