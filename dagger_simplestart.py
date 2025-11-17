"""
dagger_carracing.py

Simple DAgger loop on CarRacing-v2.

- Student: CNN policy that maps stacked frames -> actions
- Expert: placeholder (currently random) — later you’ll replace this
  with a trained PPO/SAC policy or a diffusion-based expert.

Run:
    conda activate drl-diffdist
    python dagger_carracing.py
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from collections import deque


# ========================
# 1. Student CNN Policy
# ========================

class CNNPolicy(nn.Module):
    def __init__(self, act_dim):
        super().__init__()
        # Input: (B, 4, 84, 84) — 4 stacked grayscale frames
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64, 7, 7)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),  # CarRacing actions are in [-1, 1]
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ========================
# 2. Obs preprocessing + frame stack
# ========================

def preprocess_obs(obs):
    """RGB (96,96,3) -> grayscale (84,84), float32 in [0,1]."""
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    obs = obs.astype(np.float32) / 255.0
    return obs


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(k, 84, 84),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = preprocess_obs(obs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(frame)
        stacked = np.stack(self.frames, axis=0)
        return stacked, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = preprocess_obs(obs)
        self.frames.append(frame)
        stacked = np.stack(self.frames, axis=0)
        return stacked, reward, terminated, truncated, info


def make_env(render_mode=None):
    base_env = gym.make("CarRacing-v2", render_mode=render_mode)
    env = FrameStackWrapper(base_env, k=4)
    return env


# ========================
# 3. Expert policy (stub)
# ========================

class ExpertPolicy:
    """
    Placeholder expert.

    TODO: replace get_action() with:
      - a trained PPO/SAC model, or
      - a diffusion-based trajectory expert.
    For now it returns random actions just to debug DAgger mechanics.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        # obs is (4,84,84) — you can ignore it in this random stub
        return self.action_space.sample()


# ========================
# 4. DAgger dataset
# ========================

class DAggerDataset:
    def __init__(self, max_size=200_000):
        self.max_size = max_size
        self.obs = []
        self.actions = []

    def add(self, obs, action):
        if len(self.obs) >= self.max_size:
            self.obs.pop(0)
            self.actions.pop(0)
        self.obs.append(obs.astype(np.float32))
        self.actions.append(action.astype(np.float32))

    def __len__(self):
        return len(self.obs)

    def sample_batch(self, batch_size, device):
        idxs = np.random.randint(0, len(self.obs), size=batch_size)
        obs_batch = np.stack([self.obs[i] for i in idxs], axis=0)   # [B,4,84,84]
        act_batch = np.stack([self.actions[i] for i in idxs], axis=0)  # [B, act_dim]
        obs_tensor = torch.tensor(obs_batch, device=device)
        act_tensor = torch.tensor(act_batch, device=device)
        return obs_tensor, act_tensor


# ========================
# 5. DAgger training loop
# ========================

def train_dagger(
    num_iterations=5,
    episodes_per_iter=3,
    batch_size=64,
    bc_epochs_per_iter=3,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    env = make_env(render_mode=None)
    act_dim = env.action_space.shape[0]

    student = CNNPolicy(act_dim).to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    expert = ExpertPolicy(env.action_space)
    dataset = DAggerDataset(max_size=200_000)

    for it in range(num_iterations):
        print(f"\n=== DAgger iteration {it+1}/{num_iterations} ===")

        # ----- Collect data with current student -----
        for ep in range(episodes_per_iter):
            obs, info = env.reset()
            episode_return = 0.0

            while True:
                # Student action
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)  # [1,4,84,84]
                with torch.no_grad():
                    student_action = student(obs_tensor).cpu().numpy()[0]

                # Expert label
                expert_action = expert.get_action(obs)

                # Classic DAgger: execute student action, but always label with expert
                env_action = student_action
                next_obs, reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated
                episode_return += reward

                # Add to dataset (state, expert_action)
                dataset.add(obs, expert_action)

                obs = next_obs

                if done:
                    print(
                        f"  [Iter {it+1}] Episode {ep+1}/{episodes_per_iter} "
                        f"return: {episode_return:.1f}"
                    )
                    break

        print(f"  Collected dataset size: {len(dataset)}")

        # ----- Behavior cloning on aggregated dataset -----
        if len(dataset) < batch_size:
            print("  Not enough data yet to train, skipping BC.")
            continue

        for epoch in range(bc_epochs_per_iter):
            # simple training: fixed number of gradient steps per epoch
            num_batches = 100
            for _ in range(num_batches):
                obs_batch, act_batch = dataset.sample_batch(batch_size, device=device)
                pred_actions = student(obs_batch)
                loss = loss_fn(pred_actions, act_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(
                f"  [Iter {it+1}] Epoch {epoch+1}/{bc_epochs_per_iter}, "
                f"BC loss = {loss.item():.4f}"
            )

    env.close()
    torch.save(student.state_dict(), "student_dagger_carracing.pt")
    print("\nTraining finished. Saved student to student_dagger_carracing.pt")


if __name__ == "__main__":
    train_dagger(
        num_iterations=5,
        episodes_per_iter=3,
        batch_size=64,
        bc_epochs_per_iter=3,
    )
