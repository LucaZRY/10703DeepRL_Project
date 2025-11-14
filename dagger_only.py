import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from collections import deque
import random


# ========================
# 1. Simple CNN policy
# ========================

class CNNPolicy(nn.Module):
    def __init__(self, act_dim):
        super().__init__()
        # Input: (C, H, W) = (4, 84, 84) if we stack 4 frames
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
        # x: [B, 4, 84, 84]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ========================
# 2. Frame preprocessing
# ========================

def preprocess_obs(obs):
    """
    obs: (H, W, 3) RGB uint8
    return: (84, 84) float32 in [0,1]
    """
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    obs = obs.astype(np.float32) / 255.0
    return obs


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        # obs space: k stacked grayscale frames
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
        done = terminated or truncated
        return stacked, reward, terminated, truncated, info


def make_env(render_mode=None):
    base_env = gym.make("CarRacing-v2", render_mode=render_mode)
    env = FrameStackWrapper(base_env, k=4)
    return env


# ========================
# 3. Expert policy stub
# ========================

class ExpertPolicy:
    """
    Placeholder expert.
    Replace `get_action` with:
      - a pre-trained RL policy,
      - human teleop,
      - or later: diffusion-based trajectory expert.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        # !!! TODO: replace this with your real expert.
        # For now, random actions just let you debug the DAgger pipeline.
        return self.action_space.sample()


# ========================
# 4. Dataset for DAgger
# ========================

class DAggerDataset:
    def __init__(self, max_size=200_000):
        self.max_size = max_size
        self.obs = []
        self.actions = []

    def add(self, obs, action):
        if len(self.obs) >= self.max_size:
            # simple FIFO: pop front
            self.obs.pop(0)
            self.actions.pop(0)
        self.obs.append(obs.astype(np.float32))
        self.actions.append(action.astype(np.float32))

    def __len__(self):
        return len(self.obs)

    def sample_batch(self, batch_size, device):
        idxs = np.random.randint(0, len(self.obs), size=batch_size)
        obs_batch = np.stack([self.obs[i] for i in idxs], axis=0)   # [B, 4, 84, 84]
        act_batch = np.stack([self.actions[i] for i in idxs], axis=0)  # [B, act_dim]
        obs_tensor = torch.tensor(obs_batch, device=device)
        act_tensor = torch.tensor(act_batch, device=device)
        return obs_tensor, act_tensor


# ========================
# 5. DAgger training loop
# ========================

def train_dagger(
    num_iterations=10,
    episodes_per_iter=5,
    batch_size=64,
    train_epochs_per_iter=5,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    env = make_env(render_mode=None)
    act_dim = env.action_space.shape[0]

    # Student policy to be learned
    student = CNNPolicy(act_dim).to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # Expert policy (stub)
    expert = ExpertPolicy(env.action_space)

    # DAgger dataset
    dataset = DAggerDataset(max_size=200_000)

    global_step = 0

    for it in range(num_iterations):
        print(f"\n=== DAgger iteration {it+1}/{num_iterations} ===")

        # ---- Collect data with current student + expert labels ----
        for ep in range(episodes_per_iter):
            obs, info = env.reset()
            episode_return = 0.0

            while True:
                global_step += 1

                # Student action
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    student_action = student(obs_tensor).cpu().numpy()[0]

                # Expert action (label)
                expert_action = expert.get_action(obs)

                # Here: DAgger chooses which action to execute in the env.
                # Classic DAgger: execute student action, but always query expert for label.
                env_action = student_action

                next_obs, reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated
                episode_return += reward

                # Add (state, expert_action) to dataset
                dataset.add(obs, expert_action)

                obs = next_obs

                if done:
                    print(f"  [Iter {it+1}] Episode {ep+1}/{episodes_per_iter} return: {episode_return:.1f}")
                    break

        print(f"  Collected dataset size: {len(dataset)}")

        # ---- Train student via behavior cloning on aggregated dataset ----
        if len(dataset) < batch_size:
            print("  Not enough data yet to train, skipping BC this iteration.")
            continue

        for epoch in range(train_epochs_per_iter):
            # sample multiple batches per epoch
            for _ in range(100):  # you can tune this
                obs_batch, act_batch = dataset.sample_batch(batch_size, device=device)
                pred_actions = student(obs_batch)
                loss = loss_fn(pred_actions, act_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"  [Iter {it+1}] Epoch {epoch+1}/{train_epochs_per_iter}, loss={loss.item():.4f}")

    env.close()
    # Save final student
    torch.save(student.state_dict(), "student_dagger_carracing.pt")
    print("Training finished, student policy saved to student_dagger_carracing.pt")


if __name__ == "__main__":
    train_dagger(
        num_iterations=5,          # number of DAgger rounds
        episodes_per_iter=3,       # episodes per round
        batch_size=64,
        train_epochs_per_iter=3,   # BC epochs per round
    )
