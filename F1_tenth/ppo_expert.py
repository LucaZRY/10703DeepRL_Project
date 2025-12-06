"""
Pure PPO expert for CarRacing-v2 (continuous) + dataset saver.

- Env: CarRacing-v2 with continuous actions [steer, gas, brake]
- Obs: stacked grayscale frames (4, 84, 84), float32 in [0,1]
- Outputs: carracing_ppo_dataset.npz
    - obs: (N, 4, 84, 84)
    - actions: (N, 3)
    - rewards: (N,)
    - dones: (N,)
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------
# Env + preprocessing
# --------------------------

def make_env(render_mode=None):
    """
    CarRacing-v2 continuous env with:
      - grayscale
      - 84x84
      - 4-frame stack
    """
    env = gym.make("CarRacing-v2", continuous=True, render_mode=render_mode)
    env = GrayScaleObservation(env, keep_dim=True)  # (H, W, 1)
    env = ResizeObservation(env, 84)               # (84, 84, 1)
    env = FrameStack(env, num_stack=4)             # (84, 84, 4)
    return env


def preprocess_obs(obs):
    """
    Convert env obs into shape (4, 84, 84) float32 in [0,1].
    Handles:
    - (84,84,4) from FrameStack (H,W,stack)
    - (4,84,84) already channels-first
    """
    arr = np.array(obs)

    # Squeeze weird singleton dims if any
    while arr.ndim > 3 and (arr.shape[0] == 1 or arr.shape[-1] == 1):
        if arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        elif arr.shape[0] == 1:
            arr = arr.squeeze(0)
        else:
            break

    if arr.ndim == 3:
        # (H,W,4)
        if arr.shape[-1] == 4:
            arr = arr.astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))  # -> (4,H,W)
        # (4,H,W)
        elif arr.shape[0] == 4:
            arr = arr.astype(np.float32)
        # (H,W,1) → tile to 4
        elif arr.shape[-1] == 1:
            img = arr[..., 0].astype(np.float32) / 255.0
            arr = np.tile(img[None, ...], (4, 1, 1))
        else:
            img = arr.astype(np.float32) / 255.0
            arr = np.tile(img[None, ...], (4, 1, 1))
    elif arr.ndim == 2:
        img = arr.astype(np.float32) / 255.0
        arr = np.tile(img[None, ...], (4, 1, 1))
    else:
        raise ValueError(f"Unexpected obs shape in preprocess_obs: {arr.shape}")

    return arr  # (4,84,84)


# --------------------------
# PPO network: CNN + Gaussian policy
# --------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_channels=4, act_dim=3):
        super().__init__()
        self.act_dim = act_dim

        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4),  # -> (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),            # -> (64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),            # -> (64,7,7)
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(256, act_dim)
        self.v_head = nn.Linear(256, 1)

        # log_std as a parameter (independent per action dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        # x: (B,4,84,84)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        mu = self.mu_head(x)           # (B,3)
        v = self.v_head(x).squeeze(-1) # (B,)
        return mu, v

    def get_dist_value(self, x):
        mu, v = self.forward(x)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mu, std)
        return dist, v


# --------------------------
# PPO buffer with GAE
# --------------------------

class PPOBuffer:
    def __init__(self, obs_shape, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, *obs_shape), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start = 0
        self.max_size = size

    def store(self, obs, act, rew, done, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0.0):
        """
        Call at end of trajectory or when buffer is full.
        Uses GAE-Lambda to compute advantage estimates.
        """
        end = self.ptr
        rews = np.append(self.rew_buf[self.path_start:end], last_val)
        vals = np.append(self.val_buf[self.path_start:end], last_val)

        gae = 0.0
        for t in reversed(range(end - self.path_start)):
            nonterminal = 1.0 - self.done_buf[self.path_start + t]
            delta = rews[t] + self.gamma * vals[t + 1] * nonterminal - vals[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            self.adv_buf[self.path_start + t] = gae

        self.ret_buf[self.path_start:end] = self.adv_buf[self.path_start:end] + self.val_buf[self.path_start:end]
        self.path_start = self.ptr

    def get(self):
        """
        Return all data from the buffer, with advantages normalized.
        """
        assert self.ptr == self.max_size  # buffer full
        self.ptr = 0
        self.path_start = 0

        adv = self.adv_buf
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        data = dict(obs=self.obs_buf,
                    act=self.act_buf,
                    ret=self.ret_buf,
                    adv=adv,
                    logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}


# --------------------------
# PPO config + agent
# --------------------------

@dataclass
class PPOConfig:
    total_steps: int = 200_000        # total environment steps
    steps_per_epoch: int = 2048       # rollout size
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    v_lr: float = 1e-3
    train_pi_iters: int = 80
    train_v_iters: int = 80
    max_ep_len: int = 1000


class PPOAgent:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.env = make_env(render_mode=None)
        obs_shape = (4, 84, 84)
        act_dim = 3

        self.ac = ActorCritic(obs_channels=4, act_dim=act_dim).to(device)
        self.pi_optimizer = Adam(self.ac.parameters(), lr=cfg.pi_lr)
        self.v_optimizer = Adam(self.ac.parameters(), lr=cfg.v_lr)

        self.buf = PPOBuffer(
            obs_shape=obs_shape,
            act_dim=act_dim,
            size=cfg.steps_per_epoch,
            gamma=cfg.gamma,
            lam=cfg.lam,
        )

        # For saving dataset
        self.dataset_obs = []
        self.dataset_actions = []
        self.dataset_rewards = []
        self.dataset_dones = []

    def select_action(self, obs):
        o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist, v = self.ac.get_dist_value(o)
        a = dist.sample()
        logp = dist.log_prob(a).sum(axis=-1)
        return a.squeeze(0).cpu().numpy(), v.item(), logp.item()

    def update(self):
        cfg = self.cfg
        data = self.buf.get()
        obs, act, ret, adv, logp_old = data["obs"], data["act"], data["ret"], data["adv"], data["logp"]

        # Policy update
        for _ in range(cfg.train_pi_iters):
            dist, _ = self.ac.get_dist_value(obs)
            logp = dist.log_prob(act).sum(axis=-1)
            ratio = torch.exp(logp - logp_old)

            clip_adv = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.pi_optimizer.step()

        # Value update
        for _ in range(cfg.train_v_iters):
            _, v = self.ac.get_dist_value(obs)
            loss_v = F.mse_loss(v, ret)

            self.v_optimizer.zero_grad()
            loss_v.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.v_optimizer.step()

    def train(self):
        cfg = self.cfg
        obs, info = self.env.reset()
        obs = preprocess_obs(obs)
        ep_ret = 0.0
        ep_len = 0

        total_steps = cfg.total_steps
        steps_per_epoch = cfg.steps_per_epoch
        num_epochs = total_steps // steps_per_epoch

        for epoch in range(num_epochs):
            for t in range(steps_per_epoch):
                act, v, logp = self.select_action(obs)
                next_obs, rew, terminated, truncated, info = self.env.step(act)
                done = terminated or truncated

                # store in PPO buffer
                self.buf.store(obs, act, rew, done, v, logp)

                # also store in dataset buffers
                self.dataset_obs.append(obs.copy())
                self.dataset_actions.append(act.copy())
                self.dataset_rewards.append(rew)
                self.dataset_dones.append(float(done))

                ep_ret += rew
                ep_len += 1
                obs = preprocess_obs(next_obs)

                timeout = ep_len == cfg.max_ep_len
                terminal = done or timeout
                epoch_ended = (t == steps_per_epoch - 1)

                if terminal or epoch_ended:
                    if epoch_ended and not terminal:
                        # bootstrap value if epoch ended but episode not done
                        with torch.no_grad():
                            o_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                            _, v = self.ac.get_dist_value(o_t)
                            last_val = v.item()
                        self.buf.finish_path(last_val)
                    else:
                        self.buf.finish_path(last_val=0.0)

                    if terminal:
                        print(f"Epoch {epoch+1}, episode return = {ep_ret:.2f}, len = {ep_len}")
                        obs, info = self.env.reset()
                        obs = preprocess_obs(obs)
                        ep_ret = 0.0
                        ep_len = 0

            # PPO update at end of epoch
            self.update()

        self.env.close()

    def save_dataset(self, path="carracing_ppo_dataset.npz"):
        obs = np.stack(self.dataset_obs, axis=0)         # (N,4,84,84)
        actions = np.stack(self.dataset_actions, axis=0) # (N,3)
        rewards = np.array(self.dataset_rewards, dtype=np.float32)
        dones = np.array(self.dataset_dones, dtype=np.float32)

        np.savez_compressed(
            path,
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )
        print(f"Saved dataset to {path} with {obs.shape[0]} transitions")


def main():
    cfg = PPOConfig(
        total_steps=200_000,       # increase if you want a better expert
        steps_per_epoch=2048,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        pi_lr=3e-4,
        v_lr=1e-3,
        train_pi_iters=80,
        train_v_iters=80,
        max_ep_len=1000,
    )

    agent = PPOAgent(cfg)
    agent.train()
    agent.save_dataset("carracing_ppo_dataset.npz")


if __name__ == "__main__":
    main()
