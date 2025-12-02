"""
Strong PPO expert (Option C) for CarRacing-v2 (continuous) + dataset saver.

Goal:
- Train a strong PPO expert (avg return ~600+ with enough time/steps)
- 4 stacked grayscale frames: (4, 96, 96)
- Gaussian policy with moderate exploration
- Save transitions for diffusion / distillation

Outputs:
    ppo_carracing_strong_best.pth
    ppo_carracing_strong_last.pth
    carracing_ppo_strong_dataset.npz with:
        obs:     (N, 4, 96, 96)   float32 in [0,1]
        actions: (N, 3)           [steer, gas, brake]
        rewards: (N,)
        dones:   (N,)
"""

import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# Env + preprocessing
# --------------------------

def make_env(render_mode=None):
    """
    CarRacing-v2 continuous env with:
      - grayscale
      - resize to 96 x 96
      - 4-frame stack
    """
    env = gym.make("CarRacing-v2", continuous=True, render_mode=render_mode)
    env = GrayScaleObservation(env, keep_dim=True)   # (H,W,1)
    env = ResizeObservation(env, 96)                 # (96,96,1)
    env = FrameStack(env, num_stack=4)               # (96,96,4)
    return env


def preprocess_obs(obs):
    """
    Convert env observation into shape (4, 96, 96) float32 in [0,1].
    Handles (96,96,4), (4,96,96), (96,96,1), (96,96).
    """
    arr = np.array(obs)

    # Squeeze extra singleton dims like (4,96,96,1) or (1,96,96,4)
    while arr.ndim > 3 and (arr.shape[0] == 1 or arr.shape[-1] == 1):
        if arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        elif arr.shape[0] == 1:
            arr = arr.squeeze(0)
        else:
            break

    if arr.ndim == 3:
        # (H,W,4) -> channels-first
        if arr.shape[-1] == 4:
            arr = arr.astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))  # (4,96,96)
        # (4,H,W)
        elif arr.shape[0] == 4:
            arr = arr.astype(np.float32)
        # (H,W,1) -> tile to 4
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

    return arr  # (4,96,96)


def env_action_from_raw(a_raw: np.ndarray) -> np.ndarray:
    """
    Clamp raw Gaussian sample into valid CarRacing ranges.
    a_raw: np.array(3,)
    Returns: np.array([steer, gas, brake]) float32
    """
    steer = np.clip(a_raw[0], -1.0, 1.0)
    gas   = np.clip(a_raw[1], 0.0, 1.0)
    brake = np.clip(a_raw[2], 0.0, 1.0)
    return np.array([steer, gas, brake], dtype=np.float32)


# --------------------------
# Actor-Critic Network
# --------------------------

class ActorCritic(nn.Module):
    """
    CNN encoder -> shared body -> Gaussian policy (mu) + value V(s).

    Input:  (B, 4, 96, 96)
    Output:
        dist  ~ N(mu, std)   (3-dim)
        value: (B,)
    """

    def __init__(self, obs_channels=4, act_dim=3):
        super().__init__()
        self.act_dim = act_dim

        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4),  # (32, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),            # (64, 10, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),            # (64, 8, 8)
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(256, act_dim)
        self.v_head  = nn.Linear(256, 1)

        # Smaller std for more stable steering: log_std = -1.5 -> std ≈ 0.22
        self.log_std = nn.Parameter(torch.ones(act_dim) * -1.5)

    def forward(self, x):
        x = self.conv(x)           # (B,64,8,8)
        x = x.view(x.size(0), -1)  # (B,64*8*8)
        x = self.fc(x)             # (B,256)
        mu = self.mu_head(x)       # (B,3)
        v  = self.v_head(x).squeeze(-1)  # (B,)
        return mu, v

    def get_dist_value(self, x):
        mu, v = self.forward(x)
        std = torch.exp(self.log_std)    # (3,)
        dist = torch.distributions.Normal(mu, std)
        return dist, v


# --------------------------
# PPO Buffer with GAE
# --------------------------

class PPOBuffer:
    def __init__(self, obs_shape, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf   = np.zeros((size, *obs_shape), dtype=np.float32)
        self.act_buf   = np.zeros((size, act_dim), dtype=np.float32)  # raw actions
        self.rew_buf   = np.zeros(size, dtype=np.float32)
        self.done_buf  = np.zeros(size, dtype=np.float32)
        self.val_buf   = np.zeros(size, dtype=np.float32)
        self.logp_buf  = np.zeros(size, dtype=np.float32)
        self.adv_buf   = np.zeros(size, dtype=np.float32)
        self.ret_buf   = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.lam   = lam
        self.ptr = 0
        self.path_start = 0
        self.max_size = size

    def store(self, obs, act, rew, done, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr]  = obs
        self.act_buf[self.ptr]  = act
        self.rew_buf[self.ptr]  = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr]  = val
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

        self.ret_buf[self.path_start:end] = (
            self.adv_buf[self.path_start:end] + self.val_buf[self.path_start:end]
        )
        self.path_start = self.ptr

    def get(self):
        """
        Return all data, with normalized advantages.
        """
        assert self.ptr == self.max_size
        self.ptr = 0
        self.path_start = 0

        adv = self.adv_buf
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        data = dict(
            obs   = self.obs_buf,
            act   = self.act_buf,
            ret   = self.ret_buf,
            adv   = adv,
            logp  = self.logp_buf,
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in data.items()
        }


# --------------------------
# Config (lighter defaults)
# --------------------------

@dataclass
class PPOConfig:
    # Lighter defaults to avoid crashes
    total_steps: int = 150_000        # total env steps
    steps_per_epoch: int = 4_096      # rollout size per PPO update

    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 2e-4
    v_lr: float = 5e-4
    train_pi_iters: int = 5
    train_v_iters: int = 5
    max_ep_len: int = 1000


# --------------------------
# PPO Agent
# --------------------------

class PPOAgent:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.env = make_env(render_mode=None)

        obs_shape = (4, 96, 96)
        act_dim   = 3

        self.ac = ActorCritic(obs_channels=4, act_dim=act_dim).to(device)
        self.pi_optim = Adam(self.ac.parameters(), lr=cfg.pi_lr)
        self.v_optim  = Adam(self.ac.parameters(), lr=cfg.v_lr)

        self.buf = PPOBuffer(
            obs_shape=obs_shape,
            act_dim=act_dim,
            size=cfg.steps_per_epoch,
            gamma=cfg.gamma,
            lam=cfg.lam,
        )

        # Dataset buffers
        self.dataset_obs     = []
        self.dataset_actions = []
        self.dataset_rewards = []
        self.dataset_dones   = []

        # Cap how many transitions we store to avoid huge RAM usage
        # 30k steps of (4,96,96) is already large but manageable.
        self.max_dataset_steps = 30_000

        # Tracking
        self.episode_returns = []
        self.best_return = -np.inf

    def select_action(self, obs_np):
        """
        obs_np: (4,96,96) numpy
        Returns:
            env_action: np(3,) in valid ranges
            raw_action: np(3,) pre-clipping (for PPO)
            value: float
            logp: float
        """
        o = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        dist, v = self.ac.get_dist_value(o)
        a_raw = dist.sample()                      # (1,3)
        logp  = dist.log_prob(a_raw).sum(dim=-1)   # (1,)
        a_np_raw = a_raw.squeeze(0).cpu().numpy()
        env_action = env_action_from_raw(a_np_raw)
        return env_action, a_np_raw, v.item(), logp.item()

    def update(self):
        cfg = self.cfg
        data = self.buf.get()
        obs, act_raw, ret, adv, logp_old = (
            data["obs"], data["act"], data["ret"], data["adv"], data["logp"]
        )

        # Policy update
        for _ in range(cfg.train_pi_iters):
            dist, _ = self.ac.get_dist_value(obs)
            logp = dist.log_prob(act_raw).sum(dim=-1)
            ratio = torch.exp(logp - logp_old)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            self.pi_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.pi_optim.step()

        # Value update
        for _ in range(cfg.train_v_iters):
            _, v_pred = self.ac.get_dist_value(obs)
            critic_loss = F.mse_loss(v_pred, ret)

            self.v_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.v_optim.step()

    def train(self):
        cfg = self.cfg
        env = self.env

        total_steps = cfg.total_steps
        steps_per_epoch = cfg.steps_per_epoch
        num_epochs = total_steps // steps_per_epoch

        # plotting style similar to earlier scripts
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display
        plt.ion()

        for epoch in range(num_epochs):
            self.buf.ptr = 0
            self.buf.path_start = 0
            steps_collected = 0

            while steps_collected < steps_per_epoch:
                obs_raw, _ = env.reset()
                obs = preprocess_obs(obs_raw)
                done = False
                ep_ret = 0.0
                ep_len = 0
                negative_counter = 0

                while not done and steps_collected < steps_per_epoch:
                    env_action, raw_action, value, logp = self.select_action(obs)
                    next_obs_raw, reward, terminated, truncated, _ = env.step(env_action)
                    done_flag = terminated or truncated

                    # Early termination if stuck in bad region too long
                    negative_counter = negative_counter + 1 if reward < -0.1 else 0
                    if negative_counter > 60:
                        done_flag = True

                    next_obs = preprocess_obs(next_obs_raw)

                    # Store in PPO buffer
                    self.buf.store(
                        obs, raw_action, reward, float(done_flag),
                        value, logp
                    )

                    # Store in dataset buffers (env actions), but cap size
                    if len(self.dataset_obs) < self.max_dataset_steps:
                        self.dataset_obs.append(obs.copy())
                        self.dataset_actions.append(env_action.copy())
                        self.dataset_rewards.append(float(reward))
                        self.dataset_dones.append(float(done_flag))

                    ep_ret += reward
                    ep_len += 1
                    steps_collected += 1
                    obs = next_obs
                    done = done_flag

                    timeout = (ep_len >= cfg.max_ep_len)
                    epoch_ended = (steps_collected == steps_per_epoch)

                    if done or timeout or epoch_ended:
                        if epoch_ended and not done:
                            # bootstrap value
                            with torch.no_grad():
                                o_t = torch.as_tensor(
                                    obs, dtype=torch.float32, device=device
                                ).unsqueeze(0)
                                _, v = self.ac.get_dist_value(o_t)
                                last_val = v.item()
                        else:
                            last_val = 0.0
                        self.buf.finish_path(last_val)

                        self.episode_returns.append(ep_ret)
                        print(
                            f"Epoch {epoch+1}/{num_epochs}, "
                            f"Episode Return: {ep_ret:.2f}, Length: {ep_len}"
                        )

                        # Save best/last models
                        if ep_ret > self.best_return:
                            self.best_return = ep_ret
                            torch.save(self.ac.state_dict(), "ppo_carracing_strong_best.pth")
                        torch.save(self.ac.state_dict(), "ppo_carracing_strong_last.pth")

                        # Plot
                        plt.figure(1)
                        plt.clf()
                        plt.title("PPO Training Results (Strong Expert)")
                        plt.xlabel("Episode")
                        plt.ylabel("Total Reward")
                        returns_np = np.array(self.episode_returns)
                        plt.plot(returns_np)
                        if len(returns_np) >= 20:
                            ma = np.convolve(
                                returns_np, np.ones(20)/20, mode="valid"
                            )
                            plt.plot(np.arange(19, 19+len(ma)), ma)
                        plt.pause(0.001)
                        if is_ipython:
                            display.display(plt.gcf())
                            display.clear_output(wait=True)

                        break  # break inner while; start new episode

            # PPO update at end of epoch
            self.update()

        env.close()
        plt.ioff()
        plt.figure(1)
        plt.savefig("ppo_training_result_strong.png")
        print("Training finished. Best return:", self.best_return)

    def save_dataset(self, path="carracing_ppo_strong_dataset.npz"):
        if len(self.dataset_obs) == 0:
            print("Warning: no dataset collected, not saving npz.")
            return

        obs     = np.stack(self.dataset_obs, axis=0)        # (N,4,96,96)
        actions = np.stack(self.dataset_actions, axis=0)    # (N,3)
        rewards = np.array(self.dataset_rewards, dtype=np.float32)
        dones   = np.array(self.dataset_dones, dtype=np.float32)

        np.savez_compressed(
            path,
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )
        print(f"Saved dataset to {path} with {obs.shape[0]} transitions")


# --------------------------
# Main
# --------------------------

def main():
    # Lighter config to avoid crashes and finish in reasonable time
    cfg = PPOConfig(
        total_steps=150_000,    # was 1_200_000 in original strong expert
        steps_per_epoch=4_096,  # was 16_384
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        pi_lr=2e-4,
        v_lr=5e-4,
        train_pi_iters=5,       # was 15
        train_v_iters=5,        # was 15
        max_ep_len=1000,
    )

    agent = PPOAgent(cfg)
    agent.train()
    agent.save_dataset("carracing_ppo_strong_dataset.npz")


if __name__ == "__main__":
    main()
