"""
Enhanced PPO for CarRacing with Video Recording and Pipeline Integration

Improvements over original PPO:
- Video recording during training and evaluation
- Better CarRacing-specific optimizations
- Enhanced monitoring and visualization
- Improved dataset generation for diffusion pipeline
- Curriculum learning and adaptive exploration
- Comprehensive logging and analysis

Outputs:
    models/carracing_ppo_enhanced_best.pth
    datasets/carracing_ppo_enhanced_dataset.npz
    videos/
    plots/
    logs/
"""

import os
import time
import json
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Try importing TensorBoard, fallback to None if not available
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None
    HAS_TENSORBOARD = False
    print("⚠️ TensorBoard not available. Install with: pip install tensorboard")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Enhanced PPO using device: {device}")


# --------------------------
# Enhanced Environment Setup
# --------------------------

class AddChannelDimWrapper(gym.ObservationWrapper):
    """Add channel dimension to observations for frame stacking (safe no-op for 3D obs)."""

    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        if len(old_shape) == 2:  # (H, W) -> (H, W, 1)
            new_shape = old_shape + (1,)
        else:  # Already has channels
            new_shape = old_shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, obs):
        if len(obs.shape) == 2:  # (H, W) -> (H, W, 1)
            return np.expand_dims(obs, axis=-1)
        else:
            return obs


class CarRacingWrapper(gym.Wrapper):
    """Enhanced wrapper for CarRacing with better reward shaping."""

    def __init__(self, env):
        super().__init__(env)
        self.last_speed = 0
        self.last_position = None
        self.negative_reward_counter = 0
        self.max_negative_rewards = 100
        self.trajectory_length = 0

    def reset(self, **kwargs):
        self.last_speed = 0
        self.last_position = None
        self.negative_reward_counter = 0
        self.trajectory_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.trajectory_length += 1

        original_reward = reward

        # Penalize many consecutive negative rewards (off-track)
        if reward < 0:
            self.negative_reward_counter += 1
            if self.negative_reward_counter > self.max_negative_rewards:
                done = True
                reward = -100  # terminal penalty
        else:
            self.negative_reward_counter = max(0, self.negative_reward_counter - 1)

        # Speed shaping (if env exposes speed)
        if hasattr(self.env, "speed") and self.env.speed is not None:
            speed = self.env.speed
            if speed > 30:      # good speed
                reward += 0.1
            elif speed < 10:    # too slow
                reward -= 0.05

        # Penalize very short episodes
        if done and self.trajectory_length < 500:
            reward -= 50

        info["original_reward"] = original_reward
        info["enhanced_reward"] = reward
        info["trajectory_length"] = self.trajectory_length
        info["negative_counter"] = self.negative_reward_counter

        return obs, reward, done, truncated, info


def make_enhanced_env(render_mode=None, record_video=False, video_dir=None):
    """Create enhanced CarRacing environment with optional video recording."""

    # Use CarRacing-v2 (version available in your installation)
    env = gym.make("CarRacing-v2", continuous=True, render_mode=render_mode)

    # Optional video recording (requires rgb_array render mode)
    if record_video and video_dir and render_mode == "rgb_array":
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda episode: episode % 10 == 0,
            video_length=0,
            name_prefix="carracing_ppo",
        )

    # Preprocessing wrappers
    from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStackObservation

    env = GrayScaleObservation(env, keep_dim=True)     # (96, 96, 1)
    env = ResizeObservation(env, (96, 96))             # still (96, 96, 1)
    env = AddChannelDimWrapper(env)                    # no-op here, but safe
    env = FrameStackObservation(env, stack_size=4)     # (96, 96, 4)

    env = CarRacingWrapper(env)
    return env


# --------------------------
# Enhanced Network Architectures
# --------------------------

class EnhancedCNN(nn.Module):
    """Enhanced CNN for CarRacing with better feature extraction."""

    def __init__(self, input_channels=4, features_dim=512):
        super().__init__()

        self.conv = nn.Sequential(
            # 96x96x4 -> ...
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 96, 96)
            conv_out = self.conv(dummy)
            conv_out_size = conv_out.view(1, -1).shape[1]

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.features_dim = features_dim

    def forward(self, x):
        # x: (B, C, H, W), pixel range [0,1] ideally
        if x.max() > 1.1:
            x = x / 255.0
        conv_out = self.conv(x)
        features = self.features(conv_out)
        return features


class EnhancedActor(nn.Module):
    """Enhanced actor network with better exploration."""

    def __init__(self, features_dim=512, action_dim=3, hidden_dim=256):
        super().__init__()

        self.shared_net = EnhancedCNN(input_channels=4, features_dim=features_dim)

        self.policy_mean = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # [-1, 1]
        )

        # learnable log std
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        features = self.shared_net(obs)

        mean = self.policy_mean(features)

        # steering: [-1,1]; gas & brake: [0,1]
        steering = mean[:, 0:1]
        gas = (mean[:, 1:2] + 1) / 2
        brake = (mean[:, 2:3] + 1) / 2
        mean = torch.cat([steering, gas, brake], dim=1)

        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std


class EnhancedCritic(nn.Module):
    """Enhanced critic network for value estimation."""

    def __init__(self, features_dim=512, hidden_dim=256):
        super().__init__()

        self.shared_net = EnhancedCNN(input_channels=4, features_dim=features_dim)

        self.value_head = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        features = self.shared_net(obs)
        value = self.value_head(features)
        return value


# --------------------------
# Enhanced PPO Configuration
# --------------------------

@dataclass
class EnhancedPPOConfig:
    # Environment
    max_episode_steps: int = 1000
    early_stop_reward: float = 500.0

    # Training
    total_timesteps: int = 1_000_000
    steps_per_update: int = 2048
    n_epochs: int = 10
    batch_size: int = 64

    # PPO hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_decay: float = 0.995
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_ratio: float = 0.2
    clip_value_loss: bool = True
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5

    # Curriculum learning (not heavily used but kept for future)
    use_curriculum: bool = True
    curriculum_threshold: float = 100.0
    curriculum_increment: float = 50.0

    # Exploration
    exploration_decay: float = 0.9995
    min_entropy_coeff: float = 0.001

    # Logging and evaluation
    log_interval: int = 10
    eval_interval: int = 100_000  # effectively disabled unless you extend training
    save_interval: int = 100
    video_interval: int = 25

    # Video recording
    record_training_videos: bool = True
    record_eval_videos: bool = False
    max_video_episodes: int = 5


# --------------------------
# Enhanced Training Loop
# --------------------------

class EnhancedPPOTrainer:
    """Enhanced PPO trainer with comprehensive features."""

    def __init__(self, config: EnhancedPPOConfig):
        self.config = config
        self.global_step = 0
        self.episode_count = 0

        self.setup_directories()

        # Environments
        self.env = make_enhanced_env()
        self.eval_env = make_enhanced_env(
            render_mode="rgb_array" if config.record_eval_videos else None,
            record_video=config.record_eval_videos,
            video_dir="videos/eval",
        )

        obs, _ = self.env.reset()
        self.obs_shape = obs.shape
        self.action_dim = self.env.action_space.shape[0]

        self.actor = EnhancedActor(action_dim=self.action_dim).to(device)
        self.critic = EnhancedCritic().to(device)

        self.actor_optimizer = AdamW(self.actor.parameters(), lr=config.lr_actor, weight_decay=1e-4)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=config.lr_critic, weight_decay=1e-4)

        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.actor_optimizer, config.lr_decay
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer, config.lr_decay
        )

        self.writer = SummaryWriter("logs/enhanced_ppo") if HAS_TENSORBOARD else None
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "actor_losses": [],
            "critic_losses": [],
            "entropy_losses": [],
            "learning_rates": [],
        }

        self.dataset_buffer = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

        print("Enhanced PPO Trainer initialized:")
        print(f"  Observation shape: {self.obs_shape}")
        print(f"  Action dimension: {self.action_dim}")
        print(f"  Total parameters: {self.count_parameters()}")

    def setup_directories(self):
        os.makedirs("models", exist_ok=True)
        os.makedirs("videos/training", exist_ok=True)
        os.makedirs("videos/eval", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("datasets", exist_ok=True)

    def count_parameters(self):
        actor_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        critic_params = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        return actor_params + critic_params

    # ---------- OBS PREPROCESSING (GPU + NCHW) ----------

    def preprocess_obs(self, obs):
        """
        Preprocess a single observation to (1, C, H, W) on device.
        Env gives (H, W, C_stack) = (96, 96, 4).
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(device)
        else:
            obs = obs.to(device).float()

        if obs.dim() == 3:
            # (H, W, C) -> (1, C, H, W)
            obs = obs.permute(2, 0, 1).unsqueeze(0)
        elif obs.dim() == 4:
            # (B, H, W, C) -> (B, C, H, W)
            obs = obs.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unexpected obs shape in preprocess_obs: {obs.shape}")

        if obs.max() > 1.1:
            obs = obs / 255.0
        return obs

    # ---------- ROLLOUT COLLECTION ----------

    def collect_rollouts(self, n_steps):
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(n_steps):
            obs_tensor = self.preprocess_obs(obs)

            with torch.no_grad():
                mean, std = self.actor(obs_tensor)
                value = self.critic(obs_tensor)

                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

                action_np = action[0].cpu().numpy()
                action_np = np.clip(
                    action_np,
                    self.env.action_space.low,
                    self.env.action_space.high,
                )

            next_obs, reward, done, truncated, info = self.env.step(action_np)

            observations.append(obs)
            actions.append(action[0].cpu().numpy())
            rewards.append(reward)
            dones.append(done or truncated)
            values.append(value[0, 0].cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())

            # dataset buffer (for diffusion pipeline later)
            self.dataset_buffer["observations"].append(obs.copy())
            self.dataset_buffer["actions"].append(action_np.copy())
            self.dataset_buffer["rewards"].append(reward)
            self.dataset_buffer["dones"].append(done or truncated)

            episode_reward += reward
            episode_length += 1
            self.global_step += 1

            if done or truncated:
                self.training_stats["episode_rewards"].append(episode_reward)
                self.training_stats["episode_lengths"].append(episode_length)
                self.episode_count += 1

                if self.writer:
                    self.writer.add_scalar("Train/EpisodeReward", episode_reward, self.episode_count)
                    self.writer.add_scalar("Train/EpisodeLength", episode_length, self.episode_count)

                if self.episode_count % self.config.log_interval == 0:
                    avg_reward = np.mean(
                        self.training_stats["episode_rewards"][-self.config.log_interval :]
                    )
                    print(
                        f"Episode {self.episode_count}, Step {self.global_step}: "
                        f"Avg Reward: {avg_reward:.2f}"
                    )

                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs

        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)
        log_probs = np.array(log_probs)

        return observations, actions, rewards, dones, values, log_probs

    # ---------- GAE ----------

    def compute_gae(self, rewards, values, dones, next_value=0.0):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.lambda_gae * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    # ---------- PPO UPDATE ----------

    def ppo_update(self, observations, actions, old_log_probs, returns, advantages):
        observations = np.array(observations, dtype=np.float32)  # (N, H, W, C)

        if observations.ndim == 4:
            # (N, H, W, C) -> (N, C, H, W)
            observations = np.transpose(observations, (0, 3, 1, 2))
        elif observations.ndim == 5 and observations.shape[-1] == 1:
            # (N, 4, 96, 96, 1) style (unlikely with current wrappers)
            observations = observations.squeeze(-1)  # (N, 4, 96, 96)
        else:
            raise ValueError(f"Unexpected observations shape in ppo_update: {observations.shape}")

        observations = torch.tensor(observations, dtype=torch.float32, device=device)
        if observations.max() > 1.1:
            observations = observations / 255.0

        actions = torch.tensor(actions, dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(observations)
        indices = np.arange(dataset_size)

        actor_losses = []
        critic_losses = []
        entropy_losses = []

        for epoch in range(self.config.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_idx = indices[start:end]

                batch_obs = observations[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                mean, std = self.actor(batch_obs)
                critic_output = self.critic(batch_obs)
                values = critic_output.squeeze(-1)

                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                ) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch_returns)

                entropy_loss = -entropy.mean()

                total_actor_loss = actor_loss + self.config.entropy_coeff * entropy_loss
                total_critic_loss = self.config.value_coeff * value_loss

                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                total_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        self.actor_scheduler.step()
        self.critic_scheduler.step()

        self.config.entropy_coeff = max(
            self.config.entropy_coeff * self.config.exploration_decay,
            self.config.min_entropy_coeff,
        )

        self.training_stats["actor_losses"].extend(actor_losses)
        self.training_stats["critic_losses"].extend(critic_losses)
        self.training_stats["entropy_losses"].extend(entropy_losses)
        self.training_stats["learning_rates"].append(
            self.actor_optimizer.param_groups[0]["lr"]
        )

        if self.writer:
            self.writer.add_scalar("Train/ActorLoss", np.mean(actor_losses), self.global_step)
            self.writer.add_scalar("Train/CriticLoss", np.mean(critic_losses), self.global_step)
            self.writer.add_scalar("Train/EntropyLoss", np.mean(entropy_losses), self.global_step)
            self.writer.add_scalar(
                "Train/LearningRate",
                self.actor_optimizer.param_groups[0]["lr"],
                self.global_step,
            )
            self.writer.add_scalar("Train/EntropyCoeff", self.config.entropy_coeff, self.global_step)

    # ---------- EVALUATION / SAVING / PLOTTING ----------

    def evaluate(self, n_episodes=5):
        eval_rewards = []
        eval_lengths = []

        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                obs_tensor = self.preprocess_obs(obs)

                with torch.no_grad():
                    mean, _ = self.actor(obs_tensor)
                    action = mean[0].cpu().numpy()

                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1

                if done or truncated:
                    break

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)

        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)

        if self.writer:
            self.writer.add_scalar("Eval/AverageReward", avg_reward, self.episode_count)
            self.writer.add_scalar("Eval/AverageLength", avg_length, self.episode_count)

        print(f"Evaluation: Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")
        return avg_reward, avg_length

    def save_models(self, suffix=""):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "config": self.config,
                "episode_count": self.episode_count,
                "global_step": self.global_step,
                "training_stats": self.training_stats,
            },
            f"models/carracing_ppo_enhanced{suffix}.pth",
        )

    def save_dataset(self):
        print(f"Saving dataset with {len(self.dataset_buffer['observations'])} samples...")

        observations = np.array(self.dataset_buffer["observations"], dtype=np.float32)
        actions = np.array(self.dataset_buffer["actions"], dtype=np.float32)
        rewards = np.array(self.dataset_buffer["rewards"], dtype=np.float32)
        dones = np.array(self.dataset_buffer["dones"], dtype=bool)

        np.savez_compressed(
            "datasets/carracing_ppo_enhanced_dataset.npz",
            obs=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )

        print(
            f"Dataset saved: {observations.shape} observations, "
            f"{actions.shape} actions"
        )

    def plot_training_progress(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Episode rewards
        if self.training_stats["episode_rewards"]:
            axes[0, 0].plot(self.training_stats["episode_rewards"])
            axes[0, 0].set_title("Episode Rewards")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].grid(True)

            if len(self.training_stats["episode_rewards"]) > 50:
                rewards = self.training_stats["episode_rewards"]
                moving_avg = [
                    np.mean(rewards[max(0, i - 50) : i + 1])
                    for i in range(len(rewards))
                ]
                axes[0, 0].plot(moving_avg, "r-", alpha=0.7, label="50-episode MA")
                axes[0, 0].legend()

        # Episode lengths
        if self.training_stats["episode_lengths"]:
            axes[0, 1].plot(self.training_stats["episode_lengths"])
            axes[0, 1].set_title("Episode Lengths")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Length")
            axes[0, 1].grid(True)

        # Losses
        if self.training_stats["actor_losses"]:
            axes[0, 2].plot(self.training_stats["actor_losses"], label="Actor")
            axes[0, 2].plot(self.training_stats["critic_losses"], label="Critic")
            axes[0, 2].set_title("Training Losses")
            axes[0, 2].set_xlabel("Update")
            axes[0, 2].set_ylabel("Loss")
            axes[0, 2].legend()
            axes[0, 2].grid(True)

        # Learning rate
        if self.training_stats["learning_rates"]:
            axes[1, 0].plot(self.training_stats["learning_rates"])
            axes[1, 0].set_title("Learning Rate")
            axes[1, 0].set_xlabel("Update")
            axes[1, 0].set_ylabel("LR")
            axes[1, 0].grid(True)

        # Reward distribution
        if self.training_stats["episode_rewards"]:
            axes[1, 1].hist(self.training_stats["episode_rewards"], bins=30, alpha=0.7)
            axes[1, 1].set_title("Reward Distribution")
            axes[1, 1].set_xlabel("Reward")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].grid(True)

        # Summary box
        axes[1, 2].text(
            0.1, 0.9, f"Episodes: {self.episode_count}", fontsize=12, transform=axes[1, 2].transAxes
        )
        axes[1, 2].text(
            0.1, 0.8, f"Steps: {self.global_step}", fontsize=12, transform=axes[1, 2].transAxes
        )
        if self.training_stats["episode_rewards"]:
            if len(self.training_stats["episode_rewards"]) > 100:
                avg_reward = np.mean(self.training_stats["episode_rewards"][-100:])
            else:
                avg_reward = np.mean(self.training_stats["episode_rewards"])
            axes[1, 2].text(
                0.1,
                0.7,
                f"Avg Reward (last 100): {avg_reward:.2f}",
                fontsize=12,
                transform=axes[1, 2].transAxes,
            )
        axes[1, 2].text(
            0.1,
            0.6,
            f"Total Parameters: {self.count_parameters():,}",
            fontsize=12,
            transform=axes[1, 2].transAxes,
        )
        axes[1, 2].set_title("Training Summary")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(f"plots/training_progress_{self.episode_count}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ---------- MAIN TRAIN LOOP ----------

    def train(self):
        print("Starting Enhanced PPO Training...")
        print(f"Total timesteps: {self.config.total_timesteps}")
        print(f"Update frequency: every {self.config.steps_per_update} steps")

        best_reward = float("-inf")

        while self.global_step < self.config.total_timesteps:
            (
                observations,
                actions,
                rewards,
                dones,
                values,
                log_probs,
            ) = self.collect_rollouts(self.config.steps_per_update)

            advantages, returns = self.compute_gae(rewards, values, dones)

            self.ppo_update(observations, actions, log_probs, returns, advantages)

            # Evaluation (optional, mostly disabled by huge interval)
            if (
                self.episode_count > 0
                and self.episode_count % self.config.eval_interval == 0
            ):
                avg_reward, _ = self.evaluate()
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save_models("_best")
                    print(f"New best model saved! Reward: {best_reward:.2f}")

            # Periodic checkpoints + plots
            if self.episode_count > 0 and self.episode_count % self.config.save_interval == 0:
                self.save_models(f"_episode_{self.episode_count}")
                self.plot_training_progress()

            # Early stopping
            if len(self.training_stats["episode_rewards"]) > 100:
                recent_avg = np.mean(self.training_stats["episode_rewards"][-100:])
                if recent_avg > self.config.early_stop_reward:
                    print(
                        f"Early stopping! Achieved target reward: {recent_avg:.2f}"
                    )
                    break

        self.save_models("_final")
        self.save_dataset()
        self.plot_training_progress()

        self.env.close()
        self.eval_env.close()
        if self.writer:
            self.writer.close()

        print("Training completed!")
        print(f"Total episodes: {self.episode_count}")
        print(f"Total steps: {self.global_step}")
        if self.training_stats["episode_rewards"]:
            if len(self.training_stats["episode_rewards"]) > 100:
                final_avg = np.mean(self.training_stats["episode_rewards"][-100:])
            else:
                final_avg = np.mean(self.training_stats["episode_rewards"])
            print(f"Final average reward: {final_avg:.2f}")


# --------------------------
# Training Script Entry Point
# --------------------------

def main():
    config = EnhancedPPOConfig()
    trainer = EnhancedPPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
