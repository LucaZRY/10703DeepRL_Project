"""
F1TENTH PPO Expert Training for Autonomous Racing

Adapted from the CarRacing PPO pipeline to work with F1TENTH gym environment.
This script trains a strong PPO expert on F1TENTH racing and saves datasets
for diffusion/distillation training.

Key differences from CarRacing:
- Uses LiDAR observations instead of camera images
- Different action space (steering angle + speed)
- F1TENTH-specific reward structure

Outputs:
    f1tenth_ppo_best.pth
    f1tenth_ppo_last.pth
    f1tenth_ppo_dataset.npz with:
        obs: (N, lidar_dim)     LiDAR scans
        actions: (N, 2)         [steering, speed]
        rewards: (N,)
        dones: (N,)
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib

# Try to import F1TENTH gym
try:
    import f1tenth_gym
except ImportError:
    print("F1TENTH gym not installed. Install with: pip install f1tenth-gym")
    print("Falling back to CarRacing environment for demonstration")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# Environment setup
# --------------------------

def make_f1tenth_env(render_mode=None):
    """
    F1TENTH racing environment with:
      - LiDAR observations (1080 points default)
      - Continuous steering and speed control
      - Realistic vehicle dynamics
    """
    try:
        # F1TENTH specific configuration
        env = gym.make(
            'f1tenth_gym:f1tenth-v0',
            map="Spielberg",  # F1 Austria track
            num_agents=1,
            timestep=0.01,
            integrator="rk4",
            render_mode=render_mode
        )
    except:
        print("F1TENTH gym not available, using CarRacing as fallback")
        from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
        env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
        env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, 96)
        env = FrameStackObservation(env, stack_size=4)

    return env

# --------------------------
# Policy Network Architecture
# --------------------------

class F1TenthPolicyNet(nn.Module):
    """
    Policy network for F1TENTH environment.
    Takes LiDAR input and outputs steering + speed actions.
    """
    def __init__(self, obs_dim, action_dim=2, hidden_sizes=[512, 256, 128]):
        super().__init__()

        # Handle different observation types
        if len(obs_dim) == 3:  # Image observations (CarRacing fallback)
            self.obs_type = "image"
            # Convolutional layers for image processing
            self.conv = nn.Sequential(
                nn.Conv2d(obs_dim[0], 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            # Calculate conv output size
            with torch.no_grad():
                dummy = torch.zeros(1, *obs_dim)
                conv_out_size = self.conv(dummy).shape[1]
            input_size = conv_out_size
        else:  # LiDAR observations (F1TENTH)
            self.obs_type = "lidar"
            input_size = obs_dim[0] if isinstance(obs_dim, tuple) else obs_dim

        # Shared layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size

        self.shared_net = nn.Sequential(*layers)

        # Policy head (mean)
        self.policy_mean = nn.Linear(prev_size, action_dim)

        # Policy head (log std)
        self.policy_logstd = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        if self.obs_type == "image" and len(obs.shape) == 4:
            x = self.conv(obs)
        else:
            x = obs

        x = self.shared_net(x)
        mean = self.policy_mean(x)

        # F1TENTH specific action bounds
        if self.obs_type == "lidar":
            # Steering: [-1, 1], Speed: [0, 1]
            mean[:, 0] = torch.tanh(mean[:, 0])  # steering
            mean[:, 1] = torch.sigmoid(mean[:, 1])  # speed (positive only)
        else:
            # CarRacing fallback: [steering, gas, brake]
            mean = torch.tanh(mean)

        std = torch.exp(self.policy_logstd.clamp(-20, 2))
        return mean, std

class F1TenthValueNet(nn.Module):
    """Value network for F1TENTH environment."""
    def __init__(self, obs_dim, hidden_sizes=[512, 256, 128]):
        super().__init__()

        # Handle different observation types
        if len(obs_dim) == 3:  # Image observations
            self.obs_type = "image"
            self.conv = nn.Sequential(
                nn.Conv2d(obs_dim[0], 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            with torch.no_grad():
                dummy = torch.zeros(1, *obs_dim)
                conv_out_size = self.conv(dummy).shape[1]
            input_size = conv_out_size
        else:  # LiDAR observations
            self.obs_type = "lidar"
            input_size = obs_dim[0] if isinstance(obs_dim, tuple) else obs_dim

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        if self.obs_type == "image" and len(obs.shape) == 4:
            x = self.conv(obs)
        else:
            x = obs
        return self.net(x)

# --------------------------
# PPO Training Configuration
# --------------------------

@dataclass
class F1TenthPPOConfig:
    # Environment
    max_episode_steps: int = 2000

    # Training
    total_timesteps: int = 2_000_000
    steps_per_epoch: int = 4096
    epochs_per_update: int = 10
    batch_size: int = 64

    # PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5

    # Logging
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50

# --------------------------
# Training Functions
# --------------------------

def compute_gae(rewards, values, dones, gamma=0.99, lambda_gae=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0

    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[i + 1]

        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lambda_gae * (1 - dones[i]) * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages, dtype=torch.float32)

def ppo_update(policy_net, value_net, policy_optimizer, value_optimizer,
               obs_batch, action_batch, old_logprob_batch, advantage_batch,
               return_batch, config):
    """Perform PPO update."""

    # Get current policy
    mean, std = policy_net(obs_batch)
    dist = torch.distributions.Normal(mean, std)

    # Compute ratio
    new_logprob = dist.log_prob(action_batch).sum(dim=-1)
    ratio = torch.exp(new_logprob - old_logprob_batch)

    # Clipped surrogate objective
    clip_advantage = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * advantage_batch
    policy_loss = -torch.min(ratio * advantage_batch, clip_advantage).mean()

    # Entropy bonus
    entropy_loss = -config.entropy_coeff * dist.entropy().mean()

    # Value loss
    value_pred = value_net(obs_batch).squeeze()
    value_loss = F.mse_loss(value_pred, return_batch)

    # Total loss
    total_loss = policy_loss + config.value_coeff * value_loss + entropy_loss

    # Update
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), config.max_grad_norm)
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), config.max_grad_norm)
    policy_optimizer.step()
    value_optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy_loss': entropy_loss.item(),
        'total_loss': total_loss.item()
    }

def train_f1tenth_ppo():
    """Main training loop for F1TENTH PPO."""
    config = F1TenthPPOConfig()

    # Environment
    env = make_f1tenth_env()
    obs, _ = env.reset()
    obs_dim = obs.shape

    # Determine action dimension based on environment
    if hasattr(env.action_space, 'shape'):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = 2  # Default for F1TENTH (steering, speed)

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")

    # Networks
    policy_net = F1TenthPolicyNet(obs_dim, action_dim).to(device)
    value_net = F1TenthValueNet(obs_dim).to(device)

    # Optimizers
    policy_optimizer = Adam(policy_net.parameters(), lr=config.lr)
    value_optimizer = Adam(value_net.parameters(), lr=config.lr)

    # Training data storage
    all_obs, all_actions, all_rewards, all_dones = [], [], [], []

    # Training loop
    episode_rewards = []
    episode_reward = 0
    episode_count = 0

    print("Starting F1TENTH PPO training...")

    for timestep in range(config.total_timesteps):
        # Collect experience
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            mean, std = policy_net(obs_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action_np = action.cpu().numpy()[0]

        # Environment step
        next_obs, reward, done, truncated, info = env.step(action_np)

        # Store data for dataset
        all_obs.append(obs.copy())
        all_actions.append(action_np.copy())
        all_rewards.append(reward)
        all_dones.append(done or truncated)

        episode_reward += reward
        obs = next_obs

        if done or truncated:
            episode_rewards.append(episode_reward)
            episode_count += 1
            episode_reward = 0
            obs, _ = env.reset()

            if episode_count % config.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-config.log_interval:])
                print(f"Episode {episode_count}, Avg Reward: {avg_reward:.2f}")

        # Save models periodically
        if timestep % (config.save_interval * 1000) == 0 and timestep > 0:
            torch.save(policy_net.state_dict(), f"f1tenth_ppo_policy_{timestep}.pth")
            torch.save(value_net.state_dict(), f"f1tenth_ppo_value_{timestep}.pth")

    env.close()

    # Save final models
    torch.save(policy_net.state_dict(), "f1tenth_ppo_best.pth")
    torch.save(value_net.state_dict(), "f1tenth_ppo_value_best.pth")

    # Save dataset
    print("Saving F1TENTH dataset...")
    np.savez_compressed(
        "f1tenth_ppo_dataset.npz",
        obs=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        rewards=np.array(all_rewards, dtype=np.float32),
        dones=np.array(all_dones, dtype=bool)
    )

    print("Training completed!")
    print(f"Final average reward: {np.mean(episode_rewards[-100:]):.2f}")
    return episode_rewards

if __name__ == "__main__":
    matplotlib.use('Agg')  # Non-interactive backend
    rewards = train_f1tenth_ppo()

    # Plot training progress
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Moving average
    if len(rewards) > 100:
        moving_avg = [np.mean(rewards[max(0, i-100):i+1]) for i in range(len(rewards))]
        plt.subplot(1, 2, 2)
        plt.plot(moving_avg)
        plt.title('Moving Average Reward (100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Avg Reward')

    plt.tight_layout()
    plt.savefig('f1tenth_training_results.png', dpi=150, bbox_inches='tight')
    print("Training plots saved to f1tenth_training_results.png")