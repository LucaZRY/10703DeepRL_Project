"""
F1TENTH Environment Wrappers

Custom wrappers for F1TENTH gym environment to:
1. Standardize observation preprocessing
2. Handle LiDAR data efficiently
3. Ensure compatibility with diffusion pipeline
4. Provide fallback to CarRacing if F1TENTH not available
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any
import torch

class F1TenthLidarWrapper(gym.ObservationWrapper):
    """
    Wrapper for F1TENTH LiDAR preprocessing.

    Features:
    - Normalize LiDAR ranges
    - Handle infinite values
    - Optional downsampling for efficiency
    - Add velocity information if available
    """

    def __init__(self, env,
                 lidar_range_max: float = 30.0,
                 downsample_factor: int = 1,
                 add_velocity: bool = True,
                 normalize: bool = True):
        super().__init__(env)
        self.lidar_range_max = lidar_range_max
        self.downsample_factor = downsample_factor
        self.add_velocity = add_velocity
        self.normalize = normalize

        # Get original observation space
        if isinstance(env.observation_space, gym.spaces.Dict):
            # F1TENTH typically provides dict observations
            lidar_space = env.observation_space['scans']
            self.original_lidar_dim = lidar_space.shape[0]
        else:
            # Fallback for simple Box space
            self.original_lidar_dim = env.observation_space.shape[0]

        # Calculate new dimensions
        self.lidar_dim = self.original_lidar_dim // downsample_factor

        # Additional features (velocity, pose info)
        self.extra_dim = 3 if add_velocity else 0  # [vx, vy, angular_velocity]

        # New observation space
        total_dim = self.lidar_dim + self.extra_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )

    def observation(self, obs):
        """Process F1TENTH observation."""
        if isinstance(obs, dict):
            # Extract LiDAR data
            lidar = obs['scans'][0]  # First agent

            # Handle infinite values
            lidar = np.clip(lidar, 0, self.lidar_range_max)

            # Downsample if requested
            if self.downsample_factor > 1:
                indices = np.arange(0, len(lidar), self.downsample_factor)
                lidar = lidar[indices]

            # Normalize to [0, 1]
            if self.normalize:
                lidar = lidar / self.lidar_range_max

            # Add velocity information if available
            if self.add_velocity and 'poses_x' in obs and 'poses_y' in obs and 'poses_theta' in obs:
                # Simple velocity estimation (would need proper state for actual velocity)
                # For now, just add placeholder velocity features
                velocity_features = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                observation = np.concatenate([lidar, velocity_features])
            else:
                observation = lidar

        else:
            # Fallback for non-dict observations
            observation = obs.astype(np.float32)
            if self.normalize and observation.max() > 10:  # Likely raw LiDAR
                observation = np.clip(observation, 0, self.lidar_range_max) / self.lidar_range_max

        return observation.astype(np.float32)

class F1TenthActionWrapper(gym.ActionWrapper):
    """
    Wrapper for F1TENTH action space standardization.

    Converts various action formats to consistent [steering, speed] format
    and applies safety constraints.
    """

    def __init__(self, env,
                 max_steering: float = 0.4189,  # ~24 degrees in radians
                 max_speed: float = 8.0,        # m/s
                 min_speed: float = 0.0):
        super().__init__(env)
        self.max_steering = max_steering
        self.max_speed = max_speed
        self.min_speed = min_speed

        # Standardize action space to [steering, speed]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def action(self, action):
        """Convert normalized action to F1TENTH format."""
        action = np.array(action, dtype=np.float32)

        # Ensure action is in correct format
        if len(action) != 2:
            raise ValueError(f"Expected 2D action [steering, speed], got {action.shape}")

        # Convert from normalized [-1,1] and [0,1] to actual ranges
        steering = action[0] * self.max_steering
        speed = action[1] * (self.max_speed - self.min_speed) + self.min_speed

        # F1TENTH expects dict format for multi-agent
        return {
            'drive_0': [speed, steering]  # [speed, steering] for agent 0
        }

class F1TenthRewardWrapper(gym.RewardWrapper):
    """
    Custom reward shaping for F1TENTH racing.

    Combines:
    - Progress reward (distance traveled)
    - Speed reward (encouraging faster completion)
    - Safety penalty (collision, going off-track)
    - Efficiency bonus (smooth steering, optimal racing line)
    """

    def __init__(self, env,
                 progress_weight: float = 1.0,
                 speed_weight: float = 0.1,
                 collision_penalty: float = -100.0,
                 smoothness_weight: float = 0.05):
        super().__init__(env)
        self.progress_weight = progress_weight
        self.speed_weight = speed_weight
        self.collision_penalty = collision_penalty
        self.smoothness_weight = smoothness_weight

        self.prev_action = None
        self.prev_position = None

    def reward(self, reward):
        """Enhanced reward calculation."""
        # Start with base reward (typically progress in F1TENTH)
        total_reward = reward * self.progress_weight

        # Add custom reward components here
        # (Would need access to env state for full implementation)

        return total_reward

def make_f1tenth_env(map_name: str = "Spielberg",
                     render_mode: str = None,
                     lidar_downsample: int = 2,
                     max_episode_steps: int = 2000) -> gym.Env:
    """
    Create wrapped F1TENTH environment with all preprocessing.

    Args:
        map_name: F1 track to use ("Spielberg", "Monaco", etc.)
        render_mode: Rendering mode (None, "human", "rgb_array")
        lidar_downsample: Factor to downsample LiDAR (1=no downsampling)
        max_episode_steps: Maximum steps per episode

    Returns:
        Fully wrapped F1TENTH environment
    """
    try:
        import f1tenth_gym

        # Create base F1TENTH environment
        env = gym.make(
            'f1tenth_gym:f1tenth-v0',
            map=map_name,
            num_agents=1,
            timestep=0.01,
            integrator="rk4",
            render_mode=render_mode
        )

        print(f"Created F1TENTH environment with map: {map_name}")

    except ImportError:
        print("F1TENTH gym not available, using CarRacing as fallback")
        from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation

        env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
        env = GrayscaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, 96)
        env = FrameStackObservation(env, stack_size=4)

        # Return early for CarRacing (different pipeline)
        return gym.wrappers.TimeLimit(env, max_episode_steps)

    # Apply F1TENTH specific wrappers
    env = F1TenthLidarWrapper(env, downsample_factor=lidar_downsample)
    env = F1TenthActionWrapper(env)
    env = F1TenthRewardWrapper(env)

    # Time limit wrapper
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    return env

def get_f1tenth_maps():
    """Return list of available F1TENTH maps."""
    try:
        import f1tenth_gym
        # Common F1TENTH maps
        return [
            "Spielberg",    # Austria
            "Monaco",       # Monaco
            "Silverstone",  # UK
            "Spa",          # Belgium
            "Monza",        # Italy
            "Suzuka",       # Japan
            "LVMS",         # Las Vegas
            "Nurburgring",  # Germany
        ]
    except ImportError:
        return ["CarRacing-v2"]  # Fallback

# Test the environment wrapper
if __name__ == "__main__":
    # Test environment creation
    env = make_f1tenth_env(map_name="Spielberg", render_mode=None)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test a few steps
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: obs_shape={obs.shape}, reward={reward:.3f}, done={done}")

        if done or truncated:
            obs, _ = env.reset()

    env.close()
    print("Environment test completed successfully!")