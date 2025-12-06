"""
Convert F1TENTH PPO dataset to trajectory format for diffusion training.

Adapts the CarRacing conversion pipeline for F1TENTH data:
- Handles LiDAR observations instead of images
- Processes [steering, speed] actions instead of [steer, gas, brake]
- Groups sequential data into racing trajectories
- Saves in format compatible with diffusion training

Input:  f1tenth_ppo_dataset.npz with:
        obs: (N, lidar_dim)    LiDAR observations
        actions: (N, 2)        [steering, speed]
        rewards: (N,)
        dones: (N,)

Output: data/expert_f1tenth/ with:
        states.npy:  (num_traj, T, lidar_dim)
        actions.npy: (num_traj, T, 2)
"""

import argparse
import os
import numpy as np
from typing import Tuple, List

def load_f1tenth_dataset(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load F1TENTH dataset from .npz file."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data = np.load(npz_path)

    obs = data['obs']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']

    print(f"Loaded dataset with {len(obs)} transitions")
    print(f"  Observations shape: {obs.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards range: [{rewards.min():.2f}, {rewards.max():.2f}]")
    print(f"  Episodes: {dones.sum()} completed")

    return obs, actions, rewards, dones

def segment_into_trajectories(obs: np.ndarray,
                            actions: np.ndarray,
                            rewards: np.ndarray,
                            dones: np.ndarray,
                            min_traj_length: int = 50,
                            max_traj_length: int = 1000) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Segment sequential data into racing trajectories.

    Args:
        obs: Observations array (N, obs_dim)
        actions: Actions array (N, action_dim)
        rewards: Rewards array (N,)
        dones: Done flags array (N,)
        min_traj_length: Minimum trajectory length to keep
        max_traj_length: Maximum trajectory length (truncate if longer)

    Returns:
        List of trajectory states and actions
    """
    traj_states = []
    traj_actions = []

    start_idx = 0

    for i, done in enumerate(dones):
        if done or i == len(dones) - 1:
            # End of trajectory
            end_idx = i + 1
            traj_length = end_idx - start_idx

            if traj_length >= min_traj_length:
                # Extract trajectory
                traj_obs = obs[start_idx:end_idx]
                traj_act = actions[start_idx:end_idx]

                # Truncate if too long
                if traj_length > max_traj_length:
                    traj_obs = traj_obs[:max_traj_length]
                    traj_act = traj_act[:max_traj_length]

                # Quality filter: check for reasonable reward
                traj_reward = rewards[start_idx:start_idx+len(traj_obs)]
                if traj_reward.mean() > -50:  # Filter out very poor trajectories
                    traj_states.append(traj_obs.astype(np.float32))
                    traj_actions.append(traj_act.astype(np.float32))

            start_idx = end_idx

    print(f"Extracted {len(traj_states)} trajectories")
    if len(traj_states) > 0:
        lengths = [len(traj) for traj in traj_states]
        print(f"  Trajectory lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")

    return traj_states, traj_actions

def pad_trajectories(traj_states: List[np.ndarray],
                    traj_actions: List[np.ndarray],
                    target_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad trajectories to uniform length for batch processing.

    Args:
        traj_states: List of trajectory states
        traj_actions: List of trajectory actions
        target_length: Target trajectory length (None = max length)

    Returns:
        Padded arrays (num_traj, T, dim)
    """
    if len(traj_states) == 0:
        raise ValueError("No trajectories to pad")

    # Determine target length
    if target_length is None:
        target_length = max(len(traj) for traj in traj_states)

    num_trajs = len(traj_states)
    obs_dim = traj_states[0].shape[1]
    act_dim = traj_actions[0].shape[1]

    # Initialize padded arrays
    padded_states = np.zeros((num_trajs, target_length, obs_dim), dtype=np.float32)
    padded_actions = np.zeros((num_trajs, target_length, act_dim), dtype=np.float32)

    for i, (states, actions) in enumerate(zip(traj_states, traj_actions)):
        traj_len = min(len(states), target_length)
        padded_states[i, :traj_len] = states[:traj_len]
        padded_actions[i, :traj_len] = actions[:traj_len]

    print(f"Padded trajectories to shape: states={padded_states.shape}, actions={padded_actions.shape}")
    return padded_states, padded_actions

def save_trajectory_data(states: np.ndarray,
                        actions: np.ndarray,
                        output_dir: str):
    """Save trajectory data in diffusion-compatible format."""
    os.makedirs(output_dir, exist_ok=True)

    states_path = os.path.join(output_dir, "states.npy")
    actions_path = os.path.join(output_dir, "actions.npy")

    np.save(states_path, states)
    np.save(actions_path, actions)

    print(f"Saved trajectory data to {output_dir}/")
    print(f"  states.npy: {states.shape}")
    print(f"  actions.npy: {actions.shape}")

    # Save metadata
    metadata = {
        'num_trajectories': states.shape[0],
        'trajectory_length': states.shape[1],
        'state_dim': states.shape[2],
        'action_dim': actions.shape[2],
        'total_transitions': np.prod(states.shape[:2])
    }

    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

def convert_f1tenth_dataset(npz_path: str,
                           output_dir: str,
                           min_traj_length: int = 50,
                           max_traj_length: int = 1000,
                           target_length: int = None):
    """
    Complete conversion pipeline from F1TENTH PPO data to trajectory format.

    Args:
        npz_path: Path to F1TENTH PPO dataset (.npz)
        output_dir: Output directory for trajectory data
        min_traj_length: Minimum trajectory length to keep
        max_traj_length: Maximum trajectory length
        target_length: Target padded length (None = auto)
    """
    print("Converting F1TENTH dataset to trajectory format...")

    # Load data
    obs, actions, rewards, dones = load_f1tenth_dataset(npz_path)

    # Segment into trajectories
    traj_states, traj_actions = segment_into_trajectories(
        obs, actions, rewards, dones,
        min_traj_length=min_traj_length,
        max_traj_length=max_traj_length
    )

    if len(traj_states) == 0:
        raise ValueError("No valid trajectories found. Check data quality and min_traj_length.")

    # Pad trajectories
    padded_states, padded_actions = pad_trajectories(
        traj_states, traj_actions, target_length=target_length
    )

    # Save results
    save_trajectory_data(padded_states, padded_actions, output_dir)

    print("Conversion completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Convert F1TENTH PPO dataset to trajectory format")
    parser.add_argument("--input", default="f1tenth_ppo_dataset.npz",
                       help="Input F1TENTH dataset (.npz)")
    parser.add_argument("--output", default="data/expert_f1tenth",
                       help="Output directory for trajectory data")
    parser.add_argument("--min_length", type=int, default=50,
                       help="Minimum trajectory length")
    parser.add_argument("--max_length", type=int, default=1000,
                       help="Maximum trajectory length")
    parser.add_argument("--target_length", type=int, default=None,
                       help="Target padded length (None=auto)")

    args = parser.parse_args()

    convert_f1tenth_dataset(
        npz_path=args.input,
        output_dir=args.output,
        min_traj_length=args.min_length,
        max_traj_length=args.max_length,
        target_length=args.target_length
    )

if __name__ == "__main__":
    main()