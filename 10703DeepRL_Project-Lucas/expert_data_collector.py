"""
Expert Data Collection System for CarRacing DRL Project

This module provides comprehensive tools for collecting, processing, and managing
expert demonstration data for imitation learning and DAgger algorithms.

Usage:
    python expert_data_collector.py --mode collect --episodes 50 --output expert_data_v1
    python expert_data_collector.py --mode dagger --iterations 5 --output dagger_data_v1
    python expert_data_collector.py --mode validate --input expert_data_v1
"""

import argparse
import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ppo_expert import PPOExpertPolicy
from dagger_carracing_2 import CNNPolicy, make_env, preprocess_obs


class ExpertDataCollector:
    """
    Comprehensive data collection system for expert demonstrations.
    """

    def __init__(self, expert_model_path: str, device: str = "auto"):
        """
        Initialize the data collector.

        Args:
            expert_model_path: Path to trained expert model
            device: Computing device ('cpu', 'cuda', or 'auto')
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[DataCollector] Using device: {self.device}")

        # Load expert policy
        self.expert = PPOExpertPolicy(expert_model_path, device_str=self.device)
        print(f"[DataCollector] Loaded expert from {expert_model_path}")

        # Data storage
        self.dataset = EnhancedImitationDataset()

        # Statistics tracking
        self.collection_stats = {
            'episodes_collected': 0,
            'total_steps': 0,
            'total_reward': 0.0,
            'collection_time': 0.0,
            'episode_returns': [],
            'episode_lengths': [],
            'success_rate': 0.0
        }

    def collect_expert_episodes(self, num_episodes: int, render: bool = False,
                              save_videos: bool = False) -> Dict[str, Any]:
        """
        Collect pure expert demonstration episodes.

        Args:
            num_episodes: Number of episodes to collect
            render: Whether to render environment during collection
            save_videos: Whether to save video recordings

        Returns:
            Collection statistics and metadata
        """
        print(f"\n[Collection] Starting expert data collection: {num_episodes} episodes")

        # Setup environment
        render_mode = "rgb_array" if (render or save_videos) else None
        env = make_env(render_mode=render_mode)

        # Video storage
        all_videos = [] if save_videos else None

        start_time = time.time()

        for episode in tqdm(range(num_episodes), desc="Collecting Episodes"):
            obs, info = env.reset()
            done = False
            episode_reward = 0.0
            episode_steps = 0
            episode_frames = [] if save_videos else None

            while not done:
                # Get expert action
                expert_action = self.expert.get_action(obs)

                # Execute action
                next_obs, reward, terminated, truncated, info = env.step(expert_action)
                done = terminated or truncated

                # Store transition
                self.dataset.add_transition(
                    obs, expert_action, reward, next_obs, done,
                    episode_id=self.collection_stats['episodes_collected'] + episode,
                    step_id=episode_steps,
                    collection_method="expert_demo"
                )

                # Update statistics
                episode_reward += reward
                episode_steps += 1

                # Store frame for video
                if save_videos:
                    frame = env.render()
                    if frame is not None:
                        episode_frames.append(frame)

                obs = next_obs

            # Episode completed
            self.collection_stats['episode_returns'].append(episode_reward)
            self.collection_stats['episode_lengths'].append(episode_steps)
            self.collection_stats['total_reward'] += episode_reward
            self.collection_stats['total_steps'] += episode_steps

            if save_videos and episode_frames:
                all_videos.append(episode_frames)

            # Progress logging
            if (episode + 1) % 10 == 0 or episode == num_episodes - 1:
                avg_return = np.mean(self.collection_stats['episode_returns'][-10:])
                print(f"Episode {episode + 1}/{num_episodes}, Avg Return (last 10): {avg_return:.2f}")

        # Update collection statistics
        self.collection_stats['episodes_collected'] += num_episodes
        self.collection_stats['collection_time'] += time.time() - start_time

        # Calculate success rate (episodes with return > 700)
        successful_episodes = sum(1 for r in self.collection_stats['episode_returns'] if r > 700)
        self.collection_stats['success_rate'] = successful_episodes / len(self.collection_stats['episode_returns'])

        env.close()

        # Save videos if requested
        if save_videos and all_videos:
            self._save_videos(all_videos, "expert_demos")

        return self.get_statistics()

    def collect_dagger_iteration(self, student_policy: nn.Module, num_episodes: int,
                                beta: float = 0.5) -> Dict[str, Any]:
        """
        Collect one iteration of DAgger data.

        Args:
            student_policy: Current student policy
            num_episodes: Number of episodes for this iteration
            beta: Mixing parameter (1.0 = pure expert, 0.0 = pure student)

        Returns:
            Iteration statistics
        """
        print(f"\n[DAgger] Collection iteration with β={beta:.2f}")

        env = make_env(render_mode=None)
        student_policy.eval()

        iteration_stats = {
            'episodes': 0,
            'student_actions': 0,
            'expert_corrections': 0,
            'disagreement_rate': 0.0
        }

        start_time = time.time()

        for episode in tqdm(range(num_episodes), desc="DAgger Collection"):
            obs, info = env.reset()
            done = False
            episode_reward = 0.0
            episode_steps = 0
            disagreements = 0

            while not done:
                # Get student action
                obs_proc = preprocess_obs(obs)
                obs_tensor = torch.tensor(
                    obs_proc, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    student_action = student_policy(obs_tensor).cpu().numpy()[0]

                # Get expert action (always for labeling)
                expert_action = self.expert.get_action(obs)

                # Determine execution action
                if np.random.random() < beta:
                    execution_action = expert_action
                    iteration_stats['expert_corrections'] += 1
                else:
                    execution_action = student_action
                    iteration_stats['student_actions'] += 1

                # Measure disagreement
                action_diff = np.linalg.norm(student_action - expert_action)
                if action_diff > 0.1:  # Threshold for significant disagreement
                    disagreements += 1

                # Execute action
                next_obs, reward, terminated, truncated, info = env.step(execution_action)
                done = terminated or truncated

                # Store transition with expert label
                self.dataset.add_transition(
                    obs, expert_action, reward, next_obs, done,  # Always store expert action
                    episode_id=self.collection_stats['episodes_collected'] + episode,
                    step_id=episode_steps,
                    collection_method="dagger",
                    metadata={
                        'student_action': student_action,
                        'execution_action': execution_action,
                        'action_disagreement': action_diff
                    }
                )

                episode_reward += reward
                episode_steps += 1
                obs = next_obs

            # Episode statistics
            self.collection_stats['episode_returns'].append(episode_reward)
            self.collection_stats['episode_lengths'].append(episode_steps)
            self.collection_stats['total_reward'] += episode_reward
            self.collection_stats['total_steps'] += episode_steps

            if episode_steps > 0:
                iteration_stats['disagreement_rate'] += disagreements / episode_steps

        # Finalize iteration statistics
        iteration_stats['episodes'] = num_episodes
        iteration_stats['disagreement_rate'] /= num_episodes
        self.collection_stats['episodes_collected'] += num_episodes
        self.collection_stats['collection_time'] += time.time() - start_time

        env.close()
        return iteration_stats

    def collect_recovery_scenarios(self, num_episodes: int, trigger_conditions: List[str]) -> Dict[str, Any]:
        """
        Collect expert demonstrations in recovery scenarios.

        Args:
            num_episodes: Number of recovery episodes to collect
            trigger_conditions: List of conditions to trigger recovery collection

        Returns:
            Recovery collection statistics
        """
        print(f"\n[Recovery] Collecting recovery scenarios: {trigger_conditions}")

        env = make_env(render_mode=None)
        recovery_stats = {
            'recovery_episodes': 0,
            'trigger_counts': {condition: 0 for condition in trigger_conditions}
        }

        for episode in tqdm(range(num_episodes), desc="Recovery Collection"):
            obs, info = env.reset()

            # Introduce perturbations for recovery scenarios
            if 'off_track' in trigger_conditions:
                # Simulate off-track scenario
                for _ in range(np.random.randint(10, 30)):
                    random_action = env.action_space.sample()
                    obs, _, terminated, truncated, _ = env.step(random_action)
                    if terminated or truncated:
                        obs, info = env.reset()
                        break

            # Now collect expert recovery
            done = False
            episode_reward = 0.0
            episode_steps = 0

            while not done and episode_steps < 500:  # Limit recovery episodes
                expert_action = self.expert.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(expert_action)
                done = terminated or truncated

                # Store as recovery data
                self.dataset.add_transition(
                    obs, expert_action, reward, next_obs, done,
                    episode_id=self.collection_stats['episodes_collected'] + episode,
                    step_id=episode_steps,
                    collection_method="recovery",
                    metadata={'recovery_type': 'off_track'}
                )

                episode_reward += reward
                episode_steps += 1
                obs = next_obs

            recovery_stats['recovery_episodes'] += 1
            recovery_stats['trigger_counts']['off_track'] += 1

            self.collection_stats['episode_returns'].append(episode_reward)
            self.collection_stats['total_reward'] += episode_reward
            self.collection_stats['total_steps'] += episode_steps

        self.collection_stats['episodes_collected'] += num_episodes
        env.close()
        return recovery_stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics."""
        if len(self.collection_stats['episode_returns']) == 0:
            return self.collection_stats

        stats = self.collection_stats.copy()
        returns = np.array(self.collection_stats['episode_returns'])

        stats.update({
            'avg_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'min_return': float(np.min(returns)),
            'max_return': float(np.max(returns)),
            'median_return': float(np.median(returns)),
            'avg_episode_length': float(np.mean(self.collection_stats['episode_lengths'])),
            'dataset_size': len(self.dataset)
        })

        return stats

    def save_dataset(self, filepath: str, include_metadata: bool = True):
        """Save collected dataset to file."""
        save_data = {
            'dataset': self.dataset,
            'statistics': self.get_statistics(),
            'timestamp': datetime.now().isoformat(),
            'expert_model': self.expert,
            'device': self.device
        }

        if include_metadata:
            save_data['collection_metadata'] = {
                'python_version': str(torch.__version__),
                'numpy_version': str(np.__version__),
                'collection_script': __file__
            }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"[Save] Dataset saved to {filepath}")
        print(f"[Save] Dataset size: {len(self.dataset)} transitions")

    def _save_videos(self, videos: List[List[np.ndarray]], prefix: str):
        """Save collected videos."""
        import imageio

        os.makedirs("videos", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, video in enumerate(videos[:5]):  # Save first 5 videos
            filename = f"videos/{prefix}_{timestamp}_ep{i}.mp4"
            imageio.mimwrite(filename, video, fps=30)

        print(f"[Videos] Saved {min(len(videos), 5)} videos to videos/ directory")


class EnhancedImitationDataset:
    """
    Enhanced dataset class with metadata tracking and advanced sampling.
    """

    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize enhanced dataset.

        Args:
            max_size: Maximum dataset size (None for unlimited)
        """
        self.max_size = max_size
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []

        # Metadata tracking
        self.episode_ids = []
        self.step_ids = []
        self.collection_methods = []
        self.metadata = []

        # Index management
        self.current_size = 0
        self.write_index = 0

    def add_transition(self, obs: np.ndarray, action: np.ndarray, reward: float,
                      next_obs: np.ndarray, done: bool, episode_id: int,
                      step_id: int, collection_method: str, metadata: Optional[Dict] = None):
        """
        Add a single transition to the dataset.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Episode termination flag
            episode_id: Episode identifier
            step_id: Step within episode
            collection_method: Method used to collect this data
            metadata: Additional metadata dictionary
        """
        # Preprocess observation
        obs_processed = preprocess_obs(obs)
        next_obs_processed = preprocess_obs(next_obs)
        action_array = np.array(action, dtype=np.float32)

        if self.max_size and self.current_size >= self.max_size:
            # Circular buffer behavior
            idx = self.write_index % self.max_size
            self.observations[idx] = obs_processed
            self.actions[idx] = action_array
            self.rewards[idx] = reward
            self.next_observations[idx] = next_obs_processed
            self.dones[idx] = done
            self.episode_ids[idx] = episode_id
            self.step_ids[idx] = step_id
            self.collection_methods[idx] = collection_method
            self.metadata[idx] = metadata or {}
            self.write_index += 1
        else:
            # Append new data
            self.observations.append(obs_processed)
            self.actions.append(action_array)
            self.rewards.append(reward)
            self.next_observations.append(next_obs_processed)
            self.dones.append(done)
            self.episode_ids.append(episode_id)
            self.step_ids.append(step_id)
            self.collection_methods.append(collection_method)
            self.metadata.append(metadata or {})
            self.current_size += 1

    def __len__(self):
        """Return dataset size."""
        return self.current_size

    def sample_batch(self, batch_size: int, device: str = "cpu",
                    filter_method: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch from the dataset.

        Args:
            batch_size: Size of batch to sample
            device: Device to place tensors on
            filter_method: Filter by collection method (e.g., 'expert_demo', 'dagger')

        Returns:
            Tuple of (observations, actions) tensors
        """
        if filter_method:
            # Filter by collection method
            valid_indices = [i for i, method in enumerate(self.collection_methods)
                           if method == filter_method]
            if not valid_indices:
                raise ValueError(f"No data found for method: {filter_method}")
            indices = np.random.choice(valid_indices, size=batch_size, replace=True)
        else:
            # Sample from all data
            indices = np.random.randint(0, self.current_size, size=batch_size)

        # Gather batch data
        obs_batch = np.stack([self.observations[i] for i in indices], axis=0)
        act_batch = np.stack([self.actions[i] for i in indices], axis=0)

        # Convert to tensors
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
        act_tensor = torch.tensor(act_batch, dtype=torch.float32, device=device)

        return obs_tensor, act_tensor

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self.current_size == 0:
            return {}

        stats = {
            'total_transitions': self.current_size,
            'unique_episodes': len(set(self.episode_ids)),
            'collection_methods': dict(zip(*np.unique(self.collection_methods, return_counts=True))),
            'avg_reward': float(np.mean(self.rewards)),
            'reward_std': float(np.std(self.rewards)),
            'action_stats': {
                'mean': np.mean(self.actions, axis=0).tolist(),
                'std': np.std(self.actions, axis=0).tolist(),
                'min': np.min(self.actions, axis=0).tolist(),
                'max': np.max(self.actions, axis=0).tolist()
            }
        }

        return stats


def main():
    """Main execution function with command-line interface."""
    parser = argparse.ArgumentParser(description="Expert Data Collection System")
    parser.add_argument('--mode', choices=['collect', 'dagger', 'recovery', 'validate'],
                       required=True, help='Collection mode')
    parser.add_argument('--expert_model', default='ppo_discrete_carracing.pt',
                       help='Path to expert model')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes to collect')
    parser.add_argument('--output', required=True,
                       help='Output file path (without extension)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Computing device')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during collection')
    parser.add_argument('--save_videos', action='store_true',
                       help='Save video recordings')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of DAgger iterations')

    args = parser.parse_args()

    # Initialize collector
    collector = ExpertDataCollector(args.expert_model, args.device)

    if args.mode == 'collect':
        # Pure expert collection
        print(f"Collecting {args.episodes} expert episodes...")
        stats = collector.collect_expert_episodes(
            args.episodes, args.render, args.save_videos
        )

    elif args.mode == 'dagger':
        # DAgger collection (requires student model)
        print("DAgger mode requires a trained student model")
        print("This is a placeholder - implement student loading logic")
        return

    elif args.mode == 'recovery':
        # Recovery scenario collection
        print(f"Collecting {args.episodes} recovery episodes...")
        stats = collector.collect_recovery_scenarios(
            args.episodes, ['off_track']
        )

    elif args.mode == 'validate':
        # Validate existing dataset
        print(f"Validation mode - implement dataset loading and analysis")
        return

    # Save results
    output_path = f"{args.output}.pkl"
    collector.save_dataset(output_path)

    # Print summary
    print("\n" + "="*50)
    print("COLLECTION SUMMARY")
    print("="*50)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    print(f"\nDataset saved to: {output_path}")
    print(f"Total transitions collected: {len(collector.dataset)}")


if __name__ == "__main__":
    main()