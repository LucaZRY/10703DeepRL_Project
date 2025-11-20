"""
Diffusion-Specific Expert Data Generator for CarRacing DRL Project

This module generates expert data optimized for diffusion policy training
with trajectory-level supervision and recovery behavior modeling.

Key features:
- Trajectory-based data collection for diffusion models
- Recovery scenario generation with temporal context
- Multi-modal action distributions for uncertainty modeling
- Trajectory segmentation for hierarchical learning

Usage:
    python diffusion_expert_data_generator.py --mode trajectory --episodes 100 --horizon 64
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import os

from ppo_expert import PPOExpertPolicy
from dagger_carracing_2 import make_env, preprocess_obs


@dataclass
class TrajectorySegment:
    """
    Represents a trajectory segment for diffusion policy training.
    """
    observations: np.ndarray      # Shape: (T, 4, 84, 84)
    actions: np.ndarray          # Shape: (T, 3)
    rewards: np.ndarray          # Shape: (T,)
    segment_type: str            # 'normal', 'recovery', 'challenging'
    start_condition: Dict[str, Any]  # Initial state metadata
    performance_metric: float    # Success/quality score for this segment


class DiffusionExpertDataGenerator:
    """
    Expert data generator specifically designed for diffusion policy training.

    Focuses on:
    - Trajectory-level data collection
    - Multi-horizon prediction targets
    - Recovery behavior modeling
    - Temporal consistency preservation
    """

    def __init__(self, expert_model_path: str, device: str = "auto",
                 trajectory_horizon: int = 64):
        """
        Initialize diffusion expert data generator.

        Args:
            expert_model_path: Path to expert model
            device: Computing device
            trajectory_horizon: Length of trajectory segments for diffusion training
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[DiffusionDataGen] Using device: {self.device}")
        print(f"[DiffusionDataGen] Trajectory horizon: {trajectory_horizon}")

        # Load expert policy
        self.expert = PPOExpertPolicy(expert_model_path, device_str=self.device)

        # Configuration
        self.trajectory_horizon = trajectory_horizon
        self.overlap_ratio = 0.25  # Overlap between consecutive segments
        self.min_segment_length = trajectory_horizon // 4

        # Data storage
        self.trajectory_segments = []
        self.recovery_segments = []
        self.challenging_segments = []

        # Statistics
        self.collection_stats = {
            'total_episodes': 0,
            'total_segments': 0,
            'recovery_segments': 0,
            'challenging_segments': 0,
            'avg_episode_length': 0.0,
            'avg_segment_reward': 0.0,
            'collection_time': 0.0
        }

    def collect_trajectory_episodes(self, num_episodes: int,
                                  enable_perturbations: bool = True,
                                  save_videos: bool = False) -> Dict[str, Any]:
        """
        Collect episodes with trajectory segmentation for diffusion training.

        Args:
            num_episodes: Number of episodes to collect
            enable_perturbations: Whether to introduce perturbations for recovery data
            save_videos: Whether to save video recordings

        Returns:
            Collection statistics
        """
        print(f"\n[DiffusionDataGen] Collecting {num_episodes} trajectory episodes...")
        print(f"Perturbations enabled: {enable_perturbations}")

        env = make_env(render_mode="rgb_array" if save_videos else None)
        all_videos = [] if save_videos else None

        start_time = datetime.now()

        for episode in tqdm(range(num_episodes), desc="Collecting Episodes"):
            obs, info = env.reset()
            done = False
            episode_data = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'episode_frames': [] if save_videos else None
            }

            # Randomly introduce perturbations for some episodes
            if enable_perturbations and np.random.random() < 0.3:
                episode_data = self._collect_perturbed_episode(env, episode_data, save_videos)
            else:
                episode_data = self._collect_normal_episode(env, episode_data, save_videos)

            # Process episode into trajectory segments
            segments = self._segment_trajectory(
                episode_data['observations'],
                episode_data['actions'],
                episode_data['rewards'],
                episode_id=episode
            )

            # Store segments by type
            for segment in segments:
                if segment.segment_type == 'recovery':
                    self.recovery_segments.append(segment)
                    self.collection_stats['recovery_segments'] += 1
                elif segment.segment_type == 'challenging':
                    self.challenging_segments.append(segment)
                    self.collection_stats['challenging_segments'] += 1
                else:
                    self.trajectory_segments.append(segment)

            # Store video if requested
            if save_videos and episode_data['episode_frames']:
                all_videos.append(episode_data['episode_frames'])

            # Update statistics
            self.collection_stats['total_episodes'] += 1
            self.collection_stats['total_segments'] += len(segments)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed. "
                      f"Generated {len(segments)} segments.")

        env.close()

        # Save videos if requested
        if save_videos and all_videos:
            self._save_trajectory_videos(all_videos, "diffusion_trajectories")

        # Update final statistics
        collection_time = (datetime.now() - start_time).total_seconds()
        self.collection_stats['collection_time'] = collection_time

        total_segments = len(self.trajectory_segments) + len(self.recovery_segments) + len(self.challenging_segments)
        if total_segments > 0:
            all_rewards = []
            for segments in [self.trajectory_segments, self.recovery_segments, self.challenging_segments]:
                all_rewards.extend([seg.performance_metric for seg in segments])
            self.collection_stats['avg_segment_reward'] = np.mean(all_rewards)

        return self.collection_stats

    def _collect_normal_episode(self, env, episode_data: Dict, save_videos: bool) -> Dict:
        """Collect a normal episode with expert policy."""
        obs, _ = env.reset()
        done = False

        while not done:
            # Get expert action
            action = self.expert.get_action(obs)

            # Store transition
            episode_data['observations'].append(preprocess_obs(obs))
            episode_data['actions'].append(np.array(action))

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_data['rewards'].append(reward)

            # Store frame
            if save_videos:
                frame = env.render()
                if frame is not None:
                    episode_data['episode_frames'].append(frame)

            obs = next_obs

        return episode_data

    def _collect_perturbed_episode(self, env, episode_data: Dict, save_videos: bool) -> Dict:
        """Collect episode with perturbations to generate recovery scenarios."""
        obs, _ = env.reset()
        done = False
        step_count = 0

        # Introduce initial perturbation
        perturbation_steps = np.random.randint(20, 50)
        in_recovery = False
        recovery_start = None

        while not done:
            # Determine action source
            if step_count < perturbation_steps and not in_recovery:
                # Random perturbation phase
                if np.random.random() < 0.7:
                    action = env.action_space.sample()  # Random action
                else:
                    action = self.expert.get_action(obs)  # Mix with expert
            else:
                # Expert recovery phase
                if not in_recovery:
                    recovery_start = step_count
                    in_recovery = True
                action = self.expert.get_action(obs)

            # Store transition
            episode_data['observations'].append(preprocess_obs(obs))
            episode_data['actions'].append(np.array(action))

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_data['rewards'].append(reward)

            # Store frame
            if save_videos:
                frame = env.render()
                if frame is not None:
                    episode_data['episode_frames'].append(frame)

            obs = next_obs
            step_count += 1

        # Mark recovery period in metadata
        if recovery_start is not None:
            episode_data['recovery_start'] = recovery_start

        return episode_data

    def _segment_trajectory(self, observations: List[np.ndarray],
                           actions: List[np.ndarray],
                           rewards: List[float],
                           episode_id: int) -> List[TrajectorySegment]:
        """
        Segment episode trajectory into overlapping windows for diffusion training.

        Args:
            observations: Episode observations
            actions: Episode actions
            rewards: Episode rewards
            episode_id: Episode identifier

        Returns:
            List of trajectory segments
        """
        segments = []
        episode_length = len(observations)

        if episode_length < self.min_segment_length:
            return segments  # Skip very short episodes

        # Calculate step size with overlap
        step_size = int(self.trajectory_horizon * (1 - self.overlap_ratio))

        for start_idx in range(0, episode_length - self.min_segment_length + 1, step_size):
            end_idx = min(start_idx + self.trajectory_horizon, episode_length)
            actual_length = end_idx - start_idx

            if actual_length < self.min_segment_length:
                break

            # Extract segment data
            seg_obs = np.stack(observations[start_idx:end_idx])
            seg_actions = np.stack(actions[start_idx:end_idx])
            seg_rewards = np.array(rewards[start_idx:end_idx])

            # Determine segment type based on performance
            avg_reward = np.mean(seg_rewards)
            if avg_reward < -0.5:  # Poor performance indicates recovery scenario
                segment_type = 'recovery'
            elif np.std(seg_rewards) > 2.0:  # High variance indicates challenging scenario
                segment_type = 'challenging'
            else:
                segment_type = 'normal'

            # Create segment
            segment = TrajectorySegment(
                observations=seg_obs,
                actions=seg_actions,
                rewards=seg_rewards,
                segment_type=segment_type,
                start_condition={
                    'episode_id': episode_id,
                    'start_step': start_idx,
                    'segment_length': actual_length
                },
                performance_metric=avg_reward
            )

            segments.append(segment)

        return segments

    def generate_diffusion_training_data(self, output_dir: str = "diffusion_training_data") -> str:
        """
        Generate training data in format optimized for diffusion policy training.

        Args:
            output_dir: Output directory for training data

        Returns:
            Path to generated training data
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[DiffusionDataGen] Generating training data in {output_dir}...")

        # Combine all segments
        all_segments = (self.trajectory_segments +
                       self.recovery_segments +
                       self.challenging_segments)

        if not all_segments:
            raise ValueError("No trajectory segments available. Collect data first.")

        # Prepare training data structure
        training_data = {
            'observations': [],           # (N, T, 4, 84, 84)
            'actions': [],               # (N, T, 3)
            'rewards': [],               # (N, T)
            'segment_types': [],         # (N,) segment type labels
            'trajectory_masks': [],      # (N, T) valid timestep masks
            'metadata': {
                'horizon': self.trajectory_horizon,
                'action_dim': 3,
                'observation_shape': (4, 84, 84),
                'num_segments': len(all_segments),
                'collection_stats': self.collection_stats,
                'timestamp': datetime.now().isoformat()
            }
        }

        # Process segments
        for segment in tqdm(all_segments, desc="Processing segments"):
            # Pad sequences to fixed horizon if necessary
            T = segment.observations.shape[0]

            if T < self.trajectory_horizon:
                # Pad with last observation/action
                pad_length = self.trajectory_horizon - T

                padded_obs = np.concatenate([
                    segment.observations,
                    np.repeat(segment.observations[-1:], pad_length, axis=0)
                ])
                padded_actions = np.concatenate([
                    segment.actions,
                    np.repeat(segment.actions[-1:], pad_length, axis=0)
                ])
                padded_rewards = np.concatenate([
                    segment.rewards,
                    np.zeros(pad_length)
                ])

                # Create mask (1 for valid steps, 0 for padded)
                mask = np.concatenate([
                    np.ones(T),
                    np.zeros(pad_length)
                ])
            else:
                # Truncate to horizon length
                padded_obs = segment.observations[:self.trajectory_horizon]
                padded_actions = segment.actions[:self.trajectory_horizon]
                padded_rewards = segment.rewards[:self.trajectory_horizon]
                mask = np.ones(self.trajectory_horizon)

            # Store processed segment
            training_data['observations'].append(padded_obs)
            training_data['actions'].append(padded_actions)
            training_data['rewards'].append(padded_rewards)
            training_data['segment_types'].append(segment.segment_type)
            training_data['trajectory_masks'].append(mask)

        # Convert to numpy arrays
        training_data['observations'] = np.stack(training_data['observations'])
        training_data['actions'] = np.stack(training_data['actions'])
        training_data['rewards'] = np.stack(training_data['rewards'])
        training_data['trajectory_masks'] = np.stack(training_data['trajectory_masks'])

        # Save training data
        data_path = os.path.join(output_dir, "diffusion_training_data.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(training_data, f)

        # Save metadata separately
        metadata_path = os.path.join(output_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(training_data['metadata'], f, indent=2, default=str)

        # Generate data analysis
        self._generate_data_analysis(training_data, output_dir)

        print(f"[DiffusionDataGen] Training data saved:")
        print(f"  - Data: {data_path}")
        print(f"  - Metadata: {metadata_path}")
        print(f"  - Analysis: {output_dir}/analysis/")
        print(f"  - Shape: {training_data['observations'].shape}")

        return data_path

    def generate_recovery_focused_data(self, num_recovery_episodes: int = 50,
                                     output_dir: str = "recovery_data") -> str:
        """
        Generate data specifically focused on recovery behaviors.

        Args:
            num_recovery_episodes: Number of recovery-focused episodes
            output_dir: Output directory

        Returns:
            Path to recovery data
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[DiffusionDataGen] Generating recovery-focused data...")

        env = make_env(render_mode=None)
        recovery_trajectories = []

        for episode in tqdm(range(num_recovery_episodes), desc="Recovery Episodes"):
            # Create challenging initial conditions
            obs, _ = env.reset()

            # Introduce significant perturbation
            perturbation_length = np.random.randint(30, 80)
            trajectory_data = []

            # Perturbation phase
            for step in range(perturbation_length):
                if np.random.random() < 0.8:
                    action = env.action_space.sample()  # Random action
                else:
                    action = [0, 0, 1]  # Emergency brake

                next_obs, reward, terminated, truncated, info = env.step(action)

                trajectory_data.append({
                    'obs': preprocess_obs(obs),
                    'action': np.array(action),
                    'reward': reward,
                    'phase': 'perturbation'
                })

                if terminated or truncated:
                    break
                obs = next_obs

            # Recovery phase with expert
            if not (terminated or truncated):
                recovery_start = len(trajectory_data)
                max_recovery_steps = self.trajectory_horizon * 2

                for step in range(max_recovery_steps):
                    action = self.expert.get_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)

                    trajectory_data.append({
                        'obs': preprocess_obs(obs),
                        'action': np.array(action),
                        'reward': reward,
                        'phase': 'recovery'
                    })

                    if terminated or truncated or reward > 0:  # Successful recovery
                        break
                    obs = next_obs

            # Store trajectory with recovery metadata
            if len(trajectory_data) > self.min_segment_length:
                recovery_trajectories.append({
                    'trajectory': trajectory_data,
                    'recovery_start': recovery_start if 'recovery_start' in locals() else len(trajectory_data),
                    'total_length': len(trajectory_data),
                    'episode_id': episode
                })

        env.close()

        # Process recovery trajectories
        recovery_data = self._process_recovery_trajectories(recovery_trajectories)

        # Save recovery data
        recovery_path = os.path.join(output_dir, "recovery_training_data.pkl")
        with open(recovery_path, 'wb') as f:
            pickle.dump(recovery_data, f)

        print(f"[DiffusionDataGen] Recovery data saved: {recovery_path}")
        print(f"Recovery trajectories: {len(recovery_trajectories)}")

        return recovery_path

    def _process_recovery_trajectories(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """Process recovery trajectories for training."""
        processed_data = {
            'pre_recovery_obs': [],      # Observations before recovery
            'recovery_actions': [],      # Expert recovery actions
            'recovery_outcomes': [],     # Success/failure labels
            'trajectory_embeddings': [], # Context embeddings
            'metadata': {
                'num_trajectories': len(trajectories),
                'avg_recovery_length': 0,
                'recovery_success_rate': 0
            }
        }

        recovery_lengths = []
        success_count = 0

        for traj_data in trajectories:
            trajectory = traj_data['trajectory']
            recovery_start = traj_data['recovery_start']

            if recovery_start < len(trajectory):
                # Extract pre-recovery context
                context_length = min(32, recovery_start)
                context_start = max(0, recovery_start - context_length)

                pre_recovery_obs = [step['obs'] for step in trajectory[context_start:recovery_start]]

                # Extract recovery sequence
                recovery_steps = trajectory[recovery_start:]
                recovery_actions = [step['action'] for step in recovery_steps]

                # Determine success (positive reward achieved)
                recovery_rewards = [step['reward'] for step in recovery_steps]
                success = any(r > 0 for r in recovery_rewards)

                if len(recovery_actions) > 0:
                    processed_data['pre_recovery_obs'].append(np.stack(pre_recovery_obs))
                    processed_data['recovery_actions'].append(np.stack(recovery_actions))
                    processed_data['recovery_outcomes'].append(success)

                    recovery_lengths.append(len(recovery_actions))
                    if success:
                        success_count += 1

        # Update metadata
        if recovery_lengths:
            processed_data['metadata']['avg_recovery_length'] = np.mean(recovery_lengths)
            processed_data['metadata']['recovery_success_rate'] = success_count / len(recovery_lengths)

        return processed_data

    def _generate_data_analysis(self, training_data: Dict, output_dir: str):
        """Generate comprehensive analysis of collected data."""
        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        # Segment type distribution
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 3, 1)
        segment_types = training_data['segment_types']
        type_counts = {t: segment_types.count(t) for t in set(segment_types)}
        plt.bar(type_counts.keys(), type_counts.values())
        plt.title('Segment Type Distribution')
        plt.ylabel('Count')

        # Reward distributions
        plt.subplot(2, 3, 2)
        all_rewards = training_data['rewards'].flatten()
        valid_rewards = all_rewards[training_data['trajectory_masks'].flatten() == 1]
        plt.hist(valid_rewards, bins=50, alpha=0.7)
        plt.title('Reward Distribution')
        plt.xlabel('Reward')

        # Action distributions
        plt.subplot(2, 3, 3)
        actions = training_data['actions']
        masks = training_data['trajectory_masks']
        valid_actions = actions[masks == 1]

        for i, name in enumerate(['Steering', 'Gas', 'Brake']):
            plt.hist(valid_actions[:, i], bins=30, alpha=0.5, label=name)
        plt.title('Action Distributions')
        plt.legend()

        # Trajectory length distribution
        plt.subplot(2, 3, 4)
        traj_lengths = np.sum(training_data['trajectory_masks'], axis=1)
        plt.hist(traj_lengths, bins=20, alpha=0.7)
        plt.title('Trajectory Length Distribution')
        plt.xlabel('Length')

        # Performance by segment type
        plt.subplot(2, 3, 5)
        segment_performance = {}
        for i, seg_type in enumerate(training_data['segment_types']):
            mask = training_data['trajectory_masks'][i]
            rewards = training_data['rewards'][i][mask == 1]
            if seg_type not in segment_performance:
                segment_performance[seg_type] = []
            segment_performance[seg_type].extend(rewards)

        for seg_type, rewards in segment_performance.items():
            plt.hist(rewards, alpha=0.5, label=seg_type, bins=20)
        plt.title('Performance by Segment Type')
        plt.legend()

        # Save analysis
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "data_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Generate summary statistics
        stats = {
            'total_segments': len(training_data['segment_types']),
            'segment_type_counts': type_counts,
            'avg_trajectory_length': float(np.mean(traj_lengths)),
            'avg_reward': float(np.mean(valid_rewards)),
            'action_ranges': {
                'steering': [float(np.min(valid_actions[:, 0])), float(np.max(valid_actions[:, 0]))],
                'gas': [float(np.min(valid_actions[:, 1])), float(np.max(valid_actions[:, 1]))],
                'brake': [float(np.min(valid_actions[:, 2])), float(np.max(valid_actions[:, 2]))]
            }
        }

        with open(os.path.join(analysis_dir, "statistics.json"), 'w') as f:
            json.dump(stats, f, indent=2)

    def _save_trajectory_videos(self, videos: List[List[np.ndarray]], prefix: str):
        """Save trajectory videos."""
        import imageio
        os.makedirs("videos", exist_ok=True)

        for i, video in enumerate(videos[:5]):  # Save first 5
            filename = f"videos/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ep{i}.mp4"
            imageio.mimwrite(filename, video, fps=30)

        print(f"Saved {min(len(videos), 5)} trajectory videos to videos/")

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        total_segments = (len(self.trajectory_segments) +
                         len(self.recovery_segments) +
                         len(self.challenging_segments))

        if total_segments == 0:
            return self.collection_stats

        stats = self.collection_stats.copy()

        # Segment statistics
        stats.update({
            'normal_segments': len(self.trajectory_segments),
            'recovery_segments': len(self.recovery_segments),
            'challenging_segments': len(self.challenging_segments),
            'total_segments': total_segments
        })

        # Performance statistics
        all_performance = []
        for segments in [self.trajectory_segments, self.recovery_segments, self.challenging_segments]:
            all_performance.extend([seg.performance_metric for seg in segments])

        if all_performance:
            stats.update({
                'avg_performance': float(np.mean(all_performance)),
                'std_performance': float(np.std(all_performance)),
                'min_performance': float(np.min(all_performance)),
                'max_performance': float(np.max(all_performance))
            })

        return stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Diffusion Expert Data Generator")
    parser.add_argument('--mode', choices=['trajectory', 'recovery', 'both'],
                       default='trajectory', help='Data collection mode')
    parser.add_argument('--expert_model', default='ppo_discrete_carracing.pt',
                       help='Path to expert model')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to collect')
    parser.add_argument('--horizon', type=int, default=64,
                       help='Trajectory horizon for diffusion training')
    parser.add_argument('--output_dir', default='diffusion_data',
                       help='Output directory for generated data')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Computing device')
    parser.add_argument('--save_videos', action='store_true',
                       help='Save video recordings')
    parser.add_argument('--perturbations', action='store_true', default=True,
                       help='Enable perturbations for recovery data')

    args = parser.parse_args()

    # Initialize generator
    generator = DiffusionExpertDataGenerator(
        expert_model_path=args.expert_model,
        device=args.device,
        trajectory_horizon=args.horizon
    )

    try:
        if args.mode in ['trajectory', 'both']:
            # Collect trajectory data
            print(f"Collecting trajectory data with horizon {args.horizon}...")
            stats = generator.collect_trajectory_episodes(
                num_episodes=args.episodes,
                enable_perturbations=args.perturbations,
                save_videos=args.save_videos
            )

            # Generate training data
            training_data_path = generator.generate_diffusion_training_data(
                output_dir=args.output_dir
            )

            print(f"\nTrajectory collection completed:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

        if args.mode in ['recovery', 'both']:
            # Collect recovery-focused data
            recovery_episodes = args.episodes // 2 if args.mode == 'both' else args.episodes
            recovery_data_path = generator.generate_recovery_focused_data(
                num_recovery_episodes=recovery_episodes,
                output_dir=f"{args.output_dir}_recovery"
            )

        # Print final statistics
        final_stats = generator.get_dataset_statistics()
        print(f"\n{'='*60}")
        print("DIFFUSION DATA GENERATION COMPLETE")
        print("="*60)

        for key, value in final_stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")

        print(f"\nGenerated data optimized for diffusion policy training:")
        print(f"- Trajectory horizon: {args.horizon}")
        print(f"- Temporal consistency preserved")
        print(f"- Recovery behaviors included")
        print(f"- Multi-modal action distributions captured")

    except FileNotFoundError:
        print(f"Error: Expert model '{args.expert_model}' not found.")
        print("Train an expert model first or adjust the path.")
    except Exception as e:
        print(f"Error during data generation: {e}")


if __name__ == "__main__":
    main()