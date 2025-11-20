"""
Clean Expert Data Generator for Deep Reinforcement Learning Project
CarRacing-v2 Environment

This module provides a streamlined, production-ready expert data generation system
for the 10703 Deep RL project focusing on diffusion-based imitation learning.

Features:
- Trajectory-level data collection for diffusion policy training
- Recovery scenario generation
- Data validation and quality control
- Multiple export formats (pickle, HDF5, NPZ)
- Comprehensive statistics and visualization

Usage:
    python expert_data_generator.py --episodes 100 --output expert_data
    python expert_data_generator.py --recovery --episodes 50 --output recovery_data
"""

import argparse
import os
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class DataCollectionConfig:
    """Configuration for expert data collection."""
    episodes: int = 100
    trajectory_horizon: int = 64
    expert_model_path: str = "ppo_discrete_carracing.pt"
    output_dir: str = "expert_data"
    device: str = "auto"
    enable_recovery: bool = False
    save_videos: bool = False
    validate_data: bool = True


class ExpertDataGenerator:
    """
    Clean, production-ready expert data generator for CarRacing-v2.

    Generates high-quality expert demonstrations optimized for:
    - Behavioral cloning
    - DAgger training
    - Diffusion policy learning
    """

    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.device = self._setup_device(config.device)
        self.expert_policy = self._load_expert_policy()

        # Data storage
        self.demonstrations = []
        self.statistics = {
            'collection_start': datetime.now().isoformat(),
            'episodes_collected': 0,
            'total_steps': 0,
            'total_reward': 0.0,
            'episode_returns': [],
            'episode_lengths': [],
            'success_episodes': 0
        }

    def _setup_device(self, device_str: str) -> torch.device:
        """Setup computing device."""
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)
        print(f"Using device: {device}")
        return device

    def _load_expert_policy(self):
        """Load the expert PPO policy."""
        from ppo_expert import PPOExpertPolicy

        if not os.path.exists(self.config.expert_model_path):
            raise FileNotFoundError(
                f"Expert model not found: {self.config.expert_model_path}\n"
                "Please train an expert model first using: python ppo_expert.py"
            )

        expert = PPOExpertPolicy(
            checkpoint_path=self.config.expert_model_path,
            device_str=str(self.device)
        )
        print(f"Loaded expert policy from: {self.config.expert_model_path}")
        return expert

    def _make_environment(self, render_mode=None):
        """Create CarRacing environment with proper preprocessing."""
        from dagger_carracing_2 import make_env
        return make_env(render_mode=render_mode)

    def _preprocess_observation(self, obs):
        """Preprocess observation to match expert policy input format."""
        from dagger_carracing_2 import preprocess_obs
        return preprocess_obs(obs)

    def collect_expert_demonstrations(self) -> Dict[str, Any]:
        """
        Collect pure expert demonstrations.

        Returns:
            Collection statistics and metadata
        """
        print(f"\nCollecting {self.config.episodes} expert demonstrations...")

        env = self._make_environment(
            render_mode="rgb_array" if self.config.save_videos else None
        )

        videos = [] if self.config.save_videos else None

        for episode in tqdm(range(self.config.episodes), desc="Episodes"):
            episode_data = self._collect_single_episode(env, episode, videos)

            if episode_data:
                self.demonstrations.append(episode_data)
                self._update_statistics(episode_data)

            if (episode + 1) % 20 == 0:
                self._print_progress(episode + 1)

        env.close()

        if self.config.save_videos and videos:
            self._save_videos(videos)

        self.statistics['collection_end'] = datetime.now().isoformat()
        print(f"\nCollection complete! Generated {len(self.demonstrations)} episodes")
        return self._get_final_statistics()

    def _collect_single_episode(self, env, episode_id: int, videos: Optional[List]) -> Optional[Dict]:
        """Collect a single expert episode."""
        obs, info = env.reset()
        done = False

        episode_data = {
            'episode_id': episode_id,
            'observations': [],
            'actions': [],
            'rewards': [],
            'episode_length': 0,
            'total_reward': 0.0,
            'frames': [] if self.config.save_videos else None
        }

        while not done:
            # Get expert action
            action = self.expert_policy.get_action(obs)

            # Store transition
            episode_data['observations'].append(self._preprocess_observation(obs))
            episode_data['actions'].append(np.array(action))

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_data['rewards'].append(reward)
            episode_data['total_reward'] += reward
            episode_data['episode_length'] += 1

            # Store frame for video
            if self.config.save_videos:
                frame = env.render()
                if frame is not None:
                    episode_data['frames'].append(frame)

            obs = next_obs

        # Convert lists to arrays
        episode_data['observations'] = np.stack(episode_data['observations'])
        episode_data['actions'] = np.stack(episode_data['actions'])
        episode_data['rewards'] = np.array(episode_data['rewards'])

        # Add to videos if recording
        if self.config.save_videos and episode_data['frames']:
            videos.append(episode_data['frames'])
            episode_data.pop('frames')  # Remove to save memory

        return episode_data

    def collect_recovery_demonstrations(self, num_episodes: int = None) -> Dict[str, Any]:
        """
        Collect expert demonstrations in recovery scenarios.

        Args:
            num_episodes: Number of recovery episodes (defaults to config.episodes // 4)
        """
        if num_episodes is None:
            num_episodes = max(20, self.config.episodes // 4)

        print(f"\nCollecting {num_episodes} recovery demonstrations...")

        env = self._make_environment()
        recovery_demonstrations = []

        for episode in tqdm(range(num_episodes), desc="Recovery Episodes"):
            recovery_data = self._collect_recovery_episode(env, episode)
            if recovery_data:
                recovery_demonstrations.append(recovery_data)

        env.close()

        print(f"Recovery collection complete! Generated {len(recovery_demonstrations)} episodes")
        return recovery_demonstrations

    def _collect_recovery_episode(self, env, episode_id: int) -> Optional[Dict]:
        """Collect a single recovery episode with initial perturbations."""
        obs, info = env.reset()

        # Introduce perturbations to create challenging scenarios
        perturbation_steps = np.random.randint(20, 50)
        for _ in range(perturbation_steps):
            if np.random.random() < 0.7:
                action = env.action_space.sample()  # Random action
            else:
                action = [0, 0, 0.8]  # Emergency brake

            next_obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                break
            obs = next_obs

        # Now collect expert recovery behavior
        episode_data = {
            'episode_id': episode_id,
            'observations': [],
            'actions': [],
            'rewards': [],
            'episode_length': 0,
            'total_reward': 0.0,
            'type': 'recovery'
        }

        done = False
        max_steps = self.config.trajectory_horizon * 2  # Limit episode length

        while not done and episode_data['episode_length'] < max_steps:
            action = self.expert_policy.get_action(obs)

            episode_data['observations'].append(self._preprocess_observation(obs))
            episode_data['actions'].append(np.array(action))

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_data['rewards'].append(reward)
            episode_data['total_reward'] += reward
            episode_data['episode_length'] += 1

            obs = next_obs

        # Convert to arrays
        episode_data['observations'] = np.stack(episode_data['observations'])
        episode_data['actions'] = np.stack(episode_data['actions'])
        episode_data['rewards'] = np.array(episode_data['rewards'])

        return episode_data

    def _update_statistics(self, episode_data: Dict):
        """Update collection statistics."""
        self.statistics['episodes_collected'] += 1
        self.statistics['total_steps'] += episode_data['episode_length']
        self.statistics['total_reward'] += episode_data['total_reward']
        self.statistics['episode_returns'].append(episode_data['total_reward'])
        self.statistics['episode_lengths'].append(episode_data['episode_length'])

        # Success threshold (commonly used for CarRacing)
        if episode_data['total_reward'] > 700:
            self.statistics['success_episodes'] += 1

    def _print_progress(self, episode_num: int):
        """Print collection progress."""
        if self.statistics['episode_returns']:
            avg_return = np.mean(self.statistics['episode_returns'][-20:])
            success_rate = self.statistics['success_episodes'] / episode_num
            print(f"Episode {episode_num}: Avg Return (last 20): {avg_return:.2f}, "
                  f"Success Rate: {success_rate:.2%}")

    def _get_final_statistics(self) -> Dict[str, Any]:
        """Generate final collection statistics."""
        if not self.statistics['episode_returns']:
            return self.statistics

        returns = np.array(self.statistics['episode_returns'])
        lengths = np.array(self.statistics['episode_lengths'])

        final_stats = self.statistics.copy()
        final_stats.update({
            'avg_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'min_return': float(np.min(returns)),
            'max_return': float(np.max(returns)),
            'median_return': float(np.median(returns)),
            'success_rate': self.statistics['success_episodes'] / len(returns),
            'avg_episode_length': float(np.mean(lengths)),
            'total_transitions': sum(lengths)
        })

        return final_stats

    def save_dataset(self, format_type: str = "pickle") -> str:
        """
        Save collected dataset in specified format.

        Args:
            format_type: Output format ("pickle", "hdf5", "npz")

        Returns:
            Path to saved dataset
        """
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Prepare dataset structure
        dataset = {
            'demonstrations': self.demonstrations,
            'statistics': self._get_final_statistics(),
            'config': self.config.__dict__,
            'metadata': {
                'expert_model': self.config.expert_model_path,
                'collection_date': datetime.now().isoformat(),
                'total_episodes': len(self.demonstrations),
                'format_version': '1.0'
            }
        }

        if format_type == "pickle":
            filepath = os.path.join(self.config.output_dir, "expert_dataset.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(dataset, f)

        elif format_type == "hdf5":
            filepath = self._save_hdf5(dataset)

        elif format_type == "npz":
            filepath = self._save_npz(dataset)

        else:
            raise ValueError(f"Unsupported format: {format_type}")

        print(f"Dataset saved to: {filepath}")
        return filepath

    def _save_hdf5(self, dataset: Dict) -> str:
        """Save dataset in HDF5 format."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 format. Install with: pip install h5py")

        filepath = os.path.join(self.config.output_dir, "expert_dataset.h5")

        with h5py.File(filepath, 'w') as f:
            # Save demonstrations
            demos_group = f.create_group('demonstrations')
            for i, demo in enumerate(dataset['demonstrations']):
                demo_group = demos_group.create_group(f'episode_{i}')
                demo_group.create_dataset('observations', data=demo['observations'])
                demo_group.create_dataset('actions', data=demo['actions'])
                demo_group.create_dataset('rewards', data=demo['rewards'])
                demo_group.attrs['episode_id'] = demo['episode_id']
                demo_group.attrs['episode_length'] = demo['episode_length']
                demo_group.attrs['total_reward'] = demo['total_reward']

            # Save metadata
            f.attrs['statistics'] = json.dumps(dataset['statistics'])
            f.attrs['config'] = json.dumps(dataset['config'])
            f.attrs['metadata'] = json.dumps(dataset['metadata'])

        return filepath

    def _save_npz(self, dataset: Dict) -> str:
        """Save dataset in NPZ format."""
        filepath = os.path.join(self.config.output_dir, "expert_dataset.npz")

        # Flatten all demonstrations into arrays
        all_observations = []
        all_actions = []
        all_rewards = []
        episode_starts = [0]

        for demo in dataset['demonstrations']:
            all_observations.append(demo['observations'])
            all_actions.append(demo['actions'])
            all_rewards.append(demo['rewards'])
            episode_starts.append(episode_starts[-1] + len(demo['observations']))

        np.savez_compressed(
            filepath,
            observations=np.concatenate(all_observations),
            actions=np.concatenate(all_actions),
            rewards=np.concatenate(all_rewards),
            episode_starts=np.array(episode_starts[:-1]),
            statistics=json.dumps(dataset['statistics']),
            config=json.dumps(dataset['config']),
            metadata=json.dumps(dataset['metadata'])
        )

        return filepath

    def _save_videos(self, videos: List):
        """Save collected videos."""
        try:
            import imageio
        except ImportError:
            print("Warning: imageio not available, skipping video save")
            return

        video_dir = os.path.join(self.config.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save first 5 videos
        for i, video in enumerate(videos[:5]):
            filename = os.path.join(video_dir, f"expert_demo_{timestamp}_ep{i}.mp4")
            imageio.mimwrite(filename, video, fps=30)

        print(f"Saved {min(len(videos), 5)} videos to {video_dir}")

    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report with visualizations."""
        report_dir = os.path.join(self.config.output_dir, "analysis")
        os.makedirs(report_dir, exist_ok=True)

        stats = self._get_final_statistics()

        # Create visualizations
        self._create_performance_plots(stats, report_dir)

        # Generate text report
        report_path = os.path.join(report_dir, "collection_report.txt")
        with open(report_path, 'w') as f:
            f.write("Expert Data Collection Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Collection Date: {stats['collection_start']}\n")
            f.write(f"Expert Model: {self.config.expert_model_path}\n")
            f.write(f"Total Episodes: {stats['episodes_collected']}\n")
            f.write(f"Total Transitions: {stats['total_transitions']}\n\n")

            f.write("Performance Metrics:\n")
            f.write(f"  Average Return: {stats['avg_return']:.2f}\n")
            f.write(f"  Success Rate: {stats['success_rate']:.2%}\n")
            f.write(f"  Average Episode Length: {stats['avg_episode_length']:.1f}\n")
            f.write(f"  Return Range: [{stats['min_return']:.1f}, {stats['max_return']:.1f}]\n\n")

            f.write("Data Quality Assessment:\n")
            if stats['success_rate'] >= 0.7:
                f.write("  ✓ High success rate achieved\n")
            else:
                f.write("  ⚠ Success rate below 70%\n")

            if stats['avg_return'] >= 600:
                f.write("  ✓ Good average return\n")
            else:
                f.write("  ⚠ Average return below 600\n")

        print(f"Analysis report generated: {report_path}")
        return report_path

    def _create_performance_plots(self, stats: Dict, output_dir: str):
        """Create performance visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        returns = stats['episode_returns']
        lengths = stats['episode_lengths']

        # Episode returns over time
        axes[0, 0].plot(returns, 'b-', alpha=0.7)
        axes[0, 0].axhline(y=700, color='r', linestyle='--', label='Success threshold')
        axes[0, 0].set_title('Episode Returns Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Return distribution
        axes[0, 1].hist(returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(x=np.mean(returns), color='r', linestyle='-', label='Mean')
        axes[0, 1].set_title('Return Distribution')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()

        # Episode lengths
        axes[1, 0].plot(lengths, 'g-', alpha=0.7)
        axes[1, 0].set_title('Episode Lengths Over Time')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Length (steps)')
        axes[1, 0].grid(True, alpha=0.3)

        # Success rate (rolling window)
        window_size = min(10, len(returns) // 4)
        if window_size > 1:
            import pandas as pd
            rolling_success = pd.Series(np.array(returns) > 700).rolling(window=window_size).mean()
            axes[1, 1].plot(rolling_success, 'purple', linewidth=2)
            axes[1, 1].set_title(f'Success Rate (Rolling Window: {window_size})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_analysis.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Expert Data Generator for Deep RL Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python expert_data_generator.py --episodes 100 --output expert_data
  python expert_data_generator.py --recovery --episodes 50
  python expert_data_generator.py --episodes 200 --format hdf5 --videos
        """
    )

    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to collect')
    parser.add_argument('--output', default='expert_data',
                       help='Output directory')
    parser.add_argument('--expert_model', default='ppo_discrete_carracing.pt',
                       help='Path to expert model')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Computing device')
    parser.add_argument('--format', choices=['pickle', 'hdf5', 'npz'], default='pickle',
                       help='Output format')
    parser.add_argument('--recovery', action='store_true',
                       help='Collect recovery scenarios')
    parser.add_argument('--videos', action='store_true',
                       help='Save video recordings')
    parser.add_argument('--horizon', type=int, default=64,
                       help='Trajectory horizon for diffusion training')

    args = parser.parse_args()

    # Create configuration
    config = DataCollectionConfig(
        episodes=args.episodes,
        expert_model_path=args.expert_model,
        output_dir=args.output,
        device=args.device,
        enable_recovery=args.recovery,
        save_videos=args.videos,
        trajectory_horizon=args.horizon
    )

    print("🚀 Expert Data Generator for Deep RL Project")
    print("=" * 50)
    print(f"Episodes: {config.episodes}")
    print(f"Output: {config.output_dir}")
    print(f"Expert Model: {config.expert_model_path}")
    print(f"Recovery Mode: {config.enable_recovery}")
    print()

    try:
        # Initialize generator
        generator = ExpertDataGenerator(config)

        # Collect demonstrations
        if config.enable_recovery:
            recovery_stats = generator.collect_recovery_demonstrations()
            print(f"Recovery episodes: {len(recovery_stats)}")
        else:
            stats = generator.collect_expert_demonstrations()

        # Save dataset
        dataset_path = generator.save_dataset(args.format)

        # Generate analysis
        report_path = generator.generate_analysis_report()

        # Print summary
        final_stats = generator._get_final_statistics()
        print("\n" + "=" * 50)
        print("COLLECTION SUMMARY")
        print("=" * 50)
        print(f"Episodes Collected: {final_stats['episodes_collected']}")
        print(f"Total Transitions: {final_stats['total_transitions']}")
        print(f"Average Return: {final_stats['avg_return']:.2f}")
        print(f"Success Rate: {final_stats['success_rate']:.2%}")
        print(f"Dataset: {dataset_path}")
        print(f"Report: {report_path}")

        if final_stats['success_rate'] >= 0.7 and final_stats['avg_return'] >= 600:
            print("\n✅ High-quality dataset generated!")
        else:
            print("\n⚠️ Dataset quality may need improvement")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()