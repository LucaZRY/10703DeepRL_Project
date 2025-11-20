"""
Data Validation and Quality Control Tools for Expert Data

This module provides comprehensive validation tools for expert demonstration data,
including performance analysis, data integrity checks, and visualization utilities.

Usage:
    python data_validation_tools.py --input expert_data_v1.pkl --output validation_report
    python data_validation_tools.py --compare expert_data_v1.pkl expert_data_v2.pkl
"""

import argparse
import pickle
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import torch
import json
from datetime import datetime


class DataValidator:
    """
    Comprehensive data validation and quality control system.
    """

    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize the data validator.

        Args:
            dataset_path: Path to dataset file to load
        """
        self.dataset = None
        self.statistics = None
        self.validation_results = {}

        if dataset_path:
            self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path: str):
        """Load dataset from pickle file."""
        try:
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)

            self.dataset = data['dataset']
            self.statistics = data.get('statistics', {})
            print(f"[Loaded] Dataset from {dataset_path}")
            print(f"[Loaded] Size: {len(self.dataset)} transitions")

        except Exception as e:
            print(f"[Error] Failed to load dataset: {e}")
            raise

    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Perform comprehensive data integrity checks.

        Returns:
            Dictionary with validation results
        """
        print("\n[Validation] Checking data integrity...")

        integrity_results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }

        # Check basic structure
        required_attrs = ['observations', 'actions', 'rewards', 'next_observations', 'dones']
        for attr in required_attrs:
            if not hasattr(self.dataset, attr):
                integrity_results['errors'].append(f"Missing required attribute: {attr}")
                integrity_results['passed'] = False

        if not integrity_results['passed']:
            return integrity_results

        # Check data consistency
        data_lengths = {
            'observations': len(self.dataset.observations),
            'actions': len(self.dataset.actions),
            'rewards': len(self.dataset.rewards),
            'next_observations': len(self.dataset.next_observations),
            'dones': len(self.dataset.dones)
        }

        unique_lengths = set(data_lengths.values())
        if len(unique_lengths) > 1:
            integrity_results['errors'].append(f"Inconsistent data lengths: {data_lengths}")
            integrity_results['passed'] = False

        integrity_results['checks']['data_lengths'] = data_lengths

        # Check observation shapes
        obs_shapes = [obs.shape for obs in self.dataset.observations[:100]]  # Check first 100
        unique_shapes = set(obs_shapes)
        if len(unique_shapes) > 1:
            integrity_results['warnings'].append(f"Multiple observation shapes found: {unique_shapes}")

        expected_shape = (4, 84, 84)
        if expected_shape not in unique_shapes:
            integrity_results['errors'].append(f"Expected observation shape {expected_shape} not found")
            integrity_results['passed'] = False

        integrity_results['checks']['observation_shapes'] = list(unique_shapes)

        # Check action ranges
        all_actions = np.array(self.dataset.actions)
        action_mins = np.min(all_actions, axis=0)
        action_maxs = np.max(all_actions, axis=0)

        # Expected ranges: steer[-1,1], gas[0,1], brake[0,1]
        expected_ranges = [(-1.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        for i, (min_val, max_val) in enumerate(expected_ranges):
            if action_mins[i] < min_val - 0.01 or action_maxs[i] > max_val + 0.01:
                integrity_results['warnings'].append(
                    f"Action {i} out of expected range [{min_val}, {max_val}]: "
                    f"actual range [{action_mins[i]:.3f}, {action_maxs[i]:.3f}]"
                )

        integrity_results['checks']['action_ranges'] = {
            'mins': action_mins.tolist(),
            'maxs': action_maxs.tolist(),
            'expected': expected_ranges
        }

        # Check for NaN or infinite values
        nan_checks = {}
        for attr in ['observations', 'actions', 'rewards']:
            data_array = getattr(self.dataset, attr)
            if attr == 'observations':
                # Check a sample of observations
                sample_obs = np.array([obs for obs in data_array[:1000]])
                nan_count = np.isnan(sample_obs).sum()
                inf_count = np.isinf(sample_obs).sum()
            else:
                data_array = np.array(data_array)
                nan_count = np.isnan(data_array).sum()
                inf_count = np.isinf(data_array).sum()

            nan_checks[attr] = {'nan_count': int(nan_count), 'inf_count': int(inf_count)}

            if nan_count > 0 or inf_count > 0:
                integrity_results['errors'].append(
                    f"Found {nan_count} NaN and {inf_count} inf values in {attr}"
                )
                integrity_results['passed'] = False

        integrity_results['checks']['nan_inf_checks'] = nan_checks

        # Check reward distribution
        rewards = np.array(self.dataset.rewards)
        reward_stats = {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'negative_fraction': float(np.mean(rewards < 0))
        }

        if reward_stats['negative_fraction'] > 0.8:
            integrity_results['warnings'].append(
                f"High fraction of negative rewards: {reward_stats['negative_fraction']:.2f}"
            )

        integrity_results['checks']['reward_stats'] = reward_stats

        self.validation_results['integrity'] = integrity_results
        return integrity_results

    def analyze_expert_performance(self) -> Dict[str, Any]:
        """
        Analyze expert policy performance from collected data.

        Returns:
            Performance analysis results
        """
        print("\n[Analysis] Analyzing expert performance...")

        if not hasattr(self.dataset, 'episode_ids'):
            print("[Warning] No episode metadata available for performance analysis")
            return {}

        # Group data by episodes
        episode_data = {}
        for i in range(len(self.dataset.episode_ids)):
            ep_id = self.dataset.episode_ids[i]
            if ep_id not in episode_data:
                episode_data[ep_id] = {
                    'rewards': [],
                    'actions': [],
                    'episode_length': 0
                }

            episode_data[ep_id]['rewards'].append(self.dataset.rewards[i])
            episode_data[ep_id]['actions'].append(self.dataset.actions[i])
            episode_data[ep_id]['episode_length'] += 1

        # Calculate episode returns
        episode_returns = []
        episode_lengths = []
        for ep_id, data in episode_data.items():
            episode_returns.append(sum(data['rewards']))
            episode_lengths.append(data['episode_length'])

        # Performance statistics
        performance_stats = {
            'num_episodes': len(episode_returns),
            'avg_return': float(np.mean(episode_returns)),
            'std_return': float(np.std(episode_returns)),
            'min_return': float(np.min(episode_returns)),
            'max_return': float(np.max(episode_returns)),
            'median_return': float(np.median(episode_returns)),
            'success_rate': float(np.mean(np.array(episode_returns) > 700)),  # Success threshold
            'avg_episode_length': float(np.mean(episode_lengths)),
            'std_episode_length': float(np.std(episode_lengths))
        }

        # Action analysis
        all_actions = np.array(self.dataset.actions)
        action_stats = {
            'steer_stats': {
                'mean': float(np.mean(all_actions[:, 0])),
                'std': float(np.std(all_actions[:, 0])),
                'abs_mean': float(np.mean(np.abs(all_actions[:, 0])))
            },
            'gas_stats': {
                'mean': float(np.mean(all_actions[:, 1])),
                'std': float(np.std(all_actions[:, 1])),
                'usage_rate': float(np.mean(all_actions[:, 1] > 0.1))
            },
            'brake_stats': {
                'mean': float(np.mean(all_actions[:, 2])),
                'std': float(np.std(all_actions[:, 2])),
                'usage_rate': float(np.mean(all_actions[:, 2] > 0.1))
            }
        }

        performance_results = {
            'episode_performance': performance_stats,
            'action_analysis': action_stats,
            'episode_returns': episode_returns,
            'episode_lengths': episode_lengths
        }

        self.validation_results['performance'] = performance_results
        return performance_results

    def detect_anomalies(self, threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect anomalous data points using statistical methods.

        Args:
            threshold: Z-score threshold for anomaly detection

        Returns:
            Anomaly detection results
        """
        print(f"\n[Anomaly Detection] Using threshold: {threshold}")

        anomaly_results = {
            'action_anomalies': [],
            'reward_anomalies': [],
            'episode_anomalies': []
        }

        # Action anomalies
        all_actions = np.array(self.dataset.actions)
        for i in range(all_actions.shape[1]):
            action_col = all_actions[:, i]
            z_scores = np.abs(stats.zscore(action_col))
            anomalous_indices = np.where(z_scores > threshold)[0]

            if len(anomalous_indices) > 0:
                anomaly_results['action_anomalies'].extend([
                    {
                        'index': int(idx),
                        'action_dim': i,
                        'value': float(action_col[idx]),
                        'z_score': float(z_scores[idx])
                    }
                    for idx in anomalous_indices
                ])

        # Reward anomalies
        rewards = np.array(self.dataset.rewards)
        reward_z_scores = np.abs(stats.zscore(rewards))
        anomalous_reward_indices = np.where(reward_z_scores > threshold)[0]

        anomaly_results['reward_anomalies'].extend([
            {
                'index': int(idx),
                'reward': float(rewards[idx]),
                'z_score': float(reward_z_scores[idx])
            }
            for idx in anomalous_reward_indices
        ])

        # Episode-level anomalies (if metadata available)
        if hasattr(self.dataset, 'episode_ids') and len(set(self.dataset.episode_ids)) > 1:
            episode_returns = self.validation_results.get('performance', {}).get('episode_returns', [])
            if episode_returns:
                episode_z_scores = np.abs(stats.zscore(episode_returns))
                anomalous_episodes = np.where(episode_z_scores > threshold)[0]

                anomaly_results['episode_anomalies'].extend([
                    {
                        'episode_id': int(idx),
                        'return': float(episode_returns[idx]),
                        'z_score': float(episode_z_scores[idx])
                    }
                    for idx in anomalous_episodes
                ])

        # Summary statistics
        anomaly_results['summary'] = {
            'total_action_anomalies': len(anomaly_results['action_anomalies']),
            'total_reward_anomalies': len(anomaly_results['reward_anomalies']),
            'total_episode_anomalies': len(anomaly_results['episode_anomalies']),
            'anomaly_rate': len(anomaly_results['action_anomalies']) / len(self.dataset),
            'threshold_used': threshold
        }

        self.validation_results['anomalies'] = anomaly_results
        return anomaly_results

    def create_visualizations(self, output_dir: str):
        """
        Create comprehensive visualizations of the dataset.

        Args:
            output_dir: Directory to save visualization plots
        """
        print(f"\n[Visualization] Creating plots in {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Episode performance plot
        if 'performance' in self.validation_results:
            self._plot_episode_performance(output_dir)

        # 2. Action distribution plots
        self._plot_action_distributions(output_dir)

        # 3. Reward distribution
        self._plot_reward_distribution(output_dir)

        # 4. Data integrity summary
        if 'integrity' in self.validation_results:
            self._plot_integrity_summary(output_dir)

        # 5. Anomaly visualization
        if 'anomalies' in self.validation_results:
            self._plot_anomalies(output_dir)

        print(f"[Visualization] Plots saved to {output_dir}")

    def _plot_episode_performance(self, output_dir: str):
        """Plot episode performance metrics."""
        perf_data = self.validation_results['performance']
        episode_returns = perf_data['episode_returns']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Episode returns over time
        axes[0, 0].plot(episode_returns, 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].axhline(y=700, color='r', linestyle='--', alpha=0.8, label='Success threshold')
        axes[0, 0].set_title('Episode Returns Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Return distribution
        axes[0, 1].hist(episode_returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(x=np.mean(episode_returns), color='r', linestyle='-', alpha=0.8, label='Mean')
        axes[0, 1].axvline(x=700, color='orange', linestyle='--', alpha=0.8, label='Success threshold')
        axes[0, 1].set_title('Episode Return Distribution')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()

        # Episode lengths
        episode_lengths = perf_data['episode_lengths']
        axes[1, 0].plot(episode_lengths, 'g-', alpha=0.7, linewidth=1)
        axes[1, 0].set_title('Episode Lengths Over Time')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Length (steps)')
        axes[1, 0].grid(True, alpha=0.3)

        # Success rate (rolling window)
        window_size = min(10, len(episode_returns) // 4)
        if window_size > 1:
            rolling_success = pd.Series(np.array(episode_returns) > 700).rolling(window=window_size).mean()
            axes[1, 1].plot(rolling_success, 'purple', linewidth=2)
            axes[1, 1].set_title(f'Success Rate (Rolling Window: {window_size})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/episode_performance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_action_distributions(self, output_dir: str):
        """Plot action distributions."""
        all_actions = np.array(self.dataset.actions)
        action_names = ['Steering', 'Gas', 'Brake']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for i, name in enumerate(action_names):
            # Distribution histogram
            axes[0, i].hist(all_actions[:, i], bins=50, alpha=0.7, color=f'C{i}', edgecolor='black')
            axes[0, i].set_title(f'{name} Distribution')
            axes[0, i].set_xlabel(f'{name} Value')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].grid(True, alpha=0.3)

            # Time series (sample)
            sample_size = min(1000, len(all_actions))
            indices = np.linspace(0, len(all_actions)-1, sample_size, dtype=int)
            axes[1, i].plot(indices, all_actions[indices, i], alpha=0.6, linewidth=1, color=f'C{i}')
            axes[1, i].set_title(f'{name} Over Time (Sample)')
            axes[1, i].set_xlabel('Step')
            axes[1, i].set_ylabel(f'{name} Value')
            axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/action_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_reward_distribution(self, output_dir: str):
        """Plot reward distribution analysis."""
        rewards = np.array(self.dataset.rewards)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Reward histogram
        axes[0, 0].hist(rewards, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 0].axvline(x=np.mean(rewards), color='r', linestyle='-', alpha=0.8, label='Mean')
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Cumulative reward (sample)
        sample_size = min(1000, len(rewards))
        indices = np.linspace(0, len(rewards)-1, sample_size, dtype=int)
        cumulative_reward = np.cumsum(rewards[indices])
        axes[0, 1].plot(indices, cumulative_reward, 'b-', linewidth=2)
        axes[0, 1].set_title('Cumulative Reward (Sample)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Cumulative Reward')
        axes[0, 1].grid(True, alpha=0.3)

        # Reward boxplot by quartiles
        quartile_rewards = np.array_split(rewards, 4)
        axes[1, 0].boxplot(quartile_rewards, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        axes[1, 0].set_title('Reward Distribution by Data Quartiles')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True, alpha=0.3)

        # Q-Q plot for normality check
        stats.probplot(rewards, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Reward Q-Q Plot (Normal Distribution)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/reward_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_integrity_summary(self, output_dir: str):
        """Plot data integrity summary."""
        integrity_data = self.validation_results['integrity']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Data lengths comparison
        lengths = integrity_data['checks']['data_lengths']
        axes[0, 0].bar(lengths.keys(), lengths.values(), alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Data Array Lengths')
        axes[0, 0].set_ylabel('Length')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Action ranges
        action_ranges = integrity_data['checks']['action_ranges']
        action_names = ['Steering', 'Gas', 'Brake']
        x_pos = np.arange(len(action_names))

        mins = action_ranges['mins']
        maxs = action_ranges['maxs']
        expected = action_ranges['expected']

        width = 0.35
        axes[0, 1].bar(x_pos - width/2, mins, width, label='Min', alpha=0.7)
        axes[0, 1].bar(x_pos + width/2, maxs, width, label='Max', alpha=0.7)

        # Add expected range lines
        for i, (exp_min, exp_max) in enumerate(expected):
            axes[0, 1].hlines(exp_min, i-0.4, i+0.4, colors='red', linestyles='--', alpha=0.8)
            axes[0, 1].hlines(exp_max, i-0.4, i+0.4, colors='red', linestyles='--', alpha=0.8)

        axes[0, 1].set_title('Action Ranges vs Expected')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(action_names)
        axes[0, 1].legend()

        # NaN/Inf checks
        nan_inf_data = integrity_data['checks']['nan_inf_checks']
        categories = list(nan_inf_data.keys())
        nan_counts = [nan_inf_data[cat]['nan_count'] for cat in categories]
        inf_counts = [nan_inf_data[cat]['inf_count'] for cat in categories]

        x_pos = np.arange(len(categories))
        axes[1, 0].bar(x_pos - width/2, nan_counts, width, label='NaN', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, inf_counts, width, label='Inf', alpha=0.7)
        axes[1, 0].set_title('NaN and Infinite Value Counts')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()

        # Integrity status
        status_text = "PASSED" if integrity_data['passed'] else "FAILED"
        status_color = "green" if integrity_data['passed'] else "red"

        axes[1, 1].text(0.5, 0.7, f"Integrity Check: {status_text}",
                       horizontalalignment='center', fontsize=20, fontweight='bold',
                       color=status_color, transform=axes[1, 1].transAxes)

        error_count = len(integrity_data['errors'])
        warning_count = len(integrity_data['warnings'])

        axes[1, 1].text(0.5, 0.5, f"Errors: {error_count}",
                       horizontalalignment='center', fontsize=14,
                       transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.5, 0.3, f"Warnings: {warning_count}",
                       horizontalalignment='center', fontsize=14,
                       transform=axes[1, 1].transAxes)

        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/integrity_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_anomalies(self, output_dir: str):
        """Plot anomaly detection results."""
        anomaly_data = self.validation_results['anomalies']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Anomaly counts by type
        anomaly_counts = {
            'Action': len(anomaly_data['action_anomalies']),
            'Reward': len(anomaly_data['reward_anomalies']),
            'Episode': len(anomaly_data['episode_anomalies'])
        }

        axes[0, 0].bar(anomaly_counts.keys(), anomaly_counts.values(), alpha=0.7, color='coral')
        axes[0, 0].set_title('Anomaly Counts by Type')
        axes[0, 0].set_ylabel('Count')

        # Action anomalies by dimension
        if anomaly_data['action_anomalies']:
            action_dims = [anom['action_dim'] for anom in anomaly_data['action_anomalies']]
            unique_dims, counts = np.unique(action_dims, return_counts=True)
            dim_names = ['Steering', 'Gas', 'Brake']

            axes[0, 1].bar([dim_names[dim] for dim in unique_dims], counts, alpha=0.7, color='orange')
            axes[0, 1].set_title('Action Anomalies by Dimension')
            axes[0, 1].set_ylabel('Count')

        # Z-score distributions
        if anomaly_data['action_anomalies']:
            z_scores = [anom['z_score'] for anom in anomaly_data['action_anomalies']]
            axes[1, 0].hist(z_scores, bins=20, alpha=0.7, color='red', edgecolor='black')
            axes[1, 0].axvline(x=anomaly_data['summary']['threshold_used'],
                              color='black', linestyle='--', label='Threshold')
            axes[1, 0].set_title('Action Anomaly Z-Score Distribution')
            axes[1, 0].set_xlabel('Z-Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()

        # Anomaly rate summary
        anomaly_rate = anomaly_data['summary']['anomaly_rate'] * 100
        axes[1, 1].text(0.5, 0.6, f"Anomaly Rate: {anomaly_rate:.2f}%",
                       horizontalalignment='center', fontsize=16,
                       transform=axes[1, 1].transAxes)

        total_anomalies = sum(anomaly_counts.values())
        axes[1, 1].text(0.5, 0.4, f"Total Anomalies: {total_anomalies}",
                       horizontalalignment='center', fontsize=14,
                       transform=axes[1, 1].transAxes)

        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/anomaly_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, output_path: str):
        """
        Generate comprehensive validation report.

        Args:
            output_path: Path to save the report (JSON format)
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'size': len(self.dataset),
                'has_metadata': hasattr(self.dataset, 'episode_ids')
            },
            'validation_results': self.validation_results,
            'summary': self._generate_summary()
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"[Report] Validation report saved to {output_path}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        summary = {
            'overall_status': 'PASS',
            'recommendations': []
        }

        # Check integrity
        if 'integrity' in self.validation_results:
            if not self.validation_results['integrity']['passed']:
                summary['overall_status'] = 'FAIL'
                summary['recommendations'].append("Fix data integrity issues before using dataset")

        # Check performance
        if 'performance' in self.validation_results:
            perf = self.validation_results['performance']['episode_performance']
            if perf['success_rate'] < 0.7:
                summary['recommendations'].append("Expert performance below 70% success rate - consider retraining")
            if perf['avg_return'] < 600:
                summary['recommendations'].append("Low average return - expert may need improvement")

        # Check anomalies
        if 'anomalies' in self.validation_results:
            anomaly_rate = self.validation_results['anomalies']['summary']['anomaly_rate']
            if anomaly_rate > 0.05:  # 5% threshold
                summary['recommendations'].append("High anomaly rate detected - review data collection process")

        if not summary['recommendations']:
            summary['recommendations'].append("Dataset appears to be high quality")

        return summary


def compare_datasets(dataset1_path: str, dataset2_path: str, output_dir: str):
    """
    Compare two datasets and generate comparison report.

    Args:
        dataset1_path: Path to first dataset
        dataset2_path: Path to second dataset
        output_dir: Directory to save comparison results
    """
    print(f"\n[Comparison] Comparing {dataset1_path} vs {dataset2_path}")

    validator1 = DataValidator(dataset1_path)
    validator2 = DataValidator(dataset2_path)

    # Run validations
    validator1.validate_data_integrity()
    validator1.analyze_expert_performance()

    validator2.validate_data_integrity()
    validator2.analyze_expert_performance()

    # Create comparison plots
    os.makedirs(output_dir, exist_ok=True)

    # Compare episode performance
    if ('performance' in validator1.validation_results and
        'performance' in validator2.validation_results):

        perf1 = validator1.validation_results['performance']['episode_performance']
        perf2 = validator2.validation_results['performance']['episode_performance']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Return comparison
        metrics = ['avg_return', 'success_rate', 'avg_episode_length']
        values1 = [perf1[m] for m in metrics]
        values2 = [perf2[m] for m in metrics]

        x_pos = np.arange(len(metrics))
        width = 0.35

        axes[0, 0].bar(x_pos - width/2, values1, width, label='Dataset 1', alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, values2, width, label='Dataset 2', alpha=0.7)
        axes[0, 0].set_title('Performance Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].legend()

        # Size comparison
        sizes = [len(validator1.dataset), len(validator2.dataset)]
        axes[0, 1].bar(['Dataset 1', 'Dataset 2'], sizes, alpha=0.7, color=['blue', 'orange'])
        axes[0, 1].set_title('Dataset Size Comparison')
        axes[0, 1].set_ylabel('Number of Transitions')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/dataset_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"[Comparison] Results saved to {output_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Data Validation Tools")
    parser.add_argument('--input', required=True, help='Input dataset path')
    parser.add_argument('--output', default='validation_results', help='Output directory/file prefix')
    parser.add_argument('--compare', help='Second dataset path for comparison')
    parser.add_argument('--anomaly_threshold', type=float, default=2.0,
                       help='Z-score threshold for anomaly detection')

    args = parser.parse_args()

    if args.compare:
        # Dataset comparison mode
        compare_datasets(args.input, args.compare, f"{args.output}_comparison")
    else:
        # Single dataset validation mode
        validator = DataValidator(args.input)

        # Run all validations
        print("Running comprehensive validation...")
        validator.validate_data_integrity()
        validator.analyze_expert_performance()
        validator.detect_anomalies(args.anomaly_threshold)

        # Generate outputs
        output_dir = f"{args.output}_visualizations"
        validator.create_visualizations(output_dir)
        validator.generate_report(f"{args.output}_report.json")

        # Print summary
        summary = validator._generate_summary()
        print(f"\n{'='*50}")
        print("VALIDATION SUMMARY")
        print("="*50)
        print(f"Overall Status: {summary['overall_status']}")
        print("\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"- {rec}")
        print(f"\nDetailed results saved to {args.output}_*")


if __name__ == "__main__":
    main()