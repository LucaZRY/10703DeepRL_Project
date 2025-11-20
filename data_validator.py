"""
Data Validation Tool for Expert Demonstrations
Deep Reinforcement Learning Project

This module provides comprehensive validation and quality control for expert
demonstration datasets collected for the CarRacing-v2 environment.

Features:
- Data integrity checks
- Expert performance analysis
- Statistical anomaly detection
- Visualization generation
- Format compatibility validation

Usage:
    python data_validator.py --input expert_dataset.pkl --output validation_report
    python data_validator.py --input data1.pkl --compare data2.pkl
"""

import argparse
import os
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class DataValidator:
    """
    Comprehensive validation system for expert demonstration datasets.
    """

    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize validator.

        Args:
            dataset_path: Path to dataset file
        """
        self.dataset = None
        self.metadata = None
        self.validation_results = {}

        if dataset_path:
            self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path: str):
        """Load dataset from file."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        file_ext = os.path.splitext(dataset_path)[1].lower()

        if file_ext == '.pkl':
            self._load_pickle(dataset_path)
        elif file_ext == '.h5':
            self._load_hdf5(dataset_path)
        elif file_ext == '.npz':
            self._load_npz(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        print(f"Loaded dataset from: {dataset_path}")
        print(f"Number of episodes: {len(self.dataset['demonstrations'])}")

    def _load_pickle(self, filepath: str):
        """Load pickle format dataset."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.dataset = data
        self.metadata = data.get('metadata', {})

    def _load_hdf5(self, filepath: str):
        """Load HDF5 format dataset."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 files. Install with: pip install h5py")

        with h5py.File(filepath, 'r') as f:
            demonstrations = []
            demos_group = f['demonstrations']

            for episode_key in demos_group.keys():
                demo_group = demos_group[episode_key]
                demo = {
                    'episode_id': demo_group.attrs['episode_id'],
                    'observations': demo_group['observations'][:],
                    'actions': demo_group['actions'][:],
                    'rewards': demo_group['rewards'][:],
                    'episode_length': demo_group.attrs['episode_length'],
                    'total_reward': demo_group.attrs['total_reward']
                }
                demonstrations.append(demo)

            self.dataset = {
                'demonstrations': demonstrations,
                'statistics': json.loads(f.attrs['statistics']),
                'config': json.loads(f.attrs['config']),
                'metadata': json.loads(f.attrs['metadata'])
            }
            self.metadata = self.dataset['metadata']

    def _load_npz(self, filepath: str):
        """Load NPZ format dataset."""
        data = np.load(filepath, allow_pickle=True)

        # Reconstruct episodes from flattened arrays
        observations = data['observations']
        actions = data['actions']
        rewards = data['rewards']
        episode_starts = data['episode_starts']

        demonstrations = []
        for i, start_idx in enumerate(episode_starts):
            end_idx = episode_starts[i + 1] if i + 1 < len(episode_starts) else len(observations)

            demo = {
                'episode_id': i,
                'observations': observations[start_idx:end_idx],
                'actions': actions[start_idx:end_idx],
                'rewards': rewards[start_idx:end_idx],
                'episode_length': end_idx - start_idx,
                'total_reward': float(np.sum(rewards[start_idx:end_idx]))
            }
            demonstrations.append(demo)

        self.dataset = {
            'demonstrations': demonstrations,
            'statistics': json.loads(str(data['statistics'])),
            'config': json.loads(str(data['config'])),
            'metadata': json.loads(str(data['metadata']))
        }
        self.metadata = self.dataset['metadata']

    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Perform comprehensive data integrity validation.

        Returns:
            Validation results with pass/fail status and details
        """
        print("\n🔍 Validating data integrity...")

        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }

        demonstrations = self.dataset['demonstrations']

        # Check basic structure
        if not demonstrations:
            results['errors'].append("No demonstrations found in dataset")
            results['passed'] = False
            return results

        # Check required fields
        required_fields = ['observations', 'actions', 'rewards', 'episode_length', 'total_reward']
        for i, demo in enumerate(demonstrations[:10]):  # Check first 10
            for field in required_fields:
                if field not in demo:
                    results['errors'].append(f"Episode {i} missing field: {field}")
                    results['passed'] = False

        if not results['passed']:
            return results

        # Validate shapes and consistency
        self._validate_shapes(demonstrations, results)
        self._validate_action_ranges(demonstrations, results)
        self._validate_data_types(demonstrations, results)
        self._check_missing_values(demonstrations, results)

        self.validation_results['integrity'] = results
        print(f"✅ Integrity check: {'PASSED' if results['passed'] else 'FAILED'}")
        return results

    def _validate_shapes(self, demonstrations: List[Dict], results: Dict):
        """Validate observation and action shapes."""
        expected_obs_shape = (4, 84, 84)  # CarRacing preprocessed observations
        expected_action_shape = (3,)      # [steering, gas, brake]

        shape_issues = []

        for i, demo in enumerate(demonstrations):
            obs = demo['observations']
            actions = demo['actions']

            # Check observation shapes
            if obs.shape[1:] != expected_obs_shape:
                shape_issues.append(f"Episode {i}: obs shape {obs.shape[1:]} != {expected_obs_shape}")

            # Check action shapes
            if actions.shape[1:] != expected_action_shape:
                shape_issues.append(f"Episode {i}: action shape {actions.shape[1:]} != {expected_action_shape}")

            # Check sequence length consistency
            if len(obs) != len(actions) or len(actions) != len(demo['rewards']):
                shape_issues.append(f"Episode {i}: inconsistent sequence lengths")

        if shape_issues:
            results['warnings'].extend(shape_issues[:5])  # Limit warnings
            if len(shape_issues) > 10:
                results['errors'].append(f"Too many shape inconsistencies: {len(shape_issues)}")
                results['passed'] = False

        results['checks']['shape_validation'] = {
            'total_issues': len(shape_issues),
            'expected_obs_shape': expected_obs_shape,
            'expected_action_shape': expected_action_shape
        }

    def _validate_action_ranges(self, demonstrations: List[Dict], results: Dict):
        """Validate action value ranges."""
        all_actions = np.concatenate([demo['actions'] for demo in demonstrations])

        # Expected ranges for CarRacing: steering[-1,1], gas[0,1], brake[0,1]
        expected_ranges = [(-1.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        action_names = ['steering', 'gas', 'brake']

        range_violations = []

        for i, (min_val, max_val) in enumerate(expected_ranges):
            action_col = all_actions[:, i]
            actual_min = np.min(action_col)
            actual_max = np.max(action_col)

            if actual_min < min_val - 0.01 or actual_max > max_val + 0.01:
                range_violations.append(
                    f"{action_names[i]}: range [{actual_min:.3f}, {actual_max:.3f}] "
                    f"outside expected [{min_val}, {max_val}]"
                )

        if range_violations:
            results['warnings'].extend(range_violations)

        results['checks']['action_ranges'] = {
            'violations': range_violations,
            'actual_ranges': {
                name: [float(np.min(all_actions[:, i])), float(np.max(all_actions[:, i]))]
                for i, name in enumerate(action_names)
            }
        }

    def _validate_data_types(self, demonstrations: List[Dict], results: Dict):
        """Validate data types."""
        type_issues = []

        for i, demo in enumerate(demonstrations[:10]):  # Check first 10
            if not isinstance(demo['observations'], np.ndarray):
                type_issues.append(f"Episode {i}: observations not numpy array")
            if not isinstance(demo['actions'], np.ndarray):
                type_issues.append(f"Episode {i}: actions not numpy array")
            if not isinstance(demo['rewards'], np.ndarray):
                type_issues.append(f"Episode {i}: rewards not numpy array")

        if type_issues:
            results['warnings'].extend(type_issues)

        results['checks']['data_types'] = {'issues': type_issues}

    def _check_missing_values(self, demonstrations: List[Dict], results: Dict):
        """Check for NaN or infinite values."""
        nan_inf_count = {'observations': 0, 'actions': 0, 'rewards': 0}

        for demo in demonstrations:
            # Check observations (sample first 100 to avoid memory issues)
            obs_sample = demo['observations'][:100] if len(demo['observations']) > 100 else demo['observations']
            nan_inf_count['observations'] += np.sum(~np.isfinite(obs_sample))

            # Check actions
            nan_inf_count['actions'] += np.sum(~np.isfinite(demo['actions']))

            # Check rewards
            nan_inf_count['rewards'] += np.sum(~np.isfinite(demo['rewards']))

        total_nan_inf = sum(nan_inf_count.values())
        if total_nan_inf > 0:
            results['errors'].append(f"Found {total_nan_inf} NaN/inf values")
            results['passed'] = False

        results['checks']['missing_values'] = nan_inf_count

    def analyze_expert_performance(self) -> Dict[str, Any]:
        """
        Analyze expert policy performance.

        Returns:
            Performance analysis results
        """
        print("\n📊 Analyzing expert performance...")

        demonstrations = self.dataset['demonstrations']
        episode_returns = [demo['total_reward'] for demo in demonstrations]
        episode_lengths = [demo['episode_length'] for demo in demonstrations]

        # Calculate performance metrics
        performance_stats = {
            'num_episodes': len(demonstrations),
            'avg_return': float(np.mean(episode_returns)),
            'std_return': float(np.std(episode_returns)),
            'min_return': float(np.min(episode_returns)),
            'max_return': float(np.max(episode_returns)),
            'median_return': float(np.median(episode_returns)),
            'success_rate': float(np.mean(np.array(episode_returns) > 700)),
            'avg_episode_length': float(np.mean(episode_lengths)),
            'total_transitions': sum(episode_lengths)
        }

        # Action analysis
        all_actions = np.concatenate([demo['actions'] for demo in demonstrations])
        action_stats = {
            'steering': {
                'mean': float(np.mean(all_actions[:, 0])),
                'std': float(np.std(all_actions[:, 0])),
                'abs_mean': float(np.mean(np.abs(all_actions[:, 0])))
            },
            'gas': {
                'mean': float(np.mean(all_actions[:, 1])),
                'usage_rate': float(np.mean(all_actions[:, 1] > 0.1))
            },
            'brake': {
                'mean': float(np.mean(all_actions[:, 2])),
                'usage_rate': float(np.mean(all_actions[:, 2] > 0.1))
            }
        }

        # Quality assessment
        quality_flags = []
        if performance_stats['success_rate'] < 0.6:
            quality_flags.append("Low success rate")
        if performance_stats['avg_return'] < 500:
            quality_flags.append("Low average return")
        if performance_stats['avg_episode_length'] < 100:
            quality_flags.append("Very short episodes")

        results = {
            'performance_metrics': performance_stats,
            'action_analysis': action_stats,
            'quality_flags': quality_flags,
            'episode_returns': episode_returns,
            'episode_lengths': episode_lengths
        }

        self.validation_results['performance'] = results
        print(f"✅ Performance analysis complete")
        print(f"   Average return: {performance_stats['avg_return']:.2f}")
        print(f"   Success rate: {performance_stats['success_rate']:.2%}")
        return results

    def detect_anomalies(self, threshold: float = 2.5) -> Dict[str, Any]:
        """
        Detect statistical anomalies in the dataset.

        Args:
            threshold: Z-score threshold for anomaly detection

        Returns:
            Anomaly detection results
        """
        print(f"\n🔍 Detecting anomalies (threshold: {threshold})...")

        demonstrations = self.dataset['demonstrations']
        episode_returns = [demo['total_reward'] for demo in demonstrations]

        # Episode-level anomalies
        return_z_scores = np.abs(stats.zscore(episode_returns))
        anomalous_episodes = np.where(return_z_scores > threshold)[0]

        # Action-level anomalies
        all_actions = np.concatenate([demo['actions'] for demo in demonstrations])
        action_anomalies = {}

        for i, action_name in enumerate(['steering', 'gas', 'brake']):
            action_col = all_actions[:, i]
            z_scores = np.abs(stats.zscore(action_col))
            anomalous_indices = np.where(z_scores > threshold)[0]
            action_anomalies[action_name] = len(anomalous_indices)

        results = {
            'threshold': threshold,
            'anomalous_episodes': [
                {
                    'episode_id': int(idx),
                    'return': episode_returns[idx],
                    'z_score': float(return_z_scores[idx])
                }
                for idx in anomalous_episodes
            ],
            'action_anomalies': action_anomalies,
            'summary': {
                'episode_anomaly_rate': len(anomalous_episodes) / len(episode_returns),
                'total_action_anomalies': sum(action_anomalies.values()),
                'action_anomaly_rate': sum(action_anomalies.values()) / len(all_actions)
            }
        }

        self.validation_results['anomalies'] = results
        print(f"✅ Anomaly detection complete")
        print(f"   Episode anomalies: {len(anomalous_episodes)}/{len(episode_returns)}")
        return results

    def create_visualizations(self, output_dir: str):
        """
        Create comprehensive visualizations.

        Args:
            output_dir: Directory to save plots
        """
        print(f"\n📊 Creating visualizations in {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        if 'performance' in self.validation_results:
            self._plot_performance_analysis(output_dir)

        if 'integrity' in self.validation_results:
            self._plot_integrity_summary(output_dir)

        if 'anomalies' in self.validation_results:
            self._plot_anomaly_analysis(output_dir)

        self._plot_action_distributions(output_dir)

        print(f"✅ Visualizations saved to {output_dir}")

    def _plot_performance_analysis(self, output_dir: str):
        """Create performance analysis plots."""
        perf_data = self.validation_results['performance']
        returns = perf_data['episode_returns']
        lengths = perf_data['episode_lengths']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Episode returns
        axes[0, 0].plot(returns, 'b-', alpha=0.8, linewidth=1)
        axes[0, 0].axhline(y=700, color='r', linestyle='--', alpha=0.8, label='Success threshold')
        axes[0, 0].set_title('Episode Returns Over Time')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Return histogram
        axes[0, 1].hist(returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(x=np.mean(returns), color='r', linestyle='-', label='Mean')
        axes[0, 1].axvline(x=700, color='orange', linestyle='--', label='Success threshold')
        axes[0, 1].set_title('Return Distribution')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()

        # Episode lengths
        axes[1, 0].plot(lengths, 'g-', alpha=0.8, linewidth=1)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Length (steps)')
        axes[1, 0].grid(True, alpha=0.3)

        # Performance summary
        stats = perf_data['performance_metrics']
        summary_text = f"""
        Episodes: {stats['num_episodes']}
        Avg Return: {stats['avg_return']:.1f}
        Success Rate: {stats['success_rate']:.1%}
        Total Steps: {stats['total_transitions']}
        """

        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_action_distributions(self, output_dir: str):
        """Plot action distribution analysis."""
        demonstrations = self.dataset['demonstrations']
        all_actions = np.concatenate([demo['actions'] for demo in demonstrations])
        action_names = ['Steering', 'Gas', 'Brake']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for i, name in enumerate(action_names):
            # Distribution
            axes[0, i].hist(all_actions[:, i], bins=50, alpha=0.7, color=f'C{i}', edgecolor='black')
            axes[0, i].set_title(f'{name} Distribution')
            axes[0, i].set_xlabel(f'{name} Value')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].grid(True, alpha=0.3)

            # Time series (sample)
            sample_size = min(1000, len(all_actions))
            indices = np.linspace(0, len(all_actions)-1, sample_size, dtype=int)
            axes[1, i].plot(indices, all_actions[indices, i], alpha=0.6, color=f'C{i}')
            axes[1, i].set_title(f'{name} Over Time (Sample)')
            axes[1, i].set_xlabel('Step')
            axes[1, i].set_ylabel(f'{name} Value')
            axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'action_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_integrity_summary(self, output_dir: str):
        """Plot data integrity summary."""
        integrity_data = self.validation_results['integrity']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Overall status
        status = "PASSED" if integrity_data['passed'] else "FAILED"
        status_color = "green" if integrity_data['passed'] else "red"

        ax1.text(0.5, 0.5, f"Integrity Check\n{status}", ha='center', va='center',
                fontsize=24, fontweight='bold', color=status_color,
                transform=ax1.transAxes)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')

        # Error and warning counts
        error_count = len(integrity_data['errors'])
        warning_count = len(integrity_data['warnings'])

        categories = ['Errors', 'Warnings']
        counts = [error_count, warning_count]
        colors = ['red' if error_count > 0 else 'lightgray',
                 'orange' if warning_count > 0 else 'lightgray']

        ax2.bar(categories, counts, color=colors, alpha=0.7)
        ax2.set_title('Issues Found')
        ax2.set_ylabel('Count')

        # Missing values check
        if 'missing_values' in integrity_data['checks']:
            missing_data = integrity_data['checks']['missing_values']
            ax3.bar(missing_data.keys(), missing_data.values(), alpha=0.7, color='skyblue')
            ax3.set_title('NaN/Inf Values Found')
            ax3.set_ylabel('Count')
            ax3.tick_params(axis='x', rotation=45)

        # Action ranges validation
        if 'action_ranges' in integrity_data['checks']:
            violations = len(integrity_data['checks']['action_ranges']['violations'])
            ax4.pie([violations, 3-violations], labels=['Violations', 'Valid'],
                   colors=['red' if violations > 0 else 'green', 'lightgreen'],
                   autopct='%1.0f', startangle=90)
            ax4.set_title('Action Range Validation')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'integrity_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_anomaly_analysis(self, output_dir: str):
        """Plot anomaly detection results."""
        anomaly_data = self.validation_results['anomalies']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Anomaly summary
        summary = anomaly_data['summary']
        episode_rate = summary['episode_anomaly_rate'] * 100
        action_rate = summary['action_anomaly_rate'] * 100

        rates_text = f"""
        Episode Anomaly Rate: {episode_rate:.1f}%
        Action Anomaly Rate: {action_rate:.1f}%
        Threshold: {anomaly_data['threshold']}
        """

        axes[0, 0].text(0.1, 0.5, rates_text, transform=axes[0, 0].transAxes,
                       fontsize=14, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Anomaly Summary')

        # Action anomalies by type
        action_anomalies = anomaly_data['action_anomalies']
        axes[0, 1].bar(action_anomalies.keys(), action_anomalies.values(), alpha=0.7, color='coral')
        axes[0, 1].set_title('Action Anomalies by Type')
        axes[0, 1].set_ylabel('Count')

        # Anomalous episode returns
        if anomaly_data['anomalous_episodes']:
            anomalous_returns = [ep['return'] for ep in anomaly_data['anomalous_episodes']]
            axes[1, 0].scatter(range(len(anomalous_returns)), anomalous_returns,
                              color='red', alpha=0.7, s=50)
            axes[1, 0].set_title('Anomalous Episode Returns')
            axes[1, 0].set_xlabel('Anomalous Episode Index')
            axes[1, 0].set_ylabel('Return')
            axes[1, 0].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'anomaly_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, output_path: str):
        """
        Generate comprehensive validation report.

        Args:
            output_path: Path to save the report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'num_episodes': len(self.dataset['demonstrations']),
                'metadata': self.metadata
            },
            'validation_results': self.validation_results,
            'summary': self._generate_summary()
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📄 Validation report saved: {output_path}")
        return report

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        summary = {
            'overall_status': 'PASS',
            'quality_score': 100,
            'recommendations': []
        }

        # Check integrity
        if 'integrity' in self.validation_results:
            if not self.validation_results['integrity']['passed']:
                summary['overall_status'] = 'FAIL'
                summary['quality_score'] -= 50
                summary['recommendations'].append("Fix data integrity issues")

        # Check performance
        if 'performance' in self.validation_results:
            perf = self.validation_results['performance']['performance_metrics']

            if perf['success_rate'] < 0.6:
                summary['quality_score'] -= 20
                summary['recommendations'].append("Expert success rate below 60%")

            if perf['avg_return'] < 500:
                summary['quality_score'] -= 15
                summary['recommendations'].append("Low average return")

        # Check anomalies
        if 'anomalies' in self.validation_results:
            anomaly_rate = self.validation_results['anomalies']['summary']['episode_anomaly_rate']
            if anomaly_rate > 0.1:  # 10% threshold
                summary['quality_score'] -= 10
                summary['recommendations'].append("High anomaly rate detected")

        if summary['quality_score'] >= 90:
            summary['quality_rating'] = 'Excellent'
        elif summary['quality_score'] >= 75:
            summary['quality_rating'] = 'Good'
        elif summary['quality_score'] >= 60:
            summary['quality_rating'] = 'Acceptable'
        else:
            summary['quality_rating'] = 'Poor'

        if not summary['recommendations']:
            summary['recommendations'].append("Dataset quality is good")

        return summary


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Expert Dataset Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_validator.py --input expert_dataset.pkl --output validation_report
  python data_validator.py --input data.pkl --visualizations --anomaly_threshold 2.0
        """
    )

    parser.add_argument('--input', required=True,
                       help='Input dataset file (pkl, h5, or npz)')
    parser.add_argument('--output', default='validation_report',
                       help='Output file prefix')
    parser.add_argument('--visualizations', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--anomaly_threshold', type=float, default=2.5,
                       help='Z-score threshold for anomaly detection')

    args = parser.parse_args()

    print("🔍 Expert Dataset Validation Tool")
    print("=" * 40)

    try:
        # Load and validate dataset
        validator = DataValidator(args.input)

        # Run validation suite
        integrity_results = validator.validate_data_integrity()
        performance_results = validator.analyze_expert_performance()
        anomaly_results = validator.detect_anomalies(args.anomaly_threshold)

        # Generate visualizations if requested
        if args.visualizations:
            validator.create_visualizations(f"{args.output}_plots")

        # Generate report
        report_path = f"{args.output}.json"
        report = validator.generate_report(report_path)

        # Print summary
        summary = report['summary']
        print(f"\n{'='*40}")
        print("VALIDATION SUMMARY")
        print("=" * 40)
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Quality Rating: {summary['quality_rating']} ({summary['quality_score']}/100)")
        print(f"\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")

        print(f"\n📊 Report saved: {report_path}")
        if args.visualizations:
            print(f"📈 Plots saved: {args.output}_plots/")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()