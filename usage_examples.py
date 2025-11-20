#!/usr/bin/env python3
"""
Expert Data Preparation - Usage Examples

This script demonstrates how to use the expert data preparation system
for the CarRacing DRL project.

Run with: python usage_examples.py
"""

import os
import torch
from expert_data_collector import ExpertDataCollector
from data_validation_tools import DataValidator
from data_utils import DataManager


def example_1_basic_data_collection():
    """Example 1: Basic expert data collection"""
    print("="*60)
    print("EXAMPLE 1: Basic Expert Data Collection")
    print("="*60)

    try:
        # Initialize data collector
        collector = ExpertDataCollector(
            expert_model_path="ppo_discrete_carracing.pt",
            device="auto"
        )

        # Collect expert episodes
        print("\nCollecting 10 expert demonstration episodes...")
        stats = collector.collect_expert_episodes(
            num_episodes=10,
            render=False,
            save_videos=False
        )

        # Save dataset
        output_path = collector.save_dataset(
            "example_expert_data.pkl",
            include_metadata=True
        )

        print(f"\nCollection Results:")
        print(f"- Episodes collected: {stats['episodes_collected']}")
        print(f"- Average return: {stats['avg_return']:.2f}")
        print(f"- Success rate: {stats['success_rate']:.2f}")
        print(f"- Dataset saved to: {output_path}")

    except Exception as e:
        print(f"Error in basic data collection: {e}")
        print("Make sure 'ppo_discrete_carracing.pt' exists in the current directory")


def example_2_data_validation():
    """Example 2: Data validation and quality control"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Data Validation and Quality Control")
    print("="*60)

    dataset_path = "example_expert_data.pkl"

    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Run example 1 first.")
        return

    try:
        # Initialize validator
        validator = DataValidator(dataset_path)

        # Run comprehensive validation
        print("\nRunning data integrity checks...")
        integrity_results = validator.validate_data_integrity()

        print("\nRunning performance analysis...")
        performance_results = validator.analyze_expert_performance()

        print("\nRunning anomaly detection...")
        anomaly_results = validator.detect_anomalies(threshold=2.0)

        # Generate visualizations
        print("\nCreating validation visualizations...")
        validator.create_visualizations("validation_output")

        # Generate report
        validator.generate_report("validation_report.json")

        print(f"\nValidation Results:")
        print(f"- Data integrity: {'PASS' if integrity_results['passed'] else 'FAIL'}")
        print(f"- Dataset size: {len(validator.dataset)} transitions")

        if 'performance' in validator.validation_results:
            perf = validator.validation_results['performance']['episode_performance']
            print(f"- Average return: {perf['avg_return']:.2f}")
            print(f"- Success rate: {perf['success_rate']:.2f}")

        if 'anomalies' in validator.validation_results:
            anom = validator.validation_results['anomalies']['summary']
            print(f"- Anomaly rate: {anom['anomaly_rate']*100:.2f}%")

        print(f"\nOutputs:")
        print(f"- Visualizations saved to: validation_output/")
        print(f"- Report saved to: validation_report.json")

    except Exception as e:
        print(f"Error in data validation: {e}")


def example_3_data_management():
    """Example 3: Data management and utilities"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Data Management and Utilities")
    print("="*60)

    dataset_path = "example_expert_data.pkl"

    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Run example 1 first.")
        return

    try:
        # Initialize data manager
        dm = DataManager(base_dir="managed_data")

        # Load dataset
        print("\nLoading dataset...")
        dataset = dm.load_dataset(dataset_path)

        # Get dataset info
        info = dm.get_dataset_info(dataset_path)
        print(f"\nDataset Info:")
        print(f"- Size: {info.get('dataset_size', 'unknown')} transitions")
        print(f"- File size: {info['file_size_mb']:.2f} MB")
        print(f"- Format: {info['format']}")

        # Convert to different formats
        print(f"\nConverting to different formats...")
        hdf5_path = dm.save_dataset(dataset, "expert_data_converted.h5", format="hdf5")
        npz_path = dm.save_dataset(dataset, "expert_data_converted.npz", format="npz")

        print(f"- HDF5 format saved to: {hdf5_path}")
        print(f"- NPZ format saved to: {npz_path}")

        # Split dataset
        print(f"\nSplitting dataset (80/20 train/val)...")
        train_path, val_path = dm.split_dataset(
            dataset_path,
            train_ratio=0.8,
            output_prefix="expert_data_split"
        )

        print(f"- Train set: {train_path}")
        print(f"- Validation set: {val_path}")

        # Create PyTorch DataLoader
        print(f"\nCreating PyTorch DataLoader...")
        dataloader = dm.create_dataloader(
            dataset,
            batch_size=32,
            shuffle=True
        )

        # Test the dataloader
        for batch_idx, (obs, actions, rewards) in enumerate(dataloader):
            print(f"- Batch {batch_idx}: obs {obs.shape}, actions {actions.shape}")
            if batch_idx >= 2:  # Only show first 3 batches
                break

        # List all datasets in managed directory
        print(f"\nDatasets in managed_data/:")
        datasets = dm.list_datasets()
        for ds in datasets:
            name = os.path.basename(ds['filepath'])
            size = ds.get('dataset_size', 'unknown')
            print(f"- {name}: {size} transitions")

    except Exception as e:
        print(f"Error in data management: {e}")


def example_4_advanced_collection():
    """Example 4: Advanced collection scenarios"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Advanced Collection Scenarios")
    print("="*60)

    try:
        # Initialize collector
        collector = ExpertDataCollector(
            expert_model_path="ppo_discrete_carracing.pt",
            device="auto"
        )

        # Collect recovery scenarios
        print("\nCollecting recovery scenarios...")
        recovery_stats = collector.collect_recovery_scenarios(
            num_episodes=5,
            trigger_conditions=['off_track']
        )

        print(f"Recovery Collection Results:")
        print(f"- Recovery episodes: {recovery_stats['recovery_episodes']}")
        print(f"- Trigger counts: {recovery_stats['trigger_counts']}")

        # Save combined dataset
        combined_path = collector.save_dataset(
            "advanced_expert_data.pkl",
            include_metadata=True
        )

        print(f"- Combined dataset saved to: {combined_path}")
        print(f"- Total transitions: {len(collector.dataset)}")

        # Note: DAgger collection would require a trained student model
        print(f"\nNote: DAgger collection requires a trained student model.")
        print(f"See the main training scripts for student policy training.")

    except Exception as e:
        print(f"Error in advanced collection: {e}")


def cleanup_example_files():
    """Clean up example files"""
    files_to_remove = [
        "example_expert_data.pkl",
        "advanced_expert_data.pkl",
        "validation_report.json"
    ]

    dirs_to_remove = [
        "validation_output",
        "managed_data",
        "videos"
    ]

    print("\n" + "="*60)
    print("CLEANUP (Optional)")
    print("="*60)
    print("To clean up example files, run:")

    for file in files_to_remove:
        if os.path.exists(file):
            print(f"rm {file}")

    for dir in dirs_to_remove:
        if os.path.exists(dir):
            print(f"rm -rf {dir}/")


def main():
    """Run all examples"""
    print("EXPERT DATA PREPARATION - USAGE EXAMPLES")
    print("This script demonstrates the expert data preparation system.")
    print("Make sure you have 'ppo_discrete_carracing.pt' in the current directory.\n")

    # Check if expert model exists
    if not os.path.exists("ppo_discrete_carracing.pt"):
        print("WARNING: Expert model 'ppo_discrete_carracing.pt' not found!")
        print("Some examples may fail. Train an expert model first or adjust the path.\n")

    try:
        # Run examples
        example_1_basic_data_collection()
        example_2_data_validation()
        example_3_data_management()
        example_4_advanced_collection()

        cleanup_example_files()

        print(f"\n" + "="*60)
        print("ALL EXAMPLES COMPLETED")
        print("="*60)
        print("Check the generated files and directories for results.")
        print("Review the code in this file to understand the usage patterns.")

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")


if __name__ == "__main__":
    main()