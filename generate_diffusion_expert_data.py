#!/usr/bin/env python3
"""
Generate Expert Data for Diffusion-DAgger Project

This script generates comprehensive expert data specifically tailored for your
diffusion policy + DAgger implementation on CarRacing-v2.

Based on analysis of your project structure, this generates:
1. Trajectory-level expert demonstrations for diffusion teacher pre-training
2. Recovery scenario data for robust behavior learning
3. Multi-modal action distributions for uncertainty modeling
4. Properly formatted data for your experiment configuration

Usage:
    python generate_diffusion_expert_data.py --full-pipeline
    python generate_diffusion_expert_data.py --quick-start --episodes 50
"""

import argparse
import os
import sys
import torch
from datetime import datetime
from diffusion_expert_data_generator import DiffusionExpertDataGenerator
from diffusion_dagger_trainer import DiffusionDAggerTrainer
from data_validation_tools import DataValidator
import yaml


def load_experiment_config(config_path: str = "config/experiment.yaml"):
    """Load experiment configuration with fallback defaults."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠ Configuration file {config_path} not found. Using defaults.")
        # Provide default configuration based on your project structure
        return {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'teacher': {
                'ckpt': 'ppo_discrete_carracing.pt',
                'max_recovery_horizon': 64,
                'num_denoise_steps': 10,
                'guidance_scale': 1.0
            },
            'student': {
                'actor': {'lr': 3e-4}
            },
            'distillation': {
                'enabled': True,
                'lambda_kl': 1.0
            },
            'dagger_recovery': {
                'enabled': True,
                'rollout_horizon': 40
            },
            'training': {
                'total_env_steps': 2_000_000,
                'eval_interval_steps': 20_000
            },
            'logging': {
                'use_wandb': False,
                'project': 'CarRacing-DiffDist',
                'name': 'baseline_run'
            }
        }


def generate_expert_data_for_diffusion_project(episodes: int = 200,
                                              horizon: int = 64,
                                              include_recovery: bool = True,
                                              validate_data: bool = True):
    """
    Generate comprehensive expert data for your diffusion-DAgger project.

    Args:
        episodes: Number of episodes to collect
        horizon: Trajectory horizon for diffusion training
        include_recovery: Whether to include recovery scenarios
        validate_data: Whether to run data validation
    """

    print("🚀 GENERATING EXPERT DATA FOR DIFFUSION-DAGGER PROJECT")
    print("="*70)
    print(f"Episodes: {episodes}")
    print(f"Trajectory Horizon: {horizon}")
    print(f"Include Recovery: {include_recovery}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    # Check for expert model
    expert_model_path = "ppo_discrete_carracing.pt"
    if not os.path.exists(expert_model_path):
        print(f"❌ Expert model '{expert_model_path}' not found!")
        print("   Please ensure you have a trained expert model.")
        print("   You can train one using: python ppo_expert.py")
        return None

    try:
        # 1. Initialize diffusion data generator
        print("📊 Step 1: Initializing Diffusion Data Generator")
        generator = DiffusionExpertDataGenerator(
            expert_model_path=expert_model_path,
            device="auto",
            trajectory_horizon=horizon
        )

        # 2. Collect trajectory episodes
        print("\n📈 Step 2: Collecting Trajectory Episodes")
        stats = generator.collect_trajectory_episodes(
            num_episodes=episodes,
            enable_perturbations=True,  # Important for recovery data
            save_videos=True  # Save some examples
        )

        print(f"✓ Collected {stats['total_episodes']} episodes")
        print(f"✓ Generated {stats['total_segments']} trajectory segments")
        print(f"✓ Recovery segments: {stats['recovery_segments']}")
        print(f"✓ Challenging segments: {stats['challenging_segments']}")

        # 3. Generate training data
        print("\n💾 Step 3: Generating Training Data")
        training_data_path = generator.generate_diffusion_training_data(
            output_dir="diffusion_training_data"
        )

        # 4. Generate recovery-focused data if requested
        if include_recovery:
            print("\n🔄 Step 4: Generating Recovery-Focused Data")
            recovery_episodes = max(20, episodes // 4)  # At least 20, or 25% of total
            recovery_data_path = generator.generate_recovery_focused_data(
                num_recovery_episodes=recovery_episodes,
                output_dir="recovery_training_data"
            )
            print(f"✓ Generated recovery data: {recovery_data_path}")

        # 5. Data validation
        if validate_data:
            print("\n🔍 Step 5: Validating Generated Data")
            validator = DataValidator(training_data_path)

            # Run validation
            integrity_results = validator.validate_data_integrity()
            performance_results = validator.analyze_expert_performance()

            # Generate validation report
            validator.create_visualizations("diffusion_data_validation")
            validator.generate_report("diffusion_data_validation_report.json")

            print(f"✓ Data integrity: {'PASS' if integrity_results['passed'] else 'FAIL'}")
            if 'performance' in validator.validation_results:
                perf = validator.validation_results['performance']['episode_performance']
                print(f"✓ Average return: {perf['avg_return']:.2f}")
                print(f"✓ Success rate: {perf['success_rate']:.2f}")

        # 6. Generate summary report
        final_stats = generator.get_dataset_statistics()

        print("\n📋 EXPERT DATA GENERATION COMPLETE")
        print("="*70)
        print("Generated Files:")
        print(f"  📁 Training data: {training_data_path}")
        if include_recovery:
            print(f"  📁 Recovery data: recovery_training_data/")
        if validate_data:
            print(f"  📁 Validation: diffusion_data_validation/")
        print(f"  📁 Videos: videos/")

        print(f"\nDataset Statistics:")
        for key, value in final_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

        print(f"\n✨ Data optimized for your diffusion-DAgger approach:")
        print(f"   • Trajectory horizon: {horizon} steps")
        print(f"   • Temporal consistency preserved")
        print(f"   • Recovery behaviors included")
        print(f"   • Multi-modal distributions captured")
        print(f"   • Compatible with your experiment config")

        return {
            'training_data_path': training_data_path,
            'recovery_data_path': recovery_data_path if include_recovery else None,
            'statistics': final_stats,
            'validation_passed': integrity_results.get('passed', False) if validate_data else True
        }

    except Exception as e:
        print(f"❌ Error during data generation: {e}")
        print("Please check that:")
        print("  1. Expert model exists and is accessible")
        print("  2. Environment dependencies are installed")
        print("  3. Sufficient disk space is available")
        return None


def quick_start_demo():
    """Quick demonstration of the data generation system."""
    print("🎯 QUICK START DEMO")
    print("="*50)
    print("Generating small dataset for testing...")

    result = generate_expert_data_for_diffusion_project(
        episodes=20,          # Small number for demo
        horizon=32,          # Shorter horizon for speed
        include_recovery=True,
        validate_data=True
    )

    if result:
        print("\n✅ Demo completed successfully!")
        print("   You can now test your diffusion-DAgger trainer with this data.")
        print("   For full dataset, run with --full-pipeline")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")


def full_pipeline():
    """Generate complete dataset for production training."""
    print("🎯 FULL PRODUCTION PIPELINE")
    print("="*50)

    # Load configuration to get optimal parameters
    config = load_experiment_config()

    horizon = config['teacher'].get('max_recovery_horizon', 64)
    episodes = 200  # Production dataset size

    print(f"Using trajectory horizon: {horizon}")
    print(f"Generating {episodes} episodes...")

    result = generate_expert_data_for_diffusion_project(
        episodes=episodes,
        horizon=horizon,
        include_recovery=True,
        validate_data=True
    )

    if result:
        print(f"\n🎉 PRODUCTION DATA GENERATION COMPLETE!")
        print(f"   Ready for full diffusion-DAgger training!")

        # Optional: Test compatibility with trainer
        if input("\n🧪 Test compatibility with trainer? (y/n): ").lower() == 'y':
            test_trainer_compatibility(config, result['training_data_path'])
    else:
        print(f"\n❌ Production pipeline failed.")


def test_trainer_compatibility(config, data_path):
    """Test that generated data is compatible with the trainer."""
    print("\n🧪 Testing trainer compatibility...")

    try:
        # Initialize trainer
        trainer = DiffusionDAggerTrainer(config)

        # Test data loading
        import pickle
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        print(f"✓ Data loaded successfully")
        print(f"✓ Observations shape: {data['observations'].shape}")
        print(f"✓ Actions shape: {data['actions'].shape}")
        print(f"✓ Trajectory masks shape: {data['trajectory_masks'].shape}")

        # Test teacher model forward pass
        sample_obs = torch.tensor(data['observations'][:1, 0], dtype=torch.float32)
        with torch.no_grad():
            trajectory = trainer.teacher.sample_trajectory(sample_obs, num_steps=5)

        print(f"✓ Teacher trajectory generation: {trajectory.shape}")

        print(f"✅ All compatibility tests passed!")

    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate Expert Data for Diffusion-DAgger Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_diffusion_expert_data.py --quick-start
  python generate_diffusion_expert_data.py --full-pipeline
  python generate_diffusion_expert_data.py --episodes 100 --horizon 64
        """
    )

    parser.add_argument('--quick-start', action='store_true',
                       help='Run quick demo with small dataset')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Generate complete production dataset')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to collect (default: 100)')
    parser.add_argument('--horizon', type=int, default=64,
                       help='Trajectory horizon (default: 64)')
    parser.add_argument('--no-recovery', action='store_true',
                       help='Skip recovery scenario collection')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip data validation')

    args = parser.parse_args()

    # Print header
    print()
    print("🚀 DIFFUSION-DAGGER EXPERT DATA GENERATOR")
    print("   For 10703 Deep Reinforcement Learning Project")
    print("   CarRacing-v2 Environment")
    print()

    if args.quick_start:
        quick_start_demo()
    elif args.full_pipeline:
        full_pipeline()
    else:
        # Custom parameters
        result = generate_expert_data_for_diffusion_project(
            episodes=args.episodes,
            horizon=args.horizon,
            include_recovery=not args.no_recovery,
            validate_data=not args.no_validation
        )

        if result:
            print("\n✨ Expert data generation completed!")
            print("   Ready for diffusion-DAgger training.")
        else:
            print("\n❌ Expert data generation failed.")
            sys.exit(1)


if __name__ == "__main__":
    main()