"""
Test F1TENTH Environment Setup

This script validates:
1. F1TENTH environment installation and functionality
2. Environment wrapper compatibility
3. PPO training pipeline basic functionality
4. Dataset generation format

Run this before full training to ensure everything is set up correctly.
"""

import numpy as np
import torch
import sys
import traceback
from f1tenth_wrappers import make_f1tenth_env, get_f1tenth_maps
from f1tenth_ppo import F1TenthPolicyNet, F1TenthValueNet

def test_environment_creation():
    """Test F1TENTH environment creation and basic functionality."""
    print("=== Testing Environment Creation ===")

    try:
        # Test environment creation
        env = make_f1tenth_env(map_name="Spielberg", render_mode=None)
        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        # Test reset
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Initial observation shape: {obs.shape}")
        print(f"  Observation dtype: {obs.dtype}")
        print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

        # Test a few environment steps
        print("\n--- Testing Environment Steps ---")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            print(f"Step {i+1}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Done: {done}, Truncated: {truncated}")
            print(f"  New obs shape: {obs.shape}")

            if done or truncated:
                obs, info = env.reset()
                print(f"  Environment reset after termination")
                break

        env.close()
        print("✓ Environment test completed successfully\n")
        return True

    except Exception as e:
        print(f"✗ Environment test failed: {str(e)}")
        print(f"Error details: {traceback.format_exc()}")
        return False

def test_policy_networks():
    """Test F1TENTH policy and value networks."""
    print("=== Testing Policy Networks ===")

    try:
        # Create dummy environment to get dimensions
        env = make_f1tenth_env()
        obs, _ = env.reset()
        obs_dim = obs.shape
        action_dim = env.action_space.shape[0]
        env.close()

        print(f"Network input dim: {obs_dim}")
        print(f"Network output dim: {action_dim}")

        # Test policy network
        policy_net = F1TenthPolicyNet(obs_dim, action_dim)
        print(f"✓ Policy network created: {sum(p.numel() for p in policy_net.parameters())} parameters")

        # Test value network
        value_net = F1TenthValueNet(obs_dim)
        print(f"✓ Value network created: {sum(p.numel() for p in value_net.parameters())} parameters")

        # Test forward pass
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            mean, std = policy_net(obs_tensor)
            value = value_net(obs_tensor)

        print(f"✓ Policy forward pass successful")
        print(f"  Mean shape: {mean.shape}, range: [{mean.min():.3f}, {mean.max():.3f}]")
        print(f"  Std shape: {std.shape}, range: [{std.min():.3f}, {std.max():.3f}]")
        print(f"✓ Value forward pass successful")
        print(f"  Value shape: {value.shape}, value: {value.item():.3f}")

        # Test action sampling
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        print(f"✓ Action sampling successful")
        print(f"  Sampled action: {action[0].numpy()}")
        print(f"  Log probability: {log_prob.item():.3f}")

        print("✓ Policy networks test completed successfully\n")
        return True

    except Exception as e:
        print(f"✗ Policy networks test failed: {str(e)}")
        print(f"Error details: {traceback.format_exc()}")
        return False

def test_dataset_compatibility():
    """Test dataset generation and format compatibility."""
    print("=== Testing Dataset Compatibility ===")

    try:
        # Simulate dataset creation
        env = make_f1tenth_env()
        obs, _ = env.reset()

        # Collect some dummy data
        obs_list = []
        actions_list = []
        rewards_list = []
        dones_list = []

        for i in range(100):
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)

            obs_list.append(obs.copy())
            actions_list.append(action.copy())
            rewards_list.append(reward)
            dones_list.append(done or truncated)

            obs = next_obs
            if done or truncated:
                obs, _ = env.reset()

        env.close()

        # Convert to numpy arrays (simulate dataset format)
        obs_array = np.array(obs_list, dtype=np.float32)
        actions_array = np.array(actions_list, dtype=np.float32)
        rewards_array = np.array(rewards_list, dtype=np.float32)
        dones_array = np.array(dones_list, dtype=bool)

        print(f"✓ Dataset arrays created")
        print(f"  Observations: {obs_array.shape}, dtype: {obs_array.dtype}")
        print(f"  Actions: {actions_array.shape}, dtype: {actions_array.dtype}")
        print(f"  Rewards: {rewards_array.shape}, dtype: {rewards_array.dtype}")
        print(f"  Dones: {dones_array.shape}, dtype: {dones_array.dtype}")

        # Test data ranges
        print(f"✓ Data ranges:")
        print(f"  Obs range: [{obs_array.min():.3f}, {obs_array.max():.3f}]")
        print(f"  Action range: [{actions_array.min():.3f}, {actions_array.max():.3f}]")
        print(f"  Reward range: [{rewards_array.min():.3f}, {rewards_array.max():.3f}]")
        print(f"  Episodes completed: {dones_array.sum()}")

        # Test trajectory segmentation (simulate convert_f1tenth_expert.py)
        episode_starts = [0]
        for i, done in enumerate(dones_array):
            if done and i < len(dones_array) - 1:
                episode_starts.append(i + 1)

        print(f"✓ Found {len(episode_starts)} episode boundaries")

        print("✓ Dataset compatibility test completed successfully\n")
        return True

    except Exception as e:
        print(f"✗ Dataset compatibility test failed: {str(e)}")
        print(f"Error details: {traceback.format_exc()}")
        return False

def test_installation_requirements():
    """Test if all required packages are installed."""
    print("=== Testing Installation Requirements ===")

    required_packages = {
        'numpy': 'NumPy for numerical computations',
        'torch': 'PyTorch for deep learning',
        'gymnasium': 'Gymnasium for RL environments',
        'matplotlib': 'Matplotlib for plotting'
    }

    optional_packages = {
        'f1tenth_gym': 'F1TENTH racing environment (will fallback to CarRacing if not available)'
    }

    all_good = True

    # Test required packages
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} - {description}")
        except ImportError:
            print(f"✗ {package} - {description} [MISSING]")
            all_good = False

    # Test optional packages
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} - {description}")
        except ImportError:
            print(f"⚠ {package} - {description} [OPTIONAL - using fallback]")

    # Test PyTorch device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ PyTorch device: {device}")

    if all_good:
        print("✓ All required packages are installed\n")
    else:
        print("✗ Some required packages are missing\n")

    return all_good

def main():
    """Run all tests."""
    print("F1TENTH Setup Validation")
    print("=" * 50)

    tests = [
        ("Installation Requirements", test_installation_requirements),
        ("Environment Creation", test_environment_creation),
        ("Policy Networks", test_policy_networks),
        ("Dataset Compatibility", test_dataset_compatibility)
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        results[test_name] = test_func()

    # Summary
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:.<30} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! F1TENTH setup is ready for training.")
        print("\nNext steps:")
        print("1. Run: python f1tenth_ppo.py")
        print("2. Convert data: python convert_f1tenth_expert.py")
        print("3. Train diffusion: python train_diffusion_f1tenth.py")
    else:
        print("\n❌ Some tests failed. Please address the issues before training.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)