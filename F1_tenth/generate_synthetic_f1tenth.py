"""
Generate Synthetic F1TENTH Racing Data

Uses trained diffusion model to generate synthetic F1TENTH racing trajectories.
This data can then be used for student policy training via behavioral cloning.

Input:  f1tenth_diffusion_expert.pt (trained diffusion model)
Output: data/generated_f1tenth/ with:
        states.npy:  (N, lidar_dim) synthetic observations
        actions.npy: (N, 2) synthetic [steering, speed] actions
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from diffusers import DDPMScheduler

from src.models import PolicyDiffusionTransformer

class F1TenthSyntheticGenerator:
    """
    Generate synthetic F1TENTH racing data using trained diffusion model.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """Initialize generator with trained diffusion model."""

        self.device = torch.device(device)

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']

        # Create model
        self.model = PolicyDiffusionTransformer(
            num_transformer_layers=6,  # Default values, should match training
            state_dim=self.config['state_dim'],
            act_dim=self.config['action_dim'],
            hidden_size=128,
            n_transformer_heads=8,
            device=self.device,
            target="diffusion_policy"
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Initialize noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config['diffusion_steps'],
            beta_schedule="linear",
            prediction_type="epsilon",
            clip_sample=True,
            clip_sample_range=1.0
        )

        print(f"Loaded F1TENTH diffusion model:")
        print(f"  State dim: {self.config['state_dim']}")
        print(f"  Action dim: {self.config['action_dim']}")
        print(f"  Action horizon: {self.config['action_horizon']}")
        print(f"  Observation horizon: {self.config['observation_horizon']}")

    def generate_initial_observations(self, batch_size: int) -> torch.Tensor:
        """
        Generate or sample initial LiDAR observations.

        For now, we create plausible LiDAR patterns.
        In practice, you might want to sample from real initial states.
        """

        state_dim = self.config['state_dim']
        obs_horizon = self.config['observation_horizon']

        # Generate synthetic LiDAR observations
        # Simulate a racing track with walls at various distances
        observations = []

        for _ in range(batch_size):
            # Create a sequence of LiDAR observations
            obs_seq = []

            for t in range(obs_horizon):
                # Simulate LiDAR scan around a track
                angles = np.linspace(-np.pi, np.pi, state_dim)
                ranges = []

                # Simulate track with left/right walls and some variation
                for angle in angles:
                    if -np.pi/3 < angle < np.pi/3:  # Front sector
                        base_range = 8.0 + 2.0 * np.sin(angle * 3)  # Road ahead
                    elif angle < -np.pi/2 or angle > np.pi/2:  # Behind
                        base_range = 3.0 + np.random.normal(0, 0.5)
                    else:  # Sides
                        base_range = 2.0 + 1.0 * np.random.normal(0, 0.3)

                    # Add some noise and ensure positive
                    range_val = max(0.1, base_range + np.random.normal(0, 0.2))
                    ranges.append(range_val)

                # Normalize to [0, 1] (assuming max range of 30m)
                ranges = np.array(ranges) / 30.0
                ranges = np.clip(ranges, 0, 1)

                obs_seq.append(ranges)

            observations.append(obs_seq)

        return torch.tensor(observations, dtype=torch.float32, device=self.device)

    def denoise_actions(self, initial_observations: torch.Tensor, num_inference_steps: int = 10) -> torch.Tensor:
        """
        Generate action sequences using the diffusion model.

        Args:
            initial_observations: (batch_size, obs_horizon, state_dim)
            num_inference_steps: Number of denoising steps

        Returns:
            Generated actions: (batch_size, action_horizon, action_dim)
        """

        batch_size = initial_observations.shape[0]
        action_horizon = self.config['action_horizon']
        action_dim = self.config['action_dim']

        # Set inference timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Start with random noise
        actions = torch.randn(batch_size, action_horizon, action_dim, device=self.device)

        # Prepare model inputs
        episode_timesteps = torch.arange(
            self.config['observation_horizon'], device=self.device
        ).unsqueeze(0).repeat(batch_size, 1)

        # Dummy previous actions (or could use zeros)
        previous_actions = torch.zeros(batch_size, 1, action_dim, device=self.device)

        # Denoising loop
        with torch.no_grad():
            for t in tqdm(self.noise_scheduler.timesteps, desc="Generating actions"):
                # Expand timestep to batch dimension
                timestep_batch = t.unsqueeze(0).repeat(batch_size).unsqueeze(1)

                # Predict noise
                noise_pred = self.model(
                    previous_states=initial_observations,
                    previous_actions=previous_actions,
                    noisy_actions=actions,
                    episode_timesteps=episode_timesteps,
                    noise_timesteps=timestep_batch
                )

                # Remove noise
                actions = self.noise_scheduler.step(noise_pred, t, actions).prev_sample

        return actions

    def generate_synthetic_data(self, num_samples: int = 50000, batch_size: int = 256) -> tuple:
        """
        Generate synthetic F1TENTH racing data.

        Args:
            num_samples: Total number of state-action pairs to generate
            batch_size: Generation batch size

        Returns:
            (states, actions) arrays
        """

        print(f"Generating {num_samples} synthetic F1TENTH samples...")

        all_states = []
        all_actions = []

        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Generation batches"):
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)

            # Generate initial observations
            initial_obs = self.generate_initial_observations(current_batch_size)

            # Generate actions using diffusion model
            generated_actions = self.denoise_actions(initial_obs, num_inference_steps=10)

            # Extract state-action pairs
            # For each sequence, we create multiple (state, action) pairs
            for i in range(current_batch_size):
                obs_seq = initial_obs[i]  # (obs_horizon, state_dim)
                action_seq = generated_actions[i]  # (action_horizon, action_dim)

                # Create (state, action) pairs from the sequence
                for t in range(self.config['action_horizon']):
                    if t < len(obs_seq) and t < len(action_seq):
                        # Use last observation and corresponding action
                        state = obs_seq[-1].cpu().numpy()  # Use most recent observation
                        action = action_seq[t].cpu().numpy()

                        all_states.append(state)
                        all_actions.append(action)

        # Convert to numpy arrays
        states = np.array(all_states, dtype=np.float32)
        actions = np.array(all_actions, dtype=np.float32)

        # Denormalize actions back to original ranges
        actions[:, 0] *= self.config['steering_range']  # Steering: [-0.4, 0.4] radians
        actions[:, 1] *= self.config['speed_range']     # Speed: [0, 8] m/s

        print(f"Generated synthetic data:")
        print(f"  States: {states.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Steering range: [{actions[:, 0].min():.3f}, {actions[:, 0].max():.3f}]")
        print(f"  Speed range: [{actions[:, 1].min():.3f}, {actions[:, 1].max():.3f}]")

        return states, actions

def save_synthetic_data(states: np.ndarray, actions: np.ndarray, output_dir: str):
    """Save synthetic data in format compatible with student training."""

    os.makedirs(output_dir, exist_ok=True)

    states_path = os.path.join(output_dir, "states.npy")
    actions_path = os.path.join(output_dir, "actions.npy")

    np.save(states_path, states)
    np.save(actions_path, actions)

    print(f"Saved synthetic data to {output_dir}/")
    print(f"  states.npy: {states.shape}")
    print(f"  actions.npy: {actions.shape}")

    # Save metadata
    metadata = {
        'num_samples': len(states),
        'state_dim': states.shape[1],
        'action_dim': actions.shape[1],
        'steering_range': [actions[:, 0].min(), actions[:, 0].max()],
        'speed_range': [actions[:, 1].min(), actions[:, 1].max()],
        'state_range': [states.min(), states.max()]
    }

    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic F1TENTH racing data")
    parser.add_argument("--model_path", default="f1tenth_diffusion_expert.pt",
                       help="Path to trained diffusion model")
    parser.add_argument("--output_dir", default="data/generated_f1tenth",
                       help="Output directory for synthetic data")
    parser.add_argument("--num_samples", type=int, default=50000,
                       help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Generation batch size")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cpu, cuda)")

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the diffusion model first using:")
        print("  python train_diffusion_f1tenth.py")
        return

    # Generate data
    generator = F1TenthSyntheticGenerator(args.model_path, device=device)

    states, actions = generator.generate_synthetic_data(
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )

    # Save data
    save_synthetic_data(states, actions, args.output_dir)

    print("Synthetic F1TENTH data generation completed!")

if __name__ == "__main__":
    main()