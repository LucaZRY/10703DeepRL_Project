"""
F1TENTH Diffusion Policy Training

Adapted from train_diffusion.py for F1TENTH racing data.
Trains a diffusion model on F1TENTH expert trajectories for policy generation.

Key adaptations:
- Handles LiDAR observations instead of camera images
- Uses [steering, speed] actions instead of [steer, gas, brake]
- Optimized for F1TENTH state/action dimensions
- Compatible with existing PolicyDiffusionTransformer

Input:  data/expert_f1tenth/ with:
        states.npy:  (num_traj, T, lidar_dim)
        actions.npy: (num_traj, T, 2)

Output: f1tenth_diffusion_expert.pt (trained diffusion model)
"""

import argparse
import os
import pickle
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt

from src.models import PolicyDiffusionTransformer

class F1TenthDiffusionTrainer:
    """
    Diffusion trainer specifically for F1TENTH racing trajectories.

    Handles LiDAR-based observations and [steering, speed] actions.
    """

    def __init__(
        self,
        model: PolicyDiffusionTransformer,
        optimizer: torch.optim.Optimizer,
        states_array: np.ndarray,
        actions_array: np.ndarray,
        device: Union[torch.device, str] = "cpu",
        num_train_diffusion_timesteps: int = 30,
        max_trajectory_length: Optional[int] = None,
        action_horizon: int = 8,    # How many future actions to predict
        observation_horizon: int = 16,  # How many past observations to use
    ):
        """
        Initialize F1TENTH diffusion trainer.

        Args:
            states_array: (num_traj, T, lidar_dim) LiDAR observations
            actions_array: (num_traj, T, 2) [steering, speed] actions
        """
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.model.set_device(self.device)

        assert states_array.shape[:2] == actions_array.shape[:2], (
            f"states shape {states_array.shape}, actions shape {actions_array.shape}"
        )

        self.states = states_array.astype(np.float32)
        self.actions = actions_array.astype(np.float32)

        # F1TENTH specific action normalization
        # Steering: typically [-0.4, 0.4] radians -> normalize to [-1, 1]
        # Speed: typically [0, 8] m/s -> normalize to [0, 1]
        self.steering_range = 0.4189  # ~24 degrees
        self.speed_range = 8.0        # max speed

        # Normalize actions
        self.actions[:, :, 0] = np.clip(self.actions[:, :, 0] / self.steering_range, -1, 1)  # steering
        self.actions[:, :, 1] = np.clip(self.actions[:, :, 1] / self.speed_range, 0, 1)     # speed

        num_traj, T, _ = self.states.shape
        self.max_trajectory_length = max_trajectory_length or T
        self.action_horizon = min(action_horizon, T)
        self.observation_horizon = min(observation_horizon, T)

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_diffusion_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon",
            clip_sample=True,
            clip_sample_range=1.0
        )

        print(f"F1TENTH Diffusion Trainer initialized:")
        print(f"  Trajectories: {num_traj}")
        print(f"  Max trajectory length: {self.max_trajectory_length}")
        print(f"  State dim: {self.states.shape[2]}")
        print(f"  Action dim: {self.actions.shape[2]}")
        print(f"  Action horizon: {self.action_horizon}")
        print(f"  Observation horizon: {self.observation_horizon}")

    def prepare_training_data(self, batch_size: int = 256):
        """Prepare training data for diffusion training."""

        # Extract training samples
        obs_sequences = []
        action_sequences = []

        num_traj, T, state_dim = self.states.shape

        for traj_idx in range(num_traj):
            traj_states = self.states[traj_idx]
            traj_actions = self.actions[traj_idx]

            # Find actual trajectory length (non-padded)
            actual_length = T
            for t in range(T):
                if np.allclose(traj_states[t], 0) and np.allclose(traj_actions[t], 0):
                    actual_length = t
                    break

            if actual_length < self.observation_horizon + self.action_horizon:
                continue

            # Sample segments from this trajectory
            max_start = actual_length - self.observation_horizon - self.action_horizon + 1

            for start_idx in range(0, max_start, max(1, max_start // 10)):  # Sample ~10 segments per trajectory
                # Observation sequence (past states)
                obs_seq = traj_states[start_idx:start_idx + self.observation_horizon]

                # Action sequence (future actions to predict)
                action_seq = traj_actions[start_idx + self.observation_horizon:
                                        start_idx + self.observation_horizon + self.action_horizon]

                obs_sequences.append(obs_seq)
                action_sequences.append(action_seq)

        if len(obs_sequences) == 0:
            raise ValueError("No valid training sequences found!")

        # Convert to tensors
        obs_tensor = torch.tensor(np.array(obs_sequences), dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(np.array(action_sequences), dtype=torch.float32, device=self.device)

        print(f"Prepared training data:")
        print(f"  Observation sequences: {obs_tensor.shape}")
        print(f"  Action sequences: {action_tensor.shape}")

        # Create dataloader
        dataset = TensorDataset(obs_tensor, action_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        return dataloader

    def train_step(self, obs_batch: torch.Tensor, action_batch: torch.Tensor) -> dict:
        """Single training step for diffusion model."""

        batch_size, obs_horizon, state_dim = obs_batch.shape
        _, action_horizon, action_dim = action_batch.shape

        # Sample random timesteps for diffusion
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()

        # Add noise to actions
        noise = torch.randn_like(action_batch)
        noisy_actions = self.noise_scheduler.add_noise(action_batch, noise, timesteps)

        # Create episode timesteps (trajectory time, not diffusion time)
        episode_timesteps = torch.arange(obs_horizon, device=self.device).unsqueeze(0).repeat(batch_size, 1)

        # Prepare model inputs
        # For F1TENTH, we use recent observations as "previous states"
        # and empty previous actions (or could use recent actions)
        previous_states = obs_batch  # Use all observations
        previous_actions = torch.zeros(batch_size, 1, action_dim, device=self.device)  # Dummy previous actions

        # Predict noise
        predicted_noise = self.model(
            previous_states=previous_states,
            previous_actions=previous_actions,
            noisy_actions=noisy_actions,
            episode_timesteps=episode_timesteps,
            noise_timesteps=timesteps.unsqueeze(1)
        )

        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'predicted_noise_std': predicted_noise.std().item(),
            'target_noise_std': noise.std().item()
        }

    def train(self, num_epochs: int = 100, batch_size: int = 256, log_interval: int = 10):
        """Full training loop."""

        print(f"Starting F1TENTH diffusion training for {num_epochs} epochs...")

        # Prepare data
        dataloader = self.prepare_training_data(batch_size)

        # Training loop
        losses = []
        self.model.train()

        for epoch in range(num_epochs):
            epoch_losses = []

            for batch_idx, (obs_batch, action_batch) in enumerate(dataloader):
                metrics = self.train_step(obs_batch, action_batch)
                epoch_losses.append(metrics['loss'])

                if batch_idx == 0 and epoch % log_interval == 0:
                    print(f"Epoch {epoch:3d}, Batch {batch_idx:3d}: "
                          f"Loss={metrics['loss']:.4f}, "
                          f"PredNoise={metrics['predicted_noise_std']:.4f}, "
                          f"TargetNoise={metrics['target_noise_std']:.4f}")

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

            if epoch % log_interval == 0:
                print(f"Epoch {epoch:3d}: Average Loss = {avg_loss:.4f}")

        print("Training completed!")
        return losses

    def save_model(self, save_path: str):
        """Save the trained diffusion model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'state_dim': self.states.shape[2],
                'action_dim': self.actions.shape[2],
                'action_horizon': self.action_horizon,
                'observation_horizon': self.observation_horizon,
                'steering_range': self.steering_range,
                'speed_range': self.speed_range,
                'diffusion_steps': self.noise_scheduler.config.num_train_timesteps
            }
        }, save_path)
        print(f"Model saved to {save_path}")

def load_f1tenth_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load F1TENTH trajectory data."""

    states_path = os.path.join(data_dir, "states.npy")
    actions_path = os.path.join(data_dir, "actions.npy")

    if not os.path.exists(states_path) or not os.path.exists(actions_path):
        raise FileNotFoundError(f"F1TENTH data not found in {data_dir}")

    states = np.load(states_path)
    actions = np.load(actions_path)

    print(f"Loaded F1TENTH data:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")

    return states, actions

def main():
    parser = argparse.ArgumentParser(description="Train F1TENTH diffusion expert")
    parser.add_argument("--expert_dir", default="data/expert_f1tenth",
                       help="Directory with F1TENTH expert data")
    parser.add_argument("--save_path", default="f1tenth_diffusion_expert.pt",
                       help="Path to save trained model")

    # Model hyperparameters
    parser.add_argument("--num_layers", type=int, default=6,
                       help="Number of transformer layers")
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="Hidden dimension size")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--num_diffusion_steps", type=int, default=30,
                       help="Number of diffusion timesteps")

    # Training hyperparameters
    parser.add_argument("--train_steps", type=int, default=20000,
                       help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")

    # Data hyperparameters
    parser.add_argument("--action_horizon", type=int, default=8,
                       help="Number of future actions to predict")
    parser.add_argument("--observation_horizon", type=int, default=16,
                       help="Number of past observations to use")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    states, actions = load_f1tenth_data(args.expert_dir)

    state_dim = states.shape[2]
    action_dim = actions.shape[2]

    # Create model
    model = PolicyDiffusionTransformer(
        num_transformer_layers=args.num_layers,
        state_dim=state_dim,
        act_dim=action_dim,
        hidden_size=args.hidden_size,
        n_transformer_heads=args.num_heads,
        device=device,
        target="diffusion_policy"
    )

    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Create trainer
    trainer = F1TenthDiffusionTrainer(
        model=model,
        optimizer=optimizer,
        states_array=states,
        actions_array=actions,
        device=device,
        num_train_diffusion_timesteps=args.num_diffusion_steps,
        action_horizon=args.action_horizon,
        observation_horizon=args.observation_horizon
    )

    # Train
    num_epochs = max(1, args.train_steps // len(trainer.prepare_training_data(args.batch_size)))
    print(f"Training for {num_epochs} epochs...")

    losses = trainer.train(
        num_epochs=num_epochs,
        batch_size=args.batch_size,
        log_interval=max(1, num_epochs // 10)
    )

    # Save model
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else ".", exist_ok=True)
    trainer.save_model(args.save_path)

    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('F1TENTH Diffusion Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(args.save_path.replace('.pt', '_training_loss.png'), dpi=150, bbox_inches='tight')
    print(f"Training plot saved to {args.save_path.replace('.pt', '_training_loss.png')}")

    print("F1TENTH diffusion training completed!")

if __name__ == "__main__":
    main()