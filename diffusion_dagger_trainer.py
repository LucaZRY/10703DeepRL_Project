"""
Diffusion-DAgger Training System for CarRacing DRL Project

This module implements the complete training pipeline combining:
- Diffusion Policy as teacher (trajectory-level guidance)
- Student Policy learning via DAgger with diffusion supervision
- Advantage-weighted distillation for sample efficiency

Usage:
    python diffusion_dagger_trainer.py --config config/experiment.yaml
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import yaml
import argparse
import os
from datetime import datetime
import wandb
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from diffusion_expert_data_generator import DiffusionExpertDataGenerator, TrajectorySegment
from dagger_carracing_2 import CNNPolicy, make_env, preprocess_obs
from data_utils import DataManager


class TrajectoryDiffusionPolicy(nn.Module):
    """
    Simplified trajectory diffusion policy for CarRacing.

    This serves as the teacher model that generates trajectory-level guidance
    for the student policy during DAgger training.
    """

    def __init__(self, obs_dim: Tuple[int, ...], action_dim: int,
                 trajectory_horizon: int, hidden_dim: int = 256):
        """
        Initialize trajectory diffusion policy.

        Args:
            obs_dim: Observation dimensions (C, H, W)
            action_dim: Action dimensions
            trajectory_horizon: Length of trajectory to predict
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.trajectory_horizon = trajectory_horizon
        self.hidden_dim = hidden_dim

        # CNN encoder for observations
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(obs_dim[0], 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # Calculate conv output dimension
        conv_out_dim = 64 * 7 * 7

        # Trajectory embedding network
        self.trajectory_embedding = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Action sequence decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),  # +32 for noise embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, trajectory_horizon * action_dim)
        )

        # Noise embedding for diffusion process
        self.noise_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observation to trajectory context.

        Args:
            obs: Observation tensor (B, C, H, W)

        Returns:
            Trajectory context embedding (B, hidden_dim)
        """
        # CNN encoding
        conv_features = self.obs_encoder(obs)
        conv_features = conv_features.view(conv_features.size(0), -1)

        # Trajectory embedding
        context = self.trajectory_embedding(conv_features)
        return context

    def forward(self, obs: torch.Tensor, noise_level: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for trajectory generation.

        Args:
            obs: Current observation (B, C, H, W)
            noise_level: Noise level for diffusion (B, 1)

        Returns:
            Predicted action trajectory (B, horizon, action_dim)
        """
        batch_size = obs.size(0)

        # Encode observation
        context = self.encode_observation(obs)

        # Handle noise level
        if noise_level is None:
            noise_level = torch.zeros(batch_size, 1, device=obs.device)

        # Embed noise
        noise_emb = self.noise_embedding(noise_level)

        # Combine context and noise
        combined = torch.cat([context, noise_emb], dim=1)

        # Decode action trajectory
        action_flat = self.action_decoder(combined)
        action_traj = action_flat.view(batch_size, self.trajectory_horizon, self.action_dim)

        return action_traj

    def sample_trajectory(self, obs: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        Sample trajectory using simplified diffusion process.

        Args:
            obs: Current observation
            num_steps: Number of denoising steps

        Returns:
            Sampled action trajectory
        """
        self.eval()

        with torch.no_grad():
            batch_size = obs.size(0)
            device = obs.device

            # Start with random noise
            trajectory = torch.randn(
                batch_size, self.trajectory_horizon, self.action_dim,
                device=device
            )

            # Denoising steps
            for step in range(num_steps):
                noise_level = torch.full((batch_size, 1), step / num_steps, device=device)

                # Predict noise/trajectory
                pred_trajectory = self.forward(obs, noise_level)

                # Simple denoising update
                alpha = 1.0 - (step + 1) / num_steps
                trajectory = alpha * trajectory + (1 - alpha) * pred_trajectory

                # Clamp actions to valid ranges
                trajectory[:, :, 0] = torch.clamp(trajectory[:, :, 0], -1, 1)  # Steering
                trajectory[:, :, 1:] = torch.clamp(trajectory[:, :, 1:], 0, 1)   # Gas, brake

        return trajectory


class DiffusionDAggerTrainer:
    """
    Complete training system combining diffusion teacher with DAgger student learning.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize diffusion-DAgger trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config['device'])

        # Initialize environment
        self.env = make_env(render_mode=None)

        # Get dimensions
        dummy_obs, _ = self.env.reset()
        self.obs_shape = preprocess_obs(dummy_obs).shape  # (4, 84, 84)
        self.action_dim = 3

        # Initialize diffusion teacher
        self.teacher = TrajectoryDiffusionPolicy(
            obs_dim=self.obs_shape,
            action_dim=self.action_dim,
            trajectory_horizon=config['teacher']['max_recovery_horizon'],
            hidden_dim=256
        ).to(self.device)

        # Initialize student policy
        self.student = CNNPolicy(self.action_dim).to(self.device)

        # Optimizers
        self.teacher_optimizer = optim.Adam(
            self.teacher.parameters(),
            lr=config['teacher'].get('lr', 3e-4)
        )
        self.student_optimizer = optim.Adam(
            self.student.parameters(),
            lr=config['student']['actor']['lr']
        )

        # Data management
        self.data_manager = DataManager()
        self.expert_data_generator = DiffusionExpertDataGenerator(
            expert_model_path=config['teacher'].get('ckpt', 'ppo_discrete_carracing.pt'),
            trajectory_horizon=config['teacher']['max_recovery_horizon']
        )

        # Training state
        self.current_iteration = 0
        self.training_stats = {
            'teacher_losses': [],
            'student_losses': [],
            'dagger_performance': [],
            'iteration_rewards': []
        }

        # Setup logging
        if config['logging']['use_wandb']:
            wandb.init(
                project=config['logging']['project'],
                name=config['logging']['name'],
                config=config
            )

    def pretrain_teacher(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Pre-train the diffusion teacher on expert demonstrations.

        Args:
            num_episodes: Number of episodes to collect for pre-training

        Returns:
            Training statistics
        """
        print(f"\n{'='*60}")
        print("PRE-TRAINING DIFFUSION TEACHER")
        print("="*60)

        # Collect expert trajectory data
        print("Collecting expert trajectory data...")
        self.expert_data_generator.collect_trajectory_episodes(
            num_episodes=num_episodes,
            enable_perturbations=True
        )

        # Generate training data
        training_data_path = self.expert_data_generator.generate_diffusion_training_data(
            output_dir="teacher_pretraining_data"
        )

        # Load training data
        with open(training_data_path, 'rb') as f:
            data = pickle.load(f)

        # Create data loader
        dataset = self._create_trajectory_dataset(data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True, num_workers=0
        )

        # Training loop
        self.teacher.train()
        total_loss = 0
        num_batches = 0

        print("Training diffusion teacher...")
        for epoch in tqdm(range(20), desc="Teacher Pre-training"):
            epoch_loss = 0

            for batch in dataloader:
                obs, trajectories, masks = batch
                obs = obs.to(self.device)
                trajectories = trajectories.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                self.teacher_optimizer.zero_grad()

                # Add noise for diffusion training
                noise = torch.randn_like(trajectories)
                noise_level = torch.rand(obs.size(0), 1, device=self.device)
                noisy_trajectories = trajectories + noise_level.unsqueeze(-1) * noise

                # Predict trajectory
                pred_trajectories = self.teacher.forward(obs, noise_level)

                # Compute loss (only on valid timesteps)
                loss = F.mse_loss(pred_trajectories * masks.unsqueeze(-1),
                                trajectories * masks.unsqueeze(-1))

                loss.backward()
                self.teacher_optimizer.step()

                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1

            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

        avg_loss = total_loss / num_batches
        print(f"Teacher pre-training completed. Average loss: {avg_loss:.4f}")

        # Save teacher model
        teacher_path = "diffusion_teacher_pretrained.pt"
        torch.save(self.teacher.state_dict(), teacher_path)
        print(f"Teacher model saved: {teacher_path}")

        return {"avg_loss": avg_loss, "total_batches": num_batches}

    def run_dagger_iteration(self, iteration: int) -> Dict[str, float]:
        """
        Run one DAgger iteration with diffusion teacher guidance.

        Args:
            iteration: Iteration number

        Returns:
            Iteration statistics
        """
        print(f"\n{'='*60}")
        print(f"DAGGER ITERATION {iteration + 1}")
        print("="*60)

        # Collection parameters
        num_episodes = self.config['dagger_recovery'].get('num_episodes', 20)
        beta = max(0.1, 1.0 - iteration * 0.1)  # Decay expert mixing

        # Collect data with current student
        print(f"Collecting DAgger data (β={beta:.2f})...")
        dagger_data = self._collect_dagger_data(num_episodes, beta)

        # Train student on aggregated dataset
        print("Training student policy...")
        student_stats = self._train_student(dagger_data)

        # Evaluate current student performance
        print("Evaluating student performance...")
        eval_stats = self._evaluate_student(num_episodes=10)

        # Update training statistics
        iteration_stats = {
            'iteration': iteration,
            'beta': beta,
            'student_loss': student_stats['avg_loss'],
            'eval_return': eval_stats['avg_return'],
            'eval_success_rate': eval_stats['success_rate'],
            'num_trajectories': len(dagger_data['trajectories'])
        }

        self.training_stats['dagger_performance'].append(iteration_stats)

        # Log to wandb
        if self.config['logging']['use_wandb']:
            wandb.log({
                f"dagger/iteration": iteration,
                f"dagger/beta": beta,
                f"dagger/student_loss": student_stats['avg_loss'],
                f"dagger/eval_return": eval_stats['avg_return'],
                f"dagger/success_rate": eval_stats['success_rate']
            })

        print(f"Iteration {iteration + 1} completed:")
        print(f"  Student loss: {student_stats['avg_loss']:.4f}")
        print(f"  Eval return: {eval_stats['avg_return']:.2f}")
        print(f"  Success rate: {eval_stats['success_rate']:.2f}")

        return iteration_stats

    def _collect_dagger_data(self, num_episodes: int, beta: float) -> Dict[str, Any]:
        """Collect DAgger data with diffusion teacher guidance."""
        dagger_trajectories = []

        for episode in tqdm(range(num_episodes), desc="DAgger Collection"):
            obs, _ = self.env.reset()
            trajectory_data = {
                'observations': [],
                'student_actions': [],
                'teacher_actions': [],
                'executed_actions': [],
                'rewards': []
            }

            done = False
            while not done:
                obs_processed = preprocess_obs(obs)
                obs_tensor = torch.tensor(obs_processed, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Get student action
                with torch.no_grad():
                    self.student.eval()
                    student_action = self.student(obs_tensor).cpu().numpy()[0]

                # Get teacher trajectory guidance
                with torch.no_grad():
                    self.teacher.eval()
                    teacher_trajectory = self.teacher.sample_trajectory(obs_tensor, num_steps=5)
                    teacher_action = teacher_trajectory[0, 0].cpu().numpy()  # First action of trajectory

                # Determine executed action (beta mixing)
                if np.random.random() < beta:
                    executed_action = teacher_action
                else:
                    executed_action = student_action

                # Store data
                trajectory_data['observations'].append(obs_processed)
                trajectory_data['student_actions'].append(student_action)
                trajectory_data['teacher_actions'].append(teacher_action)
                trajectory_data['executed_actions'].append(executed_action)

                # Execute action
                next_obs, reward, terminated, truncated, info = self.env.step(executed_action)
                done = terminated or truncated

                trajectory_data['rewards'].append(reward)
                obs = next_obs

            dagger_trajectories.append(trajectory_data)

        return {'trajectories': dagger_trajectories}

    def _train_student(self, dagger_data: Dict[str, Any]) -> Dict[str, float]:
        """Train student policy on DAgger data."""
        # Prepare training data
        observations = []
        actions = []  # Teacher actions for supervision

        for trajectory in dagger_data['trajectories']:
            observations.extend(trajectory['observations'])
            actions.extend(trajectory['teacher_actions'])

        # Convert to tensors
        obs_tensor = torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in observations]).to(self.device)
        act_tensor = torch.tensor(actions, dtype=torch.float32).to(self.device)

        # Training parameters
        batch_size = 64
        num_epochs = 10
        total_loss = 0
        num_batches = 0

        self.student.train()

        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(len(obs_tensor))

            for i in range(0, len(obs_tensor), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_obs = obs_tensor[batch_indices]
                batch_acts = act_tensor[batch_indices]

                # Forward pass
                self.student_optimizer.zero_grad()
                pred_actions = self.student(batch_obs)

                # Compute loss
                loss = F.mse_loss(pred_actions, batch_acts)

                # Backward pass
                loss.backward()
                self.student_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {'avg_loss': avg_loss}

    def _evaluate_student(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current student policy performance."""
        self.student.eval()

        episode_returns = []
        successful_episodes = 0

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_return = 0
            done = False

            while not done:
                obs_processed = preprocess_obs(obs)
                obs_tensor = torch.tensor(obs_processed, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    action = self.student(obs_tensor).cpu().numpy()[0]

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_return += reward

            episode_returns.append(episode_return)
            if episode_return > 700:  # Success threshold
                successful_episodes += 1

        return {
            'avg_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'success_rate': successful_episodes / num_episodes
        }

    def _create_trajectory_dataset(self, data: Dict[str, Any]):
        """Create PyTorch dataset from trajectory data."""
        class TrajectoryDataset(torch.utils.data.Dataset):
            def __init__(self, observations, actions, masks):
                self.observations = torch.tensor(observations, dtype=torch.float32)
                self.actions = torch.tensor(actions, dtype=torch.float32)
                self.masks = torch.tensor(masks, dtype=torch.float32)

            def __len__(self):
                return len(self.observations)

            def __getitem__(self, idx):
                return self.observations[idx], self.actions[idx], self.masks[idx]

        return TrajectoryDataset(
            data['observations'][:, 0],  # Use first observation of each trajectory
            data['actions'],
            data['trajectory_masks']
        )

    def train(self, num_iterations: int = 10):
        """
        Run complete diffusion-DAgger training.

        Args:
            num_iterations: Number of DAgger iterations
        """
        print(f"\n{'='*80}")
        print("DIFFUSION-DAGGER TRAINING")
        print("="*80)

        # Pre-train diffusion teacher
        teacher_stats = self.pretrain_teacher(num_episodes=100)

        # Run DAgger iterations
        for iteration in range(num_iterations):
            iteration_stats = self.run_dagger_iteration(iteration)

            # Save checkpoints
            if (iteration + 1) % 5 == 0:
                self._save_checkpoints(iteration)

        # Final evaluation and save
        final_stats = self._evaluate_student(num_episodes=20)
        self._save_final_models()
        self._generate_training_report()

        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print("="*80)
        print(f"Final student performance:")
        print(f"  Average return: {final_stats['avg_return']:.2f}")
        print(f"  Success rate: {final_stats['success_rate']:.2f}")

    def _save_checkpoints(self, iteration: int):
        """Save model checkpoints."""
        os.makedirs("checkpoints", exist_ok=True)

        torch.save(self.teacher.state_dict(), f"checkpoints/teacher_iter_{iteration}.pt")
        torch.save(self.student.state_dict(), f"checkpoints/student_iter_{iteration}.pt")

        # Save training stats
        with open(f"checkpoints/stats_iter_{iteration}.pkl", 'wb') as f:
            pickle.dump(self.training_stats, f)

    def _save_final_models(self):
        """Save final trained models."""
        torch.save(self.teacher.state_dict(), "diffusion_teacher_final.pt")
        torch.save(self.student.state_dict(), "student_dagger_diffusion_final.pt")
        print("Final models saved: diffusion_teacher_final.pt, student_dagger_diffusion_final.pt")

    def _generate_training_report(self):
        """Generate comprehensive training report."""
        # Create visualizations
        plt.figure(figsize=(15, 10))

        # DAgger performance over iterations
        plt.subplot(2, 3, 1)
        iterations = [stats['iteration'] for stats in self.training_stats['dagger_performance']]
        returns = [stats['eval_return'] for stats in self.training_stats['dagger_performance']]
        plt.plot(iterations, returns, 'b-o')
        plt.title('Student Performance Over DAgger Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Average Return')
        plt.grid(True)

        # Success rate progression
        plt.subplot(2, 3, 2)
        success_rates = [stats['eval_success_rate'] for stats in self.training_stats['dagger_performance']]
        plt.plot(iterations, success_rates, 'g-o')
        plt.title('Success Rate Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Success Rate')
        plt.grid(True)

        # Student loss over iterations
        plt.subplot(2, 3, 3)
        losses = [stats['student_loss'] for stats in self.training_stats['dagger_performance']]
        plt.plot(iterations, losses, 'r-o')
        plt.title('Student Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)

        # Beta decay
        plt.subplot(2, 3, 4)
        betas = [stats['beta'] for stats in self.training_stats['dagger_performance']]
        plt.plot(iterations, betas, 'm-o')
        plt.title('Expert Mixing Parameter (β)')
        plt.xlabel('Iteration')
        plt.ylabel('Beta')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('diffusion_dagger_training_report.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save detailed statistics
        with open('diffusion_dagger_training_stats.pkl', 'wb') as f:
            pickle.dump(self.training_stats, f)

        print("Training report saved: diffusion_dagger_training_report.png")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Diffusion-DAgger Training")
    parser.add_argument('--config', default='config/experiment.yaml',
                       help='Path to configuration file')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of DAgger iterations')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize trainer
    trainer = DiffusionDAggerTrainer(config)

    # Run training
    trainer.train(num_iterations=args.iterations)


if __name__ == "__main__":
    main()