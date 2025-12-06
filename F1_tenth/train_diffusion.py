# train_diffusion.py
import argparse
import os
import pickle
from typing import Optional, Tuple, Union

import numpy as np
import torch
from diffusers import DDPMScheduler

from src.models import PolicyDiffusionTransformer


class DiffusionExpertTrainer:
    """
    Offline trainer for the diffusion 'expert' policy.

    This is essentially a cleaned-up, self-contained version of your HW4
    TrainDiffusionPolicy class, but without any environment interaction.
    It just learns to predict noise on (state, action) trajectories.
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
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.model.set_device(self.device)

        assert states_array.shape[:2] == actions_array.shape[:2], (
            f"states shape {states_array.shape}, actions shape {actions_array.shape}"
        )

        self.states = states_array.astype(np.float32)
        self.actions = actions_array.astype(np.float32)

        self.clip_sample_range = 1.0
        self.actions = np.clip(self.actions, -self.clip_sample_range, self.clip_sample_range)

        num_traj, T, _ = self.states.shape
        self.max_trajectory_length = max_trajectory_length or T

        # compute per-trajectory valid length (assumes padding is all zeros)
        self.trajectory_lengths = []
        for i in range(num_traj):
            valid_mask = (np.abs(self.states[i]).sum(axis=-1) != 0)
            length = int(valid_mask.sum())
            if length == 0:
                length = 1
            self.trajectory_lengths.append(length)

        # normalize states & actions using only valid timesteps
        all_states = np.concatenate(
            [self.states[i, :self.trajectory_lengths[i]] for i in range(num_traj)],
            axis=0,
        )
        all_actions = np.concatenate(
            [self.actions[i, :self.trajectory_lengths[i]] for i in range(num_traj)],
            axis=0,
        )
        eps = 1e-8
        self.states_mean = all_states.mean(axis=0)
        self.states_std = all_states.std(axis=0) + eps
        self.actions_mean = all_actions.mean(axis=0)
        self.actions_std = all_actions.std(axis=0) + eps

        self.states = (self.states - self.states_mean) / self.states_std
        self.actions = (self.actions - self.actions_mean) / self.actions_std

        self.num_train_diffusion_timesteps = num_train_diffusion_timesteps

        self.training_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_diffusion_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample_range=self.clip_sample_range,
        )
        self.inference_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_diffusion_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small_log",
            clip_sample_range=self.clip_sample_range,
        )
        self.inference_scheduler.alphas_cumprod = (
            self.inference_scheduler.alphas_cumprod.to(self.device)
        )

    # batching logic (same spirit as HW4) 

    def get_training_batch(
        self,
        batch_size: int,
        max_action_len: int = 3,
        num_previous_states: int = 5,
        num_previous_actions: int = 4,
    ):
        """
        Returns:
            previous_states_batch: (B, S, state_dim)
            previous_actions_batch: (B, A, act_dim)
            actions_batch: (B, L, act_dim)  - target actions for noise prediction
            episode_timesteps_batch: (B, S)
            previous_states_padding_batch: (B, S) bool; True = padded
            previous_actions_padding_batch: (B, A) bool
            actions_padding_batch: (B, L) bool
        """
        assert num_previous_states == num_previous_actions + 1, (
            f"num_previous_states={num_previous_states} must equal "
            f"num_previous_actions + 1={num_previous_actions + 1}"
        )

        num_traj = len(self.trajectory_lengths)
        probs = np.array(self.trajectory_lengths, dtype=np.float64)
        probs /= probs.sum()

        batch_indices = np.random.choice(
            np.arange(num_traj),
            size=batch_size,
            replace=True,
            p=probs,
        )

        prev_states_list, prev_actions_list, actions_list = [], [], []
        episode_ts_list = []
        prev_states_mask_list, prev_actions_mask_list, actions_mask_list = [], [], []

        for idx in batch_indices:
            traj_len = self.trajectory_lengths[idx]

            # choose a random "current" timestep
            end_index_state = np.random.randint(1, traj_len)
            start_index_state = max(0, end_index_state - num_previous_states)

            # previous actions lead up to current state
            start_prev_act = start_index_state
            end_prev_act = end_index_state - 1

            # actions to predict start at last previous action
            start_act = end_prev_act
            end_act = min(traj_len, start_act + max_action_len)

            prev_states = self.states[idx, start_index_state:end_index_state]
            prev_actions = self.actions[idx, start_prev_act:end_prev_act]
            actions = self.actions[idx, start_act:end_act]

            state_seq_len, state_dim = prev_states.shape
            prev_act_seq_len = prev_actions.shape[0]
            act_seq_len, act_dim = actions.shape

            # pad previous states/actions
            if state_seq_len < num_previous_states:
                pad_states = np.zeros(
                    (num_previous_states - state_seq_len, state_dim), dtype=np.float32
                )
                prev_states = np.concatenate([prev_states, pad_states], axis=0)

                pad_prev_actions = np.zeros(
                    (num_previous_actions - prev_act_seq_len, act_dim), dtype=np.float32
                )
                prev_actions = np.concatenate([prev_actions, pad_prev_actions], axis=0)

                prev_states_mask = np.concatenate(
                    [
                        np.zeros(state_seq_len, dtype=bool),
                        np.ones(num_previous_states - state_seq_len, dtype=bool),
                    ],
                    axis=0,
                )
                prev_actions_mask = np.concatenate(
                    [
                        np.zeros(prev_act_seq_len, dtype=bool),
                        np.ones(num_previous_actions - prev_act_seq_len, dtype=bool),
                    ],
                    axis=0,
                )
            else:
                prev_states_mask = np.zeros(num_previous_states, dtype=bool)
                prev_actions_mask = np.zeros(num_previous_actions, dtype=bool)

            # pad predicted actions
            if act_seq_len < max_action_len:
                pad_actions = np.zeros(
                    (max_action_len - act_seq_len, act_dim), dtype=np.float32
                )
                actions = np.concatenate([actions, pad_actions], axis=0)
                actions_mask = np.concatenate(
                    [
                        np.zeros(act_seq_len, dtype=bool),
                        np.ones(max_action_len - act_seq_len, dtype=bool),
                    ],
                    axis=0,
                )
            else:
                actions_mask = np.zeros(max_action_len, dtype=bool)

            prev_states_list.append(prev_states)
            prev_actions_list.append(prev_actions)
            actions_list.append(actions)

            episode_ts = np.arange(start_index_state, start_index_state + num_previous_states)
            episode_ts_list.append(episode_ts)

            prev_states_mask_list.append(prev_states_mask)
            prev_actions_mask_list.append(prev_actions_mask)
            actions_mask_list.append(actions_mask)

        previous_states_batch = torch.from_numpy(
            np.stack(prev_states_list)
        ).float().to(self.device)
        previous_actions_batch = torch.from_numpy(
            np.stack(prev_actions_list)
        ).float().to(self.device)
        actions_batch = torch.from_numpy(
            np.stack(actions_list)
        ).float().to(self.device)
        episode_timesteps_batch = torch.from_numpy(
            np.stack(episode_ts_list)
        ).long().to(self.device)

        previous_states_padding_batch = torch.from_numpy(
            np.stack(prev_states_mask_list)
        ).bool().to(self.device)
        previous_actions_padding_batch = torch.from_numpy(
            np.stack(prev_actions_mask_list)
        ).bool().to(self.device)
        actions_padding_batch = torch.from_numpy(
            np.stack(actions_mask_list)
        ).bool().to(self.device)

        return (
            previous_states_batch,
            previous_actions_batch,
            actions_batch,
            episode_timesteps_batch,
            previous_states_padding_batch,
            previous_actions_padding_batch,
            actions_padding_batch,
        )

    def get_inference_timesteps(self) -> torch.Tensor:
        """
        Configure and return the timesteps used for inference sampling.
        """
        self.inference_scheduler.set_timesteps(
            self.num_train_diffusion_timesteps, device=self.device
        )
        return self.inference_scheduler.timesteps

    def training_step(self, batch_size: int) -> float:
        (
            previous_states_batch,
            previous_actions_batch,
            actions_batch,
            episode_timesteps_batch,
            previous_states_padding_batch,
            previous_actions_padding_batch,
            actions_padding_batch,
        ) = self.get_training_batch(batch_size=batch_size)

        # sample noise and timesteps
        noise = torch.randn_like(actions_batch)

        # scheduler timesteps are 1D (B,), model embedding uses (B, 1)
        t = torch.randint(
            low=0,
            high=self.num_train_diffusion_timesteps,
            size=(actions_batch.shape[0],),
            device=self.device,
            dtype=torch.long,
        )
        noisy_actions = self.training_scheduler.add_noise(actions_batch, noise, t)
        noise_timesteps = t.unsqueeze(1)  # shape (B, 1) for embedding lookup

        noise_pred = self.model(
            previous_states=previous_states_batch,
            previous_actions=previous_actions_batch,
            noisy_actions=noisy_actions,
            episode_timesteps=episode_timesteps_batch,
            noise_timesteps=noise_timesteps,
            previous_states_mask=previous_states_padding_batch,
            previous_actions_mask=previous_actions_padding_batch,
            actions_padding_mask=actions_padding_batch,
        )

        # MSE only over non-padded actions
        valid_mask = (~actions_padding_batch).unsqueeze(-1)  # (B, L, 1)
        valid_mask = valid_mask.expand_as(actions_batch)  # match (B, L, act_dim)

        mse_per_entry = (noise_pred - noise) ** 2
        mse = mse_per_entry[valid_mask].mean()

        self.optimizer.zero_grad()
        mse.backward()
        self.optimizer.step()

        return float(mse.detach().cpu().item())

    def diffusion_sample(
        self,
        previous_states: torch.Tensor,
        previous_actions: torch.Tensor,
        episode_timesteps: torch.Tensor,
        previous_states_padding_mask: Optional[torch.Tensor] = None,
        previous_actions_padding_mask: Optional[torch.Tensor] = None,
        actions_padding_mask: Optional[torch.Tensor] = None,
        max_action_len: int = 3,
    ) -> torch.Tensor:
        """
        Sample actions via reverse diffusion, conditioned on history.
        """
        was_training = self.model.training
        self.model.eval()

        previous_states = previous_states.to(self.device)
        previous_actions = previous_actions.to(self.device)
        episode_timesteps = episode_timesteps.to(self.device)

        batch_size = previous_states.shape[0]
        act_dim = self.actions.shape[-1]

        x_t = torch.randn(batch_size, max_action_len, act_dim, device=self.device)

        if previous_states_padding_mask is None:
            previous_states_padding_mask = torch.zeros(
                batch_size,
                previous_states.shape[1],
                device=self.device,
                dtype=torch.bool,
            )
        else:
            previous_states_padding_mask = previous_states_padding_mask.to(
                self.device
            )

        if previous_actions_padding_mask is None:
            previous_actions_padding_mask = torch.zeros(
                batch_size,
                previous_actions.shape[1],
                device=self.device,
                dtype=torch.bool,
            )
        else:
            previous_actions_padding_mask = previous_actions_padding_mask.to(
                self.device
            )

        if actions_padding_mask is None:
            actions_padding_mask = torch.zeros(
                batch_size, max_action_len, device=self.device, dtype=torch.bool
            )
        else:
            actions_padding_mask = actions_padding_mask.to(self.device)

        timesteps = self.get_inference_timesteps()

        with torch.no_grad():
            for t in timesteps:
                timestep_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
                noise_timesteps = torch.full(
                    (batch_size, 1),
                    fill_value=timestep_int,
                    device=self.device,
                    dtype=torch.long,
                )
                noise_pred = self.model(
                    previous_states=previous_states,
                    previous_actions=previous_actions,
                    noisy_actions=x_t,
                    episode_timesteps=episode_timesteps,
                    noise_timesteps=noise_timesteps,
                    previous_states_mask=previous_states_padding_mask,
                    previous_actions_mask=previous_actions_padding_mask,
                    actions_padding_mask=actions_padding_mask,
                )
                step_out = self.inference_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=x_t,
                )
                x_t = step_out.prev_sample

        actions_mean = torch.as_tensor(
            self.actions_mean, device=self.device, dtype=torch.float32
        )
        actions_std = torch.as_tensor(
            self.actions_std, device=self.device, dtype=torch.float32
        )
        pred_actions = x_t * actions_std.view(1, 1, -1) + actions_mean.view(1, 1, -1)
        pred_actions = torch.clamp(
            pred_actions, -self.clip_sample_range, self.clip_sample_range
        )

        if was_training:
            self.model.train()

        return pred_actions

    def generate_synthetic_dataset(
        self,
        num_samples: int,
        batch_size: int = 64,
        max_action_len: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic (state, action) pairs using the trained diffusion expert.

        Args:
            num_samples: total number of pairs to produce.
            batch_size: contexts sampled per iteration.
            max_action_len: number of actions generated per context.

        Returns:
            synthetic_states: (num_samples, state_dim)
            synthetic_actions: (num_samples, act_dim)
        """
        assert num_samples > 0, "num_samples must be positive"
        assert max_action_len >= 1, "max_action_len must be >= 1"

        synthetic_states: list[np.ndarray] = []
        synthetic_actions: list[np.ndarray] = []

        states_mean_t = torch.as_tensor(
            self.states_mean, device=self.device, dtype=torch.float32
        )
        states_std_t = torch.as_tensor(
            self.states_std, device=self.device, dtype=torch.float32
        )

        while len(synthetic_states) < num_samples:
            current_batch_size = min(batch_size, num_samples - len(synthetic_states))

            (
                prev_states_batch,
                prev_actions_batch,
                _actions_batch,
                episode_ts_batch,
                prev_states_mask_batch,
                prev_actions_mask_batch,
                actions_padding_batch,
            ) = self.get_training_batch(
                batch_size=current_batch_size,
                max_action_len=max_action_len,
            )

            sampled_actions = self.diffusion_sample(
                previous_states=prev_states_batch,
                previous_actions=prev_actions_batch,
                episode_timesteps=episode_ts_batch,
                previous_states_padding_mask=prev_states_mask_batch,
                previous_actions_padding_mask=prev_actions_mask_batch,
                actions_padding_mask=actions_padding_batch,
                max_action_len=max_action_len,
            )

            actions_valid = (~actions_padding_batch).cpu()
            prev_states_mask_cpu = prev_states_mask_batch.cpu()

            for b in range(current_batch_size):
                valid_state_mask = ~prev_states_mask_cpu[b]
                if valid_state_mask.any():
                    last_state_idx = int(torch.nonzero(valid_state_mask)[-1])
                else:
                    last_state_idx = prev_states_batch.shape[1] - 1

                last_state_norm = prev_states_batch[b, last_state_idx, :]
                last_state_raw = last_state_norm * states_std_t + states_mean_t
                state_np = last_state_raw.detach().cpu().numpy()

                for l in range(max_action_len):
                    if not actions_valid[b, l]:
                        continue
                    action_np = sampled_actions[b, l, :].detach().cpu().numpy()
                    synthetic_states.append(state_np)
                    synthetic_actions.append(action_np)
                    if len(synthetic_states) >= num_samples:
                        break
                if len(synthetic_states) >= num_samples:
                    break

        synthetic_states_np = np.stack(synthetic_states, axis=0).astype(np.float32)
        synthetic_actions_np = np.stack(synthetic_actions, axis=0).astype(np.float32)
        return synthetic_states_np, synthetic_actions_np

    def train(
        self,
        num_training_steps: int,
        batch_size: int,
        print_every: int = 1000,
        save_path: Optional[str] = None,
    ):
        self.model.train()
        losses = []
        for step in range(1, num_training_steps + 1):
            loss = self.training_step(batch_size)
            losses.append(loss)
            if step % print_every == 0:
                print(f"[diffusion expert] step {step}/{num_training_steps}, loss={loss:.6f}")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "states_mean": self.states_mean,
                "states_std": self.states_std,
                "actions_mean": self.actions_mean,
                "actions_std": self.actions_std,
            }
            torch.save(checkpoint, save_path)

        return np.array(losses, dtype=np.float32)


# data loading helper

def load_expert_data(expert_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tries a few reasonable filenames so you can reuse HW4 data easily.
    Expected shapes: (num_traj, max_T, state_dim) and (num_traj, max_T, act_dim)
    """

    # npy format: data/expert/states.npy, actions.npy
    states_npy = os.path.join(expert_dir, "states.npy")
    actions_npy = os.path.join(expert_dir, "actions.npy")
    if os.path.exists(states_npy) and os.path.exists(actions_npy):
        return np.load(states_npy), np.load(actions_npy)

    # npz format: data/expert/expert_trajectories.npz with keys 'states', 'actions'
    npz_path = os.path.join(expert_dir, "expert_trajectories.npz")
    if os.path.exists(npz_path):
        arr = np.load(npz_path)
        return arr["states"], arr["actions"]

    # HW4-style pickles: states_BC.pkl, actions_BC.pkl
    states_pkl = os.path.join(expert_dir, "states_BC.pkl")
    actions_pkl = os.path.join(expert_dir, "actions_BC.pkl")
    if os.path.exists(states_pkl) and os.path.exists(actions_pkl):
        with open(states_pkl, "rb") as f:
            states = pickle.load(f)
        with open(actions_pkl, "rb") as f:
            actions = pickle.load(f)
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)

    raise FileNotFoundError(
        f"Could not find expert data in {expert_dir}. "
        "Tried states.npy/actions.npy, expert_trajectories.npz, states_BC.pkl/actions_BC.pkl."
    )


# CLI entrypoint 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_dir", type=str, default="data/expert")
    parser.add_argument("--save_path", type=str, default="results/diffusion_expert/expert_model.pt")
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--num_diffusion_steps", type=int, default=30)
    parser.add_argument("--train_steps", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    states, actions = load_expert_data(args.expert_dir)
    num_traj, max_T, state_dim = states.shape
    act_dim = actions.shape[-1]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = PolicyDiffusionTransformer(
        num_transformer_layers=args.num_layers,
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=args.hidden_size,
        max_episode_length=max_T,
        n_transformer_heads=args.num_heads,
        device=device,
        target="diffusion_policy",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)

    trainer = DiffusionExpertTrainer(
        model=model,
        optimizer=optimizer,
        states_array=states,
        actions_array=actions,
        device=device,
        num_train_diffusion_timesteps=args.num_diffusion_steps,
        max_trajectory_length=max_T,
    )

    losses = trainer.train(
        num_training_steps=args.train_steps,
        batch_size=args.batch_size,
        print_every=1000,
        save_path=args.save_path,
    )
    print(f"Finished training diffusion expert. Final loss={losses[-1]:.6f}")


if __name__ == "__main__":
    main()
