# tests/test_diffusion_expert.py
import numpy as np
import torch

from src.models import PolicyDiffusionTransformer
from train_diffusion import DiffusionExpertTrainer


def make_dummy_data(num_traj=4, T=10, state_dim=6, act_dim=2):
    states = np.random.randn(num_traj, T, state_dim).astype(np.float32)
    actions = np.random.randn(num_traj, T, act_dim).astype(np.float32)

    # add some padding (zeros) at the end of each trajectory
    for i in range(num_traj):
        cut = np.random.randint(T // 2, T)
        states[i, cut:] = 0.0
        actions[i, cut:] = 0.0

    return states, actions


def test_diffusion_forward_shapes():
    states, actions = make_dummy_data()
    num_traj, T, state_dim = states.shape
    act_dim = actions.shape[-1]

    device = torch.device("cpu")
    model = PolicyDiffusionTransformer(
        num_transformer_layers=1,
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=32,
        max_episode_length=T,
        n_transformer_heads=1,
        device=device,
        target="diffusion_policy",
    )

    # construct a single batch via the trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = DiffusionExpertTrainer(
        model=model,
        optimizer=optimizer,
        states_array=states,
        actions_array=actions,
        device=device,
        num_train_diffusion_timesteps=5,
        max_trajectory_length=T,
    )

    batch = trainer.get_training_batch(batch_size=2)
    (
        prev_states_batch,
        prev_actions_batch,
        actions_batch,
        episode_ts_batch,
        prev_states_mask_batch,
        prev_actions_mask_batch,
        actions_padding_batch,
    ) = batch

    # build a forward pass manually to check shapes
    noise = torch.randn_like(actions_batch)
    t = torch.randint(
        low=0,
        high=5,
        size=(actions_batch.shape[0],),
        device=device,
        dtype=torch.long,
    )
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=5,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample_range=1.0,
    )
    noisy_actions = scheduler.add_noise(actions_batch, noise, t)
    noise_timesteps = t.unsqueeze(1)

    with torch.no_grad():
        noise_pred = model(
            previous_states=prev_states_batch,
            previous_actions=prev_actions_batch,
            noisy_actions=noisy_actions,
            episode_timesteps=episode_ts_batch,
            noise_timesteps=noise_timesteps,
            previous_states_mask=prev_states_mask_batch,
            previous_actions_mask=prev_actions_mask_batch,
            actions_padding_mask=actions_padding_batch,
        )

    assert noise_pred.shape == actions_batch.shape


def test_training_step_runs_and_returns_float():
    states, actions = make_dummy_data()
    num_traj, T, state_dim = states.shape
    act_dim = actions.shape[-1]

    device = torch.device("cpu")
    model = PolicyDiffusionTransformer(
        num_transformer_layers=1,
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=32,
        max_episode_length=T,
        n_transformer_heads=1,
        device=device,
        target="diffusion_policy",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = DiffusionExpertTrainer(
        model=model,
        optimizer=optimizer,
        states_array=states,
        actions_array=actions,
        device=device,
        num_train_diffusion_timesteps=5,
        max_trajectory_length=T,
    )

    loss = trainer.training_step(batch_size=2)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert not np.isnan(loss)


def test_diffusion_sampling_shapes_and_numerics():
    states, actions = make_dummy_data()
    num_traj, T, state_dim = states.shape
    act_dim = actions.shape[-1]

    device = torch.device("cpu")
    model = PolicyDiffusionTransformer(
        num_transformer_layers=1,
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=32,
        max_episode_length=T,
        n_transformer_heads=1,
        device=device,
        target="diffusion_policy",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = DiffusionExpertTrainer(
        model=model,
        optimizer=optimizer,
        states_array=states,
        actions_array=actions,
        device=device,
        num_train_diffusion_timesteps=5,
        max_trajectory_length=T,
    )

    (
        prev_states_batch,
        prev_actions_batch,
        actions_batch,
        episode_ts_batch,
        prev_states_mask_batch,
        prev_actions_mask_batch,
        actions_padding_batch,
    ) = trainer.get_training_batch(batch_size=2)

    _, max_action_len, _ = actions_batch.shape

    sampled_actions = trainer.diffusion_sample(
        previous_states=prev_states_batch,
        previous_actions=prev_actions_batch,
        episode_timesteps=episode_ts_batch,
        previous_states_padding_mask=prev_states_mask_batch,
        previous_actions_padding_mask=prev_actions_mask_batch,
        actions_padding_mask=actions_padding_batch,
        max_action_len=max_action_len,
    )

    assert sampled_actions.shape == actions_batch.shape
    assert not torch.isnan(sampled_actions).any()
    clip = trainer.clip_sample_range + 1e-5
    assert sampled_actions.max().item() <= clip
    assert sampled_actions.min().item() >= -clip


def test_generate_synthetic_dataset_shapes():
    states, actions = make_dummy_data()
    num_traj, T, state_dim = states.shape
    act_dim = actions.shape[-1]

    device = torch.device("cpu")
    model = PolicyDiffusionTransformer(
        num_transformer_layers=1,
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=32,
        max_episode_length=T,
        n_transformer_heads=1,
        device=device,
        target="diffusion_policy",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = DiffusionExpertTrainer(
        model=model,
        optimizer=optimizer,
        states_array=states,
        actions_array=actions,
        device=device,
        num_train_diffusion_timesteps=5,
        max_trajectory_length=T,
    )

    num_samples = 10
    synth_states, synth_actions = trainer.generate_synthetic_dataset(
        num_samples=num_samples,
        batch_size=4,
        max_action_len=1,
    )

    assert synth_states.shape == (num_samples, state_dim)
    assert synth_actions.shape == (num_samples, act_dim)
    assert not np.isnan(synth_states).any()
    assert not np.isnan(synth_actions).any()