import os
import numpy as np
import torch

from train_diffusion import DiffusionExpertTrainer, load_expert_data
from src.models import PolicyDiffusionTransformer  # same as in train_diffusion.py

def main():
    expert_dir = "data/expert_carracing"
    ckpt_path  = "results/diffusion_expert/carracing_expert_96.pt"  
    out_dir    = "data/generated_carracing"

    os.makedirs(out_dir, exist_ok=True)

    # 1) Load expert trajectories
    states, actions = load_expert_data(expert_dir)
    num_traj, max_T, state_dim = states.shape
    act_dim = actions.shape[-1]

    # 2) Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # 3) Rebuild model
    model = PolicyDiffusionTransformer(
        num_transformer_layers=6,
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=128,
        max_episode_length=max_T,
        n_transformer_heads=1,
        device=device,
        target="diffusion_policy",
    )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 4) Build trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
    trainer = DiffusionExpertTrainer(
        model=model,
        optimizer=optimizer,
        states_array=states,
        actions_array=actions,
        device=device,
        num_train_diffusion_timesteps=30,
        max_trajectory_length=max_T,
    )

    # 5) Sample synthetic data
    num_samples = 200_000
    synthetic_states, synthetic_actions = trainer.generate_synthetic_dataset(
        num_samples=num_samples,
        batch_size=256,
        max_action_len=1,
    )

    # 6) Save dataset
    np.save(os.path.join(out_dir, "states.npy"),  synthetic_states)
    np.save(os.path.join(out_dir, "actions.npy"), synthetic_actions)
    print(f"Saved synthetic dataset to {out_dir}")
    print("synthetic_states:", synthetic_states.shape)
    print("synthetic_actions:", synthetic_actions.shape)

if __name__ == "__main__":
    main()
