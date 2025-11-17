import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def select_device() -> torch.device:
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class StudentMLP(nn.Module):
    """
    Simple MLP used for both baseline and diffusion-distilled students.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        input_dim = state_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, act_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_dataset(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load states/actions stored as .npy files.
    """
    states_path = os.path.join(dataset_dir, "states.npy")
    actions_path = os.path.join(dataset_dir, "actions.npy")
    if not os.path.exists(states_path) or not os.path.exists(actions_path):
        raise FileNotFoundError(
            f"Expected states/actions .npy files in {dataset_dir}, "
            "found none. Did you run train_diffusion/generate_synthetic_dataset?"
        )

    states = np.load(states_path).astype(np.float32)
    actions = np.load(actions_path).astype(np.float32)
    if states.shape[:2] != actions.shape[:2]:
        raise ValueError(
            f"State/action leading dims must match, got {states.shape} vs {actions.shape}"
        )
    return states, actions


def build_dataloader(
    states: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(states.reshape(-1, states.shape[-1])).float(),
        torch.from_numpy(actions.reshape(-1, actions.shape[-1])).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_student_on_arrays(
    states: np.ndarray,
    actions: np.ndarray,
    hidden_dim: int = 256,
    num_hidden_layers: int = 2,
    num_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> Tuple[StudentMLP, List[float]]:
    """
    Train a student policy using MSE bc loss on provided arrays.
    """
    if device is None:
        device = select_device()

    state_dim = states.shape[-1]
    act_dim = actions.shape[-1]

    dataloader = build_dataloader(states, actions, batch_size=batch_size, shuffle=True)
    model = StudentMLP(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    losses: List[float] = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            pred_actions = model(batch_states)
            loss = criterion(pred_actions, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(len(dataloader), 1)
        losses.append(epoch_loss)
        if (epoch + 1) % max(num_epochs // 5, 1) == 0:
            print(f"[student] epoch {epoch+1}/{num_epochs}, loss={epoch_loss:.6f}")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)

    return model, losses


def train_student_from_dir(
    data_dir: str,
    hidden_dim: int = 256,
    num_hidden_layers: int = 2,
    num_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    save_path: Optional[str] = None,
) -> Tuple[StudentMLP, List[float]]:
    states, actions = load_dataset(data_dir)
    return train_student_on_arrays(
        states=states,
        actions=actions,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        save_path=save_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train student policies.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "offline_distill", "both"],
        default="both",
    )
    parser.add_argument("--baseline_data_dir", type=str, default="data/expert")
    parser.add_argument("--offline_data_dir", type=str, default="data/generated")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.mode in ("baseline", "both"):
        baseline_path = os.path.join(
            args.results_dir, "student_baseline", "student_baseline_model.pt"
        )
        print("Training baseline student (expert data)...")
        train_student_from_dir(
            data_dir=args.baseline_data_dir,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_path=baseline_path,
        )

    if args.mode in ("offline_distill", "both"):
        offline_path = os.path.join(
            args.results_dir, "student_offline_distill", "student_offline_model.pt"
        )
        print("Training offline-distilled student (synthetic data)...")
        train_student_from_dir(
            data_dir=args.offline_data_dir,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_path=offline_path,
        )


if __name__ == "__main__":
    main()

