import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

# --- 1. ARCHITECTURES ---

class StudentMLP(nn.Module):
    """
    Simple MLP for vector-based environments (BipedalWalker, etc.).
    NOT recommended for CarRacing images.
    """
    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int = 256, num_hidden_layers: int = 2):
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

class CNNPolicy(nn.Module):
    """
    CNN Policy specifically for CarRacing (4x96x96 input).
    """
    def __init__(self, obs_channels=4, act_dim=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc_body = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(256, act_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc_body(x)
        raw = self.fc_out(x)
        # CarRacing Actions: Steer[-1,1], Gas[0,1], Brake[0,1]
        steer = torch.tanh(raw[:, 0:1])
        gas   = torch.sigmoid(raw[:, 1:2])
        brake = torch.sigmoid(raw[:, 2:3])
        return torch.cat([steer, gas, brake], dim=1)


# --- 2. UTILS & ENV SETUP ---

def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def make_env(render_mode=None):
    env = gym.make("CarRacing-v2", continuous=True, render_mode=render_mode)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, 96)
    env = FrameStack(env, num_stack=4)
    return env

def preprocess_obs(obs):
    """Standardize observation to (4, 96, 96) float32 in [0, 1]."""
    arr = np.array(obs)
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.ndim == 3:
        if arr.shape[-1] == 4: arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 2:
        arr = np.tile(arr[None, ...], (4, 1, 1))
    return arr.astype(np.float32) / 255.0

def load_dataset(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    states_path = os.path.join(dataset_dir, "states.npy")
    actions_path = os.path.join(dataset_dir, "actions.npy")
    if not os.path.exists(states_path):
        # Fallback to .npz if .npy doesn't exist (support your format)
        npz_path = os.path.join(dataset_dir, "expert_trajectories.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            return data["states"].astype(np.float32), data["actions"].astype(np.float32)
        raise FileNotFoundError(f"No data found in {dataset_dir}")
    return np.load(states_path).astype(np.float32), np.load(actions_path).astype(np.float32)


# --- 3. TRAINING LOOPS ---

def build_dataloader(states: np.ndarray, actions: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    # If using CNN, input is likely (N, 36864) flattened, need to reshape to (N, 4, 96, 96) inside model or here
    # For simplicity, we assume the CNNPolicy expects (B, 4, 96, 96)
    # If dataset is flattened, we reshape it on the fly or in the dataset
    
    # Check if this is image data (flattened size 36864)
    if states.shape[-1] == 36864: 
        # Reshape to (N, 4, 96, 96) for CNN
        states_tensor = torch.from_numpy(states).float().view(-1, 4, 96, 96)
    else:
        states_tensor = torch.from_numpy(states).float()
        
    dataset = TensorDataset(states_tensor, torch.from_numpy(actions).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_student_bc(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int,
    device: torch.device,
    lr: float = 1e-4,
    print_prefix: str = "[BC]"
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        steps = 0
        for batch_states, batch_actions in dataloader:
            batch_states, batch_actions = batch_states.to(device), batch_actions.to(device)
            
            pred_actions = model(batch_states)
            loss = criterion(pred_actions, batch_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            
        avg_loss = epoch_loss / max(steps, 1)
        losses.append(avg_loss)
        if (epoch + 1) % max(num_epochs // 5, 1) == 0:
            print(f"{print_prefix} Epoch {epoch+1}/{num_epochs}, Loss={avg_loss:.6f}")
            
    return losses


# --- 4. DAGGER COMPONENTS ---

class ImitationDataset:
    def __init__(self):
        self.obs = []
        self.acts = []
    def add(self, obs, act):
        self.obs.append(obs)
        self.acts.append(act)
    def __len__(self): return len(self.obs)
    def sample_batch(self, batch_size, device):
        idxs = np.random.randint(0, len(self.obs), size=batch_size)
        obs = np.stack([self.obs[i] for i in idxs])
        act = np.stack([self.acts[i] for i in idxs])
        return torch.tensor(obs, dtype=torch.float32, device=device), torch.tensor(act, dtype=torch.float32, device=device)

def run_dagger(
    student: nn.Module,
    expert_model_path: str,
    offline_data_path: str,
    num_iterations: int = 5,
    episodes_per_iter: int = 3,
    device: torch.device = torch.device("cpu")
):
    print(f"--- Starting DAgger ---")
    
    # 1. Load Expert (Diffusion)
    # We import locally to avoid dependency errors if not present
    from dagger_online import DiffusionExpert # Assumes dagger_online.py is in same folder
    
    if not os.path.exists(expert_model_path):
        print(f"Error: Expert model not found at {expert_model_path}")
        return

    expert = DiffusionExpert(expert_model_path, device=str(device))
    dataset = ImitationDataset()
    env = make_env()

    # 2. Warm Start (Load Offline Data)
    # Reuse your existing loading logic here, adapted for the dataset class
    from dagger_online import load_fixed_offline_buffer, DAggerConfig
    cfg = DAggerConfig(ppo_npz_path=offline_data_path, seed_max_samples=50000)
    load_fixed_offline_buffer(dataset, cfg)
    
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # Initial BC
    print("[DAgger] Warm Start BC...")
    for _ in range(5): # 5 epochs
        # Simple training loop for ImitationDataset
        student.train()
        steps = len(dataset) // 64
        for _ in range(steps):
            obs, act = dataset.sample_batch(64, device)
            optimizer.zero_grad()
            loss_fn(student(obs), act).backward()
            optimizer.step()

    # 3. DAgger Loop
    for it in range(num_iterations):
        print(f"\n=== DAgger Iteration {it+1}/{num_iterations} ===")
        
        # Collect
        student.eval()
        for ep in range(episodes_per_iter):
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done:
                obs_proc = preprocess_obs(obs)
                obs_t = torch.tensor(obs_proc, device=device).unsqueeze(0)
                
                # Student Act
                with torch.no_grad():
                    student_act = student(obs_t).cpu().numpy()[0]
                
                # Expert Query (Pass 'steps'!)
                expert_act = expert.get_action(obs_proc, t=steps)
                
                dataset.add(obs_proc, expert_act)
                obs, _, term, trunc, _ = env.step(student_act)
                done = term or trunc
                steps += 1
            print(f"  Ep {ep+1}: Collected {steps} steps")

        # Train
        student.train()
        print(f"  Training on {len(dataset)} samples...")
        for epoch in range(5):
            avg_loss = 0
            steps = max(1, len(dataset) // 64)
            for _ in range(steps):
                obs, act = dataset.sample_batch(64, device)
                optimizer.zero_grad()
                loss = loss_fn(student(obs), act)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            print(f"  Epoch {epoch+1} Loss: {avg_loss/steps:.4f}")
            
    env.close()
    return student


# --- 5. MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Train student policies.")
    parser.add_argument("--mode", type=str, choices=["baseline", "offline_distill", "dagger"], default="dagger")
    parser.add_argument("--data_dir", type=str, default="data/human_expert", help="Folder containing .npz data")
    parser.add_argument("--expert_ckpt", type=str, default="results/diffusion_expert/human_expert_96.pt")
    parser.add_argument("--save_path", type=str, default="results/student_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # 1. Initialize CNN Policy (CarRacing)
    # We ignore MLP args because we know this is for CarRacing
    student = CNNPolicy().to(device)
    print(f"Initialized CNN Student on {device}")

    # 2. Select Mode
    if args.mode == "dagger":
        # Online DAgger Training
        student = run_dagger(
            student=student,
            expert_model_path=args.expert_ckpt,
            offline_data_path=os.path.join(args.data_dir, "expert_trajectories.npz"),
            device=device
        )
        torch.save(student.state_dict(), args.save_path)
        print(f"Saved DAgger student to {args.save_path}")

    else:
        # Offline BC Training (Baseline or Distill)
        print(f"Loading offline data from {args.data_dir}...")
        try:
            states, actions = load_dataset(args.data_dir)
            dataloader = build_dataloader(states, actions, batch_size=256)
            
            print(f"Training Offline BC ({args.mode})...")
            train_student_bc(student, dataloader, num_epochs=50, device=device)
            
            torch.save(student.state_dict(), args.save_path)
            print(f"Saved offline student to {args.save_path}")
            
        except Exception as e:
            print(f"Failed to run offline training: {e}")

if __name__ == "__main__":
    main()