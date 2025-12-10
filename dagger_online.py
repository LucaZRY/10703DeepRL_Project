import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from dataclasses import dataclass
from diffusers import DDPMScheduler

# --- Import your Diffusion Model Architecture ---
# This assumes you have a folder 'src' with 'models.py' inside
try:
    from src.models import PolicyDiffusionTransformer
except ImportError:
    raise ImportError("Could not import PolicyDiffusionTransformer. Ensure 'src/models.py' exists.")

# --------------------------------------------------
# Configuration
# --------------------------------------------------
@dataclass
class DAggerConfig:
    # Training Parameters
    num_iterations: int = 5            # Number of DAgger rounds
    dagger_episodes_per_iter: int = 3  # Episodes to run per round (Online Collection)
    batch_size: int = 64
    bc_epochs_init: int = 10           # Epochs for initial warm-start
    bc_epochs_per_iter: int = 5        # Epochs for retraining after each round
    
    # Data Parameters
    seed_max_samples: int = 50000      # How many offline samples to keep in buffer
    eval_episodes: int = 3
    
    # Paths
    ppo_npz_path: str = "carracing_dqn_dataset.npz"
    diffusion_data_dir: str = "data/expert_carracing" # Optional extra offline data
    diffusion_model_path="results/diffusion_expert/carracing_expert_96.pt"
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Environment & Preprocessing
# --------------------------------------------------
def make_env(render_mode=None):
    env = gym.make("CarRacing-v2", continuous=True, render_mode=render_mode)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, 96)
    env = FrameStack(env, num_stack=4)
    return env

def preprocess_obs(obs):
    """
    Standardize observation to (4, 96, 96) float32 in [0, 1].
    Handles (4, 96, 96, 1) coming from FrameStack + GrayScale.
    """
    arr = np.array(obs)
    
    # 1. Remove trailing singleton dimensions (e.g., (4, 96, 96, 1) -> (4, 96, 96))
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
        
    # 2. Handle (H, W, C) -> (C, H, W)
    if arr.ndim == 3:
        # If shape is (96, 96, 4), transpose it
        if arr.shape[-1] == 4: 
            arr = np.transpose(arr, (2, 0, 1))
        # If shape is (4, 96, 96), leave it alone
        elif arr.shape[0] == 4: 
            pass
            
    # 3. Handle single grayscale frame (96, 96) -> tile to (4, 96, 96)
    elif arr.ndim == 2: # (96, 96)
        arr = np.tile(arr[None, ...], (4, 1, 1))
        
    return arr.astype(np.float32) / 255.0

# --------------------------------------------------
# Student Policy (CNN)
# --------------------------------------------------
class CNNPolicy(nn.Module):
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

# --------------------------------------------------
# Online Diffusion Expert (The Teacher)
# --------------------------------------------------
class DiffusionExpert:
    def __init__(self, checkpoint_path, device="cuda"):
        print(f"[Expert] Loading Diffusion Model from {checkpoint_path}...")
        self.device = device
        
        # 1. Load Checkpoint FIRST to check dimensions
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # 2. Dynamically determine the max_episode_length from the weights
        # The embedding weight shape is (max_len, hidden_size)
        emb_weight = ckpt["model_state_dict"]["episode_timestep_embedding.weight"]
        trained_max_len = emb_weight.shape[0]
        trained_hidden_size = emb_weight.shape[1]
        
        print(f"[Expert] Detected max_episode_length={trained_max_len}, hidden_size={trained_hidden_size}")
        
        # 3. Load Stats
        self.states_mean = torch.as_tensor(ckpt["states_mean"], device=device).float()
        self.states_std = torch.as_tensor(ckpt["states_std"], device=device).float()
        self.actions_mean = torch.as_tensor(ckpt["actions_mean"], device=device).float()
        self.actions_std = torch.as_tensor(ckpt["actions_std"], device=device).float()
        
        # 4. Initialize Model with the CORRECT detected length
        self.state_dim = 4 * 96 * 96
        self.model = PolicyDiffusionTransformer(
            num_transformer_layers=6,
            state_dim=self.state_dim,
            act_dim=3,
            hidden_size=trained_hidden_size, # Use detected hidden size
            max_episode_length=trained_max_len, # <--- FIXED: Uses 317 (or whatever is in ckpt)
            n_transformer_heads=1,
            device=device,
            target="diffusion_policy"
        )
        
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        
        # Initialize Scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=30,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small_log",
            clip_sample_range=1.0,
        )
        self.scheduler.set_timesteps(30, device=device)

    def get_action(self, obs_np):
        """
        Queries the diffusion model for an action given a single observation.
        obs_np: (4, 96, 96) float32 [0,1]
        """
        # 1. Flatten & Normalize State
        obs_flat = torch.tensor(obs_np.flatten(), device=self.device, dtype=torch.float32)
        obs_norm = (obs_flat - self.states_mean) / self.states_std
        
        # 2. Prepare Inputs (Batch=1, Seq=1)
        # We treat the current state as a sequence of length 1
        previous_states = obs_norm.view(1, 1, -1) 
        
        # Dummy previous action (zeros)
        previous_actions = torch.zeros((1, 1, 3), device=self.device) 
        
        # --- FIXED LINE BELOW ---
        # Episode timestep must be (Batch, Seq) -> Shape (1, 1)
        # The previous error happened because it was Shape (1,)
        episode_timesteps = torch.tensor([[0]], device=self.device, dtype=torch.long)
        
        # 3. Reverse Diffusion Process
        batch_size = 1
        x_t = torch.randn(batch_size, 1, 3, device=self.device) 
        
        with torch.no_grad():
            for t in self.scheduler.timesteps:
                noise_timesteps = torch.full((batch_size, 1), t.item(), device=self.device, dtype=torch.long)
                
                noise_pred = self.model(
                    previous_states=previous_states,
                    previous_actions=previous_actions,
                    noisy_actions=x_t,
                    episode_timesteps=episode_timesteps,
                    noise_timesteps=noise_timesteps
                )
                
                x_t = self.scheduler.step(noise_pred, t, x_t).prev_sample

        # 4. Denormalize Action
        pred_action_norm = x_t[0, 0]
        pred_action = pred_action_norm * self.actions_std + self.actions_mean
        
        # Clip to valid range
        pred_action = torch.clamp(pred_action, torch.tensor([-1.0, 0.0, 0.0], device=self.device), 
                                               torch.tensor([1.0, 1.0, 1.0], device=self.device))
        
        return pred_action.cpu().numpy()

# --------------------------------------------------
# Dataset & Buffer Management
# --------------------------------------------------
class ImitationDataset:
    def __init__(self):
        self.obs = []
        self.acts = []

    def add(self, obs, act):
        self.obs.append(obs)
        self.acts.append(act)

    def __len__(self):
        return len(self.obs)

    def sample_batch(self, batch_size, device):
        idxs = np.random.randint(0, len(self.obs), size=batch_size)
        obs_batch = np.stack([self.obs[i] for i in idxs])
        act_batch = np.stack([self.acts[i] for i in idxs])
        return (torch.tensor(obs_batch, dtype=torch.float32, device=device),
                torch.tensor(act_batch, dtype=torch.float32, device=device))

def load_fixed_offline_buffer(dataset, cfg):
    """
    Loads local files (NPZ or NPY) into the ImitationDataset to serve as the fixed buffer.
    """
    print("[Buffer] Loading fixed offline data...")
    count = 0
    
    # 1. Try Loading NPZ (Best source)
    if os.path.exists(cfg.ppo_npz_path):
        print(f"[Buffer] Found {cfg.ppo_npz_path}")
        data = np.load(cfg.ppo_npz_path)
        obs_data = data['obs'] # (N, 4, 96, 96)
        act_data = data['actions']
        
        # Add a subset to dataset
        limit = min(len(obs_data), cfg.seed_max_samples)
        for i in range(limit):
            dataset.add(obs_data[i], act_data[i])
            count += 1
            
    # 2. Try Loading NPY (Diffusion source)
    elif os.path.exists(cfg.diffusion_data_dir):
        s_path = os.path.join(cfg.diffusion_data_dir, "states.npy")
        a_path = os.path.join(cfg.diffusion_data_dir, "actions.npy")
        if os.path.exists(s_path):
            print(f"[Buffer] Found NPY files in {cfg.diffusion_data_dir}")
            states = np.load(s_path) # Might be (N, T, Dim) or (N, Dim)
            actions = np.load(a_path)
            
            # Flatten if trajectory format
            if states.ndim == 3:
                states = states.reshape(-1, states.shape[-1])
                actions = actions.reshape(-1, actions.shape[-1])
                
            # Filter zeros/padding
            valid = ~np.all(states == 0, axis=1)
            states = states[valid]
            actions = actions[valid]
            
            limit = min(len(states), cfg.seed_max_samples - count)
            for i in range(limit):
                # Reshape Flat State -> Image
                img = states[i].reshape(4, 96, 96)
                dataset.add(img, actions[i])
                count += 1
                
    print(f"[Buffer] Total initial samples in buffer: {count}")

# --------------------------------------------------
# Training Loops
# --------------------------------------------------
def train_bc_epoch(student, dataset, optimizer, batch_size, device):
    student.train()
    if len(dataset) < batch_size: return 0.0
    
    loss_fn = nn.MSELoss()
    total_loss = 0
    steps = len(dataset) // batch_size
    
    for _ in range(steps):
        obs, act = dataset.sample_batch(batch_size, device)
        pred = student(obs)
        loss = loss_fn(pred, act)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / steps

def collect_online_data(env, student, expert, dataset, episodes, device):
    """
    DAgger Rollout: Student acts, Diffusion Expert labels.
    """
    student.eval()
    print(f"[Rollout] collecting {episodes} episodes...")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            obs_proc = preprocess_obs(obs) # (4, 96, 96)
            
            # 1. Student Action (for environment interaction)
            obs_t = torch.tensor(obs_proc, device=device).unsqueeze(0)
            with torch.no_grad():
                student_action = student(obs_t).cpu().numpy()[0]
                
            # 2. Expert Label (Online Diffusion Query)
            # We query the expert on the *same* observation
            expert_action = expert.get_action(obs_proc)
            
            # 3. Add to Dataset (Aggregation)
            dataset.add(obs_proc, expert_action)
            
            # 4. Step Environment
            obs, _, term, trunc, _ = env.step(student_action)
            done = term or trunc
            steps += 1
            
        print(f"  Ep {ep+1}: {steps} steps collected.")

def evaluate(env, student, episodes, device):
    student.eval()
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ret = 0
        while not done:
            obs_proc = preprocess_obs(obs)
            obs_t = torch.tensor(obs_proc, device=device).unsqueeze(0)
            with torch.no_grad():
                action = student(obs_t).cpu().numpy()[0]
            obs, r, term, trunc, _ = env.step(action)
            ret += r
            done = term or trunc
        returns.append(ret)
    return np.mean(returns)

# --------------------------------------------------
# Main
# --------------------------------------------------
def train_dagger(cfg: DAggerConfig):
    print(f"--- Starting DAgger (Online Diffusion) on {cfg.device} ---")
    
    # 1. Setup
    env = make_env()
    student = CNNPolicy().to(cfg.device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    dataset = ImitationDataset()
    
    # 2. Load Fixed Offline Buffer
    load_fixed_offline_buffer(dataset, cfg)
    
    # 3. Load Online Expert
    if not os.path.exists(cfg.diffusion_model_path):
        print(f"ERROR: Expert model not found at {cfg.diffusion_model_path}")
        return
        
    expert = DiffusionExpert(cfg.diffusion_model_path, device=cfg.device)
    
    # 4. Warm Start (Train on Buffer)
    print("\n[Warm Start] Training on fixed buffer...")
    for i in range(cfg.bc_epochs_init):
        loss = train_bc_epoch(student, dataset, optimizer, cfg.batch_size, cfg.device)
        if (i+1) % 5 == 0:
            print(f"  Epoch {i+1}: Loss {loss:.4f}")
            
    score = evaluate(env, student, cfg.eval_episodes, cfg.device)
    print(f"[Warm Start] Eval Score: {score:.2f}")
    
    # 5. DAgger Loop
    for it in range(cfg.num_iterations):
        print(f"\n=== Iteration {it+1}/{cfg.num_iterations} ===")
        
        # A. Collect (Online)
        collect_online_data(env, student, expert, dataset, cfg.dagger_episodes_per_iter, cfg.device)
        print(f"  Dataset Size: {len(dataset)}")
        
        # B. Train (Behavior Cloning)
        for i in range(cfg.bc_epochs_per_iter):
            loss = train_bc_epoch(student, dataset, optimizer, cfg.batch_size, cfg.device)
        print(f"  Train Loss: {loss:.4f}")
        
        # C. Evaluate
        score = evaluate(env, student, cfg.eval_episodes, cfg.device)
        print(f"  Eval Score: {score:.2f}")
        
    # Save
    torch.save(student.state_dict(), "student_dagger_diffusion.pt")
    print("\nSaved student_dagger_diffusion.pt")
    env.close()

if __name__ == "__main__":
    config = DAggerConfig()
    train_dagger(config)