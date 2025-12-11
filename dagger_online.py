import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from dataclasses import dataclass
from diffusers import DDPMScheduler

# --- Import your Diffusion Model Architecture ---
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
    num_iterations: int = 10           # Increased iterations for better convergence
    dagger_episodes_per_iter: int = 5  # Collect more data per round
    batch_size: int = 64
    bc_epochs_init: int = 5            # Warm start epochs (lower if loading pre-trained)
    bc_epochs_per_iter: int = 5        # Epochs for retraining after each round
    
    # Data Parameters
    seed_max_samples: int = 50000      # How many offline samples to keep
    eval_episodes: int = 3
    
    # --- PATHS (Updated for Phase 4) ---
    # 1. Offline Data for Warm Start (Use Human Data)
    ppo_npz_path: str = "data/human_expert/expert_trajectories.npz"
    
    # 2. Expert Model (Use Human-Cloned Expert)
    diffusion_model_path: str = "results/diffusion_expert/perfect_expert_96.pt"  #
    
    # 3. Pre-Trained Student (Optional Phase 3 Output)
    pretrained_student_path: str = "results/student_pretrain.pt"
    
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
    """
    arr = np.array(obs)
    
    # 1. Remove trailing singleton dimensions
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
        
    # 2. Handle (H, W, C) -> (C, H, W)
    if arr.ndim == 3:
        if arr.shape[-1] == 4: 
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.shape[0] == 4: 
            pass
            
    # 3. Handle single grayscale frame (96, 96) -> tile to (4, 96, 96)
    elif arr.ndim == 2: 
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
        
        # Load Checkpoint (safely handling weights_only for newer pytorch)
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Auto-Detect Dimensions
        emb_weight = ckpt["model_state_dict"]["episode_timestep_embedding.weight"]
        trained_max_len = emb_weight.shape[0]
        trained_hidden_size = emb_weight.shape[1]
        
        print(f"[Expert] Detected max_episode_length={trained_max_len}, hidden_size={trained_hidden_size}")
        
        self.states_mean = torch.as_tensor(ckpt["states_mean"], device=device).float()
        self.states_std = torch.as_tensor(ckpt["states_std"], device=device).float()
        self.actions_mean = torch.as_tensor(ckpt["actions_mean"], device=device).float()
        self.actions_std = torch.as_tensor(ckpt["actions_std"], device=device).float()
        
        self.state_dim = 4 * 96 * 96
        self.model = PolicyDiffusionTransformer(
            num_transformer_layers=6,
            state_dim=self.state_dim,
            act_dim=3,
            hidden_size=trained_hidden_size,
            max_episode_length=trained_max_len,
            n_transformer_heads=1,
            device=device,
            target="diffusion_policy"
        )
        
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        
        self.scheduler = DDPMScheduler(
            num_train_timesteps=30,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small_log",
            clip_sample_range=1.0,
        )
        self.scheduler.set_timesteps(30, device=device)

    def get_action(self, obs_np, t):
        """
        Queries the diffusion model for an action given a single observation and timestep.
        """
        obs_flat = torch.tensor(obs_np.flatten(), device=self.device, dtype=torch.float32)
        obs_norm = (obs_flat - self.states_mean) / self.states_std
        
        previous_states = obs_norm.view(1, 1, -1) 
        previous_actions = torch.zeros((1, 1, 3), device=self.device) 
        
        # Fix: Use actual timestep 't', clamped to model limit
        safe_t = min(t, self.model.max_episode_length - 1)
        episode_timesteps = torch.tensor([[safe_t]], device=self.device, dtype=torch.long)
        
        batch_size = 1
        x_t = torch.randn(batch_size, 1, 3, device=self.device) 
        
        with torch.no_grad():
            for t_diff in self.scheduler.timesteps:
                noise_timesteps = torch.full((batch_size, 1), t_diff.item(), device=self.device, dtype=torch.long)
                
                noise_pred = self.model(
                    previous_states=previous_states,
                    previous_actions=previous_actions,
                    noisy_actions=x_t,
                    episode_timesteps=episode_timesteps,
                    noise_timesteps=noise_timesteps
                )
                
                x_t = self.scheduler.step(noise_pred, t_diff, x_t).prev_sample

        pred_action_norm = x_t[0, 0]
        pred_action = pred_action_norm * self.actions_std + self.actions_mean
        
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

# REPLACE THIS FUNCTION in dagger_online.py

def load_fixed_offline_buffer(dataset, cfg):
    print(f"[Buffer] Loading fixed offline data from {cfg.ppo_npz_path}...")
    count = 0
    
    if os.path.exists(cfg.ppo_npz_path):
        data = np.load(cfg.ppo_npz_path)
        
        # --- 1. HANDLE KEYS & SHAPES ---
        if 'obs' in data:
            obs_data = data['obs']
            act_data = data['actions']
        elif 'states' in data:
            print("[Buffer] Found 'states' key (Flattened). Un-flattening...")
            obs_data = data['states'] 
            act_data = data['actions']
            
            # --- FIX: Handle (Episodes, Time, Dim) -> (Total_Samples, Dim) ---
            if obs_data.ndim == 3:
                # Shape is (N_Episodes, Time, Dim) e.g. (20, 1000, 36864)
                # We need to flatten the first two dims to get a list of samples
                N, T, D = obs_data.shape
                obs_data = obs_data.reshape(N * T, D)
                act_data = act_data.reshape(N * T, act_data.shape[-1])
                
                # Filter out padding (all-zero rows) if any exist
                # This removes the empty steps at the end of episodes if you used padding
                non_zero_mask = np.abs(obs_data).sum(axis=1) > 0.0001
                obs_data = obs_data[non_zero_mask]
                act_data = act_data[non_zero_mask]
                print(f"[Buffer] Flattened {N} episodes. Valid samples: {len(obs_data)}")
            # ------------------------------------------------------------------

            # Reshape Flattened Vectors back to Images (N, 4, 96, 96)
            N = obs_data.shape[0]
            obs_data = obs_data.reshape(N, 4, 96, 96)
        else:
            raise KeyError(f"Dataset must contain 'obs' or 'states'. Found: {list(data.keys())}")

        # Limit samples if requested
        limit = min(len(obs_data), cfg.seed_max_samples)
        
        # --- 2. AUTO-SCALING LOGIC ---
        sample_max = np.max(obs_data[:100])
        scale_factor = 1.0
        
        if sample_max > 1.0:
            print("[Buffer] Detected range 0-255. Dividing by 255.")
            scale_factor = 1.0 / 255.0
        else:
            print("[Buffer] Detected range 0-1. No scaling needed.")
            scale_factor = 1.0

        for i in range(limit):
            raw_obs = obs_data[i]
            obs_norm = raw_obs.astype(np.float32) * scale_factor
            obs_norm = np.clip(obs_norm, 0.0, 1.0)
            dataset.add(obs_norm, act_data[i])
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
    student.eval()
    print(f"[Rollout] collecting {episodes} episodes...")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            obs_proc = preprocess_obs(obs)
            obs_t = torch.tensor(obs_proc, device=device).unsqueeze(0)
            
            with torch.no_grad():
                student_action = student(obs_t).cpu().numpy()[0]
                
            # Query Expert with Correct Timestep
            expert_action = expert.get_action(obs_proc, t=steps)
            
            dataset.add(obs_proc, expert_action)
            
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
    
    env = make_env()
    student = CNNPolicy().to(cfg.device)
    
    # --- Load Pre-Trained Student (Phase 3 Integration) ---
    if os.path.exists(cfg.pretrained_student_path):
        print(f"Loading pre-trained student from {cfg.pretrained_student_path}...")
        try:
            student.load_state_dict(torch.load(cfg.pretrained_student_path, map_location=cfg.device))
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights ({e}). Starting from scratch.")
    else:
        print("No pre-trained student found. Starting from scratch.")
    # ------------------------------------------------------

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    dataset = ImitationDataset()
    
    # 1. Load Fixed Offline Buffer (Warm Start Data)
    load_fixed_offline_buffer(dataset, cfg)
    
    # 2. Load Online Expert
    if not os.path.exists(cfg.diffusion_model_path):
        print(f"ERROR: Expert model not found at {cfg.diffusion_model_path}")
        return
    expert = DiffusionExpert(cfg.diffusion_model_path, device=cfg.device)
    
    # 3. Warm Start BC Training
    if cfg.bc_epochs_init > 0:
        print("\n[Warm Start] Training on fixed buffer...")
        for i in range(cfg.bc_epochs_init):
            loss = train_bc_epoch(student, dataset, optimizer, cfg.batch_size, cfg.device)
            if (i+1) % 5 == 0:
                print(f"  Epoch {i+1}: Loss {loss:.4f}")
    
    score = evaluate(env, student, cfg.eval_episodes, cfg.device)
    print(f"[Warm Start] Eval Score: {score:.2f}")
    
    # 4. DAgger Loop
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
        
    torch.save(student.state_dict(), "student_dagger_final.pt")
    print("\nSaved student_dagger_final.pt")
    env.close()

if __name__ == "__main__":
    config = DAggerConfig()
    train_dagger(config)