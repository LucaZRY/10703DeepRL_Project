import sys
import os
import torch
import imageio

# Add parent directory to path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the correct Expert class and utilities from dagger_online
from dagger_online import DiffusionExpert, make_env, preprocess_obs

def record_video(policy, save_path="expert_run.mp4", max_steps=1500, device="cpu"):
    # Use the SAME env setup as training
    env = make_env(render_mode="rgb_array")
    obs, info = env.reset()

    frames = []
    total_reward = 0.0

    print(f"Recording video to {save_path}...")

    for step in range(max_steps):
        # 1. Preprocess (Standardize input to 0-1 float)
        obs_proc = preprocess_obs(obs)  # -> (4, 96, 96)
        
        # 2. Get Action
        # Pass the timestep 't' because the Diffusion model needs it
        action = policy.get_action(obs_proc, t=step)

        # 3. Step Env
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 4. Capture Frame
        frame = env.render()
        frames.append(frame)

        if terminated or truncated:
            print(f"Episode finished at step {step}")
            break

    env.close()
    
    # FIX: macro_block_size=1 prevents resizing artifacts
    # FIX: pixelformat='yuv420p' ensures compatibility with all video players
    imageio.mimwrite(save_path, frames, fps=30, macro_block_size=1, pixelformat='yuv420p')
    
    print(f"[Saved video] {save_path}, Total Return = {total_reward:.2f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- LOAD MODEL CORRECTLY ---
    # Try to find the file in the parent folder structure
    expert_path = os.path.join(parent_dir, "results/diffusion_expert/perfect_expert_96.pt")
    
    # Fallback if running from a different location
    if not os.path.exists(expert_path):
        expert_path = "results/diffusion_expert/perfect_expert_96.pt"

    if not os.path.exists(expert_path):
        print(f"ERROR: Could not find expert model at {expert_path}")
        exit(1)

    print(f"Loading expert from: {expert_path}")

    # Load Diffusion Expert
    expert = DiffusionExpert(expert_path, device=device)

    # Record video
    record_video(expert, "expert_run.mp4", device=device)