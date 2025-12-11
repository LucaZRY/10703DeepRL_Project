import torch
import numpy as np
import gymnasium as gym
from dagger_online import make_env, preprocess_obs, DiffusionExpert

# --- CONFIGURATION ---
MODEL_PATH = "results/diffusion_expert/perfect_expert_96.pt" # Your human expert
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RENDER_MODE = "human" # Watch the car drive

def run_expert_test():
    print(f"--- Testing Diffusion Expert on {DEVICE} ---")
    
    # 1. Load Environment
    env = make_env(render_mode=RENDER_MODE)
    
    # 2. Load Expert
    try:
        expert = DiffusionExpert(MODEL_PATH, device=DEVICE)
    except Exception as e:
        print(f"FAILED to load expert: {e}")
        return

    # 3. Run Evaluation Loop
    total_reward = 0
    obs, _ = env.reset()
    done = False
    step = 0
    
    print("Starting Drive...")
    while not done:
        obs_proc = preprocess_obs(obs)
        
        # --- FIX: PASS THE TIMESTEP 't' ---
        action = expert.get_action(obs_proc, t=step) 
        # ----------------------------------
        
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step}, Cumulative Reward: {total_reward:.2f}")

    print(f"\n--- FINAL RESULT ---")
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    run_expert_test()