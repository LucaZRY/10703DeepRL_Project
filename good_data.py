import gymnasium as gym
import numpy as np
import os
import torch
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub

# CONFIG
REPO_ID = "igpaub/ppo-CarRacing-v2"
FILENAME = "ppo-CarRacing-v2.zip"
OUTPUT_DIR = "data/human_expert"
OUTPUT_FILE = "expert_trajectories.npz"
NUM_EPISODES = 200
MAX_STEPS = 1000 # Force a consistent max length for stacking

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def generate_data():
    print(f"Downloading perfect PPO agent from {REPO_ID}...")
    try:
        checkpoint = load_from_hub(REPO_ID, FILENAME)
        model = PPO.load(checkpoint, device="cpu")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
    
    # Store lists of episodes
    all_states = []
    all_actions = []
    
    print(f"Generating {NUM_EPISODES} episodes...")
    
    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        score = 0
        steps = 0
        
        # Buffers for this episode
        ep_states = []
        ep_actions = []
        
        stack = [np.zeros((96, 96), dtype=np.float32) for _ in range(4)]
        
        while not done and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            
            # Process Frame
            frame = obs
            if np.max(frame) > 1.0: frame = frame.astype(np.float32) / 255.0
            gray = rgb2gray(frame)
            if gray.ndim == 3: gray = gray.squeeze()
            
            stack.pop(0)
            stack.append(gray)
            
            state_stack = np.array(stack, dtype=np.float32)
            state_flat = state_stack.flatten() 
            
            ep_states.append(state_flat)
            ep_actions.append(action)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            steps += 1

        print(f"  Episode {ep+1}: {steps} steps, Score: {score:.1f}")

        # PAD OR TRUNCATE TO MAX_STEPS
        # This ensures we can stack them into a (N, T, Dim) array
        ep_states = np.array(ep_states)
        ep_actions = np.array(ep_actions)
        
        curr_len = len(ep_states)
        if curr_len < MAX_STEPS:
            # Pad with zeros
            pad_len = MAX_STEPS - curr_len
            state_pad = np.zeros((pad_len, 36864), dtype=np.float32)
            act_pad = np.zeros((pad_len, 3), dtype=np.float32)
            
            ep_states = np.concatenate([ep_states, state_pad], axis=0)
            ep_actions = np.concatenate([ep_actions, act_pad], axis=0)
        
        all_states.append(ep_states)
        all_actions.append(ep_actions)

    # Convert to (N, T, Dim)
    states_out = np.stack(all_states).astype(np.float32) # (20, 1000, 36864)
    actions_out = np.stack(all_actions).astype(np.float32) # (20, 1000, 3)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    print(f"Saving data shape {states_out.shape} to {out_path}...")
    np.savez(out_path, states=states_out, actions=actions_out)
    print("Done! Data is now properly structured as separate episodes.")

if __name__ == "__main__":
    generate_data()