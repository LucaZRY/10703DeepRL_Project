"""
record_human.py
Drive the car yourself to create a high-quality expert dataset.
"""
import gymnasium as gym
import numpy as np
import os
import pygame

# CONFIG
OUTPUT_FILE = "carracing_human_dataset.npz"
NUM_EPISODES = 15  # Drive 5 good laps

def record_human():
    env = gym.make("CarRacing-v2", continuous=True, render_mode="human")
    
    obs_list = []
    action_list = []
    
    print("Controls:")
    print("  Arrow Keys: Steer / Gas / Brake")
    print("  ESC: Quit")
    print(f"Recording {NUM_EPISODES} episodes...")

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        step = 0
        score = 0
        
        # Buffer for current episode
        ep_obs = []
        ep_acts = []

        while not done:
            # 1. Capture Human Action from Keyboard
            # CarRacing-v2 internal logic handles keyboard -> action mapping
            # but we need to grab it. 
            # Note: We must construct the action manually to save it.
            
            keys = pygame.key.get_pressed()
            action = np.array([0.0, 0.0, 0.0])

            if keys[pygame.K_LEFT]:
                action[0] = -1.0
            elif keys[pygame.K_RIGHT]:
                action[0] = +1.0
            
            if keys[pygame.K_UP]:
                action[1] = +1.0
            
            if keys[pygame.K_DOWN]:
                action[2] = +0.8 # Brake
            
            # 2. Step Environment
            # Important: We pass the explicit action so the env matches our data
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            step += 1
            
            # 3. Save Data (Only save if you want to keep this frame)
            # You might want to skip the "Zoom in" start frames (first 50 steps)
            if step > 50:
                # Resize/Grayscale is handled in dagger loader, 
                # but we should save raw (96,96,3) or (96,96) here.
                # To match your pipeline, let's save the Raw RGB 
                # and let the loader handle preprocessing.
                
                # Standardize action for saving
                ep_obs.append(obs)
                ep_acts.append(action)

        print(f"Episode {ep+1} Score: {score:.2f}")
        
        # Only save if the run was good (e.g. positive score)
        if score > 500:
            print("  -> Saved (Good Run)")
            obs_list.extend(ep_obs)
            action_list.extend(ep_acts)
        else:
            print("  -> Discarded (Bad Run - Try to stay on track!)")

    env.close()
    
    # Save to NPZ
    print(f"Saving {len(obs_list)} samples to {OUTPUT_FILE}...")
    np.savez_compressed(OUTPUT_FILE, obs=np.array(obs_list), actions=np.array(action_list))
    print("Done!")

if __name__ == "__main__":
    record_human()