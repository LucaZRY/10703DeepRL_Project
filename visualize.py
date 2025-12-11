import gymnasium as gym
import torch
import numpy as np
import os
from gymnasium.wrappers import RecordVideo, GrayScaleObservation, ResizeObservation, FrameStack
from train_student import CNNPolicy, preprocess_obs # Re-using your existing file

# CONFIG
MODEL_PATH = "student_dagger_final.pt"
VIDEO_DIR = "results/videos"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_env_video(video_folder):
    # We use render_mode="rgb_array" for recording
    env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
    
    # Wrap to record video (records episode 0, 1, 2...)
    env = RecordVideo(env, video_folder, episode_trigger=lambda x: True, name_prefix="dagger_agent")
    
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, 96)
    env = FrameStack(env, num_stack=4)
    return env

def record():
    print(f"--- Recording Video for {MODEL_PATH} ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model {MODEL_PATH} not found.")
        return

    # 1. Setup
    env = make_env_video(VIDEO_DIR)
    student = CNNPolicy().to(DEVICE)
    student.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    student.eval()

    # 2. Record 3 Episodes
    for i in range(3):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            obs_proc = preprocess_obs(obs)
            obs_tensor = torch.tensor(obs_proc, device=DEVICE).unsqueeze(0)
            
            with torch.no_grad():
                action = student(obs_tensor).cpu().numpy()[0]
                
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
        print(f"Episode {i+1} Recorded. Score: {total_reward:.2f}")

    env.close()
    print(f"\nVideos saved in '{VIDEO_DIR}' folder!")

if __name__ == "__main__":
    record()