import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from train_student import CNNPolicy, preprocess_obs  # Import from your script

# CONFIG
MODEL_PATH = "results/student_pretrain.pt"  # The file you just trained
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_env():
    # Use 'human' render mode to watch the car
    env = gym.make("CarRacing-v2", continuous=True, render_mode="human")
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, 96)
    env = FrameStack(env, num_stack=4)
    return env

def test_drive():
    print(f"--- Testing Student Model: {MODEL_PATH} ---")
    
    # 1. Setup Environment
    env = make_env()
    
    # 2. Load Student
    student = CNNPolicy().to(DEVICE)
    if not torch.cuda.is_available():
        # Map cuda weights to cpu if needed
        student.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    else:
        student.load_state_dict(torch.load(MODEL_PATH))
    student.eval()
    
    # 3. Drive Loop
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print("Driving... (Press ESC in the window to stop)")
    while not done:
        # Preprocess
        obs_proc = preprocess_obs(obs)
        obs_tensor = torch.tensor(obs_proc, device=DEVICE).unsqueeze(0) # (1, 4, 96, 96)
        
        # Inference
        with torch.no_grad():
            action = student(obs_tensor).cpu().numpy()[0]
            
        # Step
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        if steps % 100 == 0:
            print(f"Step {steps}, Reward: {total_reward:.2f}, Action: {action}")

    print(f"Final Score: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    test_drive()