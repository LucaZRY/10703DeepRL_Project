import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_student import CNNPolicy, preprocess_obs

# CONFIG
MODEL_PATH = "student_dagger_final.pt"
NUM_EPISODES = 20  # Run 20 laps to get good statistics
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_env():
    env = gym.make("CarRacing-v2", continuous=True, render_mode=None) # No render for speed
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
    env = gym.wrappers.ResizeObservation(env, 96)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env

def evaluate_and_plot():
    print(f"Evaluating {MODEL_PATH} over {NUM_EPISODES} episodes...")
    
    env = make_env()
    student = CNNPolicy().to(DEVICE)
    student.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    student.eval()
    
    scores = []
    
    for i in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            obs_proc = preprocess_obs(obs)
            obs_t = torch.tensor(obs_proc, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action = student(obs_t).cpu().numpy()[0]
            obs, r, term, trunc, _ = env.step(action)
            score += r
            done = term or trunc
        
        scores.append(score)
        print(f"  Ep {i+1}: {score:.1f}")
        
    env.close()
    
    # --- PLOTTING ---
    plt.figure(figsize=(10, 5))
    
    # 1. Box Plot (Shows consistency)
    plt.subplot(1, 2, 1)
    plt.boxplot(scores, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title("Score Distribution")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    
    # 2. Bar Chart (Shows individual runs)
    plt.subplot(1, 2, 2)
    plt.bar(range(1, NUM_EPISODES+1), scores, color='green')
    plt.axhline(y=900, color='r', linestyle='--', label='Perfect (900)')
    plt.axhline(y=np.mean(scores), color='b', linestyle='-', label=f'Mean ({np.mean(scores):.1f})')
    plt.title("Individual Episode Scores")
    plt.xlabel("Episode")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/final_performance.png")
    print(f"\nPlot saved to results/final_performance.png")
    print(f"Average Score: {np.mean(scores):.2f}")

if __name__ == "__main__":
    evaluate_and_plot()