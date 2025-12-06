import gymnasium as gym
import numpy as np
import torch
from ppo_expert import PPOExpertPolicy
from dagger_carracing_2 import StudentPolicy  # whichever student policy class you use
import imageio

def record_video(agent, save_path="dagger_result.mp4", max_frames=2000):
    env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
    obs, _ = env.reset()

    frames = []
    total_reward = 0

    for _ in range(max_frames):
        # agent expects (4,84,84) from preprocess_obs
        action = agent.get_action(obs)

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward

        frame = env.render()  # (H,W,3)
        frames.append(frame)

        if term or trunc:
            break

    env.close()

    imageio.mimwrite(save_path, frames, fps=30)
    print(f"[Saved] {save_path}, Return = {total_reward:.2f}")
