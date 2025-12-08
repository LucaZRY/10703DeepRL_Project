# ppo_train_praveen.py
# cite from https://github.com/praveenVnktsh/CarRacingv0-PPO-pytorch

# ppo_train_praveen.py

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import torch

from ppo_praveen_style import CarRacingNet, PPOPraveenStyle


def make_env():
    """
    CarRacing-v2 env with:
    - grayscale
    - 96x96
    - 4-frame stack
    """
    env = gym.make("CarRacing-v2", render_mode=None)
    env = GrayScaleObservation(env, keep_dim=False)  # (96,96)
    env = ResizeObservation(env, 96)
    env = FrameStack(env, 4)                         # LazyFrames, shape (4,96,96)
    return env


def preprocess_obs(obs):
    """
    Convert LazyFrames / whatever into np.ndarray (4,96,96).
    """
    if not isinstance(obs, np.ndarray):
        obs = np.asarray(obs, dtype=np.float32)

    # If shape is (96,96,4), move channels to first dim
    if obs.shape[-1] == 4 and obs.shape[0] != 4:
        obs = np.transpose(obs, (2, 0, 1))

    # If shape is (4,96,96) already, just ensure float32
    obs = obs.astype(np.float32)
    return obs


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()

    net = CarRacingNet(action_dim=3)
    agent = PPOPraveenStyle(
        net,
        device,
        gamma=0.99,
        clip_param=0.2,
        ppo_epoch=4,
        buffer_capacity=2048,
        batch_size=256,
        lr=1e-3,
    )

    max_episodes = 200
    max_steps = 800

    for ep in range(max_episodes):
        obs, _ = env.reset()
        obs = preprocess_obs(obs)

        episode_return = 0.0

        for t in range(max_steps):
            # --- use the new signature: returns env_action, logp, a_beta ---
            env_action, logp, a_beta = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            next_obs_proc = preprocess_obs(next_obs)

            # store raw Beta action (a_beta), not env_action
            agent.store_transition(obs, a_beta, logp, reward, next_obs_proc)

            obs = next_obs_proc
            episode_return += reward

            if agent._ready_to_update():
                agent.update()

            if done:
                break

        print(f"[EP {ep}] return = {episode_return:.1f}")

    env.close()


if __name__ == "__main__":
    train()
