# ppo_train_praveen.py
# Praveen-style PPO on CarRacing-v2 with 4x96x96 grayscale frames
# Also saves a dataset: carracing_ppo_dataset_fast.npz
# Format:
#   obs:     (N, 4, 96, 96)
#   actions: (N, 3)
#   rewards: (N,)
#   dones:   (N,)

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import torch
import matplotlib.pyplot as plt  # <<< NEW: for plotting

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


def moving_average(x, window=10):
    """
    Simple moving average for smoothing returns.
    """
    if len(x) < window:
        return np.array(x, dtype=np.float32)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    # pad to same length for easier plotting
    pad = np.full(window - 1, ma[0], dtype=np.float32)
    return np.concatenate([pad, ma])


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
        lr=3e-4,
    )

    max_episodes =1000
    max_steps = 800

    # ---- dataset buffers ----
    all_obs = []      # unflattened obs: (4,96,96)
    all_actions = []  # env actions: (3,)
    all_rewards = []  # scalar rewards
    all_dones = []    # bool flags

    # how many episodes to skip before logging (so PPO warms up a bit)
    warmup_episodes = 100

    # <<< NEW: track per-episode returns >>>
    episode_returns = []

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

            # store raw Beta action (a_beta), not env_action, for PPO
            agent.store_transition(obs, a_beta, logp, reward, next_obs_proc)

            # ---- log data for diffusion / DAgger dataset AFTER warmup ----
            if ep >= warmup_episodes:
                all_obs.append(obs.copy())                          # (4,96,96)
                all_actions.append(env_action.astype(np.float32))   # (3,)
                all_rewards.append(np.float32(reward))
                all_dones.append(done)

            obs = next_obs_proc
            episode_return += reward

            if agent._ready_to_update():
                agent.update()

            if done:
                break

        # <<< NEW: record return for this episode >>>
        episode_returns.append(episode_return)

        print(f"[EP {ep}] return = {episode_return:.1f}")

    env.close()

    # ---- NEW: Plot training curve ----
    if len(episode_returns) > 0:
        returns_np = np.array(episode_returns, dtype=np.float32)
        ma_returns = moving_average(returns_np, window=10)

        plt.figure(figsize=(8, 5))
        plt.plot(returns_np, label="Episode return")
        plt.plot(ma_returns, label="Moving avg (window=10)")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("PPO Training on CarRacing-v2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("ppo_training_returns.png")
        # Uncomment this if you want to see the plot interactively:
        # plt.show()
        print("Saved training curve to ppo_training_returns.png")

    # ---- save dataset to carracing_ppo_dataset_fast.npz ----
    if len(all_obs) == 0:
        print("Warning: no data collected (maybe warmup_episodes too large?).")
        return

    obs_arr = np.stack(all_obs, axis=0)        # (N, 4, 96, 96)
    actions = np.vstack(all_actions)           # (N, 3)
    rewards = np.array(all_rewards)
    dones = np.array(all_dones, dtype=bool)

    np.savez(
        "carracing_ppo_dataset_1.npz",
        obs=obs_arr,           # what convert_ppo_expert.py expects
        actions=actions,
        rewards=rewards,
        dones=dones,
    )

    print("Saved carracing_ppo_dataset_fast.npz")
    print("  obs     :", obs_arr.shape)
    print("  actions :", actions.shape)
    print("  rewards :", rewards.shape)
    print("  dones   :", dones.shape)


if __name__ == "__main__":
    train()
