import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from student.policy import Actor
from student.critic import Critic
from utils.replay_buffer import ReplayBuffer
from envs.wrappers import make_carracing_env


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- 1) Create environment ----
    env = make_carracing_env(render_mode=None)
    obs, _ = env.reset()

    obs_dim = obs.shape
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    # ---- 2) Student policy (Actor) + Critic ----
    actor = Actor(act_dim).to(device)
    critic = Critic().to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = optim.Adam(critic.parameters(), lr=3e-4)

    # ---- 3) Replay buffer ----
    rb = ReplayBuffer(capacity=1000000, obs_shape=obs_dim, act_dim=act_dim)

    total_steps = 200000
    start_steps = 10000

    for step in range(total_steps):

        # Use random actions at the start (better exploration)
        if step < start_steps:
            action = env.action_space.sample()
        else:
            # Policy action
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = actor(obs_tensor).cpu().detach().numpy()[0]

        # ---- 4) Step environment ----
        next_obs, reward, done, trunc, info = env.step(action)
        rb.add(obs, action, reward, next_obs, done or trunc)

        obs = next_obs

        if done or trunc:
            obs, _ = env.reset()

        # ---- 5) Update policy after enough data ----
        if step > start_steps:
            batch = rb.sample(256, device=device)
            update(actor, critic, actor_opt, critic_opt, batch, device)

        # ---- Logging ----
        if step % 1000 == 0:
            print("Step:", step)

    env.close()


def update(actor, critic, actor_opt, critic_opt, batch, device):
    obs, act, rew, next_obs, done = batch

    # ----- Critic loss -----
    with torch.no_grad():
        target_q = rew + 0.99 * (1 - done) * critic(next_obs)

    current_q = critic(obs)
    critic_loss = ((current_q - target_q) ** 2).mean()

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    # ----- Actor loss -----
    new_actions = actor(obs)
    actor_loss = -critic(obs).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()


if __name__ == "__main__":
    train()
