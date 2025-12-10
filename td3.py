import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

# ============================================================
#  Utils
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Force GPU if available, else warn user
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using device: {device} ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("⚠️  Using device: cpu (Training will be slow!)")


# ============================================================
#  Replay Buffer (GPU OPTIMIZED)
# ============================================================

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Store as uint8 (1 byte) on CPU RAM
        self.obs = np.zeros((max_size, obs_dim), dtype=np.uint8) 
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.uint8)
        
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs.astype(np.uint8)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs.astype(np.uint8)
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        # OPTIMIZATION: Move uint8 to GPU first, then cast to float/normalize.
        # This reduces PCIe bandwidth usage by 4x compared to normalizing on CPU.
        obs = torch.as_tensor(self.obs[idxs], device=device).float() / 255.0
        next_obs = torch.as_tensor(self.next_obs[idxs], device=device).float() / 255.0

        actions = torch.as_tensor(self.actions[idxs], device=device)
        rewards = torch.as_tensor(self.rewards[idxs], device=device)
        dones = torch.as_tensor(self.dones[idxs], device=device)

        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return self.size


# ============================================================
#  Networks: Actor & Critic
# ============================================================

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action, hidden_sizes=(256, 256)):
        super().__init__()
        self.max_action = max_action

        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        a = self.net(obs)
        return torch.tanh(a) * self.max_action


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()

        # Q1
        q1_layers = []
        last_dim = obs_dim + action_dim
        for h in hidden_sizes:
            q1_layers.append(nn.Linear(last_dim, h))
            q1_layers.append(nn.ReLU())
            last_dim = h
        q1_layers.append(nn.Linear(last_dim, 1))
        self.q1 = nn.Sequential(*q1_layers)

        # Q2
        q2_layers = []
        last_dim = obs_dim + action_dim
        for h in hidden_sizes:
            q2_layers.append(nn.Linear(last_dim, h))
            q2_layers.append(nn.ReLU())
            last_dim = h
        q2_layers.append(nn.Linear(last_dim, 1))
        self.q2 = nn.Sequential(*q2_layers)

    def forward(self, obs, action):
        xu = torch.cat([obs, action], dim=-1)
        q1 = self.q1(xu)
        q2 = self.q2(xu)
        return q1, q2

    def q1_only(self, obs, action):
        xu = torch.cat([obs, action], dim=-1)
        return self.q1(xu)


# ============================================================
#  TD3 Agent
# ============================================================

class TD3Agent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        max_action,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(obs_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, action_dim).to(device)
        self.critic_target = Critic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.total_it = 0

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        # Optimization: Move uint8 to GPU first
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        # Convert numpy uint8 -> tensor on GPU -> float -> normalize
        obs_t = torch.as_tensor(obs, device=device).float() / 255.0
            
        action = self.actor(obs_t)
        return action.cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return

        self.total_it += 1

        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1_only(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, net, target_net):
        for p, p_targ in zip(net.parameters(), target_net.parameters()):
            p_targ.data.mul_(1.0 - self.tau)
            p_targ.data.add_(self.tau * p.data)

    def save(self, prefix):
        torch.save(self.actor.state_dict(), f"{prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{prefix}_critic.pt")

    def load(self, prefix):
        # map_location ensures we load to the correct device (gpu if available)
        self.actor.load_state_dict(torch.load(f"{prefix}_actor.pt", map_location=device))
        self.critic.load_state_dict(torch.load(f"{prefix}_critic.pt", map_location=device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


# ============================================================
#  Training Loop
# ============================================================

def make_carracing_env(seed=0):
    env = gym.make("CarRacing-v2", continuous=True, render_mode=None)
    env.reset(seed=seed)
    return env

def save_td3_dataset(save_path, all_states, all_actions, all_rewards, all_dones, episode_lengths, state_dim, action_dim, append_to_existing=True):
    if len(all_states) == 0: return

    states = np.array(all_states, dtype=np.uint8).reshape(-1, state_dim)
    actions = np.array(all_actions, dtype=np.float32).reshape(-1, action_dim)
    rewards = np.array(all_rewards, dtype=np.float32).reshape(-1, 1)
    dones = np.array(all_dones, dtype=np.float32).reshape(-1, 1)
    ep_lens = np.array(episode_lengths, dtype=np.int32)

    if append_to_existing and os.path.exists(save_path):
        print(f"[Dataset] Appending to existing dataset at {save_path}")
        old = np.load(save_path)
        states = np.concatenate([old["states"], states], axis=0) 
        actions = np.concatenate([old["actions"], actions], axis=0)
        rewards = np.concatenate([old["rewards"], rewards], axis=0)
        dones = np.concatenate([old["dones"], dones], axis=0)
        ep_lens = np.concatenate([old["episode_lengths"], ep_lens], axis=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, states=states, actions=actions, rewards=rewards, dones=dones, episode_lengths=ep_lens)
    print(f"[Dataset] Saved dataset to {save_path}")

def train_td3():
    env_name="CarRacing-v2"
    seed=0
    max_episodes=500
    max_steps=1000
    start_timesteps=25_000
    batch_size=256
    save_prefix="td3_carracing"
    dataset_path="data/td3_carracing_dataset.npz"

    set_seed(seed)
    env = make_carracing_env(seed)
    obs, _ = env.reset()

    if isinstance(obs, np.ndarray) and obs.ndim > 1:
        obs_dim = int(np.prod(obs.shape))
        flatten_obs = True
    else:
        obs_dim = obs.shape[0]
        flatten_obs = False

    action_space = env.action_space
    action_dim = action_space.shape[0]
    max_action = float(np.max(np.abs(action_space.high)))

    buffer = ReplayBuffer(obs_dim, action_dim) 
    agent = TD3Agent(obs_dim, action_dim, max_action)

    total_steps = 0
    best_eval_return = -np.inf

    # Storage buffers
    all_states, all_actions, all_rewards, all_dones, episode_lengths = [], [], [], [], []
    episode_rewards = []

    for episode in range(1, max_episodes + 1):
        obs, info = env.reset()
        state = obs.astype(np.uint8).reshape(-1) if flatten_obs else obs.astype(np.uint8)
        
        episode_reward = 0.0
        steps_in_ep = 0

        for step in range(max_steps):
            total_steps += 1
            steps_in_ep += 1

            if total_steps < start_timesteps:
                action = action_space.sample()
                if isinstance(action, np.ndarray): action = action.tolist()
            else:
                action = agent.select_action(state)
                noise = np.random.normal(0, 0.1, size=action_dim)
                action = (action + noise).clip(action_space.low, action_space.high)
                action = action.tolist()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = next_obs.astype(np.uint8).reshape(-1) if flatten_obs else next_obs.astype(np.uint8)

            buffer.add(state, action, reward, next_state, float(done))

            all_states.append(state.copy())
            all_actions.append(action)
            all_rewards.append(reward)
            all_dones.append(float(done))

            state = next_state
            episode_reward += reward

            if total_steps >= start_timesteps:
                agent.train(buffer, batch_size=batch_size)

            if done: break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps_in_ep)
        print(f"[Episode {episode:4d}] Reward: {episode_reward:8.2f} TotalSteps: {total_steps}")

        if episode % 10 == 0:
            eval_ret = evaluate_policy(env, agent, flatten_obs, n_episodes=3)
            if eval_ret > best_eval_return:
                best_eval_return = eval_ret
                agent.save(save_prefix)

        if episode % 50 == 0:
            save_td3_dataset(dataset_path, all_states, all_actions, all_rewards, all_dones, episode_lengths, obs_dim, action_dim)

    env.close()

def evaluate_policy(env, agent, flatten_obs, n_episodes=5):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = obs.astype(np.uint8).reshape(-1) if flatten_obs else obs.astype(np.uint8)
        done, ep_ret = False, 0.0
        while not done:
            action = agent.select_action(state, deterministic=True).clip(env.action_space.low, env.action_space.high).tolist()
            obs, reward, terminated, truncated, _ = env.step(action)
            state = obs.astype(np.uint8).reshape(-1) if flatten_obs else obs.astype(np.uint8)
            ep_ret += reward
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))

if __name__ == "__main__":
    train_td3()