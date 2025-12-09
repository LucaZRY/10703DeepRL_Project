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
    torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
#  Replay Buffer (MODIFIED FOR HIGH-DIM OBSERVATIONS)
# ============================================================

class ReplayBuffer:
    # Reduced default max_size to 1e5 to prevent memory error 
    # and changed dtype to uint8 for observations.
    def __init__(self, obs_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Store observations as np.uint8 (1 byte) for massive memory savings
        self.obs = np.zeros((max_size, obs_dim), dtype=np.uint8) 
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.uint8)
        
        # Actions, rewards, and dones remain float32
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        # Cast observations to uint8 before storing
        self.obs[self.ptr] = obs.astype(np.uint8)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs.astype(np.uint8)
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        # Scale uint8 obs (0-255) to float32 (0.0-1.0) for the network
        obs = torch.as_tensor(self.obs[idxs] / 255.0, dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions[idxs], device=device)
        rewards = torch.as_tensor(self.rewards[idxs], device=device)
        next_obs = torch.as_tensor(self.next_obs[idxs] / 255.0, dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.dones[idxs], device=device)

        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return self.size


# ============================================================
#  Networks: Actor & Critic (Twin Q)
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
        # tanh to [-1, 1], then scale by max_action
        return torch.tanh(a) * self.max_action


class Critic(nn.Module):
    """
    Twin Q-network: Q1 and Q2 share no parameters.
    Takes (obs, action) as input.
    """
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
        self.obs_dim = obs_dim
        self.action_dim = action_dim
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
        """
        obs: np.array of shape (obs_dim,) - Should be uint8 [0, 255]
        returns: np.array of shape (action_dim,)
        """
        # Convert uint8 obs to float32 [0.0, 1.0] tensor for the network
        if obs.ndim == 1:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device) / 255.0
            
        action = self.actor(obs_t)
        # For td3 exploration, noise is added outside this function
        return action.cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return

        self.total_it += 1

        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = self.actor_target(next_obs)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q
            target_q1, target_q2 = self.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.gamma * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.critic(obs, actions)

        # Critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Actor loss: maximize Q1(obs, actor(obs)) => minimize -Q1
            actor_actions = self.actor(obs)
            actor_loss = -self.critic.q1_only(obs, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
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
        self.actor.load_state_dict(torch.load(f"{prefix}_actor.pt", map_location=device))
        self.critic.load_state_dict(torch.load(f"{prefix}_critic.pt", map_location=device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


# ============================================================
#  Environment helper
# ============================================================

def make_carracing_env(seed=0):
    """
    Simple CarRacing-v2 env.
    You can replace this with your own make_env() + preprocess_obs() later.
    """
    env = gym.make("CarRacing-v2", continuous=True, render_mode=None)
    env.reset(seed=seed)
    return env


# ============================================================
#  Dataset Saving (continuous / appendable)
# ============================================================

def save_td3_dataset(
    save_path,
    all_states,
    all_actions,
    all_rewards,
    all_dones,
    episode_lengths,
    state_dim,
    action_dim,
    append_to_existing=True,
):
    """
    Save transitions in a format usable for diffusion / offline expert:

    - states: (N, state_dim)
    - actions: (N, action_dim)
    - rewards: (N, 1)
    - dones: (N, 1)
    - episode_lengths: (num_episodes,)

    If append_to_existing is True and save_path exists, data will be concatenated.
    Note: Stored states will be uint8 to save memory.
    """
    if len(all_states) == 0:
        print("[Dataset] No transitions to save, skipping.")
        return

    # Ensure states are saved as uint8 for memory efficiency
    states = np.array(all_states, dtype=np.uint8).reshape(-1, state_dim)
    actions = np.array(all_actions, dtype=np.float32).reshape(-1, action_dim)
    rewards = np.array(all_rewards, dtype=np.float32).reshape(-1, 1)
    dones = np.array(all_dones, dtype=np.float32).reshape(-1, 1)
    ep_lens = np.array(episode_lengths, dtype=np.int32)

    if append_to_existing and os.path.exists(save_path):
        print(f"[Dataset] Appending to existing dataset at {save_path}")
        old = np.load(save_path)
        
        # Ensure correct concatenation for uint8 states
        states = np.concatenate([old["states"], states], axis=0) 
        actions = np.concatenate([old["actions"], actions], axis=0)
        rewards = np.concatenate([old["rewards"], rewards], axis=0)
        dones = np.concatenate([old["dones"], dones], axis=0)
        ep_lens = np.concatenate([old["episode_lengths"], ep_lens], axis=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        episode_lengths=ep_lens,
        state_dim=np.array([state_dim], dtype=np.int32),
        action_dim=np.array([action_dim], dtype=np.int32),
    )
    print(f"[Dataset] Saved dataset to {save_path}")
    print(f"          states: {states.shape} (dtype: {states.dtype}), actions: {actions.shape}")


# ============================================================
#  Training Loop
# ============================================================

def train_td3(
    env_name="CarRacing-v2",
    max_episodes=500,
    max_steps=1000,
    start_timesteps=25_000,  # pure random at beginning
    expl_noise=0.1,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_freq=2,
    actor_lr=3e-4,
    critic_lr=3e-4,
    seed=0,
    save_prefix="td3_carracing",
    dataset_path="data/td3_carracing_dataset.npz",
    append_dataset=True,
    save_interval_episodes=50,
):
    set_seed(seed)

    env = make_carracing_env(seed)
    obs, _ = env.reset()

    # If your obs is image or stacked frames, you may want to flatten:
    if isinstance(obs, np.ndarray) and obs.ndim > 1:
        obs_dim = int(np.prod(obs.shape))
        flatten_obs = True
    else:
        obs_dim = obs.shape[0]
        flatten_obs = False

    action_space = env.action_space
    assert isinstance(action_space, gym.spaces.Box), "TD3 needs continuous actions (Box)."
    action_dim = action_space.shape[0]

    # For CarRacing: steer in [-1, 1], gas/brake in [0, 1]
    # TD3 assumes symmetric range, so we use the max of |low| and |high|
    max_action = float(np.max(np.abs(action_space.high)))

    print(f"Obs dim: {obs_dim}, action dim: {action_dim}, max_action: {max_action}")

    # ReplayBuffer now defaults to max_size=1e5 and uses uint8 for observations
    buffer = ReplayBuffer(obs_dim, action_dim) 
    agent = TD3Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        tau=tau,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_freq=policy_freq,
    )

    total_steps = 0
    best_eval_return = -np.inf

    # For logging & dataset
    episode_rewards = []
    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    episode_lengths = []

    for episode in range(1, max_episodes + 1):
        obs, info = env.reset()
        
        # State is stored as np.uint8 [0, 255]
        if flatten_obs and isinstance(obs, np.ndarray):
            state = obs.astype(np.uint8).reshape(-1) 
        else:
            state = obs.astype(np.uint8)

        episode_reward = 0.0
        steps_in_ep = 0

        for step in range(max_steps):
            total_steps += 1
            steps_in_ep += 1

            if total_steps < start_timesteps:
                action = action_space.sample()
                # FIX: Convert numpy array to list for Box2D compatibility
                if isinstance(action, np.ndarray):
                    action = action.tolist()
            else:
                action = agent.select_action(state)
                # Add exploration noise
                noise = np.random.normal(0, expl_noise, size=action_dim)
                action = action + noise
                # Clip to env bounds
                action = np.clip(action, action_space.low, action_space.high)
                
                # FIX: Convert numpy array (of float32) to a standard Python list of floats for env.step()
                action = action.tolist() 

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Next state is stored as np.uint8 [0, 255]
            if flatten_obs and isinstance(next_obs, np.ndarray):
                next_state = next_obs.astype(np.uint8).reshape(-1)
            else:
                next_state = next_obs.astype(np.uint8)

            # buffer.add handles casting to np.uint8, but the state should already be uint8 here
            buffer.add(
                state,
                action,
                reward,
                next_state,
                float(done),
            )

            # Collect transitions for diffusion dataset
            all_states.append(state.copy()) # copy is important if state is mutated
            all_actions.append(action.copy())
            all_rewards.append(reward)
            all_dones.append(float(done))

            state = next_state
            episode_reward += reward

            # TD3 update
            if total_steps >= start_timesteps:
                agent.train(buffer, batch_size=batch_size)

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps_in_ep)

        print(
            f"[Episode {episode:4d}] Reward: {episode_reward:8.2f} "
            f"Steps: {steps_in_ep:4d}  TotalSteps: {total_steps}"
        )

        # Simple eval every N episodes
        if episode % 10 == 0:
            eval_ret = evaluate_policy(env, agent, flatten_obs, n_episodes=3)
            print(f"  -> Eval return (avg over 3): {eval_ret:.2f}")
            if eval_ret > best_eval_return:
                best_eval_return = eval_ret
                print(f"  -> New best eval return! Saving model to {save_prefix}_*.pt")
                agent.save(save_prefix)

        # Periodic dataset + model snapshot to avoid losing progress
        if episode % save_interval_episodes == 0:
            print(f"[Episode {episode}] Periodic save of model + dataset.")
            agent.save(save_prefix)
            save_td3_dataset(
                dataset_path,
                all_states,
                all_actions,
                all_rewards,
                all_dones,
                episode_lengths,
                state_dim=obs_dim,
                action_dim=action_dim,
                append_to_existing=append_dataset,
            )

    env.close()

    # Final save of model and dataset
    print("[Training] Finished. Saving final model and dataset.")
    agent.save(save_prefix)
    save_td3_dataset(
        dataset_path,
        all_states,
        all_actions,
        all_rewards,
        all_dones,
        episode_lengths,
        state_dim=obs_dim,
        action_dim=action_dim,
        append_to_existing=append_dataset,
    )

    # Save and plot rewards
    np.save(f"{save_prefix}_episode_rewards.npy", np.array(episode_rewards, dtype=np.float32))
    plot_rewards(episode_rewards, save_prefix)


# ============================================================
#  Evaluation
# ============================================================

def evaluate_policy(env, agent, flatten_obs, n_episodes=5, max_steps=1000):
    action_space = env.action_space
    returns = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        if flatten_obs and isinstance(obs, np.ndarray):
            state = obs.astype(np.uint8).reshape(-1)
        else:
            state = obs.astype(np.uint8)

        done = False
        ep_ret = 0.0
        steps = 0

        while not done and steps < max_steps:
            steps += 1
            with torch.no_grad():
                # Agent's select_action returns a NumPy array
                action = agent.select_action(state, deterministic=True) 
            action = np.clip(action, action_space.low, action_space.high)
            
            # FIX: Convert NumPy array (of float32) to a standard Python list of floats for env.step()
            action = action.tolist() 

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward

            if flatten_obs and isinstance(next_obs, np.ndarray):
                state = next_obs.astype(np.uint8).reshape(-1)
            else:
                state = next_obs.astype(np.uint8)

        returns.append(ep_ret)

    return float(np.mean(returns))


# ============================================================
#  Plotting
# ============================================================

def plot_rewards(episode_rewards, save_prefix):
    if len(episode_rewards) == 0:
        print("[Plot] No rewards to plot.")
        return

    rewards = np.array(episode_rewards, dtype=np.float32)
    plt.figure()
    plt.plot(rewards, label="Episode reward")

    if len(rewards) >= 10:
        window = 10
        kernel = np.ones(window) / window
        moving_avg = np.convolve(rewards, kernel, mode="valid")
        plt.plot(
            np.arange(window - 1, len(rewards)),
            moving_avg,
            label=f"{window}-episode moving avg",
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("TD3 Training Reward")
    plt.legend()
    plt.tight_layout()

    fig_path = f"{save_prefix}_reward_curve.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"[Plot] Saved reward curve to {fig_path}")


# ============================================================
#  Entry point
# ============================================================

if __name__ == "__main__":
    train_td3()