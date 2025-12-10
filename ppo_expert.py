import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
import cv2
import numpy as np
import random
import pickle
import os
import string
import glob

# --- Setup ---
# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable interactive mode for matplotlib (prevents blocking)
plt.ion()

# --- Helper Functions ---

def image_preprocessing(img):
    """
    Resize image to 84x84 and convert to grayscale.
    Returns normalized float image.
    """
    img = cv2.resize(img, dsize=(84, 84))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img

def plot_results(rewards):
    """
    Plots the training rewards and saves the graph to a file.
    This function is called periodically to update the training curve.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Training Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig("training_curve.png")
    plt.close() # Close the figure to free memory
    print("Graph updated: training_curve.png")

def animate(imgs, video_name, _return=True):
    """
    Compiles a list of image frames into a video file (.webm).
    """
    if video_name is None:
        video_name = ''.join(random.choice(string.ascii_letters) for i in range(18)) + '.webm'
    
    if len(imgs) == 0:
        print("No frames to animate.")
        return

    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)
    video.release()
    print(f"Video saved as {video_name}")

# --- Environment Wrapper ---

class CarEnvironment(gym.Wrapper):
    def __init__(self, env, skip_frames=4, stack_frames=4, no_operation=50, **kwargs):
        super().__init__(env, **kwargs)
        self._no_operation = no_operation
        self._skip_frames = skip_frames
        self._stack_frames = stack_frames
        self.stack_state = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # Perform no-ops to randomize start
        for i in range(self._no_operation):
            observation, reward, terminated, truncated, info = self.env.step(0)

        observation = image_preprocessing(observation)
        # Stack the initial frame multiple times
        self.stack_state = np.tile(observation, (self._stack_frames, 1, 1))
        return self.stack_state, info

    def step(self, action):
        total_reward = 0
        for i in range(self._skip_frames):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        observation = image_preprocessing(observation)
        # Update stack: drop oldest frame, add newest
        self.stack_state = np.concatenate((self.stack_state[1:], observation[np.newaxis]), axis=0)
        return self.stack_state, total_reward, terminated, truncated, info

# --- Neural Networks ---

class Actor(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_features = 32 * 9 * 9

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self._n_features, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view((-1, self._n_features))
        x = self.fc(x)
        return x

class Critic(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_features = 32 * 9 * 9

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self._n_features, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view((-1, self._n_features))
        x = self.fc(x)
        return x

# --- PPO Agent ---

class PPO:
    def __init__(self, action_dim=5, obs_dim=4, episodes=1500, trajectories=300, 
                 gamma=0.99, lr_actor=0.0001, lr_critic=0.0001, clip=0.4, 
                 n_updates=3, lambda_=0.99):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.episodes = episodes
        self.trajectories = trajectories
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.clip = clip
        self.n_updates = n_updates
        self.lambda_ = lambda_
        self._total_rewards = []
        
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim, 1).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_critic)

    def get_action(self, obs):
        """
        Feed observation to Actor, sample action from distribution.
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action_probs = self.actor(obs)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().cpu().numpy(), log_prob.detach()

    def collect_trajectories(self):
        """
        Collect trajectories (observations, rewards, etc.) using current policy.
        """
        batch_obs = []
        batch_rewards = []
        batch_log_probs = []
        batch_actions = []
        batch_dones = []
        t = 0

        # Create env
        env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
        env = CarEnvironment(env)

        while True:
            obs, _ = env.reset()

            while True:
                batch_obs.append(obs)

                a, log_prob = self.get_action(obs)
                batch_actions.append(a)
                batch_log_probs.append(log_prob)

                obs, rew, terminated, truncated, _ = env.step(a.item())
                batch_rewards.append(rew)

                t += 1

                if terminated or truncated or t == self.trajectories:
                    batch_dones.append(1)
                    break
                else:
                    batch_dones.append(0)

            if t == self.trajectories:
                env.close()
                break

        self._total_rewards.append(sum(batch_rewards))

        # Convert to tensors
        batch_obs = np.array(batch_obs)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long)

        # Reward Normalization
        batch_rewards = (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-8)

        return batch_obs, batch_rewards, batch_log_probs, batch_actions, batch_dones

    def compute_discounted_sum(self, batch_rewards, V, batch_dones):
        """
        Computing the discounted reward sum with GAE.
        """
        discounted_sum = []
        gae = 0
        zero = torch.tensor([0])
        V = torch.cat((V.cpu(), zero))

        for i in reversed(range(len(batch_rewards))):
            delta = batch_rewards[i] + self.gamma * V[i + 1] * (1 - batch_dones[i]) - V[i]
            gae = delta + self.gamma * self.lambda_ * gae * (1 - batch_dones[i])
            discounted_sum.insert(0, gae)

        return discounted_sum

    def train(self):
        """
        Main Training Loop with Live Plotting
        """
        print(f"Starting training for {self.episodes} episodes...")
        for episode in range(self.episodes):

            if episode % 10 == 0:
                print(f"Episode {episode} | Last Rewards: {self._total_rewards[-5:]}")

            if (1 + episode) % 50 == 0:
                print(f"Saving Checkpoint: {episode + 1}")
                torch.save(self.actor.state_dict(), f'actor_weights_{episode + 1}.pth')
                torch.save(self.critic.state_dict(), f'critic_weights_{episode + 1}.pth')
                with open('statistics.pkl', 'wb') as f:
                    pickle.dump((self._total_rewards), f)
                
                # --- UPDATE PLOT EVERY 50 EPISODES ---
                plot_results(self._total_rewards)
                # -------------------------------------

            # Collecting the batches with the information
            batch_obs, batch_rewards, batch_log_probs, batch_actions, batch_dones = self.collect_trajectories()

            # Compute V values with the critic network in current states
            V = self.critic(batch_obs.to(device)).squeeze()

            # Compute the discounted sum
            discounted_sum = self.compute_discounted_sum(batch_rewards, V, batch_dones)
            discounted_sum = torch.tensor(discounted_sum, dtype=torch.float32)

            # The advantages to maximize
            advantages = discounted_sum - V.detach().cpu()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update the network
            for update in range(self.n_updates):
                actions_probs = self.actor(batch_obs.to(device))
                action_log_probs = actions_probs.gather(1, batch_actions.to(device)).squeeze()
                ratios = torch.exp(action_log_probs - batch_log_probs.to(device)).cpu()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
                loss = -torch.min(surr1, surr2).mean()

                self.actor_optim.zero_grad()
                loss.backward(retain_graph=True)
                self.actor_optim.step()

                V = self.critic(batch_obs.to(device)).squeeze()
                value_loss = nn.MSELoss()(V, discounted_sum.detach().to(device))

                self.critic_optim.zero_grad()
                value_loss.backward()
                self.critic_optim.step()

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Initialize and Train
    # Set episodes=1500 as per your original request
    model = PPO(episodes=1500) 
    model.train()

    # 2. Final Plot (just in case)
    print("Final Plot Generation...")
    plot_results(model._total_rewards)

    # 3. Evaluation Phase
    print("Starting Evaluation...")
    eval_env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
    eval_env = CarEnvironment(eval_env)

    frames = []
    scores = 0
    s, _ = eval_env.reset()

    done, ret = False, 0

    while not done:
        frames.append(eval_env.render())
        s = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Select Best Action (Argmax) for evaluation
        a = torch.argmax(model.actor(s), dim=-1)
        discrete_action = a.item() % 5
        
        s_prime, r, terminated, truncated, info = eval_env.step(discrete_action)
        s = s_prime
        ret += r
        done = terminated or truncated
        
        if terminated:
            print("Evaluation Episode Terminated")
    
    scores += ret
    print(f"Final Evaluation Score: {scores}")

    # 4. Generate Video
    animate(frames, "carracing_result.webm")