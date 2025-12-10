import gymnasium as gym
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count
import random
import math
import pickle
import os # Added for file path management, though not strictly required by original code

# --- Setup for plotting and device ---
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Image Preprocessing Function ---
def image_preprocessing(img):
    """
    Preprocesses the raw image observation from the environment.
    1. Resizes to 84x84.
    2. Converts from RGB to Grayscale.
    3. Normalizes pixel values to [0.0, 1.0].
    """
    # 
    img = cv2.resize(img, dsize=(84, 84))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img

# --- Custom Car Environment Wrapper ---
class CarEnvironment(gym.Wrapper):
    """
    A custom wrapper for the CarRacing-v2 environment to:
    1. Apply a fixed 'no_operation' action at reset for variety.
    2. Implement frame skipping (repeat the action for 'skip_frames').
    3. Implement frame stacking (stack 'stack_frames' preprocessed images).
    """
    def __init__(self, env, skip_frames=3, stack_frames=4, no_operation=50, **kwargs):
        super().__init__(env, **kwargs)
        self._no_operation = no_operation
        self._skip_frames = skip_frames
        self._stack_frames = stack_frames
        # The new observation space is a stack of preprocessed frames
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(stack_frames, 84, 84), dtype=np.float32
        )
        self.stack_state = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # Apply no-op at the start
        for i in range(self._no_operation):
            # The original code uses a hardcoded 0, assuming 'no action'
            observation, reward, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                 break # Break if environment terminates early during no-op

        observation = image_preprocessing(observation)
        # Initialize the stack with the first frame repeated
        self.stack_state = np.tile(observation, (self._stack_frames, 1, 1))
        return self.stack_state, info

    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False
        info = {}

        # Frame skipping
        for i in range(self._skip_frames):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        observation = image_preprocessing(observation)
        # Frame stacking: remove the oldest frame and append the new one
        # 
        self.stack_state = np.concatenate((self.stack_state[1:], observation[np.newaxis]), axis=0)
        return self.stack_state, total_reward, terminated, truncated, info

# --- Convolutional Neural Network (CNN) Architecture ---
class CNN(nn.Module):
    """
    The Q-network architecture using Convolutional Layers to process stacked frames.
    Input: Stack of 4 84x84 grayscale frames.
    Output: Q-values for each discrete action.
    """
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Calculate the size of the features after the convolutions
        # 84x84 -> Conv(k8, s4) -> 20x20
        # 20x20 -> Conv(k4, s2) -> 9x9
        self._n_features = 32 * 9 * 9 

        self.conv = nn.Sequential(
            # Input: 4x84x84
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            # Output: 16x20x20
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            # Output: 32x9x9
        )

        self.fc = nn.Sequential(
            nn.Linear(self._n_features, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels),
        )


    def forward(self, x):
        x = self.conv(x)
        # Flatten the convolutional output
        x = x.view((-1, self._n_features))
        x = self.fc(x)
        return x

# --- Replay Memory ---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
# 

class ReplayMemory(object):
    """
    A memory buffer to store and sample past experiences (transitions).
    Used to break correlations between sequential samples during training.
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- Deep Q-Network (DQN) Agent ---
class DQN:
    """
    Deep Q-Network implementation with a policy network and a target network.
    It handles action selection (epsilon-greedy) and the training step.
    """
    def __init__(self, action_space, batch_size=256, gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000, lr=0.001):
        # State space: 4 stacked frames
        self._n_observation = 4
        # Action space: 5 discrete actions for CarRacing-v2
        self._n_actions = 5
        self._action_space = action_space # Gymnasium Discrete action space object
        self._batch_size = batch_size
        self._gamma = gamma
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay
        self._lr = lr
        self._total_steps = 0
        self._evaluate_loss = []
        
        # Policy Network (Q) and Target Network (Q')
        # 
        self.network = CNN(self._n_observation, self._n_actions).to(device)
        self.target_network = CNN(self._n_observation, self._n_actions).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.AdamW(self.network.parameters(), lr=self._lr, amsgrad=True)
        self._memory = ReplayMemory(10000)

    def select_action(self, state, evaluation_phase=False):
        """
        Selects an action using the epsilon-greedy strategy during training
        or a purely greedy strategy during evaluation.
        """
        # Calculating the threshold (epsilon) - decreases over time
        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1. * self._total_steps / self._eps_decay)
        
        if not evaluation_phase:
            self._total_steps += 1

        if evaluation_phase:
            # Evaluation mode: use target network for stable, greedy action selection
            with torch.no_grad():
                return self.target_network(state).max(1).indices.view(1, 1)
        
        elif random.random() > eps_threshold:
            # Exploitation: select the action with the highest Q-value from the policy network
            with torch.no_grad():
                return self.network(state).max(1).indices.view(1, 1)
        
        else:
            # Exploration: randomly select an action
            return torch.tensor([[self._action_space.sample()]], device=device, dtype=torch.long)

    def train(self):
        """
        Performs one step of optimization on the policy network.
        Calculates the target Q-value using the target network (Q').
        """
        if len(self._memory) < self._batch_size:
            return

        # Sample a batch of transitions from the replay memory
        transitions = self._memory.sample(self._batch_size)
        batch = Transition(*zip(*transitions))

        # Create a mask for non-final states (where the episode didn't end)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        # Concatenate non-final next states
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the Q-values of the *selected* actions at state s_t
        # network(state_batch) outputs Q-values for all actions.
        # gather(1, action_batch) selects the Q-value for the action taken (a)
        state_action_values = self.network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) = max_a' Q'(s_{t+1}, a') for all next states.
        # Initialize next_state_values to zero
        next_state_values = torch.zeros(self._batch_size, device=device)
        
        with torch.no_grad():
            # For non-final states, use the target network to find the max Q' value
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values

        # Compute the expected Q values (The Bellman Target)
        # Expected Q = R + gamma * max_a' Q'(s_{t+1}, a')
        expected_state_action_values = (next_state_values * self._gamma) + reward_batch

        # Compute the loss using SmoothL1Loss (Huber loss)
        # The loss is (Q(s_t, a) - Expected_Q)^2
        criterion = nn.SmoothL1Loss()
        # unsqueeze(1) is needed to match the dimensions for the loss function
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.optimizer.step()

        self._evaluate_loss.append(loss.item())
        
        return

    def copy_weights(self):
        """Copies weights from the policy network to the target network."""
        self.target_network.load_state_dict(self.network.state_dict())

    def get_loss(self):
        """Returns the list of recorded losses."""
        return self._evaluate_loss

    def save_model(self, i):
        """Saves the target network weights."""
        torch.save(self.target_network.state_dict(), f'model_weights_{i}.pth')

    def load_model(self, i):
        """Loads weights into the target network."""
        self.target_network.load_state_dict(torch.load(f'model_weights_{i}.pth'))

# --- Plotting Function ---
def plot_statistics(x, y, title, x_axis, y_axis):
    """Generates and saves a plot of training statistics."""
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.grid(True)
    plt.savefig(f'{title}.png')
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    plt.close() # Close the figure to free memory

# --- Video Animation Function ---
def animate(imgs, video_name, _return=True):
    """
    Creates a video file (webm) from a list of image frames.
    Requires opencv-python (cv2).
    """
    # This block requires cv2 to run, assuming it's available from imports
    height, width, layers = imgs[0].shape
    
    if video_name is None:
        import string
        video_name = ''.join(random.choice(string.ascii_letters) for i in range(18)) + '.webm'
    
    # VP90 codec is for webm format
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height)) # 10 FPS

    for img in imgs:
        # OpenCV expects BGR, but gym renders RGB, so convert
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)
    video.release()
    print(f"Evaluation video saved as: {video_name}")

# --- Main Training Loop ---
def main_training_loop():
    """Initializes environment, agent, and runs the training loop."""
    rewards_per_episode = []
    episode_duration = []
    average_episode_loss = []

    episodes = 1000
    C = 5 # Target network update frequency

    # Initialize the base environment and agent
    base_env = gym.make('CarRacing-v2', continuous=False)
    n_actions = base_env.action_space
    agent = DQN(n_actions)
    base_env.close() # Close the temporary environment

    # Try to load existing model/stats if they exist
    try:
        agent.load_model(episodes) # Try loading the final model if available
        with open('statistics.pkl', 'rb') as f:
            (episode_duration, rewards_per_episode, average_episode_loss) = pickle.load(f)
        print("Loaded previous model weights and statistics.")
        start_episode = len(rewards_per_episode) + 1
    except FileNotFoundError:
        print("No previous model or statistics found, starting from episode 1.")
        start_episode = 1
    except Exception as e:
        print(f"Error loading model or statistics: {e}. Starting from episode 1.")
        start_episode = 1


    for episode in range(start_episode, episodes + 1):
        if episode % 10 == 0:
            print(f"--- Starting episode {episode} ---")
            
        # Create a new environment for the episode and wrap it
        env = gym.make('CarRacing-v2', continuous=False)
        env = CarEnvironment(env)
        
        state, info = env.reset()
        
        # Convert initial state to a tensor, ready for the network (shape: [1, C, H, W])
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        episode_total_reward = 0
        agent._evaluate_loss = [] # Reset loss tracking for the episode

        for t in count():
            # Agent selects an action (epsilon-greedy)
            action = agent.select_action(state)
            
            # Environment step
            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            # Convert reward to tensor and update total reward
            reward = torch.tensor([reward], device=device)
            episode_total_reward += reward
            
            done = terminated or truncated

            if terminated:
                next_state = None
                # print("Finished the lap successfully!") # This often means running out of time, not necessarily success in car racing
            else:
                # Convert next observation to tensor
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in replay memory
            agent._memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform a training optimization step
            agent.train()

            if done:
                env.close()
                # Only record statistics if enough experience is in memory
                if len(agent._memory) >= agent._batch_size:
                    episode_duration.append(t + 1)
                    rewards_per_episode.append(episode_total_reward.item())
                    ll = agent.get_loss()
                    if ll:
                        average_episode_loss.append(sum(ll) / len(ll))
                    
                    # print(f"Episode {episode} finished in {t+1} steps with reward: {episode_total_reward.item():.2f}")
                
                # Plot and save statistics every 10 episodes
                if episode % 10 == 0 and len(rewards_per_episode) > 0:
                    x = [k for k in range(len(rewards_per_episode))]
                    plot_statistics(x, rewards_per_episode, "Rewards for every episode", "Episode", "Reward")
                    if average_episode_loss:
                        plot_statistics(x, average_episode_loss, "Average loss for every episode", "Episode", "Average Loss")
                    plot_statistics(x, episode_duration, "Duration (in steps) for every episode", "Episode", "Duration")

                # Save model weights and stats periodically
                if episode % 100 == 0:
                    agent.save_model(episode)
                    with open('statistics.pkl', 'wb') as f:
                        pickle.dump((episode_duration, rewards_per_episode, average_episode_loss), f)
                        
                break

        # Update the target network (hard update)
        if episode % C == 0:
            agent.copy_weights()
            # print(f"Target network updated at episode {episode}")

    # Final save
    agent.save_model(episodes)
    with open('statistics.pkl', 'wb') as f:
        pickle.dump((episode_duration, rewards_per_episode, average_episode_loss), f)
        
    print("\n--- Training finished ---")
    print(f"Final model saved as 'model_weights_{episodes}.pth'")
    
    return agent

# --- Evaluation of the Agent ---
def evaluate_agent(agent, episodes):
    """Runs a single evaluation episode with rendering and video recording."""
    print("\n--- Starting Evaluation ---")
    
    # Load the trained model for evaluation
    try:
        agent.load_model(episodes)
    except FileNotFoundError:
        print(f"Warning: Could not find model_weights_{episodes}.pth. Using current weights.")
    except Exception as e:
        print(f"Error loading model: {e}. Using current weights.")

    # Create the evaluation environment with rendering enabled
    eval_env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
    # Use the custom wrapper for preprocessing/stacking
    eval_env = CarEnvironment(eval_env)
    
    frames = []
    
    # Ensure a reproducible starting track/position for evaluation
    # This might not be strictly necessary, but good practice
    # eval_env.np_random = np.random.default_rng(42) # This line from original code is complex with Gymnasium v26
    
    # Reset the environment
    s, _ = eval_env.reset()

    done, ret = False, 0
    
    # Evaluation loop
    while not done:
        # Capture the frame for video recording
        frames.append(eval_env.render())
        
        # Convert state to tensor
        s_tensor = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Agent selects a GREEDY action (evaluation_phase=True)
        a_tensor = agent.select_action(s_tensor, evaluation_phase=True)
        discrete_action = a_tensor.item() % 5 # Get the discrete action index
        
        # Step the environment
        s_prime, r, terminated, truncated, info = eval_env.step(discrete_action)
        
        # Update state, return, and done status
        s = s_prime
        ret += r
        done = terminated or truncated
        
        if terminated:
            print("Episode terminated during evaluation.")

    eval_env.close()
    
    print(f"Evaluation finished with total reward: {ret:.2f}")

    # Generate the animation video
    if frames:
        animate(frames, 'car_racing_evaluation.webm')

# --- Run the main functions ---
if __name__ == '__main__':
    
    # NOTE: In a standard Python environment, you must first run the installation commands manually.
    # !pip install swig
    # !pip install gymnasium[box2d]

    # Set episodes for the training and evaluation
    EPISODES = 1000
    
    # Run the training loop
    agent = main_training_loop()
    
    # Run the evaluation
    evaluate_agent(agent, EPISODES)

    # Re-run plotting from the final saved statistics for confirmation
    try:
        with open('statistics.pkl', 'rb') as f:
            data_tuple = pickle.load(f)
        
        episode_duration, rewards_per_episode, average_episode_loss = data_tuple
        
        # Only plot for the number of episodes actually completed and recorded
        recorded_episodes = len(rewards_per_episode)
        if recorded_episodes > 0:
            x = [k for k in range(recorded_episodes)]
            print("\n--- Final Plots ---")
            plot_statistics(x, rewards_per_episode, "Rewards for every episode", "Episode", "Reward")
            if average_episode_loss:
                plot_statistics(x, average_episode_loss, "Average loss for every episode", "Episode", "Average Loss")
            plot_statistics(x, episode_duration, "Duration (in steps) for every episode", "Episode", "Duration")
        else:
            print("No episode statistics recorded to plot.")
            
    except FileNotFoundError:
        print("Final statistics file not found.")