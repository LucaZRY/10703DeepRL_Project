import gymnasium as gym
import torch
import numpy as np
import os
import cv2
from DQN import DQN, CarEnvironment

# --- 1. Helper: Convert Discrete DQN Actions to Continuous ---
def discrete_to_continuous(action_int):
    mapping = {
        0: [0.0, 0.0, 0.0],
        1: [-1.0, 0.0, 0.0],
        2: [+1.0, 0.0, 0.0],
        3: [0.0, 1.0, 0.0], 
        4: [0.0, 0.0, 0.8],
    }
    return np.array(mapping.get(int(action_int), [0.0, 0.0, 0.0]), dtype=np.float32)

def main():
    # --- 2. Setup Device & Environment ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DQN expects 84x84, so we keep this environment as is
    env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
    env = CarEnvironment(env)
    
    # --- 3. Initialize DQN Agent ---
    action_space = env.action_space
    agent = DQN(action_space)
    
    # IMPORTANT: Update this to your best model weight file
    weight_file = 'model_weights_1000.pth' 
    
    if os.path.exists(weight_file):
        print(f"Loading weights from {weight_file}...")
        agent.target_network.load_state_dict(
            torch.load(weight_file, map_location=device)
        )
        agent.network.load_state_dict(agent.target_network.state_dict())
        print("Weights loaded successfully.")
    else:
        print(f"Error: {weight_file} not found. Cannot generate expert data.")
        return

    # --- 4. Collection Loop ---
    num_episodes = 50 
    print(f"Starting data collection for {num_episodes} episodes...")
    
    # Lists for .npy (Traj, Time, FlatState)
    episodes_flat_states = []
    episodes_actions = []

    # Lists for .npz (TotalSteps, 4, 96, 96)
    all_obs_img = []
    all_actions_img = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        # Temp lists for current episode (for .npy structure)
        curr_ep_flat = []
        curr_ep_acts = []
        
        while not done:
            # Prepare state for Agent (84x84)
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Select Action
            action_tensor = agent.select_action(state_tensor, evaluation_phase=True)
            discrete_action = action_tensor.item()
            continuous_action = discrete_to_continuous(discrete_action)
            
            # --- Image Processing ---
            # 1. Transpose to HWC (84, 84, 4) for resizing
            obs_hwc = np.transpose(obs, (1, 2, 0)) 
            
            # 2. Resize to 96x96
            obs_96_hwc = cv2.resize(obs_hwc, (96, 96), interpolation=cv2.INTER_LINEAR)

            # 3. Transpose back to CHW (4, 96, 96)
            obs_96 = np.transpose(obs_96_hwc, (2, 0, 1))
            
            # Normalize to [0,1] for .npz (standard format for DAgger)
            obs_96_norm = obs_96.astype(np.float32) / 255.0
            
            # Flatten for .npy (36864,)
            flat_state_96 = obs_96.reshape(-1).astype(np.float32) / 255.0

            # Store for .npy (Episode structure)
            curr_ep_flat.append(flat_state_96)
            curr_ep_acts.append(continuous_action)

            # Store for .npz (Flat structure)
            all_obs_img.append(obs_96_norm)
            all_actions_img.append(continuous_action)
            
            # Step Env
            next_obs, reward, terminated, truncated, _ = env.step(discrete_action)
            done = terminated or truncated
            obs = next_obs
            
        # Append episode to .npy lists
        episodes_flat_states.append(np.stack(curr_ep_flat))
        episodes_actions.append(np.stack(curr_ep_acts))
        
        print(f"Episode {episode+1}/{num_episodes} collected.")

    env.close()
    
    # --- 5. Save .npy (Padded Trajectories) ---
    max_T = max(len(ep) for ep in episodes_flat_states)
    num_traj = len(episodes_flat_states)
    state_dim = episodes_flat_states[0].shape[1] 
    act_dim = episodes_actions[0].shape[1]
    
    print(f"\nProcessing .npy (Traj format). Max length: {max_T}")
    padded_states = np.zeros((num_traj, max_T, state_dim), dtype=np.float32)
    padded_actions = np.zeros((num_traj, max_T, act_dim), dtype=np.float32)
    
    for i in range(num_traj):
        traj_len = len(episodes_flat_states[i])
        padded_states[i, :traj_len, :] = episodes_flat_states[i]
        padded_actions[i, :traj_len, :] = episodes_actions[i]

    npy_dir = "data/expert_carracing"
    os.makedirs(npy_dir, exist_ok=True)
    np.save(os.path.join(npy_dir, "states.npy"), padded_states)
    np.save(os.path.join(npy_dir, "actions.npy"), padded_actions)
    print(f"Saved .npy to {npy_dir}")

    # --- 6. Save .npz (Flat Image Data) ---
    # Convert lists to arrays
    np_obs = np.array(all_obs_img, dtype=np.float32)    # (N, 4, 96, 96)
    np_acts = np.array(all_actions_img, dtype=np.float32) # (N, 3)

    npz_filename = "carracing_dqn_dataset.npz"
    np.savez_compressed(npz_filename, obs=np_obs, actions=np_acts)
    
    print(f"Saved .npz to {npz_filename}")
    print(f"  .npz Obs shape: {np_obs.shape}")
    print(f"  .npz Acts shape: {np_acts.shape}")

if __name__ == "__main__":
    main()