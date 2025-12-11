"""
humandata_process.py
Converts raw human RGB recordings into the format required by train_diffusion.py
Includes "Idle Filtering" to remove frames where the car is stopped/coasting.
"""
import numpy as np
import os

# --- CONFIG ---
# 1. Input: Your raw recording
INPUT_PATH = "carracing_human_dataset.npz" 

# 2. Output: Where train_diffusion.py expects to find the data
OUTPUT_DIR = "data/human_expert"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "expert_trajectories.npz")

def rgb2gray(rgb):
    """Convert RGB to Grayscale."""
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def process_data():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Could not find {INPUT_PATH}")
        return

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {INPUT_PATH}...")
    data = np.load(INPUT_PATH)
    
    # Handle keys (obs vs states)
    if 'obs' in data:
        raw_obs = data['obs']
        raw_acts = data['actions']
    elif 'states' in data:
        raw_obs = data['states']
        raw_acts = data['actions']
    else:
        raise KeyError("File contains neither 'obs' nor 'states'")

    print(f"Raw Observations Shape: {raw_obs.shape}")
    
    processed_states = []
    processed_actions = []
    
    # Initialize stack of 4 empty frames (96x96)
    # Ensure dtype is float32 to match processing
    stack = [np.zeros((96, 96), dtype=np.float32) for _ in range(4)]
    
    kept_count = 0
    dropped_count = 0
    
    print("Processing frames (Filtering idle steps)...")
    
    for i in range(len(raw_obs)):
        # --- 1. IDLE FILTERING ---
        act = raw_acts[i] # [Steer, Gas, Brake]
        
        # Check if action is "doing nothing" (Steer ~0 AND Gas ~0)
        # We generally keep braking frames because stopping is an important skill,
        # but pure coasting often teaches the model to be lazy.
        is_idle = (abs(act[0]) < 0.05) and (act[1] < 0.05) and (act[2] < 0.05)
        
        if is_idle:
            dropped_count += 1
            # We still need to update the stack even if we drop the sample
            # so the next frame has the correct history.
            # However, simpler logic is to just skip adding to the TRAINING set.
            # But we must update the stack history.
            
            frame = raw_obs[i]
             # Normalize & Gray just for the stack update
            if np.issubdtype(frame.dtype, np.integer) or np.max(frame) > 1.0:
                frame = frame.astype(np.float32) / 255.0
            else:
                frame = frame.astype(np.float32)
                
            if frame.ndim == 3 and frame.shape[-1] == 3:
                gray = rgb2gray(frame)
            else:
                gray = frame
            
            if gray.ndim == 3: gray = gray.squeeze()
            
            stack.pop(0)
            stack.append(gray)
            continue 
        # -------------------------

        frame = raw_obs[i] # Expecting (96, 96, 3)
        
        # 2. Normalize to [0, 1]
        if np.issubdtype(frame.dtype, np.integer) or np.max(frame) > 1.0:
            frame = frame.astype(np.float32) / 255.0
        else:
            frame = frame.astype(np.float32)

        # 3. Convert to Grayscale (96, 96)
        if frame.ndim == 3 and frame.shape[-1] == 3:
            gray = rgb2gray(frame)
        else:
            gray = frame

        # 4. SAFETY CHECK: Remove extra dimensions if any (e.g. 96x96x1 -> 96x96)
        if gray.ndim == 3:
            gray = gray.squeeze()
            
        # 5. Update Stack
        stack.pop(0)
        stack.append(gray)
        
        # 6. Stack & Flatten
        try:
            state_stack = np.array(stack, dtype=np.float32) # (4, 96, 96)
            state_flat = state_stack.flatten() # (36864,)
            
            processed_states.append(state_flat)
            processed_actions.append(act) # Keep the action sync'd
            kept_count += 1
            
        except ValueError as e:
            print(f"Error stacking frame {i}. Shapes in stack: {[s.shape for s in stack]}")
            raise e

    # Reshape for Trainer: (Num_Trajectories, Time, Dim)
    states_out = np.array(processed_states).astype(np.float32) 
    actions_out = np.array(processed_actions).astype(np.float32)
    
    # Add Batch Dimension (1, N, Dim)
    states_out = states_out[np.newaxis, ...]
    actions_out = actions_out[np.newaxis, ...]

    print(f"Filtering Complete: Kept {kept_count} samples, Dropped {dropped_count} idle samples.")
    print(f"Final States Shape: {states_out.shape}")
    print(f"Final Actions Shape: {actions_out.shape}")
    
    np.savez(OUTPUT_PATH, states=states_out, actions=actions_out)
    print(f"Saved processed data to {OUTPUT_PATH}")
    print("Ready for training!")

if __name__ == "__main__":
    process_data()