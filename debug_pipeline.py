import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from dagger_online import DiffusionExpert  # Re-use your class

# CONFIG
DATA_PATH = "data/human_expert/expert_trajectories.npz"
MODEL_PATH = "results/diffusion_expert/perfect_expert_96.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def debug_pipeline():
    print(f"--- DIAGNOSING PIPELINE ---")
    
    # 1. CHECK DATA
    if not os.path.exists(DATA_PATH):
        print(f"❌ FAIL: Data file not found at {DATA_PATH}")
        return

    print(f"Loading {DATA_PATH}...")
    data = np.load(DATA_PATH)
    states = data['states']
    actions = data['actions']
    
    print(f"States Shape: {states.shape}")   # Expected: (1, N, 36864)
    print(f"Actions Shape: {actions.shape}") # Expected: (1, N, 3)
    
    # Check Normalization (Images)
    s_min, s_max, s_mean = np.min(states), np.max(states), np.mean(states)
    print(f"States Stats: Min={s_min:.4f}, Max={s_max:.4f}, Mean={s_mean:.4f}")
    if s_max > 1.0:
        print("⚠️ WARNING: States are NOT normalized (Max > 1.0). Diffusion expects [0, 1].")
    elif s_max < 0.01:
        print("⚠️ WARNING: States seem too dark (Double Division?).")
    else:
        print("✅ States look normalized (0-1).")

    # Check Actions
    a_min, a_max, a_mean = np.min(actions), np.max(actions), np.mean(actions)
    print(f"Actions Stats: Min={a_min:.4f}, Max={a_max:.4f}, Mean={a_mean:.4f}")
    
    # Check for "Lazy" Data (All zeros)
    gas_usage = np.mean(actions[0, :, 1] > 0.05) * 100
    print(f"Gas Usage: {gas_usage:.1f}% of frames have gas > 5%")
    if gas_usage < 10:
        print("❌ FAIL: The expert hardly ever presses gas! Data is bad.")
        return

    # 2. CHECK MODEL LOADING
    print(f"\nLoading Model {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ FAIL: Model file not found.")
        return
        
    try:
        expert = DiffusionExpert(MODEL_PATH, device=DEVICE)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ FAIL: Model load error: {e}")
        return

    # 3. CHECK PREDICTION (Overfitting Test)
    # Pick a real frame from the dataset where the expert turned/accelerated
    # We want to see if the model can reproduce it.
    
    idx = 100 # Pick an arbitrary step
    if idx >= states.shape[1]: idx = 0
    
    real_obs_flat = states[0, idx]
    real_action = actions[0, idx]
    
    # Reshape flattened state to (4, 96, 96) for the get_action method
    real_obs_img = real_obs_flat.reshape(4, 96, 96)
    
    print(f"\n--- Prediction Test (Step {idx}) ---")
    print(f"Ground Truth Action: {real_action}")
    
    # Force model to predict
    pred_action = expert.get_action(real_obs_img, t=idx)
    print(f"Model Prediction:    {pred_action}")
    
    # Error
    mse = np.mean((real_action - pred_action)**2)
    print(f"MSE Error: {mse:.4f}")
    
    if mse < 0.05:
        print("✅ PASS: Model remembers the training data.")
        print("If the car still crashes, it's an environment/preprocessing mismatch.")
    else:
        print("❌ FAIL: Model output is totally different from expert data.")
        print("The model did not train correctly (underfitted). Increase steps or check loss.")

if __name__ == "__main__":
    debug_pipeline()