import numpy as np

# Path to your dataset
DATA_PATH = "carracing_dqn_dataset.npz" 

try:
    data = np.load(DATA_PATH)
    print(f"--- Inspecting {DATA_PATH} ---")
    
    # Check Observations
    obs = data['obs']
    print(f"Observation Shape: {obs.shape}")
    print(f"Observation Type:  {obs.dtype}")
    print(f"Observation Range: [{np.min(obs)}, {np.max(obs)}]")
    
    # Check Actions (The Critical Part)
    actions = data['actions']
    print(f"\nAction Shape: {actions.shape}")
    print(f"Action Sample (First 5): \n{actions[:5]}")
    
    # Diagnosis
    if len(actions.shape) == 1 or (len(actions.shape) == 2 and actions.shape[1] == 1):
        print("\n[DIAGNOSIS]: ❌ DISCRETE DATA DETECTED.")
        print("Your data is single integers (Discrete). Your DAgger Student expects vectors of size 3 (Continuous).")
        print("You MUST convert these integers to vectors.")
    elif actions.shape[1] == 3:
        print("\n[DIAGNOSIS]: ✅ CONTINUOUS DATA DETECTED.")
        print("The shape looks correct. Check if values are in range [-1, 1].")
    else:
        print(f"\n[DIAGNOSIS]: ❓ UNKNOWN FORMAT. Shape is {actions.shape}")

except FileNotFoundError:
    print(f"File not found: {DATA_PATH}")