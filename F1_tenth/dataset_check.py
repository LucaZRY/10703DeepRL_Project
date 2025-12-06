import numpy as np
import os

def check_dataset(path):
    print(f"\n=== Checking dataset at: {path} ===")

    if path.endswith(".npz"):
        data = np.load(path)
        print("PPO dataset detected:")
        print(f" obs:     {data['obs'].shape}")
        print(f" actions: {data['actions'].shape}")
        print(f" dones:   {data['dones'].shape}")
        print(f" samples: {data['obs'].shape[0]}")
        print(f" state_dim: {data['obs'][0].size}")
        return

    states_path = os.path.join(path, "states.npy")
    actions_path = os.path.join(path, "actions.npy")

    if os.path.exists(states_path) and os.path.exists(actions_path):
        states = np.load(states_path)
        actions = np.load(actions_path)
        print("Numpy dataset detected:")
        print(f" states:  {states.shape}")
        print(f" actions: {actions.shape}")
        return

    print("❌ Unknown dataset format.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    check_dataset(args.path)
