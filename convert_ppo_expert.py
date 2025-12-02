import os
import numpy as np

def main():
    # 1) Load PPO dataset
    ppo_path = "carracing_ppo_strong_dataset.npz"  # or strong / fast version
    data = np.load(ppo_path)
    obs     = data["obs"]      # (N,4,96,96)
    actions = data["actions"]  # (N,3)
    dones   = data["dones"]    # (N,)

    N = obs.shape[0]
    state_dim = np.prod(obs.shape[1:])  # 4*96*96
    act_dim   = actions.shape[-1]

    # 2) Split into episodes using dones
    episodes_states = []
    episodes_actions = []
    cur_states = []
    cur_actions = []

    for i in range(N):
        s = obs[i].reshape(-1)    # flatten to (state_dim,)
        a = actions[i]            # (3,)

        cur_states.append(s)
        cur_actions.append(a)

        if dones[i] == 1.0 or i == N - 1:
            episodes_states.append(np.stack(cur_states, axis=0))
            episodes_actions.append(np.stack(cur_actions, axis=0))
            cur_states, cur_actions = [], []

    num_traj = len(episodes_states)
    max_T = max(ep.shape[0] for ep in episodes_states)
    print(f"Found {num_traj} episodes, max length {max_T}")

    # 3) Pad to (num_traj, max_T, dim)
    states_arr  = np.zeros((num_traj, max_T, state_dim), dtype=np.float32)
    actions_arr = np.zeros((num_traj, max_T, act_dim),   dtype=np.float32)

    for i, (ep_s, ep_a) in enumerate(zip(episodes_states, episodes_actions)):
        T = ep_s.shape[0]
        states_arr[i, :T, :]  = ep_s
        actions_arr[i, :T, :] = ep_a

    # 4) Save in data/expert_carracing
    out_dir = "data/expert_carracing"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "states.npy"),  states_arr)
    np.save(os.path.join(out_dir, "actions.npy"), actions_arr)
    print(f"Saved expert trajectories to {out_dir}")
    print(f"  states.npy:  {states_arr.shape}")
    print(f"  actions.npy: {actions_arr.shape}")

if __name__ == "__main__":
    main()
