import argparse
import os
from typing import Any, Dict, List

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - handled at runtime
    gym = None

from train_student import StudentMLP, select_device


def rollout_policy(
    env: Any,
    model: torch.nn.Module,
    device: torch.device,
    num_episodes: int = 10,
    max_steps: int = 1600,
    render: bool = False,
) -> List[float]:
    """
    Roll out a policy in an environment that matches the gymnasium API.
    """
    model.eval()
    episode_returns: List[float] = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        step_count = 0
        ep_return = 0.0

        while not done and not truncated and step_count < max_steps:
            if render:
                env.render()

            obs_tensor = torch.from_numpy(np.asarray(obs)).float().to(device).unsqueeze(0)
            with torch.no_grad():
                action_tensor = model(obs_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()

            obs, reward, done, truncated, _ = env.step(action)
            ep_return += float(reward)
            step_count += 1

        episode_returns.append(ep_return)

    return episode_returns


def evaluate_student(
    env_id: str,
    model_path: str,
    hidden_dim: int = 256,
    num_hidden_layers: int = 2,
    num_episodes: int = 10,
    max_steps: int = 1600,
    render: bool = False,
) -> Dict[str, float]:
    """
    Load a trained student policy and evaluate it inside a Gymnasium env.
    """
    if gym is None:
        raise ImportError(
            "gymnasium is required for evaluation but is not installed. "
            "Install gymnasium or skip environment rollouts."
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = select_device()

    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = StudentMLP(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    try:
        returns = rollout_policy(
            env=env,
            model=model,
            device=device,
            num_episodes=num_episodes,
            max_steps=max_steps,
            render=render,
        )
    finally:
        env.close()

    returns_array = np.array(returns, dtype=np.float32)
    stats = {
        "mean": float(returns_array.mean()),
        "median": float(np.median(returns_array)),
        "max": float(returns_array.max()),
        "std": float(returns_array.std()),
    }

    print(
        f"Evaluated {model_path} on {env_id}: "
        f"mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
        f"max={stats['max']:.2f}, std={stats['std']:.2f}"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate student policies in gym envs.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "offline_distill", "both"],
        default="both",
        help="Which student checkpoint to evaluate.",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="BipedalWalker-v3",
        help="Gymnasium environment id.",
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1600)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    baseline_path = os.path.join(
        args.results_dir, "student_baseline", "student_baseline_model.pt"
    )
    offline_path = os.path.join(
        args.results_dir, "student_offline_distill", "student_offline_model.pt"
    )

    if args.mode in ("baseline", "both"):
        print("\n=== Evaluating baseline student ===")
        evaluate_student(
            env_id=args.env_id,
            model_path=baseline_path,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            render=args.render,
        )

    if args.mode in ("offline_distill", "both"):
        print("\n=== Evaluating offline-distilled student ===")
        evaluate_student(
            env_id=args.env_id,
            model_path=offline_path,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            render=args.render,
        )


if __name__ == "__main__":
    main()

