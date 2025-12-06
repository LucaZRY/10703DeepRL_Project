import torch
import imageio

from dagger_carracing_2 import make_env   # same env as training
from ppo_expert import PPOExpertPolicy


def record_expert_video(expert, save_path="expert_run.mp4", max_steps=1500, device="cpu"):
    # Use the SAME wrappers as DAgger (GrayScale, Resize 84, FrameStack 4), but with rgb_array rendering
    env = make_env(render_mode="rgb_array")
    obs, info = env.reset()

    frames = []
    total_reward = 0.0

    for _ in range(max_steps):
        # PPOExpertPolicy expects raw obs (any of 84x84x4, 4x84x84, etc)
        action = expert.get_action(obs)  # returns [steer, gas, brake]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame = env.render()  # rgb_array frame
        frames.append(frame)

        if terminated or truncated:
            break

    env.close()
    imageio.mimwrite(save_path, frames, fps=30)
    print(f"[Saved expert video] {save_path}, total return = {total_reward:.2f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trained PPO expert (discrete→continuous wrapper)
    expert = PPOExpertPolicy("ppo_discrete_carracing.pt", device_str=device)

    # Record one rollout
    record_expert_video(expert, "expert_run.mp4", device=device)
