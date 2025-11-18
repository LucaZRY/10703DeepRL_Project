import torch
import imageio

from dagger_carracing_2 import CNNPolicy, make_env, preprocess_obs


def record_video(policy, save_path="dagger_student.mp4", max_steps=1500, device="cpu"):
    # Use the SAME env setup as training (grayscale, 84x84, FrameStack=4)
    env = make_env(render_mode="rgb_array")
    obs, info = env.reset()

    frames = []
    total_reward = 0.0

    for _ in range(max_steps):
        # obs is (84,84,4) uint8 from FrameStack
        obs_proc = preprocess_obs(obs)  # -> (4,84,84) float32
        s_t = torch.tensor(obs_proc, dtype=torch.float32, device=device).unsqueeze(0)  # (1,4,84,84)

        with torch.no_grad():
            action = policy(s_t).cpu().numpy()[0]  # [steer, gas, brake]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        frame = env.render()  # rgb_array frame for video
        frames.append(frame)

        if terminated or truncated:
            break

    env.close()
    imageio.mimwrite(save_path, frames, fps=30)
    print(f"[Saved video] {save_path}, total return = {total_reward:.2f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trained DAgger student
    student = CNNPolicy().to(device)
    student.load_state_dict(torch.load("student_dagger_carracing.pt", map_location=device))
    student.eval()

    # Record video
    record_video(student, "dagger_student.mp4", device=device)
