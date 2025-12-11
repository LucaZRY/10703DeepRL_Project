import sys
import os
import torch
import imageio


current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)


from dagger_online import CNNPolicy, make_env, preprocess_obs


def record_video(policy, save_path="dagger_student.mp4", max_steps=1500, device="cpu"):
    # Use the SAME env setup as training
    env = make_env(render_mode="rgb_array")
    obs, info = env.reset()

    frames = []
    total_reward = 0.0

    print(f"Recording video to {save_path}...")

    for step in range(max_steps):
        # 1. Preprocess (uses the FIXED function from dagger_online.py)
        obs_proc = preprocess_obs(obs)  # -> (4, 96, 96)
        
        # 2. Prepare Tensor
        s_t = torch.tensor(obs_proc, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4, 96, 96)

        # 3. Model Inference
        with torch.no_grad():
            action = policy(s_t).cpu().numpy()[0]

        # 4. Step Env
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 5. Capture Frame
        frame = env.render()
        frames.append(frame)

        if terminated or truncated:
            print(f"Episode finished at step {step}")
            break

    env.close()
    imageio.mimwrite(save_path, frames, fps=30)
    print(f"[Saved video] {save_path}, Total Return = {total_reward:.2f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 3. LOAD MODEL CORRECTLY ---
    # Construct path relative to parent_dir to be safe
    model_path = os.path.join(parent_dir, "student_dagger_diffusion.pt")
    
    if not os.path.exists(model_path):
        # Fallback to the absolute path you provided if relative fails
        model_path = "/home/ruiyangz/Desktop/10703Project/10703DeepRL_Project/student_dagger_diffusion.pt"

    print(f"Loading model from: {model_path}")

    student = CNNPolicy().to(device)
    # weights_only=False is needed because we are loading a full state dict, not just weights
    # (though typically load_state_dict handles the dict object)
    student.load_state_dict(torch.load(model_path, map_location=device))
    student.eval()

    # Record video
    record_video(student, "dagger_student.mp4", device=device)