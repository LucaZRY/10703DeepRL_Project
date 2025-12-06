"""
environment_test.py
"""

import torch
import gymnasium as gym

def test_pytorch():
    print("=== PyTorch Test ===")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    print()

def test_gymnasium():
    print("=== Gymnasium + CarRacing Test ===")
    try:
        env = gym.make("CarRacing-v2", render_mode=None)
        obs, _ = env.reset()
        print("CarRacing obs shape:", obs.shape)
        env.close()
        print("CarRacing environment loaded successfully.")
    except Exception as e:
        print("CarRacing FAILED to load:", e)
    print()

def test_box2d():
    print("=== Box2D Test ===")
    try:
        from Box2D import b2World
        world = b2World()
        body = world.CreateStaticBody(position=(0, 0))
        print("Box2D loaded OK. Created body at:", body.position)
    except Exception as e:
        print("Box2D FAILED:", e)
    print()

if __name__ == "__main__":
    print("\n============================")
    print(" Running Environment Tests ")
    print("============================\n")

    test_pytorch()
    test_gymnasium()
    test_box2d()

    print("All tests finished.\n")
