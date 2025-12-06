# # debug_f1_env.py
# import time
# import numpy as np
# import gymnasium as gym

# import f1_register  # makes sure F1Racing-v0 is registered

# def main():
#     # render_mode="human" opens a pygame window
#     env = gym.make("F1Racing-v0", render_mode="human")
#     obs, info = env.reset(seed=0)

#     print("Observation shape:", obs.shape)

#     for t in range(1000):
#         # simple random policy just to see things move
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)

#         # print some info every 50 steps
#         if t % 50 == 0:
#             print(
#                 f"t={t}, reward={reward:.2f}, "
#                 f"speed={info['speed']:.2f}, "
#                 f"dist_center={info['dist_from_center']:.2f}"
#             )

#         if terminated or truncated:
#             print("Episode ended, resetting...")
#             obs, info = env.reset()

#         # small delay so it doesn’t run too fast
#         time.sleep(0.02)

#     env.close()

# if __name__ == "__main__":
#     main()

# debug_f1_env.py
import time
import numpy as np
import gymnasium as gym

import f1_register  # makes sure F1Racing-v0 is registered

def main():
    # render_mode="human" opens a pygame window
    env = gym.make("F1Racing-v0", render_mode="human")
    obs, info = env.reset(seed=0)

    print("Observation shape:", obs.shape)

    for t in range(1000):
        # ---- RENDER FRAME ----
        env.render()  # <-- THIS WAS MISSING

        # simple random policy just to see things move
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # print some info every 50 steps
        if t % 50 == 0:
            print(
                f"t={t}, reward={reward:.2f}, "
                f"speed={info['speed']:.2f}, "
                f"dist_center={info['dist_from_center']:.2f}"
            )

        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()

        time.sleep(0.02)

    env.close()

if __name__ == "__main__":
    main()
