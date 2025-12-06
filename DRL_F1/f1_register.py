# from gymnasium.envs.registration import register

# register(
#     id="F1Racing-v0",
#     entry_point="f1_env:F1Env",
#     max_episode_steps=1000,
# )
# f1_register.py
# f1_register.py
from gymnasium.envs.registration import register

register(
    id="F1Racing-v0",
    entry_point="f1_env:F1Env",
    max_episode_steps=1000,
)
