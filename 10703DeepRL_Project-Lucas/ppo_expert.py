import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import cv2
from dataclasses import dataclass
from typing import Tuple, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Image preprocessing + env
# -------------------------

def image_preprocessing(img: np.ndarray) -> np.ndarray:
    """Resize to 84x84 and convert to grayscale in [0,1]."""
    img = cv2.resize(img, dsize=(84, 84))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img.astype(np.float32)


class CarEnvironment(gym.Wrapper):
    """
    Discrete CarRacing-v2 wrapper with:
      - no-op on reset
      - frame skipping
      - frame stacking (4x84x84)
    """

    def __init__(self, env, skip_frames: int = 4, stack_frames: int = 4, no_operation: int = 50, **kwargs):
        super().__init__(env, **kwargs)
        self._no_operation = no_operation
        self._skip_frames = skip_frames
        self._stack_frames = stack_frames
        self.stack_state = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # Do-nothing steps to randomize initial state a bit
        for _ in range(self._no_operation):
            observation, reward, terminated, truncated, info = self.env.step(0)

        observation = image_preprocessing(observation)
        self.stack_state = np.tile(observation, (self._stack_frames, 1, 1))
        return self.stack_state, info

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        observation = None

        for _ in range(self._skip_frames):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        observation = image_preprocessing(observation)
        self.stack_state = np.concatenate((self.stack_state[1:], observation[np.newaxis]), axis=0)
        return self.stack_state, total_reward, terminated, truncated, info


# -------------------------
# Networks
# -------------------------

class Actor(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self._n_features = 32 * 9 * 9
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self._n_features, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(-1, self._n_features)
        x = self.fc(x)
        return x  # logits over discrete actions


class Critic(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self._n_features = 32 * 9 * 9
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self._n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(-1, self._n_features)
        x = self.fc(x)
        return x.squeeze(-1)  # value scalar


# -------------------------
# PPO Agent (discrete)
# -------------------------

@dataclass
class PPOConfig:
    action_dim: int = 5
    obs_channels: int = 4
    episodes: int = 1500
    horizon: int = 1024
    gamma: float = 0.99
    lam: float = 0.95
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    clip_eps: float = 0.2
    n_updates: int = 3
    batch_size: int = 64
    save_path: str = "ppo_discrete_carracing.pt"


class PPOAgent:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.actor = Actor(cfg.obs_channels, cfg.action_dim).to(device)
        self.critic = Critic(cfg.obs_channels).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.total_rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        state: (4,84,84) numpy -> discrete action index, log_prob
        """
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self.actor(s)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return int(action.item()), float(logp.item())

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        logits = self.actor(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states)
        return logp, entropy, values

    def compute_gae(self, rewards, dones, values, last_value):
        cfg = self.cfg
        T = len(rewards)
        adv = torch.zeros(T, device=device)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * last_value * mask - values[t]
            gae = delta + cfg.gamma * cfg.lam * mask * gae
            adv[t] = gae
            last_value = values[t]
        returns = adv + values
        return adv, returns

    def train(self):
        cfg = self.cfg

        env = gym.make("CarRacing-v2", continuous=False)
        env = CarEnvironment(env)

        for ep in range(cfg.episodes):
            state, _ = env.reset()
            done = False
            ep_reward = 0.0

            states = []
            actions = []
            logps = []
            rewards = []
            dones = []
            values = []

            t = 0
            while not done and t < cfg.horizon:
                action, logp = self.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done_flag = terminated or truncated

                # store
                states.append(state)
                actions.append(action)
                logps.append(logp)
                rewards.append(reward)
                dones.append(float(done_flag))

                # value of current state
                with torch.no_grad():
                    v = self.critic(
                        torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    ).item()
                values.append(v)

                state = next_state
                ep_reward += reward
                done = done_flag
                t += 1

            # bootstrap value
            with torch.no_grad():
                if done:
                    last_value = 0.0
                else:
                    last_value = self.critic(
                        torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    ).item()

            states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.long, device=device)
            old_logp_t = torch.tensor(logps, dtype=torch.float32, device=device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
            values_t = torch.tensor(values, dtype=torch.float32, device=device)

            adv_t, ret_t = self.compute_gae(rewards_t, dones_t, values_t, last_value)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            # PPO updates
            n_samples = states_t.size(0)
            for _ in range(cfg.n_updates):
                idxs = np.random.permutation(n_samples)
                for start in range(0, n_samples, cfg.batch_size):
                    end = start + cfg.batch_size
                    mb_idx = idxs[start:end]

                    mb_states = states_t[mb_idx]
                    mb_actions = actions_t[mb_idx]
                    mb_old_logp = old_logp_t[mb_idx]
                    mb_adv = adv_t[mb_idx]
                    mb_ret = ret_t[mb_idx]

                    logp, entropy, values_pred = self.evaluate_actions(mb_states, mb_actions)
                    ratio = torch.exp(logp - mb_old_logp)

                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    critic_loss = F.mse_loss(values_pred, mb_ret)

                    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                    self.actor_optim.zero_grad()
                    self.critic_optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.actor_optim.step()
                    self.critic_optim.step()

            self.total_rewards.append(ep_reward)
            print(f"[PPO] Episode {ep+1}/{cfg.episodes}, return = {ep_reward:.2f}")

        env.close()
        torch.save(self.actor.state_dict(), cfg.save_path)
        print(f"Saved PPO expert actor to {cfg.save_path}")


# -------------------------
# Expert wrapper for DAgger
# -------------------------

# -------------------------
# Expert wrapper for DAgger
# -------------------------

DISCRETE_TO_CONTINUOUS = [
    np.array([0.0, 0.0, 0.0], dtype=np.float32),   # do nothing
    np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # left
    np.array([+1.0, 0.0, 0.0], dtype=np.float32),  # right
    np.array([0.0, 1.0, 0.0], dtype=np.float32),   # gas
    np.array([0.0, 0.0, 0.8], dtype=np.float32),   # brake
]


class PPOExpertPolicy:
    """
    Wrapper to use the trained discrete PPO actor as an expert
    in a continuous CarRacing-v2 environment with stacked frames.

    Expected obs from dagger_carracing:
      - raw obs (84,84,4) uint8, OR
      - preprocessed (4,84,84) float32 in [0,1]
    """

    def __init__(self, checkpoint_path: str, device_str: str = None):
        self.device = torch.device(device_str) if device_str is not None else device
        self.actor = Actor(in_channels=4, out_dim=len(DISCRETE_TO_CONTINUOUS)).to(self.device)

        # Safe-ish load (you can add weights_only=True if your torch version supports it)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

    def _ensure_preprocessed(self, obs) -> np.ndarray:
        """
        Normalize obs into shape (4, 84, 84), float32 in [0, 1].

        Handles:
        - (84, 84, 4)   from H,W,stack
        - (4, 84, 84)   already channels-first
        - (4, 84, 84,1) or (84,84,4,1) with extra singleton channel
        - (84, 84, 1)   single grayscale frame -> tile to 4
        """
        arr = np.array(obs)

        # Squeeze extra singleton dims if present, e.g. (4,84,84,1) -> (4,84,84)
        # or (1,84,84,4) -> (84,84,4)
        while arr.ndim > 3 and (arr.shape[0] == 1 or arr.shape[-1] == 1):
            if arr.shape[-1] == 1:
                arr = arr.squeeze(-1)
            elif arr.shape[0] == 1:
                arr = arr.squeeze(0)
            else:
                break

        # Now handle 3D / 2D cases
        if arr.ndim == 3:
            # Case (H, W, 4): last dim is stack
            if arr.shape[-1] == 4:
                arr = arr.astype(np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))  # -> (4, H, W)

            # Case (4, H, W): already stacked channels-first
            elif arr.shape[0] == 4:
                arr = arr.astype(np.float32)

            # Case (H, W, 1): single grayscale frame -> tile to 4
            elif arr.shape[-1] == 1:
                img = arr[..., 0].astype(np.float32) / 255.0
                arr = np.tile(img[None, ...], (4, 1, 1))

            else:
                # Fallback: treat as single frame (H, W)
                img = arr.astype(np.float32) / 255.0
                arr = np.tile(img[None, ...], (4, 1, 1))

        elif arr.ndim == 2:
            # Single grayscale frame (H, W) -> tile to 4
            img = arr.astype(np.float32) / 255.0
            arr = np.tile(img[None, ...], (4, 1, 1))

        else:
            raise ValueError(f"Unexpected obs shape in PPOExpertPolicy: {arr.shape}")

        return arr

    def get_action(self, obs) -> np.ndarray:
        """
        obs: state from DAgger env (either stacked frames or raw).
        Returns continuous action [steer, gas, brake].
        """
        s = self._ensure_preprocessed(obs)                     # (4,84,84)
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,4,84,84)
        with torch.no_grad():
            logits = self.actor(s_t)
            probs = torch.softmax(logits, dim=-1)
            # Greedy action for expert
            action_idx = torch.argmax(probs, dim=-1).item()
        return DISCRETE_TO_CONTINUOUS[action_idx].copy()


if __name__ == "__main__":
    cfg = PPOConfig()
    agent = PPOAgent(cfg)
    agent.train()

