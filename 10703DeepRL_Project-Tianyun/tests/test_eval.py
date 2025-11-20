import numpy as np
import torch

from eval import rollout_policy
from train_student import StudentMLP


class DummyEnv:
    """
    Minimal environment with Gymnasium-like API for testing rollouts.
    """

    def __init__(self, state_dim=4, act_dim=2, max_steps=15):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_steps = max_steps
        self._step = 0
        self._state = np.zeros(self.state_dim, dtype=np.float32)
        self._rng = np.random.default_rng(0)

    def reset(self):
        self._step = 0
        self._state = self._rng.normal(size=self.state_dim).astype(np.float32)
        return self._state.copy(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        noise = self._rng.normal(scale=0.05, size=self.state_dim).astype(np.float32)
        self._state = self._state + 0.1 * action.mean() + noise
        reward = float(-np.linalg.norm(self._state))
        self._step += 1
        done = False
        truncated = self._step >= self.max_steps
        return self._state.copy(), reward, done, truncated, {}

    def render(self):
        return None

    def close(self):
        return None


def test_rollout_policy_shapes_and_returns():
    env = DummyEnv(state_dim=6, act_dim=3, max_steps=10)
    device = torch.device("cpu")
    model = StudentMLP(state_dim=6, act_dim=3, hidden_dim=16, num_hidden_layers=1).to(device)

    num_episodes = 5
    returns = rollout_policy(
        env=env,
        model=model,
        device=device,
        num_episodes=num_episodes,
        max_steps=8,
        render=False,
    )

    assert isinstance(returns, list)
    assert len(returns) == num_episodes
    for ret in returns:
        assert isinstance(ret, float)
        assert not np.isnan(ret)

    env.close()

