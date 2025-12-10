import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class CarRacingNet(nn.Module):
    """
    Actor-Critic network similar in spirit to praveen's Net, but
    adapted for 4x96x96 image input (CarRacing-v2).
    - Input: (B, C=4, H=96, W=96)
    - Output: (alpha, beta) for each of 3 actions + scalar value.
    """

    def __init__(self, action_dim: int = 3):
        super().__init__()

        # Simple CNN encoder – you can make this deeper if needed
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.conv_out_size = 64 * 8 * 8  # 4096

        # --- ADJUSTED: Deeper Shared FC Network for richer feature extraction ---
        self.fc_shared = nn.Sequential(
            nn.Linear(self.conv_out_size, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),              
            nn.ReLU(),
        )
        # -----------------------------------------------------------------------

        # Value head
        self.v_head = nn.Linear(256, 1)
        self.alpha_head = nn.Linear(256, action_dim)
        self.beta_head = nn.Linear(256, action_dim)
    
    def forward(self, obs):
        """
        obs: (B, 4, 96, 96) float tensor in [0,1] or similar.
        Returns: (alpha, beta), v
        """
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        x = self.fc_shared(x)

        v = self.v_head(x)

        alpha = F.softplus(self.alpha_head(x)) + 1.0
        beta = F.softplus(self.beta_head(x)) + 1.0

        return (alpha, beta), v


class PPOPraveenStyle:
    """
    PPO agent implementing praveen's algorithm:
    - Beta policy in [0,1]^3
    - Clipped surrogate objective
    - Shared actor-critic network

    NOTE:
    - The buffer stores RAW Beta samples a_beta in [0,1]^3
      (closer to the original implementation).
    - We convert to env actions (steer, gas, brake) only when stepping the env.
    """

    def __init__(
        self,
        net: CarRacingNet,
        device,
        gamma=0.99,
        clip_param=0.2,
        ppo_epoch=4,
        buffer_capacity=8192, # Adjusted in ppo_train, matching here for consistency
        batch_size=256,
        lr=3e-4,
    ):
        self.net = net.to(device)
        self.device = device
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        # Transition buffer: (s, a_beta, logp_old, r, s_)
        self.counter = 0
        self.buffer = {
            "s": np.zeros((buffer_capacity, 4, 96, 96), dtype=np.float32),
            "a": np.zeros((buffer_capacity, 3), dtype=np.float32),   # raw Beta samples
            "logp": np.zeros((buffer_capacity, 1), dtype=np.float32),
            "r": np.zeros((buffer_capacity, 1), dtype=np.float32),
            "s_": np.zeros((buffer_capacity, 4, 96, 96), dtype=np.float32),
        }

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.training_step = 0

    @torch.no_grad()
    def select_action(self, obs_np):
        """
        obs_np: numpy array (4,96,96) or LazyFrames
        Returns:
          env_action: np.array (3,) in CarRacing space (steer, gas, brake)
          logp: float (log-prob of the Beta-sampled action)
          a_beta: np.array (3,) raw Beta action in [0,1]^3
        """
        # Allow LazyFrames, etc.
        if not isinstance(obs_np, np.ndarray):
            obs_np = np.asarray(obs_np, dtype=np.float32)

        # Make sure channels-first: (4,96,96)
        if obs_np.shape[-1] == 4 and obs_np.shape[0] != 4:
            obs_np = np.transpose(obs_np, (2, 0, 1))

        obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)

        (alpha, beta), _ = self.net(obs)

        # EXTRA SAFETY: clamp to strictly positive
        alpha = torch.clamp(alpha, min=1e-4)
        beta  = torch.clamp(beta,  min=1e-4)

        dist = Beta(alpha, beta)
        a_beta = dist.sample()                      # in [0, 1]^3
        logp = dist.log_prob(a_beta).sum(dim=1)     # (B,)

        a_beta_np = a_beta.squeeze(0).cpu().numpy()  # (3,)

        # Map Beta [0,1]^3 -> CarRacing action space
        steer = 2.0 * a_beta_np[0] - 1.0   # [-1,1]
        gas   = a_beta_np[1]               # [0,1]
        brake = a_beta_np[2]               # [0,1]
        env_action = np.array([steer, gas, brake], dtype=np.float32)

        return env_action, float(logp.item()), a_beta_np

    def store_transition(self, s, a_beta, logp, r, s_):
        """
        s, s_: (4,96,96) np.float32
        a_beta: (3,) np.float32 in [0,1]^3 (raw Beta sample)
        logp: float (log prob of a_beta under old policy)
        r: float (reward)
        """
        idx = self.counter
        self.buffer["s"][idx] = s
        self.buffer["a"][idx] = a_beta
        self.buffer["logp"][idx] = logp
        self.buffer["r"][idx] = r
        self.buffer["s_"][idx] = s_

        self.counter += 1

    def _ready_to_update(self):
        return self.counter >= self.buffer_capacity

    def update(self):
        if not self._ready_to_update():
            return  # not enough data yet

        self.training_step += 1
        print(f"[PPO] Update step {self.training_step}")

        # Convert buffer to tensors
        s = torch.tensor(self.buffer["s"], dtype=torch.float32, device=self.device)
        a = torch.tensor(self.buffer["a"], dtype=torch.float32, device=self.device)        # [0,1]^3
        old_logp = torch.tensor(self.buffer["logp"], dtype=torch.float32, device=self.device)
        r = torch.tensor(self.buffer["r"], dtype=torch.float32, device=self.device)
        s_ = torch.tensor(self.buffer["s_"], dtype=torch.float32, device=self.device)

        # Compute target value & advantage (one-step TD)
        with torch.no_grad():
            (_, v_s) = self.net(s)
            (_, v_s_) = self.net(s_)
            target_v = r + self.gamma * v_s_
            advantage = target_v - v_s

        total_loss = 0.0

        for _ in range(self.ppo_epoch):
            sampler = BatchSampler(
                SubsetRandomSampler(range(self.buffer_capacity)),
                self.batch_size,
                drop_last=False,
            )
            for batch_idx in sampler:
                b_idx = torch.tensor(batch_idx, device=self.device)

                (alpha, beta), v = self.net(s[b_idx])

                # EXTRA SAFETY: guarantee strictly positive params
                alpha = torch.clamp(alpha, min=1e-4)
                beta  = torch.clamp(beta,  min=1e-4)

                dist = Beta(alpha, beta)

                a_b = a[b_idx]  # raw Beta actions in [0,1]
                logp = dist.log_prob(a_b).sum(dim=1, keepdim=True)
                ratio = torch.exp(logp - old_logp[b_idx])

                adv_b = advantage[b_idx]
                target_v_b = target_v[b_idx]

                surr1 = ratio * adv_b
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                ) * adv_b

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.smooth_l1_loss(v, target_v_b)
                loss = actor_loss + 2.0 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss += loss.item()

        print(f"[PPO] Loss = {total_loss:.4f}")

        # Reset buffer
        self.counter = 0

    # ---------- save/load methods ----------

    def save(self, path: str):
        """
        Save only the network weights. To reload, you must recreate
        CarRacingNet and PPOPraveenStyle with the same architecture/hparams.
        """
        torch.save(self.net.state_dict(), path)
        print(f"[PPO] Saved model to {path}")

    def load(self, path: str):
        """
        Load network weights from a file saved by `save()`.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state_dict)
        print(f"[PPO] Loaded model from {path}")