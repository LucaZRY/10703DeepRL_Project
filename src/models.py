# src/models.py
import math
from typing import Union

import torch
from torch import nn, Tensor

class PolicyDiffusionTransformer(nn.Module):
    def __init__(
        self,
        num_transformer_layers,
        state_dim,
        act_dim,
        hidden_size,
        max_episode_length=1600,
        n_transformer_heads=8,
        device="cpu",
        target="diffusion_policy",
    ):
        super().__init__()
        assert target in ["diffusion_policy", "value_model"], (
            f"target must be either 'diffusion_policy' or 'value_model', got {target}"
        )
        self.num_transformer_layers = num_transformer_layers
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_episode_length = max_episode_length
        self.n_transformer_heads = n_transformer_heads
        self.device = device

        # learnable episode timestep embedding (trajectory time)
        self.episode_timestep_embedding = nn.Embedding(
            self.max_episode_length, self.hidden_size
        )

        # fixed sinusoidal timestep embeddings for diffusion noise level
        self.sinusoidal_timestep_embeddings = self.get_all_sinusoidal_timestep_embeddings(
            self.hidden_size, max_period=10000
        )
        self.sinusoidal_linear_layer = nn.Linear(self.hidden_size, self.hidden_size)

        # embed state, action
        self.state_embedding = nn.Linear(self.state_dim, self.hidden_size)
        self.previous_act_embedding = nn.Linear(self.act_dim, self.hidden_size)
        self.act_embedding = nn.Linear(self.act_dim, self.hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_transformer_heads,
            dim_feedforward=4 * self.hidden_size,
            dropout=0.01,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.num_transformer_layers,
        )

        if target == "diffusion_policy":
            self.predict_noise = nn.Linear(self.hidden_size, self.act_dim)
        else:
            self.predict_noise = nn.Sequential(
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid(),
            )

        self.to(self.device)

    def get_all_sinusoidal_timestep_embeddings(
        self, dim, max_period=10000, num_timesteps=1000
    ):
        timesteps = torch.arange(0, num_timesteps, device=self.device)
        half = dim // 2
        logs = -math.log(max_period)
        arange = torch.arange(start=0, end=half, dtype=torch.float32)
        logfreqs = logs * arange / half
        freqs = torch.exp(logfreqs).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def set_device(self, device: Union[torch.device, str]):
        self.device = device
        self.to(device)

    def forward(
        self,
        previous_states,
        previous_actions,
        noisy_actions,
        episode_timesteps,
        noise_timesteps,
        previous_states_mask=None,
        previous_actions_mask=None,
        actions_padding_mask=None,
    ):
        """
        previous_states: (B, S, state_dim)
        previous_actions: (B, A, act_dim)
        noisy_actions: (B, L, act_dim)
        episode_timesteps: (B, S)  (integer timesteps within episode)
        noise_timesteps: (B, 1)   (diffusion time index, same for whole sequence)
        *_mask: bool, True = padded, False = real
        """
        batch_size, input_seq_length = noisy_actions.shape[0], noisy_actions.shape[1]
        prev_actions_seq_length = previous_actions.shape[1]
        prev_states_seq_length = previous_states.shape[1]

        prev_state_emb = self.state_embedding(previous_states)
        prev_action_emb = self.previous_act_embedding(previous_actions)
        noisy_action_emb = self.act_embedding(noisy_actions)

        # episode timestep embeddings (trajectory time, not diffusion time!)
        episode_timestep_emb = self.episode_timestep_embedding(episode_timesteps)

        # diffusion timestep embedding
        self.sinusoidal_timestep_embeddings = self.sinusoidal_timestep_embeddings.to(
            self.device
        )
        noise_timestep_emb = self.sinusoidal_timestep_embeddings[noise_timesteps]
        # now shape: (B, 1, hidden)
        noise_timestep_emb = self.sinusoidal_linear_layer(noise_timestep_emb)

        # add episode timestep to states + previous actions
        prev_state_emb = prev_state_emb + episode_timestep_emb
        prev_action_emb = prev_action_emb + episode_timestep_emb[
            :, 0:prev_actions_seq_length, :
        ]

        # concat states + actions along seq dim
        prev_obs = torch.cat((prev_state_emb, prev_action_emb), dim=1)

        # prepend diffusion-time token at the beginning of memory
        prev_obs = torch.cat((noise_timestep_emb, prev_obs), dim=1)
        obs_seq_length = prev_obs.shape[1]

        if previous_states_mask is None:
            previous_states_mask = torch.zeros(
                batch_size, prev_states_seq_length, device=self.device, dtype=torch.bool
            )
        if previous_actions_mask is None:
            previous_actions_mask = torch.zeros(
                batch_size, prev_actions_seq_length, device=self.device, dtype=torch.bool
            )

        # memory mask has one extra token for diffusion timestep
        obs_padding_mask = torch.cat(
            (
                torch.zeros(batch_size, 1, device=self.device, dtype=torch.bool),
                previous_states_mask,
                previous_actions_mask,
            ),
            dim=1,
        )

        if actions_padding_mask is None:
            actions_padding_mask = torch.zeros(
                batch_size, input_seq_length, device=self.device, dtype=torch.bool
            )

        # causal mask on the target actions (predict future actions only from past)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            input_seq_length, device=self.device
        )

        output = self.decoder(
            tgt=noisy_action_emb,
            memory=prev_obs,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=actions_padding_mask,
            memory_key_padding_mask=obs_padding_mask,
        )

        noise_preds = self.predict_noise(output)
        return noise_preds
