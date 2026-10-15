#!/usr/bin/env python

"""Layer-wise flow-matching head adapted from the official StarVLA Qwen-PI.

Each action-expert block cross-attends to the matching hidden layer from Qwen,
rather than making the whole action expert attend only to Qwen's final layer.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Beta

from policy.starvla_groot.action_head.cross_attention_dit import (
    BasicTransformerBlock,
    TimestepEncoder,
)
from policy.starvla_groot.action_head.flow_matching_head import MLP, ActionEncoder


@dataclass
class LayerwiseActionHeadConfig:
    hidden_size: int
    num_layers: int
    attention_head_dim: int
    action_dim: int
    state_dim: int
    action_horizon: int
    cross_attention_dim: int | None = None
    num_inference_timesteps: int = 10
    num_target_vision_tokens: int = 32
    add_pos_embed: bool = True
    max_seq_len: int = 1024
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    dropout: float = 0.0
    final_dropout: bool = False
    activation_fn: str = "geglu"


class LayerwiseFlowmatchingActionHead(nn.Module):
    """Flow-matching DiT whose block ``i`` consumes Qwen hidden layer ``i``."""

    def __init__(self, config: LayerwiseActionHeadConfig):
        super().__init__()
        if config.hidden_size % config.attention_head_dim:
            raise ValueError(
                "Action expert hidden_size must be divisible by layerwise_attention_head_dim: "
                f"{config.hidden_size} % {config.attention_head_dim} != 0."
            )
        self.config = config
        self.action_dim = int(config.action_dim)
        self.action_horizon = int(config.action_horizon)
        self.num_inference_timesteps = int(config.num_inference_timesteps)
        num_heads = config.hidden_size // config.attention_head_dim

        self.timestep_encoder = TimestepEncoder(config.hidden_size)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=config.hidden_size,
                    num_attention_heads=num_heads,
                    attention_head_dim=config.attention_head_dim,
                    dropout=config.dropout,
                    cross_attention_dim=config.cross_attention_dim or config.hidden_size,
                    activation_fn=config.activation_fn,
                    attention_bias=True,
                    norm_elementwise_affine=False,
                    norm_type="ada_norm",
                    final_dropout=config.final_dropout,
                    positional_embeddings=None,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.state_encoder = (
            MLP(config.state_dim, config.hidden_size, config.hidden_size) if config.state_dim else None
        )
        self.action_encoder = ActionEncoder(config.action_dim, config.hidden_size)
        self.action_decoder = MLP(config.hidden_size, config.hidden_size, config.action_dim)
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, config.hidden_size)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        self.position_embedding = None
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        sample = sample.clamp(max=self.config.noise_s)
        return (self.config.noise_s - sample) / self.config.noise_s

    def _validate_hidden_layers(self, vl_embs_list: list[torch.Tensor]) -> None:
        if len(vl_embs_list) != len(self.transformer_blocks):
            raise ValueError(
                "Qwen hidden-state count must match layer-wise DiT block count: "
                f"got {len(vl_embs_list)} and {len(self.transformer_blocks)}."
            )

    def _prepare_action_mask(
        self, action_mask: torch.Tensor | None, actions: torch.Tensor
    ) -> torch.Tensor | None:
        if action_mask is None:
            return None
        action_mask = action_mask.to(device=actions.device, dtype=actions.dtype)
        if action_mask.ndim == 2:
            action_mask = action_mask[:, None, :].expand(actions.shape[0], actions.shape[1], -1)
        return action_mask

    def _encode_tokens(
        self, actions: torch.Tensor, timesteps: torch.Tensor, state: torch.Tensor | None
    ) -> torch.Tensor:
        action_features = self.action_encoder(actions, timesteps)
        if self.position_embedding is not None:
            pos_ids = torch.arange(action_features.shape[1], device=actions.device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        tokens = [
            self.future_tokens.weight.unsqueeze(0).expand(actions.shape[0], -1, -1),
            action_features,
        ]
        if state is not None:
            state_features = self.state_encoder(state)
            if self.position_embedding is not None and state_features.shape[1] > 1:
                state_pos = torch.arange(state_features.shape[1], device=actions.device)
                state_features = state_features + self.position_embedding(state_pos).unsqueeze(0)
            tokens.insert(0, state_features)
        return torch.cat(tokens, dim=1)

    def _layerwise_dit(
        self,
        tokens: torch.Tensor,
        timesteps: torch.Tensor,
        vl_embs_list: list[torch.Tensor],
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        self._validate_hidden_layers(vl_embs_list)
        temb = self.timestep_encoder(timesteps)
        hidden = tokens
        for block, vl_hidden in zip(self.transformer_blocks, vl_embs_list, strict=True):
            hidden = block(
                hidden_states=hidden,
                encoder_hidden_states=vl_hidden,
                encoder_attention_mask=encoder_attention_mask,
                temb=temb,
            )
        return hidden

    def forward(
        self,
        vl_embs_list: list[torch.Tensor],
        actions: torch.Tensor,
        state: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_mask = self._prepare_action_mask(action_mask, actions)
        noise = torch.randn_like(actions)
        if action_mask is not None:
            actions = actions * action_mask
            noise = noise * action_mask

        time = self.sample_time(actions.shape[0], actions.device, actions.dtype)
        time_broadcast = time[:, None, None]
        noisy_actions = (1.0 - time_broadcast) * noise + time_broadcast * actions
        target_velocity = actions - noise
        timesteps = (time * self.config.num_timestep_buckets).long()

        tokens = self._encode_tokens(noisy_actions, timesteps, state)
        hidden = self._layerwise_dit(tokens, timesteps, vl_embs_list, encoder_attention_mask)
        predicted_velocity = self.action_decoder(hidden[:, -actions.shape[1] :])
        losses = functional.mse_loss(predicted_velocity, target_velocity, reduction="none")
        if action_mask is not None:
            losses = losses * action_mask
            return losses.sum(dim=(1, 2)) / action_mask.sum(dim=(1, 2)).clamp_min(1.0)
        return losses.mean(dim=(1, 2))

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list[torch.Tensor],
        state: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_hidden_layers(vl_embs_list)
        base_hidden = vl_embs_list[-1]
        actions = noise if noise is not None else torch.randn(
            base_hidden.shape[0], self.action_horizon, self.action_dim,
            device=base_hidden.device, dtype=base_hidden.dtype,
        )
        action_mask = self._prepare_action_mask(action_mask, actions)
        if action_mask is not None:
            actions = actions * action_mask

        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            timestep = int(step / self.num_inference_timesteps * self.config.num_timestep_buckets)
            timesteps = torch.full((actions.shape[0],), timestep, device=actions.device, dtype=torch.long)
            tokens = self._encode_tokens(actions, timesteps, state)
            hidden = self._layerwise_dit(tokens, timesteps, vl_embs_list, encoder_attention_mask)
            velocity = self.action_decoder(hidden[:, -self.action_horizon :])
            if action_mask is not None:
                velocity = velocity * action_mask
            actions = actions + dt * velocity
            if action_mask is not None:
                actions = actions * action_mask
        return actions
