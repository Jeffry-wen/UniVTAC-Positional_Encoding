#!/usr/bin/env python

"""Configuration for the StarVLA Qwen-PI layer-wise flow-matching policy."""

from dataclasses import dataclass, field

from policy.vla_common.engine.configs import PreTrainedConfig
from policy.vla_common.engine.configs import NormalizationMode
from policy.starvla_groot.configuration_starvla_groot import StarvlaGrootConfig


@PreTrainedConfig.register_subclass("starvla_pi")
@dataclass
class StarvlaPiConfig(StarvlaGrootConfig):
    """Qwen-VL plus one layer-wise cross-attention DiT block per selected text layer.

    Sensor routing and preprocessing reuse the repository StarVLA interfaces.
    State and action use quantile normalization before internal action padding.
    """

    # PI0.5-compatible internal action width. Dataset actions remain their real
    # width (8D for UniVTAC joints) and are padded only after normalizing.
    base_vlm: str = "resources/pretrained_models/Qwen3.5-2B"
    use_cuda_graph: bool = True
    cuda_graph_vlm: bool = True
    cuda_graph_full_vlm: bool = True
    cuda_graph_full_chunk: bool = True
    max_action_dim: int = 32

    # None means all text-transformer layers. Qwen3.5-2B therefore selects 24
    # hidden states and builds 24 matching DiT blocks.
    layerwise_num_layers: int | None = None
    legacy_layerwise_layout: bool = False  # Compatibility with historical selected-layer checkpoints
    layerwise_attention_head_dim: int = 64
    layerwise_dropout: float = 0.0
    layerwise_final_dropout: bool = False
    layerwise_activation_fn: str = "geglu"

    # Match PI0.5 preprocessing for a controlled architecture comparison.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # PI0.5 uses ten Euler denoising steps and no inference-time noise ensemble.
    num_inference_timesteps: int = 10
    repeated_diffusion_steps: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.max_action_dim < 1:
            raise ValueError("max_action_dim must be positive.")
        if self.layerwise_num_layers is not None and self.layerwise_num_layers < 1:
            raise ValueError("layerwise_num_layers must be positive when provided.")
        if self.layerwise_attention_head_dim < 1:
            raise ValueError("layerwise_attention_head_dim must be positive.")
        if not 0.0 <= self.layerwise_dropout < 1.0:
            raise ValueError("layerwise_dropout must be in [0, 1).")
        if self.num_inference_timesteps < 1:
            raise ValueError("num_inference_timesteps must be positive.")
        if self.repeated_diffusion_steps != 1:
            raise ValueError("StarvlaPi supports repeated_diffusion_steps=1 only.")
