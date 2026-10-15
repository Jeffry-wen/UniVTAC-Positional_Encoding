#!/usr/bin/env python

"""LeRobot policy wrapper for StarVLA's Qwen-PI architecture."""

import importlib.util
from contextlib import nullcontext

import torch
import torch.nn.functional as functional
from torch import Tensor

from policy.vla_common.pretrained import PreTrainedPolicy
from policy.starvla_groot.modeling_starvla_groot import StarvlaGrootPolicy
from policy.starvla_groot.qwen_vl_interface import QwenVLInterface
from policy.vla_common.engine.utils.constants import ACTION, OBS_STATE

from policy.vla_common.tactile_encode import TactileEncoder

from .configuration_starvla_pi import StarvlaPiConfig
from .cuda_graph import LayerwiseActionCudaGraph, LayerwiseQwen35TextCudaGraph, LayerwiseQwen35PrefixCudaGraph
from .layerwise_flow_matching_head import (
    LayerwiseActionHeadConfig,
    LayerwiseFlowmatchingActionHead,
)


class StarvlaPiPolicy(StarvlaGrootPolicy):
    """Qwen-VL with layer-wise hidden-state injection into a flow action expert."""

    config_class = StarvlaPiConfig
    name = "starvla_pi"

    def __init__(self, config: StarvlaPiConfig, **kwargs):
        if importlib.util.find_spec("transformers") is None:
            raise ImportError("StarvlaPi requires transformers and the starvla_groot dependencies.")
        # Skip StarvlaGrootPolicy.__init__: Qwen-PI constructs a different action head.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        torch.backends.cuda.matmul.allow_tf32 = False

        self.original_action_dim = int(config.output_features[ACTION].shape[0])
        if self.original_action_dim > config.max_action_dim:
            raise ValueError(
                f"Dataset action dim {self.original_action_dim} exceeds max_action_dim "
                f"{config.max_action_dim}."
            )
        self.action_dim = int(config.max_action_dim)
        if config.state_mode == "none":
            self.state_dim = 0
        elif config.state_dim is not None:
            self.state_dim = int(config.state_dim)
        elif OBS_STATE in config.input_features:
            self.state_dim = int(config.input_features[OBS_STATE].shape[0])
        else:
            self.state_dim = 0

        load_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        self.qwen_vl = QwenVLInterface(
            base_vlm=config.base_vlm,
            from_config=config.vlm_from_config,
            attn_implementation=config.attn_implementation,
            dtype=load_dtype,
            image_resolution=config.image_resolution,
            batch_vision_attention=config.training_batch_vision_attention,
        )
        text_config = getattr(self.qwen_vl.model.config, "text_config", self.qwen_vl.model.config)
        available_layers = int(text_config.num_hidden_layers)
        self.layerwise_num_layers = int(config.layerwise_num_layers or available_layers)
        if not config.legacy_layerwise_layout and self.layerwise_num_layers != available_layers:
            raise ValueError("PI action layer count must equal the VLM text layer count")
        if self.layerwise_num_layers > available_layers:
            raise ValueError(
                f"layerwise_num_layers={self.layerwise_num_layers} exceeds the Qwen backbone's "
                f"{available_layers} text layers."
            )

        self.action_head = LayerwiseFlowmatchingActionHead(
            LayerwiseActionHeadConfig(
                hidden_size=config.action_expert_hidden_size or self.qwen_vl.hidden_size,
                cross_attention_dim=self.qwen_vl.hidden_size,
                num_layers=self.layerwise_num_layers,
                attention_head_dim=config.layerwise_attention_head_dim,
                action_dim=self.action_dim,
                state_dim=self.state_dim,
                action_horizon=config.chunk_size,
                num_inference_timesteps=config.num_inference_timesteps,
                num_target_vision_tokens=config.num_target_vision_tokens,
                add_pos_embed=config.add_pos_embed,
                max_seq_len=config.max_seq_len,
                noise_beta_alpha=config.noise_beta_alpha,
                noise_beta_beta=config.noise_beta_beta,
                noise_s=config.noise_s,
                num_timestep_buckets=config.num_timestep_buckets,
                dropout=config.layerwise_dropout,
                final_dropout=config.layerwise_final_dropout,
                activation_fn=config.layerwise_activation_fn,
            )
        )
        self._action_cuda_graph = LayerwiseActionCudaGraph(
            self.action_head, self._denoise_layers,
            config.cuda_graph_warmup_steps, config.cuda_graph_max_graphs,
        )
        self._chunk_cuda_graph = LayerwiseActionCudaGraph(
            self.action_head, self._denoise_layerwise_chunk,
            config.cuda_graph_warmup_steps, config.cuda_graph_max_graphs,
        )
        self._layerwise_text_cuda_graph = None
        self._layerwise_prefix_cuda_graph = None
        if self.qwen_vl._text_cuda_graph is not None:
            self._layerwise_text_cuda_graph = LayerwiseQwen35TextCudaGraph(
                self.qwen_vl.model.model.language_model, self.layerwise_num_layers,
                config.cuda_graph_warmup_steps, config.cuda_graph_max_graphs,
            )
        if self.qwen_vl._prefix_cuda_graph is not None:
            self._layerwise_prefix_cuda_graph = LayerwiseQwen35PrefixCudaGraph(
                self.qwen_vl, self.layerwise_num_layers,
                config.cuda_graph_warmup_steps, config.cuda_graph_max_graphs,
            )
        self.tactile_encoder = (
            TactileEncoder(config, self.qwen_vl.hidden_size) if config.tactile_mode == "encode" else None
        )
        if self.tactile_encoder is not None and config.tactile_insert_location == "encoder":
            if not self.qwen_vl.supports_prefix_injection():
                raise ValueError("Encoder tactile injection requires a Qwen3.5-compatible backbone.")
        self._set_requires_grad()
        if config.gradient_checkpointing:
            self.qwen_vl.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.to(config.device)
        self.reset()

    def clear_cuda_graphs(self):
        for name in ("_action_cuda_graph", "_chunk_cuda_graph",
                     "_layerwise_text_cuda_graph", "_layerwise_prefix_cuda_graph"):
            runner = getattr(self, name, None)
            if runner is not None:
                runner.clear()
        self.qwen_vl.clear_cuda_graphs()

    def _denoise_layers(self, stacked, state, mask, action_mask, noise):
        return self.action_head.predict_action(
            list(stacked.unbind(0)), state, encoder_attention_mask=mask,
            action_mask=action_mask, noise=noise,
        )

    def _denoise_layerwise_chunk(self, stacked, state, mask, action_mask, noise):
        return self._denoise_layers(stacked, state, mask, action_mask, noise)[..., :self.original_action_dim].float()

    def _encode_prefix_layers(self, batch: dict[str, Tensor]):
        self.qwen_vl.use_cuda_graph = False
        graph_vlm = (self.config.use_cuda_graph and self.config.cuda_graph_vlm
                     and not self.training and not torch.is_grad_enabled())
        images = self._build_images(batch)
        graph_vlm = graph_vlm and images.is_cuda
        instructions = self._get_instructions(batch, len(images))
        tactile = None
        if self.tactile_encoder is not None:
            with self._vlm_autocast(images.device):
                tactile = self.tactile_encoder.forward_flat(batch)
        extra = tactile if self.config.tactile_insert_location == "encoder" else None
        encoded = None
        if graph_vlm and self.config.cuda_graph_full_vlm and self._layerwise_prefix_cuda_graph is not None:
            with self._vlm_autocast(images.device):
                encoded = self._layerwise_prefix_cuda_graph(images, instructions, extra)
        if encoded is not None:
            all_layers, attention_mask = encoded
        elif extra is not None:
            all_layers, attention_mask = self.qwen_vl.forward_with_prefix_tokens(
                images, instructions, tactile, return_hidden_states=True,
                text_graph=self._layerwise_text_cuda_graph if graph_vlm else None
            )
        else:
            inputs = self.qwen_vl.build_qwenvl_inputs(images=images, instructions=instructions)
            attention_mask = inputs.get("attention_mask")
            if graph_vlm and self._layerwise_text_cuda_graph is not None:
                with self._vlm_autocast(images.device):
                    all_layers = self.qwen_vl.forward_graph_features(
                        inputs, text_graph=self._layerwise_text_cuda_graph
                    )
            else:
                outputs = self.qwen_vl(
                    **inputs, output_attentions=False, output_hidden_states=True,
                    return_dict=True, use_cache=False,
                )
                all_layers = outputs.hidden_states
        hidden_layers = list(all_layers[-self.layerwise_num_layers:])
        if len(hidden_layers) != self.layerwise_num_layers:
            raise RuntimeError("Qwen hidden-state count does not match the layer-wise action head.")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=hidden_layers[-1].device, dtype=torch.bool)
        if tactile is not None and self.config.tactile_insert_location == "decoder":
            tactile = tactile.to(device=hidden_layers[-1].device, dtype=hidden_layers[-1].dtype)
            if attention_mask is None:
                attention_mask = torch.ones(hidden_layers[-1].shape[:2], device=tactile.device, dtype=torch.bool)
            hidden_layers = [torch.cat((layer, tactile), dim=1) for layer in hidden_layers]
            attention_mask = torch.cat((attention_mask, torch.ones(tactile.shape[:2], device=tactile.device, dtype=torch.bool)), dim=1)
        return hidden_layers, attention_mask

    def _action_mask_with_padded_tail(
        self,
        batch: dict[str, Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        mask = torch.ones(batch_size, self.config.chunk_size, self.original_action_dim,
                          device=device, dtype=dtype)
        # Padding in the internal action width is always excluded. Training also
        # excludes repeated tail frames at the end of a demonstration.
        if "action_mask" in batch:
            supplied = batch["action_mask"].to(device=device, dtype=dtype)
            if supplied.ndim == 2:
                supplied = supplied[:, None, :]
            mask = mask * supplied[..., :self.original_action_dim]
        if "is_pad" in batch:
            mask = mask * (~batch["is_pad"].to(device=device, dtype=torch.bool))[:, -self.config.chunk_size:, None]
        return functional.pad(mask, (0, self.action_dim - self.original_action_dim))

    @staticmethod
    def _head_autocast(device: torch.device):
        if device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float32)
        return nullcontext()

    @staticmethod
    def _vlm_autocast(device: torch.device):
        if device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean", *,
                return_loss_dict: bool = True) -> tuple[Tensor, dict]:
        hidden_layers, attention_mask = self._encode_prefix_layers(batch)
        base_hidden = hidden_layers[-1]
        with self._head_autocast(base_hidden.device):
            actions = batch[ACTION].to(device=base_hidden.device, dtype=base_hidden.dtype)
            actions = actions[:, -self.config.chunk_size :, :]
            actions = functional.pad(actions, (0, self.action_dim - actions.shape[-1]))
            state = self._get_state(batch, base_hidden.device, base_hidden.dtype)
            action_mask = self._action_mask_with_padded_tail(
                batch, base_hidden.shape[0], base_hidden.device, base_hidden.dtype
            )
            per_sample_loss = self.action_head(
                hidden_layers,
                actions,
                state,
                encoder_attention_mask=attention_mask,
                action_mask=action_mask,
            )
        loss = per_sample_loss.mean()
        value = loss.item() if return_loss_dict else None
        loss_dict = {"action_loss": value, "loss": value} if return_loss_dict else {}
        return (per_sample_loss, loss_dict) if reduction == "none" else (loss, loss_dict)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        hidden_layers, attention_mask = self._encode_prefix_layers(batch)
        base_hidden = hidden_layers[-1]
        with self._head_autocast(base_hidden.device):
            state = self._get_state(batch, base_hidden.device, base_hidden.dtype)
            action_mask = self._action_mask_with_padded_tail(
                batch, base_hidden.shape[0], base_hidden.device, base_hidden.dtype
            )
            noise = kwargs.get("noise")
            expected_shape = (base_hidden.shape[0], self.config.chunk_size, self.action_dim)
            if noise is None:
                noise = torch.randn(expected_shape, device=base_hidden.device, dtype=base_hidden.dtype)
            elif tuple(noise.shape) != expected_shape or noise.device != base_hidden.device or noise.dtype != base_hidden.dtype:
                raise ValueError(f"noise must have shape {expected_shape}, device {base_hidden.device} and dtype {base_hidden.dtype}")
            stacked = torch.stack(hidden_layers)
            if self.config.use_cuda_graph:
                runner = self._chunk_cuda_graph if self.config.cuda_graph_full_chunk else self._action_cuda_graph
                actions = runner(stacked, state, attention_mask, action_mask, noise)
            else:
                actions = self._denoise_layers(stacked, state, attention_mask, action_mask, noise)
        return actions[..., :self.original_action_dim].float()

    def _get_default_peft_targets(self) -> dict:
        return {
            "target_modules": r"(.*self_attn\.(q|v)_proj)",
            "modules_to_save": ["action_head"],
        }
