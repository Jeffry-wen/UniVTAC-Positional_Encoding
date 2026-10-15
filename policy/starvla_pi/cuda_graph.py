"""CUDA Graph helpers for selected Qwen layers and layer-wise denoising.

Reuse the bounded cache, input refresh and owned-output semantics of GR00T.
Selected text states are stacked into a single [layers, batch, tokens, hidden]
tensor, so graph replay retains every conditioning layer.
"""
import torch

from policy.starvla_groot.cuda_graph import (
    TensorCudaGraph, Qwen35TextCudaGraph, Qwen35PrefixCudaGraph, _PrefixPlan,
)


class LayerwiseQwen35TextCudaGraph(Qwen35TextCudaGraph):
    def __init__(self, text_model, num_layers, warmup_steps=3, max_graphs=2):
        super().__init__(text_model, warmup_steps, max_graphs)
        self.num_layers = num_layers

    def _forward_on_device(self, embeds, positions, causal_mask, linear_mask):
        model = self.module
        if positions.ndim == 2:
            positions = positions[None, ...].expand(4, positions.shape[0], -1)
        if positions.shape[0] == 4:
            text_positions, positions = positions[0], positions[1:]
        else:
            text_positions = None
        position_embeddings = model.rotary_emb(embeds, positions)
        hidden, selected = embeds, []
        count = model.config.num_hidden_layers
        for i, layer in enumerate(model.layers[:count]):
            mask = linear_mask if model.config.layer_types[i] == "linear_attention" else causal_mask
            hidden = layer(
                hidden, position_embeddings=position_embeddings,
                attention_mask=mask, position_ids=text_positions,
                past_key_values=None, use_cache=False,
            )
            # HF hidden_states ends with final norm, replacing the raw last
            # layer state. Omit the embedding state and preserve this ordering.
            if count - self.num_layers <= i < count - 1:
                selected.append(hidden)
        selected.append(model.norm(hidden))
        return torch.stack(selected)


class LayerwiseQwen35PrefixCudaGraph(Qwen35PrefixCudaGraph):
    def __init__(self, qv, num_layers, warmup_steps=3, max_graphs=2):
        def make_plan(qv, images, instructions, extra):
            return _PrefixPlan(
                qv, images, instructions, extra,
                text_compute=LayerwiseQwen35TextCudaGraph(qv.model.model.language_model, num_layers),
            )
        super().__init__(qv, warmup_steps, max_graphs, plan_factory=make_plan)


class LayerwiseActionCudaGraph(TensorCudaGraph):
    def __call__(self, hidden_layers, state, mask, action_mask, noise):
        return super().__call__(
            hidden_layers, state, mask, action_mask, noise,
            extra_key=self.module.num_inference_timesteps,
        )
