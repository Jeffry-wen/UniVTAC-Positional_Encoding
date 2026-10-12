"""Bounded CUDA Graph caches for GR00T denoising and Qwen3.5 text compute.

Noise is sampled by the caller, outside capture. Each invocation copies fresh
conditioning and noise into stable buffers and returns an owned output tensor.
One policy instance must be used from one inference thread/stream at a time.

The full-prefix and full-chunk capture boundaries follow fast_vlm.py and
fast_dit.py from the local lerobot-v0.5.1-test-cuda_graph reference. Dimensions,
tactile inputs, masks, noise, and output ownership follow UniVTAC's contract.
"""
from dataclasses import dataclass

import torch


@dataclass
class _Entry:
    graph: torch.cuda.CUDAGraph
    inputs: tuple
    output: torch.Tensor


class TensorCudaGraph:
    def __init__(self, module, function, warmup_steps=3, max_graphs=2):
        self.module = module
        self.function = function
        self.warmup_steps = warmup_steps
        self.max_graphs = max_graphs
        self.entries = {}
        self.captures = 0
        self.replays = 0
        self.eager_calls = 0

    def clear(self):
        self.entries.clear()

    @staticmethod
    def _signature(tensor):
        if tensor is None:
            return None
        return (tuple(tensor.shape), tensor.dtype, tensor.device)

    @torch.no_grad()
    def __call__(self, *inputs, extra_key=None):
        hidden = inputs[0]
        key = (
            tuple(self._signature(t) for t in inputs),
            extra_key,
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
            torch.is_inference_mode_enabled(),
        )
        entry = self.entries.get(key)
        if self.module.training or not hidden.is_cuda or (
            entry is None and len(self.entries) >= self.max_graphs
        ):
            self.eager_calls += 1
            return self.function(*inputs)

        with torch.cuda.device(hidden.device):
            if entry is None:
                static = tuple(t.clone() if t is not None else None for t in inputs)
                # Warm up on a side stream before capture (cuBLAS/SDPA lazy init).
                current = torch.cuda.current_stream(hidden.device)
                stream = torch.cuda.Stream(device=hidden.device)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    for _ in range(self.warmup_steps):
                        self.function(*static)
                current.wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = self.function(*static)
                current.wait_stream(stream)
                entry = _Entry(graph, static, output)
                self.entries[key] = entry
                self.captures += 1
            for dest, source in zip(entry.inputs, inputs, strict=True):
                if dest is not None:
                    dest.copy_(source)
            entry.graph.replay()
            self.replays += 1
            # select_action queues views; returning static storage would silently
            # overwrite queued/retained actions on the next replay.
            return entry.output.clone()


class ActionHeadCudaGraph(TensorCudaGraph):
    def __init__(self, head, warmup_steps=3, max_graphs=2):
        super().__init__(
            head, lambda h, s, m, n: head.predict_action(h, s, m, noise=n),
            warmup_steps, max_graphs,
        )

    def __call__(self, hidden, state, mask, noise):
        return super().__call__(
            hidden, state, mask, noise, extra_key=self.module.num_inference_timesteps
        )


class Qwen35TextCudaGraph(TensorCudaGraph):
    """Capture Qwen3.5 decoder compute after eager, data-dependent mask setup.

    Mirrors Qwen3_5TextModel.forward with use_cache=False. RoPE/image metadata
    and padding decisions stay eager; all layer math stays identical. Using
    the prepared masks avoids tensor-to-Python conditionals during capture.
    """
    def __init__(self, text_model, warmup_steps=3, max_graphs=2):
        super().__init__(text_model, self._forward, warmup_steps, max_graphs)

    def _forward(self, embeds, positions, causal_mask, linear_mask):
        # The transformers torch delta-rule fallback creates its zero recurrent
        # state on CPU then calls .to(value), which is forbidden during capture.
        # A scoped default device allocates the identical zeros directly on GPU.
        with torch.device(embeds.device):
            return self._forward_on_device(embeds, positions, causal_mask, linear_mask)

    def _forward_on_device(self, embeds, positions, causal_mask, linear_mask):
        model = self.module
        if positions.ndim == 2:
            positions = positions[None, ...].expand(4, positions.shape[0], -1)
        if positions.shape[0] == 4:
            text_positions, positions = positions[0], positions[1:]
        else:
            text_positions = None
        position_embeddings = model.rotary_emb(embeds, positions)
        hidden = embeds
        for i, layer in enumerate(model.layers[:model.config.num_hidden_layers]):
            mask = linear_mask if model.config.layer_types[i] == "linear_attention" else causal_mask
            hidden = layer(
                hidden, position_embeddings=position_embeddings,
                attention_mask=mask, position_ids=text_positions,
                past_key_values=None, use_cache=False,
            )
        return model.norm(hidden)

    def __call__(self, embeds, positions, attention_mask):
        from transformers.masking_utils import create_causal_mask

        text_positions = positions if positions.ndim == 2 else (
            positions[0] if positions.shape[0] == 4 else None
        )
        causal_mask = create_causal_mask(
            config=self.module.config, inputs_embeds=embeds,
            attention_mask=attention_mask, past_key_values=None,
            position_ids=text_positions,
        )
        linear_mask = self.module._update_linear_attn_mask(attention_mask, None)
        return super().__call__(embeds, positions, causal_mask, linear_mask)


class _PrefixPlan:
    """Immutable prompt/grid metadata, prepared before capture without HF patches."""
    def __init__(self, qv, images, instructions, extra):
        from transformers.masking_utils import create_causal_mask

        self.qv = qv
        self.core = core = qv.model.model
        visual = core.visual
        batch_size, views = images.shape[:2]
        inputs = qv.build_qwenvl_inputs(images, instructions)
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        self.prompt_length = ids.shape[1]
        if extra is not None:
            filler = torch.full(
                extra.shape[:2], qv.processor.tokenizer.pad_token_id or 0,
                dtype=ids.dtype, device=ids.device,
            )
            ids = torch.cat([ids, filler], dim=1)
            mask = torch.cat([mask, torch.ones_like(filler)], dim=1)
        self.mask = mask
        self.base_embeds = core.get_input_embeddings()(ids)
        mm_types = (ids == qv.model.config.image_token_id).long()
        video_id = getattr(qv.model.config, "video_token_id", -1)
        if video_id >= 0:
            mm_types[ids == video_id] = 2
        self.positions = core.get_rope_index(
            ids, mm_types, image_grid_thw=inputs["image_grid_thw"], attention_mask=mask,
        )[0]
        # Use every sample's actual indices; instructions may differ within a batch.
        self.image_indices = (ids.reshape(-1) == qv.model.config.image_token_id).nonzero().flatten()
        grid = inputs["image_grid_thw"].cpu()
        self.vision_positions = visual.fast_pos_embed_interpolate(grid)
        rotary = visual.rot_pos_emb(grid)
        rotary = rotary.reshape(rotary.shape[0], -1)
        rotary = torch.cat([rotary, rotary], dim=-1)
        self.vision_rope = (rotary.cos(), rotary.sin())
        # SDPA vision consumes this metadata using .tolist(); keep it on CPU.
        lengths = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(0, dtype=torch.int32)
        self.cu_seqlens = torch.nn.functional.pad(lengths, (1, 0))
        self.mean = images.new_tensor(qv._img_mean).view(1, 3, 1, 1)
        self.std = images.new_tensor(qv._img_std).view(1, 3, 1, 1)
        self.batch_size, self.views = batch_size, views
        self.causal_mask = create_causal_mask(
            config=core.language_model.config, inputs_embeds=self.base_embeds,
            attention_mask=mask, past_key_values=None, position_ids=None,
        )
        self.linear_mask = core.language_model._update_linear_attn_mask(mask, None)
        self.text_compute = Qwen35TextCudaGraph(core.language_model)

    def forward(self, images, extra):
        qv, visual = self.qv, self.core.visual
        n, c, height, width = self.batch_size * self.views, 3, images.shape[-2], images.shape[-1]
        p, m, tp = qv._patch_size, qv._merge_size, qv._temporal_patch_size
        gh, gw = height // p, width // p
        x = (images.reshape(n, c, height, width) - self.mean) / self.std
        x = x.unsqueeze(1).repeat(1, tp, 1, 1, 1)
        x = x.reshape(n, 1, tp, c, gh // m, m, p, gw // m, m, p)
        pixel_values = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(n * gh * gw, c * tp * p * p)
        hidden = visual.patch_embed(pixel_values) + self.vision_positions
        for block in visual.blocks:
            hidden = block(hidden, cu_seqlens=self.cu_seqlens, position_embeddings=self.vision_rope)
        image_embeds = visual.merger(hidden).to(self.base_embeds.dtype)
        embeds = self.base_embeds.clone()
        embeds.reshape(-1, embeds.shape[-1]).index_copy_(0, self.image_indices, image_embeds)
        if extra is not None:
            embeds[:, self.prompt_length:] = extra.to(embeds.dtype)
        return self.text_compute._forward(embeds, self.positions, self.causal_mask, self.linear_mask)


class Qwen35PrefixCudaGraph:
    """FastVLM-style patchify/vision/text graph with dynamic tactile embeddings.

    Each prompt tuple owns its metadata and graph. Unlike the reference's
    class-wide monkey patches, constants cannot leak between policy instances.
    """
    def __init__(self, qv, warmup_steps=3, max_graphs=2):
        self.qv = qv
        self.warmup_steps, self.max_graphs = warmup_steps, max_graphs
        self.entries = {}
        self.captures = 0
        self.replays = 0
        self.eager_calls = 0

    def clear(self):
        self.entries.clear()

    @torch.no_grad()
    def __call__(self, images, instructions, extra):
        sig = TensorCudaGraph._signature
        key = (tuple(instructions), sig(images), sig(extra), torch.is_inference_mode_enabled())
        if images.shape[-2:] != (self.qv._target_h, self.qv._target_w):
            self.eager_calls += 1
            return None
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_graphs:
                self.eager_calls += 1
                return None
            plan = _PrefixPlan(self.qv, images, instructions, extra)
            runner = TensorCudaGraph(self.qv, plan.forward, self.warmup_steps, max_graphs=1)
            entry = (plan, runner)
            self.entries[key] = entry
        plan, runner = entry
        captures_before = runner.captures
        output = runner(images, extra)
        self.captures += runner.captures - captures_before
        self.replays += 1
        return output, plan.mask.to(dtype=torch.bool).clone()
