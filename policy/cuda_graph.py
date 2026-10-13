"""Bounded, deterministic CUDA Graph inference for tensor pytrees.

One policy instance is used on one inference thread/stream at a time. Callers
sample noise outside capture, so warmup/replay never consumes their RNG stream.
CPU inputs, training and excess shape variants use the eager implementation.
"""
from dataclasses import dataclass

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, treespec_dumps


@dataclass
class _Entry:
    graph: torch.cuda.CUDAGraph
    leaves: list
    output: torch.Tensor


class InferenceCudaGraph:
    def __init__(self, module, function, warmup_steps=3, max_graphs=2):
        if warmup_steps < 1 or max_graphs < 1:
            raise ValueError("CUDA Graph warmup_steps and max_graphs must be positive")
        self.module, self.function = module, function
        self.warmup_steps, self.max_graphs = warmup_steps, max_graphs
        self.entries = {}
        self.captures = self.replays = self.eager_calls = 0

    def clear(self):
        self.entries.clear()

    @torch.no_grad()
    def __call__(self, *args):
        leaves, spec = tree_flatten(args)
        tensors = [t for t in leaves if isinstance(t, torch.Tensor)]
        if self.module.training or not tensors or any(not t.is_cuda for t in tensors):
            self.eager_calls += 1
            return self.function(*args)
        device = tensors[0].device
        if any(t.device != device for t in tensors):
            self.eager_calls += 1
            return self.function(*args)
        signature = tuple((tuple(t.shape), t.dtype, t.device) if isinstance(t, torch.Tensor)
                          else (type(t), t) for t in leaves)
        key = (treespec_dumps(spec), signature, torch.is_autocast_enabled("cuda"),
               torch.get_autocast_dtype("cuda"), torch.is_inference_mode_enabled(),
               torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32)
        entry = self.entries.get(key)
        if entry is None and len(self.entries) >= self.max_graphs:
            self.eager_calls += 1
            return self.function(*args)
        with torch.cuda.device(device):
            if entry is None:
                static = [t.clone() if isinstance(t, torch.Tensor) else t for t in leaves]
                static_args = tree_unflatten(static, spec)
                current = torch.cuda.current_stream(device)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    for _ in range(self.warmup_steps):
                        self.function(*static_args)
                current.wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = self.function(*static_args)
                current.wait_stream(stream)
                entry = _Entry(graph, static, output)
                self.entries[key] = entry
                self.captures += 1
                print(f"CUDA Graph captured: {type(self.module).__name__} "
                      f"(shape variants={len(self.entries)}/{self.max_graphs})", flush=True)
            for dest, source in zip(entry.leaves, leaves, strict=True):
                if isinstance(dest, torch.Tensor):
                    dest.copy_(source)
            entry.graph.replay()
            self.replays += 1
            # Queued actions/retained outputs must survive the next replay.
            return entry.output.clone()


class CudaGraphPolicyMixin:
    """Invalidate captures before parameter replacement, device moves or training."""
    def clear_cuda_graphs(self):
        runner = getattr(self, "_action_cuda_graph", None)
        if runner is not None:
            runner.clear()

    def train(self, mode=True):
        if mode:
            self.clear_cuda_graphs()
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self.clear_cuda_graphs()
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.clear_cuda_graphs()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def attention_block_mask(values, *, dtype, device):
    """Materialize Python 0/1 block metadata using GPU fills, without host copies."""
    from itertools import groupby
    mask = torch.zeros(len(values), dtype=dtype, device=device)
    offset = 0
    for value, group in groupby(values):
        count = sum(1 for _ in group)
        if value:
            mask[offset:offset + count] = value
        offset += count
    return mask
