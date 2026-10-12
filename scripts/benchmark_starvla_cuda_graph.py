"""Benchmark eager/CUDA Graph StarVLA on real checkpoint + HDF5 observations.

Run from the repository root with the inference environment. Latencies use CUDA
synchronization and include Python dispatch; capture is reported separately.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.unified_deploy import Policy
from scripts.test_policy import hdf5_observation


def timed(fn, device):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = fn()
    torch.cuda.synchronize(device)
    return output, (time.perf_counter() - start) * 1000


def summary(values):
    return {"mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "samples": len(values)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--task", default="lift_can")
    parser.add_argument("--frames", default="0,10,20")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--head-only", action="store_true", help="Capture only the action head")
    parser.add_argument("--previous-graph", action="store_true", help="Use earlier text-only/split-head graphs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 4 or args.warmup < 1:
        parser.error("Use at least 4 iterations and 1 warmup")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(123)
    policy = Policy({"checkpoint": args.checkpoint, "task_name": args.task,
                     "device": str(device)})
    model = policy.model
    if policy.kind != "starvla_groot":
        raise ValueError("Expected a StarVLA checkpoint")
    model.config.cuda_graph_vlm = not args.head_only
    model.config.cuda_graph_full_vlm = not args.previous_graph
    model.config.cuda_graph_full_chunk = not args.previous_graph
    frames = [int(v) for v in args.frames.split(",")]
    observations = [hdf5_observation(args.episode, frame, policy.run) for frame in frames]
    batches = [policy.encode_obs(obs) for obs in observations]
    report = {"checkpoint": str(policy.run["checkpoint"]), "step": policy.run["step"],
              "episode": str(Path(args.episode).resolve()), "frames": frames,
              "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
              "python": platform.python_version(), "batch_size": 1,
              "chunk_size": model.config.chunk_size,
              "repeated_diffusion_steps": model.config.repeated_diffusion_steps,
              "denoise_steps": model.config.num_inference_timesteps,
              "tactile_mode": model.config.tactile_mode,
              "tactile_insert_location": model.config.tactile_insert_location,
              "scope": "action_head" if args.head_only else (
                  "action_head + Qwen3.5 text decoder" if args.previous_graph else "full VLM + full action chunk"
              ),
              "timing": "synchronized wall time; ABBA interleaving; capture excluded from steady state"}
    hidden, mask = model._encode_prefix(batches[0])
    reps = max(1, model.config.repeated_diffusion_steps)
    state = model._get_state(batches[0], hidden.device, hidden.dtype)
    h = hidden.repeat(reps, 1, 1)
    s = state.repeat(reps, 1, 1) if state is not None else None
    m = mask.repeat(reps, 1) if mask is not None else None
    noise = torch.randn((h.shape[0], model.config.chunk_size, model.action_dim),
                        device=device, dtype=h.dtype)
    report["hidden_shape"] = list(h.shape)
    report["hidden_dtype"] = str(h.dtype)
    report["head_parameter_dtype"] = str(next(model.action_head.parameters()).dtype)
    print(json.dumps(report, indent=2), flush=True)
    # Match the policy's existing action-head precision context exactly.
    @torch.autocast("cuda", dtype=torch.float32)
    def eager_head():
        return model.action_head.predict_action(h, s, m, noise=noise)

    @torch.autocast("cuda", dtype=torch.float32)
    def graph_head():
        return model._action_cuda_graph(h, s, m, noise)
    for _ in range(args.warmup):
        eager_head()
    torch.cuda.synchronize(device)
    memory_before = torch.cuda.memory_allocated(device)
    _, capture_ms = timed(graph_head, device)
    report["head_capture_and_first_replay_ms"] = capture_ms
    report["graph_extra_allocated_MiB"] = (torch.cuda.memory_allocated(device) - memory_before) / 2**20
    model.config.use_cuda_graph = True
    _, first_chunk_ms = timed(lambda: model.predict_action_chunk(batches[0], noise=noise), device)
    report["first_chunk_with_vlm_capture_ms"] = first_chunk_ms
    report["graph_extra_allocated_MiB"] = (torch.cuda.memory_allocated(device) - memory_before) / 2**20

    # Identical noise controls the stochastic flow-matching sampler. Capture
    # must leave the default RNG untouched; regular inference must advance it.
    checks, artifacts = [], []
    for frame, batch in zip(frames, batches, strict=True):
        for seed in (0, 1, 42):
            torch.manual_seed(seed)
            fixed_noise = torch.randn(noise.shape, device=device, dtype=noise.dtype)
            model.config.use_cuda_graph = False
            eager = model.predict_action_chunk(batch, noise=fixed_noise)
            model.config.use_cuda_graph = True
            rng_before = torch.cuda.get_rng_state(device)
            graph = model.predict_action_chunk(batch, noise=fixed_noise)
            assert torch.equal(rng_before, torch.cuda.get_rng_state(device)), "Graph consumed RNG"
            torch.testing.assert_close(graph, eager, atol=1e-5, rtol=1e-5)
            delta = (graph - eager).float()
            joint_delta = delta * delta.new_tensor(policy.stats["action"]["std"])
            checks.append({"frame": frame, "seed": seed,
                           "max_abs": delta.abs().max().item(),
                           "mean_abs": delta.abs().mean().item(),
                           "rmse": delta.square().mean().sqrt().item(),
                           "joint_max_abs": joint_delta.abs().max().item(),
                           "exact_equal": torch.equal(eager, graph)})
            artifacts.append({"frame": frame, "seed": seed,
                              "eager": eager.cpu(), "cuda_graph": graph.cpu()})
    # Replay must not mutate a previously returned chunk.
    retained = graph_head()
    saved = retained.clone()
    with torch.autocast("cuda", dtype=torch.float32):
        changed = model._action_cuda_graph(h + 0.25, s, m, noise + 0.5)
    torch.testing.assert_close(retained, saved, rtol=0, atol=0)
    assert not torch.equal(changed, retained), "Replay ignored changed inputs"
    model.config.use_cuda_graph = False
    torch.manual_seed(991)
    eager_random = model.predict_action_chunk(batches[0])
    eager_rng = torch.cuda.get_rng_state(device)
    model.config.use_cuda_graph = True
    torch.manual_seed(991)
    graph_random = model.predict_action_chunk(batches[0])
    torch.testing.assert_close(eager_random, graph_random, atol=1e-5, rtol=1e-5)
    assert torch.equal(eager_rng, torch.cuda.get_rng_state(device)), "RNG sequence changed"
    another = model.predict_action_chunk(batches[0])
    assert not torch.equal(another, graph_random), "Noise was frozen by capture"
    report["output_comparison"] = checks
    report["checks"] = {"owned_output": True, "changed_input": True,
                        "fresh_random_noise": True, "same_rng_progression": True,
                        "same_noise_allclose_1e-5": True}

    # Alternate eager, graph, graph, eager to reduce clock/thermal order bias.
    latencies = {}
    for scope in ("action_head", "predict_action_chunk", "deployment_predict_chunk"):
        samples = {"eager": [], "cuda_graph": []}
        for enabled in (False, True):
            model.config.use_cuda_graph = enabled
            for _ in range(args.warmup):
                if scope == "action_head":
                    (graph_head if enabled else eager_head)()
                elif scope == "predict_action_chunk":
                    model.predict_action_chunk(batches[0])
                else:
                    policy.predict_chunk(observations[0])
        for i in range(args.iterations):
            for enabled in ((False, True) if i % 2 == 0 else (True, False)):
                model.config.use_cuda_graph = enabled
                if scope == "action_head":
                    fn = graph_head if enabled else eager_head
                elif scope == "predict_action_chunk":
                    fn = lambda: model.predict_action_chunk(batches[i % len(batches)])
                else:
                    fn = lambda: policy.predict_chunk(observations[i % len(observations)])
                _, elapsed = timed(fn, device)
                samples["cuda_graph" if enabled else "eager"].append(elapsed)
        entry = {name: summary(values) for name, values in samples.items()}
        entry["speedup"] = entry["eager"]["median_ms"] / entry["cuda_graph"]["median_ms"]
        entry["raw_ms"] = samples
        latencies[scope] = entry
        print(scope, json.dumps({k: v for k, v in entry.items() if k != "raw_ms"}), flush=True)
    report["latency"] = latencies
    report["graph_stats"] = {"captures": model._action_cuda_graph.captures,
                             "replays": model._action_cuda_graph.replays,
                             "eager_fallbacks": model._action_cuda_graph.eager_calls}
    if model.qwen_vl._text_cuda_graph is not None:
        text_graph = model.qwen_vl._text_cuda_graph
        report["text_graph_stats"] = {"captures": text_graph.captures, "replays": text_graph.replays,
                                      "eager_fallbacks": text_graph.eager_calls}
    for name, manager in (("chunk_graph_stats", model._chunk_cuda_graph),
                          ("prefix_graph_stats", model.qwen_vl._prefix_cuda_graph)):
        if manager is not None:
            report[name] = {"captures": manager.captures, "replays": manager.replays,
                            "eager_fallbacks": manager.eager_calls}
    assert report["graph_stats"]["replays"] > 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts, args.output.with_suffix(".outputs.pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
