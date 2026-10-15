"""Benchmark StarVLA-PI eager/graph inference on a checkpoint and HDF5 frames."""
from pathlib import Path
import argparse
import json
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


@torch.no_grad()
def benchmark(policy, observations, iterations=20, warmup=3):
    model, device = policy.model, policy.device
    batches = [policy.encode_obs(obs) for obs in observations]
    config = model.config
    report = {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
              "layerwise_num_layers": model.layerwise_num_layers,
              "chunk_size": config.chunk_size, "denoise_steps": model.action_head.num_inference_timesteps,
              "tactile_mode": config.tactile_mode, "tactile_type": config.tactile_type,
              "tactile_insert_location": config.tactile_insert_location,
              "timing": "synchronized wall time; alternating AB/BA; capture excluded"}
    model.clear_cuda_graphs()
    config.use_cuda_graph = False
    layers, _ = model._encode_prefix_layers(batches[0])
    noise_shape = (layers[-1].shape[0], config.chunk_size, model.action_dim)
    noise = torch.randn(noise_shape, device=device, dtype=layers[-1].dtype)
    del layers
    config.use_cuda_graph = True
    _, report["capture_and_first_replay_ms"] = timed(lambda: model.predict_action_chunk(batches[0], noise=noise), device)
    checks = []
    for i, batch in enumerate(batches):
        for seed in (0, 1, 42):
            torch.manual_seed(seed)
            fixed = torch.randn(noise_shape, device=device, dtype=noise.dtype)
            config.use_cuda_graph = False
            eager = model.predict_action_chunk(batch, noise=fixed)
            config.use_cuda_graph = True
            rng = torch.cuda.get_rng_state(device)
            graph = model.predict_action_chunk(batch, noise=fixed)
            assert torch.equal(rng, torch.cuda.get_rng_state(device)), "Capture/replay consumed RNG"
            torch.testing.assert_close(graph, eager, rtol=0, atol=0)
            checks.append({"observation": i, "seed": seed, "max_abs": (graph-eager).abs().max().item()})
    retained = graph.clone()
    model.predict_action_chunk(batches[0])
    torch.testing.assert_close(graph, retained, rtol=0, atol=0)
    config.use_cuda_graph = False
    torch.manual_seed(99)
    eager = model.predict_action_chunk(batches[0])
    eager_rng = torch.cuda.get_rng_state(device)
    config.use_cuda_graph = True
    torch.manual_seed(99)
    graph = model.predict_action_chunk(batches[0])
    torch.testing.assert_close(graph, eager, rtol=0, atol=0)
    assert torch.equal(eager_rng, torch.cuda.get_rng_state(device)), "RNG progression changed"
    assert not torch.equal(graph, model.predict_action_chunk(batches[0])), "Noise frozen"
    report["comparisons"] = checks
    report["owned_output"] = report["same_rng_progression"] = report["fresh_noise"] = True
    report["latency"] = {}
    for scope in ("predict_action_chunk", "deployment_predict_chunk"):
        samples = {"eager": [], "graph": []}
        for enabled in (False, True):
            config.use_cuda_graph = enabled
            for _ in range(warmup):
                if scope == "predict_action_chunk":
                    model.predict_action_chunk(batches[0])
                else:
                    policy.predict_chunk(observations[0])
        for i in range(iterations):
            for enabled in ((False, True) if i % 2 == 0 else (True, False)):
                config.use_cuda_graph = enabled
                index = i % len(batches)
                fn = (lambda: model.predict_action_chunk(batches[index])) if scope == "predict_action_chunk" else (lambda: policy.predict_chunk(observations[index]))
                _, elapsed = timed(fn, device)
                samples["graph" if enabled else "eager"].append(elapsed)
        row = {name: {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values,95))}
               for name,values in samples.items()}
        row["speedup"] = row["eager"]["median_ms"] / row["graph"]["median_ms"]
        report["latency"][scope] = row
    report["graphs"] = {}
    for name in ("_chunk_cuda_graph", "_action_cuda_graph", "_layerwise_prefix_cuda_graph", "_layerwise_text_cuda_graph"):
        runner = getattr(model, name)
        if runner is not None:
            report["graphs"][name] = {k: getattr(runner,k) for k in ("captures","replays","eager_calls")}
    config.use_cuda_graph = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--task", default="lift_can")
    parser.add_argument("--frames", default="0,10,20")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 4 or args.warmup < 1:
        parser.error("Use at least 4 iterations and 1 warmup")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    policy = Policy({"checkpoint": args.checkpoint, "task_name": args.task, "device": str(device)})
    if policy.kind != "starvla_pi":
        parser.error("Expected a starvla_pi checkpoint")
    policy.model.config.cuda_graph_vlm = not args.head_only
    policy.model.config.cuda_graph_full_vlm = not args.text_only
    observations = [hdf5_observation(args.episode, int(frame), policy.run) for frame in args.frames.split(",")]
    report = benchmark(policy, observations, args.iterations, args.warmup)
    report.update(checkpoint=str(policy.run["checkpoint"]), episode=args.episode, frames=args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
