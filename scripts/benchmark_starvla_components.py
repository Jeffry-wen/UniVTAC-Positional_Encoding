"""VAE full component latency without observation preprocessing/denormalization.

Vision/text are independently captured ONLY for attribution; production fuses
them. A second benchmark retains the fused prefix boundary, with already
prepared pixel values as input, to measure the complete model-only pipeline.
The tactile encoder stays eager in both modes, as in current deployment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.starvla_groot.cuda_graph import TensorCudaGraph, _PrefixPlan
from policy.unified_deploy import Policy
from scripts.benchmark_starvla_cuda_graph import summary, timed
from scripts.test_policy import hdf5_observation


def parameters(module):
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--task", default="lift_can")
    parser.add_argument("--frames", default="0,10,20")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--ensemble", type=int, help="Override inference ensemble size; 1 disables ensemble")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 10 or args.warmup < 1:
        parser.error("Use at least 10 samples and 1 warmup")
    if args.ensemble is not None and args.ensemble < 1:
        parser.error("Ensemble size must be at least 1")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(123)
    policy = Policy({"checkpoint": args.checkpoint, "task_name": args.task, "device": str(device)})
    model, qv = policy.model, policy.model.qwen_vl
    if args.ensemble is not None:
        model.config.repeated_diffusion_steps = args.ensemble
    tactile = model.tactile_encoder
    if tactile is None or tactile.method != "VAE" or tactile.mode != "full":
        raise ValueError("This benchmark requires VAE full tactile tokens")
    if model.config.tactile_insert_location != "encoder":
        raise ValueError("Expected encoder-side tactile fusion")
    core, visual = qv.model.model, qv.model.model.visual
    text = core.language_model
    reps = max(1, int(model.config.repeated_diffusion_steps))
    frames = [int(v) for v in args.frames.split(",")]
    batches = [policy.encode_obs(hdf5_observation(args.episode, f, policy.run)) for f in frames]

    # Prepare all observations ONCE, before any measurements.
    samples = []
    for batch in batches:
        images = model._build_images(batch)
        pixels, grid = qv.gpu_pixel_values(images.flatten(0, 1))
        tactile_images = tuple(F.interpolate(batch[k], size=(tactile.image_size, tactile.image_size),
                                            mode="bilinear", align_corners=False) for k in tactile.keys)
        samples.append({"batch": batch, "images": images, "pixels": pixels, "grid": grid,
                        "tactile_images": tactile_images})

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def tactile_eager(tactile_images):
        tokens = [tactile.proj(tactile.backbone.latent_tokens(x).to(tactile.proj.weight.dtype))
                  for x in tactile_images]
        return torch.cat(tokens, dim=1)

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def vision_eager(pixels, grid):
        out = core.get_image_features(pixels, grid, return_dict=True)
        return torch.cat(out.pooler_output, dim=0)

    for s in samples:
        s["tactile"] = tactile_eager(s["tactile_images"])
        s["vision"] = vision_eager(s["pixels"], s["grid"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            original_tactile = tactile.forward_flat(s["batch"])
        torch.testing.assert_close(s["tactile"], original_tactile, atol=0, rtol=0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        plan = _PrefixPlan(qv, samples[0]["images"],
                           model._get_instructions(batches[0], 1), samples[0]["tactile"])

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def vision_compute(pixels):
        hidden = visual.patch_embed(pixels.to(visual.dtype)) + plan.vision_positions
        for block in visual.blocks:
            hidden = block(hidden, cu_seqlens=plan.cu_seqlens, position_embeddings=plan.vision_rope)
        return visual.merger(hidden)

    def merge_tokens(vision, tactile_tokens):
        embeds = plan.base_embeds.clone()
        embeds.reshape(-1, embeds.shape[-1]).index_copy_(0, plan.image_indices, vision.to(embeds.dtype))
        embeds[:, plan.prompt_length:] = tactile_tokens.to(embeds.dtype)
        return embeds

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def text_eager(embeds):
        return text(inputs_embeds=embeds, position_ids=plan.positions,
                    attention_mask=plan.mask, use_cache=False).last_hidden_state

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def text_compute(embeds):
        return plan.text_compute._forward(embeds, plan.positions, plan.causal_mask, plan.linear_mask)

    @torch.autocast("cuda", dtype=torch.float32)
    def head_eager(hidden, state, mask, noise):
        return model._denoise_chunk(hidden, state, mask, noise)

    # Capture the same numerical vision/text computations as deployment, while
    # excluding normalization/patchify from both the graph and eager timing.
    vision_graph = TensorCudaGraph(visual, vision_compute)
    text_graph = TensorCudaGraph(text, text_compute)

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def prefix_compute(pixels, tactile_tokens):
        return text_compute(merge_tokens(vision_compute(pixels), tactile_tokens))

    prefix_graph = TensorCudaGraph(core, prefix_compute)

    @torch.autocast("cuda", dtype=torch.float32)
    def head_graph(hidden, state, mask, noise):
        return model._chunk_cuda_graph(hidden, state, mask, noise,
                                       extra_key=(reps, model.action_head.num_inference_timesteps))

    for s in samples:
        s["embeds"] = merge_tokens(s["vision"], s["tactile"])
        s["hidden"] = text_eager(s["embeds"])
        s["state"] = model._get_state(s["batch"], device, s["hidden"].dtype)
        s["mask"] = plan.mask.to(dtype=torch.bool)
        s["noise"] = torch.randn((reps, model.config.chunk_size, model.action_dim),
                                  device=device, dtype=s["hidden"].dtype)

    def pipeline(s, accelerated):
        tac = tactile_eager(s["tactile_images"])
        if accelerated:
            hidden = prefix_graph(s["pixels"], tac)
            return head_graph(hidden, s["state"], s["mask"], s["noise"])
        vis = vision_eager(s["pixels"], s["grid"])
        hidden = text_eager(merge_tokens(vis, tac))
        return head_eager(hidden, s["state"], s["mask"], s["noise"])

    functions = {
        "vision_encoder": (
            lambda s: vision_eager(s["pixels"], s["grid"]),
            lambda s: vision_graph(s["pixels"])),
        "tactile_encoder": (
            lambda s: tactile_eager(s["tactile_images"]),
            lambda s: tactile_eager(s["tactile_images"])),
        "vlm_text_decoder": (
            lambda s: text_eager(s["embeds"]),
            lambda s: text_graph(s["embeds"])),
        "action_head": (
            lambda s: head_eager(s["hidden"], s["state"], s["mask"], s["noise"]),
            lambda s: head_graph(s["hidden"], s["state"], s["mask"], s["noise"])),
        "model_only_fused_pipeline": (
            lambda s: pipeline(s, False), lambda s: pipeline(s, True)),
    }
    checks, latencies = {}, {}
    for name, (original, accelerated) in functions.items():
        checks[name] = []
        for s in samples:
            expected, actual = original(s), accelerated(s)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            checks[name].append({"max_abs": (actual - expected).abs().max().item(),
                                 "exact_equal": torch.equal(actual, expected)})
        for fn in (original, accelerated):
            for _ in range(args.warmup):
                fn(samples[0])
        raw = {"eager": [], "accelerated": []}
        captures_before = (vision_graph.captures, text_graph.captures, prefix_graph.captures,
                           model._chunk_cuda_graph.captures)
        for i in range(args.iterations):
            for enabled in ((False, True) if i % 2 == 0 else (True, False)):
                fn = accelerated if enabled else original
                _, elapsed = timed(lambda: fn(samples[i % len(samples)]), device)
                raw["accelerated" if enabled else "eager"].append(elapsed)
        captures_after = (vision_graph.captures, text_graph.captures, prefix_graph.captures,
                          model._chunk_cuda_graph.captures)
        assert captures_before == captures_after, "Capture leaked into steady-state timing"
        entry = {k: summary(v) for k, v in raw.items()}
        entry["speedup_mean"] = entry["eager"]["mean_ms"] / entry["accelerated"]["mean_ms"]
        entry["raw_ms"] = raw
        latencies[name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k != "raw_ms"}), flush=True)

    first = samples[0]
    # Validate the measurement-only pipeline against actual policy inference.
    model.config.use_cuda_graph = False
    original_policy = model.predict_action_chunk(first["batch"], noise=first["noise"])
    model.config.use_cuda_graph = True
    accelerated_policy = model.predict_action_chunk(first["batch"], noise=first["noise"])
    for output in (pipeline(first, False), pipeline(first, True), accelerated_policy):
        torch.testing.assert_close(output, original_policy, rtol=0, atol=0)

    report = {
        "checkpoint": str(policy.run["checkpoint"]), "step": policy.run["step"],
        "episode": str(Path(args.episode).resolve()), "frames": frames,
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "iterations_per_mode_per_component": args.iterations, "warmup_per_mode": args.warmup,
        "timing": "synchronized wall time, arithmetic mean, ABBA mode alternation",
        "excluded": ["HDF5 decode", "observation adapter", "CPU/GPU transfers", "image/tactile resize",
                     "normalization", "image patchify", "tokenizer", "prompt/M-RoPE setup",
                     "state normalization", "noise generation", "action denormalization", "first capture", "warmup"],
        "attribution": "vision/text captured separately for component microbenchmarks; fused for model-only total",
        "tactile_acceleration": "unchanged eager in both modes, matching current deployment",
        "layout": {"vision_images": list(first["images"].shape),
                   "pixel_values": list(first["pixels"].shape),
                   "tactile_images": [list(t.shape) for t in first["tactile_images"]],
                   "tactile_tokens": list(first["tactile"].shape),
                   "vlm_tokens": list(first["hidden"].shape),
                   "ensemble": reps, "denoise_steps": model.action_head.num_inference_timesteps,
                   "action_shape": list(original_policy.shape)},
        "architecture": {"vision_depth": len(visual.blocks), "vision_parameters": parameters(visual),
                         "tactile_parameters": parameters(tactile), "tactile_backbone": str(type(tactile.backbone)),
                         "text_depth": len(text.layers), "text_layer_types": dict(Counter(text.config.layer_types)),
                         "text_parameters": parameters(text), "head_parameters": parameters(model.action_head),
                         "head_depth": len(model.action_head.model.transformer_blocks)},
        "latency": latencies, "output_comparison": checks, "matches_production_output_exactly": True,
        "graph_stats": {name: {"captures": manager.captures, "replays": manager.replays,
                               "eager_fallbacks": manager.eager_calls}
                        for name, manager in (("vision_isolated", vision_graph), ("text_isolated", text_graph),
                                              ("prefix_no_preprocessing", prefix_graph),
                                              ("chunk", model._chunk_cuda_graph))},
    }
    components = list(functions)[:4]
    report["isolated_component_sum_ms"] = {
        mode: sum(latencies[n][mode]["mean_ms"] for n in components) for mode in ("eager", "accelerated")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
