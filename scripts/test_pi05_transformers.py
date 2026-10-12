"""Check pretrained PI05 image scale, gradients and cached denoising on a GPU.

Run: python scripts/test_pi05_transformers.py --device cuda:0
"""
from pathlib import Path
import argparse
import sys

import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.pi05.configuration_pi05 import PI05Config
from policy.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from policy.vla_common.engine.configs import FeatureType, PolicyFeature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    config = PI05Config(
        device=args.device, chunk_size=4, n_action_steps=4,
        num_inference_steps=3, tactile_mode='none',
        input_features={'observation.images.cam_high': PolicyFeature(FeatureType.VISUAL, (3, 224, 224))},
        output_features={'action': PolicyFeature(FeatureType.ACTION, (8,))},
        top_camera_keys=['observation.images.cam_high'], wrist_camera_keys=[],
    )
    policy = PI05Policy.from_pretrained(
        ROOT / 'resources/pretrained_models/pi05_base', config=config, local_files_only=True,
    )
    core = policy.model
    vlm = core.paligemma_with_expert
    images = [torch.rand(1, 3, 224, 224, device=args.device) * 2 - 1]
    img_masks = [torch.ones(1, dtype=torch.bool, device=args.device)]
    tokens = torch.tensor([[2, 10, 20, 30, 40, 50, 0, 0]], device=args.device)
    masks = tokens != 0
    actions = torch.randn(1, 4, 32, device=args.device)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.5], device=args.device)
    policy.eval()
    with torch.no_grad():
        reference = vlm.paligemma.model.multi_modal_projector(
            vlm.paligemma.model.vision_tower(images[0]).last_hidden_state,
        )
        for dtype in (torch.float32, torch.bfloat16):
            image = images[0].to(dtype)
            expected = vlm.paligemma.model.multi_modal_projector(
                vlm.paligemma.model.vision_tower(image.float()).last_hidden_state,
            ).to(dtype)
            actual = vlm.embed_image(image)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.dtype == dtype
        upstream = vlm.paligemma.model.get_image_features(images[0]).pooler_output
        torch.testing.assert_close(reference, upstream, rtol=0, atol=0)
    print(f'PASS image scale and FP32/BF16 output dtype; transformers={transformers.__version__}', flush=True)

    policy.train()
    loss = core(images, img_masks, tokens, masks, actions, noise, time).mean()
    assert torch.isfinite(loss)
    loss.backward()
    counts = {}
    for name, param in policy.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name
            if param.grad.count_nonzero():
                for group in ('vision_tower', 'multi_modal_projector', 'gemma_expert', 'action_in_proj', 'action_out_proj', 'time_mlp'):
                    if group in name:
                        counts[group] = counts.get(group, 0) + 1
    assert len(counts) == 6, counts
    print(f'PASS forward/backward loss={loss.item():.6f}; nonzero finite gradient tensors={counts}', flush=True)
    policy.zero_grad(set_to_none=True)
    policy.eval()
    with torch.no_grad():
        # Match sample_actions/denoise_step, which use eager attention for
        # the FP32 additive mask with BF16 queries.
        vlm.paligemma.model.language_model.config._attn_implementation = 'eager'
        vlm.gemma_expert.model.config._attn_implementation = 'eager'
        prefix, pads, att = core.embed_prefix(images, img_masks, tokens, masks)
        prefix = prefix.to(torch.bfloat16)
        mask4d = core._prepare_attention_masks_4d(make_att_2d_masks(pads, att))
        (_, _), cache = vlm(
            attention_mask=mask4d, position_ids=pads.cumsum(1)-1,
            inputs_embeds=[prefix, None], use_cache=True,
        )
        prefix_len = cache.get_seq_length()
        assert prefix_len == prefix.shape[1]
        for t in (0.9, 0.5, 0.1):
            timestep = torch.tensor([t], device=args.device)
            cached = core.denoise_step(pads, cache, noise, timestep)
            assert cache.get_seq_length() == prefix_len
            suffix, suffix_pads, suffix_att, cond = core.embed_suffix(noise, timestep)
            all_pads = torch.cat([pads, suffix_pads], dim=1)
            all_att = torch.cat([att, suffix_att], dim=1)
            (_, output), _ = vlm(
                attention_mask=core._prepare_attention_masks_4d(make_att_2d_masks(all_pads, all_att)),
                position_ids=all_pads.cumsum(1)-1,
                inputs_embeds=[prefix, suffix.to(torch.bfloat16)],
                use_cache=False, adarms_cond=[None, cond],
            )
            expected = core.action_out_proj(output[:, -4:].float())
            torch.testing.assert_close(cached, expected, rtol=0.03, atol=0.03)
            print(f'PASS cached/full-prefix equivalence t={t}; max_abs_diff={(cached-expected).abs().max().item():.6f}', flush=True)
        result = core.sample_actions(images, img_masks, tokens, masks, noise=noise, num_steps=3)
        repeat = core.sample_actions(images, img_masks, tokens, masks, noise=noise, num_steps=3)
        assert result.shape == (1, 4, 32) and torch.isfinite(result).all()
        torch.testing.assert_close(result, repeat, rtol=0, atol=0)
    torch.cuda.synchronize()
    print(f'PASS three-step cached sampling, repeatability; peak_allocated_GiB={torch.cuda.max_memory_allocated()/1024**3:.2f}', flush=True)


if __name__ == '__main__':
    main()
