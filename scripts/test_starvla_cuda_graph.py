"""CUDA regression checks for changing inputs, masks, shapes, and ownership."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.starvla_groot.cuda_graph import ActionHeadCudaGraph, Qwen35PrefixCudaGraph, Qwen35TextCudaGraph, TensorCudaGraph


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CudaGraphTests(unittest.TestCase):
    @torch.no_grad()
    def test_policy_invalidates_captured_weights(self):
        from policy.starvla_groot.modeling_starvla_groot import StarvlaGrootPolicy

        class Head(torch.nn.Module):
            num_inference_timesteps = 1

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0, device="cuda"))

            def predict_action(self, hidden, state, mask, noise):
                return noise + self.weight * hidden.mean() * mask.float().mean() + state.mean()

        policy = StarvlaGrootPolicy.__new__(StarvlaGrootPolicy)
        torch.nn.Module.__init__(policy)
        policy.config = SimpleNamespace(repeated_diffusion_steps=3, train_expert_only=False,
                                        freeze_vision_encoder=False)
        policy.action_head = Head()
        policy.qwen_vl = torch.nn.Identity()
        policy.qwen_vl.clear_cuda_graphs = lambda: None
        policy._action_cuda_graph = ActionHeadCudaGraph(policy.action_head)
        policy._chunk_cuda_graph = TensorCudaGraph(policy.action_head, policy._denoise_chunk)
        policy.eval()
        hidden = torch.ones(1, 5, 8, device="cuda")
        state = torch.ones(1, 1, 2, device="cuda")
        mask = torch.ones(1, 5, device="cuda", dtype=torch.bool)
        noise = torch.randn(3, 4, 2, device="cuda")
        first = policy._chunk_cuda_graph(hidden, state, mask, noise)
        policy.load_state_dict({"action_head.weight": torch.tensor(4.0, device="cuda")}, assign=True)
        self.assertFalse(policy._chunk_cuda_graph.entries)
        changed = policy._chunk_cuda_graph(hidden, state, mask, noise)
        torch.testing.assert_close(changed, first + 3)
        policy.train()
        self.assertFalse(policy._chunk_cuda_graph.entries)
        policy.eval()
        policy._chunk_cuda_graph(hidden, state, mask, noise)
        policy.to(dtype=torch.float64)
        self.assertFalse(policy._chunk_cuda_graph.entries)

    @torch.no_grad()
    def test_input_refresh_output_ownership_and_bounded_cache(self):
        layer = torch.nn.Linear(4, 4, device="cuda").eval()
        graph = TensorCudaGraph(layer, lambda x, bias: layer(x) + bias, max_graphs=1)
        x, bias = torch.randn(2, 4, device="cuda"), torch.randn(2, 4, device="cuda")
        first = graph(x, bias)
        saved = first.clone()
        x.add_(1)
        bias.mul_(2)
        torch.testing.assert_close(graph(x, bias), layer(x) + bias)
        torch.testing.assert_close(first, saved, rtol=0, atol=0)
        x2, bias2 = x[:1], bias[:1]
        torch.testing.assert_close(graph(x2, bias2), layer(x2) + bias2)
        self.assertEqual(len(graph.entries), 1)
        self.assertEqual(graph.eager_calls, 1)
        graph.clear()
        self.assertFalse(graph.entries)
        graph(x2, bias2)
        self.assertEqual(graph.captures, 2)

    @torch.no_grad()
    def test_qwen_text_padding_positions_and_shape_changes(self):
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

        config = Qwen3_5TextConfig(
            vocab_size=32, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 1.0, "mrope_section": [6, 5, 5]},
        )
        config._attn_implementation = "sdpa"
        model = Qwen3_5TextModel(config).to(device="cuda", dtype=torch.bfloat16).eval()
        graph = Qwen35TextCudaGraph(model, max_graphs=2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for length, padding, offset in ((9, 0, 0), (9, 2, 0), (9, 2, 7), (13, 1, 4)):
                embeds = torch.randn(2, length, 128, device="cuda", dtype=torch.bfloat16)
                mask = torch.ones(2, length, device="cuda", dtype=torch.long)
                mask[0, :padding] = 0
                positions = (torch.arange(length, device="cuda") + offset)[None, None].expand(3, 2, -1)
                expected = model(inputs_embeds=embeds, position_ids=positions,
                                 attention_mask=mask, use_cache=False).last_hidden_state
                actual = graph(embeds, positions, mask)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(graph.captures, 2)
        self.assertGreater(graph.replays, graph.captures)
        self.assertEqual(graph.eager_calls, 1)

    @torch.no_grad()
    def test_full_prefix_mixed_prompts_padding_and_tactile(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from policy.starvla_groot.qwen_vl_interface import QwenVLInterface

        config = Qwen3_5Config(
            image_token_id=29, video_token_id=30, vision_start_token_id=28, vision_end_token_id=27,
            text_config=dict(
                vocab_size=32, hidden_size=128, intermediate_size=256,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16,
                linear_num_key_heads=2, linear_num_value_heads=4,
                layer_types=["linear_attention", "full_attention"],
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                 "partial_rotary_factor": 1.0, "mrope_section": [6, 5, 5]},
            ),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                               out_hidden_size=128, patch_size=2, temporal_patch_size=1,
                               spatial_merge_size=2, num_position_embeddings=16),
        )
        config._attn_implementation = "sdpa"
        qv = QwenVLInterface.__new__(QwenVLInterface)
        torch.nn.Module.__init__(qv)
        qv.model = Qwen3_5ForConditionalGeneration(config).to(device="cuda", dtype=torch.bfloat16).eval()
        qv.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        qv._patch_size, qv._merge_size, qv._temporal_patch_size = 2, 2, 1
        qv._target_h, qv._target_w = 8, 8
        qv._img_mean, qv._img_std = [0.5] * 3, [0.25] * 3
        qv.use_cuda_graph, qv._text_cuda_graph = False, None
        qv.eval()

        def text_inputs(instructions, views):
            rows = [[1] + [token for _ in range(views) for token in (28, 29, 29, 29, 29, 27)]
                    + [ord(char) % 20 + 1 for char in instruction] for instruction in instructions]
            length = max(map(len, rows))
            ids = torch.tensor([[0] * (length - len(row)) + row for row in rows], device="cuda")
            return ids, (ids != 0).long()

        qv.build_text_inputs = text_inputs
        manager = Qwen35PrefixCudaGraph(qv, max_graphs=2)
        images = torch.rand(2, 2, 3, 8, 8, device="cuda")
        extra = torch.randn(2, 2, 128, device="cuda", dtype=torch.bfloat16)
        for instructions in (("a", "bbb"), ("b", "ccc"), ("b", "ccc")):
            images = images * 0.9
            extra = extra + 0.25
            with torch.autocast("cuda", dtype=torch.bfloat16):
                expected, mask = qv.forward_with_prefix_tokens(images, instructions, extra)
                actual, actual_mask = manager(images, instructions, extra)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_mask, mask, rtol=0, atol=0)
        self.assertEqual(manager.captures, 2)
        self.assertIsNone(manager(images, ("d", "eee"), extra))
        self.assertEqual(manager.eager_calls, 1)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
