"""Layer-wise StarVLA-PI training, tactile routing and checkpoint contracts."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.starvla_pi import StarvlaPiConfig
from policy.starvla_pi.modeling_starvla_pi import StarvlaPiPolicy
from policy.starvla_pi.layerwise_flow_matching_head import LayerwiseActionHeadConfig, LayerwiseFlowmatchingActionHead
from policy.vla_common.engine.configs import FeatureType, PolicyFeature, NormalizationMode

CAM = "observation.images.cam_high"
TAC = "observation.images.tac_left"


class TinyQwen(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(1, 3, 16))
        self.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2))
        self.hidden_size = 16
        self.views = None
        self._target_h = self._target_w = 16
        self._text_cuda_graph = self._prefix_cuda_graph = None

    def clear_cuda_graphs(self):
        pass

    def supports_prefix_injection(self):
        return True

    def build_qwenvl_inputs(self, images, instructions):
        self.views = images.shape[1]
        return {"attention_mask": torch.ones(len(images), 3, dtype=torch.bool)}

    def forward(self, attention_mask, **kwargs):
        x = self.embed.expand(len(attention_mask), -1, -1)
        return SimpleNamespace(hidden_states=(x * 0, x, x * 2))

    def forward_with_prefix_tokens(self, images, instructions, extra_embeds, **kwargs):
        self.views = images.shape[1]
        x = self.embed.expand(len(images), -1, -1)
        x = torch.cat((x, extra_embeds), dim=1)
        # A tiny differentiable stand-in for transformer prefix fusion.
        x = x + x.mean(1, keepdim=True)
        return (x * 0, x, x * 2), torch.ones(x.shape[:2], dtype=torch.bool)


class TinyTactile(nn.Module):
    def __init__(self, config, hidden):
        super().__init__()
        self.proj = nn.Linear(3, hidden)

    def forward_flat(self, batch):
        return self.proj(batch[TAC].mean((-1, -2))).unsqueeze(1)


class StarvlaPiTests(unittest.TestCase):
    def config(self, mode="none", location="encoder", state=True):
        features = {CAM: PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
                    "observation.state": PolicyFeature(FeatureType.STATE, (8,))}
        if mode != "none":
            features[TAC] = PolicyFeature(FeatureType.VISUAL, (3, 16, 16))
        return StarvlaPiConfig(input_features=features,
            output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
            top_camera_keys=[CAM], wrist_camera_keys=[], tactile_keys=[TAC],
            tactile_mode=mode, tactile_insert_location=location,
            tactile_encoder_config={"method": "ResNet", "size": "S", "input_mode": "marker_rgb"} if mode == "encode" else None,
            state_mode="absolute_joint" if state else "none", action_mode="relative_joint",
            device="cpu", chunk_size=3, n_action_steps=2,
            action_expert_hidden_size=16,
            layerwise_attention_head_dim=8, num_inference_timesteps=2, num_target_vision_tokens=2)

    def batch(self):
        return {CAM: torch.rand(2, 3, 16, 16), TAC: torch.rand(2, 3, 16, 16),
                "observation.state": torch.randn(2, 8), "action": torch.randn(2, 3, 8),
                "is_pad": torch.tensor([[False, False, True], [False, True, True]]),
                "task": ["lift can", "lift can"]}

    @patch("policy.starvla_pi.modeling_starvla_pi.QwenVLInterface", TinyQwen)
    @patch("policy.starvla_pi.modeling_starvla_pi.TactileEncoder", TinyTactile)
    def test_routing_backward_reload_and_queue(self):
        for mode, location in (("none", "encoder"), ("as_image", "encoder"),
                               ("encode", "encoder"), ("encode", "decoder")):
            for state in (True, False):
                with self.subTest(mode=mode, location=location, state=state):
                    policy = StarvlaPiPolicy(self.config(mode, location, state))
                    batch = self.batch()
                    loss, _ = policy(batch)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertGreater(policy.qwen_vl.embed.grad.abs().sum().item(), 0)
                    for block in policy.action_head.transformer_blocks:
                        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters()))
                    if mode == "encode":
                        self.assertGreater(policy.tactile_encoder.proj.weight.grad.abs().sum().item(), 0)
                    self.assertEqual(policy.qwen_vl.views, 2 if mode == "as_image" else 1)
                    replacement = StarvlaPiPolicy(self.config(mode, location, state))
                    replacement.load_state_dict(policy.state_dict(), strict=True)
                    torch.manual_seed(4)
                    expected = policy.predict_action_chunk(batch)
                    torch.manual_seed(4)
                    actual = replacement.predict_action_chunk(batch)
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(actual.shape, (2, 3, 8))
                    self.assertEqual(replacement.select_action(batch).shape, (2, 8))
                    replacement.reset()
                    self.assertEqual(len(replacement._action_queue), 0)

    @patch("policy.starvla_pi.modeling_starvla_pi.QwenVLInterface", TinyQwen)
    def test_optional_metrics_preserve_loss_and_gradients(self):
        policy = StarvlaPiPolicy(self.config())
        batch = self.batch()
        torch.manual_seed(12)
        loss, metrics = policy(batch)
        loss.backward()
        gradients = {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None}
        policy.zero_grad(set_to_none=True)
        torch.manual_seed(12)
        other, skipped = policy(batch, return_loss_dict=False)
        other.backward()
        self.assertEqual(skipped, {})
        self.assertEqual(metrics["loss"], loss.item())
        torch.testing.assert_close(loss, other, rtol=0, atol=0)
        for name, parameter in policy.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)

    @patch("policy.starvla_pi.modeling_starvla_pi.QwenVLInterface", TinyQwen)
    def test_padding_does_not_change_loss(self):
        policy = StarvlaPiPolicy(self.config())
        batch = self.batch()
        modified = {**batch, "action": batch["action"].clone()}
        modified["action"][batch["is_pad"]] = 1e5
        torch.manual_seed(7)
        loss, _ = policy(batch)
        torch.manual_seed(7)
        other, _ = policy(modified)
        torch.testing.assert_close(loss, other)
        mask = policy._action_mask_with_padded_tail(batch, 2, torch.device("cpu"), torch.float32)
        self.assertEqual(mask[..., 8:].count_nonzero(), 0)
        self.assertEqual(mask[batch["is_pad"]].count_nonzero(), 0)

    def test_layer_correspondence_and_masked_width(self):
        head = LayerwiseFlowmatchingActionHead(LayerwiseActionHeadConfig(
            hidden_size=16, num_layers=2, attention_head_dim=8,
            action_dim=32, state_dim=0, action_horizon=3, num_target_vision_tokens=2))
        layers = [torch.randn(2, 4, 16, requires_grad=True) for _ in range(2)]
        mask = torch.zeros(2, 3, 32)
        mask[..., :8] = 1
        action = torch.randn(2, 3, 32)
        altered = action.clone()
        altered[..., 8:] = 1e5
        torch.manual_seed(3)
        loss = head(layers, action, action_mask=mask)
        torch.manual_seed(3)
        torch.testing.assert_close(loss, head(layers, altered, action_mask=mask))
        loss.mean().backward()
        for layer in layers:
            self.assertGreater(layer.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            head.predict_action(layers[:1])
        output = head.predict_action(layers, action_mask=mask)
        self.assertEqual(output[..., 8:].count_nonzero(), 0)

    def test_defaults(self):
        config = self.config()
        self.assertEqual(config.type, "starvla_pi")
        self.assertEqual(config.normalization_mapping["ACTION"], NormalizationMode.QUANTILES)
        self.assertTrue(config.use_cuda_graph)
        with self.assertRaises(ValueError):
            StarvlaPiConfig(repeated_diffusion_steps=2)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
