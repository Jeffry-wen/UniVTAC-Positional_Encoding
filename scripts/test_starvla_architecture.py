"""Expert-width defaults, unequal cross-attention widths and legacy checkpoints."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import test_starvla_pi as helpers
TinyQwen, CAM = helpers.TinyQwen, helpers.CAM
from policy.starvla_groot.configuration_starvla_groot import StarvlaGrootConfig
from policy.starvla_groot.modeling_starvla_groot import StarvlaGrootPolicy
from policy.starvla_pi import StarvlaPiConfig
from policy.starvla_pi.modeling_starvla_pi import StarvlaPiPolicy
from policy.unified_deploy import make_vla_config
from policy.vla_common.engine.configs import PreTrainedConfig


class ArchitectureTests(unittest.TestCase):
    def run_config(self, kind, base):
        return dict(policy=kind, directory=Path("/tmp"), camera_keys=[CAM],
                    options=dict(tactile_mode="none", tactile_input_mode="rgb_only", tactile_type="cls",
                                 tactile_insert_location="encoder", action_mode="absolute_joint", chunk_size=3),
                    config=dict(base=base, input_features={CAM: [3, 16, 16], "observation.state": [8]},
                                output_features={"action": [8]}, state_mode="absolute_joint", action_gap=1,
                                chunk_size=3, n_action_steps=2))

    def test_default_widths_and_deployment_migration(self):
        for kind, cls in (("starvla_groot", StarvlaGrootConfig), ("starvla_pi", StarvlaPiConfig)):
            self.assertEqual(cls().action_expert_hidden_size, 1024)
            base = yaml.safe_load((ROOT / "policy" / kind / "base_config.yml").read_text())
            self.assertEqual(base["action_expert_hidden_size"], 1024)
            restored = make_vla_config(self.run_config(kind, {}), "cpu", ROOT / "resources/pretrained_models")
            self.assertIsNone(restored.action_expert_hidden_size)
            current = make_vla_config(self.run_config(kind, base), "cpu", ROOT / "resources/pretrained_models")
            self.assertEqual(current.action_expert_hidden_size, 1024)

    def test_old_and_new_serialized_configs(self):
        for cls in (StarvlaGrootConfig, StarvlaPiConfig):
            with tempfile.TemporaryDirectory() as directory:
                cls(device="cpu")._save_pretrained(Path(directory))
                restored = PreTrainedConfig.from_pretrained(directory)
                self.assertEqual(restored.action_expert_hidden_size, 1024)
                path = Path(directory) / "config.json"
                data = json.loads(path.read_text())
                data.pop("action_expert_hidden_size")
                data.pop("legacy_layerwise_layout", None)
                path.write_text(json.dumps(data))
                restored = PreTrainedConfig.from_pretrained(directory)
                self.assertIsNone(restored.action_expert_hidden_size)
                if cls is StarvlaPiConfig:
                    self.assertTrue(restored.legacy_layerwise_layout)

    @patch("policy.starvla_pi.modeling_starvla_pi.QwenVLInterface", TinyQwen)
    def test_pi_decoupled_width_backward_and_legacy_strict_reload(self):
        helper = helpers.StarvlaPiTests()
        cfg = helper.config()
        cfg.action_expert_hidden_size = 32
        model = StarvlaPiPolicy(cfg)
        block = model.action_head.transformer_blocks[0]
        self.assertEqual(block.attn1.to_q.in_features, 32)
        self.assertEqual(block.attn1.to_k.in_features, 16)
        loss, _ = model(helper.batch(), return_loss_dict=False)
        loss.backward()
        self.assertGreater(model.qwen_vl.embed.grad.abs().sum().item(), 0)
        for block in model.action_head.transformer_blocks:
            self.assertIsNotNone(block.attn1.to_k.weight.grad)
        # Pre-width PI used the VLM width; retain keys and strict loading.
        old = helper.config()
        old.action_expert_hidden_size = None
        legacy = StarvlaPiPolicy(old).eval()
        restored = StarvlaPiPolicy(old).eval()
        restored.load_state_dict(legacy.state_dict(), strict=True)
        batch = helper.batch()
        torch.manual_seed(14)
        expected = legacy.predict_action_chunk(batch)
        torch.manual_seed(14)
        actual = restored.predict_action_chunk(batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        cfg.layerwise_num_layers = 1
        with self.assertRaisesRegex(ValueError, "must equal"):
            StarvlaPiPolicy(cfg)
        old.layerwise_num_layers, old.legacy_layerwise_layout = 1, True
        self.assertEqual(len(StarvlaPiPolicy(old).action_head.transformer_blocks), 1)

    @patch("policy.starvla_groot.modeling_starvla_groot.QwenVLInterface", TinyQwen)
    def test_groot_legacy_preset_and_new_width_strict_reload(self):
        for width, expected in ((None, 768), (32, 32)):
            cfg = make_vla_config(self.run_config("starvla_groot", dict(
                action_expert_hidden_size=width, action_head_hidden_size=32,
                diffusion_model_cfg=dict(num_layers=1, output_dim=32, dropout=0.0,
                                         positional_embeddings=None,
                                         **({"attention_head_dim": 8} if width is not None else {})))), "cpu", ROOT)
            cfg.num_inference_timesteps = 1
            cfg.use_cuda_graph = False
            model = StarvlaGrootPolicy(cfg).eval()
            self.assertEqual(model.action_head.input_embedding_dim, expected)
            replacement = StarvlaGrootPolicy(cfg).eval()
            replacement.load_state_dict(model.state_dict(), strict=True)
            batch = helpers.StarvlaPiTests().batch()
            torch.manual_seed(14)
            expected_actions = model.predict_action_chunk(batch)
            torch.manual_seed(14)
            actual = replacement.predict_action_chunk(batch)
            torch.testing.assert_close(actual, expected_actions, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
