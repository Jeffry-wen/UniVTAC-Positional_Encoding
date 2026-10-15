"""N0 runtime, native tactile, and portable adapter checks without CUDA."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import cv2
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.n0vtla.configuration_n0vtla import N0PolicyConfig
from policy.n0vtla.observation_adapter import make_observation, masked_action_loss, tokenize_batch, TACTILE_KEYS
from policy.n0vtla.native.tactile_predictor import TactileActionPredictor
from policy.train_unified import UniVTACDataset
from policy.unified_deploy import Policy
from scripts.test_policy import hdf5_observation


class N0Tests(unittest.TestCase):
    def test_runtime_sources_are_repository_local(self):
        import policy.n0vtla.modeling_n0vtla
        self.assertFalse(any(name == "n0vtla" or name.startswith("n0vtla.") for name in sys.modules))
        for name, module in list(sys.modules.items()):
            if name.startswith("policy.n0vtla") and getattr(module, "__file__", None):
                self.assertTrue(Path(module.__file__).resolve().is_relative_to(ROOT), name)

    def test_native_cache_matches_joint_attention_and_is_read_only(self):
        from transformers import GemmaConfig
        from policy.vla_common.pi_gemma import PiGemmaModel
        from policy.n0vtla.native.gemma_pytorch import PaliGemmaWithExpertModel
        torch.manual_seed(9)
        prefix_config = GemmaConfig(vocab_size=16, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16)
        suffix_config = GemmaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            use_adarms=True, adarms_cond_dim=16)
        player, expert = PiGemmaModel(prefix_config), PiGemmaModel(suffix_config)
        model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
        torch.nn.Module.__init__(model)
        model.paligemma = SimpleNamespace(language_model=player,
            model=SimpleNamespace(language_model=player), config=SimpleNamespace(text_config=prefix_config))
        model.gemma_expert = SimpleNamespace(model=expert)
        model.eval()
        prefix, suffix, cond = torch.randn(1, 3, 32), torch.randn(1, 2, 16), torch.randn(1, 16)
        mask = torch.zeros(1, 1, 5, 5)
        mask[:, :, :3, 3:] = -2.3819763e38
        positions = torch.arange(5)[None]
        joint, _ = model(attention_mask=mask, position_ids=positions,
                         inputs_embeds=[prefix, suffix], adarms_cond=[None, cond], use_cache=False)
        _, cache = model(attention_mask=mask[:, :, :3, :3], position_ids=positions[:, :3],
                         inputs_embeds=[prefix, None], use_cache=True)
        before = [[t.clone() for t in group] for group in cache]
        for _ in range(2):
            cached, _ = model(attention_mask=mask[:, :, 3:, :], position_ids=positions[:, 3:],
                              inputs_embeds=[None, suffix], past_key_values=cache,
                              adarms_cond=[None, cond], use_cache=False)
            torch.testing.assert_close(joint[1], cached[1], rtol=1e-5, atol=1e-6)
        for original, current in zip(before, cache):
            for a, b in zip(original, current):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_siglip_fp32_embeddings_bf16_encoder(self):
        from transformers import SiglipVisionConfig, SiglipVisionModel
        from policy.n0vtla.native.gemma_pytorch import PaliGemmaWithExpertModel
        vision = SiglipVisionModel(SiglipVisionConfig(hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, image_size=28, patch_size=14)).to(torch.bfloat16)
        vision.vision_model.embeddings.float()
        model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
        torch.nn.Module.__init__(model)
        model.paligemma = SimpleNamespace(model=SimpleNamespace(vision_tower=vision,
            multi_modal_projector=torch.nn.Linear(32, 16).to(torch.bfloat16)))
        output = model.embed_image(torch.rand(1, 3, 28, 28))
        self.assertEqual(tuple(output.shape), (1, 4, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_image_scale_matches_transformers55_and_has_gradients(self):
        import inspect
        from transformers import SiglipVisionConfig, SiglipVisionModel
        from transformers.models.paligemma.modeling_paligemma import PaliGemmaModel
        from policy.n0vtla.native.gemma_pytorch import PaliGemmaWithExpertModel
        torch.manual_seed(17)
        vision = SiglipVisionModel(SiglipVisionConfig(hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, image_size=28, patch_size=14)).to(torch.bfloat16)
        vision.vision_model.embeddings.float()
        projector = torch.nn.Linear(32, 16).to(torch.bfloat16)
        backbone = SimpleNamespace(vision_tower=vision, multi_modal_projector=projector,
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=16)))
        model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
        torch.nn.Module.__init__(model)
        model.paligemma = SimpleNamespace(model=backbone)
        image = torch.rand(1, 3, 28, 28, requires_grad=True)
        # Use HF's complete vision forward as the reference, supplying only
        # the FP32-embedding -> BF16-encoder transition required by N0.
        handle = vision.vision_model.embeddings.register_forward_hook(
            lambda module, inputs, output: output.to(torch.bfloat16))
        try:
            with torch.no_grad():
                expected = inspect.unwrap(PaliGemmaModel.get_image_features)(backbone, image).pooler_output
        finally:
            handle.remove()
        output = model.embed_image(image)
        self.assertEqual(output.dtype, torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.float().square().mean().backward()
        for name, parameter in [('image', image), ('projector', projector.weight),
                ('patch_embedding', vision.vision_model.embeddings.patch_embedding.weight),
                ('encoder', vision.vision_model.encoder.layers[0].self_attn.q_proj.weight)]:
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(int(parameter.grad.count_nonzero()), 0, name)

    def test_masked_loss_and_gradients(self):
        loss = torch.ones(1, 3, 32, requires_grad=True)
        with torch.no_grad():
            loss[:, 2] = 1e6
            loss[..., 8:] = 1e6
        value = masked_action_loss(loss, torch.tensor([[False, False, True]]))
        self.assertEqual(float(value), 1.)
        value.backward()
        self.assertEqual(int((loss.grad != 0).sum()), 16)

    def test_observation_slots_and_raw_difference(self):
        batch = {"observation.state": torch.ones(2, 8), "observation.images.cam_high": torch.ones(2, 3, 224, 224) * .8,
                 "observation.images.tac_left": torch.ones(2, 3, 224, 224) * .7,
                 "observation.images.tac_left.baseline": torch.ones(2, 3, 224, 224) * .2,
                 "observation.language.tokens": torch.zeros(2, 200, dtype=torch.long),
                 "observation.language.attention_mask": torch.ones(2, 200, dtype=torch.bool)}
        obs = make_observation(batch)
        torch.testing.assert_close(obs.images[TACTILE_KEYS[0]] - obs.images[TACTILE_KEYS[0]+".baseline"], torch.ones(2, 3, 224, 224))
        self.assertFalse(obs.image_masks["right_wrist_0_rgb"].any())
        self.assertFalse(obs.image_masks[TACTILE_KEYS[1]].any())
        self.assertEqual(tuple(obs.state.shape), (2, 32))
        self.assertFalse(obs.state[:, 8:].any())
        batch.pop("observation.images.tac_left.baseline")
        with self.assertRaisesRegex(ValueError, "both current and baseline"):
            make_observation(batch)

    def test_predictor_handles_masked_views(self):
        torch.manual_seed(7)
        model = TactileActionPredictor(32, n_latent=5, n_layers=1, n_heads=4, predictor_arch="tactile_kv")
        context = torch.randn(2, 6, 32)
        tactile = torch.randn(2, 10, 32, requires_grad=True)
        mask = torch.tensor([[True]*5+[False]*5, [False]*10])
        output = model(context, tactile, g_mask=mask, vl_ctx_mask=torch.ones(2, 6, dtype=torch.bool))
        self.assertTrue(torch.isfinite(output).all())
        changed = tactile.detach().clone()
        changed[~mask] = 1000
        torch.testing.assert_close(output, model(context, changed, g_mask=mask, vl_ctx_mask=torch.ones(2, 6, dtype=torch.bool)))
        output.square().mean().backward()
        self.assertGreater(float(tactile.grad[0, :5].abs().sum()), 0)
        self.assertEqual(float(tactile.grad[~mask].abs().sum()), 0)

    @unittest.skipUnless((ROOT / "resources/pretrained_models/pi05_base/paligemma-3b-pt-224-tokenizer/tokenizer.json").is_file(), "Local tokenizer not installed")
    def test_tokenizer_and_scheduler(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "resources/pretrained_models/pi05_base/paligemma-3b-pt-224-tokenizer", local_files_only=True)
        batch = {"task": ["lift_can"], "observation.state": torch.zeros(1, 8)}
        tokenize_batch(batch, tokenizer, 200)
        tokens = batch["observation.language.tokens"][0]
        mask = batch["observation.language.attention_mask"][0]
        expected = [tokenizer.bos_token_id] + tokenizer.encode("Task: lift can, State: 128 128 128 128 128 128 128 128;\nAction: ", add_special_tokens=False)
        self.assertEqual(tokens[mask].tolist(), expected)
        self.assertTrue((tokens[~mask] == 0).all())
        config = N0PolicyConfig()
        optimizer = torch.optim.AdamW([{"params": [torch.nn.Parameter(torch.ones(1))], "lr": config.optimizer_lr},
                                       {"params": [torch.nn.Parameter(torch.ones(1))], "lr": config.optimizer_lr*.1}])
        scheduler = config.get_scheduler_preset().build(optimizer, 20_000)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"], .1)

    def test_training_offline_and_live_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "0.hdf5"
            joints = np.arange(27, dtype=np.float32).reshape(3, 9) / 100
            images = [np.full((24, 32, 3), value, np.uint8) for value in (20, 80, 140)]
            with h5py.File(path, "w") as f:
                f["embodiment/joint"] = joints
                dtype = h5py.vlen_dtype(np.dtype("uint8"))
                for key in ("observation/head/rgb", "tactile/left_tactile/rgb", "tactile/right_tactile/rgb",
                            "tactile/left_tactile/rgb_marker", "tactile/right_tactile/rgb_marker"):
                    dataset = f.create_dataset(key, (3,), dtype=dtype)
                    for i, image in enumerate(images):
                        sample = image.copy()
                        if key.endswith("rgb_marker"):
                            cv2.circle(sample, (8 + i * 3, 12), 3, (0, 0, 0), -1)
                        dataset[i] = cv2.imencode(".jpg", sample)[1]
                fixed = np.array([(x, y) for y in (40, 120, 200) for x in (40, 160, 280)], dtype=np.float32)
                markers = np.stack([np.stack([fixed, fixed + step * 4]) for step in range(3)])
                for side in ("left", "right"):
                    f[f"tactile/{side}_tactile/marker"] = markers
                    f[f"tactile/{side}_tactile/depth"] = np.stack([np.full((24, 32), 34-step, np.float32) for step in range(3)])
            stats = {name: {"q01": [-1.]*8, "q99": [1.]*8} for name in ("state", "action")}
            for mode in ("rgb_only", "marker_rgb", "marker_only", "depth_deform_20", "depth_deform_80", "depth_deform_160", "depth_deform_240", "depth_deform_nonlinear", "depth_deform_nonlinear_radial"):
                with self.subTest(mode=mode):
                    dataset = UniVTACDataset({"test": [path]}, cameras=["head"], tactile_mode="native", tactile_input_mode=mode,
                                            chunk_size=2, stats=stats, policy="n0vtla", image_size=224, action_mode="relative")
                    try:
                        policy = Policy.__new__(Policy)
                        policy.kind, policy.stats, policy.task_name = "n0vtla", stats, "test"
                        policy.camera_keys = ["observation.images.cam_high"]
                        policy.tactile_mode, policy.tactile_input_mode = "native", mode
                        policy.tokenizer, policy.device, policy.model = None, torch.device("cpu"), object()
                        policy.reset()
                        run = {"camera_keys": policy.camera_keys, "options": {"tactile_mode": "native", "tactile_input_mode": mode}}
                        train = dataset[1]
                        offline_obs = hdf5_observation(path, 1, run)
                        offline = policy.encode_obs(offline_obs)
                        for key in train:
                            if key.startswith("observation."):
                                torch.testing.assert_close(train[key], offline[key][0], rtol=0, atol=0)
                        policy.reset()
                        first_obs = hdf5_observation(path, 0, run)
                        for side in ("left", "right"):
                            first_obs["tactile"][side+"_tactile"].pop("baseline_rgb", None)
                            offline_obs["tactile"][side+"_tactile"].pop("baseline_rgb", None)
                            first_obs["tactile"][side+"_tactile"].pop("baseline", None)
                            offline_obs["tactile"][side+"_tactile"].pop("baseline", None)
                        policy.encode_obs(first_obs)
                        live = policy.encode_obs(offline_obs)
                        torch.testing.assert_close(live["observation.images.tac_left.baseline"], offline["observation.images.tac_left.baseline"])
                        policy.reset()
                        new_episode = policy.encode_obs(offline_obs)
                        torch.testing.assert_close(new_episode["observation.images.tac_left.baseline"], new_episode["observation.images.tac_left"])
                    finally:
                        for handle in dataset._files.values():
                            handle.close()

    @unittest.skipUnless((ROOT / "resources/pretrained_models/n0-vtla-base/model.safetensors").is_file(), "Local N0 base not installed")
    def test_full_base_shapes_and_tied_embedding(self):
        from safetensors import safe_open
        from policy.n0vtla.modeling_n0vtla import N0Policy
        with torch.device("meta"):
            model = N0Policy(N0PolicyConfig())
        base = model.model
        self.assertIs(base.paligemma_with_expert.paligemma.lm_head.weight,
                      base.paligemma_with_expert.paligemma.language_model.embed_tokens.weight)
        with safe_open(ROOT / "resources/pretrained_models/n0-vtla-base/model.safetensors", framework="pt") as f:
            expected = base.state_dict()
            self.assertFalse(set(f.keys()) - set(expected))
            self.assertEqual(set(expected) - set(f.keys()), {"paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"})
            for key in f.keys():
                self.assertEqual(tuple(expected[key].shape), tuple(f.get_slice(key).get_shape()), key)


if __name__ == "__main__":
    unittest.main()
