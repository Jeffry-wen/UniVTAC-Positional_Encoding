"""Raw physical inputs, trainable policy tokens and portable checkpoint restore."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from encoder.DINO_TactileRoPE.encoder import TactileRoPEViT
from encoder.DINO_TactileRoPE_V2.encoder import TactileRoPEViTV2
from encoder.modality import read_tactile
from encoder.image_io import encode_rgb
from policy.vla_common.tactile_encode import TactileEncoder
from policy.train_unified import UniVTACDataset


def config(metadata, path=None):
    return SimpleNamespace(tactile_encoder_path=path, tactile_encoder_config=metadata,
                           tactile_input_mode='depth_deform_raw', tactile_type='cls',
                           freeze_tactile_encoder=False,
                           tactile_encoder_keys=lambda: ['left', 'right'])


class RoPEPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_policy_routing_preserves_physical_fields(self):
        from policy.starvla_groot.configuration_starvla_groot import StarvlaGrootConfig
        from policy.vla_common.engine.configs import FeatureType, PolicyFeature
        camera, tactile = 'observation.images.cam_high', 'observation.images.tac_left'
        cfg = StarvlaGrootConfig(device='cpu', tactile_mode='encode', tactile_type='cls',
            tactile_input_mode='depth_deform_raw', tactile_insert_location='encoder',
            tactile_encoder_path='/unused/encoder.pth',
            top_camera_keys=[camera], wrist_camera_keys=[], tactile_keys=[tactile],
            input_features={camera: PolicyFeature(FeatureType.VISUAL, (3,224,224)),
                            tactile: PolicyFeature(FeatureType.VISUAL, (5,224,224))})
        self.assertFalse(cfg.freeze_tactile_encoder)
        self.assertNotIn(tactile, cfg.normalizer_input_features())
        self.assertNotIn(tactile, cfg.image_features)
        self.assertIn(camera, cfg.image_features)

    def test_legacy_dinov2_still_loads(self):
        from encoder.vit_encoder import TactileViT
        reference = TactileViT('S', image_size=32)
        metadata = dict(method='DINOv2', size='S', input_mode='marker_rgb', in_chans=3,
                        image_size=32, color_order='RGB', encoder=reference.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'encoder.pth'
            torch.save(metadata, path)
            cfg = config(None, str(path))
            cfg.tactile_input_mode = 'marker_rgb'
            adapter = TactileEncoder(cfg, 16)
            image = torch.rand(1,3,32,32)
            torch.testing.assert_close(adapter.backbone(image), reference(image))
            self.assertEqual(adapter({'left':image,'right':image}).shape, (1,2,16))

    def test_sensor_fields_gradients_and_policy_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5path = Path(tmp) / 'episode.h5'
            ref = np.array([[2,2], [60,2], [2,44], [60,44]], dtype=np.float32)
            with h5py.File(h5path, 'w') as f:
                f['embodiment/joint'] = np.zeros((2, 8), dtype=np.float32)
                jpeg = np.frombuffer(encode_rgb(np.zeros((48, 64, 3), dtype=np.uint8)), dtype=np.uint8)
                f['observation/head/rgb'] = np.stack([jpeg, jpeg])
                for side in ('left', 'right'):
                    group = f.create_group(f'tactile/{side}_tactile')
                    group['depth'] = np.full((2, 48, 64), 28.5, dtype=np.float32)
                    group['marker'] = np.stack([np.stack([ref, ref])]*2)
                    # Prefer full markers, not the reduced marker fallback.
                    group['marker_full'] = np.stack([np.stack([ref, ref+[7., -4.]])]*2)
            with h5py.File(h5path, 'r') as f:
                raw = read_tactile(f, 'left', 0, 'depth_deform_raw', 32)
            np.testing.assert_allclose(raw[:3].mean((1,2)), [28.5, 7., -4.], atol=1e-4)
            np.testing.assert_allclose(raw[3,0], (np.arange(32)+.5)*2)
            np.testing.assert_allclose(raw[4,:,0], (np.arange(32)+.5)*1.5)
            stats = {key: dict(mean=np.zeros(8), std=np.ones(8)) for key in ('state', 'action')}
            dataset = UniVTACDataset({'test_task': [h5path]}, cameras=['head'],
                                     tactile_mode='encode', tactile_input_mode='depth_deform_raw',
                                     chunk_size=2, stats=stats, policy='starvla_groot', image_size=32)
            item = dataset[0]
            for side in ('left', 'right'):
                torch.testing.assert_close(item[f'observation.images.tac_{side}'], raw)
            for handle in dataset._files.values():
                handle.close()
            for method, arch, cls in [('DINO_TactileRoPE', 'tactile_rope_v1', TactileRoPEViT),
                                      ('DINO_TactileRoPE_V2', 'tactile_rope_v2', TactileRoPEViTV2)]:
                with self.subTest(method=method):
                    torch.manual_seed(7)
                    reference = cls().eval()
                    metadata = dict(method=method, architecture=arch, size='S',
                                    input_mode='depth_deform_raw', in_chans=5, image_size=32,
                                    model_config={'rope': reference.config})
                    path = Path(tmp) / f'{method}.pth'
                    torch.save(dict(metadata, encoder=reference.state_dict()), path)
                    cfg = config(None, str(path))
                    adapter = TactileEncoder(cfg, 16)
                    x = raw.unsqueeze(0)
                    torch.testing.assert_close(adapter.backbone(x), reference(x))
                    out = adapter({'left': x, 'right': x})
                    self.assertEqual(out.shape, (1, 2, 16))
                    torch.testing.assert_close(out[:, :1], adapter.proj(reference(x))[:, None])
                    # Absolute sensor coordinates must reach attention unmodified.
                    changed = x.clone()
                    changed[:, 3, :, 16:] += 200
                    self.assertGreater((adapter.backbone(changed)-adapter.backbone(x)).abs().max().item(), 1e-8)
                    before = adapter.backbone.patch_embed.weight.detach().clone()
                    optimizer = torch.optim.SGD(adapter.parameters(), lr=.01)
                    out.square().mean().backward()
                    for name in ['patch_embed.weight', 'blocks.0.qkv.weight', 'blocks.0.rest_project.weight']:
                        grad = dict(adapter.backbone.named_parameters())[name].grad
                        self.assertIsNotNone(grad, name)
                        self.assertTrue(torch.isfinite(grad).all(), name)
                        self.assertGreater(grad.abs().sum().item(), 0, name)
                    optimizer.step()
                    self.assertFalse(torch.equal(before, adapter.backbone.patch_embed.weight))
                    # A policy checkpoint embeds constructor metadata and all weights;
                    # restoring it must not depend on the original initializer file.
                    path.unlink()
                    restored = TactileEncoder(config(cfg.tactile_encoder_config), 16)
                    restored.load_state_dict(adapter.state_dict(), strict=True)
                    torch.testing.assert_close(restored({'left':x,'right':x}), adapter({'left':x,'right':x}))
                    temporal = x[:, None].expand(-1, 2, -1, -1, -1)
                    self.assertEqual(adapter({'left':temporal,'right':temporal}).shape, (1,4,16))
                    with self.assertRaisesRegex(ValueError, 'raw'):
                        adapter({'left':x[:,:3], 'right':x[:,:3]})
                    bad = copy.deepcopy(metadata)
                    bad['architecture'] = 'wrong'
                    with self.assertRaisesRegex(ValueError, 'architecture'):
                        TactileEncoder(config(bad), 16)
                    bad = copy.deepcopy(metadata)
                    del bad['model_config']
                    with self.assertRaisesRegex(ValueError, 'calibration'):
                        TactileEncoder(config(bad), 16)
                    frozen_cfg = config(metadata)
                    frozen_cfg.freeze_tactile_encoder = True
                    frozen = TactileEncoder(frozen_cfg, 16)
                    self.assertFalse(any(p.requires_grad for p in frozen.backbone.parameters()))
                    self.assertTrue(frozen.proj.weight.requires_grad)
                    del reference, adapter, restored, frozen, optimizer


if __name__ == '__main__':
    unittest.main()
