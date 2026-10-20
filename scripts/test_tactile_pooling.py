"""Policy-side tactile pooling, gradients, and legacy configuration contracts."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F
from policy.checkpoints import training_options
from policy.vla_common.tactile_encode import TactileEncoder
from policy.vla_common.tactile_pooling import (
    pool_spatial_tokens, pool_tactile_tokens, pool_act_tactile_outputs,
)


class FakeEncoder(nn.Module):
    embed_dim = 4
    grid_size = 14

    def __init__(self, *args, **kwargs):
        super().__init__()

    def tokens(self, x):
        offsets = torch.arange(197, device=x.device)[None, :, None]
        return x.mean((1, 2, 3))[:, None, None].expand(-1, 197, 4) + offsets

    def patch_tokens(self, x):
        return self.tokens(x)[:, 1:]

    def latent_tokens(self, x):
        return self.patch_tokens(x)[:, :49]

    def forward(self, x):
        return self.tokens(x)[:, 0]


def adapter(method='MAE', mode='full'):
    model = TactileEncoder.__new__(TactileEncoder)
    nn.Module.__init__(model)
    model.backbone = FakeEncoder()
    model.proj = nn.Linear(4, 6)
    model.keys = ['left', 'right']
    model.pretrained_mode = 'univtac'
    model.method, model.mode = method, mode
    model.image_size, model.num_frames = 224, 1
    return model


class PoolingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_values_prefix_time_and_gradients(self):
        tokens = torch.randn(2, 1 + 2 * 7 * 7, 4, requires_grad=True)
        actual = pool_spatial_tokens(tokens, (7, 7), 1)
        expected = torch.stack([
            F.adaptive_avg_pool2d(tokens[:, 1+i*49:1+(i+1)*49].transpose(1, 2)
                                 .reshape(2, 4, 7, 7), (3, 3)).flatten(2).transpose(1, 2)
            for i in range(2)], dim=1).flatten(1, 2)
        torch.testing.assert_close(actual[:, 0], tokens[:, 0])
        torch.testing.assert_close(actual[:, 1:], expected)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue((tokens.grad > 0).all())

    def test_vla_full_cls_frames_and_sensors(self):
        x = torch.randn(2, 3, 3, 16, 16, requires_grad=True)
        for method, per_image in [('MAE', 10), ('DINOv2', 10), ('VAE', 9),
                                  ('I-JEPA', 9), ('V-JEPA', 9)]:
            with self.subTest(method=method):
                model = adapter(method, mode='pool')
                y = model({'left': x, 'right': x + 1})
                self.assertEqual(y.shape, (2, 2 * 3 * per_image, 6))
                y.sum().backward(retain_graph=True)
                model.mode = 'cls'
                pooled = model({'left': x, 'right': x})
                model.mode = 'cls'
                torch.testing.assert_close(pooled, model({'left': x, 'right': x}))
                self.assertEqual(pooled.shape, (2, 6, 6))

    def test_disabled_unchanged_and_strict_weights(self):
        model = adapter()
        x = torch.randn(2, 3, 16, 16)
        torch.testing.assert_close(model._tokens(x), model._unpooled_tokens(x))
        state = model.state_dict()
        model.mode = 'pool'
        self.assertEqual(set(state), set(model.state_dict()))
        model.load_state_dict(state, strict=True)
        self.assertEqual(model._tokens(x).shape, (2, 10, 4))

    def test_external_rectangular_and_temporal_grids(self):
        for method, geometry, count, expected in [
            ('V-JEPA', dict(hw=(320, 240), model=SimpleNamespace(patch_size=16)), 600, 18),
            ('AnyTouch2', dict(hw=(224, 224), model=SimpleNamespace()), 392, 18),
            ('Wan2.2-VAE', dict(wan_patch_size=2), 98, 18),
            ('V-JEPA2', dict(hw=(256, 256), model=SimpleNamespace(config=SimpleNamespace(patch_size=16))), 512, 18),
        ]:
            with self.subTest(method=method):
                tokens = torch.randn(2, count, 4)
                result = pool_tactile_tokens(tokens, SimpleNamespace(**geometry), method, 'vision')
                self.assertEqual(result.shape, (2, expected, 4))
        result = pool_tactile_tokens(torch.randn(2, 64, 4), None, 'ResNet', 'univtac')
        self.assertEqual(result.shape, (2, 9, 4))

    def test_act_maps_and_flat_cls_features(self):
        class ACT(nn.Module):
            tactile_type = 'full'
            pooling3x3 = True
            pooling_pretrained_mode = 'univtac'
            pooling_method = 'MAE'
            backbone = FakeEncoder()

            @pool_act_tactile_outputs
            def forward(self, feature, pos):
                return [feature], [pos]

        model = ACT()
        flat, positions = torch.randn(2, 4, 1, 197), torch.randn(2, 8, 1, 197)
        feature, pos = model(flat, positions)
        self.assertEqual(feature[0].shape, (2, 4, 1, 10))
        self.assertEqual(pos[0].shape, (2, 8, 1, 10))
        torch.testing.assert_close(feature[0][:, :, :, 0], flat[:, :, :, 0])
        spatial = torch.randn(2, 4, 7, 7)
        feature, _ = model(spatial, spatial)
        torch.testing.assert_close(feature[0], F.adaptive_avg_pool2d(spatial, (3, 3)))
        model.tactile_type = 'cls'
        single = torch.randn(2, 4, 1, 1)
        feature, _ = model(single, single)
        self.assertIs(feature[0], single)
        model.tactile_type, model.pooling3x3 = 'full', False
        feature, _ = model(flat, positions)
        self.assertIs(feature[0], flat)

    def test_legacy_and_new_config_restore(self):
        self.assertEqual(training_options({})['tactile_type'], 'cls')
        self.assertEqual(training_options({'routing': {'tactile_type': 'full'}})['tactile_type'], 'full')
        self.assertEqual(training_options({'routing': {'tactile_type': 'pool'}})['tactile_type'], 'pool')
        from policy.unified_deploy import make_vla_config
        from policy.train_unified import make_config
        camera = 'observation.images.cam_high'
        for kind in ('pi05', 'starvla_groot', 'starvla_pi'):
            for mode in ('cls', 'full', 'pool'):
                with self.subTest(policy=kind, mode=mode):
                    args = SimpleNamespace(policy=kind, chunk_size=3, n_action_steps=2,
                        image_size=224, tactile_mode='encode', tactile_input_mode='rgb_only',
                        tactile_type=mode, tactile_insert_location='encoder', encoder_ckpt=Path('/tmp/unused'),
                        encoder_pretrained_mode='univtac', tactile_num_frames=1,
                        tactile_frame_stride=1, action_mode='absolute_joint', device='cpu')
                    config = make_config(args, {}, [camera], ['observation.images.tac_left'])
                    self.assertEqual(config.tactile_type, mode)
                    saved = dict(base={}, input_features={camera: [3, 224, 224], 'observation.state': [8]},
                        output_features={'action': [8]}, state_mode='absolute_joint', action_gap=1,
                        chunk_size=3, n_action_steps=2, tactile_encoder_config=dict(
                            method='MAE', size='S', image_size=224, in_chans=3,
                            input_mode='rgb_only', color_order='RGB'))
                    options = dict(tactile_mode='encode', tactile_input_mode='rgb_only', tactile_type=mode,
                        tactile_insert_location='encoder', action_mode='absolute_joint', chunk_size=3)
                    run = dict(policy=kind, directory=Path('/tmp'), config=saved, options=options,
                               camera_keys=[camera])
                    restored = make_vla_config(run, 'cpu', ROOT / 'resources/pretrained_models')
                    self.assertEqual(restored.tactile_type, mode)

    def test_act_builder_modes_and_strict_legacy_load(self):
        sys.path.insert(0, str(ROOT / 'policy/ACT/detr'))
        sys.path.insert(0, str(ROOT / 'encoder'))
        from policy.ACT.detr.models.backbone import build_tactile_backbone
        state = dict(method='MAE', size='S', input_mode='rgb_only', color_order='RGB',
                     image_size=224, in_chans=3)
        args = SimpleNamespace(lr_tactile_backbone=0, tactile_masks=False,
            tactile_type='full', hidden_dim=512, position_embedding='sine',
            tactile_encoder_method='MAE', tactile_encoder_config=state,
            tactile_ckpt=None, tactile_encoder_size='S', tactile_input_mode='rgb_only')
        x = torch.randn(2, 3, 16, 16)
        with patch('vit_encoder.TactileViT', FakeEncoder):
            full = build_tactile_backbone(args)
            expected = full(x)
            args.tactile_type = 'pool'
            pooled = build_tactile_backbone(args)
            pooled.load_state_dict(full.state_dict(), strict=True)
            feature, position = pooled(x)
            self.assertEqual(feature[0].shape, (2, 4, 1, 10))
            self.assertEqual(position[0].shape, (2, 512, 1, 10))
            torch.testing.assert_close(feature[0][..., 0], expected[0][0][..., 0])
            args.tactile_type = 'cls'
            cls = build_tactile_backbone(args)
            self.assertEqual(cls(x)[0][0].shape, (2, 4, 1, 1))


if __name__ == '__main__':
    unittest.main()
