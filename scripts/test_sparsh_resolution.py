"""Sparsh resolution, official weight loading, gradients and saved policy restore."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoder.external import ExternalEncoder
from encoder.pretrained_registry import constructor_metadata
from policy.vla_common.tactile_pooling import pool_spatial_tokens, pool_act_tactile_outputs
from policy.vla_common.tactile_encode import TactileEncoder


class SparshResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_official_initializers_square_inputs_tokens_and_gradients(self):
        for method, size, count, dim in (
            ('MAE', 'S', 196, 384), ('DINOv2', 'B', 196, 768),
            ('I-JEPA', 'S', 196, 384), ('V-JEPA', 'S', 392, 384),
        ):
            with self.subTest(method=method):
                model = ExternalEncoder('tactile', method, size).eval()
                images = torch.rand(1, 4, 3, 24, 32, requires_grad=True)
                self.assertEqual(model.hw, (224, 224))
                self.assertEqual(model.metadata['image_size'], [224, 224])
                self.assertEqual(model.preprocess(images).shape, (1, 4, 3, 224, 224))
                with torch.no_grad():
                    source = model.preprocess(images).flip(1)
                    source = source.transpose(1, 2) if method == 'V-JEPA' else source.flatten(1, 2)
                    features = model.model.forward_features(source)
                    patches = features['x_norm_patchtokens']
                    prefix = features['x_norm_regtokens']
                    self.assertEqual(patches.shape, (1, count, dim))
                    expected = torch.cat((prefix, pool_spatial_tokens(patches, (14, 14))), dim=1)
                pooled = model.pooled_tokens(images)
                self.assertEqual(pooled.shape, (1, (18 if method == 'V-JEPA' else 9) + (method != 'MAE'), dim))
                torch.testing.assert_close(pooled, expected)
                pooled[..., 0].sum().backward()
                self.assertTrue(torch.isfinite(images.grad).all())
                self.assertGreater(images.grad.abs().sum().item(), 0)
                del features, patches, prefix, pooled, model, images

    def test_anytouch_cls_sensor_exclusion_and_policy_pooling_routes(self):
        model = ExternalEncoder('tactile', 'AnyTouch2', 'B').eval()
        images = torch.rand(1, 4, 3, 24, 32, requires_grad=True)
        with torch.no_grad():
            raw = model.model(model.preprocess(images))
            self.assertEqual(raw.shape, (1, 398, 512))
            expected = torch.cat((raw[:, :1], pool_spatial_tokens(raw[:, 6:], (14, 14))), dim=1)
            torch.testing.assert_close(model.patch_tokens(images), raw[:, 6:])
            torch.testing.assert_close(model(images), raw[:, 6:].mean(1))
        # Exercise the VLA route actually used by StarVLA-GR00T.
        adapter = TactileEncoder.__new__(TactileEncoder)
        torch.nn.Module.__init__(adapter)
        adapter.backbone = model
        adapter.pretrained_mode, adapter.method = 'tactile', 'AnyTouch2'
        for mode, size in [('pool', 3), ('pool3x3', 3), ('pool5x5', 5), ('pool7x7', 7)]:
            adapter.mode = mode
            with torch.no_grad():
                actual = adapter._tokens(images)
                self.assertEqual(actual.shape, (1, 1 + 2*size*size, 512))
                torch.testing.assert_close(actual[:, :1], raw[:, :1])
                if size == 3:
                    torch.testing.assert_close(actual, expected)

        class ACT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = model
                self.tactile_type, self.pooling_pretrained_mode = 'full', 'tactile'
                self.pooling_size = 3
                self.position_embedding = torch.nn.Identity()

            @pool_act_tactile_outputs
            def forward(self, x):
                raise AssertionError('Pooling must preserve the prefix before ACT patch-only output')

        features, positions = ACT()(images)
        self.assertEqual(features[0].shape, (1, 512, 1, 19))
        self.assertEqual(positions[0].shape, features[0].shape)
        torch.testing.assert_close(features[0].squeeze(2).transpose(1, 2), expected)
        features[0].square().mean().backward()
        self.assertTrue(torch.isfinite(images.grad).all())
        cls_grad = model.model.touch_model.embeddings.class_embedding.grad
        self.assertIsNotNone(cls_grad)
        self.assertGreater(cls_grad.abs().sum().item(), 0)

    def test_metadata_restore_preserves_resolution_weights_and_outputs(self):
        for method in ('MAE', 'V-JEPA'):
            for hw in ([224, 224], [320, 240]):
                with self.subTest(method=method, hw=hw):
                    original = ExternalEncoder('tactile', method, 'S',
                                               metadata={'image_size': hw}, pretrained=False).eval()
                    metadata = constructor_metadata(original.metadata)
                    restored = ExternalEncoder('tactile', method, 'S',
                                               metadata=metadata, pretrained=False).eval()
                    restored.load_state_dict(original.state_dict(), strict=True)
                    images = torch.rand(1, 4, 3, 24, 32)
                    with torch.no_grad():
                        expected = original.patch_tokens(images)
                        actual = restored.patch_tokens(images)
                        torch.testing.assert_close(actual, expected)
                        torch.testing.assert_close(restored(images), actual.mean(1))
                    count = (hw[0] // 16) * (hw[1] // 16) * (2 if method == 'V-JEPA' else 1)
                    self.assertEqual(actual.shape, (1, count, 384))
                    self.assertEqual(restored.hw, tuple(hw))
                    del original, restored, actual, expected


if __name__ == '__main__':
    unittest.main()
