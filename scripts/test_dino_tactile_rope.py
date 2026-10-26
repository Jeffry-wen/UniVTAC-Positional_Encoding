"""Behavioral tests for physical fields, relative phases and masked-data leakage."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'encoder'))
import numpy as np
import torch
from DINO_TactileRoPE.encoder import AxisRoPE, TactileRoPEViT
from DINO_TactileRoPE.data import raw_fields
from DINO_TactileRoPE.model import DINOTactileRoPE
from modality import tactile_from_arrays


def sample(size=32):
    x = torch.zeros(2, 5, size, size)
    x[:, 0] = 29.
    x[:, 1] = 2.
    x[:, 2] = -3.
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    x[:, 3], x[:, 4] = xx, yy
    return x


class RoPETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        cls.encoder = TactileRoPEViT().eval()

    def test_relative_rotation_identity(self):
        rope = AxisRoPE(64, 5)
        q, k = torch.randn(2, 4, 64), torch.randn(2, 4, 64)
        a, b = torch.randn(2, 4, 5), torch.randn(2, 4, 5)
        actual = (rope(q, a)*rope(k, b)).sum(-1)
        expected = (q*rope(k, b-a)).sum(-1)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(rope(q, a).norm(dim=-1), q.norm(dim=-1))

    def test_uniform_offset_distinguishes_rest_but_not_pairwise_phase(self):
        rope = AxisRoPE(64, 3)
        q = torch.randn(1, 3, 64)
        d = torch.randn(1, 3, 3)
        offset = torch.tensor([0., 0., 1.])
        self.assertFalse(torch.allclose(rope(q, d), rope(q, d+offset)))
        torch.testing.assert_close((d+offset)[:, 1:]-(d+offset)[:, :1], d[:, 1:]-d[:, :1])

    def test_raw_fields_preserve_signed_displacement_and_depth(self):
        reference = np.array([[2,2],[20,2],[2,20],[20,20]], dtype=np.float32)
        markers = np.stack((reference, reference+[50., -40.])).astype(np.float32)
        data = dict(depth=np.full((24, 32), 23., dtype=np.float32), marker=markers)
        raw = tactile_from_arrays(data, 'depth_deform_raw', 32)
        torch.testing.assert_close(raw[0], torch.full((32,32),23.))
        torch.testing.assert_close(raw[1], torch.full((32,32),50.), atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(raw[2], torch.full((32,32),-40.), atol=1e-4, rtol=1e-5)
        self.assertGreater(raw[3].max().item(), 30)

    def test_calibration_and_zero_reference(self):
        encoder = self.encoder
        deformation, coordinates = encoder.fields(sample())
        torch.testing.assert_close(deformation[0,:,0,0], torch.tensor([.1,-.15,.2]))
        rest = sample(); rest[:,0]=30.; rest[:,1:3]=0.
        self.assertEqual(encoder.fields(rest)[0].count_nonzero(), 0)
        with torch.no_grad():
            original = encoder.pixel_scale.clone()
            encoder.pixel_scale.copy_(torch.tensor([.1,.2]))
            calibrated, _ = encoder.fields(sample())
            encoder.pixel_scale.copy_(original)
        torch.testing.assert_close(calibrated[0,:,0,0], torch.tensor([.01,-.03,.2]))

    def test_masked_deformation_cannot_leak(self):
        images = sample()
        changed = images.clone(); changed[:,:3,:16,:16] += 100
        mask = torch.zeros(2,4,dtype=torch.bool); mask[:,0] = True
        token = torch.zeros(1,1,384)
        with torch.no_grad():
            a = self.encoder.tokens(images, patch_mask=mask, mask_token=token)
            b = self.encoder.tokens(changed, patch_mask=mask, mask_token=token)
            self.assertEqual(a.shape, (2,5,384))
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            self.assertFalse(torch.allclose(self.encoder(images), self.encoder(changed)))

    def test_augments_keep_units_and_coordinate_alignment(self):
        images = sample(224)
        images[:,1] = images[:,3]*.5
        views = DINOTactileRoPE.make_views(None,images)
        for view in views:
            torch.testing.assert_close(view[:,1], view[:,3]*.5)
            torch.testing.assert_close(view[:,0], torch.full_like(view[:,0],29.))
            torch.testing.assert_close(view[:,2], torch.full_like(view[:,2],-3.))

    def test_both_branches_receive_gradients(self):
        model = self.encoder
        model.zero_grad(set_to_none=True)
        out = model(sample())
        (out*torch.randn_like(out)).sum().backward()
        for name in ('rest_project','qkv','fuse'):
            grad = getattr(model.blocks[0],name).weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(),0)


if __name__ == '__main__':
    unittest.main()
