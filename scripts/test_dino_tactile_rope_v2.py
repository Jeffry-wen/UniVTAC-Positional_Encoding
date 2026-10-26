"""V2 invariants, masked-information isolation and checkpoint round trip."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'encoder'))
import torch
from DINO_TactileRoPE_V2.encoder import TactileRoPEViTV2, load_encoder
from test_dino_tactile_rope import sample


class V2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = TactileRoPEViTV2().eval()

    def test_static_zero_and_offset(self):
        block = self.model.blocks[0]
        z = torch.randn(2, 4, 384)
        d = torch.zeros(2, 4, 3)
        self.assertEqual(block.static_increment(z, d).count_nonzero(), 0)
        self.assertGreater(block.static_increment(z, d + 1).abs().sum(), 0)

    def test_partial_relative_identity(self):
        block = self.model.blocks[0]
        q, k = torch.randn(2, 6, 4, 64), torch.randn(2, 6, 4, 64)
        a, b = torch.randn(2, 4, 5), torch.randn(2, 4, 5)
        actual = (block.rotate_relation(q, a) * block.rotate_relation(k, b)).sum(-1)
        expected = (q * block.rotate_relation(k, b - a)).sum(-1)
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)
        torch.testing.assert_close(block.rotate_relation(q, a)[..., :24], q[..., :24])

    def test_mask_hides_values_and_missingness(self):
        a = sample()
        b = a.clone(); b[:, :3, :16, :16] = float('nan')
        mask = torch.zeros(2, 4, dtype=torch.bool); mask[:, 0] = True
        token = torch.randn(1, 1, 384)
        with torch.no_grad():
            x = self.model.tokens(a, mask, token)
            y = self.model.tokens(b, mask, token)
        torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_missing_is_not_rest(self):
        a = sample(); a[:, 0] = 30.; a[:, 1:3] = 0.
        b = a.clone(); b[:, :3, :16, :16] = float('nan')
        with torch.no_grad():
            x, y = self.model.tokens(a), self.model.tokens(b)
        self.assertTrue(torch.isfinite(y).all())
        self.assertFalse(torch.allclose(x, y))

    def test_gradients_both_branches(self):
        self.model.zero_grad(set_to_none=True)
        out = self.model(sample())
        (out * torch.randn_like(out)).sum().backward()
        block = self.model.blocks[0]
        for module in (block.rest_project, block.rest_output, block.qkv, block.context_output):
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(module.weight.grad.abs().sum(), 0)

    def test_checkpoint_and_variable_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'encoder.pth'
            torch.save(dict(method='DINO_TactileRoPE_V2', architecture='tactile_rope_v2',
                            size='S', input_mode='depth_deform_raw',
                            model_config={'rope': self.model.config}, encoder=self.model.state_dict()), path)
            restored = load_encoder(path)
            with torch.no_grad():
                for side in (32, 48):
                    torch.testing.assert_close(restored(sample(side)), self.model(sample(side)), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
