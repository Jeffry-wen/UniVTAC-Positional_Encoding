"""Global-objective token-parallel wiring; optional real multi-GPU execution."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'encoder'))
import torch
from DINO_TactileRoPE.model import DINOTactileRoPE, _TokenForward
from DINO_TactileRoPE.encoder import TactileRoPEViT


class TokenParallelTest(unittest.TestCase):
    def test_scattered_mask_token_and_gradient_match(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        encoder = TactileRoPEViT().eval()
        images = torch.randn(4, 5, 32, 32)
        images[:, 0] += 30
        mask = torch.tensor([[True, False, True, False]]).expand(4, -1)
        token = torch.randn(1, 1, 384, requires_grad=True)
        probe = torch.randn(4, 5, 384)
        full = encoder.tokens(images, patch_mask=mask, mask_token=token)
        (full*probe).sum().backward()
        grad = token.grad.clone()
        encoder_grad = encoder.blocks[0].rest_project.weight.grad.clone()
        token.grad = None; encoder.zero_grad(set_to_none=True)
        adapter = _TokenForward(encoder)
        parts = [adapter(images[i:i+1], mask[i:i+1], token.expand(4,-1,-1)[i:i+1]) for i in range(4)]
        split = torch.cat(parts)
        (split*probe).sum().backward()
        torch.testing.assert_close(full, split, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(grad, token.grad, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(encoder_grad, encoder.blocks[0].rest_project.weight.grad, atol=5e-5, rtol=5e-4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_global_loss_and_checkpoint_keys(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = DINOTactileRoPE(rope={}, centering='sinkhorn').cuda()
        keys = set(model.state_dict())
        model.enable_token_parallel()
        self.assertEqual(keys, set(model.state_dict()))
        # At least eight samples for an eight-GPU host, still small on one GPU.
        images = torch.randn(8, 5, 224, 224, device='cuda')
        images[:, 0] += 30
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, metrics = model(images)
        self.assertEqual(loss.ndim, 0)  # no per-replica loss vector
        expected = metrics['cls_distillation'] + model.patch_weight*metrics['patch_distillation'] + model.koleo_weight*metrics['koleo'] + model.head_koleo_weight*metrics['head_koleo']
        torch.testing.assert_close(loss.detach(), expected)
        loss.backward()
        self.assertTrue(torch.isfinite(model.mask_token.grad).all())
        self.assertGreater(model.mask_token.grad.abs().sum().item(), 0)
        self.assertIsNone(next(model.teacher.parameters()).grad)
        print(f'CUDA adapter tested on {torch.cuda.device_count()} visible GPU(s).')


if __name__ == '__main__':
    unittest.main()
