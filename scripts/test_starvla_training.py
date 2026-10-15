"""Numerical and image-isolation checks for batched StarVLA vision attention."""
from pathlib import Path
import sys
from types import MethodType
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionAttention
from policy.starvla_groot.qwen_vl_interface import _batched_vision_attention


class VisionBatchTests(unittest.TestCase):
    def attention(self, device, dtype):
        config = Qwen3_5VisionConfig(hidden_size=64, num_heads=4)
        config._attn_implementation = "sdpa"
        module = Qwen3_5VisionAttention(config).to(device=device, dtype=dtype).train()
        module._univtac_original_forward = module.forward
        module._univtac_vision_layout = {}
        return module

    def compare(self, device, dtype, lengths):
        torch.manual_seed(8)
        module = self.attention(device, dtype)
        total = sum(lengths)
        x = torch.randn(total, 64, device=device, dtype=dtype, requires_grad=True)
        angles = torch.randn(total, 16, device=device)
        positions = angles.cos(), angles.sin()
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
        expected = module(x, cu, position_embeddings=positions)
        expected.float().square().sum().backward()
        gradients = [x.grad.clone(), *[p.grad.clone() for p in module.parameters()]]
        module.zero_grad(set_to_none=True)
        x.grad = None
        keys = set(module.state_dict())
        module.forward = MethodType(_batched_vision_attention, module)
        actual = module(x, cu, position_embeddings=positions)
        actual.float().square().sum().backward()
        tolerance = 0.025 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        for a, b in zip([x.grad, *[p.grad for p in module.parameters()]], gradients):
            torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
        self.assertEqual(keys, set(module.state_dict()))
        # Changing another image must never affect the first image's output.
        modified = x.detach().clone()
        modified[lengths[0]:] += 100
        changed = module(modified, cu, position_embeddings=positions)
        torch.testing.assert_close(changed[:lengths[0]], actual[:lengths[0]], rtol=0, atol=0)

    def test_cpu_uniform_and_ragged_outputs_gradients(self):
        for lengths in ([16] * 3, [8, 16, 24]):
            self.compare("cpu", torch.float32, lengths)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_fp32_and_bf16_outputs_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            self.compare("cuda:0", dtype, [16] * 3)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
