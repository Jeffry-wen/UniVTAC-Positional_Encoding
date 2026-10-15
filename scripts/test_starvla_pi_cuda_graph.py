"""Exact eager/graph parity for layer-wise Qwen features and action chunks."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.starvla_pi.cuda_graph import LayerwiseQwen35TextCudaGraph, LayerwiseQwen35PrefixCudaGraph
from policy.starvla_pi.modeling_starvla_pi import StarvlaPiPolicy
from scripts import test_starvla_pi as helpers

TinyQwen = helpers.TinyQwen


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class LayerwiseCudaGraphTests(unittest.TestCase):
    @patch("policy.starvla_pi.modeling_starvla_pi.QwenVLInterface", TinyQwen)
    def test_fp32_action_head_with_bf16_backbone_training_and_prediction(self):
        helper = helpers.StarvlaPiTests()
        config = helper.config()
        config.device, config.use_cuda_graph = "cuda", False
        torch.backends.cuda.matmul.allow_tf32 = True
        policy = StarvlaPiPolicy(config)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        policy.qwen_vl.to(dtype=torch.bfloat16)
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in helper.batch().items()}
        dtypes = []
        hooks = [module.register_forward_hook(lambda module, args, output: dtypes.append(output.dtype))
                 for module in policy.action_head.modules() if isinstance(module, torch.nn.Linear)]
        try:
            loss, _ = policy(batch, return_loss_dict=False)
            loss.backward()
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(dtypes and all(dtype == torch.float32 for dtype in dtypes))
            dtypes.clear()
            actions = policy.predict_action_chunk(batch)
            self.assertEqual(actions.dtype, torch.float32)
            self.assertTrue(dtypes and all(dtype == torch.float32 for dtype in dtypes))
        finally:
            for hook in hooks:
                hook.remove()

    def make_qwen(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from policy.starvla_groot.qwen_vl_interface import QwenVLInterface
        config = Qwen3_5Config(
            image_token_id=29, video_token_id=30, vision_start_token_id=28, vision_end_token_id=27,
            text_config=dict(vocab_size=32, hidden_size=128, intermediate_size=256,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16,
                linear_num_key_heads=2, linear_num_value_heads=4,
                layer_types=["linear_attention", "full_attention"],
                rope_parameters={"rope_type":"default", "rope_theta":10000.,
                                 "partial_rotary_factor":1., "mrope_section":[6,5,5]}),
            vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,
                out_hidden_size=128,patch_size=2,temporal_patch_size=1,spatial_merge_size=2,num_position_embeddings=16))
        config._attn_implementation = "sdpa"
        qv = QwenVLInterface.__new__(QwenVLInterface)
        torch.nn.Module.__init__(qv)
        qv.model = Qwen3_5ForConditionalGeneration(config).to("cuda", torch.bfloat16).eval()
        qv.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        qv._patch_size, qv._merge_size, qv._temporal_patch_size = 2, 2, 1
        qv._target_h = qv._target_w = 8
        qv._img_mean, qv._img_std = [0.5]*3, [0.25]*3
        qv.use_cuda_graph, qv._text_cuda_graph, qv._prefix_cuda_graph = False, None, None
        qv.eval()
        def text_inputs(instructions, views):
            rows = [[1]+[t for _ in range(views) for t in (28,29,29,29,29,27)]
                    +[ord(c)%20+1 for c in instruction] for instruction in instructions]
            n = max(map(len,rows))
            ids = torch.tensor([[0]*(n-len(row))+row for row in rows],device="cuda")
            return ids,(ids!=0).long()
        qv.build_text_inputs = text_inputs
        return qv

    @torch.no_grad()
    def test_text_and_full_prefix_all_layers_padding_refresh_and_cache(self):
        qv = self.make_qwen()
        images = torch.rand(2,2,3,8,8,device="cuda")
        extra = torch.randn(2,2,128,device="cuda",dtype=torch.bfloat16)
        for count in (1,2):
            prefix = LayerwiseQwen35PrefixCudaGraph(qv,count,max_graphs=1)
            text = LayerwiseQwen35TextCudaGraph(qv.model.model.language_model,count,max_graphs=1)
            for tactile in (extra,extra+.2):
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    eager,mask = qv.forward_with_prefix_tokens(images,("a","bbb"),tactile,return_hidden_states=True)
                    expected = torch.stack(eager[-count:])
                    actual,actual_mask = prefix(images,("a","bbb"),tactile)
                    fallback,fallback_mask = qv.forward_with_prefix_tokens(images,("a","bbb"),tactile,return_hidden_states=True,text_graph=text)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                torch.testing.assert_close(actual_mask,mask,rtol=0,atol=0)
                torch.testing.assert_close(fallback,expected,rtol=0,atol=0)
                torch.testing.assert_close(fallback_mask,mask,rtol=0,atol=0)
                saved=actual.clone()
                images=images*.9
            self.assertEqual(prefix.captures,1)
            self.assertEqual(text.captures,1)
            self.assertIsNone(prefix(images,("new","prompt"),extra))
            torch.testing.assert_close(actual,saved,rtol=0,atol=0)
            prefix.clear();text.clear()
            self.assertFalse(prefix.entries)
            self.assertFalse(text.entries)
        # No extra tokens: compare the entire multimodal prefix to native HF.
        prefix = LayerwiseQwen35PrefixCudaGraph(qv,2)
        inputs=qv.build_qwenvl_inputs(images,("a","bbb"))
        expected=torch.stack(qv(**inputs,output_hidden_states=True,use_cache=False).hidden_states[-2:])
        with torch.autocast("cuda",dtype=torch.bfloat16):
            actual,_=prefix(images,("a","bbb"),None)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    @patch("policy.starvla_pi.modeling_starvla_pi.QwenVLInterface",TinyQwen)
    @torch.no_grad()
    def test_action_output_ownership_masks_steps_and_lifecycle(self):
        helper=helpers.StarvlaPiTests()
        config=helper.config();config.device="cuda"
        config.cuda_graph_max_graphs=1
        policy=StarvlaPiPolicy(config).eval()
        batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in helper.batch().items()}
        for full in (True,False):
            config.cuda_graph_full_chunk=full
            for change in (False,True):
                if change:
                    batch["observation.state"].add_(.5)
                    batch["is_pad"].zero_()
                config.use_cuda_graph=False;torch.manual_seed(7)
                expected=policy.predict_action_chunk(batch)
                config.use_cuda_graph=True;torch.manual_seed(7)
                actual=policy.predict_action_chunk(batch)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                saved=actual.clone()
                policy.predict_action_chunk(batch)
                torch.testing.assert_close(actual,saved,rtol=0,atol=0)
        self.assertEqual(policy._chunk_cuda_graph.captures,1)
        self.assertEqual(policy._action_cuda_graph.captures,1)
        # Timestep count changes must not replay an obsolete denoising loop.
        policy.action_head.num_inference_timesteps=3
        config.use_cuda_graph=False;torch.manual_seed(9);expected=policy.predict_action_chunk(batch)
        config.use_cuda_graph=True;torch.manual_seed(9);actual=policy.predict_action_chunk(batch)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertGreater(policy._action_cuda_graph.eager_calls,0)
        policy.load_state_dict(policy.state_dict())
        self.assertFalse(policy._action_cuda_graph.entries)
        policy.predict_action_chunk(batch);policy.train()
        self.assertFalse(policy._action_cuda_graph.entries)
        policy.eval();policy.predict_action_chunk(batch);policy.to(dtype=torch.float64)
        self.assertFalse(policy._action_cuda_graph.entries)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
