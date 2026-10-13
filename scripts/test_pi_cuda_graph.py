"""CUDA Graph cache, input ownership and parameter lifecycle regressions."""
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.cuda_graph import InferenceCudaGraph, CudaGraphPolicyMixin


class ToyPolicy(CudaGraphPolicyMixin, nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.,device=device))
        self._action_cuda_graph = InferenceCudaGraph(self,self.compute,max_graphs=1)

    def compute(self,batch,noise):
        return batch['views'][0]*self.weight+noise

    def forward(self,batch,noise):
        return self._action_cuda_graph(batch,noise)


class CacheTests(unittest.TestCase):
    def test_cpu_fallback(self):
        model=ToyPolicy().eval()
        out=model({'views':[torch.ones(2)],'task':['lift can']},torch.ones(2))
        torch.testing.assert_close(out,torch.full((2,),3.))
        self.assertEqual(model._action_cuda_graph.captures,0)
        self.assertEqual(model._action_cuda_graph.eager_calls,1)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_dynamic_inputs_owned_outputs_and_shape_limit(self):
        model=ToyPolicy('cuda').eval()
        batch={'views':[torch.ones(2,device='cuda')],'task':['lift can']}
        noise=torch.ones(2,device='cuda')
        first=model(batch,noise);saved=first.clone()
        batch['views'][0].fill_(4);noise.fill_(3)
        out=model(batch,noise)
        torch.testing.assert_close(out,torch.full((2,),11.,device='cuda'))
        torch.testing.assert_close(first,saved,rtol=0,atol=0)
        model({'views':[torch.ones(3,device='cuda')],'task':['lift can']},torch.ones(3,device='cuda'))
        self.assertEqual(model._action_cuda_graph.captures,1)
        self.assertEqual(model._action_cuda_graph.replays,2)
        self.assertEqual(model._action_cuda_graph.eager_calls,1)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_weight_load_train_and_device_move_invalidate(self):
        model=ToyPolicy('cuda').eval()
        batch={'views':[torch.ones(2,device='cuda')]};noise=torch.ones(2,device='cuda')
        model(batch,noise)
        model.load_state_dict({'weight':torch.tensor(5.,device='cuda')},assign=True)
        self.assertFalse(model._action_cuda_graph.entries)
        torch.testing.assert_close(model(batch,noise),torch.full((2,),6.,device='cuda'))
        model.train();self.assertFalse(model._action_cuda_graph.entries)
        model.eval();model(batch,noise)
        model.to('cpu');self.assertFalse(model._action_cuda_graph.entries)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_pi05_fp32_vision_is_stable_and_graph_safe(self):
        from transformers import SiglipVisionConfig, SiglipVisionModel
        from policy.pi05.modeling_pi05 import PaliGemmaWithExpertModel
        vision=SiglipVisionModel(SiglipVisionConfig(hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,image_size=28,patch_size=14)).cuda()
        vlm=PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
        nn.Module.__init__(vlm)
        vlm.paligemma=nn.Module();vlm.paligemma.model=nn.Module()
        vlm.paligemma.model.vision_tower=vision
        vlm.paligemma.model.multi_modal_projector=nn.Linear(32,16).cuda()
        vlm.freeze_vision_encoder=vlm.train_expert_only=False
        vlm.eval()
        image=torch.rand(1,3,28,28,device='cuda')
        runner=InferenceCudaGraph(vlm,vlm.embed_image)
        enabled=torch.backends.cuda.mem_efficient_sdp_enabled()
        with torch.no_grad():
            eager=vlm.embed_image(image)
            torch.testing.assert_close(eager,vlm.embed_image(image),rtol=0,atol=0)
            torch.testing.assert_close(eager,runner(image),rtol=0,atol=0)
            changed=image+0.1
            torch.testing.assert_close(vlm.embed_image(changed),runner(changed),rtol=0,atol=0)
        self.assertEqual(enabled,torch.backends.cuda.mem_efficient_sdp_enabled())


if __name__=='__main__':
    unittest.main()
