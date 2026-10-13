"""Compare PI05/N0 eager and CUDA Graph inference on real checkpoint observations."""
import argparse
import ast
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.unified_deploy import Policy
from scripts.test_policy import hdf5_observation


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = fn()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - start) * 1000


def metrics(values):
    return {'median_ms': statistics.median(values), 'mean_ms': statistics.mean(values),
            'samples': len(values)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--task', default='lift_can')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=12)
    args = parser.parse_args()
    if args.iterations < 4:
        parser.error('Use at least four timing iterations')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    from transformers.initialization import no_init_weights
    # Policy strictly loads the complete state dict before it can be used.
    with no_init_weights():
        policy = Policy({'checkpoint':args.checkpoint, 'task_name':args.task,
                         'device':'cuda', 'use_cuda_graph':False})
    model = policy.model
    if policy.kind not in ('pi05', 'n0vtla'):
        raise ValueError('Expected a PI05 or N0VTLA checkpoint')
    observations = [hdf5_observation(args.episode, i, policy.run) for i in (0, 10, 20)]
    batches = [policy.encode_obs(obs) for obs in observations]
    shape = (1, model.config.chunk_size, 32)
    noise = torch.randn(shape, device='cuda')
    report = {'checkpoint':args.checkpoint, 'policy':policy.kind,
              'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__,
              'scope':'full action chunk, including GPU image preprocessing, vision, prefix and denoising',
              'denoise_steps':model.config.num_inference_steps,
              'execution_horizon':policy.execution_horizon, 'comparisons':[]}
    # Compare the fixed-count N0 loop with the original GPU-tensor while loop.
    if policy.kind == 'n0vtla':
        source = subprocess.check_output(['git','show','HEAD:policy/n0vtla/native/n0vtla_policy.py'],text=True)
        cls = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='N0VTLAPolicy')
        func = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_sample_actions_predictor')
        func.decorator_list=[]
        space={'torch':torch}
        exec(compile(ast.Module(body=[func],type_ignores=[]),'<original N0 sampler>','exec'),space)
        original=types.MethodType(space[func.name],model.model)
        from policy.n0vtla.observation_adapter import make_observation
        old=original(torch.device('cuda'),make_observation(batches[0]),noise=noise,num_steps=model.config.num_inference_steps)[...,:8]
        new=model.predict_action_chunk(batches[0],noise=noise)
        torch.testing.assert_close(old,new,rtol=0,atol=0)
        report['original_loop_exact_equal']=True
    for batch in batches:
        model.predict_action_chunk(batch,noise=noise)
    before = torch.cuda.memory_allocated()
    model.config.use_cuda_graph=True
    _, report['capture_first_replay_ms']=timed(lambda:model.predict_action_chunk(batches[0],noise=noise))
    report['graph_extra_allocated_MiB']=(torch.cuda.memory_allocated()-before)/2**20
    for frame,batch in zip((0,10,20),batches):
        for seed in (0,1,42):
            torch.manual_seed(seed)
            n=torch.randn(shape,device='cuda')
            model.config.use_cuda_graph=False
            eager=model.predict_action_chunk(batch,noise=n)
            model.config.use_cuda_graph=True
            graph=model.predict_action_chunk(batch,noise=n)
            print(f"Check frame={frame} seed={seed}",flush=True)
            torch.testing.assert_close(eager,graph,rtol=1e-5,atol=1e-6)
            delta=(eager-graph).abs()
            report['comparisons'].append({'frame':frame,'seed':seed,'max_abs':delta.max().item(),
                                          'exact_equal':torch.equal(eager,graph)})
    retained=model.predict_action_chunk(batches[0],noise=noise)
    saved=retained.clone()
    changed=model.predict_action_chunk(batches[2],noise=noise+0.5)
    torch.testing.assert_close(retained,saved,rtol=0,atol=0)
    assert not torch.equal(changed,retained),'Replay ignored changed inputs'
    report['output_ownership_verified']=True
    # First-capture warmup and every subsequent replay must leave RNG behavior intact.
    model.clear_cuda_graphs()
    model.config.use_cuda_graph=False
    torch.manual_seed(991)
    eager=model.predict_action_chunk(batches[0]);eager_rng=torch.cuda.get_rng_state()
    model.config.use_cuda_graph=True
    torch.manual_seed(991)
    graph=model.predict_action_chunk(batches[0]);graph_rng=torch.cuda.get_rng_state()
    torch.testing.assert_close(eager,graph,rtol=1e-5,atol=1e-6)
    assert torch.equal(eager_rng,graph_rng),'Graph warmup/capture changed CUDA RNG consumption'
    report['rng_verified']=True
    times={False:[],True:[]}
    for i in range(args.iterations):
        for enabled in (False,True,True,False):
            model.config.use_cuda_graph=enabled
            _,ms=timed(lambda:model.predict_action_chunk(batches[i%3],noise=noise))
            times[enabled].append(ms)
    report['eager']=metrics(times[False]);report['cuda_graph']=metrics(times[True])
    report['speedup']=report['eager']['median_ms']/report['cuda_graph']['median_ms']
    end_to_end={False:[],True:[]}
    for i in range(args.iterations):
        for enabled in (False,True,True,False):
            model.config.use_cuda_graph=enabled
            _,ms=timed(lambda:policy.predict_chunk(observations[i%3]))
            end_to_end[enabled].append(ms)
    report['adapter_eager']=metrics(end_to_end[False]);report['adapter_cuda_graph']=metrics(end_to_end[True])
    report['adapter_speedup']=report['adapter_eager']['median_ms']/report['adapter_cuda_graph']['median_ms']
    report['graph_captures']=model._action_cuda_graph.captures
    report['graph_replays']=model._action_cuda_graph.replays
    assert model._action_cuda_graph.replays>1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
