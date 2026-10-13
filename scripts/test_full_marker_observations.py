"""Check full-coordinate capture leaves legacy marker output/RNG unchanged."""
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
BACKEND=Path('resources/third_party/TacEx/source/tacex/tacex/simulation_approaches/fem_based/sim/tactile_sensor_sapienipc_modified.py')


def method(source,name,namespace):
    node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated sensor method>','exec'),namespace)
    return namespace[name]


def sensor():
    x,y=np.meshgrid(np.linspace(40,280,9),np.linspace(35,205,7))
    ref=torch.tensor(np.column_stack((x.ravel(),y.ravel(),np.ones(x.size))),dtype=torch.float32)
    current=ref+torch.tensor([-120.,60.,0.])
    return SimpleNamespace(reference_surface_vertices_camera=ref,marker_surf_idx=np.arange(63)[:,None],
                           marker_weight=np.ones((63,1),np.float32),
                           get_camera_pose_world=lambda:None,get_surface_vertices_camera=lambda **kw:current,
                           gen_marker_uv=lambda points:points[:,:2],tactile_img_width=320,tactile_img_height=240,
                           marker_lose_tracking_probability=0.,marker_random_noise=0.,num_markers=63,normalize=False)


class FullMarkerTest(unittest.TestCase):
    def test_capture_preserves_legacy_values_and_rng(self):
        original=subprocess.check_output(['git','show',f'HEAD:{BACKEND}'],cwd=ROOT,text=True)
        current=(ROOT/BACKEND).read_text()
        # The isolated legacy method requests cuda:0; use a CPU tensor constructor
        # to test its numerical behavior without starting Isaac Sim or CUDA.
        ns=lambda:dict(np=np,torch=SimpleNamespace(tensor=lambda data,device:torch.tensor(data)))
        old=method(original,'gen_marker_flow',ns());new=method(current,'gen_marker_flow',ns())
        a,b=sensor(),sensor()
        np.random.seed(123);expected=old(a);expected_rng=np.random.get_state()
        np.random.seed(123);actual=new(b);actual_rng=np.random.get_state()
        np.testing.assert_array_equal(actual.numpy(),expected.numpy())
        np.testing.assert_array_equal(actual_rng[1],expected_rng[1])
        self.assertEqual(actual_rng[2:],expected_rng[2:])
        self.assertEqual(b.full_marker_flow.shape,(2,63,2))
        self.assertTrue((b.full_marker_flow[1,:,0]<0).any())
        self.assertTrue((b.full_marker_flow[1,:,1]>240).any())
        self.assertLess(b.full_marker_in_view.sum(),63)
        self.assertTrue((actual[1,:,0]>0).all())
        self.assertTrue((actual[1,:,1]<240).all())

    def test_public_full_fields_keep_stable_ids_and_outside_coordinates(self):
        b=sensor()
        backend=method((ROOT/BACKEND).read_text(),'gen_marker_flow',dict(np=np,torch=SimpleNamespace(tensor=lambda data,device:torch.tensor(data))))
        backend(b)
        wrapper=SimpleNamespace(device='cpu',sensor=SimpleNamespace(marker_motion_simulator=SimpleNamespace(marker_motion_sim=b)))
        get=method((ROOT/'resources/envs/sensors/tactile.py').read_text(),'get_observations',dict(torch=torch,np=np))
        result=get(wrapper,['marker_full','marker_id','marker_in_view'])
        np.testing.assert_array_equal(result['marker_full'].numpy(),b.full_marker_flow)
        self.assertEqual(result['marker_id'].tolist(),list(range(63)))
        np.testing.assert_array_equal(result['marker_in_view'].numpy(),b.full_marker_in_view)


if __name__=='__main__':unittest.main()
