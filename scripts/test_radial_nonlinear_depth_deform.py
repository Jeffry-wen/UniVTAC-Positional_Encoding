"""Radial geometry, actual TPS semantics, preview parity and checkpoint routing."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'encoder'))
from radial_nonlinear_depth_deform import encode_radial_displacement, convert_nonlinear_radial, decode_radial_displacement
from nonlinear_depth_deform import encode_displacement, convert_nonlinear
from modality import INPUT_CHANNELS, tactile_from_arrays
from linear_depth_deform import DEPTH_DEFORM_SPECS, depth_deform_spec, require_depth_deform_encoding


class RadialNonlinearTest(unittest.TestCase):
    def test_core_and_rotational_equivariance(self):
        rng=np.random.default_rng(42)
        theta=rng.uniform(-np.pi,np.pi,1000)
        r=rng.uniform(0,20,1000)
        v=np.column_stack([r*np.cos(theta),r*np.sin(theta)])
        np.testing.assert_allclose(encode_radial_displacement(v),encode_displacement(v),atol=6e-8)
        v*=20
        e=encode_radial_displacement(v).astype(float)-.5
        cross=v[:,0]*e[:,1]-v[:,1]*e[:,0]
        np.testing.assert_allclose(cross,0,atol=3e-5)
        rotated=np.column_stack([-v[:,1],v[:,0]])
        np.testing.assert_allclose(encode_radial_displacement(rotated)-.5,np.column_stack([-e[:,1],e[:,0]]),atol=6e-8)
        np.testing.assert_allclose(encode_radial_displacement([0,0]),[.5,.5])
        np.testing.assert_allclose(encode_radial_displacement([[20,0],[0,-20]]),[[.8,.5],[.5,.2]],atol=6e-8)
        self.assertTrue((np.linalg.norm(e,axis=1)<.5).all())

    def test_tps_full_markers_duplicate_padding_and_shared_depth(self):
        fixed=np.array([[2,2],[6,2],[2,6],[6,6]],dtype=np.float32)
        full=np.stack([fixed,fixed+[200,20]]).astype(np.float32)
        depth=np.full((8,8),27.5,dtype=np.float32)
        raw=dict(depth=depth,marker=np.stack([fixed,fixed]),marker_full=full,press_depth=np.full_like(depth,99))
        actual=tactile_from_arrays(raw,'depth_deform_nonlinear_radial',8).numpy()
        expected=encode_radial_displacement([200,20])
        np.testing.assert_allclose(actual[1:],np.broadcast_to(expected[:,None,None],(2,8,8)),atol=1e-6)
        np.testing.assert_array_equal(actual[0],convert_nonlinear(depth,full,8)[0])
        np.testing.assert_allclose(actual[0],.5,atol=1e-7)
        padded=np.concatenate([full,full[:,-1:]],axis=1)
        np.testing.assert_allclose(convert_nonlinear_radial(depth,padded,8),actual,atol=1e-6)
        self.assertEqual(actual.dtype,np.float32)

    def test_registry_and_incompatible_weights(self):
        expected={f'depth_deform_{n}' for n in (20,80,160,240)}|{'depth_deform_nonlinear','depth_deform_nonlinear_radial'}
        self.assertEqual(set(DEPTH_DEFORM_SPECS),expected)
        self.assertNotIn('depth_deform_3stage',INPUT_CHANNELS)
        mode='depth_deform_nonlinear_radial';spec=depth_deform_spec(mode)
        require_depth_deform_encoding(dict(depth_deform_encoding=spec),mode)
        for wrong in (None,depth_deform_spec('depth_deform_nonlinear'),{**spec,'deform_core_radius_px':40}):
            with self.assertRaises(ValueError):require_depth_deform_encoding(dict(depth_deform_encoding=wrong),mode)
        spec['deform_core_radius_px']=99
        self.assertEqual(depth_deform_spec(mode)['deform_core_radius_px'],20)

    def test_invalid_vectors(self):
        for bad in ([np.nan,1],[1,np.inf],[1,2,3],np.ones((2,3))):
            with self.assertRaises(ValueError):encode_radial_displacement(bad)

    def test_inverse_and_radial_projection(self):
        vectors=np.array([[0,0],[3,4],[20,0],[200,20],[-231,70]],dtype=np.float64)
        decoded,mask=decode_radial_displacement(encode_radial_displacement(vectors))
        np.testing.assert_allclose(decoded,vectors,atol=.002)
        self.assertFalse(mask.any())
        decoded,mask=decode_radial_displacement([[1,1],[.5,.5]],400)
        self.assertTrue(mask[0]);self.assertFalse(mask[1])
        np.testing.assert_allclose(np.linalg.norm(decoded[0]),400,atol=1e-4)
        np.testing.assert_allclose(decoded[0,0],decoded[0,1])


if __name__=='__main__':unittest.main()
