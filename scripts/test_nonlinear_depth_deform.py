"""Hybrid mapping, genuine TPS interpolation, and new-mode reader parity."""
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'encoder'))
from nonlinear_depth_deform import convert_nonlinear, encode_displacement, _tps_interpolation_map
from modality import tactile_from_arrays, read_tactile, depth_deform_spec
from train_common import TactileFrames
from policy.n0vtla.observation_adapter import read_native_tactile
from scripts.test_policy_input_recording import make_policy, observation
from scripts.test_policy import hdf5_observation
from image_io import encode_rgb, STANDARD_ENCODING


class NonlinearDepthDeformTest(unittest.TestCase):
    def test_linear_endpoints_and_smooth_monotone_symmetric_tails(self):
        x = np.array([-20., -10., 0., 10., 20.], np.float32)
        np.testing.assert_allclose(encode_displacement(x), [.2, .35, .5, .65, .8], atol=1e-7)
        x = np.linspace(-100, 100, 2001, dtype=np.float32)
        y = encode_displacement(x)
        self.assertTrue((np.diff(y) > 0).all())
        self.assertTrue(((y > 0) & (y < 1)).all())
        np.testing.assert_allclose(y + encode_displacement(-x), 1., atol=1e-7)
        for boundary in (-20., 20.):
            nearby = encode_displacement(np.array([boundary-.01, boundary, boundary+.01]))
            np.testing.assert_allclose(np.diff(nearby)/.01, .015, atol=1e-5)

    def test_depth_uses_camera_baseline_and_full_markers(self):
        raw = observation()['tactile']['left_tactile']
        raw['depth'] = np.tile(np.array([31, 30, 27.5, 25, 24], np.float32), (5, 1))
        raw['press_depth'] = np.full((5, 5), 100, np.float32)
        raw['marker_full'] = raw['marker'].copy()
        raw['marker_full'][1] = raw['marker_full'][0] + [20, -20]
        # Current coordinates may be outside the source image; no filtering.
        result = tactile_from_arrays(raw, 'depth_deform_nonlinear', 5).numpy()
        np.testing.assert_allclose(result[0, 0], [0, 0, .5, 1, 1])
        np.testing.assert_allclose(result[1], .8, atol=1e-6)
        np.testing.assert_allclose(result[2], .2, atol=1e-6)
        self.assertEqual(result.dtype, np.float32)

    def test_duplicate_training_padding_is_merged_without_losing_motion(self):
        raw = observation()['tactile']['left_tactile']
        markers = raw['marker']
        padded = np.concatenate((markers, markers[:, -1:, :]), axis=1)
        padded[:, -1, 0] += 3e-5
        expected = convert_nonlinear(raw['depth'], markers, 12)
        actual = convert_nonlinear(raw['depth'], padded, 12)
        np.testing.assert_allclose(actual, expected, atol=2e-6)

    def test_large_test_displacements_remain_distinguishable(self):
        values = encode_displacement([20, 80, 160, 240, 1000])
        self.assertTrue((np.diff(values) > 0).all())
        self.assertTrue((values < 1).all())

    def test_tps_matches_independent_fit_for_non_affine_motion_and_caches(self):
        raw = observation()['tactile']['left_tactile']
        fixed = raw['marker'][0]
        displacement = np.column_stack((5*np.sin(fixed[:, 0]/50), 7*np.cos(fixed[:, 1]/40)))
        moved = (fixed + displacement).astype(np.float32)
        size = 12
        actual = convert_nonlinear(raw['depth'], np.stack((fixed, moved)), size)
        x, y = np.meshgrid((np.arange(size)+.5)*320/size, (np.arange(size)+.5)*240/size)
        query = np.column_stack((x.ravel(), y.ravel()))
        reference = RBFInterpolator(fixed.astype(np.float64), (moved-fixed).astype(np.float64),
                                    kernel='thin_plate_spline', degree=1)(query)
        expected = encode_displacement(reference).reshape(size, size, 2).transpose(2, 0, 1)
        np.testing.assert_allclose(actual[1:], expected, atol=2e-6)
        weights = _tps_interpolation_map(fixed.tobytes(), len(fixed), 240, 320, size)
        self.assertIs(weights, _tps_interpolation_map(fixed.tobytes(), len(fixed), 240, 320, size))
        np.testing.assert_allclose(weights.sum(axis=1), 1., atol=2e-6)

    def test_all_readers_and_offline_observations_match(self):
        for mode in ('depth_deform_nonlinear', 'depth_deform_nonlinear_radial'):
            with self.subTest(mode=mode):
                self.check_all_readers(mode)

    def check_all_readers(self, mode):
        obs = observation()
        for raw in obs['tactile'].values():
            raw['marker_full'] = raw['marker'].copy()
            raw['marker_full'][1] = raw['marker_full'][0] + [50, -100]
            raw['marker_full'][1, 0] = [-35, 400]
            raw['marker_id'] = np.arange(63, dtype=np.int64)
            raw['marker_in_view'] = np.ones(63, bool)
            raw['marker_in_view'][0] = False
            raw['press_depth'] = np.full_like(raw['depth'], 2.5)
        raw = obs['tactile']['left_tactile']
        expected = tactile_from_arrays(raw, mode)
        policy, run = make_policy(mode)
        run['options']['tactile_mode'] = 'encode'
        torch.testing.assert_close(policy.encode_obs(obs)['observation.images.tac_left'][0], expected)
        for compression in (None, 'gzip'):
            with self.subTest(compression=compression), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'episode.hdf5'
                with h5py.File(path, 'w') as f:
                    f.create_dataset('embodiment/joint', data=np.zeros((1, 9), np.float32))
                    for camera in ('head', 'wrist'):
                        encoded = np.frombuffer(encode_rgb(obs['observation'][camera]['rgb']), dtype=np.uint8)
                        ds = f.create_dataset(f'observation/{camera}/rgb', data=encoded[None])
                        ds.attrs['image_encoding'] = STANDARD_ENCODING
                    for side in ('left', 'right'):
                        for key in ('depth', 'press_depth', 'marker', 'marker_full', 'marker_id', 'marker_in_view'):
                            f.create_dataset(f'tactile/{side}_tactile/{key}', data=raw[key][None], compression=compression)
                with h5py.File(path, 'r') as f:
                    torch.testing.assert_close(read_tactile(f, 'left', 0, mode), expected)
                    torch.testing.assert_close(read_native_tactile(f, 'left', 0, mode), expected)
                dataset = TactileFrames([path], input_mode=mode)
                try:
                    torch.testing.assert_close(dataset[0], expected)
                    self.assertEqual(dataset.mapped_arrays[(str(path), 'left')] is None, compression is not None)
                finally:
                    for handle in dataset.handles.values():
                        handle.close()
                restored = hdf5_observation(path, 0, run)
                torch.testing.assert_close(policy.encode_obs(restored)['observation.images.tac_left'][0], expected)
                np.testing.assert_array_equal(restored['tactile']['left_tactile']['marker_id'], raw['marker_id'])
                np.testing.assert_array_equal(restored['tactile']['left_tactile']['marker_full'], raw['marker_full'])

    def test_metadata_and_invalid_inputs(self):
        spec = depth_deform_spec('depth_deform_nonlinear')
        self.assertEqual(spec['reference_depth_mm'], 30.)
        self.assertEqual(spec['depth_range_mm'], 5.)
        spec['deform_linear_output_range'][0] = 99
        self.assertEqual(depth_deform_spec('depth_deform_nonlinear')['deform_linear_output_range'][0], .2)
        with self.assertRaises(ValueError):
            encode_displacement([np.nan])
        with self.assertRaises(ValueError):
            convert_nonlinear(np.ones((3, 3)), np.zeros((2, 2, 2)))


if __name__ == '__main__':
    unittest.main()
