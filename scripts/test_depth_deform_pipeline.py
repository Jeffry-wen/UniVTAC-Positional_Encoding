"""Check train mmap, HDF5 and deployment parity for 30/5 depth with 20/80/160/240px."""
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'encoder'))
from modality import read_tactile, tactile_from_arrays
from train_common import TactileFrames
from policy.n0vtla.observation_adapter import read_native_tactile
from scripts.test_policy_input_recording import make_policy, observation


class DepthDeformPipelineTest(unittest.TestCase):
    def test_all_readers_match_full_markers_and_camera_depth(self):
        for limit in (20, 80, 160, 240):
            with self.subTest(limit=limit):
                self.check_readers(limit)

    def check_readers(self, limit):
        mode = f'depth_deform_{limit}'
        obs = observation()
        raw = obs['tactile']['left_tactile']
        # Distinguish ignored saved indentation from camera depth, and full
        # out-of-view coordinates from the deliberately conflicting old markers.
        raw['press_depth'] = np.full_like(raw['depth'], 2.5)
        raw['marker_full'] = raw['marker'].copy()
        raw['marker_full'][1] = raw['marker_full'][0] + [50, -100]
        raw['marker'][1] = raw['marker'][0]
        expected = tactile_from_arrays(raw, mode)
        np.testing.assert_allclose(expected[:, 100, 100].numpy()*255,
                                   [153,128+127*min(50/limit,1),128+127*max(-100/limit,-1)], atol=2e-5)
        policy,_ = make_policy(mode)
        torch.testing.assert_close(policy.encode_obs(obs)['observation.images.tac_left'][0], expected)
        for compression in (None, 'gzip'):
            with self.subTest(compression=compression), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'episode.hdf5'
                with h5py.File(path,'w') as f:
                    for key in ('depth','press_depth','marker','marker_full'):
                        f.create_dataset(f'tactile/left_tactile/{key}',data=raw[key][None],compression=compression)
                with h5py.File(path,'r') as f:
                    torch.testing.assert_close(read_tactile(f,'left',0,mode),expected)
                    torch.testing.assert_close(read_native_tactile(f,'left',0,mode),expected)
                data = TactileFrames([path],input_mode=mode)
                torch.testing.assert_close(data[0],expected)
                self.assertEqual(data.mapped_arrays[(str(path),'left')] is None, compression is not None)
                for handle in data.handles.values():
                    handle.close()

    def test_legacy_training_file_uses_30mm_camera_baseline(self):
        raw = observation()['tactile']['left_tactile']
        output = tactile_from_arrays(raw,'depth_deform_80')
        np.testing.assert_allclose(output[:, 100, 100].numpy()*255,
            [153,128+127*8/80,128-127*5/80],atol=2e-5)

    def test_all_depth_channels_match_nonlinear_exactly(self):
        raw = observation()['tactile']['left_tactile']
        # Include values straddling the clipping thresholds in resized pixels.
        raw['depth'] = np.random.default_rng(42).uniform(23, 33, raw['depth'].shape).astype(np.float32)
        raw['press_depth'] = np.full_like(raw['depth'], 99)
        expected = tactile_from_arrays(raw, 'depth_deform_nonlinear', 37)[0]
        for limit in (20, 80, 160, 240):
            with self.subTest(limit=limit):
                torch.testing.assert_close(tactile_from_arrays(raw, f'depth_deform_{limit}', 37)[0],
                                           expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
