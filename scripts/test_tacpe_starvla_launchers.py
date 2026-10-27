"""Eight-rank launchers preserve the original tactile ablation budget."""
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def launch(self, version, **overrides):
        return subprocess.run(['bash', str(ROOT / 'scripts' / f'train_dino_tacpe_v{version}_starvla_8gpu.sh'), '--dry-run'],
                              env={**os.environ, 'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7', **overrides},
                              text=True, capture_output=True)

    def test_versions_and_baseline_budget(self):
        for version, method in [(1,'DINO_TactileRoPE'),(2,'DINO_TactileRoPE_V2')]:
            result = self.launch(version, BATCH_SIZE='128', TACTILE_TYPE='full', STEPS='1')
            self.assertEqual(result.returncode, 0, result.stderr)
            settings = dict(line.split('=',1) for line in result.stdout.splitlines())
            expected = dict(POLICY='starvla_groot', NUM_PROCESSES='8', BATCH_SIZE='16', STEPS='30000',
                            CHUNK_SIZE='32', N_ACTION_STEPS='30', SAVE_FREQ='5000',
                            TACTILE_TYPE='cls', TACTILE_INSERT_LOCATION='encoder',
                            TACTILE_INPUT_MODE='depth_deform_raw', ENCODER_METHOD=method,
                            ACTION_MODE='relative_joint', per_gpu_batch='2')
            for key, value in expected.items():
                self.assertEqual(settings[key], value, key)

    def test_invalid_gpu_lists_fail_before_training(self):
        for ids in ['0,1', '0,1,2,3,4,5,6,6', '0,1,2,3,4,5,6,7,', '0,1,2,3,4,5,6,x']:
            self.assertNotEqual(self.launch(1, CUDA_VISIBLE_DEVICES=ids).returncode, 0)


if __name__ == '__main__':
    unittest.main()
