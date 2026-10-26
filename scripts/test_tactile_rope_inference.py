"""Public package import, checkpoint dispatch and deterministic feature extraction."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from encoder.DINO_TactileRoPE.encoder import TactileRoPEViT
from encoder.DINO_TactileRoPE_V2.encoder import TactileRoPEViTV2
from encoder.tactile_rope_inference import load_tactile_rope, infer_fields


class InferenceTest(unittest.TestCase):
    def test_both_versions_match_direct_forward(self):
        torch.set_num_threads(2)
        fields = torch.randn(2, 5, 32, 32)
        fields[:, 0] += 30
        for version, cls in ((1, TactileRoPEViT), (2, TactileRoPEViTV2)):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                model = cls().eval()
                method = 'DINO_TactileRoPE' + ('_V2' if version == 2 else '')
                checkpoint = Path(tmp) / 'encoder.pth'
                torch.save(dict(method=method, architecture=f'tactile_rope_v{version}',
                                size='S', input_mode='depth_deform_raw',
                                model_config={'rope':model.config}, encoder=model.state_dict()), checkpoint)
                loaded, metadata = load_tactile_rope(checkpoint)
                self.assertEqual(metadata['method'], method)
                self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
                actual = infer_fields(loaded, fields)
                with torch.no_grad():
                    expected = model.tokens(fields)
                torch.testing.assert_close(actual['global_features'], expected[:, 0], atol=0, rtol=0)
                torch.testing.assert_close(actual['patch_features'], expected[:, 1:], atol=0, rtol=0)
                single = infer_fields(loaded, fields[0])
                torch.testing.assert_close(single['global_features'], actual['global_features'][:1])
                with self.assertRaises(ValueError):
                    infer_fields(loaded, torch.full_like(fields, float('nan')))


if __name__ == '__main__':
    unittest.main()
