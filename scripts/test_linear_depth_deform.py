"""Check fixed linear encoding endpoints, rest zero, and legacy isolation."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'encoder'))
from linear_depth_deform import encode_linear_fields,convert_linear,require_depth_deform_encoding
from depth_marker_to_deform import convert
from modality import INPUT_CHANNELS, tactile_from_arrays, depth_deform_spec


def markers(displacement=(0,0)):
    x,y=np.meshgrid(np.linspace(40,280,9),np.linspace(35,205,7))
    p=np.column_stack((x.ravel(),y.ravel())).astype(np.float32)
    return np.stack((p,p+np.asarray(displacement,dtype=np.float32)))


class LinearEncodingTest(unittest.TestCase):
    def test_modes_use_selected_range_and_clip(self):
        for limit in (20, 80, 160, 240):
            mode = f'depth_deform_{limit}'
            with self.subTest(mode=mode):
                raw = dict(depth=np.full((240, 320), 27, np.float32),
                           marker=markers((limit / 2, -limit * 2)))
                result = tactile_from_arrays(raw, mode).numpy()
                np.testing.assert_allclose(result[:, 100, 100] * 255,
                                           [153, 191.5, 1], atol=2e-5)
                self.assertEqual(depth_deform_spec(mode)['deform_range_px'], limit)
                self.assertEqual(depth_deform_spec(mode)['depth_range_mm'], 5)
                self.assertEqual(depth_deform_spec(mode)['reference_depth_mm'], 30)
        self.assertNotIn('depth_deform', INPUT_CHANNELS)
        with self.assertRaises(ValueError):
            tactile_from_arrays(raw, 'depth_deform')

    def test_rest_and_signed_endpoints(self):
        result=encode_linear_fields(np.zeros((2,2)),np.zeros((2,2,2)))
        expected=np.array([0,128,128],dtype=np.uint8)
        np.testing.assert_array_equal(np.rint(result[:,0,0]*255).astype(np.uint8),expected)
        field=np.array([[[100,-100]],[[0,50]]],dtype=np.float32)
        result=encode_linear_fields(np.array([[4,1.5]],dtype=np.float32),field)
        np.testing.assert_allclose(result[:,0,0]*255,[204,255,128],atol=2e-5)
        np.testing.assert_allclose(result[:,0,1]*255,[76.5,1,191.5],atol=1e-5)

    def test_known_rest_baseline_and_contact_not_first_frame_zero(self):
        rest=np.full((240,320),30,dtype=np.float32)
        output=convert_linear(rest,markers())
        np.testing.assert_allclose(output[0],0)
        np.testing.assert_allclose(output[1:],128/255)
        contact=rest-.5
        output=convert_linear(contact,markers())
        np.testing.assert_allclose(output[0],.1,atol=1e-6)
        direct=convert_linear(contact,markers(),press_depth=np.full_like(rest,100))
        np.testing.assert_array_equal(output,direct)

    def test_rejects_invalid_depth_reference(self):
        with self.assertRaises(ValueError):
            convert_linear(np.full((240,320),28),markers(),reference_depth_mm=np.nan)
        with self.assertRaises(ValueError):
            encode_linear_fields(np.zeros((1,1)),np.zeros((2,1,1)),depth_range_mm=0)

    def test_legacy_conversion_is_unchanged(self):
        old=convert(np.full((240,320),28,dtype=np.float32),markers())
        np.testing.assert_allclose(old[0],.6,atol=1e-6)
        np.testing.assert_allclose(old[1:],.5)

    def test_old_depth_encoding_checkpoints_are_rejected(self):
        for mode in ('depth_deform_20', 'depth_deform_80', 'depth_deform_160',
                     'depth_deform_240', 'depth_deform_nonlinear', 'depth_deform_nonlinear_radial'):
            require_depth_deform_encoding({'depth_deform_encoding': depth_deform_spec(mode)}, mode)
            with self.assertRaisesRegex(ValueError, 'retrain'):
                require_depth_deform_encoding({}, mode)
        outdated = depth_deform_spec('depth_deform_80')
        outdated.update(reference_depth_mm=28., depth_range_mm=3.)
        with self.assertRaisesRegex(ValueError, 'retrain'):
            require_depth_deform_encoding({'depth_deform_encoding': outdated}, 'depth_deform_80')


if __name__=='__main__':unittest.main()
