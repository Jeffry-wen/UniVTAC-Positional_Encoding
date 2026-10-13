"""Check fixed linear encoding endpoints, rest zero, and legacy isolation."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'encoder'))
from linear_depth_deform import encode_linear_fields,convert_linear
from depth_marker_to_deform import convert


def markers(displacement=(0,0)):
    x,y=np.meshgrid(np.linspace(40,280,9),np.linspace(35,205,7))
    p=np.column_stack((x.ravel(),y.ravel())).astype(np.float32)
    return np.stack((p,p+np.asarray(displacement,dtype=np.float32)))


class LinearEncodingTest(unittest.TestCase):
    def test_rest_and_signed_endpoints(self):
        result=encode_linear_fields(np.zeros((2,2)),np.zeros((2,2,2)))
        expected=np.array([0,128,128],dtype=np.uint8)
        np.testing.assert_array_equal(np.rint(result[:,0,0]*255).astype(np.uint8),expected)
        field=np.array([[[100,-100]],[[0,50]]],dtype=np.float32)
        result=encode_linear_fields(np.array([[4,1.5]],dtype=np.float32),field)
        np.testing.assert_allclose(result[:,0,0]*255,[255,255,128],atol=1e-5)
        np.testing.assert_allclose(result[:,0,1]*255,[127.5,1,191.5],atol=1e-5)

    def test_known_rest_baseline_and_contact_not_first_frame_zero(self):
        rest=np.full((240,320),28,dtype=np.float32)
        output=convert_linear(rest,markers(),reference_depth_mm=28)
        np.testing.assert_allclose(output[0],0)
        np.testing.assert_allclose(output[1:],128/255)
        contact=rest-.5
        output=convert_linear(contact,markers(),reference_depth_mm=28)
        np.testing.assert_allclose(output[0],1/6,atol=1e-6)
        direct=convert_linear(contact,markers(),press_depth=np.full_like(rest,.5))
        np.testing.assert_array_equal(output,direct)

    def test_requires_explicit_depth_reference(self):
        with self.assertRaises(ValueError):
            convert_linear(np.full((240,320),28),markers())
        with self.assertRaises(ValueError):
            encode_linear_fields(np.zeros((1,1)),np.zeros((2,1,1)),depth_range_mm=0)

    def test_legacy_conversion_is_unchanged(self):
        old=convert(np.full((240,320),28,dtype=np.float32),markers())
        np.testing.assert_allclose(old[0],.6,atol=1e-6)
        np.testing.assert_allclose(old[1:],.5)


if __name__=='__main__':unittest.main()
