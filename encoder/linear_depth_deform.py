"""Shared fixed-range linear depth/deform encoding, 30 mm depth baseline.

Byte-space convention: R=255*clip(indentation/D,0,1);
G/B=128+127*clip(displacement/U,-1,1). Returns normalized float32 CHW.
The symmetric G/B range is 1..255 so zero is exactly 128 with equal signed gain.
Reference-position interpolation is identical to depth_marker_to_deform.
The historical converter remains available in depth_marker_to_deform.
"""
import cv2
import numpy as np
from copy import deepcopy
if __package__:
    from .depth_marker_to_deform import _interpolation_map
    from .nonlinear_depth_deform import NONLINEAR_DEPTH_DEFORM_SPEC
    from .radial_nonlinear_depth_deform import RADIAL_NONLINEAR_DEPTH_DEFORM_SPEC
else:
    from depth_marker_to_deform import _interpolation_map
    from nonlinear_depth_deform import NONLINEAR_DEPTH_DEFORM_SPEC
    from radial_nonlinear_depth_deform import RADIAL_NONLINEAR_DEPTH_DEFORM_SPEC

DEPTH_DEFORM_SPEC = {
    'version': 'linear_camera_30mm_5mm_100px_v2',
    'depth_range_mm': 5.0, 'deform_range_px': 100.0,
    'depth_source': 'camera_depth', 'interpolation': 'linear_idw',
    'reference_depth_mm': 30.0, 'zero_displacement_byte': 128,
    'displacement_gain_byte': 127, 'interpolation_coordinates': 'reference',
}

DEPTH_DEFORM_SPECS = {
    f'depth_deform_{limit}': {
        **DEPTH_DEFORM_SPEC,
        'version': f'linear_camera_30mm_5mm_{limit}px_v2',
        'deform_range_px': float(limit),
    }
    for limit in (20, 80, 160, 240)
}
DEPTH_DEFORM_SPECS['depth_deform_nonlinear'] = NONLINEAR_DEPTH_DEFORM_SPEC
DEPTH_DEFORM_SPECS['depth_deform_nonlinear_radial'] = RADIAL_NONLINEAR_DEPTH_DEFORM_SPEC


def depth_deform_spec(mode):
    return deepcopy(DEPTH_DEFORM_SPECS[mode])


def require_depth_deform_encoding(metadata, mode):
    """Reject weights trained with different depth/deform preprocessing."""
    if mode in DEPTH_DEFORM_SPECS and metadata.get('depth_deform_encoding') != depth_deform_spec(mode):
        raise ValueError(f'{mode} checkpoint depth_deform_encoding does not match current preprocessing; '
                         'retrain with the current encoding')


def encode_linear_fields(indentation_mm, displacement_px, depth_range_mm=5., deform_range_px=100.):
    """Encode [H,W] indentation and [2,H,W] reference-coordinate displacement."""
    indentation=np.asarray(indentation_mm,dtype=np.float32)
    field=np.asarray(displacement_px,dtype=np.float32)
    if indentation.ndim!=2 or field.shape!=(2,*indentation.shape):
        raise ValueError('Expected indentation [H,W] and displacement [2,H,W]')
    if not np.isfinite(indentation).all() or not np.isfinite(field).all():
        raise ValueError('Input arrays must be finite')
    if not np.isfinite(depth_range_mm) or not np.isfinite(deform_range_px) or depth_range_mm<=0 or deform_range_px<=0:
        raise ValueError('Ranges must be finite and positive')
    r=np.clip(indentation/depth_range_mm,0,1)
    gb=(128.+127.*np.clip(field/deform_range_px,-1,1))/255.
    return np.concatenate((r[None],gb),axis=0).astype(np.float32)


def convert_linear(depth,markers,image_size=224,*,press_depth=None,reference_depth_mm=30.,
                   depth_range_mm=5.,deform_range_px=100.):
    """Convert camera depth with the same depth semantics as nonlinear mode.

press_depth is accepted for caller compatibility but is not used: recorded
indentation can have a different baseline. Resize camera depth before mapping,
identically to convert_nonlinear. Markers may lie out of view. Explicit range
and reference overrides are available for historical comparison scripts.
"""
    depth=np.asarray(depth,dtype=np.float32)
    marker=np.asarray(markers,dtype=np.float32)
    if depth.ndim!=2 or marker.ndim!=3 or marker.shape[0]!=2 or marker.shape[-1]!=2:
        raise ValueError('Expected depth [H,W] and markers [2,N,2]')
    if reference_depth_mm is None or not np.isfinite(reference_depth_mm):
        raise ValueError('reference_depth_mm must be finite')
    if not np.isfinite(depth).all() or not np.isfinite(marker).all():
        raise ValueError('Depth and markers must be finite')
    resized=cv2.resize(depth,(image_size,image_size),interpolation=cv2.INTER_AREA)
    indentation=np.maximum(float(reference_depth_mm)-resized,0)
    reference,current=marker
    vertices,weights=_interpolation_map(reference.tobytes(),len(reference),*depth.shape,image_size)
    field=np.einsum('nvc,nv->nc',(current-reference)[vertices],weights)
    field=field.reshape(image_size,image_size,2).transpose(2,0,1)
    return encode_linear_fields(indentation,field,depth_range_mm,deform_range_px)
