"""Unclipped native sensor fields, packed together to keep crops aligned."""
import cv2
import numpy as np
import torch
try:
    from ..nonlinear_depth_deform import _tps_interpolation_map
except ImportError:
    from nonlinear_depth_deform import _tps_interpolation_map


def raw_fields(depth, markers, image_size=224):
    depth = np.asarray(depth, dtype=np.float32)
    markers = np.asarray(markers, dtype=np.float32)
    if depth.ndim != 2 or markers.ndim != 3 or markers.shape[0] != 2 or markers.shape[-1] != 2:
        raise ValueError('Expected depth [H,W], reference/current markers [2,N,2]')
    if not np.isfinite(depth).all() or not np.isfinite(markers).all():
        raise ValueError('Nonfinite sensor data')
    h, w = depth.shape
    reference, current = markers
    weights = _tps_interpolation_map(reference.tobytes(), len(reference), h, w, image_size)
    uv = (weights @ (current-reference)).reshape(image_size, image_size, 2)
    x, y = np.meshgrid((np.arange(image_size)+.5)*w/image_size,
                       (np.arange(image_size)+.5)*h/image_size)
    return torch.from_numpy(np.stack((cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_AREA),
                                     uv[..., 0], uv[..., 1], x, y)).astype(np.float32))
