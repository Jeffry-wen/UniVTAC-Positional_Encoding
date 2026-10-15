"""Convert contact HDF5 observations to the tactile encoder input modes."""
import cv2
import numpy as np
import torch
if __package__:
    from .image_io import read_rgb
    from .linear_depth_deform import convert_linear, DEPTH_DEFORM_SPECS, depth_deform_spec
    from .nonlinear_depth_deform import convert_nonlinear
    from .radial_nonlinear_depth_deform import convert_nonlinear_radial
else:
    from image_io import read_rgb
    from linear_depth_deform import convert_linear, DEPTH_DEFORM_SPECS, depth_deform_spec
    from nonlinear_depth_deform import convert_nonlinear
    from radial_nonlinear_depth_deform import convert_nonlinear_radial


INPUT_CHANNELS = {'marker_only': 3, 'rgb_only': 3, 'marker_rgb': 3,
                  **{mode: 3 for mode in DEPTH_DEFORM_SPECS}}


def tactile_from_arrays(data, mode, image_size=224):
    """Encode raw RGB/depth/marker arrays identically for training and deployment."""
    if mode not in INPUT_CHANNELS:
        raise ValueError(f'Unknown tactile input mode: {mode}')
    if mode in ('rgb_only', 'marker_rgb'):
        key = 'rgb' if mode == 'rgb_only' else 'rgb_marker'
        image = np.asarray(data[key])
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
    if mode == 'marker_only':
        markers = np.asarray(data['marker'])[1, :63]
        canvas = np.zeros((image_size, image_size), dtype=np.uint8)
        for x, y in markers:
            cx = int(round(float(x) * image_size / 320))
            cy = int(round(float(y) * image_size / 240))
            if 0 <= cx < image_size and 0 <= cy < image_size:
                cv2.circle(canvas, (cx, cy), 2, 255, -1)
        # Replicate the marker raster into RGB; each channel contains the same dots.
        return torch.from_numpy(canvas.copy()).unsqueeze(0).repeat(3, 1, 1).float() / 255
    if mode == 'depth_deform_nonlinear':
        return torch.from_numpy(convert_nonlinear(
            data['depth'], data.get('marker_full', data.get('marker')), image_size))
    if mode == 'depth_deform_nonlinear_radial':
        return torch.from_numpy(convert_nonlinear_radial(
            data['depth'], data.get('marker_full', data.get('marker')), image_size))
    spec = depth_deform_spec(mode)
    channels = convert_linear(
        data['depth'], data.get('marker_full', data.get('marker')), image_size,
        reference_depth_mm=spec['reference_depth_mm'],
        depth_range_mm=spec['depth_range_mm'], deform_range_px=spec['deform_range_px'])
    return torch.from_numpy(channels)


def read_tactile(h5, side, step, mode, image_size=224):
    prefix = f'tactile/{side}_tactile'
    if mode in ('rgb_only', 'marker_rgb'):
        key = 'rgb' if mode == 'rgb_only' else 'rgb_marker'
        data = {key: read_rgb(h5[f'{prefix}/{key}'], step)}
    elif mode == 'marker_only':
        data = {'marker': h5[f'{prefix}/marker'][step]}
    elif mode in DEPTH_DEFORM_SPECS:
        data = {'depth': h5[f'{prefix}/depth'][step], 'marker': h5[f'{prefix}/marker'][step]}
        for key in ('press_depth', 'marker_full'):
            if f'{prefix}/{key}' in h5:
                data[key] = h5[f'{prefix}/{key}'][step]
    else:
        raise ValueError(f'Unknown tactile input mode: {mode}')
    return tactile_from_arrays(data, mode, image_size)
