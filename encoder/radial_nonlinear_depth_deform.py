"""Direction-preserving nonlinear encoding of reference-coordinate TPS vectors."""
import numpy as np

if __package__:
    from .nonlinear_depth_deform import NONLINEAR_DEPTH_DEFORM_SPEC, _convert_tps
else:
    from nonlinear_depth_deform import NONLINEAR_DEPTH_DEFORM_SPEC, _convert_tps


RADIAL_NONLINEAR_DEPTH_DEFORM_SPEC = {
    **{key: value for key, value in NONLINEAR_DEPTH_DEFORM_SPEC.items()
       if key.startswith('tps_') or key in ('reference_depth_mm', 'depth_range_mm',
           'depth_source', 'interpolation', 'interpolation_coordinates')},
    'version': 'nonlinear_radial_30mm_5mm_tps_v1',
    'deform_mapping': 'radial',
    'deform_radius': 'l2_sensor_pixels',
    'deform_core_radius_px': 20.0,
    'deform_core_slope': 0.015,
    'deform_amplitude_at_core_radius': 0.3,
    'deform_amplitude_asymptote': 0.5,
    'deform_tail': 'rational',
    'deform_tail_scale_px': 40.0 / 3.0,
    'zero_displacement': 0.5,
    'deform_deadzone_px': 0.0,
    'deform_hard_clip': False,
}


def encode_radial_displacement(displacement_px):
    """Encode [...,2] dx/dy vectors; r<=20 matches the original linear core.

    GB = .5 + A(r) * vector/r, with A(r)=.015r in the core and
    A(r)=.5-.2/(1+(r-20)/(40/3)) outside. Both components share the
    same positive gain. Zero vectors map to (.5,.5). Float32 output.
    """
    field = np.asarray(displacement_px, dtype=np.float64)
    if field.ndim < 1 or field.shape[-1] != 2:
        raise ValueError('Expected displacement vectors [...,2] with dx,dy last')
    if not np.isfinite(field).all():
        raise ValueError('Displacements must be finite')
    radius = np.hypot(field[..., 0], field[..., 1])
    amplitude = np.where(radius <= 20., .015 * radius,
                         .5 - .2 / (1. + np.maximum(radius - 20., 0.) / (40. / 3.)))
    gain = np.divide(amplitude, radius, out=np.full_like(radius, .015), where=radius > 0)
    return (.5 + field * gain[..., None]).astype(np.float32)


def convert_nonlinear_radial(depth, markers, image_size=224):
    """Same 30/5 camera-depth and full-marker TPS semantics as nonlinear."""
    return _convert_tps(depth, markers, image_size, encode_radial_displacement)


def decode_radial_displacement(encoded_gb, max_radius_px=400.):
    """Invert [...,2] vectors with an explicit radial visualization bound.

    VAE outputs can leave the encoding disk. Project their radius, retaining
    direction, to A(max_radius_px) before inversion. Return decoded vectors
    and a per-vector mask identifying every projection. This bound is only
    for decoding/visualization; it does not clip the training encoding.
    """
    gb = np.asarray(encoded_gb, dtype=np.float64)
    if gb.ndim < 1 or gb.shape[-1] != 2 or not np.isfinite(gb).all():
        raise ValueError('Expected finite encoded vectors [...,2]')
    if not np.isfinite(max_radius_px) or max_radius_px <= 0:
        raise ValueError('Decoding radius must be finite and positive')
    centered = gb - .5
    amplitude = np.hypot(centered[..., 0], centered[..., 1])
    maximum = (.015 * max_radius_px if max_radius_px <= 20 else
               .5 - .2 / (1. + (max_radius_px - 20.) / (40. / 3.)))
    bounded = np.minimum(amplitude, maximum)
    radius = np.where(bounded <= .3, bounded / .015,
                      20. + (40. / 3.) * (.2 / np.maximum(.5 - bounded, 1e-12) - 1.))
    gain = np.divide(radius, amplitude, out=np.zeros_like(radius), where=amplitude > 0)
    return (centered * gain[..., None]).astype(np.float32), amplitude > maximum
