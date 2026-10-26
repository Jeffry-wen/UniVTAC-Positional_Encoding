"""DINO触觉RoPE: rest-relative rotation + inter-patch rotary attention.

Input BCHW channels: camera depth (mm), reference-frame u/v (pixels), x/y
reference coordinates (pixels). Calibration and normalization live in config.
No learned absolute position table, reference tokens, or auxiliary heads.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class AxisRoPE(nn.Module):
    """Distribute channel pairs over axes; all phases computed in float32."""
    def __init__(self, dim, axes, min_frequency=0.01, max_frequency=1.0):
        super().__init__()
        if dim % 2 or dim // 2 < axes:
            raise ValueError('RoPE requires an even dimension and at least one pair per axis')
        if not 0 < min_frequency <= max_frequency:
            raise ValueError('Invalid RoPE frequency range')
        axis = torch.arange(dim // 2) % axes
        frequencies = torch.empty(dim // 2)
        for a in range(axes):
            count = int((axis == a).sum())
            frequencies[axis == a] = torch.logspace(
                math.log10(min_frequency), math.log10(max_frequency), count)
        self.register_buffer('axis', axis)
        self.register_buffer('frequencies', frequencies)

    def forward(self, features, coordinates):
        # features B,N,D or B,H,N,D; coordinates B,N,A.
        phase = coordinates.float()[..., self.axis] * self.frequencies.float()
        if features.ndim == 4:
            phase = phase[:, None]
        pairs = features.float().reshape(*features.shape[:-1], -1, 2)
        a, b = pairs.unbind(-1)
        cos, sin = phase.cos(), phase.sin()
        return torch.stack((a*cos - b*sin, a*sin + b*cos), -1).flatten(-2).to(features.dtype)


class DualRoPEBlock(nn.Module):
    def __init__(self, dim, heads, min_frequency, max_frequency):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.rest_project = nn.Linear(dim, dim)
        self.rest_rope = AxisRoPE(dim, 3, min_frequency, max_frequency)
        self.qkv = nn.Linear(dim, dim * 3)
        self.relation_rope = AxisRoPE(dim // heads, 5, min_frequency, max_frequency)
        self.fuse = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, coordinates):
        normalized = self.norm1(x)
        # R(d0)^T R(dt) = R(dt-d0); deformation has already been zero-referenced.
        static = self.rest_rope(self.rest_project(normalized), coordinates[..., 2:])
        b, n, dim = x.shape
        q, k, v = self.qkv(normalized).reshape(b, n, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q = self.relation_rope(q, coordinates)
        k = self.relation_rope(k, coordinates)
        context = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, dim)
        x = x + self.fuse(torch.cat((static, context), -1))
        return x + self.mlp(self.norm2(x))


class TactileRoPEViT(nn.Module):
    def __init__(self, size='S', reference_depth_mm=30.0, mm_per_pixel=None,
                 spatial_scales=(160., 120.), deformation_scales=(20., 20., 5.),
                 min_frequency=0.01, max_frequency=1.0):
        super().__init__()
        if size not in ('S', 'B'):
            raise ValueError('Expected S or B')
        self.embed_dim, heads = (384, 6) if size == 'S' else (768, 12)
        if not math.isfinite(reference_depth_mm):
            raise ValueError('reference_depth_mm must be finite')
        if mm_per_pixel is not None and (len(mm_per_pixel) != 2 or any(not math.isfinite(v) or v <= 0 for v in mm_per_pixel)):
            raise ValueError('mm_per_pixel must be two positive finite values')
        scales = [*spatial_scales, *deformation_scales]
        if len(spatial_scales) != 2 or len(deformation_scales) != 3 or any(not math.isfinite(v) or v <= 0 for v in scales):
            raise ValueError('Expected positive finite x/y and u/v/h scales')
        self.config = dict(reference_depth_mm=reference_depth_mm, mm_per_pixel=mm_per_pixel,
                           spatial_scales=list(spatial_scales), deformation_scales=list(deformation_scales),
                           min_frequency=min_frequency, max_frequency=max_frequency)
        self.register_buffer('reference_depth', torch.tensor(float(reference_depth_mm)))
        self.register_buffer('pixel_scale', torch.tensor(mm_per_pixel or (1., 1.), dtype=torch.float32))
        self.register_buffer('scales', torch.tensor(scales, dtype=torch.float32))
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=16, stride=16)
        self.blocks = nn.ModuleList([DualRoPEBlock(self.embed_dim, heads, min_frequency, max_frequency) for _ in range(12)])
        self.norm = nn.LayerNorm(self.embed_dim)

    def fields(self, images):
        if images.ndim != 4 or images.shape[1] != 5 or images.shape[-1] % 16 or images.shape[-2] % 16:
            raise ValueError('Expected [B,5,H,W] raw depth/u/v/x/y, H/W multiples of 16')
        # No clamping, brightness jitter, or per-frame max normalization.
        depth, u, v, x, y = images.float().unbind(1)
        sx, sy = self.pixel_scale
        h = self.reference_depth - depth
        coordinates = torch.stack((x*sx, y*sy, u*sx, v*sy, h), 1) / self.scales[None, :, None, None]
        return coordinates[:, 2:], coordinates

    def tokens(self, images, patch_mask=None, mask_token=None):
        deformation, fields = self.fields(images)
        coordinates = F.avg_pool2d(fields, 16, 16).flatten(2).transpose(1, 2)
        # Mask BEFORE embedding; hidden patch values must not reach either branch.
        if patch_mask is not None:
            b, _, h, w = deformation.shape
            if patch_mask.shape != (b, (h//16)*(w//16)) or mask_token is None:
                raise ValueError('Invalid patch mask or missing mask token')
            pixel_mask = patch_mask.reshape(b, 1, h//16, w//16).repeat_interleave(16, 2).repeat_interleave(16, 3)
            deformation = deformation.masked_fill(pixel_mask, 0)
            coordinates = torch.cat((coordinates[..., :2], coordinates[..., 2:].masked_fill(patch_mask[..., None], 0)), -1)
        x = self.patch_embed(deformation).flatten(2).transpose(1, 2)
        if patch_mask is not None:
            x = torch.where(patch_mask[..., None], mask_token.to(x.dtype), x)
        for block in self.blocks:
            x = block(x, coordinates)
        x = self.norm(x)
        # A pooled global token preserves DINO's CLS + patch head interface.
        return torch.cat((x.mean(1, keepdim=True), x), 1)

    def forward(self, images):
        return self.tokens(images)[:, 0]

    def patch_tokens(self, images):
        return self.tokens(images)[:, 1:]


def load_encoder(checkpoint, device='cpu'):
    """Load this training variant explicitly; it is not a plain TactileViT."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('method') != 'DINO_TactileRoPE' or saved.get('architecture') != 'tactile_rope_v1':
        raise ValueError('Expected a DINO触觉RoPE checkpoint')
    if saved.get('input_mode') != 'depth_deform_raw' or 'model_config' not in saved:
        raise ValueError('Missing raw-input/calibration metadata')
    model = TactileRoPEViT(saved['size'], **saved['model_config']['rope'])
    model.load_state_dict(saved['encoder'], strict=True)
    return model.to(device).eval()
