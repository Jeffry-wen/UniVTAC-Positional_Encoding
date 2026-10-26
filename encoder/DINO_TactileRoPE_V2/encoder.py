"""Dual-reference tactile encoder; no auxiliary prediction heads.

Raw channels and units are identical to V1. A scalar rest depth is an explicit
assumption, not a calibrated indentation measurement. No per-frame scaling.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
if __package__.startswith('encoder.'):
    from ..DINO_TactileRoPE.encoder import AxisRoPE, TactileRoPEViT
else:
    from DINO_TactileRoPE.encoder import AxisRoPE, TactileRoPEViT


class DualReferenceBlock(nn.Module):
    def __init__(self, dim, heads, min_frequency=.01, max_frequency=1.,
                 layer_scale_init=.001):
        super().__init__()
        if dim // heads != 64:
            raise ValueError('V2 uses 64 channels/head: content 24, position 16, deformation 24')
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.rest_project = nn.Linear(dim, dim)
        self.rest_rope = AxisRoPE(dim, 3, min_frequency, max_frequency)
        # No bias: zero deformation must give exactly zero static update.
        self.rest_output = nn.Linear(dim, dim, bias=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.position_rope = AxisRoPE(16, 2, min_frequency, max_frequency)
        self.deformation_rope = AxisRoPE(24, 3, min_frequency, max_frequency)
        self.context_output = nn.Linear(dim, dim)
        self.gamma_static = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.gamma_context = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.gamma_mlp = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def static_increment(self, normalized, deformation):
        projected = self.rest_project(normalized)
        return self.rest_output(self.rest_rope(projected, deformation) - projected)

    def rotate_relation(self, features, coordinates):
        content, position, deformation = features.split((24, 16, 24), dim=-1)
        return torch.cat((content, self.position_rope(position, coordinates[..., :2]),
                          self.deformation_rope(deformation, coordinates[..., 2:])), dim=-1)

    def forward(self, x, coordinates):
        z = self.norm1(x)
        static = self.static_increment(z, coordinates[..., 2:])
        b, n, dim = x.shape
        q, k, v = self.qkv(z).reshape(b, n, 3, self.heads, 64).permute(2, 0, 3, 1, 4).unbind(0)
        context = F.scaled_dot_product_attention(
            self.rotate_relation(q, coordinates), self.rotate_relation(k, coordinates), v
        ).transpose(1, 2).reshape(b, n, dim)
        x = x + self.gamma_static * static + self.gamma_context * self.context_output(context)
        return x + self.gamma_mlp * self.mlp(self.norm2(x))


class TactileRoPEViTV2(TactileRoPEViT):
    def __init__(self, size='S', layer_scale_init=.001, invalid_depth_ge_mm=None, **kwargs):
        if not math.isfinite(layer_scale_init) or layer_scale_init <= 0:
            raise ValueError('layer_scale_init must be positive and finite')
        if invalid_depth_ge_mm is not None and not math.isfinite(invalid_depth_ge_mm):
            raise ValueError('invalid_depth_ge_mm must be finite or null')
        super().__init__(size, **kwargs)
        self.invalid_depth_ge_mm = invalid_depth_ge_mm
        self.config.update(layer_scale_init=layer_scale_init, invalid_depth_ge_mm=invalid_depth_ge_mm)
        self.blocks = nn.ModuleList([DualReferenceBlock(
            self.embed_dim, 6 if size == 'S' else 12,
            self.config['min_frequency'], self.config['max_frequency'], layer_scale_init
        ) for _ in range(12)])
        # Distinguish missing measurements from a valid zero-deformation patch.
        self.invalid_embedding = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.invalid_embedding, std=.02)

    def tokens(self, images, patch_mask=None, mask_token=None):
        deformation, fields = self.fields(images)
        b, _, h, w = deformation.shape
        valid = torch.isfinite(images[:, :3]).all(1, keepdim=True)
        if self.invalid_depth_ge_mm is not None:
            valid = valid & (images[:, :1] < self.invalid_depth_ge_mm)
        if not torch.isfinite(images[:, 3:]).all():
            raise ValueError('Sensor coordinates must be finite')
        deformation = torch.where(valid, deformation, 0.)
        coverage = F.avg_pool2d(valid.float(), 16, 16).flatten(2).transpose(1, 2)
        positions = F.avg_pool2d(fields[:, :2], 16, 16).flatten(2).transpose(1, 2)
        means = F.avg_pool2d(deformation, 16, 16).flatten(2).transpose(1, 2) / coverage.clamp_min(1e-8)
        if patch_mask is not None:
            if patch_mask.dtype != torch.bool or patch_mask.shape != (b, (h // 16) * (w // 16)) or mask_token is None:
                raise ValueError('Invalid patch mask or missing mask token')
            pixels = patch_mask.reshape(b, 1, h // 16, w // 16).repeat_interleave(16, 2).repeat_interleave(16, 3)
            deformation = deformation.masked_fill(pixels, 0.)
            means = means.masked_fill(patch_mask[..., None], 0.)
        coordinates = torch.cat((positions, means), -1)
        x = self.patch_embed(deformation).flatten(2).transpose(1, 2)
        x = x + (1. - coverage).to(x.dtype) * self.invalid_embedding.to(x.dtype)
        if patch_mask is not None:
            # Replace validity information too; hidden data cannot affect student.
            x = torch.where(patch_mask[..., None], mask_token.to(x.dtype), x)
        for block in self.blocks:
            x = block(x, coordinates)
        x = self.norm(x)
        return torch.cat((x.mean(1, keepdim=True), x), 1)


def load_encoder(checkpoint, device='cpu'):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('method') != 'DINO_TactileRoPE_V2' or saved.get('architecture') != 'tactile_rope_v2':
        raise ValueError('Expected a DINO_TactileRoPE_V2 checkpoint')
    if saved.get('input_mode') != 'depth_deform_raw' or 'model_config' not in saved:
        raise ValueError('Missing raw-input/calibration metadata')
    model = TactileRoPEViTV2(saved['size'], **saved['model_config']['rope'])
    model.load_state_dict(saved['encoder'], strict=True)
    return model.to(device).eval()
