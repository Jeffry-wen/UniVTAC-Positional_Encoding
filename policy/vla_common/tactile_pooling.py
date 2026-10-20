"""Policy-side spatial pooling; no parameters or checkpoint tensor keys."""
from functools import wraps
from math import isqrt

import torch
from torch.nn import functional as F


def pool_spatial_tokens(tokens, grid_size, prefix_tokens=0):
    """Pool each temporal slice to 3x3, preserving any leading CLS tokens."""
    height, width = grid_size
    batch, count, channels = tokens.shape
    patches = tokens[:, prefix_tokens:]
    if patches.shape[1] % (height * width):
        raise ValueError(f'Cannot reshape {count} tokens into {grid_size} spatial grids')
    frames = patches.shape[1] // (height * width)
    grid = patches.reshape(batch * frames, height, width, channels).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid, (3, 3)).flatten(2).transpose(1, 2)
    pooled = pooled.reshape(batch, frames * 9, channels)
    return torch.cat((tokens[:, :prefix_tokens], pooled), dim=1) if prefix_tokens else pooled


def tactile_token_grid(backbone, method, pretrained_mode, token_count):
    """Return the spatial grid, independently of the number of temporal slices."""
    if pretrained_mode == 'univtac':
        side = (7 if method == 'VAE' else isqrt(token_count) if method == 'ResNet'
                else backbone.grid_size)
        return side, side
    if method == 'Wan2.2-VAE':
        side = 14 // backbone.wan_patch_size
        return side, side
    model = backbone.model
    config = getattr(model, 'config', None)
    patch = (16 if method == 'AnyTouch2' else
             getattr(model, 'patch_size', getattr(config, 'patch_size', 16)))
    ph, pw = (patch, patch) if isinstance(patch, int) else patch
    return backbone.hw[0] // ph, backbone.hw[1] // pw


def pool_tactile_tokens(tokens, backbone, method, pretrained_mode, *, includes_cls=False):
    prefix = int(includes_cls)
    grid = tactile_token_grid(backbone, method, pretrained_mode, tokens.shape[1] - prefix)
    return pool_spatial_tokens(tokens, grid, prefix)


def pool_act_tactile_outputs(forward):
    """Pool ACT features and their positions without wrapping registered modules."""
    @wraps(forward)
    def pooled_forward(self, *args, **kwargs):
        features, positions = forward(self, *args, **kwargs)
        if not getattr(self, 'pooling3x3', False) or self.tactile_type != 'full':
            return features, positions
        pooled_features, pooled_positions = [], []
        for feature, position in zip(features, positions):
            if feature.shape[-2] > 1:
                feature = F.adaptive_avg_pool2d(feature, (3, 3))
                position = F.adaptive_avg_pool2d(position, (3, 3))
            else:
                includes_cls = (self.pooling_pretrained_mode == 'univtac'
                                and self.pooling_method in ('MAE', 'DINOv2'))
                def pool(value):
                    tokens = value.squeeze(2).transpose(1, 2)
                    tokens = pool_tactile_tokens(tokens, self.backbone, self.pooling_method,
                                                 self.pooling_pretrained_mode, includes_cls=includes_cls)
                    return tokens.transpose(1, 2).unsqueeze(2)
                feature, position = pool(feature), pool(position)
            pooled_features.append(feature)
            pooled_positions.append(position)
        return pooled_features, pooled_positions
    return pooled_forward
