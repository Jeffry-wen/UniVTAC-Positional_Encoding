"""DINO触觉RoPE; original DINO/iBOT/KoLeo objectives, no auxiliary supervision."""
import torch
from torch import nn
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF
from DINOv2.model import DINOv2
from .encoder import TactileRoPEViT


class DINOTactileRoPE(DINOv2):
    def __init__(self, size='S', in_chans=5, rope=None, **kwargs):
        if in_chans != 5:
            raise ValueError('DINO_TactileRoPE requires depth_deform_raw input')
        encoder = TactileRoPEViT(size, **(rope or {}))
        super().__init__(size, in_chans=in_chans, encoder=encoder, **kwargs)

    def enable_token_parallel(self):
        # Plain dict intentionally keeps execution adapters out of state_dict.
        # Both adapters reference the registered encoder/teacher parameters.
        self._token_parallel = {
            id(encoder): nn.DataParallel(_TokenForward(encoder))
            for encoder in (self.encoder, self.teacher)
        }

    def encode_tokens(self, encoder, view, **kwargs):
        parallel = getattr(self, '_token_parallel', {}).get(id(encoder))
        if parallel is None:
            return encoder.tokens(view, **kwargs)
        mask = kwargs.get('patch_mask')
        token = kwargs.get('mask_token')
        # DataParallel scatters every tensor on dim 0. Expand the shared token
        # to batch length first; backward through expand sums its gradients.
        expanded = token.expand(len(view), -1, -1) if token is not None else None
        return parallel(view, patch_mask=mask, expanded_mask_token=expanded)

    def make_views(self, images):
        views = []
        for local in (False, False, True):
            output = []
            for image in images:
                i, j, h, w = RandomResizedCrop.get_params(image, (.55, .8) if local else (.8, 1.), (.85, 1.15))
                # Crop all five channels together; retain original coordinate frame
                # and physical magnitudes. No color jitter or value clipping.
                output.append(TF.resized_crop(image, i, j, h, w, [224, 224]))
            views.append(torch.stack(output))
        return views


class _TokenForward(nn.Module):
    """DataParallel adapter; the original encoder remains the state-dict owner."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, images, patch_mask=None, expanded_mask_token=None):
        return self.encoder.tokens(images, patch_mask=patch_mask, mask_token=expanded_mask_token)
