"""Public inference API for V1/V2 frozen tactile encoders (not an ACT policy)."""
import torch
from .modality import tactile_from_arrays


def load_tactile_rope(checkpoint, device='cpu'):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = (saved.get('method'), saved.get('architecture'))
    if identity == ('DINO_TactileRoPE', 'tactile_rope_v1'):
        from .DINO_TactileRoPE.encoder import TactileRoPEViT as Encoder
    elif identity == ('DINO_TactileRoPE_V2', 'tactile_rope_v2'):
        from .DINO_TactileRoPE_V2.encoder import TactileRoPEViTV2 as Encoder
    else:
        raise ValueError(f'Unsupported tactile RoPE checkpoint: {identity}')
    if saved.get('input_mode') != 'depth_deform_raw':
        raise ValueError('Expected depth_deform_raw checkpoint')
    model = Encoder(saved['size'], **saved['model_config']['rope'])
    model.load_state_dict(saved['encoder'], strict=True)
    model.requires_grad_(False).to(device).eval()
    metadata = {key: saved.get(key) for key in (
        'method', 'architecture', 'size', 'model_config', 'epoch', 'step', 'input_mode')}
    return model, metadata


@torch.inference_mode()
def infer_fields(model, fields):
    """Raw [B,5,H,W] or [5,H,W]: depth mm, u/v/x/y native pixels.

    Return float32 global [B,D] and patch [B,N,D] features on CPU.
    H/W must be multiples of 16. No random crops or normalization beyond model config.
    """
    fields = torch.as_tensor(fields, dtype=torch.float32)
    if fields.ndim == 3:
        fields = fields.unsqueeze(0)
    if not torch.isfinite(fields).all():
        raise ValueError('Inference inputs must be finite')
    model.eval()
    tokens = model.tokens(fields.to(next(model.parameters()).device))
    return {'global_features': tokens[:, 0].float().cpu(),
            'patch_features': tokens[:, 1:].float().cpu()}


def prepare_observation(depth, markers, image_size=224):
    """Depth [H,W] in mm; markers [2,N,2] ordered reference/current, in pixels."""
    return tactile_from_arrays({'depth': depth, 'marker': markers},
                               'depth_deform_raw', image_size=image_size)
