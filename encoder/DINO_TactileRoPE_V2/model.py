"""Keep V1 augmentations and DINO objectives; change only the encoder."""
from DINOv2.model import DINOv2
from DINO_TactileRoPE.model import DINOTactileRoPE
from .encoder import TactileRoPEViTV2


class DINOTactileRoPEV2(DINOTactileRoPE):
    def __init__(self, size='S', in_chans=5, rope=None, **kwargs):
        if in_chans != 5:
            raise ValueError('DINO_TactileRoPE_V2 requires depth_deform_raw')
        DINOv2.__init__(self, size, in_chans=in_chans,
                        encoder=TactileRoPEViTV2(size, **(rope or {})), **kwargs)
