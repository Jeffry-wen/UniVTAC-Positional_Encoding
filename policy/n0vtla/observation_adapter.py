"""N0's image slots and discrete state prompt, without upstream data/JAX imports."""
from dataclasses import dataclass, replace

import numpy as np
import cv2
import torch
import torch.nn.functional as F

TACTILE_KEYS = ("left_wrist_left_tactile", "left_wrist_right_tactile",
                "right_wrist_left_tactile", "right_wrist_right_tactile")


def resize_rgb(image, size=224):
    """Aspect-preserving RGB resize with black padding before [-1,1] conversion."""
    height, width = image.shape[:2]
    ratio = max(width / size, height / size)
    h, w = max(1, int(height / ratio)), max(1, int(width / ratio))
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - h) // 2, (size - w) // 2
    canvas[top:top+h, left:left+w] = image
    return torch.from_numpy(canvas.copy()).permute(2, 0, 1).float() / 255


def native_tactile_from_arrays(data, mode="rgb_only"):
    """Build the selected three-channel representation before temporal differencing."""
    if mode in ("rgb_only", "marker_rgb"):
        key = "rgb" if mode == "rgb_only" else "rgb_marker"
        return resize_rgb(np.asarray(data[key]))
    from encoder.modality import tactile_from_arrays
    return tactile_from_arrays(data, mode, 224)


def read_native_tactile(h5, side, step, mode="rgb_only"):
    prefix = f"tactile/{side}_tactile"
    if mode in ("rgb_only", "marker_rgb"):
        key = "rgb" if mode == "rgb_only" else "rgb_marker"
        image = cv2.imdecode(np.frombuffer(h5[f"{prefix}/{key}"][step], np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {h5.filename}:{prefix}/{key}[{step}]")
        data = {key: cv2.cvtColor(image, cv2.COLOR_BGR2RGB)}
    else:
        data = {"marker": h5[f"{prefix}/marker"][step]}
        if mode == "depth_deform":
            data["depth"] = h5[f"{prefix}/depth"][step]
    return native_tactile_from_arrays(data, mode)


@dataclass
class Observation:
    images: dict
    image_masks: dict
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    expert_images: object = None
    token_ar_mask: object = None
    token_loss_mask: object = None

    def replace(self, **kwargs):
        return replace(self, **kwargs)


def tokenize_batch(batch, tokenizer, max_length):
    """Match N0's PI0.5 prompt: quantile-normalized 8D state, BOS, no EOS."""
    state = batch["observation.state"].detach().cpu().numpy()
    discrete = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
    rows, masks = [], []
    for task, values in zip(batch["task"], discrete):
        task = task.strip().replace("_", " ").replace("\n", " ")
        prompt = f"Task: {task}, State: {' '.join(map(str, values))};\nAction: "
        ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
        ids = ids[:max_length]
        masks.append([True] * len(ids) + [False] * (max_length - len(ids)))
        rows.append(ids + [0] * (max_length - len(ids)))
    batch["observation.language.tokens"] = torch.tensor(rows, dtype=torch.long)
    batch["observation.language.attention_mask"] = torch.tensor(masks, dtype=torch.bool)


def make_observation(batch):
    state = batch["observation.state"]
    template = next(value for key, value in batch.items() if key.startswith("observation.images.cam_"))
    images, masks = {}, {}
    for slot, key in (("base_0_rgb", "cam_high"), ("left_wrist_0_rgb", "cam_wrist"),
                      ("right_wrist_0_rgb", None)):
        image = batch.get(f"observation.images.{key}") if key else None
        images[slot] = image * 2 - 1 if image is not None else torch.zeros_like(template)
        masks[slot] = torch.full((len(state),), image is not None, dtype=torch.bool, device=state.device)
    for index, slot in enumerate(TACTILE_KEYS):
        side = ("left", "right")[index] if index < 2 else None
        key = f"observation.images.tac_{side}"
        image, baseline = batch.get(key), batch.get(key + ".baseline")
        if (image is None) != (baseline is None):
            raise ValueError(f"N0 tactile view requires both current and baseline: {key}")
        real = image is not None
        images[slot] = image * 2 - 1 if real else torch.zeros_like(template)
        images[slot + ".baseline"] = baseline * 2 - 1 if real else torch.zeros_like(template)
        mask = torch.full((len(state),), real, dtype=torch.bool, device=state.device)
        masks[slot], masks[slot + ".baseline"] = mask, mask
    return Observation(images, masks, F.pad(state, (0, 24)),
        batch["observation.language.tokens"], batch["observation.language.attention_mask"])


def masked_action_loss(loss, is_pad):
    valid = (~is_pad).unsqueeze(-1).expand(-1, -1, 8)
    if not valid.any():
        raise ValueError("N0 batch has no valid action targets")
    return loss[..., :8].masked_select(valid).mean()
