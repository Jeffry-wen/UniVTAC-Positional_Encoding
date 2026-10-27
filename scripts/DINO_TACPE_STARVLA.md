# Tactile RoPE V1/V2 + StarVLA-GR00T

These runs replace the legacy DINOv2 tactile backbone with the trained positional-encoding variants. They use the same StarVLA-GR00T policy, action objective and optimizer settings as `train_starvla_groot_grid.sh` and `policy/starvla_groot/base_config.yml`.

| Run | Checkpoint method | Architecture | Launcher |
|---|---|---|---|
| DINO_tacpe_v1_starvla | DINO_TactileRoPE | tactile_rope_v1 | `scripts/train_dino_tacpe_v1_starvla_8gpu.sh` |
| DINO_tacpe_v2_starvla | DINO_TactileRoPE_V2 | tactile_rope_v2 | `scripts/train_dino_tacpe_v2_starvla_8gpu.sh` |

## Training contract

- Single node, **8 DDP ranks**, global batch **16**, per-GPU batch **2**. This preserves the legacy ablation's global batch; eight GPUs do not multiply it to 128.
- Multi-task, all tasks/episodes and both camera views; relative arm-joint actions and absolute gripper target.
- 30,000 optimizer steps; action chunk 32; execute 30 actions per replan; checkpoint every 5,000 steps; image size 224; seed 42 by default.
- `TACTILE_MODE=encode`, `TACTILE_TYPE=cls`, `TACTILE_INSERT_LOCATION=encoder`. One global tactile token per sensor/frame, linearly projected into the VLM input dimension.
- The pretrained RoPE architecture exposes its mean-pooled patch representation at token index zero. This is its existing **CLS-interface global token**, not a newly added learned CLS parameter. The pretrained architecture is retained exactly.
- The tactile backbone and projection remain trainable. StarVLA's action flow-matching loss backpropagates through them; no DINO pretraining heads, teacher or auxiliary DINO loss are added.
- The existing GR00T base config remains unchanged: AdamW lr 2.5e-5, betas (0.9, 0.95), weight decay .01, gradient clipping 1.0, warmup 1,000 steps and decay lr 2.5e-6. Other VLM/action settings are inherited unchanged.

## Physical positional inputs

The policy HDF5 loader uses the **same `encoder.modality.read_tactile` / `raw_fields` conversion as encoder training**. For each left/right sensor it reads camera depth and reference/current markers (preferring `marker_full` when present), interpolates displacement, and packs:

`[depth_mm, u_native_pixels, v_native_pixels, x_native_pixels, y_native_pixels]` → `[B,5,224,224]`.

The x/y coordinate values remain in native sensor coordinates after sampling to 224×224. There is no RGB scaling, per-frame normalization, clipping, or replacement by a generic position table. The backbone computes rest-relative depth and RoPE phases using **`model_config.rope` from the trained checkpoint**, including reference depth, spatial/deformation scales and V2 validity settings. Input representation, architecture and calibration metadata are validated on load. The adapter does not resize raw fields after they have been prepared.

The policy checkpoint persists this architecture/calibration metadata and all fine-tuned backbone/projection weights, so restoration does not require the original encoder export path. Deployment also passes depth and full markers into the raw-fields conversion.

## Prerequisites and launch

Use a Python environment containing `requirements_train.txt` (a DINO-only environment is insufficient). Provision the same baseline assets under this checkout:

- `resources/data/isaac51/<task>/hdf5/` with robot demonstrations, camera RGB, tactile depth and reference/current markers.
- `resources/pretrained_models/Qwen3.5-2B/` with the baseline VLM.
- V1/V2 **`encoder.pth`**, not `last.pth`, from [jeffry77/tactile_rope_dinov2](https://huggingface.co/jeffry77/tactile_rope_dinov2).

Pass an absolute checkpoint path, or place each export at `resources/pretrained_models/tactile_rope/<method>/encoder.pth`.

```bash
# Inspect parameters without loading models, using GPUs or starting training.
bash scripts/train_dino_tacpe_v1_starvla_8gpu.sh --dry-run
bash scripts/train_dino_tacpe_v2_starvla_8gpu.sh --dry-run

# Run each command on its own eight-GPU allocation (or sequentially).
PYTHON_BIN=/path/to/starvla-env/bin/python \
ENCODER_CKPT=/path/to/DINO_TactileRoPE/encoder.pth \
bash scripts/train_dino_tacpe_v1_starvla_8gpu.sh

PYTHON_BIN=/path/to/starvla-env/bin/python \
ENCODER_CKPT=/path/to/DINO_TactileRoPE_V2/encoder.pth \
bash scripts/train_dino_tacpe_v2_starvla_8gpu.sh
```

`CUDA_VISIBLE_DEVICES` defaults to `0,1,2,3,4,5,6,7`. Each launcher checks for eight unique IDs, matching checkpoint metadata, baseline assets and BF16-capable GPUs before starting torchrun. `OUTPUT_DIR` may be set to a new output directory. Do not run both launchers simultaneously on the same eight GPUs. These scripts do not submit DLC jobs themselves.

## Validation

```bash
python scripts/test_tactile_rope_policy.py
python scripts/test_tacpe_starvla_launchers.py
python scripts/test_tactile_rope_inference.py
```

The adapter test covers HDF5→policy physical fields, full-marker preference, exact pretrained feature loading, coordinate-sensitive output, backbone gradients and parameter updates, temporal tokens, checkpoint identity rejection and path-free policy-weight restoration. Launcher tests check eight-GPU validation and preservation of the baseline global batch and training budget. These checks do not substitute for a full eight-GPU VLM training run with real demonstrations.
