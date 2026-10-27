#!/usr/bin/env bash
# Shared implementation; invoke the V1/V2 wrappers instead.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${1:?Expected encoder method}"; shift
case "$METHOD" in
  DINO_TactileRoPE) RUN_TAG=DINO_tacpe_v1_starvla ;;
  DINO_TactileRoPE_V2) RUN_TAG=DINO_tacpe_v2_starvla ;;
  *) echo "Unsupported tactile encoder: $METHOD" >&2; exit 2 ;;
esac
export RUN_TAG
DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1; shift ;;
  --help|-h)
    echo 'Single-node 8-GPU StarVLA-GR00T with trainable tactile RoPE CLS-interface tokens.'
    echo 'Matches the legacy tactile grid: global batch 16, 30000 steps, chunk 32, save every 5000.'
    echo 'Set PYTHON_BIN to a StarVLA-capable environment; ENCODER_CKPT to the matching encoder.pth.'
    echo 'Weights default to resources/pretrained_models/tactile_rope/<method>/encoder.pth.'
    echo 'Download the matching export from jeffry77/tactile_rope_dinov2 or pass its local path.'
    echo 'Usage: bash <v1-or-v2-wrapper> [--dry-run]'
    exit 0 ;;
esac
if (($#)); then echo "Unexpected argument: $1" >&2; exit 2; fi
export PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if ((${#GPU_IDS[@]} != 8)) || [[ "$CUDA_VISIBLE_DEVICES" == *, ]]; then
  echo 'Exactly eight GPU IDs are required.' >&2; exit 2
fi
declare -A SEEN=()
for gpu in "${GPU_IDS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${SEEN[$gpu]:-}" ]]; then
    echo "Invalid or repeated GPU ID: $gpu" >&2; exit 2
  fi
  SEEN[$gpu]=1
done
# Preserve the legacy grid's optimizer-update budget and effective batch size.
export NUM_PROCESSES=8 POLICY=starvla_groot TRAINING_MODE=multi_task
export TASKS='' EPISODES=0 CAMERAS=all ACTION_MODE=relative_joint
export STEPS=30000 BATCH_SIZE=16 CHUNK_SIZE=32 N_ACTION_STEPS=30 SAVE_FREQ=5000
export IMAGE_SIZE=224 TACTILE_MODE=encode TACTILE_INPUT_MODE=depth_deform_raw
export TACTILE_TYPE=cls TACTILE_INSERT_LOCATION=encoder
export ENCODER_PRETRAINED_MODE=univtac ENCODER_METHOD="$METHOD" ENCODER_SIZE=S ENCODER_INIT=latest
export ENCODER_CKPT="${ENCODER_CKPT:-$ROOT/resources/pretrained_models/tactile_rope/$METHOD/encoder.pth}"
export WORKERS="${WORKERS:-4}"
cd "$ROOT"
if ((DRY_RUN)); then
  # Offline inspection; do not create output dirs, load weights, or initialize CUDA.
  for key in RUN_TAG POLICY NUM_PROCESSES CUDA_VISIBLE_DEVICES TRAINING_MODE TASKS EPISODES CAMERAS ACTION_MODE \
             STEPS BATCH_SIZE CHUNK_SIZE N_ACTION_STEPS SAVE_FREQ IMAGE_SIZE WORKERS \
             TACTILE_MODE TACTILE_INPUT_MODE TACTILE_TYPE TACTILE_INSERT_LOCATION \
             ENCODER_PRETRAINED_MODE ENCODER_METHOD ENCODER_SIZE ENCODER_CKPT PYTHON_BIN; do
    printf '%s=%q\n' "$key" "${!key}"
  done
  printf 'per_gpu_batch=2\ncommand=bash %q\n' "$ROOT/train_policy.sh"
  exit 0
fi
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import torch
root = Path.cwd()
checkpoint = Path(os.environ['ENCODER_CKPT'])
if not checkpoint.is_file():
    raise SystemExit(f'Missing encoder.pth: {checkpoint}; set ENCODER_CKPT to your trained export.')
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
method = os.environ['ENCODER_METHOD']
arch = {'DINO_TactileRoPE':'tactile_rope_v1', 'DINO_TactileRoPE_V2':'tactile_rope_v2'}[method]
if (state.get('method'), state.get('architecture'), state.get('size'), state.get('input_mode')) != (method, arch, 'S', 'depth_deform_raw'):
    raise SystemExit('Checkpoint identity/input mismatch; do not interchange V1 and V2 exports.')
if 'encoder' not in state or not state.get('model_config', {}).get('rope'):
    raise SystemExit('Expected encoder.pth with encoder weights and model_config.rope calibration.')
for directory in ('resources/data/isaac51', 'resources/pretrained_models/Qwen3.5-2B'):
    if not (root / directory).is_dir():
        raise SystemExit(f'Missing {root / directory}; provision the same policy data/VLM as the baseline.')
if torch.cuda.device_count() != 8:
    raise SystemExit('Eight visible CUDA GPUs required; training not started.')
for i in range(8):
    with torch.cuda.device(i):
        if not torch.cuda.is_bf16_supported():
            raise SystemExit(f'GPU {i} does not support BF16.')
print(f'{method}: pretrained backbone remains trainable; raw depth/u/v/x/y; global batch 16 (2/GPU).', flush=True)
PY
exec bash "$ROOT/train_policy.sh"
