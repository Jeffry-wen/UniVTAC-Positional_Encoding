#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.device_count() == 1, "Select exactly one GPU"; assert torch.cuda.is_bf16_supported(), "BF16 GPU required"; print(torch.cuda.get_device_name(0))'
cd "$ROOT"
exec bash train_encoder.sh --pretrained-mode univtac --method DINO_TactileRoPE_V2 \
    --size S --input depth_deform_raw "$@"
