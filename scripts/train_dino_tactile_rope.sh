#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ROOT"
exec bash train_encoder.sh --pretrained-mode univtac --method DINO_TactileRoPE \
    --size "${ENCODER_SIZE:-S}" --input depth_deform_raw "$@"
