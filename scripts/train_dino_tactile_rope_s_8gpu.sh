#!/usr/bin/env bash
# Single node, eight GPUs. Global batch 64 (8 images/GPU per encoder view).
# Encoders run in parallel; DINO heads/Sinkhorn/KoLeo use the global batch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
cd "$ROOT"
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/train_dino_tactile_rope_s_8gpu.sh [--dry-run] [train.py options]'
    echo 'Defaults: DINO触觉RoPE-S, global batch 64, lr 1e-4, 100000 frames, up to 30 epochs.'
    echo 'Override Python/GPU IDs with PYTHON_BIN and CUDA_VISIBLE_DEVICES.'
    echo 'Examples: --max-steps 2 --max-val-steps 1 --epochs 1; --resume /path/last.pth'
    exit 0
fi
dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
# Preserve eight-GPU execution and the requested S/global-batch configuration.
for arg in "$@"; do
    case "$arg" in
        --size|--size=*|--method|--method=*|--input-mode|--input-mode=*|--batch-size|--batch-size=*|--device|--device=*|--config|--config=*)
            echo "This launcher fixes S/raw input/global batch 64/cuda/config; unsupported override: $arg" >&2
            exit 2 ;;
    esac
done
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#gpu_ids[@]} != 8 || "$CUDA_VISIBLE_DEVICES" == *, ]]; then
    echo 'Exactly eight comma-separated GPU IDs are required.' >&2; exit 2
fi
declare -A seen=()
for gpu in "${gpu_ids[@]}"; do
    if [[ -z "$gpu" || -n "${seen[$gpu]:-}" ]]; then
        echo 'GPU IDs must be nonempty and unique.' >&2; exit 2
    fi
    seen[$gpu]=1
done
command=(bash "$ROOT/train_encoder.sh" --pretrained-mode univtac
    --method DINO_TactileRoPE --size S --input depth_deform_raw
    --config "$ROOT/encoder/DINO_TactileRoPE/config.yml"
    -- --multi-gpu --device cuda --batch-size 64 "$@")
if ((dry_run)); then
    printf 'PYTHON_BIN=%q CUDA_VISIBLE_DEVICES=%q ' "$PYTHON_BIN" "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
"$PYTHON_BIN" - <<'PY'
import torch
n = torch.cuda.device_count()
if n != 8:
    raise SystemExit(f'Requires 8 visible CUDA GPUs, found {n}; no training started.')
for i in range(n):
    with torch.cuda.device(i):
        if not torch.cuda.is_bf16_supported():
            raise SystemExit(f'GPU {i} does not support the configured BF16 training.')
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
PY
printf 'DINO触觉RoPE-S: 8 GPUs, global batch=64, per-GPU encoder batch=8; global DINO loss.\n'
exec "${command[@]}"
