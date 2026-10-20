#!/usr/bin/env bash
# Train one tactile encoder. Optimization defaults live in encoder/<method>/config.yml.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
PRETRAINED_MODE="${PRETRAINED_MODE:-univtac}" # univtac | tactile | vision
METHOD="${ENCODER_METHOD:-ResNet}"
SIZE="${ENCODER_SIZE:-${VIT_SIZE:-}}"
INPUT="${INPUT_MODE:-marker_rgb}"
DATA_ROOT="$ROOT/resources/data/contact"
CONFIG=""
EXTRA=()
usage() {
    echo "Usage: $0 [--pretrained-mode univtac|tactile|vision] [--method METHOD] [--size S|B|L] [--input INPUT_MODE] [--data-root DIR] [--config FILE] [-- additional train.py options]"
    echo "univtac: ResNet VAE MAE DINOv2 I-JEPA V-JEPA; tactile: MAE DINOv2 I-JEPA V-JEPA AnyTouch2; vision: DINOv3 SigLIP2 V-JEPA2 Wan2.2-VAE"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained-mode) PRETRAINED_MODE="$2"; shift 2 ;;
        --method|--model) METHOD="$2"; shift 2 ;;
        --size) SIZE="$2"; shift 2 ;;
        --input|--input-mode) INPUT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --) shift; EXTRA+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ -z "$SIZE" ]]; then
    SIZE="$(PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c 'import sys; from encoder.pretrained_registry import default_size; print(default_size(*sys.argv[1:]))' "$PRETRAINED_MODE" "$METHOD")"
fi
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c 'import sys; from encoder.pretrained_registry import validate_identity; validate_identity(*sys.argv[1:])' "$PRETRAINED_MODE" "$METHOD" "$SIZE"
case "$INPUT" in marker_only|rgb_only|marker_rgb|depth_deform_20|depth_deform_80|depth_deform_160|depth_deform_240|depth_deform_nonlinear|depth_deform_nonlinear_radial) ;; *) echo "Invalid input: $INPUT" >&2; exit 2 ;; esac
if [[ "$PRETRAINED_MODE" == univtac ]]; then CONFIG="${CONFIG:-$ROOT/encoder/$METHOD/config.yml}"; fi
RUN_DIR="$ROOT/encoder_results/$INPUT/$PRETRAINED_MODE/$METHOD/$SIZE/$(date +%Y%m%d-%H%M%S-%6N)"
for ((i=0; i<${#EXTRA[@]}; i++)); do
    if [[ "${EXTRA[i]}" == --output-dir && $((i + 1)) -lt ${#EXTRA[@]} ]]; then
        RUN_DIR="${EXTRA[i+1]}"
    elif [[ "${EXTRA[i]}" == --output-dir=* ]]; then
        RUN_DIR="${EXTRA[i]#--output-dir=}"
    fi
done
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/train.log"
TEE_ARGS=()
for arg in "${EXTRA[@]}"; do
    if [[ "$arg" == --resume || "$arg" == --resume=* ]]; then TEE_ARGS=(-a); fi
done
{
    DISPLAY_METHOD="$METHOD"
    if [[ "$PRETRAINED_MODE" == univtac && "$METHOD" == V-JEPA ]]; then DISPLAY_METHOD='V-JEPA*'; fi
    echo "Model=$DISPLAY_METHOD size=$SIZE input=$INPUT batch_size=64 config=$CONFIG output=$RUN_DIR"
    if [[ "$PRETRAINED_MODE" == univtac ]]; then
        "$PYTHON_BIN" -u "$ROOT/encoder/$METHOD/train.py" --size "$SIZE" --input-mode "$INPUT" \
            --data-root "$DATA_ROOT" --config "$CONFIG" --output-dir "$RUN_DIR" "${EXTRA[@]}"
    else
        CONFIG_ARGS=()
        if [[ -n "$CONFIG" ]]; then CONFIG_ARGS=(--config "$CONFIG"); fi
        "$PYTHON_BIN" -u "$ROOT/encoder/train_external.py" --pretrained-mode "$PRETRAINED_MODE" --method "$METHOD" \
            --size "$SIZE" --input-mode "$INPUT" --data-root "$DATA_ROOT" --output-dir "$RUN_DIR" "${CONFIG_ARGS[@]}" "${EXTRA[@]}"
    fi
} 2>&1 | tee "${TEE_ARGS[@]}" "$LOG"
