#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/train_dino_tacpe_starvla_8gpu_common.sh" DINO_TactileRoPE_V2 "$@"
