#!/usr/bin/env bash
# Node 4: depth_deform_80 by default (override TACTILE_INPUT_MODE), full; six methods, two four-GPU DDP jobs concurrently.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/train_starvla_groot_node.sh" "${TACTILE_INPUT_MODE:-depth_deform_80}" full "$@"
