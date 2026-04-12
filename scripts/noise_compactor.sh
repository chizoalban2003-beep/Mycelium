#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PACK_SIZE="${SEDIMENT_PACK_SIZE:-6}"

cd "$ROOT_DIR"
"$PYTHON_BIN" scripts/resonance_pruning_protocol.py --sediment-pack-size "$PACK_SIZE"
