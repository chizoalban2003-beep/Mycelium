#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo ".venv not found. Create it with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements/base.txt" >&2
  exit 1
fi

source .venv/bin/activate

# Workaround for environments that cannot write __pycache__ / .pyc files.
export PYTHONDONTWRITEBYTECODE=1

# Default to Hive enabled for Parent hub.
export HIVE_ENABLED="${HIVE_ENABLED:-true}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec uvicorn mycelium_app.main:app --host "$HOST" --port "$PORT"
