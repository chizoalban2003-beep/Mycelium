#!/usr/bin/env bash
set -euo pipefail

# Rebuild the local dev virtualenv.
# This exists because Nexus Homeostasis may notice when the "limb" (.venv)
# is missing and recommend the repair.

cd "$(dirname "$0")/.."

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "ERROR: python3 not found" >&2
  exit 1
fi

echo "Using: $($PY --version)"

# Basic disk check (avoid repeating the ENOSPC issues).
FREE_MB=$(df -Pm . | awk 'NR==2 {print $4}')
if [[ -n "$FREE_MB" && "$FREE_MB" -lt 1024 ]]; then
  echo "WARN: Low disk free: ${FREE_MB}MB. Consider cleaning caches before install." >&2
fi

if [[ ! -d .venv ]]; then
  echo "Creating .venv..."
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements/base.txt

echo "\nOK: environment ready"
echo "Run: uvicorn mycelium_app.main:app --reload --port 8000"
