#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo ".venv not found. Create it with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements/base.txt" >&2
  exit 1
fi

source .venv/bin/activate

BASE_URL="${PARENT_HUB_URL:-http://127.0.0.1:8000}"
DEVICE_ID="${NEXUS_DEVICE_ID:-child-1}"
TOKEN="${HIVE_INGEST_TOKEN:-}"

export NEXUS_DEVICE_ID="$DEVICE_ID"

if [[ -z "$TOKEN" ]]; then
  echo "HIVE_INGEST_TOKEN is empty. For SaaS mode, set it (or implement per-user auth tokens)." >&2
fi

echo "Child starting: device=$DEVICE_ID parent=$BASE_URL"

# 1) Passive telemetry (local observation) -> posts to /api/nexus/telemetry/ingest (requires bearer auth today)
# If you want the child to run telemetry, use scripts/passive_telemetry_daemon.py with --token or --email/--password.
#
# 2) Minimal Hive connectivity smoketest (headless): concept import via X-Hive-Token
PARENT_HUB_URL="$BASE_URL" HIVE_INGEST_TOKEN="$TOKEN" python scripts/child_smoketest_ingest.py
