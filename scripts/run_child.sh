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
AUTO_CAPTURE="${CHILD_AUTO_CAPTURE_TRAJECTORIES:-false}"
AUTO_CAPTURE_WINDOW="${CHILD_TRAJECTORY_WINDOW_SIZE:-3}"
AUTO_CAPTURE_COOLDOWN="${CHILD_TRAJECTORY_COOLDOWN_SECONDS:-600}"
AUTO_CAPTURE_INCLUDE="${CHILD_TRAJECTORY_MUST_INCLUDE_CSV:-mycelium}"

# For telemetry + trajectory capture auth (bearer): prefer token, else email/password.
CHILD_BEARER_TOKEN="${CHILD_BEARER_TOKEN:-}"
CHILD_EMAIL="${CHILD_EMAIL:-}"
CHILD_PASSWORD="${CHILD_PASSWORD:-}"

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

if [[ "${AUTO_CAPTURE,,}" == "true" ]]; then
  echo "Auto trajectory capture enabled. Starting passive telemetry daemon..."

  AUTH_ARGS=()
  if [[ -n "$CHILD_BEARER_TOKEN" ]]; then
    AUTH_ARGS+=(--token "$CHILD_BEARER_TOKEN")
  elif [[ -n "$CHILD_EMAIL" && -n "$CHILD_PASSWORD" ]]; then
    AUTH_ARGS+=(--email "$CHILD_EMAIL" --password "$CHILD_PASSWORD")
  else
    echo "Auto capture requested but missing auth. Set CHILD_BEARER_TOKEN or CHILD_EMAIL+CHILD_PASSWORD." >&2
    exit 2
  fi

  python scripts/passive_telemetry_daemon.py \
    --base-url "$BASE_URL" \
    --device-id "$DEVICE_ID" \
    --trajectory-capture-enabled \
    --trajectory-window-size "$AUTO_CAPTURE_WINDOW" \
    --trajectory-cooldown-seconds "$AUTO_CAPTURE_COOLDOWN" \
    --trajectory-must-include-csv "$AUTO_CAPTURE_INCLUDE" \
    "${AUTH_ARGS[@]}"
fi
