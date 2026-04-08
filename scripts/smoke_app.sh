#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
TIMEOUT_SECONDS="${2:-30}"

deadline=$((SECONDS + TIMEOUT_SECONDS))

echo "Waiting for ${BASE_URL}/health ..."
until curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for ${BASE_URL}/health" >&2
    exit 1
  fi
  sleep 1
done

echo "Health OK"
curl -fsS "${BASE_URL}/health"
echo
echo "Live state:"
curl -fsS "${BASE_URL}/api/nexus/live/state?window_minutes=30"
