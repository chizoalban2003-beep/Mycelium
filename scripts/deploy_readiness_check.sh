#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
TOKEN="${TOKEN:-}"
EXPECTED_GIT_SHA="${EXPECTED_GIT_SHA:-}"
EXPECTED_ANDROID_PACKAGE="${EXPECTED_ANDROID_PACKAGE:-}"
EXPECTED_ANDROID_FINGERPRINTS_CSV="${EXPECTED_ANDROID_FINGERPRINTS_CSV:-}"
REQUIRE_ASSETLINKS="${REQUIRE_ASSETLINKS:-false}"

STRICT_PRODUCTION=true "$PYTHON_BIN" scripts/db_migration_preflight.py
BASE_URL="$BASE_URL" "$PYTHON_BIN" scripts/health_smoketest.py

if [[ -n "$TOKEN" ]]; then
  ARGS=(
    --base-url "$BASE_URL"
    --token "$TOKEN"
  )
  if [[ -n "$EXPECTED_GIT_SHA" ]]; then
    ARGS+=(--expected-git-sha "$EXPECTED_GIT_SHA")
  fi
  if [[ -n "$EXPECTED_ANDROID_PACKAGE" ]]; then
    ARGS+=(--expected-android-package "$EXPECTED_ANDROID_PACKAGE")
  fi
  if [[ -n "$EXPECTED_ANDROID_FINGERPRINTS_CSV" ]]; then
    ARGS+=(--expected-android-fingerprints-csv "$EXPECTED_ANDROID_FINGERPRINTS_CSV")
  fi
  if [[ "${REQUIRE_ASSETLINKS,,}" == "true" ]]; then
    ARGS+=(--require-assetlinks)
  fi
  "$PYTHON_BIN" scripts/public_alpha_checklist.py "${ARGS[@]}"
fi

echo "deploy_readiness_check=ok"
