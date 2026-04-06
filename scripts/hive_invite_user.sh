#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASE_URL:-}" ]]; then
  echo "ERROR: BASE_URL is required"
  exit 1
fi
if [[ -z "${OWNER_TOKEN:-}" ]]; then
  echo "ERROR: OWNER_TOKEN is required"
  exit 1
fi
if [[ -z "${INVITE_EMAIL:-}" ]]; then
  echo "ERROR: INVITE_EMAIL is required"
  exit 1
fi
if [[ -z "${INVITE_PASS:-}" ]]; then
  echo "ERROR: INVITE_PASS is required"
  exit 1
fi

INVITE_NAME="${INVITE_NAME:-Hive Member}"
ROLE="${ROLE:-viewer}" # owner|editor|viewer
PROJECT_ID="${PROJECT_ID:-}"

echo "[1/2] Ensure invited user exists..."
set +e
REG=$(curl -sS -X POST "$BASE_URL/api/auth/register" \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"$INVITE_EMAIL\",\"password\":\"$INVITE_PASS\",\"full_name\":\"$INVITE_NAME\"}")
set -e

echo "register_response=$REG"

if [[ -n "$PROJECT_ID" ]]; then
  echo "[2/2] Add invited user to project $PROJECT_ID as $ROLE..."
  ADD=$(curl -sS -X POST "$BASE_URL/api/projects/$PROJECT_ID/members" \
    -H "Authorization: Bearer $OWNER_TOKEN" \
    -H 'Content-Type: application/json' \
    -d "{\"email\":\"$INVITE_EMAIL\",\"role\":\"$ROLE\"}")
  echo "member_response=$ADD"
else
  echo "[2/2] PROJECT_ID not set, skipped project membership."
fi

echo "Done. Share login with invited user: email=$INVITE_EMAIL"
