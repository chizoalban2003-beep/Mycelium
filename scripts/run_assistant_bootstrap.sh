#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASE_URL:-}" ]]; then
  echo "ERROR: BASE_URL is required (e.g. https://your-railway-domain)"
  exit 1
fi
if [[ -z "${EMAIL:-}" ]]; then
  echo "ERROR: EMAIL is required"
  exit 1
fi
if [[ -z "${PASS:-}" ]]; then
  echo "ERROR: PASS is required"
  exit 1
fi

echo "[1/4] Register user (idempotent)..."
set +e
REG_OUT=$(curl -sS -X POST "$BASE_URL/api/auth/register" \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASS\",\"full_name\":\"Assistant Bootstrap\"}")
REG_CODE=$?
set -e
if [[ $REG_CODE -ne 0 ]]; then
  echo "Register call failed"
  echo "$REG_OUT"
fi

echo "[2/4] Login + token..."
LOGIN_JSON=$(curl -sS -X POST "$BASE_URL/api/auth/login" \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode "username=$EMAIL" \
  --data-urlencode "password=$PASS")

TOKEN=$(printf '%s' "$LOGIN_JSON" | python3 -c 'import json,sys; \
raw=sys.stdin.read().strip(); \
\
\
\
print((json.loads(raw).get("access_token","") if raw else ""))')

if [[ -z "$TOKEN" ]]; then
  echo "ERROR: login failed"
  echo "$LOGIN_JSON"
  exit 1
fi

echo "Token prefix: ${TOKEN:0:20}..."

echo "[3/4] Force telemetry assistant tick..."
TICK_JSON=$(curl -sS -X POST "$BASE_URL/api/nexus/telemetry/assistant/tick" \
  -H "Authorization: Bearer $TOKEN")
echo "$TICK_JSON"

echo "[4/4] Fetch latest unseen nudges..."
NUDGES_JSON=$(curl -sS "$BASE_URL/api/nexus/nudges/recent?limit=5&unseen_only=true" \
  -H "Authorization: Bearer $TOKEN")
echo "$NUDGES_JSON"

echo "Done. Open $BASE_URL on phone, install app (PWA), and check nudge banner."
