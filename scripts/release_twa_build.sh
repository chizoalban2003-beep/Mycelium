#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DOMAIN="${MYCELIUM_RAILWAY_DOMAIN:-}"
PACKAGE_ID="${MYCELIUM_TWA_PACKAGE_ID:-com.mycelium.nexus.alpha}"
MANIFEST_URL="${MYCELIUM_MANIFEST_URL:-}"
KEYSTORE_PATH="${MYCELIUM_KEYSTORE_PATH:-$ROOT_DIR/mycelium-release.jks}"
KEYSTORE_ALIAS="${MYCELIUM_KEYSTORE_ALIAS:-mycelium}"
OUT_DIR="${MYCELIUM_TWA_OUT_DIR:-$ROOT_DIR/twa}"
FINGERPRINT_OUT="${MYCELIUM_FINGERPRINT_OUT:-$ROOT_DIR/twa-keystore-fingerprint.txt}"
LOCAL_ENV_FILE="${MYCELIUM_TWA_LOCAL_ENV_FILE:-$ROOT_DIR/.env}"

if [[ ! -f "$ROOT_DIR/twa-manifest.json" ]]; then
  echo "Missing twa-manifest.json at $ROOT_DIR/twa-manifest.json" >&2
  exit 2
fi

MANIFEST_VERSION_NAME="$(python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('twa-manifest.json').read_text(encoding='utf-8'))
print(str(data.get('versionName', '')).strip())
PY
)"

MANIFEST_VERSION_CODE="$(python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('twa-manifest.json').read_text(encoding='utf-8'))
print(str(data.get('versionCode', '')).strip())
PY
)"

if [[ -z "$DOMAIN" ]]; then
  echo "Set MYCELIUM_RAILWAY_DOMAIN to your production HTTPS domain." >&2
  exit 2
fi

if [[ -z "$MANIFEST_URL" ]]; then
  MANIFEST_URL="https://$DOMAIN/static/manifest.webmanifest"
fi

if [[ ! -f "$KEYSTORE_PATH" ]]; then
  echo "Missing release keystore: $KEYSTORE_PATH" >&2
  echo "Generate one with:" >&2
  echo "  keytool -genkeypair -v -keystore mycelium-release.jks -alias mycelium -keyalg RSA -keysize 2048 -validity 10000" >&2
  exit 2
fi

if ! command -v bubblewrap >/dev/null 2>&1; then
  echo "bubblewrap is not installed. Install it with: npm install -g @bubblewrap/cli" >&2
  exit 2
fi

if ! command -v keytool >/dev/null 2>&1; then
  echo "keytool is not installed or not on PATH." >&2
  exit 2
fi

echo "== Mycelium TWA release build =="
echo "Domain:      $DOMAIN"
echo "Package ID:  $PACKAGE_ID"
echo "Manifest:    $MANIFEST_URL"
echo "Version:     ${MANIFEST_VERSION_NAME:-unknown} (${MANIFEST_VERSION_CODE:-unknown})"
echo "Keystore:    $KEYSTORE_PATH"
echo "Output dir:  $OUT_DIR"

echo
echo "== Keystore SHA-256 fingerprint =="
FINGERPRINT="$(keytool -list -v -keystore "$KEYSTORE_PATH" -alias "$KEYSTORE_ALIAS" | awk '/SHA256:/{print $2; exit}' || true)"
if [[ -z "$FINGERPRINT" ]]; then
  echo "Unable to extract SHA-256 fingerprint from keystore." >&2
  exit 2
fi
printf '%s\n' "$FINGERPRINT" | tee "$FINGERPRINT_OUT"
echo "Fingerprint written to: $FINGERPRINT_OUT"

if [[ -f "$LOCAL_ENV_FILE" ]]; then
  LOCAL_SYNC_VALUE="$(awk -F= '
    BEGIN { value = "" }
    /^[[:space:]]*(export[[:space:]]+)?HIVE_ANDROID_SHA256=/ {
      line = $0
      sub(/^[[:space:]]*(export[[:space:]]+)?HIVE_ANDROID_SHA256=[[:space:]]*/, "", line)
      gsub(/^[\"\047]|[\"\047]$/, "", line)
      value = line
      print value
      exit
    }
  ' "$LOCAL_ENV_FILE" || true)"
  if [[ -z "$LOCAL_SYNC_VALUE" ]]; then
    echo "Local env file found at $LOCAL_ENV_FILE but HIVE_ANDROID_SHA256 is missing." >&2
    echo "Add the latest keystore fingerprint to that file or export the variable before building." >&2
    exit 2
  fi
  if [[ "$LOCAL_SYNC_VALUE" != "$FINGERPRINT" ]]; then
    echo "Local env file fingerprint does not match the keystore fingerprint." >&2
    echo "  keystore:   $FINGERPRINT" >&2
    echo "  local env:  $LOCAL_SYNC_VALUE" >&2
    echo "Sync Railway's HIVE_ANDROID_SHA256 before continuing." >&2
    exit 2
  fi
  echo "Local env fingerprint matches the keystore fingerprint."
fi

echo
if [[ -t 0 ]]; then
  read -r -p "Is this value synced to Railway's HIVE_ANDROID_SHA256? [y/N] " PRECHECK_CONFIRM
else
  PRECHECK_CONFIRM="${MYCELIUM_TWA_PRECHECK_CONFIRM:-}"
fi

case "${PRECHECK_CONFIRM:-}" in
  y|Y|yes|YES|true|TRUE|1)
    ;;
  *)
    echo "Preflight aborted: confirm the SHA-256 is synced to Railway and rerun the build." >&2
    exit 2
    ;;
esac

echo
echo "== Bubblewrap init =="
if [[ ! -d "$OUT_DIR" ]]; then
  echo "Bubblewrap project directory not found at $OUT_DIR" >&2
  echo "Run this once to scaffold it:" >&2
  echo "  bubblewrap init --manifest \"$MANIFEST_URL\"" >&2
  echo "Then rerun this script." >&2
  exit 2
fi

echo
echo "== Bubblewrap build =="
pushd "$OUT_DIR" >/dev/null
bubblewrap build
popd >/dev/null

echo
echo "Build complete. Check $OUT_DIR for the generated APK/AAB artifacts."
echo "Update Railway env vars with the SHA-256 fingerprint in $FINGERPRINT_OUT before shipping."
echo ""
echo "Railway env vars to set:"
echo "  ANDROID_APP_PACKAGE_NAME=$PACKAGE_ID"
echo "  HIVE_ANDROID_SHA256=$(cat \"$FINGERPRINT_OUT\")"
echo "  ANDROID_APP_SHA256_CERT_FINGERPRINTS_CSV=$(cat \"$FINGERPRINT_OUT\")"
