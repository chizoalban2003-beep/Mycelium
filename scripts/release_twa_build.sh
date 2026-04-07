#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXIT_USAGE=2
EXIT_MISSING_MANIFEST=10
EXIT_MISSING_KEYSTORE=11
EXIT_MISSING_TOOLS=12
EXIT_FINGERPRINT_EXTRACT=13
EXIT_SYNC_CONFIRMATION=14
EXIT_FINGERPRINT_MISMATCH=15
EXIT_BUILD_FAILURE=16

CI_MODE=false
AUTO_CONFIRM="${MYCELIUM_TWA_PRECHECK_CONFIRM:-}"

read_env_value() {
  local key="$1"
  local env_file="$2"
  if [[ -f "$env_file" ]]; then
    awk -v key="$key" '
      BEGIN { prefix = "(^[[:space:]]*(export[[:space:]]+)?" key "[[:space:]]*=)" }
      $0 ~ prefix {
        line = $0
        sub(prefix, "", line)
        gsub(/^[\"\047]|[\"\047]$/, "", line)
        print line
        exit
      }
    ' "$env_file"
  fi
}

usage() {
  cat <<'EOF'
Usage: scripts/release_twa_build.sh [--ci]

Options:
  --ci    Run without interactive prompts. Requires CONFIRMED_RAILWAY_SYNC=true.
  -h, --help  Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ci)
      CI_MODE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit "$EXIT_USAGE"
      ;;
  esac
done

DOMAIN="${MYCELIUM_RAILWAY_DOMAIN:-}"
PACKAGE_ID="${MYCELIUM_TWA_PACKAGE_ID:-com.mycelium.nexus.alpha}"
MANIFEST_URL="${MYCELIUM_MANIFEST_URL:-}"
KEYSTORE_PATH="${MYCELIUM_KEYSTORE_PATH:-$ROOT_DIR/mycelium-release.jks}"
KEYSTORE_ALIAS="${MYCELIUM_KEYSTORE_ALIAS:-mycelium}"
OUT_DIR="${MYCELIUM_TWA_OUT_DIR:-$ROOT_DIR/twa}"
FINGERPRINT_OUT="${MYCELIUM_FINGERPRINT_OUT:-$ROOT_DIR/twa-keystore-fingerprint.txt}"
LOCAL_ENV_FILE="${MYCELIUM_TWA_LOCAL_ENV_FILE:-$ROOT_DIR/.env}"
VERSION_BUMP_FILE="${MYCELIUM_TWA_VERSION_BUMP_FILE:-$ROOT_DIR/version_bump.txt}"

if [[ ! -f "$ROOT_DIR/twa-manifest.json" ]]; then
  echo "Missing twa-manifest.json at $ROOT_DIR/twa-manifest.json" >&2
  exit "$EXIT_MISSING_MANIFEST"
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
  exit "$EXIT_USAGE"
fi

if [[ -z "$MANIFEST_URL" ]]; then
  MANIFEST_URL="https://$DOMAIN/static/manifest.webmanifest"
fi

if [[ ! -f "$KEYSTORE_PATH" ]]; then
  echo "Missing release keystore: $KEYSTORE_PATH" >&2
  echo "Generate one with:" >&2
  echo "  keytool -genkeypair -v -keystore mycelium-release.jks -alias mycelium -keyalg RSA -keysize 2048 -validity 10000" >&2
  exit "$EXIT_MISSING_KEYSTORE"
fi

if ! command -v bubblewrap >/dev/null 2>&1; then
  echo "bubblewrap is not installed. Install it with: npm install -g @bubblewrap/cli" >&2
  exit "$EXIT_MISSING_TOOLS"
fi

if ! command -v keytool >/dev/null 2>&1; then
  echo "keytool is not installed or not on PATH." >&2
  exit "$EXIT_MISSING_TOOLS"
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
  exit "$EXIT_FINGERPRINT_EXTRACT"
fi
printf '%s\n' "$FINGERPRINT" | tee "$FINGERPRINT_OUT"
echo "Fingerprint written to: $FINGERPRINT_OUT"

SYNC_CONFIRMED=false

SYNC_FLAG_VALUE="$(read_env_value CONFIRMED_RAILWAY_SYNC "$LOCAL_ENV_FILE" || true)"
if [[ -z "$SYNC_FLAG_VALUE" ]]; then
  SYNC_FLAG_VALUE="${CONFIRMED_RAILWAY_SYNC:-}"
fi
LOCAL_SYNC_VALUE="$(read_env_value HIVE_ANDROID_SHA256 "$LOCAL_ENV_FILE" || true)"
if [[ -z "$LOCAL_SYNC_VALUE" ]]; then
  LOCAL_SYNC_VALUE="${HIVE_ANDROID_SHA256:-}"
fi

if [[ -n "$LOCAL_SYNC_VALUE" && "$LOCAL_SYNC_VALUE" != "$FINGERPRINT" ]]; then
  echo "Fingerprint does not match the keystore fingerprint." >&2
  echo "  keystore:   $FINGERPRINT" >&2
  echo "  synced:     $LOCAL_SYNC_VALUE" >&2
  echo "Sync Railway's HIVE_ANDROID_SHA256 before continuing." >&2
  exit "$EXIT_FINGERPRINT_MISMATCH"
fi

if [[ -n "$LOCAL_SYNC_VALUE" ]]; then
  echo "Found matching HIVE_ANDROID_SHA256 fingerprint in local config or environment."
fi

if [[ "$SYNC_FLAG_VALUE" =~ ^(1|true|TRUE|yes|YES|y|Y)$ ]]; then
  SYNC_CONFIRMED=true
fi

if [[ "$SYNC_CONFIRMED" != true ]]; then
  if [[ "$CI_MODE" == true ]]; then
    if [[ "$SYNC_CONFIRMED" != true ]]; then
      echo "CI preflight aborted: set CONFIRMED_RAILWAY_SYNC=true after syncing Railway's HIVE_ANDROID_SHA256." >&2
      exit "$EXIT_SYNC_CONFIRMATION"
    fi
  else
    echo
    if [[ -t 0 ]]; then
      read -r -p "Is this value synced to Railway's HIVE_ANDROID_SHA256? [y/N] " AUTO_CONFIRM
    fi
    case "${AUTO_CONFIRM:-}" in
      y|Y|yes|YES|true|TRUE|1)
        SYNC_CONFIRMED=true
        ;;
      *)
        echo "Preflight aborted: confirm the SHA-256 is synced to Railway and rerun the build." >&2
        exit "$EXIT_SYNC_CONFIRMATION"
        ;;
    esac
  fi
fi

echo
echo "== Bubblewrap init =="
if [[ ! -d "$OUT_DIR" ]]; then
  echo "Bubblewrap project directory not found at $OUT_DIR" >&2
  echo "Run this once to scaffold it:" >&2
  echo "  bubblewrap init --manifest \"$MANIFEST_URL\"" >&2
  echo "Then rerun this script." >&2
  exit "$EXIT_USAGE"
fi

echo
echo "== Bubblewrap build =="
pushd "$OUT_DIR" >/dev/null
if bubblewrap build; then
  :
else
  build_status=$?
  popd >/dev/null
  echo "Bubblewrap build failed with exit code $build_status." >&2
  exit "$EXIT_BUILD_FAILURE"
fi
popd >/dev/null

echo
echo "Build complete. Check $OUT_DIR for the generated APK/AAB artifacts."
echo "Update Railway env vars with the SHA-256 fingerprint in $FINGERPRINT_OUT before shipping."
echo ""
echo "Railway env vars to set:"
echo "  ANDROID_APP_PACKAGE_NAME=$PACKAGE_ID"
echo "  HIVE_ANDROID_SHA256=$(cat \"$FINGERPRINT_OUT\")"
echo "  ANDROID_APP_SHA256_CERT_FINGERPRINTS_CSV=$(cat \"$FINGERPRINT_OUT\")"

previous_build_count=0
if [[ -f "$VERSION_BUMP_FILE" ]]; then
  previous_build_count="$(awk -F= '/^[[:space:]]*build_count=/ { print $2; exit }' "$VERSION_BUMP_FILE" 2>/dev/null || echo 0)"
fi
if ! [[ "$previous_build_count" =~ ^[0-9]+$ ]]; then
  previous_build_count=0
fi
build_count=$((previous_build_count + 1))
cat > "$VERSION_BUMP_FILE" <<EOF
build_count=$build_count
version_name=${MANIFEST_VERSION_NAME:-unknown}
version_code=${MANIFEST_VERSION_CODE:-unknown}
fingerprint=$FINGERPRINT
last_build_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
echo "Version bump tracker written to: $VERSION_BUMP_FILE"
