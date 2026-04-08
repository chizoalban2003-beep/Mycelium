#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BACKUP_DIR="${1:-storage/backups}"
mkdir -p "$BACKUP_DIR"

DATABASE_URL="${DATABASE_URL:-sqlite:///storage/mycelium.db}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ "$DATABASE_URL" == sqlite:* ]]; then
  SQLITE_PATH="${DATABASE_URL#sqlite:///}"
  if [[ ! -f "$SQLITE_PATH" ]]; then
    echo "SQLite database not found: $SQLITE_PATH" >&2
    exit 1
  fi
  OUT_FILE="${BACKUP_DIR}/mycelium-${STAMP}.sqlite3"
  cp "$SQLITE_PATH" "$OUT_FILE"
  echo "$OUT_FILE"
  exit 0
fi

if [[ "$DATABASE_URL" == postgresql* ]]; then
  if ! command -v pg_dump >/dev/null 2>&1; then
    echo "pg_dump not found. Install PostgreSQL client tools first." >&2
    exit 1
  fi
  OUT_FILE="${BACKUP_DIR}/mycelium-${STAMP}.dump"
  pg_dump "$DATABASE_URL" -Fc -f "$OUT_FILE"
  echo "$OUT_FILE"
  exit 0
fi

echo "Unsupported DATABASE_URL scheme: $DATABASE_URL" >&2
exit 1
