#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <backup-file>" >&2
  exit 1
fi

BACKUP_FILE="$1"
if [[ ! -f "$BACKUP_FILE" ]]; then
  echo "Backup file not found: $BACKUP_FILE" >&2
  exit 1
fi

DATABASE_URL="${DATABASE_URL:-sqlite:///storage/mycelium.db}"

if [[ "$DATABASE_URL" == sqlite:* ]]; then
  SQLITE_PATH="${DATABASE_URL#sqlite:///}"
  mkdir -p "$(dirname "$SQLITE_PATH")"
  cp "$BACKUP_FILE" "$SQLITE_PATH"
  echo "$SQLITE_PATH"
  exit 0
fi

if [[ "$DATABASE_URL" == postgresql* ]]; then
  if ! command -v pg_restore >/dev/null 2>&1; then
    echo "pg_restore not found. Install PostgreSQL client tools first." >&2
    exit 1
  fi
  pg_restore --clean --if-exists --no-owner --dbname "$DATABASE_URL" "$BACKUP_FILE"
  echo "Restored into $DATABASE_URL"
  exit 0
fi

echo "Unsupported DATABASE_URL scheme: $DATABASE_URL" >&2
exit 1
