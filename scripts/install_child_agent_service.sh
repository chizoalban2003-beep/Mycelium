#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo ".venv not found. Create it with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements/base.txt" >&2
  exit 1
fi

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl not found; this installer targets systemd user sessions." >&2
  exit 1
fi

REPO_ROOT="$(pwd)"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_FILE="${UNIT_DIR}/mycelium-child-agent.service"
ENV_DIR="${HOME}/.config/mycelium"
ENV_FILE="${ENV_DIR}/child-agent.env"

mkdir -p "$UNIT_DIR" "$ENV_DIR"

cat > "$UNIT_FILE" <<EOF
[Unit]
Description=Mycelium Child Agent
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${REPO_ROOT}
ExecStart=${REPO_ROOT}/scripts/run_child.sh
Restart=always
RestartSec=5
EnvironmentFile=-${ENV_FILE}

[Install]
WantedBy=default.target
EOF

if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" <<'EOF'
# Edit these values before enabling the service.
PARENT_HUB_URL=http://127.0.0.1:8000
NEXUS_DEVICE_ID=child-1
CHILD_AUTO_START_TELEMETRY=true
CHILD_AUTO_CAPTURE_TRAJECTORIES=true
CHILD_TRAJECTORY_WINDOW_SIZE=3
CHILD_TRAJECTORY_COOLDOWN_SECONDS=600
CHILD_TRAJECTORY_MUST_INCLUDE_CSV=mycelium
# Provide one of the following auth modes:
# CHILD_BEARER_TOKEN=...
# CHILD_EMAIL=you@example.com
# CHILD_PASSWORD=...
# HIVE_INGEST_TOKEN=...
EOF
fi

systemctl --user daemon-reload
systemctl --user enable --now mycelium-child-agent.service

echo "Installed and started mycelium-child-agent.service"
echo "Unit file: ${UNIT_FILE}"
echo "Env file:  ${ENV_FILE}"
