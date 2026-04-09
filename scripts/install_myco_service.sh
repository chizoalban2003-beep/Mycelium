#!/usr/bin/env bash
set -euo pipefail

# Install Myco as systemd user services (runs on login, stops on logout)
# Usage: bash scripts/install_myco_service.sh

MYCO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$MYCO_DIR/.venv/bin"

echo "🌱 Installing Myco systemd services from $MYCO_DIR"

mkdir -p ~/.config/systemd/user

# Service 1: Web server
cat > ~/.config/systemd/user/myco-web.service << EOF
[Unit]
Description=Myco Web Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$MYCO_DIR
ExecStart=$VENV/uvicorn mycelium_app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=PYTHONDONTWRITEBYTECODE=1

[Install]
WantedBy=default.target
EOF

# Service 2: Signal collector
cat > ~/.config/systemd/user/myco-collector.service << EOF
[Unit]
Description=Myco Signal Collector
After=myco-web.service

[Service]
Type=simple
WorkingDirectory=$MYCO_DIR
ExecStart=$VENV/python3 scripts/collector_standalone.py --interval 15
Restart=always
RestartSec=10
Environment=PYTHONDONTWRITEBYTECODE=1

[Install]
WantedBy=default.target
EOF

# Service 3: Learning engine
cat > ~/.config/systemd/user/myco-learner.service << EOF
[Unit]
Description=Myco Learning Engine
After=myco-web.service

[Service]
Type=simple
WorkingDirectory=$MYCO_DIR
ExecStart=$VENV/python3 scripts/learner_standalone.py --interval 120
Restart=always
RestartSec=30
Environment=PYTHONDONTWRITEBYTECODE=1

[Install]
WantedBy=default.target
EOF

# Enable and start
systemctl --user daemon-reload
systemctl --user enable myco-web.service myco-collector.service myco-learner.service
systemctl --user start myco-web.service
sleep 2
systemctl --user start myco-collector.service myco-learner.service

# Enable linger so services run even when not logged in via SSH
loginctl enable-linger "$(whoami)" 2>/dev/null || true

echo ""
echo "✅ Myco services installed and running"
echo ""
echo "  Check status:  systemctl --user status myco-web myco-collector myco-learner"
echo "  View logs:     journalctl --user -u myco-web -f"
echo "  Stop all:      systemctl --user stop myco-web myco-collector myco-learner"
echo "  Disable:       systemctl --user disable myco-web myco-collector myco-learner"
echo ""
echo "  Open: http://localhost:8000"
echo ""
