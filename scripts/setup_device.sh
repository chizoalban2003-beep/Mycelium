#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  Myco — Install on Your Device
#
#  Run this ONE command on your Linux laptop:
#    bash <(curl -sS https://raw.githubusercontent.com/chizoalban2003-beep/Mycelium/main/scripts/setup_device.sh)
#
#  Or if you already cloned:
#    cd ~/myco && bash scripts/setup_device.sh
# ============================================================

echo ""
echo "  🌱 Myco — Setting up your digital companion"
echo ""

# Detect if we're in the repo or need to clone
if [[ -f "mycelium_app/main.py" ]]; then
  MYCO_DIR="$(pwd)"
  echo "  Found existing Myco at $MYCO_DIR"
else
  MYCO_DIR="${MYCO_DIR:-$HOME/myco}"
  if [[ -d "$MYCO_DIR/.git" ]]; then
    echo "  Updating $MYCO_DIR..."
    cd "$MYCO_DIR"
    git pull --ff-only origin main 2>/dev/null || true
  else
    echo "  Cloning to $MYCO_DIR..."
    git clone https://github.com/chizoalban2003-beep/Mycelium.git "$MYCO_DIR"
    cd "$MYCO_DIR"
  fi
fi

cd "$MYCO_DIR"

# Python check
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "  ERROR: Python 3 not found. Install Python 3.11+ first."
  exit 1
fi
echo "  Python: $($PY --version)"

# Create venv
if [[ ! -d .venv ]]; then
  echo "  Creating virtual environment..."
  $PY -m venv .venv
fi
source .venv/bin/activate

# Install dependencies
echo "  Installing dependencies..."
pip install --upgrade pip wheel setuptools -q
pip install -r requirements/base.txt -q

# Create fresh .env
if [[ ! -f .env ]]; then
  cp .env.example .env
  # Generate a random secret key
  SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
  sed -i "s/SECRET_KEY=change-me-please/SECRET_KEY=$SECRET/" .env
  echo "  Created .env with secure secret key"
fi

# Ensure storage directory
mkdir -p storage

# Install xdotool for window focus tracking (Linux)
if command -v apt-get >/dev/null 2>&1 && ! command -v xdotool >/dev/null 2>&1; then
  echo "  Installing xdotool for window focus tracking..."
  sudo apt-get install -y xdotool -q 2>/dev/null || echo "  (optional: install xdotool manually for window tracking)"
fi

echo ""
echo "  ✅ Myco installed at $MYCO_DIR"
echo ""
echo "  To start your companion:"
echo ""
echo "    cd $MYCO_DIR"
echo "    source .venv/bin/activate"
echo "    uvicorn mycelium_app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "  Then open: http://localhost:8000"
echo ""
echo "  ─────────────────────────────────────────"
echo "  For background operation (runs on boot):"
echo ""
echo "    # Terminal 1: Web server"
echo "    uvicorn mycelium_app.main:app --host 0.0.0.0 --port 8000 &"
echo ""
echo "    # Terminal 2: Signal collector (independent)"
echo "    python3 scripts/collector_standalone.py &"
echo ""
echo "    # Terminal 3: Learning engine (independent)"
echo "    python3 scripts/learner_standalone.py &"
echo ""
echo "  Or install as systemd service:"
echo "    bash scripts/install_myco_service.sh"
echo ""
echo "  🌱 Grow with Data"
echo ""
