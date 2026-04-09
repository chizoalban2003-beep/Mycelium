#!/usr/bin/env bash
set -euo pipefail

# Myco — One-Line Installer
# Usage: curl -sSL https://raw.githubusercontent.com/chizoalban2003-beep/Mycelium/main/scripts/install.sh | bash

echo ""
echo "  🌱 Myco — Grow with Data"
echo "  Your digital companion that grows with you"
echo ""

# Check Python
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "ERROR: Python 3 not found. Install Python 3.11+ first."
  exit 1
fi

PY_VERSION=$($PY --version 2>&1)
echo "  Using: $PY_VERSION"

# Clone or update
INSTALL_DIR="${MYCO_DIR:-$HOME/myco}"
if [[ -d "$INSTALL_DIR/.git" ]]; then
  echo "  Updating existing installation at $INSTALL_DIR..."
  cd "$INSTALL_DIR"
  git pull --ff-only origin main || true
else
  echo "  Installing to $INSTALL_DIR..."
  git clone https://github.com/chizoalban2003-beep/Mycelium.git "$INSTALL_DIR"
  cd "$INSTALL_DIR"
fi

# Create venv and install deps
if [[ ! -d .venv ]]; then
  echo "  Creating virtual environment..."
  $PY -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip wheel setuptools -q
pip install -r requirements/base.txt -q

# Create .env if needed
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "  Created .env from template"
fi

# Create storage dir
mkdir -p storage

echo ""
echo "  ✅ Myco installed successfully!"
echo ""
echo "  To start your companion:"
echo "    cd $INSTALL_DIR"
echo "    source .venv/bin/activate"
echo "    uvicorn mycelium_app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "  Then open: http://localhost:8000"
echo ""
echo "  To install as a background service (Linux):"
echo "    bash scripts/install_child_agent_service.sh"
echo ""
echo "  🌱 Grow with Data"
echo ""
