#!/usr/bin/env bash
# setup.sh — One-click environment setup for Sana on Apple Silicon (M2/M3/M4)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PATH=""

echo "=== Sana M2 Setup ==="
echo ""

# ── Prerequisites ─────────────────────────────────────────────────────────────

if ! xcode-select -p &>/dev/null; then
  echo "ERROR: Xcode Command Line Tools not installed."
  echo "       Run: xcode-select --install"
  echo "       Then re-run this script."
  exit 1
fi

if ! command -v python3.11 &>/dev/null; then
  echo "ERROR: Python 3.11 not found."
  echo "       Install via: brew install python@3.11"
  echo "       Then re-run this script."
  exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────

if [ -d ".venv" ]; then
  VENV_PATH=".venv"
  echo "Using existing virtual environment: $VENV_PATH"
elif [ -d "Sana-main/.venv" ]; then
  VENV_PATH="Sana-main/.venv"
  echo "Using existing virtual environment: $VENV_PATH"
else
  VENV_PATH=".venv"
  echo "Creating virtual environment ($VENV_PATH) ..."
  python3.11 -m venv "$VENV_PATH"
fi

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

# ── Dependencies ──────────────────────────────────────────────────────────────

echo "Upgrading pip ..."
python -m pip install -q -U pip wheel "setuptools<82"

echo "Installing core dependencies (requirements-macos-mps.txt) ..."
python -m pip install -r requirements-macos-mps.txt || { echo "ERROR: Dependency install failed. Check the output above."; exit 1; }

# ── Verify MPS ────────────────────────────────────────────────────────────────

echo ""
echo "Verifying MPS / Metal ..."
python verify_mps.py

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "=== Setup complete! ==="
echo ""
echo "  Generate an image (CLI):"
echo "    ./launch.sh generate \"a futuristic city at sunset\""
echo ""
echo "  Launch the web UI:"
echo "    ./launch.sh ui"
echo ""
echo "  Check MPS anytime:"
echo "    ./launch.sh verify"
echo ""
