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
# Only use root .venv, never reuse Sana-main/.venv

if [ -d ".venv" ]; then
  if [ -x ".venv/bin/python" ] && ".venv/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "Using existing virtual environment: .venv"
  else
    echo "Existing .venv appears broken; recreating it ..."
    rm -rf .venv
    echo "Creating virtual environment (.venv) ..."
    python3.11 -m venv .venv
  fi
else
  echo "Creating virtual environment (.venv) ..."
  python3.11 -m venv .venv
fi

source ".venv/bin/activate"

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
