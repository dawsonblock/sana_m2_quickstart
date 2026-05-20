#!/usr/bin/env bash
# setup.sh — One-click environment setup for Sana on Apple Silicon (M2/M3/M4)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment (.venv) ..."
  python3.11 -m venv .venv
else
  echo "Virtual environment already exists — skipping creation."
fi

# shellcheck source=/dev/null
source .venv/bin/activate

# ── Dependencies ──────────────────────────────────────────────────────────────

echo "Upgrading pip ..."
pip install -q -U pip setuptools wheel

echo "Installing core dependencies (requirements-macos-mps.txt) ..."
pip install -q -r requirements-macos-mps.txt

echo "Installing Gradio (web UI) ..."
pip install -q gradio

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
