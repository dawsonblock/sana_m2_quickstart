#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Sana Mac M2 Smoke Check ==="
echo ""

echo "[1/5] Bash syntax"
bash -n setup.sh
bash -n launch.sh
bash -n smoke_check.sh

echo "[2/5] Python compile"
python3 -m py_compile \
  verify_mps.py \
  run_sana_mps.py \
  run_sana_grid.py \
  api_server.py \
  benchmark_sana_m2.py \
  app_m2.py \
  tests/test_api_server.py \
  tests/test_gallery_presets.py \
  sana_core/__init__.py \
  sana_core/engine.py \
  sana_core/gallery.py \
  sana_core/grid.py \
  sana_core/metadata.py \
  sana_core/paths.py \
  sana_core/presets.py \
  sana_core/schemas.py

if [ -f "sana_core/network.py" ]; then
  python3 -m py_compile sana_core/network.py
fi

echo "[3/5] Python tests"
python3 -m unittest discover -s tests -p 'test_*.py'

echo "[4/5] Root dependency guard (no CUDA-only packages)"
if grep -R -n "flash-attn\|xformers\|bitsandbytes\|cu128\|cuda-toolkit\|triton\|mmcv" \
  requirements-macos-mps.txt \
  requirements-ui.txt \
  requirements-api.txt \
  setup.sh \
  launch.sh \
  run_sana_mps.py \
  run_sana_grid.py \
  api_server.py \
  app_m2.py \
  benchmark_sana_m2.py \
  sana_core/*.py; then
  echo ""
  echo "FAIL: Found forbidden CUDA/NVIDIA dependency references in root wrapper files."
  exit 1
fi

if grep -R -n "Sana-main/.venv" \
  run_sana_mps.py \
  run_sana_grid.py \
  api_server.py \
  app_m2.py \
  benchmark_sana_m2.py \
  sana_core/*.py; then
  echo ""
  echo "FAIL: Root wrapper files must not reference Sana-main/.venv."
  exit 1
fi

echo "[5/5] Wrapper structure guard"
if [[ ! -f "Sana-main/DO_NOT_RUN_ON_MAC_M2.md" ]]; then
  echo ""
  echo "FAIL: Missing Sana-main/DO_NOT_RUN_ON_MAC_M2.md warning file."
  exit 1
fi

if grep -R -n "VENV_PATH=.*Sana-main/.venv\|source .*Sana-main/.venv" launch.sh setup.sh; then
  echo ""
  echo "FAIL: Root launcher must never reuse Sana-main/.venv."
  exit 1
fi

find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -name "*.pyc" -delete

echo ""
echo "PASS: Smoke check completed successfully."
echo "Next on a real Mac M2: ./setup.sh && ./launch.sh verify && ./launch.sh generate \"test prompt\""
