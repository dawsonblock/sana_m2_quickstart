#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Sana Mac M2 Smoke Check ==="
echo ""

echo "[1/4] Bash syntax"
bash -n setup.sh
bash -n launch.sh
bash -n smoke_check.sh

echo "[2/4] Python compile"
python3 -m py_compile \
  verify_mps.py \
  run_sana_mps.py \
  benchmark_sana_m2.py \
  app_m2.py \
  sana_core/__init__.py \
  sana_core/engine.py \
  sana_core/metadata.py \
  sana_core/paths.py \
  sana_core/schemas.py

echo "[3/4] Root dependency guard (no CUDA-only packages)"
if grep -R -n "flash-attn\|xformers\|bitsandbytes\|cu128\|cuda-toolkit\|triton\|mmcv" \
  requirements-macos-mps.txt setup.sh launch.sh run_sana_mps.py app_m2.py benchmark_sana_m2.py; then
  echo ""
  echo "FAIL: Found forbidden CUDA/NVIDIA dependency references in root wrapper files."
  exit 1
fi

echo "[4/4] Wrapper structure guard"
if [[ ! -f "Sana-main/DO_NOT_RUN_ON_MAC_M2.md" ]]; then
  echo ""
  echo "FAIL: Missing Sana-main/DO_NOT_RUN_ON_MAC_M2.md warning file."
  exit 1
fi

find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -name "*.pyc" -delete

echo ""
echo "PASS: Smoke check completed successfully."
echo "Next on a real Mac M2: ./setup.sh && ./launch.sh verify && ./launch.sh generate \"test prompt\""
