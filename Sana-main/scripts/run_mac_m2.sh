#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
python run_sana_m2.py
