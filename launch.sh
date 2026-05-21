#!/usr/bin/env bash
# launch.sh — Unified launcher for Sana on Apple Silicon
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Handle help early, before venv check ──────────────────────────────────

CMD="${1:-help}"

if [[ "$CMD" == "help" || "$CMD" == "--help" || "$CMD" == "-h" ]]; then
  cat <<'EOF'
Usage: ./launch.sh <command> [options]

Commands:
  generate "your prompt"   Generate an image from a text prompt
  ui                       Launch the Gradio web UI (http://127.0.0.1:7860)
  verify                   Check MPS / Metal availability
  benchmark                Run the M2 performance benchmark

Generate options (all optional):
  --prompt TEXT            Text description of the image (or pass as first arg)
  --model MODEL_ID         HuggingFace model ID  (default: Sana_600M_512px_diffusers)
  --height N               Image height in pixels (default: 512)
  --width  N               Image width  in pixels (default: 512)
  --steps  N               Inference steps        (default: 20)
  --guidance FLOAT         Guidance scale         (default: 4.5)
  --negative-prompt TEXT   Optional negative prompt (when supported)
  --seed   N               Random seed            (default: 42)
  --dtype  float16|float32 Precision              (default: float16 on MPS)
  --output FILE            Output filename        (default: sana_m2_output.png)

Examples:
  ./launch.sh generate "a small robot building a glowing circuit board"
  ./launch.sh generate --steps 8 "moon base interior"
  ./launch.sh generate "moon base interior" --steps 8 --height 512 --width 512
  ./launch.sh generate "portrait" --model Efficient-Large-Model/Sana_600M_1024px_diffusers --height 1024 --width 1024
  ./launch.sh ui
EOF
  exit 0
fi

# ── Require setup to have been run ───────────────────────────────────────────

if [ ! -d ".venv" ]; then
  echo "ERROR: Virtual environment not found. Run ./setup.sh first."
  exit 1
fi

VENV_PATH=".venv"

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

# Required for unsupported MPS ops to fall back to CPU gracefully.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

# ── Dispatch ─────────────────────────────────────────────────────────────────

shift || true

case "$CMD" in
  generate|gen)
    python run_sana_mps.py "$@"
    ;;

  ui)
    if ! python -c "import gradio" >/dev/null 2>&1; then
      echo "Installing UI dependency: gradio"
      pip install -r requirements-ui.txt
    fi
    echo "Starting Gradio web UI at http://127.0.0.1:7860"
    echo "(Press Ctrl+C to stop)"
    python app_m2.py "$@"
    ;;

  verify)
    python verify_mps.py "$@"
    ;;

  benchmark)
    python benchmark_sana_m2.py "$@"
    ;;



  *)
    echo "ERROR: Unknown command '$CMD'"
    echo "Run './launch.sh --help' for usage."
    exit 2
    ;;
esac
