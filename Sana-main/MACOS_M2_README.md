# Sana on Mac M2

This is the isolated Apple Silicon path for running Sana text-to-image locally with Hugging Face Diffusers and PyTorch MPS.

Do not run `environment_setup.sh` or install the full project with `pip install -e .` for this path. Those paths pull in CUDA-oriented packages such as xformers, flash-attn, triton, and bitsandbytes.

## Setup

```bash
xcode-select --install
cd /Users/dawsonblock/Downloads/sana_m2_quickstart/Sana-main
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install torch torchvision torchaudio
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    print(torch.ones(1, device="mps"))
else:
    raise SystemExit("MPS is not available. Check Python architecture, macOS version, and PyTorch install.")
PY
pip install -r requirements-macos-m2.txt
```

## Run

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python run_sana_m2.py
```

Or use the launcher:

```bash
./scripts/run_mac_m2.sh
```

The minimum success condition is a generated `sana_m2_output.png` from the 600M 512px Diffusers checkpoint.

## Benchmark

```bash
python benchmark_sana_m2.py
```

This writes `benchmark_512x512_12steps.png` and `benchmark_512x512_20steps.png` with timing output.

## Local UI

Install Gradio only after command-line generation works:

```bash
pip install gradio
python app_m2.py
```

Then open:

```text
http://127.0.0.1:7860
```

The UI includes:

- Generate tab with prompt, model, resolution, steps, guidance, precision, seed, saved PNG downloads, and runtime status.
- Error Settings tab with Safe Recovery, Balanced Default, and Try 1024px presets.
- Suggestions tab with reusable prompt ideas and a random prompt picker.
- A single-image queue to keep MPS memory pressure predictable.

## Troubleshooting

If MPS is unavailable, verify that Python is native ARM:

```bash
python -c "import platform; print(platform.machine())"
```

Expected output is `arm64`. If it says `x86_64`, install or select a native ARM Python.

If output is grey, black, or unstable, change the pipeline dtype in `run_sana_m2.py` from `torch.float16` to `torch.float32` and retest.

If memory pressure is high, use 512px output, reduce `num_inference_steps` to `8` or `12`, and generate one image at a time.
