# Sana on Mac M2

This folder is a minimal Apple Silicon path for the uploaded Sana repo.
It intentionally avoids the repo's CUDA-only dependencies and runs Sana through Hugging Face Diffusers on PyTorch MPS.

## Setup

```bash
xcode-select --install
cd ~/Downloads/Sana-main
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements-macos-mps.txt
python verify_mps.py
```

## Run

```bash
python run_sana_mps.py --prompt "a small robot building a glowing circuit board" --height 512 --width 512 --steps 20
```

For a stronger Mac, try:

```bash
python run_sana_mps.py --model Efficient-Large-Model/Sana_600M_1024px_diffusers --height 1024 --width 1024 --steps 20
```

Avoid the repository's `environment_setup.sh` on Mac. It installs CUDA, xformers CUDA wheels, flash-attn, triton, and other NVIDIA/Linux-oriented packages.
