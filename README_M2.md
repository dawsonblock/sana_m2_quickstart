# Sana on Apple Silicon (M2 / M3 / M4)

Local text-to-image generation on macOS using [Sana](https://github.com/NVlabs/Sana) via Hugging Face Diffusers and PyTorch MPS (Metal). No CUDA, no NVIDIA GPU required.

## Quick Start

```bash
./setup.sh                                          # one-time setup
./launch.sh generate "a small robot on a workbench" # generate an image
./launch.sh ui                                       # open the web UI
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| macOS 13 Ventura or later | Sonoma/Sequoia recommended |
| Apple Silicon (M1/M2/M3/M4) | Intel Macs not supported |
| Xcode Command Line Tools | `xcode-select --install` |
| Python 3.11 | `brew install python@3.11` |

---

## Setup

Run once. Creates a `.venv`, installs all dependencies, and verifies Metal/MPS is working:

```bash
./setup.sh
```

If `Sana-main/.venv` already exists, setup reuses it automatically.

---

## Usage

### Generate an image (CLI)

```bash
./launch.sh generate "your prompt here"
```

Common options:

```bash
# Quick test (fastest)
./launch.sh generate "sunset over mountains" --steps 8

# High-quality 512px (default)
./launch.sh generate "a futuristic city" --steps 20

# 1024px — needs 16 GB+ unified memory
./launch.sh generate "macro photo of a circuit board" \
  --model Efficient-Large-Model/Sana_600M_1024px_diffusers \
  --height 1024 --width 1024

# Save to a specific file
./launch.sh generate "cozy reading room" --output my_image.png
```

Run `./launch.sh` with no arguments to see all options.
Unknown commands now fail fast with a clear error and non-zero exit code.

### Web UI

```bash
./launch.sh ui
```

Opens a Gradio interface at **http://127.0.0.1:7860** with model selection, resolution controls, prompt library, and recovery presets.

### Other commands

```bash
./launch.sh verify     # confirm MPS / Metal is active
./launch.sh benchmark  # run the M2 performance benchmark
```

---

## Models

| Model | Resolution | Speed | Memory | Notes |
|---|---|---|---|---|
| `Sana_600M_512px_diffusers` | 512 × 512 | Fast | ~6 GB | **Default.** Best starting point |
| `Sana_600M_1024px_diffusers` | 1024 × 1024 | Medium | ~10 GB | Sharper detail |
| `Sana_1600M_512px_diffusers` | 512 × 512 | Slow | ~12 GB | Most detailed at 512px |

All models are downloaded automatically from Hugging Face on first use (~1–4 GB each).

---

## Troubleshooting

**Grey or black output**
```bash
./launch.sh generate "your prompt" --dtype float32
```
fp32 is slower but avoids fp16 artifacts on some Mac configurations.

**Out of memory / kernel panics**
- Use 512 × 512 resolution
- Reduce steps to 8–12
- Generate one image at a time
- Quit other memory-heavy applications

**MPS not detected**
```bash
./launch.sh verify
```
Ensure you are on Apple Silicon (`python3 -c "import platform; print(platform.machine())"` should print `arm64`).

**Slow first run**
The model downloads from Hugging Face and is cached in `~/.cache/huggingface`. Subsequent runs use the cache and start much faster.

---

## What to avoid

Do **not** run `Sana-main/environment_setup.sh` or `pip install -e .` on Mac. These install CUDA wheels (xformers, flash-attn, triton, bitsandbytes) that are Linux/NVIDIA-only and will fail or silently break the environment.
