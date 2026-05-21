# Sana for Mac (Apple Silicon)

Clean, local text-to-image generation on macOS using Sana through Diffusers and PyTorch MPS.

This project is a Mac-first wrapper around upstream Sana code. It is designed for Apple Silicon inference, not CUDA workflows.

## Quick start

```bash
chmod +x setup.sh launch.sh smoke_check.sh
./setup.sh
./launch.sh verify
./launch.sh generate "a small robot building a glowing circuit board"
./launch.sh grid --prompt "moon base interior" --seeds 1,2,3,4 --columns 2 --output outputs/moon_grid.png
./launch.sh api
./launch.sh phone
./launch.sh ui
./launch.sh benchmark
```

## Requirements

- macOS 13 or newer (Sonoma/Sequoia recommended)
- Apple Silicon (M1, M2, M3, M4)
- Xcode Command Line Tools: `xcode-select --install`
- Python 3.11 recommended: `brew install python@3.11`
- Python 3.12 is also supported by `setup.sh` as a fallback

## What this wrapper does

- Uses PyTorch MPS for Apple GPU acceleration
- Uses Diffusers SanaPipeline for inference
- Keeps upstream Sana files as reference only
- Avoids CUDA and NVIDIA-only package paths in the root launcher flow

## Safety: what not to run

Do not run upstream CUDA setup on Mac:

```bash
cd Sana-main
bash environment_setup.sh
pip install -e .
```

Why: those commands may pull CUDA-oriented dependencies that do not work on Apple Silicon.

See `Sana-main/DO_NOT_RUN_ON_MAC_M2.md` for details.

## Command reference

### Verify MPS

```bash
./launch.sh verify
```

Expected: MPS built and available are true on a real Apple Silicon runtime.

### Generate (positional prompt)

```bash
./launch.sh generate "a compact sci-fi generator on a lab bench"
```

### Generate (explicit prompt flag)

```bash
./launch.sh generate --prompt "moon base interior" --steps 8
```

### Generate with negative prompt (optional)

```bash
./launch.sh generate "moon base interior" \
  --negative-prompt "blurry, noisy, low quality"
```

If the current `SanaPipeline` build does not support `negative_prompt`,
the wrapper logs a warning and continues generation without failing.

### Common options

```bash
# Fast smoke run
./launch.sh generate "sunset over mountains" --steps 8

# Stability-first mode
./launch.sh generate "test prompt" --dtype float32 --steps 8 --output test_fp32.png

# Higher resolution (optional, more memory)
./launch.sh generate "prompt" \
  --model Efficient-Large-Model/Sana_600M_1024px_diffusers \
  --height 1024 --width 1024 --steps 12
```

### Launch UI

```bash
./launch.sh ui
```

UI auto-installs Gradio if missing and serves at `http://127.0.0.1:7860`.

The UI now includes a `Library` tab that lets you:

- browse and apply shared prompt presets
- save the current prompt/settings as a reusable preset
- inspect recent output metadata sidecars
- preview recent generated images inside the app

### Launch API

```bash
./launch.sh api
```

The API serves at `http://127.0.0.1:7861` and exposes `/health`, `/generate`,
`/generate/batch`, `/generate/grid`, `/outputs`, `/metadata`, `/presets`, and
`/gallery`.

The `/gallery` route serves a lightweight browser for recent outputs and links
directly to image and metadata files.

### Launch Phone Mode

```bash
./launch.sh phone
```

Phone mode starts the API on `0.0.0.0:7861`, prints your Mac LAN URL, and
shows a terminal QR code when `qrcode[pil]` is installed.

Use phone mode safely:

1. Keep the Mac and phone on the same Wi-Fi network.
2. Open the printed URL (or scan the QR code) from your phone.
3. Use the token from `SANA_PHONE_TOKEN` or the temporary startup token.
4. Do not port-forward the server.

Safe Recovery defaults in phone UI:

1. `512x512`
2. `8` steps
3. Force FP32
4. `Efficient-Large-Model/Sana_600M_512px_diffusers`

### Generate Grid

```bash
./launch.sh grid \
  --prompt "moon base interior" \
  --seeds 1,2,3,4 \
  --steps 12 \
  --columns 2 \
  --output outputs/moon_grid.png
```

This generates one image per seed plus a contact-sheet PNG and JSON metadata.

## Generation metadata

Every generated image writes a same-name JSON sidecar for reproducibility.

Examples:

- `sana_m2_output.png` and `sana_m2_output.json` (CLI)
- `outputs/sana_m2_*.png` and `outputs/sana_m2_*.json` (UI)

Metadata includes prompt, model, dimensions, steps, guidance, seed,
dtype/device, runtime seconds, and Torch/Diffusers versions.

### Run benchmark

```bash
./launch.sh benchmark
```

Default benchmark set:

- 512x512 at 8 steps
- 512x512 at 12 steps
- 512x512 at 20 steps

## Recommended defaults

- Model: `Efficient-Large-Model/Sana_600M_512px_diffusers`
- Resolution: 512x512
- Steps: 20 for quality, 8 for quick checks
- Dtype: float16 on MPS

If output looks unstable, try float32 for numerical stability.

## Troubleshooting

### Black or gray output

- Reduce to 512x512
- Use fewer steps (`--steps 8`)
- Use float32 for stability (`--dtype float32`)

### Slow run or swap pressure

- Keep one generation at a time
- Use 512x512
- Use fewer steps
- Close memory-heavy apps

Note: float32 is usually slower and may use more memory than float16.

### MPS unavailable

- Confirm arm64 Python: `python3 -c 'import platform; print(platform.machine())'`
- Reinstall PyTorch packages in this environment
- Ensure you are on Apple Silicon with a supported macOS version

### Phone cannot connect

- Confirm both devices are on the same Wi-Fi network.
- Check macOS firewall settings.
- Open the Mac LAN IP URL manually if QR does not work.
- Confirm server startup prints `--host 0.0.0.0` in phone mode.
- Try Safari or Chrome on phone.

## One-command static validation

Run local smoke checks before packaging:

```bash
./smoke_check.sh
```

This validates:

- shell syntax
- Python compile checks
- root dependency guard against CUDA-only packages
- upstream warning file presence

## Project layout

```text
sana_m2_quickstart/
  setup.sh
  launch.sh
  smoke_check.sh
  run_sana_mps.py
  run_sana_grid.py
  api_server.py
  app_m2.py
  benchmark_sana_m2.py
  verify_mps.py
  requirements-macos-mps.txt
  requirements-ui.txt
  requirements-api.txt
  sana_core/
    __init__.py
    engine.py
    gallery.py
    grid.py
    metadata.py
    paths.py
    presets.py
    schemas.py
  presets/
    prompt_presets.json
  static/
    gallery.html
    gallery.css
    gallery.js
  Sana-main/
    DO_NOT_RUN_ON_MAC_M2.md
    ...upstream files
```

## Scope and non-goals

Supported baseline:

- Mac inference via Diffusers + MPS
- CLI, UI, and benchmark paths

Not baseline-supported in this wrapper:

- Native Sana training
- SANA-Video / LongSANA / SANA-WM workflows
- CUDA-specific optimizations (flash-attn, xformers, bitsandbytes)

## License

- Wrapper scripts in this repository are licensed under MIT: `LICENSE`
- Upstream Sana code under `Sana-main/` follows its own upstream license: `Sana-main/LICENSE`
