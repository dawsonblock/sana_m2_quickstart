# Sana for Mac (Apple Silicon)

Clean, local text-to-image generation on macOS using Sana through Diffusers and PyTorch MPS.

This project is a Mac-first wrapper around upstream Sana code. It is designed for Apple Silicon inference, not CUDA workflows.

## Quick start

```bash
./setup.sh
./launch.sh verify
./launch.sh generate "a small robot building a glowing circuit board"
./launch.sh ui
./launch.sh benchmark
```

## Requirements

- macOS 13 or newer (Sonoma/Sequoia recommended)
- Apple Silicon (M1, M2, M3, M4)
- Xcode Command Line Tools: `xcode-select --install`
- Python 3.11: `brew install python@3.11`

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
	app_m2.py
	benchmark_sana_m2.py
	verify_mps.py
	requirements-macos-mps.txt
	requirements-ui.txt
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

See root LICENSE and `Sana-main/LICENSE`.
