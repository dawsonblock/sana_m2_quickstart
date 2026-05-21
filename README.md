# Sana Mac M2 Quickstart

A clean, Mac-first launcher for generating images with Sana using Diffusers + PyTorch MPS.

This wrapper is built for Apple Silicon inference and intentionally avoids CUDA-only setup paths.

## Why this repo

- Apple Silicon focused (M1/M2/M3/M4)
- One-command setup and launch flow
- CLI + Gradio UI + benchmark included
- Upstream Sana kept as reference, not default runtime path

## Quick Start

```bash
chmod +x setup.sh launch.sh smoke_check.sh
./setup.sh
./launch.sh verify
./launch.sh generate "a small robot building a glowing circuit board"
./launch.sh ui
```

## Commands

```bash
./launch.sh --help
./launch.sh verify
./launch.sh generate "your prompt"
./launch.sh generate --prompt "your prompt" --steps 8
./launch.sh generate "your prompt" --negative-prompt "blurry, low quality"
./launch.sh ui
./launch.sh benchmark
```

Each generation now writes a PNG plus a same-name JSON metadata sidecar.
Example: `sana_m2_output.png` and `sana_m2_output.json`.

## Recommended baseline

- Model: Efficient-Large-Model/Sana_600M_512px_diffusers
- Resolution: 512x512
- Steps: 8 for smoke tests, 20 for higher quality
- Dtype: float16 on MPS

If output is unstable, use float32 for numerical stability:

```bash
./launch.sh generate "your prompt" --steps 8 --dtype float32
```

Negative prompts are passed only when supported by the installed
`SanaPipeline` version. If unsupported, generation continues and prints
a warning.

## Important safety note

Do not run upstream CUDA installer commands on Mac:

```bash
cd Sana-main
bash environment_setup.sh
pip install -e .
```

Use root-level scripts instead.

## Validation before release

Run:

```bash
./smoke_check.sh
```

This checks shell syntax, Python compile health, and guards against CUDA-only root dependency drift.

## Docs

- Detailed Mac operator guide: README_M2.md
- Upstream warning file: Sana-main/DO_NOT_RUN_ON_MAC_M2.md

## License

- Wrapper scripts in this repository are licensed under MIT: `LICENSE`
- Upstream Sana code under `Sana-main/` follows its own upstream license: `Sana-main/LICENSE`
