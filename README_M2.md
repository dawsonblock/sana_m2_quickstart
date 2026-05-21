# Sana on Apple Silicon (M2 / M3 / M4)

Local text-to-image generation on macOS using [Sana](https://github.com/NVlabs/Sana) via Hugging Face Diffusers and PyTorch MPS (Metal). No CUDA, no NVIDIA GPU required.

## What this is

This is a **Mac Apple Silicon launcher** for Sana image generation using:
- PyTorch **MPS** (Metal Performance Shaders)
- Hugging Face **Diffusers**
- **SanaPipeline** from Diffusers

It does **not** use the upstream CUDA installer.

---

## Prerequisites

| Requirement | Notes |
| --- | --- |
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

---

## Usage

### Verify MPS

```bash
./launch.sh verify
```

Output should show `MPS available: True` and a successful test tensor.

### Generate an image (CLI)

```bash
./launch.sh generate "a small robot building a glowing circuit board"
```

Saves to `sana_m2_output.png` by default.

### Common CLI options

```bash
# Quick test (faster, lower memory)
./launch.sh generate "sunset over mountains" --steps 8

# Float32 (more stable, potentially slower, may use more memory)
./launch.sh generate "test prompt" --dtype float32 --steps 8 --output test_fp32.png

# Higher resolution (requires more memory)
./launch.sh generate "prompt" --model Efficient-Large-Model/Sana_600M_1024px_diffusers --height 1024 --width 1024

# Custom seed
./launch.sh generate "prompt" --seed 12345
```

### Full CLI reference

```bash
./launch.sh generate --help
```

### Launch the web UI

```bash
./launch.sh ui
```

Opens at `http://127.0.0.1:7860`. UI automatically installs Gradio on first run if needed.

### Benchmark performance

```bash
./launch.sh benchmark
```

Runs several test configurations (512×512 at 8, 12, 20 steps) and reports timing.

---

## Recommended first model

**Sana 600M 512px** (default)

```text
Efficient-Large-Model/Sana_600M_512px_diffusers
```

Works reliably on M2/M3 with 20 steps, float16, 512×512.

---

## Try later (after 512px is stable)

- `Efficient-Large-Model/Sana_600M_1024px_diffusers` — Requires more memory
- `Efficient-Large-Model/Sana_1600M_512px_diffusers` — Larger model; slow on M2

---

## Not recommended on basic M2

- SANA-Video
- LongSANA
- SANA-WM
- 4-bit CUDA quantization
- Native training
- Batch generation

---

## Do not run

```bash
cd Sana-main
bash environment_setup.sh
pip install -e .
```

These commands install CUDA/NVIDIA packages that fail on Apple Silicon. See [Sana-main/DO_NOT_RUN_ON_MAC_M2.md](./Sana-main/DO_NOT_RUN_ON_MAC_M2.md) for details.

---

## Troubleshooting

### MPS not available

- Verify Python is native arm64: `python3 -c 'import platform; print(platform.machine())'` should print `arm64`
- Ensure macOS 13+
- Reinstall PyTorch: `pip install --upgrade torch torchvision torchaudio`

### Output is black or grey

- Use Safe Recovery preset in the UI tab "Error Settings"
- Or from CLI: `./launch.sh generate "prompt" --steps 8 --dtype float32`
- Reduce resolution to 512×512

### Mac is slow or swapping

- Reduce steps: `--steps 8`
- Use float32 for stability: `--dtype float32` (may use more memory)
- Reduce resolution: `--height 512 --width 512`
- Generate one image at a time (MPS works best with serial execution)

### Gradio UI won't load

- Verify Gradio is installed: `python -c 'import gradio; print(gradio.__version__)'`
- Reinstall: `pip install -r requirements-ui.txt`

---

## Output

Images are saved as:
- CLI: `sana_m2_output.png` (default) or custom filename via `--output`
- UI: `outputs/sana_m2_<timestamp>_seed<N>_<WxH>.png`

---

## FAQ

**Q: Why not the upstream Sana repo directly?**  
A: The upstream repo is CUDA/NVIDIA-first. Diffusers provides a MPS-compatible pipeline that's more reliable on Mac.

**Q: Can I use 1600M or larger models?**  
A: Possibly, but they run slowly on M2. Start with 600M to verify stability.

**Q: Can I train models?**  
A: No. This is inference only. Training requires CUDA.

**Q: Can I use the Dockerfile?**  
A: The included Dockerfile is for CUDA/GPU environments. It won't work on Mac M2.

**Q: Why float16 by default?**  
A: It's faster and uses less memory on MPS. If output is poor, use `--dtype float32`.

---

## Advanced: Accessing Sana-main

The `Sana-main/` directory contains the original upstream repository for reference. You can explore:
- Model configurations: `Sana-main/configs/`
- Training scripts: `Sana-main/train_scripts/`
- LoRA/DreamBooth examples: `Sana-main/docs/sana_lora_dreambooth.md`

But do not run any scripts from that directory on Mac M2 without explicit porting.

---

## License

See LICENSE and Sana-main/LICENSE for details.
