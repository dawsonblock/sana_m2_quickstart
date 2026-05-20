# Do not run this upstream installer on Mac M2

## ⚠️ This directory contains CUDA/NVIDIA-oriented code

Do **not** run:

```bash
cd Sana-main
bash environment_setup.sh
pip install -e .
```

This upstream repository is designed for CUDA/NVIDIA GPUs and may install packages that **fail or cause conflicts on Apple Silicon**:

- `xformers` — CUDA kernel library, incompatible with MPS
- `flash-attn` — NVIDIA CUDA extension, incompatible with MPS
- `bitsandbytes` — CUDA quantization library, incompatible with MPS
- CUDA PyTorch wheels — Wrong architecture for Mac M2
- CUDA toolkit — Not available on macOS

## ✅ Use the root launcher instead

From the project root (`cd ..`), use:

```bash
./setup.sh
./launch.sh verify
./launch.sh generate "your prompt"
./launch.sh ui
./launch.sh benchmark
```

These commands use:
- **PyTorch MPS** (Metal Performance Shaders)
- **Hugging Face Diffusers**
- No CUDA, no NVIDIA dependencies

## Reference only

This directory is kept as a reference for:
- Original Sana model configuration files
- Upstream documentation
- Training scripts (not for inference on Mac)

Do not modify or run any scripts in this directory unless you understand the CUDA dependencies.
