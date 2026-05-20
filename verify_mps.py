import platform
import sys
import torch

print("Python arch:", platform.machine())
print("Torch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    x = torch.ones(1, device="mps")
    print("MPS test tensor:", x)
else:
    print("ERROR: MPS is not available.")
    print(
        "  - Verify Python is native arm64: "
        "python3 -c 'import platform; print(platform.machine())'"
    )
    print("  - Ensure macOS 13+ and Apple Silicon")
    print("  - Reinstall PyTorch: pip install torch torchvision torchaudio")
    sys.exit(1)
