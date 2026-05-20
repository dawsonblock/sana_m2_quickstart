import platform
import torch

print("Python arch:", platform.machine())
print("Torch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    x = torch.ones(1, device="mps")
    print("MPS test tensor:", x)
else:
    print("MPS is not available. Check that Python is arm64, macOS is current, and PyTorch is installed correctly.")
