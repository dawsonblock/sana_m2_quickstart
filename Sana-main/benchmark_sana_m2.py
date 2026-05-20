import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from diffusers import SanaPipeline


TESTS = [
    {
        "model": "Efficient-Large-Model/Sana_600M_512px_diffusers",
        "height": 512,
        "width": 512,
        "steps": 12,
    },
    {
        "model": "Efficient-Large-Model/Sana_600M_512px_diffusers",
        "height": 512,
        "width": 512,
        "steps": 20,
    },
]

PROMPT = "a compact sci-fi generator on a lab bench, clean lighting, technical detail"


def run_test(test):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    pipe = SanaPipeline.from_pretrained(
        test["model"],
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    ).to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    start = time.perf_counter()
    image = pipe(
        prompt=PROMPT,
        height=test["height"],
        width=test["width"],
        guidance_scale=4.5,
        num_inference_steps=test["steps"],
        generator=torch.Generator(device="cpu").manual_seed(123),
    ).images[0]
    elapsed = time.perf_counter() - start

    out = f"benchmark_{test['height']}x{test['width']}_{test['steps']}steps.png"
    image.save(out)
    print(
        {
            "model": test["model"],
            "resolution": f"{test['height']}x{test['width']}",
            "steps": test["steps"],
            "device": device,
            "seconds": round(elapsed, 2),
            "output": out,
        }
    )


def main():
    for test in TESTS:
        run_test(test)


if __name__ == "__main__":
    main()
