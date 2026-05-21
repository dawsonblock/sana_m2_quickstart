#!/usr/bin/env python3
"""Benchmark Sana 600M on Mac M2 with various resolutions and step counts."""

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
        "steps": 8,
    },
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


def load_sana_pipeline(model_id: str, dtype: torch.dtype):
    """Load Sana pipeline with fallback for missing fp16 variant."""
    load_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if dtype == torch.float16:
        load_kwargs["variant"] = "fp16"
    try:
        return SanaPipeline.from_pretrained(model_id, **load_kwargs)
    except OSError:
        # Fallback if model repo does not expose a separate fp16 variant
        load_kwargs.pop("variant", None)
        return SanaPipeline.from_pretrained(model_id, **load_kwargs)


def run_test(test):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    
    print(f"\nLoading {test['model']} on {device}...")
    pipe = load_sana_pipeline(test["model"], dtype).to(device)

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
    print("=== Sana M2 Benchmark ===")
    for test in TESTS:
        run_test(test)
    print("\n=== Benchmark complete ===")


if __name__ == "__main__":
    main()
