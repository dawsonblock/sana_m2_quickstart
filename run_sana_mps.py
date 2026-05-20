#!/usr/bin/env python3
"""Run Sana text-to-image locally on Apple Silicon through Diffusers + PyTorch MPS.

Default model is the smaller Sana 0.6B 512px Diffusers checkpoint because it is
more realistic on Mac M2 than the repo's CUDA-first native pipeline.
"""

import argparse
import os

# Must be set before importing torch for unsupported MPS ops to fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from diffusers import SanaPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a cyberpunk cat with a neon sign that says Sana")
    parser.add_argument("--model", default="Efficient-Large-Model/Sana_600M_512px_diffusers")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--output", default="sana_m2_output.png")
    parser.add_argument("--no-attention-slicing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if args.dtype == "float16" and device == "mps" else torch.float32

    print(f"Loading {args.model}")
    print(f"Device={device}, dtype={dtype}")

    pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(device)

    if not args.no_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=generator,
    )

    image = result.images[0] if hasattr(result, "images") else result[0][0]
    image.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
