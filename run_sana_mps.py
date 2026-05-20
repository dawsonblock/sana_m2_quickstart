#!/usr/bin/env python3
"""Run Sana text-to-image locally on Apple Silicon.

Default model is the smaller Sana 0.6B 512px Diffusers checkpoint because it is
more realistic on Mac M2 than the repo's CUDA-first native pipeline.
"""

import argparse
import os

# Must be set before importing torch so unsupported MPS operations
# can fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from diffusers import SanaPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt", default="a cyberpunk cat with a neon sign that says Sana"
    )
    parser.add_argument(
        "--model", default="Efficient-Large-Model/Sana_600M_512px_diffusers"
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype", choices=["float16", "float32"], default="float16"
    )
    parser.add_argument("--output", default="sana_m2_output.png")
    parser.add_argument("--no-attention-slicing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive integers")
    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError("--height and --width must be divisible by 32")
    if args.steps <= 0:
        raise ValueError("--steps must be greater than 0")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if args.dtype == "float16" and device == "cpu":
        print("float16 requested on CPU; falling back to float32.")
    dtype = torch.float32
    if args.dtype == "float16" and device == "mps":
        dtype = torch.float16

    print(f"Loading {args.model}")
    print(f"Device={device}, dtype={dtype}")

    pipe = SanaPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(device)

    if (
        not args.no_attention_slicing
        and hasattr(pipe, "enable_attention_slicing")
    ):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=generator,
    )

    if not hasattr(result, "images") or not result.images:
        raise RuntimeError(
            "Pipeline returned no images. "
            "Try --dtype float32 or reduce resolution."
        )
    image = result.images[0]
    image.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
