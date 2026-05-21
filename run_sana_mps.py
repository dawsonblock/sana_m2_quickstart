#!/usr/bin/env python3
"""Run Sana text-to-image locally on Apple Silicon."""

import argparse
import os

# Must be set before torch is imported indirectly by the shared engine.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt_positional",
        nargs="?",
        help="Optional positional prompt text",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text. If omitted, positional prompt or default is used.",
    )
    parser.add_argument(
        "--model",
        default="Efficient-Large-Model/Sana_600M_512px_diffusers",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negative-prompt",
        default="",
        help=(
            "Optional negative prompt, when supported by this "
            "pipeline version."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--no-attention-slicing",
        action="store_true",
        help=(
            "Disable attention slicing for compatibility with "
            "earlier CLI behavior."
        ),
    )
    parser.add_argument("--output", default="sana_m2_output.png")
    return parser.parse_args()


def main() -> None:
    from sana_core.engine import generate_image
    from sana_core.schemas import GenerationRequest

    args = parse_args()
    prompt = args.prompt or args.prompt_positional or (
        "a cyberpunk cat with a neon sign that says Sana"
    )
    result = generate_image(
        GenerationRequest(
            prompt=prompt,
            negative_prompt=args.negative_prompt or None,
            model=args.model,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
            dtype=args.dtype,
            attention_slicing=not args.no_attention_slicing,
            output=args.output,
        )
    )
    print(f"Saved {result.image_path}")
    print(f"Saved {result.metadata_path}")


if __name__ == "__main__":
    main()
