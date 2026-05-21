#!/usr/bin/env python3
"""Generate multiple Sana images and compose a contact-sheet grid."""

import argparse
import json
import os
from pathlib import Path
from typing import List

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def parse_seeds(raw_seeds: str) -> List[int]:
    tokens = [token.strip() for token in raw_seeds.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--seeds must include at least one integer seed.")

    parsed: List[int] = []
    for token in tokens:
        try:
            seed = int(token)
        except ValueError as error:
            raise ValueError(f"Invalid seed value: {token}") from error
        if seed < 0:
            raise ValueError("Seeds must be non-negative integers.")
        parsed.append(seed)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt text used for all images in the grid.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Optional negative prompt.",
    )
    parser.add_argument(
        "--model",
        default="Efficient-Large-Model/Sana_600M_512px_diffusers",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument("--no-attention-slicing", action="store_true")
    parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated seed list, e.g. 1,2,3,4",
    )
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--output", default="outputs/grid.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    from sana_core.grid import generate_grid
    from sana_core.schemas import GenerationRequest

    base_request = GenerationRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        model=args.model,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=seeds[0],
        dtype=args.dtype,
        attention_slicing=not args.no_attention_slicing,
    )

    grid_metadata = generate_grid(
        base_request=base_request,
        seeds=seeds,
        columns=args.columns,
        output_name=args.output,
    )

    grid_path = Path(grid_metadata["grid_path"])
    metadata_path = grid_path.with_suffix(".json")
    print(f"Saved {grid_path}")
    print(f"Saved {metadata_path}")
    print(json.dumps(grid_metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
