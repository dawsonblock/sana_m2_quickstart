#!/usr/bin/env python3
"""Run Sana text-to-image locally on Apple Silicon.

Default model is the smaller Sana 0.6B 512px Diffusers checkpoint because it is
more realistic on Mac M2 than the repo's CUDA-first native pipeline.
"""

import argparse
import inspect
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Must be set before importing torch so unsupported MPS operations
# can fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from diffusers import SanaPipeline, __version__ as diffusers_version  # noqa: E402


def load_sana_pipeline(model_id: str, dtype: torch.dtype):
    """Load Sana pipeline with fallback for missing fp16 variant.
    
    Tries to load with variant="fp16" for float16 dtype.
    Falls back to loading without variant if the repo does not expose it.
    """
    load_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if dtype == torch.float16:
        load_kwargs["variant"] = "fp16"
    try:
        return SanaPipeline.from_pretrained(model_id, **load_kwargs)
    except (OSError, ValueError) as first_error:
        # Fallback if model repo does not expose a separate fp16 variant
        if "variant" in load_kwargs:
            print("Warning: fp16 variant not available, loading without variant")
            load_kwargs.pop("variant", None)
            try:
                return SanaPipeline.from_pretrained(model_id, **load_kwargs)
            except (OSError, ValueError) as second_error:
                raise RuntimeError(
                    "Failed to load Sana model after fp16 fallback. "
                    "Check model id, internet access, and Hugging Face auth (HF_TOKEN)."
                ) from second_error
        raise RuntimeError(
            "Failed to load Sana model. "
            "Check model id, internet access, and Hugging Face auth (HF_TOKEN)."
        ) from first_error


def supports_negative_prompt(pipe: SanaPipeline) -> bool:
    """Return whether this pipeline accepts negative_prompt at call time."""
    try:
        call_params = inspect.signature(pipe.__class__.__call__).parameters
    except (TypeError, ValueError):
        return False
    return "negative_prompt" in call_params


def resolve_output_path(raw_output: str) -> Path:
    """Resolve a safe output path under the current workspace."""
    output_path = Path(raw_output)
    if output_path.is_absolute():
        raise ValueError("--output must be a relative path inside the project directory")
    if ".." in output_path.parts:
        raise ValueError("--output cannot use parent-directory traversal")
    resolved = (Path.cwd() / output_path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


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
        "--model", default="Efficient-Large-Model/Sana_600M_512px_diffusers"
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Optional negative prompt. Passed only when supported by this SanaPipeline version.",
    )
    parser.add_argument(
        "--dtype", choices=["float16", "float32"], default="float16"
    )
    parser.add_argument("--output", default="sana_m2_output.png")
    parser.add_argument("--no-attention-slicing", action="store_true")
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    prompt = args.prompt or args.prompt_positional or (
        "a cyberpunk cat with a neon sign that says Sana"
    )
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

    pipe = load_sana_pipeline(args.model, dtype)
    pipe = pipe.to(device)

    if (
        not args.no_attention_slicing
        and hasattr(pipe, "enable_attention_slicing")
    ):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    pipe_kwargs = {
        "prompt": prompt,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance,
        "num_inference_steps": args.steps,
        "generator": generator,
    }
    if args.negative_prompt:
        if supports_negative_prompt(pipe):
            pipe_kwargs["negative_prompt"] = args.negative_prompt
        else:
            print("Warning: negative_prompt is not supported by this SanaPipeline version.")

    result = pipe(
        **pipe_kwargs,
    )

    if not hasattr(result, "images") or not result.images:
        raise RuntimeError(
            "Pipeline returned no images. "
            "Try --dtype float32 or reduce resolution."
        )

    output_path = resolve_output_path(args.output)
    image = result.images[0]
    image.save(output_path)

    elapsed = time.perf_counter() - started
    dtype_name = "float16" if dtype == torch.float16 else "float32"
    metadata = {
        "prompt": prompt,
        "negative_prompt": args.negative_prompt or None,
        "model": args.model,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "dtype": dtype_name,
        "device": device,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(elapsed, 4),
        "torch_version": torch.__version__,
        "diffusers_version": diffusers_version,
    }
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved {output_path}")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
