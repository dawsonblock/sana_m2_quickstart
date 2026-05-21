#!/usr/bin/env python3
"""Benchmark Sana 600M on Mac M2 with fixed baseline step counts."""

import inspect
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from diffusers import SanaPipeline, __version__ as diffusers_version


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
NEGATIVE_PROMPT = ""
LOG_DIR = Path("logs")


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
    except (OSError, ValueError) as first_error:
        # Fallback if model repo does not expose a separate fp16 variant
        load_kwargs.pop("variant", None)
        try:
            return SanaPipeline.from_pretrained(model_id, **load_kwargs)
        except (OSError, ValueError) as second_error:
            raise RuntimeError(
                "Failed to load Sana model for benchmark. "
                "Check model id, internet access, and Hugging Face auth (HF_TOKEN)."
            ) from second_error


def supports_negative_prompt(pipe: SanaPipeline) -> bool:
    """Return whether this pipeline accepts negative_prompt at call time."""
    try:
        call_params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return False
    return "negative_prompt" in call_params


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


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
    pipe_kwargs = {
        "prompt": PROMPT,
        "height": test["height"],
        "width": test["width"],
        "guidance_scale": 4.5,
        "num_inference_steps": test["steps"],
        "generator": torch.Generator(device="cpu").manual_seed(123),
    }
    if NEGATIVE_PROMPT and supports_negative_prompt(pipe):
        pipe_kwargs["negative_prompt"] = NEGATIVE_PROMPT

    result = pipe(
        **pipe_kwargs,
    )
    elapsed = time.perf_counter() - start

    out = f"benchmark_{test['height']}x{test['width']}_{test['steps']}steps.png"
    if not hasattr(result, "images") or not result.images:
        raise RuntimeError("Benchmark pipeline returned no image output")
    image = result.images[0]
    image.save(out)

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT or None,
        "model": test["model"],
        "height": test["height"],
        "width": test["width"],
        "steps": test["steps"],
        "guidance": 4.5,
        "seed": 123,
        "dtype": "float16" if device == "mps" else "float32",
        "device": device,
        "runtime_seconds": round(elapsed, 4),
        "torch_version": torch.__version__,
        "diffusers_version": diffusers_version,
        "output": out,
    }

    metadata_path = Path(out).with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    append_jsonl(LOG_DIR / "benchmarks.jsonl", record)
    print(json.dumps(record))


def main():
    print("=== Sana M2 Benchmark ===")
    for test in TESTS:
        run_test(test)
    print("\n=== Benchmark complete ===")


if __name__ == "__main__":
    main()
