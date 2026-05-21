#!/usr/bin/env python3
"""Benchmark Sana 600M on Mac M2 with fixed baseline step counts."""

import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


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

PROMPT = (
    "a compact sci-fi generator on a lab bench, clean lighting, "
    "technical detail"
)
NEGATIVE_PROMPT = ""
LOG_DIR = Path("logs")


def run_test(test):
    from sana_core.engine import generate_image
    from sana_core.metadata import append_jsonl
    from sana_core.schemas import GenerationRequest

    out = (
        f"benchmark_{test['height']}x{test['width']}_"
        f"{test['steps']}steps.png"
    )
    started = time.perf_counter()
    result = generate_image(
        GenerationRequest(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT or None,
            model=test["model"],
            height=test["height"],
            width=test["width"],
            steps=test["steps"],
            guidance=4.5,
            seed=123,
            dtype="float16",
            output=out,
        )
    )

    record = dict(result.metadata)
    record.update(
        {
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "output": out,
        }
    )
    append_jsonl(LOG_DIR / "benchmarks.jsonl", record)
    print(record)


def main():
    print("=== Sana M2 Benchmark ===")
    for test in TESTS:
        run_test(test)
    print("\n=== Benchmark complete ===")


if __name__ == "__main__":
    main()
