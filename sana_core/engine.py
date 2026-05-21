import gc
import inspect
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import diffusers
import torch
from diffusers import SanaPipeline

from .metadata import append_jsonl, write_json
from .paths import LOG_DIR, ensure_dirs, safe_output_path
from .schemas import GenerationRequest, GenerationResult


_PIPE_CACHE: dict[Tuple[str, str, str], SanaPipeline] = {}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_from_string(dtype: str, device: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16" and device == "mps":
        return torch.float16
    return torch.float32


def clear_pipeline_cache() -> None:
    _PIPE_CACHE.clear()
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def load_sana_pipeline(model_id: str, dtype: torch.dtype):
    load_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
    }
    if dtype == torch.float16:
        load_kwargs["variant"] = "fp16"
    try:
        return SanaPipeline.from_pretrained(model_id, **load_kwargs)
    except (OSError, ValueError) as first_error:
        if "variant" in load_kwargs:
            load_kwargs.pop("variant", None)
            try:
                return SanaPipeline.from_pretrained(model_id, **load_kwargs)
            except (OSError, ValueError) as second_error:
                raise RuntimeError(
                    "Failed to load Sana model after fp16 fallback. "
                    "Check model id, internet access, and Hugging Face auth "
                    "(HF_TOKEN)."
                ) from second_error
        raise RuntimeError(
            "Failed to load Sana model. "
            "Check model id, internet access, and Hugging Face auth "
            "(HF_TOKEN)."
        ) from first_error


def get_pipeline(model_id: str, dtype: torch.dtype, device: str):
    key = (model_id, str(dtype), device)
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]
    if _PIPE_CACHE:
        clear_pipeline_cache()
    pipe = load_sana_pipeline(model_id, dtype)
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    _PIPE_CACHE[key] = pipe
    return pipe


def supports_negative_prompt(pipe: SanaPipeline) -> bool:
    try:
        if not callable(pipe):
            return False
        call_params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return False
    return "negative_prompt" in call_params


def validate_request(req: GenerationRequest) -> None:
    if not req.prompt.strip():
        raise ValueError("Prompt must not be empty.")
    if req.width <= 0 or req.height <= 0:
        raise ValueError("Width and height must be positive.")
    if req.width % 32 != 0 or req.height % 32 != 0:
        raise ValueError("Width and height must be divisible by 32.")
    if req.steps <= 0:
        raise ValueError("Steps must be positive.")
    if req.guidance <= 0:
        raise ValueError("Guidance must be positive.")
    if req.seed < 0:
        raise ValueError("Seed must be non-negative.")
    if req.dtype not in {"float16", "float32"}:
        raise ValueError("dtype must be float16 or float32.")


def default_output_name(req: GenerationRequest) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return (
        f"outputs/sana_m2_{timestamp}_seed{req.seed}_"
        f"{req.width}x{req.height}.png"
    )


def _report_progress(
    progress: Optional[Callable[..., Any]],
    fraction: float,
    desc: str,
) -> None:
    if progress is None:
        return
    try:
        progress(fraction, desc=desc)
    except TypeError:
        progress(fraction)


def generate_image(
    req: GenerationRequest,
    progress: Optional[Callable[..., Any]] = None,
) -> GenerationResult:
    ensure_dirs()
    validate_request(req)

    device = get_device()
    dtype = dtype_from_string(req.dtype, device)
    output_name = req.output or default_output_name(req)
    output_path = safe_output_path(output_name)
    metadata_path = output_path.with_suffix(".json")

    _report_progress(progress, 0.02, f"Loading {req.model}")
    pipe = get_pipeline(req.model, dtype, device)

    if req.attention_slicing:
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
    elif hasattr(pipe, "disable_attention_slicing"):
        pipe.disable_attention_slicing()

    generator = torch.Generator(device="cpu").manual_seed(req.seed)
    pipe_kwargs = {
        "prompt": req.prompt,
        "height": req.height,
        "width": req.width,
        "guidance_scale": req.guidance,
        "num_inference_steps": req.steps,
        "generator": generator,
    }
    negative_prompt_used = False
    if req.negative_prompt:
        if supports_negative_prompt(pipe):
            pipe_kwargs["negative_prompt"] = req.negative_prompt
            negative_prompt_used = True
        else:
            print(
                "Warning: negative_prompt is not supported by this "
                "SanaPipeline version."
            )

    _report_progress(progress, 0.35, "Generating image")
    started = time.perf_counter()
    result = pipe(**pipe_kwargs)

    if not hasattr(result, "images") or not result.images:
        raise RuntimeError(
            "Pipeline returned no images. Try --dtype float32 or reduce "
            "resolution."
        )

    image = result.images[0]
    image.save(output_path)
    runtime_seconds = round(time.perf_counter() - started, 4)

    metadata = {
        "prompt": req.prompt,
        "negative_prompt": (
            req.negative_prompt if negative_prompt_used else None
        ),
        "negative_prompt_requested": req.negative_prompt,
        "negative_prompt_used": negative_prompt_used,
        "model": req.model,
        "width": req.width,
        "height": req.height,
        "steps": req.steps,
        "guidance": req.guidance,
        "seed": req.seed,
        "dtype": req.dtype,
        "attention_slicing": req.attention_slicing,
        "device": device,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": runtime_seconds,
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "image_path": str(output_path.relative_to(Path.cwd())),
        "metadata_path": str(metadata_path.relative_to(Path.cwd())),
    }

    write_json(metadata_path, metadata)
    append_jsonl(LOG_DIR / "generations.jsonl", metadata)
    _report_progress(progress, 1.0, "Done")

    return GenerationResult(
        image_path=str(output_path),
        metadata_path=str(metadata_path),
        metadata=metadata,
    )
