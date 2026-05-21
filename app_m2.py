import inspect
import os
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_sana_pipeline(model_id, dtype):
    from sana_core.engine import load_sana_pipeline as core_load_sana_pipeline

    return core_load_sana_pipeline(model_id, dtype)


def supports_negative_prompt(pipe) -> bool:
    try:
        if not callable(pipe):
            return False
        call_params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return False
    return "negative_prompt" in call_params


MODEL_OPTIONS = {
    "Sana 600M - 512px": "Efficient-Large-Model/Sana_600M_512px_diffusers",
    "Sana 600M - 1024px": "Efficient-Large-Model/Sana_600M_1024px_diffusers",
    "Sana 1600M - 512px": "Efficient-Large-Model/Sana_1600M_512px_diffusers",
}

RESOLUTION_OPTIONS = {
    "512 x 512": (512, 512),
    "768 x 768": (768, 768),
    "1024 x 1024": (1024, 1024),
}

DEFAULT_PROMPT = (
    "a clean futuristic workbench with a small robot assembling a glowing "
    "circuit board, sharp details"
)

OUTPUT_DIR = Path("outputs")
LOG_DIR = Path("logs")
PIPE_CACHE: dict = {}


def append_jsonl(path: Path, payload: dict) -> None:
    from sana_core.metadata import append_jsonl as core_append_jsonl

    core_append_jsonl(path, payload)


def clear_pipeline_cache():
    """Clear the model cache to free unified memory on Mac."""
    from sana_core.engine import clear_pipeline_cache as core_clear_pipeline_cache

    core_clear_pipeline_cache()


def get_device_and_dtype(precision_mode="Auto"):
    from sana_core.engine import dtype_from_string, get_device

    device = get_device()
    requested_dtype = (
        "float32" if precision_mode == "Force FP32" else "float16"
    )
    dtype = dtype_from_string(requested_dtype, device)
    return device, dtype


def load_pipeline(model_label, precision_mode, progress):
    from sana_core.engine import get_pipeline as core_get_pipeline

    model_id = MODEL_OPTIONS[model_label]
    device, dtype = get_device_and_dtype(precision_mode)
    if progress is not None:
        progress(0.02, desc="Loading shared engine")
    try:
        pipe = core_get_pipeline(model_id, dtype, device)
    except RuntimeError as error:
        raise gr.Error(str(error)) from error
    return pipe, device, dtype


def generate_image(
    prompt,
    negative_prompt,
    model_label,
    resolution_label,
    steps,
    guidance,
    seed,
    precision_mode,
    progress=None,
):
    from sana_core.engine import generate_image as core_generate_image
    from sana_core.schemas import GenerationRequest
    import torch

    if progress is None:
        progress = gr.Progress(track_tqdm=True)
    prompt = (prompt or "").strip()
    negative_prompt = (negative_prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt before generating.")

    width, height = RESOLUTION_OPTIONS[resolution_label]
    seed = int(seed)
    steps = int(steps)
    guidance = float(guidance)
    if height <= 0 or width <= 0:
        raise gr.Error("Height and width must be positive integers.")
    if height % 32 != 0 or width % 32 != 0:
        raise gr.Error("Height and width must be divisible by 32.")
    if steps <= 0:
        raise gr.Error("Steps must be greater than 0.")
    if guidance <= 0:
        raise gr.Error("Guidance must be greater than 0.")
    if seed < 0:
        raise gr.Error("Seed must be a non-negative integer.")

    started = time.perf_counter()
    try:
        _, device, dtype = load_pipeline(model_label, precision_mode, progress)
        req = GenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            model=MODEL_OPTIONS[model_label],
            height=height,
            width=width,
            steps=steps,
            guidance=guidance,
            seed=seed,
            dtype="float16" if dtype == torch.float16 else "float32",
        )
        result = core_generate_image(req, progress=progress)
        elapsed = time.perf_counter() - started
        out_path = Path(result.image_path)
        metadata_path = Path(result.metadata_path)
        metadata = dict(result.metadata)
        metadata["runtime_seconds"] = round(elapsed, 4)
        dtype_name = metadata.get("dtype", "float32")

        status = (
            f"Saved {out_path.name}\n"
            f"Metadata: {metadata_path.name}\n"
            f"Model: {model_label}\n"
            f"Resolution: {width}x{height}, steps: {steps}, guidance: "
            f"{guidance}\n"
            f"Seed: {seed}, device: {device}, dtype: {dtype_name}, "
            f"time: {elapsed:.1f}s"
        )
        return str(out_path), str(out_path), status
    except gr.Error as error:
        append_jsonl(
            LOG_DIR / "errors.jsonl",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "error": str(error),
                "model": MODEL_OPTIONS.get(model_label, model_label),
                "resolution": resolution_label,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
            },
        )
        raise
    except Exception as error:  # pragma: no cover - defensive UI path
        append_jsonl(
            LOG_DIR / "errors.jsonl",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "error": str(error),
                "traceback": traceback.format_exc(),
                "model": MODEL_OPTIONS.get(model_label, model_label),
                "resolution": resolution_label,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
            },
        )
        raise gr.Error(f"Generation failed: {error}") from error


def random_seed():
    return random.randint(0, 2**31 - 1)


def device_status():
    import torch

    device, dtype = get_device_and_dtype()
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    return (
        f"Device: {device}\n"
        f"Dtype: {dtype}\n"
        f"MPS built: {mps_built}\n"
        f"MPS available: {mps_available}"
    )


def safe_settings():
    return (
        "Sana 600M - 512px",
        "512 x 512",
        8,
        3.8,
        "Force FP32",
        "Applied Safe Recovery. Go to Generate and run with lower resolution, fewer steps, and FP32 for numerical stability.",
        gr.Tabs(selected="generate"),
    )


def balanced_settings():
    return (
        "Sana 600M - 512px",
        "512 x 512",
        20,
        4.5,
        "Auto",
        "Applied Balanced Default. Go to Generate and run the tested 512px MPS path.",
        gr.Tabs(selected="generate"),
    )


def quality_settings():
    return (
        "Sana 600M - 1024px",
        "1024 x 1024",
        20,
        4.5,
        "Auto",
        "Applied Try 1024px. Use this only if 512px is stable on this Mac.",
        gr.Tabs(selected="generate"),
    )


def apply_suggestion(prompt_text):
    return (
        prompt_text,
        "Prompt copied to Generate. Adjust settings there, then press Generate.",
        gr.Tabs(selected="generate"),
    )


def random_suggestion():
    suggestions = [
        "a compact sci-fi generator on a lab bench, clean lighting, technical detail",
        "a realistic product render of a translucent handheld AI device on a white seamless background",
        "a cozy modern reading room with warm lamps, soft textiles, and balanced natural light",
        "a cinematic moon base workshop at sunrise, functional architecture, crisp atmospheric detail",
        "a clean macro photograph of a tiny robot repairing a glowing circuit board, shallow depth of field",
        "a calm Japanese-inspired workspace with a laptop, sketchbook, and small bonsai, soft morning light",
    ]
    return random.choice(suggestions)


def suggestion_one():
    return "a compact sci-fi generator on a lab bench, clean lighting, technical detail"


def suggestion_two():
    return "a realistic product render of a translucent handheld AI device on a white seamless background"


def suggestion_three():
    return "a cozy modern reading room with warm lamps, soft textiles, and balanced natural light"


def suggestion_four():
    return "a cinematic moon base workshop at sunrise, functional architecture, crisp atmospheric detail"


def suggestion_five():
    return "a clean macro photograph of a tiny robot repairing a glowing circuit board, shallow depth of field"


def suggestion_six():
    return "a calm Japanese-inspired workspace with a laptop, sketchbook, and small bonsai, soft morning light"


CSS = """
.app-shell {
    max-width: 1180px;
    margin: 0 auto;
}
.title-row h1 {
    font-size: 28px;
    line-height: 1.15;
    margin-bottom: 4px;
}
.subtle {
    color: #5f6b7a;
    font-size: 14px;
}
.status-box textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 12px;
}
"""


with gr.Blocks(
    title="Sana on Mac M2",
    css=CSS,
    theme=gr.themes.Soft(),
) as demo:
    with gr.Column(elem_classes="app-shell"):
        with gr.Row(elem_classes="title-row"):
            gr.Markdown(
                "# Sana on Mac M2\n"
                "<span class='subtle'>Local text-to-image generation using Diffusers and Apple MPS.</span>"
            )

        with gr.Tabs(selected="generate") as tabs:
            with gr.Tab("Generate", id="generate"):
                with gr.Row():
                    with gr.Column(scale=5):
                        prompt = gr.Textbox(
                            label="Prompt",
                            value=DEFAULT_PROMPT,
                            lines=4,
                            max_lines=8,
                            placeholder="Describe the image you want to generate...",
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt (optional)",
                            value="",
                            lines=2,
                            max_lines=4,
                            placeholder="Describe what to avoid in the image...",
                        )

                        with gr.Row():
                            generate_btn = gr.Button("Generate", variant="primary")
                            clear_btn = gr.ClearButton(value="Clear", components=[prompt, negative_prompt])

                        with gr.Accordion("Settings", open=True):
                            model_label = gr.Dropdown(
                                label="Model",
                                choices=list(MODEL_OPTIONS.keys()),
                                value="Sana 600M - 512px",
                            )
                            resolution_label = gr.Dropdown(
                                label="Resolution",
                                choices=list(RESOLUTION_OPTIONS.keys()),
                                value="512 x 512",
                            )
                            steps = gr.Slider(4, 30, value=20, step=1, label="Steps")
                            guidance = gr.Slider(1.0, 8.0, value=4.5, step=0.1, label="Guidance")
                            precision_mode = gr.Radio(
                                label="Precision",
                                choices=["Auto", "Force FP32"],
                                value="Auto",
                            )
                            with gr.Row():
                                seed = gr.Number(label="Seed", value=42, precision=0)
                                seed_btn = gr.Button("Random Seed")

                        device_info = gr.Textbox(
                            label="Runtime",
                            value=device_status,
                            lines=4,
                            interactive=False,
                            elem_classes="status-box",
                        )

                    with gr.Column(scale=6):
                        output_image = gr.Image(label="Generated Image", type="filepath", height=620)
                        output_file = gr.File(label="Saved PNG")
                        status = gr.Textbox(
                            label="Generation Details",
                            lines=6,
                            interactive=False,
                            elem_classes="status-box",
                        )

            with gr.Tab("Error Settings", id="errors"):
                gr.Markdown(
                    "Use these presets when generation fails, output looks black or grey, "
                    "or the Mac starts swapping heavily."
                )
                with gr.Row():
                    safe_btn = gr.Button("Safe Recovery", variant="primary")
                    balanced_btn = gr.Button("Balanced Default")
                    quality_btn = gr.Button("Try 1024px")

                preset_status = gr.Textbox(
                    label="Preset Status",
                    value="No preset applied yet.",
                    lines=2,
                    interactive=False,
                    elem_classes="status-box",
                )

                with gr.Row():
                    with gr.Group():
                        gr.Markdown(
                            "**Safe Recovery**\n\n"
                            "512px, 8 steps, lower guidance, FP32. Use this for black/grey output, "
                            "NaN-like artifacts, or memory pressure."
                        )
                    with gr.Group():
                        gr.Markdown(
                            "**Balanced Default**\n\n"
                            "512px, 20 steps, automatic MPS fp16. This is the normal tested path."
                        )
                    with gr.Group():
                        gr.Markdown(
                            "**Try 1024px**\n\n"
                            "1024px, 20 steps, automatic MPS fp16. Use only after 512px is stable."
                        )

                error_notes = gr.Textbox(
                    label="Troubleshooting Notes",
                    value=(
                        "If MPS is unavailable, verify Python is native arm64.\n"
                        "If output is black or grey, use Safe Recovery.\n"
                        "If 1024px fails or the Mac slows down, return to 512 x 512.\n"
                        "Generate one image at a time on MPS."
                    ),
                    lines=6,
                    interactive=False,
                    elem_classes="status-box",
                )

            with gr.Tab("Suggestions", id="suggestions"):
                gr.Markdown("Pick a prompt idea, send it to the Generate tab, then adjust details.")
                with gr.Row():
                    suggestion_random_btn = gr.Button("Random Suggestion", variant="primary")
                    suggestion_output = gr.Textbox(
                        label="Selected Suggestion",
                        value=suggestion_one(),
                        lines=3,
                    )
                use_suggestion_btn = gr.Button("Use This Prompt")

                suggestion_status = gr.Textbox(
                    label="Suggestion Status",
                    value="Choose or randomize a suggestion, then use it as the prompt.",
                    lines=2,
                    interactive=False,
                    elem_classes="status-box",
                )

                with gr.Row():
                    suggestion_btn_1 = gr.Button("Sci-Fi Lab")
                    suggestion_btn_2 = gr.Button("Product Render")
                    suggestion_btn_3 = gr.Button("Reading Room")
                with gr.Row():
                    suggestion_btn_4 = gr.Button("Moon Base")
                    suggestion_btn_5 = gr.Button("Robot Macro")
                    suggestion_btn_6 = gr.Button("Calm Workspace")

        seed_btn.click(fn=random_seed, outputs=seed)
        safe_btn.click(
            fn=safe_settings,
            outputs=[model_label, resolution_label, steps, guidance, precision_mode, preset_status, tabs],
        )
        balanced_btn.click(
            fn=balanced_settings,
            outputs=[model_label, resolution_label, steps, guidance, precision_mode, preset_status, tabs],
        )
        quality_btn.click(
            fn=quality_settings,
            outputs=[model_label, resolution_label, steps, guidance, precision_mode, preset_status, tabs],
        )
        suggestion_random_btn.click(fn=random_suggestion, outputs=suggestion_output)
        suggestion_btn_1.click(fn=suggestion_one, outputs=suggestion_output)
        suggestion_btn_2.click(fn=suggestion_two, outputs=suggestion_output)
        suggestion_btn_3.click(fn=suggestion_three, outputs=suggestion_output)
        suggestion_btn_4.click(fn=suggestion_four, outputs=suggestion_output)
        suggestion_btn_5.click(fn=suggestion_five, outputs=suggestion_output)
        suggestion_btn_6.click(fn=suggestion_six, outputs=suggestion_output)
        use_suggestion_btn.click(
            fn=apply_suggestion,
            inputs=suggestion_output,
            outputs=[prompt, suggestion_status, tabs],
        )
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, negative_prompt, model_label, resolution_label, steps, guidance, seed, precision_mode],
            outputs=[output_image, output_file, status],
        )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
