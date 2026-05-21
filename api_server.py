from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from sana_core.engine import generate_image, get_device
from sana_core.gallery import list_gallery_items, list_metadata_files
from sana_core.grid import generate_batch, generate_grid
from sana_core.metadata import append_jsonl, read_json
from sana_core.paths import LOG_DIR, PROJECT_ROOT, ensure_dirs
from sana_core.presets import delete_preset, list_presets, save_preset
from sana_core.schemas import GenerationRequest


MODEL_OPTIONS = [
    "Efficient-Large-Model/Sana_600M_512px_diffusers",
    "Efficient-Large-Model/Sana_600M_1024px_diffusers",
    "Efficient-Large-Model/Sana_1600M_512px_diffusers",
]


class GenerateBody(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    model: str = MODEL_OPTIONS[0]
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance: float = 4.5
    seed: int = 42
    dtype: str = "float16"
    attention_slicing: bool = True
    output: Optional[str] = None


class BatchBody(GenerateBody):
    seeds: List[int]


class GridBody(BatchBody):
    columns: int = 2
    output: Optional[str] = "outputs/grid.png"


class PresetBody(BaseModel):
    id: str
    name: str
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    steps: int = 12
    guidance: float = 4.5
    dtype: str = "float16"
    tags: List[str] = []


app = FastAPI(title="Sana M2 API", version="0.1.0")
ensure_dirs()
app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "static")),
    name="static",
)


def _to_request(
    body: GenerateBody,
    seed: Optional[int] = None,
) -> GenerationRequest:
    return GenerationRequest(
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        model=body.model,
        width=body.width,
        height=body.height,
        steps=body.steps,
        guidance=body.guidance,
        seed=body.seed if seed is None else seed,
        dtype=body.dtype,
        attention_slicing=body.attention_slicing,
        output=body.output,
    )


def _safe_project_path(relative_path: str) -> Path:
    raw_path = Path(relative_path)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise HTTPException(status_code=400, detail="Invalid path")
    resolved = (PROJECT_ROOT / raw_path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid path") from error
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return resolved


def _log_api_request(endpoint: str, payload: Dict[str, Any]) -> None:
    append_jsonl(
        LOG_DIR / "api_requests.jsonl",
        {
            "endpoint": endpoint,
            "payload": payload,
        },
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "device": get_device()}


@app.get("/models")
def models() -> Dict[str, Any]:
    return {"models": MODEL_OPTIONS}


@app.post("/generate")
def generate(body: GenerateBody) -> Dict[str, Any]:
    _log_api_request("/generate", body.model_dump())
    result = generate_image(_to_request(body))
    return {
        "image_path": result.image_path,
        "metadata_path": result.metadata_path,
        "metadata": result.metadata,
    }


@app.post("/generate/batch")
def generate_batch_endpoint(body: BatchBody) -> Dict[str, Any]:
    if not body.seeds:
        raise HTTPException(
            status_code=400,
            detail="Seed list must not be empty",
        )
    _log_api_request("/generate/batch", body.model_dump())
    items = generate_batch(_to_request(body, seed=body.seeds[0]), body.seeds)
    return {"items": items}


@app.post("/generate/grid")
def generate_grid_endpoint(body: GridBody) -> Dict[str, Any]:
    if not body.seeds:
        raise HTTPException(
            status_code=400,
            detail="Seed list must not be empty",
        )
    _log_api_request("/generate/grid", body.model_dump())
    return generate_grid(
        base_request=_to_request(body, seed=body.seeds[0]),
        seeds=body.seeds,
        columns=body.columns,
        output_name=body.output or "outputs/grid.png",
    )


@app.get("/outputs")
def outputs() -> Dict[str, Any]:
    return {"items": list_gallery_items()}


@app.get("/outputs/{image_filename}")
def output_file(image_filename: str) -> FileResponse:
    return FileResponse(_safe_project_path(f"outputs/{image_filename}"))


@app.get("/metadata")
def metadata_list() -> Dict[str, Any]:
    items = [
        str(path.relative_to(PROJECT_ROOT))
        for path in list_metadata_files()
    ]
    return {"items": items}


@app.get("/metadata/{metadata_filename}")
def metadata_file(metadata_filename: str) -> Dict[str, Any]:
    path = _safe_project_path(f"outputs/{metadata_filename}")
    return read_json(path)


@app.get("/file/{path:path}")
def project_file(path: str) -> FileResponse:
    return FileResponse(_safe_project_path(path))


@app.get("/gallery")
def gallery() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "static" / "gallery.html")


@app.get("/presets")
def presets() -> Dict[str, Any]:
    return {"items": list_presets()}


@app.post("/presets")
def preset_save(body: PresetBody) -> Dict[str, Any]:
    _log_api_request("/presets", body.model_dump())
    return save_preset(body.model_dump())


@app.delete("/presets/{preset_id}")
def preset_delete(preset_id: str) -> Dict[str, Any]:
    deleted = delete_preset(preset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"deleted": True, "preset_id": preset_id}


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="127.0.0.1", port=7861, reload=False)
