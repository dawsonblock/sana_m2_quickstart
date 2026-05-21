import argparse
import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from sana_core.engine import generate_image, get_device
from sana_core.gallery import list_gallery_items, list_metadata_files
from sana_core.grid import generate_batch, generate_grid
from sana_core.metadata import append_jsonl, read_json
from sana_core.network import get_lan_ip
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
    tags: List[str] = Field(default_factory=list)


app = FastAPI(title="Sana M2 API", version="0.1.0")
ensure_dirs()
app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "static")),
    name="static",
)

PHONE_MODE = False
PHONE_TOKEN: Optional[str] = None
PHONE_HOST = "127.0.0.1"
PHONE_PORT = 7111
PHONE_LAN_IP = "127.0.0.1"


def _phone_url(token: Optional[str] = None) -> str:
    base_url = f"http://{PHONE_LAN_IP}:{PHONE_PORT}/phone"
    if token:
        return f"{base_url}?token={token}"
    return base_url


def _print_phone_qr(url: str) -> None:
    try:
        import qrcode  # type: ignore[import-not-found,import-untyped]
    except ImportError:
        print("QR: qrcode dependency not installed; open URL manually.")
        return
    try:
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception:
        print("QR: unable to render terminal QR; open URL manually.")


def initialize_phone_mode(host: str, port: int, phone_mode: bool) -> None:
    global PHONE_MODE
    global PHONE_TOKEN
    global PHONE_HOST
    global PHONE_PORT
    global PHONE_LAN_IP

    PHONE_MODE = phone_mode
    PHONE_HOST = host
    PHONE_PORT = port
    PHONE_LAN_IP = get_lan_ip()

    if not phone_mode:
        return

    env_token = os.environ.get("SANA_PHONE_TOKEN", "").strip()
    if env_token:
        PHONE_TOKEN = env_token
        print("Phone mode token source: SANA_PHONE_TOKEN")
    else:
        PHONE_TOKEN = secrets.token_urlsafe(12)
        print("Phone mode token source: temporary (session only)")

    print("Phone mode is LAN-only. Do not port-forward this server.")
    print(f"Phone URL: {_phone_url(PHONE_TOKEN)}")
    print(f"Phone token: {PHONE_TOKEN}")
    _print_phone_qr(_phone_url(PHONE_TOKEN))


def _get_phone_token() -> Optional[str]:
    if PHONE_TOKEN:
        return PHONE_TOKEN
    env_token = os.environ.get("SANA_PHONE_TOKEN", "").strip()
    if env_token:
        return env_token
    return None


def require_phone_token(request: Request) -> None:
    expected_token = _get_phone_token()
    if not expected_token:
        raise HTTPException(
            status_code=401,
            detail="Phone token is not configured",
        )

    header_token = request.headers.get("X-Sana-Token", "").strip()
    query_token = request.query_params.get("token", "").strip()
    provided = header_token or query_token
    if provided != expected_token:
        raise HTTPException(status_code=401, detail="Invalid phone token")


def require_phone_token_if_enabled(request: Request) -> None:
    if PHONE_MODE:
        require_phone_token(request)


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


def _raise_api_error(error: Exception) -> NoReturn:
    if isinstance(error, HTTPException):
        raise error
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail=str(error)) from error
    raise HTTPException(status_code=500, detail="Generation failed") from error


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "device": get_device()}


@app.get("/models")
def models() -> Dict[str, Any]:
    return {"models": MODEL_OPTIONS}


@app.post("/generate")
def generate(body: GenerateBody, request: Request) -> Dict[str, Any]:
    require_phone_token_if_enabled(request)
    _log_api_request("/generate", body.model_dump())
    try:
        result = generate_image(_to_request(body))
    except Exception as error:
        _raise_api_error(error)
    return {
        "image_path": result.image_path,
        "metadata_path": result.metadata_path,
        "metadata": result.metadata,
    }


@app.post("/generate/batch")
def generate_batch_endpoint(
    body: BatchBody,
    request: Request,
) -> Dict[str, Any]:
    require_phone_token_if_enabled(request)
    if not body.seeds:
        raise HTTPException(
            status_code=400,
            detail="Seed list must not be empty",
        )
    _log_api_request("/generate/batch", body.model_dump())
    try:
        items = generate_batch(
            _to_request(body, seed=body.seeds[0]),
            body.seeds,
        )
    except Exception as error:
        _raise_api_error(error)
    return {"items": items}


@app.post("/generate/grid")
def generate_grid_endpoint(body: GridBody, request: Request) -> Dict[str, Any]:
    require_phone_token_if_enabled(request)
    if not body.seeds:
        raise HTTPException(
            status_code=400,
            detail="Seed list must not be empty",
        )
    _log_api_request("/generate/grid", body.model_dump())
    try:
        return generate_grid(
            base_request=_to_request(body, seed=body.seeds[0]),
            seeds=body.seeds,
            columns=body.columns,
            output_name=body.output or "outputs/grid.png",
        )
    except Exception as error:
        _raise_api_error(error)


@app.get("/outputs")
def outputs() -> Dict[str, Any]:
    return {"items": list_gallery_items()}


@app.get("/outputs/{image_path:path}")
def output_file(image_path: str, request: Request) -> FileResponse:
    require_phone_token_if_enabled(request)
    return FileResponse(_safe_project_path(f"outputs/{image_path}"))


@app.get("/file/{image_path:path}")
def file_output(image_path: str, request: Request) -> FileResponse:
    require_phone_token_if_enabled(request)
    return FileResponse(_safe_project_path(f"outputs/{image_path}"))


@app.get("/metadata")
def metadata_list() -> Dict[str, Any]:
    items = [
        str(path.relative_to(PROJECT_ROOT))
        for path in list_metadata_files()
    ]
    return {"items": items}


@app.get("/metadata/{metadata_filename:path}")
def metadata_file(metadata_filename: str, request: Request) -> Dict[str, Any]:
    require_phone_token_if_enabled(request)
    if Path(metadata_filename).suffix.lower() != ".json":
        raise HTTPException(
            status_code=400,
            detail="Metadata file must be .json",
        )
    path = _safe_project_path(f"outputs/{metadata_filename}")
    return read_json(path)


@app.get("/gallery")
def gallery() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "static" / "gallery.html")


@app.get("/phone")
def phone(request: Request) -> FileResponse:
    require_phone_token(request)
    return FileResponse(PROJECT_ROOT / "static" / "phone.html")


@app.get("/api/phone/status")
def phone_status(request: Request) -> Dict[str, Any]:
    require_phone_token(request)
    return {
        "ok": True,
        "device": get_device(),
        "mps_available": get_device() == "mps",
        "selected_model": MODEL_OPTIONS[0],
        "server_url": _phone_url(),
        "phone_mode": PHONE_MODE,
    }


@app.post("/api/phone/generate")
def phone_generate(body: GenerateBody, request: Request) -> Dict[str, Any]:
    require_phone_token(request)
    return generate(body, request)


@app.post("/api/phone/grid")
def phone_grid(body: GridBody, request: Request) -> Dict[str, Any]:
    require_phone_token(request)
    result = generate_grid_endpoint(body, request)
    return {
        "grid_image_path": result.get("grid_path"),
        "grid_metadata_path": str(
            Path(result.get("grid_path", "")).with_suffix(".json")
        ),
        "items": result.get("items", []),
        "columns": result.get("columns"),
        "seeds": result.get("seeds"),
    }


@app.get("/api/phone/gallery")
def phone_gallery(request: Request) -> Dict[str, Any]:
    require_phone_token(request)
    return {"items": list_gallery_items()}


@app.get("/api/phone/presets")
def phone_presets(request: Request) -> Dict[str, Any]:
    require_phone_token(request)
    return {"items": list_presets()}


@app.post("/api/phone/presets")
def phone_preset_save(body: PresetBody, request: Request) -> Dict[str, Any]:
    require_phone_token(request)
    _log_api_request("/api/phone/presets", body.model_dump())
    return save_preset(body.model_dump())


@app.get("/api/phone/file/{image_path:path}")
def phone_output_file(image_path: str, request: Request) -> FileResponse:
    require_phone_token(request)
    return FileResponse(_safe_project_path(f"outputs/{image_path}"))


@app.get("/api/phone/metadata/{metadata_filename:path}")
def phone_metadata_file(
    metadata_filename: str,
    request: Request,
) -> Dict[str, Any]:
    require_phone_token(request)
    if Path(metadata_filename).suffix.lower() != ".json":
        raise HTTPException(
            status_code=400,
            detail="Metadata file must be .json",
        )
    path = _safe_project_path(f"outputs/{metadata_filename}")
    return read_json(path)


@app.get("/presets")
def presets() -> Dict[str, Any]:
    return {"items": list_presets()}


@app.post("/presets")
def preset_save(body: PresetBody, request: Request) -> Dict[str, Any]:
    require_phone_token_if_enabled(request)
    _log_api_request("/presets", body.model_dump())
    return save_preset(body.model_dump())


@app.delete("/presets/{preset_id}")
def preset_delete(preset_id: str, request: Request) -> Dict[str, Any]:
    require_phone_token_if_enabled(request)
    deleted = delete_preset(preset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"deleted": True, "preset_id": preset_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sana M2 FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7111)
    parser.add_argument("--phone", action="store_true")
    args = parser.parse_args()

    initialize_phone_mode(args.host, args.port, args.phone)
    uvicorn.run("api_server:app", host=args.host, port=args.port, reload=False)
