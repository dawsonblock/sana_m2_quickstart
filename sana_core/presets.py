import json
from typing import Any, Dict, List

from .metadata import write_json
from .paths import PRESET_DIR


PRESET_PATH = PRESET_DIR / "prompt_presets.json"


DEFAULT_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "wheelchair_drone_interface",
        "name": "Wheelchair Drone Interface",
        "prompt": (
            "technical render of a chin-controlled drone interface mounted on "
            "a "
            "power wheelchair, clean white background, product design style"
        ),
        "negative_prompt": "blurry, messy, low quality, distorted",
        "width": 512,
        "height": 512,
        "steps": 12,
        "guidance": 4.5,
        "dtype": "float16",
        "tags": ["assistive-tech", "drone", "interface"],
    },
    {
        "id": "judge_atlas_dashboard",
        "name": "JUDGE_ATLASX Dashboard",
        "prompt": (
            "clean dashboard UI for legal evidence map system, Canada map, "
            "case "
            "markers, evidence timeline, modern white interface"
        ),
        "negative_prompt": "cluttered, blurry, dark, low quality",
        "width": 512,
        "height": 512,
        "steps": 12,
        "guidance": 4.5,
        "dtype": "float16",
        "tags": ["ui", "legal-tech", "dashboard"],
    },
]


def ensure_preset_file() -> None:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    if not PRESET_PATH.exists():
        write_json(PRESET_PATH, {"presets": DEFAULT_PRESETS})


def _load_preset_document() -> Dict[str, Any]:
    ensure_preset_file()
    return json.loads(PRESET_PATH.read_text(encoding="utf-8"))


def list_presets() -> List[Dict[str, Any]]:
    document = _load_preset_document()
    return list(document.get("presets", []))


def save_preset(preset: Dict[str, Any]) -> Dict[str, Any]:
    presets = list_presets()
    preset_id = preset["id"]
    updated = [item for item in presets if item.get("id") != preset_id]
    updated.append(preset)
    write_json(PRESET_PATH, {"presets": updated})
    return preset


def delete_preset(preset_id: str) -> bool:
    presets = list_presets()
    updated = [item for item in presets if item.get("id") != preset_id]
    if len(updated) == len(presets):
        return False
    write_json(PRESET_PATH, {"presets": updated})
    return True
