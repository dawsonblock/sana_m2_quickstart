import json
from pathlib import Path
from typing import Any, Dict, List

from .paths import OUTPUT_DIR, PROJECT_ROOT


def list_metadata_files() -> List[Path]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(
        OUTPUT_DIR.rglob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def list_gallery_items(limit: int = 100) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for metadata_path in list_metadata_files()[:limit]:
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        image_path = data.get("image_path")
        if not image_path:
            continue

        image_file = PROJECT_ROOT / image_path
        items.append(
            {
                "image_path": image_path,
                "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
                "prompt": data.get("prompt"),
                "negative_prompt": data.get("negative_prompt"),
                "model": data.get("model"),
                "width": data.get("width"),
                "height": data.get("height"),
                "steps": data.get("steps"),
                "guidance": data.get("guidance"),
                "seed": data.get("seed"),
                "created_at": data.get("created_at"),
                "runtime_seconds": data.get("runtime_seconds"),
                "image_exists": image_file.exists(),
            }
        )
    return items
