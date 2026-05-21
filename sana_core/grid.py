from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from .engine import generate_image
from .metadata import write_json
from .paths import safe_output_path
from .schemas import GenerationRequest


def batch_output_name(base_request: GenerationRequest, seed: int) -> str:
    if base_request.output:
        output_path = Path(base_request.output)
        return str(
            output_path.with_name(
                f"{output_path.stem}_seed{seed}{output_path.suffix}"
            )
        )
    return (
        f"outputs/batch_seed{seed}_"
        f"{base_request.width}x{base_request.height}.png"
    )


def make_grid(
    image_paths: List[str],
    output_path: Path,
    columns: int = 2,
    padding: int = 16,
) -> None:
    if columns <= 0:
        raise ValueError("Grid columns must be greater than 0.")

    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        raise ValueError("No images to include in grid.")

    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns

    grid_width = columns * width + (columns + 1) * padding
    grid_height = rows * height + (rows + 1) * padding
    grid_image = Image.new("RGB", (grid_width, grid_height), "white")

    for index, image in enumerate(images):
        x_offset = padding + (index % columns) * (width + padding)
        y_offset = padding + (index // columns) * (height + padding)
        grid_image.paste(image, (x_offset, y_offset))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_image.save(output_path)


def generate_batch(
    base_request: GenerationRequest,
    seeds: List[int],
) -> List[Dict[str, Any]]:
    if not seeds:
        raise ValueError("Seed list must not be empty.")

    batch_items: List[Dict[str, Any]] = []
    for seed in seeds:
        request = replace(
            base_request,
            seed=seed,
            output=batch_output_name(base_request, seed),
        )
        result = generate_image(request)
        batch_items.append(dict(result.metadata))
    return batch_items


def generate_grid(
    base_request: GenerationRequest,
    seeds: List[int],
    columns: int = 2,
    output_name: str = "outputs/grid.png",
) -> Dict[str, Any]:
    if columns <= 0:
        raise ValueError("Grid columns must be greater than 0.")

    output_path = safe_output_path(output_name)

    batch_metadata = generate_batch(base_request, seeds)
    image_paths = [item["image_path"] for item in batch_metadata]
    make_grid(image_paths, output_path, columns=columns)

    grid_metadata = {
        "grid_path": str(output_path),
        "columns": columns,
        "seeds": seeds,
        "items": batch_metadata,
    }
    metadata_path = output_path.with_suffix(".json")
    write_json(metadata_path, grid_metadata)
    return grid_metadata
