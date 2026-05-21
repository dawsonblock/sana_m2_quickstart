from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
PRESET_DIR = PROJECT_ROOT / "presets"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRESET_DIR.mkdir(parents=True, exist_ok=True)


def safe_output_path(name: str) -> Path:
    path = Path(name)
    if path.is_absolute():
        raise ValueError("Output path must be relative.")
    if ".." in path.parts:
        raise ValueError("Output path must not contain '..'.")
    if path.suffix.lower() != ".png":
        raise ValueError("Output path must end with .png.")
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(
            "Output path must stay inside the project directory."
        ) from error
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
