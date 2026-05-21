from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: Optional[str] = None
    model: str = "Efficient-Large-Model/Sana_600M_512px_diffusers"
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance: float = 4.5
    seed: int = 42
    dtype: str = "float16"
    attention_slicing: bool = True
    output: Optional[str] = None


@dataclass
class GenerationResult:
    image_path: str
    metadata_path: str
    metadata: Dict[str, Any]
