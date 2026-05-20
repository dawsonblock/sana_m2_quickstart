import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from diffusers import SanaPipeline


MODEL_ID = "Efficient-Large-Model/Sana_600M_512px_diffusers"


def main():
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    pipe = SanaPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )
    pipe = pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    prompt = "a clean futuristic workbench with a small robot assembling a glowing circuit board, sharp details"
    image = pipe(
        prompt=prompt,
        height=512,
        width=512,
        guidance_scale=4.5,
        num_inference_steps=20,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]
    image.save("sana_m2_output.png")
    print("Saved sana_m2_output.png")


if __name__ == "__main__":
    main()
