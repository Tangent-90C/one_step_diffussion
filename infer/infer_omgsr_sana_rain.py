import os
import sys
import argparse
import glob
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from diffusers import AutoencoderDC, SanaPipeline, SanaTransformer2DModel
from diffusers.training_utils import free_memory


def _parse_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.lower()
    if dtype_str == "fp32":
        return torch.float32
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported weight_dtype: {dtype_str}")


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype: torch.dtype) -> torch.Tensor:
    encoded = vae.encode(pixels.to(vae.dtype))
    if hasattr(encoded, "latent_dist"):
        pixel_latents = encoded.latent_dist.sample()
    elif hasattr(encoded, "latent"):
        pixel_latents = encoded.latent
    else:
        raise AttributeError("EncoderOutput has neither latent_dist nor latent")

    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    pixel_latents = (pixel_latents - shift_factor) * scaling_factor
    return pixel_latents.to(weight_dtype)


def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


def get_sana_setting_timesteps(n: int = 999, resolution: int = 1024) -> list[float]:
    latent_size = resolution // 32
    return get_schedule(n, latent_size * latent_size, shift=True)


def _gather_images(input_path: str) -> list[str]:
    if input_path.lower().endswith(".txt"):
        with open(input_path, "r", encoding="utf-8") as f:
            return [l.strip() for l in f.readlines() if l.strip()]

    if os.path.isdir(input_path):
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        files: list[str] = []
        for root, _, filenames in os.walk(input_path):
            for name in filenames:
                if os.path.splitext(name)[1].lower() in exts:
                    files.append(os.path.join(root, name))
        return sorted(files)

    return [input_path]


def _resize_for_process(img: Image.Image, process_size: int) -> tuple[Image.Image, tuple[int, int], bool]:
    """Resize keeping aspect so that max(H,W) == process_size (if needed), then make dims divisible by 32."""
    ori_w, ori_h = img.size

    scale = process_size / max(ori_w, ori_h)
    resized = False
    if abs(scale - 1.0) > 1e-6:
        new_w = max(1, int(round(ori_w * scale)))
        new_h = max(1, int(round(ori_h * scale)))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        resized = True

    # Sana latent is /32; enforce divisible by 32.
    w, h = img.size
    w32 = max(32, w - (w % 32))
    h32 = max(32, h - (h % 32))
    if (w32, h32) != (w, h):
        img = img.resize((w32, h32), Image.LANCZOS)
        resized = True

    return img, (ori_w, ori_h), resized


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    weight_dtype = _parse_dtype(args.weight_dtype)

    # 1) Encode prompt once.
    text_encoding_pipeline = SanaPipeline.from_pretrained(
        args.model_path, transformer=None, vae=None, torch_dtype=weight_dtype
    ).to(device)

    prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(args.prompt)

    text_encoding_pipeline = text_encoding_pipeline.to("cpu")
    del text_encoding_pipeline
    free_memory()

    # 2) Load base models.
    fixed_vae = AutoencoderDC.from_pretrained(args.model_path, subfolder="vae", torch_dtype=weight_dtype)
    fixed_vae.requires_grad_(False)
    fixed_vae.eval().to(device)

    # lora_vae: encoder has LoRA; decoder not needed for encoding.
    import copy

    lora_vae = copy.deepcopy(fixed_vae)
    lora_vae.requires_grad_(False)
    if hasattr(lora_vae, "decoder"):
        del lora_vae.decoder
    free_memory()

    sana_transformer = SanaTransformer2DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=weight_dtype
    )
    sana_transformer.requires_grad_(False)

    # 3) Load LoRA adapters.
    from peft import PeftModel

    sana_adapter_path = os.path.join(args.lora_path, "sana_adapter")
    vae_encoder_adapter_path = os.path.join(args.lora_path, "vae_encoder_adapter")

    sana_transformer = PeftModel.from_pretrained(sana_transformer, sana_adapter_path)
    lora_vae.encoder = PeftModel.from_pretrained(lora_vae.encoder, vae_encoder_adapter_path)

    sana_transformer.eval().to(device)
    lora_vae.eval().to(device)

    # 4) Mid-timestep setup (match training resolution).
    sana_timesteps = get_sana_setting_timesteps(resolution=args.process_size)
    sigma_t = sana_timesteps[-(args.mid_timestep + 1)]

    shift_factor = getattr(fixed_vae.config, "shift_factor", 0.0)
    scaling_factor = getattr(fixed_vae.config, "scaling_factor", 1.0)

    def one_mid_timestep_pred(lq_latent: torch.Tensor) -> torch.Tensor:
        bsz = lq_latent.shape[0]
        guidance_vec = torch.full((bsz,), args.guidance, device=device, dtype=lq_latent.dtype)

        model_pred = sana_transformer(
            hidden_states=lq_latent,
            encoder_hidden_states=prompt_embeds,
            timestep=torch.tensor([sigma_t], device=device),
            encoder_attention_mask=prompt_attention_mask,
            guidance=guidance_vec,
            return_dict=False,
        )[0]

        refined_latent = lq_latent - sigma_t * model_pred

        refined_latent = (refined_latent / scaling_factor) + shift_factor
        pred_img = fixed_vae.decode(refined_latent.to(fixed_vae.dtype), return_dict=False)[0]
        return pred_img

    image_names = _gather_images(args.input_image)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"There are {len(image_names)} images.")

    for image_name in tqdm(image_names):
        inp = Image.open(image_name).convert("RGB")
        proc_img, (ori_w, ori_h), resized = _resize_for_process(inp, args.process_size)

        lq_img = F.to_tensor(proc_img).unsqueeze(0).to(device=device, dtype=weight_dtype) * 2 - 1

        lq_latent = encode_images(lq_img, lora_vae, weight_dtype)
        pred_img = one_mid_timestep_pred(lq_latent)

        out = (pred_img * 0.5 + 0.5).clamp(0, 1).float()
        out_pil = transforms.ToPILImage()(out[0].cpu())

        if resized:
            out_pil = out_pil.resize((ori_w, ori_h), Image.LANCZOS)

        bname = os.path.splitext(os.path.basename(image_name))[0] + ".png"
        out_pil.save(os.path.join(args.output_dir, bname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMGSR + Sana (rain) inference")

    parser.add_argument("--input_image", type=str, required=True, help="Input image / directory / txt list")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--model_path", type=str, required=True, help="Base Sana model path")
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/home/chenjn/OMGSR/omgsr_trainings/omgsr_sana_1024_rain/weight-12000",
        help="Trained LoRA folder (contains sana_adapter/ and vae_encoder_adapter/)",
    )

    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument("--process_size", type=int, default=1024, help="Reference resolution for schedule + resizing")
    parser.add_argument("--weight_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])

    parser.add_argument("--prompt", type=str, default="", help="Prompt for conditioning (usually empty)")
    parser.add_argument("--mid_timestep", type=int, default=244, help="Mid timestep used in training")
    parser.add_argument("--guidance", type=float, default=5.0, help="Guidance value used in Sana forward")

    main(parser.parse_args())
