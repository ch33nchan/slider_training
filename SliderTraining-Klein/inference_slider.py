"""
Inference script for Klein 9B directional LoRA slider.

Generates images at different slider scales from a source image,
producing a stacked visualization showing continuous concept variation.

Usage:
  python inference_slider.py \
    --config config/smile_slider.yaml \
    --lora_path outputs/weights/slider_latest.safetensors \
    --source_image /path/to/portrait.png \
    --prompt "a person" \
    --scales -5 -2.5 0 2.5 5 \
    --output outputs/inference_result.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

from models.transformer import Flux2, Klein9BParams
from models.autoencoder import AutoEncoder, AutoEncoderParams
from models.sampling import (
    batched_prc_img,
    batched_prc_txt,
    default_prep,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from utils.text_encoder import load_text_encoder, encode_prompt
from utils.lora import LoRANetwork


def load_transformer(path: str, device: str, dtype=torch.bfloat16):
    """Load Klein 9B transformer from safetensors."""
    params = Klein9BParams()
    transformer = Flux2(params)
    sd = load_file(path)
    transformer.load_state_dict(sd)
    transformer.to(device, dtype=dtype)
    transformer.requires_grad_(False)
    transformer.eval()
    return transformer


def load_vae(path: str, device: str, dtype=torch.bfloat16):
    """Load VAE from safetensors."""
    params = AutoEncoderParams()
    vae = AutoEncoder(params)
    sd = load_file(path)
    vae.load_state_dict(sd)
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    return vae


@torch.no_grad()
def denoise_img2img(
    transformer,
    network,
    packed_noise,
    noise_ids,
    txt,
    txt_ids,
    timesteps,
    ref_tokens,
    ref_ids,
    device,
    dtype,
):
    """
    Denoise from noise with reference image conditioning and LoRA.

    Standard Euler denoising loop:
      x = x + (t_prev - t_curr) * pred
    where pred is the transformer output conditioned on reference tokens + text.
    """
    img = packed_noise
    for t_curr, t_prev in tqdm(
        zip(timesteps[:-1], timesteps[1:]),
        total=len(timesteps) - 1,
        desc="Denoising",
        leave=False,
    ):
        t_vec = torch.full((1,), t_curr, device=device, dtype=dtype)

        img_input = torch.cat([img, ref_tokens], dim=1)
        img_input_ids = torch.cat([noise_ids, ref_ids], dim=1)

        with network:
            pred = transformer(
                x=img_input,
                x_ids=img_input_ids,
                timesteps=t_vec,
                ctx=txt,
                ctx_ids=txt_ids,
                guidance=None,
            )
        pred = pred[:, : img.shape[1]]
        img = img + (t_prev - t_curr) * pred

    return img


def latent_to_pil(vae, packed_latent, latent_ids, dtype):
    """Unpack latent tokens and decode to PIL image."""
    output_latent = torch.cat(scatter_ids(packed_latent, latent_ids)).squeeze(2)
    output_img = vae.decode(output_latent.to(dtype)).float()
    output_img = (output_img.clamp(-1, 1) + 1) / 2
    output_np = output_img[0].permute(1, 2, 0).cpu().numpy()
    output_np = (output_np * 255).astype(np.uint8)
    return Image.fromarray(output_np)


def main():
    parser = argparse.ArgumentParser(description="Klein 9B slider inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--source_image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a person")
    parser.add_argument(
        "--scales", type=float, nargs="+", default=[-5, -2.5, 0, 2.5, 5]
    )
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strength", type=float, default=0.6,
                        help="img2img strength 0-1: 0=no change, 1=full regen from noise")
    parser.add_argument("--output", type=str, default="outputs/inference_result.png")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = cfg.device
    dtype = torch.bfloat16

    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------
    print("Loading transformer...")
    transformer = load_transformer(cfg.transformer_path, device, dtype)

    print("Loading VAE...")
    vae = load_vae(cfg.vae_path, device, dtype)

    # Load text encoder, encode prompt, then offload to save VRAM
    print("Loading text encoder...")
    text_encoder, tokenizer = load_text_encoder(cfg.te_path, device, dtype)
    prompt_embeds = encode_prompt(text_encoder, tokenizer, args.prompt, device)
    text_encoder.to("cpu")
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    txt, txt_ids = batched_prc_txt(prompt_embeds)

    # -------------------------------------------------------------------------
    # Encode source image as reference tokens + as starting latent (img2img)
    # -------------------------------------------------------------------------
    print("Encoding reference image...")
    source_img = Image.open(args.source_image).convert("RGB").resize(
        (cfg.width, cfg.height)
    )
    ref_tokens, ref_ids = encode_image_refs(vae, [source_img])
    ref_tokens = ref_tokens.to(device, dtype=dtype)
    ref_ids = ref_ids.to(device)

    # Encode source as starting latent for img2img
    source_tensor = default_prep(source_img, limit_pixels=cfg.height * cfg.width)
    if isinstance(source_tensor, list):
        source_tensor = source_tensor[0]
    with torch.no_grad():
        source_latent = vae.encode(source_tensor.unsqueeze(0).to(device, dtype=dtype))[0]
    packed_source, _ = batched_prc_img(source_latent.unsqueeze(0))

    # -------------------------------------------------------------------------
    # Create LoRA network and load weights
    # -------------------------------------------------------------------------
    print("Loading LoRA weights...")
    lora_sd = load_file(args.lora_path)
    # detect rank from checkpoint
    lora_rank = cfg.rank
    for k, v in lora_sd.items():
        if "lora_A" in k or "lora_down" in k:
            lora_rank = v.shape[0]
            break
    network = LoRANetwork(
        transformer,
        rank=lora_rank,
        multiplier=1.0,
        alpha=cfg.alpha,
        train_method=cfg.train_method,
        save_dir=".",
    ).to(device, dtype=dtype)
    # Remap ai-toolkit key format to inference_slider format if needed
    if any(k.startswith("diffusion_model.") for k in lora_sd):
        remapped = {}
        for k, v in lora_sd.items():
            # strip "diffusion_model." prefix
            k2 = k.replace("diffusion_model.", "")
            # replace dots with underscores up to lora_A/lora_B
            parts = k2.split(".")
            lora_suffix = parts[-2]  # lora_A or lora_B
            weight_suffix = parts[-1]  # weight
            module_parts = parts[:-2]
            module_key = "_".join(module_parts)
            lora_dir = "lora_down" if lora_suffix == "lora_A" else "lora_up"
            new_key = f"{module_key}.{lora_dir}.{weight_suffix}"
            remapped[new_key] = v
        # add alpha keys (use cfg.alpha value for all modules)
        alpha_val = torch.tensor(float(cfg.alpha))
        for k in list(remapped.keys()):
            if k.endswith(".lora_down.weight"):
                alpha_key = k.replace(".lora_down.weight", ".alpha")
                remapped[alpha_key] = alpha_val
        # filter to only keys present in the network (drops MLP keys etc.)
        network_keys = set(network.state_dict().keys())
        lora_sd = {k: v for k, v in remapped.items() if k in network_keys}
    network.load_state_dict(lora_sd, strict=False)

    # -------------------------------------------------------------------------
    # Generate images at different slider scales
    # -------------------------------------------------------------------------
    height_latent = cfg.height // 16
    width_latent = cfg.width // 16

    slider_images = []
    print(f"\nGenerating images at scales: {args.scales}")

    for scale in args.scales:
        print(f"\nScale: {scale}")
        network.set_lora_slider(scale)

        generator = torch.Generator(device=device).manual_seed(args.seed)
        noise = torch.randn(
            1, 128, height_latent, width_latent,
            generator=generator, device=device, dtype=dtype,
        )
        packed_noise, noise_ids = batched_prc_img(noise)

        timesteps = get_schedule(args.num_steps, packed_noise.shape[1])

        if args.strength >= 1.0:
            # Full generation from noise
            packed_start = packed_noise
            timesteps_use = timesteps
        else:
            # img2img: blend source latent with noise at t_start, then denoise the rest
            start_idx = int((1.0 - args.strength) * (len(timesteps) - 1))
            t_start = timesteps[start_idx]
            packed_start = (1.0 - t_start) * packed_source + t_start * packed_noise
            timesteps_use = timesteps[start_idx:]

        # Denoise with reference conditioning + LoRA
        packed_output = denoise_img2img(
            transformer, network,
            packed_start, noise_ids,
            txt, txt_ids,
            timesteps_use,
            ref_tokens, ref_ids,
            device, dtype,
        )

        # Decode to PIL
        output_pil = latent_to_pil(vae, packed_output, noise_ids, dtype)
        slider_images.append(output_pil)

    # -------------------------------------------------------------------------
    # Create stacked visualization
    # -------------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Include source image in visualization
    n = len(slider_images) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    # Source image
    axes[0].imshow(np.array(source_img))
    axes[0].axis("off")
    axes[0].set_title("Source", fontsize=14)

    # Slider outputs
    for i, (img, scale) in enumerate(zip(slider_images, args.scales)):
        axes[i + 1].imshow(np.array(img))
        axes[i + 1].axis("off")
        axes[i + 1].set_title(f"Scale: {scale}", fontsize=14)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved visualization to {output_path}")

    # Save individual images
    for img, scale in zip(slider_images, args.scales):
        individual_path = output_path.parent / f"scale_{scale:+.1f}.png"
        img.save(str(individual_path))
        print(f"  Saved {individual_path}")


if __name__ == "__main__":
    main()
