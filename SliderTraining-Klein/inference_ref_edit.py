"""
Reference token editing inference for Klein 9B.

Instead of modifying transformer weights (LoRA), this script shifts the
reference tokens in VAE latent space along a precomputed gaze direction:

    ref_shifted = ref_source + scale * gaze_delta

Since Klein is designed to faithfully reproduce the reference image,
shifting reference tokens toward "right gaze" causes the model to generate
the source identity with shifted gaze.

Usage:
    # First compute the delta:
    python compute_gaze_delta.py --device cuda:1

    # Then run inference:
    python inference_ref_edit.py \
        --delta_path gaze_delta.pt \
        --source_image data/eye_gaze_v3/neutral/face_0100.png \
        --output outputs/ref_edit/result.png \
        --device cuda:1
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from models.autoencoder import AutoEncoder, AutoEncoderParams
from models.sampling import (
    batched_prc_img,
    batched_prc_txt,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from models.transformer import Flux2, Klein9BParams
from utils.lora import LoRANetwork
from utils.text_encoder import encode_prompt, load_text_encoder


def load_transformer(path: str, device: str, dtype=torch.bfloat16):
    params = Klein9BParams()
    transformer = Flux2(params)
    sd = load_file(path)
    transformer.load_state_dict(sd)
    transformer.to(device, dtype=dtype)
    transformer.requires_grad_(False)
    transformer.eval()
    return transformer


def load_vae(path: str, device: str, dtype=torch.bfloat16):
    params = AutoEncoderParams()
    vae = AutoEncoder(params)
    sd = load_file(path)
    vae.load_state_dict(sd)
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    return vae


@torch.no_grad()
def generate(
    transformer,
    vae,
    ref_tokens,
    ref_ids,
    prompt_txt,
    prompt_txt_ids,
    device,
    dtype,
    height,
    width,
    num_steps=28,
    seed=42,
):
    height_latent = height // 16
    width_latent = width // 16

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        1, 128, height_latent, width_latent,
        generator=generator, device=device, dtype=dtype,
    )
    packed_noise, noise_ids = batched_prc_img(noise)
    timesteps = get_schedule(num_steps, packed_noise.shape[1])

    img = packed_noise
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((1,), t_curr, device=device, dtype=dtype)
        img_input = torch.cat([img, ref_tokens], dim=1)
        img_input_ids = torch.cat([noise_ids, ref_ids], dim=1)
        pred = transformer(
            x=img_input, x_ids=img_input_ids, timesteps=t_vec,
            ctx=prompt_txt, ctx_ids=prompt_txt_ids, guidance=None,
        )
        pred = pred[:, : img.shape[1]]
        img = img + (t_prev - t_curr) * pred

    output_latent = torch.cat(scatter_ids(img, noise_ids)).squeeze(2)
    output_img = vae.decode(output_latent.to(dtype)).float()
    output_img = (output_img.clamp(-1, 1) + 1) / 2
    output_np = output_img[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((output_np * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_path", default="models/klein-9b/flux-2-klein-9b.safetensors")
    parser.add_argument("--vae_path", default="models/klein-9b/ae.safetensors")
    parser.add_argument("--te_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--delta_path", required=True, help="Path to gaze_delta.pt")
    parser.add_argument("--source_image", required=True)
    parser.add_argument("--output", default="outputs/ref_edit/result.png")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--prompt", default="a person")
    parser.add_argument("--scales", nargs="+", type=float, default=[-5, -2.5, 0, 2.5, 5])
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--lora_path", default=None, help="Optional LoRA weights to combine with ref edit")
    parser.add_argument("--lora_config", default=None, help="Config yaml if using LoRA")
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16

    print("Loading transformer...")
    transformer = load_transformer(args.transformer_path, device, dtype)
    print("Loading VAE...")
    vae = load_vae(args.vae_path, device, dtype)

    print("Loading text encoder...")
    text_encoder, tokenizer = load_text_encoder(args.te_path, device, dtype)
    prompt_embeds = encode_prompt(text_encoder, tokenizer, args.prompt, device)
    prompt_txt, prompt_txt_ids = batched_prc_txt(prompt_embeds)
    text_encoder.to("cpu")
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    # Optional LoRA
    network = None
    if args.lora_path:
        from omegaconf import OmegaConf
        from safetensors.torch import load_file as load_sf
        cfg = OmegaConf.load(args.lora_config)
        network = LoRANetwork(
            transformer, rank=cfg.rank, multiplier=1.0,
            alpha=cfg.alpha, train_method=cfg.train_method,
        ).to(device, dtype=dtype)
        network.load_state_dict(load_sf(args.lora_path))
        print(f"Loaded LoRA from {args.lora_path}")

    print(f"Loading gaze delta from {args.delta_path}...")
    mean_delta = torch.load(args.delta_path, map_location="cpu")  # (1, N, C)
    print(f"  Shape: {mean_delta.shape}, norm: {mean_delta.norm():.4f}")
    mean_delta = mean_delta.to(device, dtype=dtype)

    print("Encoding source image...")
    source_pil = Image.open(args.source_image).convert("RGB").resize((args.width, args.height))
    with torch.no_grad():
        ref_tokens, ref_ids = encode_image_refs(vae, [source_pil])
    ref_tokens = ref_tokens.to(device, dtype=dtype)
    ref_ids = ref_ids.to(device)

    print(f"Generating at scales: {args.scales}")
    images = [source_pil]
    for scale in args.scales:
        ref_shifted = ref_tokens + scale * mean_delta
        if network is not None:
            network.set_lora_slider(scale)

        img = generate(
            transformer, vae, ref_shifted, ref_ids,
            prompt_txt, prompt_txt_ids,
            device, dtype, args.height, args.width, args.num_steps,
        )
        images.append(img)
        print(f"  scale={scale:+.1f} done")

    n = len(images)
    labels = ["Source"] + [f"Scale: {s:+.1f}" for s in args.scales]
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(np.array(img))
        ax.axis("off")
        ax.set_title(label, fontsize=14)
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
