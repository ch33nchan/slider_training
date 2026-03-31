"""
Text-based directional LoRA slider for Klein 9B.

Instead of image pairs, uses text triplets:
  target:   "a person"
  positive: "a person looking to the right"
  negative: "a person looking to the left"

Neutral face images provide identity context (ref tokens + source latent).
The directional signal is purely text-driven — avoids reference-override problem.

Per-step algorithm:
  1. Pick random neutral face -> ref_tokens + neutral_latent
  2. Sample t ~ U(0, 1) -> x_t = (1-t)*neutral_latent + t*noise
  3. Three no-LoRA passes (same x_t + ref, different text):
       target_pred = transformer(x_t + ref, target_text)
       pos_pred    = transformer(x_t + ref, positive_text)
       neg_pred    = transformer(x_t + ref, negative_text)
  4. gt = target_pred + eta * (pos_pred - neg_pred)
       gt = (gt / gt.norm()) * pos_pred.norm()
  5. LoRA forward (x_t + ref, target_text, scale=1)
  6. Loss = MSE(lora_pred, gt)
  7. Backward + optimize

Usage:
  python train_text_slider.py --config config/eye_gaze_text_v1.yaml
"""

import argparse
import os
import random
from contextlib import ExitStack
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

from models.transformer import Flux2, Klein9BParams
from models.autoencoder import AutoEncoder, AutoEncoderParams
from models.sampling import (
    batched_prc_img,
    batched_prc_txt,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from utils.text_encoder import load_text_encoder, encode_prompt
from utils.lora import LoRANetwork
from safetensors.torch import load_file


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


def load_neutral_images(neutral_dir: str, height: int, width: int):
    """Load all neutral face images from directory."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    paths = sorted(
        p for p in Path(neutral_dir).iterdir()
        if p.suffix.lower() in exts
    )
    assert paths, f"No images found in {neutral_dir}"
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((width, height))
        tensor = transforms.ToTensor()(img) * 2 - 1
        images.append((tensor, img))
    return images


@torch.no_grad()
def generate_sample(
    transformer, vae, network, ref_pil,
    target_txt, target_txt_ids,
    device, dtype, height, width,
    scales=(-5, -2.5, 0, 2.5, 5),
    num_steps=28,
    seed=42,
):
    ref_tokens, ref_ids = encode_image_refs(vae, [ref_pil])
    ref_tokens = ref_tokens.to(device, dtype=dtype)
    ref_ids = ref_ids.to(device)

    height_latent = height // 16
    width_latent = width // 16
    images = []

    for scale in scales:
        network.set_lora_slider(scale)
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
            with network:
                pred = transformer(
                    x=img_input, x_ids=img_input_ids, timesteps=t_vec,
                    ctx=target_txt, ctx_ids=target_txt_ids, guidance=None,
                )
            pred = pred[:, :img.shape[1]]
            img = img + (t_prev - t_curr) * pred

        output_latent = torch.cat(scatter_ids(img, noise_ids)).squeeze(2)
        output_img = vae.decode(output_latent.to(dtype)).float()
        output_img = (output_img.clamp(-1, 1) + 1) / 2
        output_np = output_img[0].permute(1, 2, 0).cpu().numpy()
        output_np = (output_np * 255).astype(np.uint8)
        images.append(Image.fromarray(output_np))

    return images


def save_sample_grid(images, scales, save_path):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for i, (img, scale) in enumerate(zip(images, scales)):
        axes[i].imshow(np.array(img))
        axes[i].axis("off")
        axes[i].set_title(f"Scale: {scale}", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss(losses, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(losses)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    window = min(20, max(1, len(losses) // 10))
    if len(losses) >= window:
        ma = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax2.plot(ma)
        ax2.set_title(f"Moving Average Loss (window={window})")
    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = cfg.device
    dtype = torch.bfloat16

    output_dir = Path(cfg.output_dir)
    weight_dir = output_dir / "weights"
    sample_dir = output_dir / "samples"
    for d in [output_dir, weight_dir, sample_dir]:
        d.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, str(output_dir / "config.yaml"))

    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------
    print("Loading transformer...")
    transformer = load_transformer(cfg.transformer_path, device, dtype)

    print("Loading VAE...")
    vae = load_vae(cfg.vae_path, device, dtype)

    print("Loading text encoder...")
    text_encoder, tokenizer = load_text_encoder(cfg.te_path, device, dtype)

    # -------------------------------------------------------------------------
    # Pre-encode all three prompts, then offload text encoder
    # -------------------------------------------------------------------------
    target_prompt  = cfg.get("target_prompt",   "a person")
    positive_prompt = cfg.get("positive_prompt", "a person looking to the right")
    negative_prompt = cfg.get("negative_prompt", "a person looking to the left")

    print(f"Encoding prompts:")
    print(f"  target:   {target_prompt}")
    print(f"  positive: {positive_prompt}")
    print(f"  negative: {negative_prompt}")

    all_embeds = encode_prompt(
        text_encoder, tokenizer,
        [target_prompt, positive_prompt, negative_prompt],
        device,
    )
    # all_embeds: (3, L, D)
    target_embeds  = all_embeds[0:1]   # (1, L, D)
    positive_embeds = all_embeds[1:2]
    negative_embeds = all_embeds[2:3]

    target_txt,   target_txt_ids   = batched_prc_txt(target_embeds)
    positive_txt, positive_txt_ids = batched_prc_txt(positive_embeds)
    negative_txt, negative_txt_ids = batched_prc_txt(negative_embeds)

    print("Offloading text encoder to CPU...")
    text_encoder.to("cpu")
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Load neutral face images
    # -------------------------------------------------------------------------
    neutral_images = load_neutral_images(cfg.neutral_image_dir, cfg.height, cfg.width)
    print(f"Loaded {len(neutral_images)} neutral faces from {cfg.neutral_image_dir}")

    # Use first neutral face for periodic sample visualization
    _, sample_ref_pil = neutral_images[0]

    # -------------------------------------------------------------------------
    # Create LoRA network
    # -------------------------------------------------------------------------
    print("Creating LoRA network...")
    network = LoRANetwork(
        transformer,
        rank=cfg.rank,
        multiplier=1.0,
        alpha=cfg.alpha,
        train_method=cfg.train_method,
        save_dir=str(output_dir),
    ).to(device, dtype=dtype)

    params = network.prepare_optimizer_params()
    total_params = sum(p.numel() for pg in params for p in pg["params"])
    print(f"Created LoRA for Klein 9B: trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(params, lr=cfg.lr)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    print(f"\nStarting training for {cfg.max_train_steps} steps...")
    print(f"  Rank: {cfg.rank}, Alpha: {cfg.alpha}, LR: {cfg.lr}, Eta: {cfg.eta}")
    print(f"  Image size: {cfg.width}x{cfg.height}")
    print(f"  Train method: {cfg.train_method}")
    print(f"  Neutral faces: {len(neutral_images)}")

    losses = []
    scales = (-5, -2.5, 0, 2.5, 5)
    progress_bar = tqdm(range(cfg.max_train_steps), desc="Training")

    for step in progress_bar:
        # ---- Pick random neutral face ----
        src_tensor, src_pil = random.choice(neutral_images)
        src_tensor_b = src_tensor.unsqueeze(0).to(device, dtype=dtype)

        # ---- VAE encode neutral latent ----
        with torch.no_grad():
            src_latent = vae.encode(src_tensor_b)

        # ---- Encode neutral face as reference tokens ----
        with torch.no_grad():
            ref_tokens, ref_ids = encode_image_refs(vae, [src_pil])
            ref_tokens = ref_tokens.to(device, dtype=dtype)
            ref_ids = ref_ids.to(device)

        # ---- FlowMatch noise: x_t = (1-t)*src + t*noise ----
        t = torch.rand(1, device=device, dtype=dtype)
        noise = torch.randn_like(src_latent)
        t_expand = t.view(1, 1, 1, 1)
        x_t = (1 - t_expand) * src_latent + t_expand * noise

        packed_x_t, x_ids = batched_prc_img(x_t)
        t_vec = torch.full((1,), t.item(), device=device, dtype=dtype)

        # Concatenate noise tokens + reference tokens
        x_with_ref = torch.cat([packed_x_t, ref_tokens], dim=1)
        x_with_ref_ids = torch.cat([x_ids, ref_ids], dim=1)

        # ---- Three no-LoRA forward passes (same x_t + ref, different text) ----
        with torch.no_grad():
            target_pred = transformer(
                x=x_with_ref, x_ids=x_with_ref_ids, timesteps=t_vec,
                ctx=target_txt, ctx_ids=target_txt_ids, guidance=None,
            )
            target_pred = target_pred[:, :packed_x_t.shape[1]]

            pos_pred = transformer(
                x=x_with_ref, x_ids=x_with_ref_ids, timesteps=t_vec,
                ctx=positive_txt, ctx_ids=positive_txt_ids, guidance=None,
            )
            pos_pred = pos_pred[:, :packed_x_t.shape[1]]

            neg_pred = transformer(
                x=x_with_ref, x_ids=x_with_ref_ids, timesteps=t_vec,
                ctx=negative_txt, ctx_ids=negative_txt_ids, guidance=None,
            )
            neg_pred = neg_pred[:, :packed_x_t.shape[1]]

        # ---- Directional target ----
        direction = pos_pred - neg_pred
        gt = target_pred + cfg.eta * direction
        gt = (gt / gt.norm()) * pos_pred.norm()

        # ---- LoRA forward pass (target text, scale=1) ----
        network.set_lora_slider(1.0)
        with ExitStack() as stack:
            stack.enter_context(network)
            lora_pred = transformer(
                x=x_with_ref, x_ids=x_with_ref_ids, timesteps=t_vec,
                ctx=target_txt, ctx_ids=target_txt_ids, guidance=None,
            )
            lora_pred = lora_pred[:, :packed_x_t.shape[1]]

        # ---- Loss ----
        loss = torch.mean(
            ((lora_pred.float() - gt.float()) ** 2).reshape(gt.shape[0], -1), 1
        ).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        # ---- Periodic checkpoint + sample ----
        if step > 0 and step % cfg.sample_every == 0:
            network.save_weights(
                str(weight_dir / f"slider_{step:06d}.safetensors"), dtype=dtype
            )
            try:
                sample_images = generate_sample(
                    transformer, vae, network, sample_ref_pil,
                    target_txt, target_txt_ids,
                    device, dtype, cfg.height, cfg.width, scales=scales,
                )
                save_sample_grid(
                    sample_images, scales,
                    sample_dir / f"step_{step:06d}.png",
                )
                print(f"\n  Saved sample at step {step}")
            except Exception as e:
                print(f"\n  Warning: sample generation failed at step {step}: {e}")
            plot_loss(losses, output_dir / "loss.png")

    # -------------------------------------------------------------------------
    # Final save
    # -------------------------------------------------------------------------
    network.save_weights(str(weight_dir / "slider_latest.safetensors"), dtype=dtype)
    plot_loss(losses, output_dir / "loss.png")

    try:
        sample_images = generate_sample(
            transformer, vae, network, sample_ref_pil,
            target_txt, target_txt_ids,
            device, dtype, cfg.height, cfg.width, scales=scales,
        )
        save_sample_grid(sample_images, scales, sample_dir / "final.png")
    except Exception as e:
        print(f"Warning: final sample generation failed: {e}")

    print(f"\nTraining complete!")
    print(f"  Weights: {weight_dir}")
    print(f"  Samples: {sample_dir}")
    print(f"  Loss plot: {output_dir / 'loss.png'}")


if __name__ == "__main__":
    main()
