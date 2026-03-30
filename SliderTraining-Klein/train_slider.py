"""
Directional LoRA Slider Trainer for FLUX Klein 9B (image-pair-driven).

Trains directional LoRA sliders using paired images to define the concept
direction. The direction comes from swapping reference images (neg vs pos)
while keeping text constant.

Per-step algorithm:
  1. Sample random pair (neg_image, pos_image) from dataset
  2. VAE encode neg_image -> neg_latent
  3. Encode reference tokens for both images
  4. Sample random timestep t in [0, 1]
  5. FlowMatch noise: x_t = (1-t) * neg_latent + t * noise
  6. Pack x_t to tokens
  7. TWO forward passes WITHOUT LoRA (same x_t + text, different ref tokens):
     - neg_pred = transformer(x_t + ref_neg, neutral_text)
     - pos_pred = transformer(x_t + ref_pos, neutral_text)
  8. Ground truth: gt = neg_pred + eta * (pos_pred - neg_pred), normalized
  9. ONE forward pass WITH LoRA:
     - lora_pred = transformer(x_t + ref_neg, neutral_text)
  10. Loss = MSE(lora_pred, gt)
  11. Backward + optimizer step

Usage:
  python train_slider.py --config config/smile_slider.yaml
"""

import argparse
import os
import random
from contextlib import ExitStack
from dataclasses import dataclass
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


@dataclass
class ImagePair:
    neg_path: str
    pos_path: str


def load_transformer(path: str, device: str, dtype=torch.bfloat16):
    """Load Klein 9B transformer from safetensors."""
    print(f"  Loading transformer from {path}")
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
    print(f"  Loading VAE from {path}")
    params = AutoEncoderParams()
    vae = AutoEncoder(params)
    sd = load_file(path)
    vae.load_state_dict(sd)
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def load_image_pairs(neg_dir: str, pos_dir: str):
    """Load paired images from neg/ and pos/ directories, matched by filename."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}

    neg_files = {
        os.path.splitext(f)[0]: os.path.join(neg_dir, f)
        for f in os.listdir(neg_dir)
        if os.path.splitext(f)[1].lower() in exts
    }
    pos_files = {
        os.path.splitext(f)[0]: os.path.join(pos_dir, f)
        for f in os.listdir(pos_dir)
        if os.path.splitext(f)[1].lower() in exts
    }

    # Match by stem (filename without extension)
    common_stems = sorted(set(neg_files.keys()) & set(pos_files.keys()))
    pairs = [ImagePair(neg_path=neg_files[s], pos_path=pos_files[s]) for s in common_stems]
    return pairs


def load_and_preprocess_image(img_path: str, height: int, width: int):
    """Load image, resize, return tensor [-1,1] and PIL image."""
    img = Image.open(img_path).convert("RGB").resize((width, height))
    tensor = transforms.ToTensor()(img) * 2 - 1
    return tensor, img


@torch.no_grad()
def generate_sample(
    transformer,
    vae,
    network,
    source_img_pil,
    prompt_txt,
    prompt_txt_ids,
    device,
    dtype,
    height,
    width,
    scales=(-5, -2.5, 0, 2.5, 5),
    num_steps=28,
    seed=42,
):
    """Generate sample images at different slider scales for visualization."""
    ref_tokens, ref_ids = encode_image_refs(vae, [source_img_pil])
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
                    ctx=prompt_txt, ctx_ids=prompt_txt_ids, guidance=None,
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
    """Save a grid of images with scale labels."""
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
    """Plot training loss and moving average."""
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
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Klein 9B directional LoRA slider")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = cfg.device
    dtype = torch.bfloat16

    # Create output directories
    output_dir = Path(cfg.get("output_dir", "outputs"))
    weight_dir = output_dir / "weights"
    sample_dir = output_dir / "samples"
    for d in [output_dir, weight_dir, sample_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(cfg, str(output_dir / "config.yaml"))

    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------
    print("Loading models...")
    transformer = load_transformer(cfg.transformer_path, device, dtype)
    vae = load_vae(cfg.vae_path, device, dtype)

    print("Loading text encoder...")
    text_encoder, tokenizer = load_text_encoder(cfg.te_path, device, dtype)

    # -------------------------------------------------------------------------
    # Pre-encode the single neutral prompt
    # -------------------------------------------------------------------------
    print("Pre-encoding prompt...")
    prompt_embeds = encode_prompt(text_encoder, tokenizer, cfg.prompt, device)
    neutral_txt, neutral_txt_ids = batched_prc_txt(prompt_embeds)

    # Offload text encoder to save ~16GB VRAM
    print("Offloading text encoder to CPU...")
    text_encoder.to("cpu")
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Load image pairs
    # -------------------------------------------------------------------------
    pairs = load_image_pairs(cfg.neg_image_dir, cfg.pos_image_dir)
    print(f"Found {len(pairs)} image pairs")
    assert len(pairs) > 0, (
        f"No matching image pairs found in {cfg.neg_image_dir} and {cfg.pos_image_dir}"
    )

    # Keep one neg image for periodic sampling visualization
    _, sample_img_pil = load_and_preprocess_image(pairs[0].neg_path, cfg.height, cfg.width)

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
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)

    # Count trainable parameters
    total_params = sum(p.numel() for pg in params for p in pg["params"])
    print(f"Trainable parameters: {total_params:,}")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    print(f"\nStarting training for {cfg.max_train_steps} steps...")
    print(f"  Rank: {cfg.rank}, Alpha: {cfg.alpha}, LR: {cfg.lr}, Eta: {cfg.eta}")
    print(f"  Image size: {cfg.width}x{cfg.height}")
    print(f"  Train method: {cfg.train_method}")
    print(f"  Prompt: \"{cfg.prompt}\"")
    print(f"  Image pairs: {len(pairs)}")

    losses = []
    scales = (-5, -2.5, 0, 2.5, 5)
    progress_bar = tqdm(range(cfg.max_train_steps), desc="Training")

    for step in progress_bar:
        # Sample random image pair
        pair = random.choice(pairs)
        neg_tensor, neg_pil = load_and_preprocess_image(pair.neg_path, cfg.height, cfg.width)
        pos_tensor, pos_pil = load_and_preprocess_image(pair.pos_path, cfg.height, cfg.width)

        neg_tensor = neg_tensor.unsqueeze(0).to(device, dtype=dtype)

        # VAE encode neg image for noisy input
        with torch.no_grad():
            neg_latent = vae.encode(neg_tensor)

        # Reference tokens for both images
        with torch.no_grad():
            ref_neg, ref_neg_ids = encode_image_refs(vae, [neg_pil])
            ref_neg = ref_neg.to(device, dtype=dtype)
            ref_neg_ids = ref_neg_ids.to(device)

            ref_pos, ref_pos_ids = encode_image_refs(vae, [pos_pil])
            ref_pos = ref_pos.to(device, dtype=dtype)
            ref_pos_ids = ref_pos_ids.to(device)

        # Sample random timestep t in [0, 1]
        t = torch.rand(1, device=device, dtype=dtype)

        # FlowMatch noise: x_t = (1-t) * neg_latent + t * noise
        noise = torch.randn_like(neg_latent)
        t_expand = t.view(1, 1, 1, 1)
        x_t = (1 - t_expand) * neg_latent + t_expand * noise

        # Pack x_t to tokens
        packed_x_t, x_ids = batched_prc_img(x_t)

        t_vec = torch.full((1,), t.item(), device=device, dtype=dtype)

        # ----- Two forward passes WITHOUT LoRA (different refs, same text) -----
        with torch.no_grad():
            # Neg reference pass
            x_neg = torch.cat([packed_x_t, ref_neg], dim=1)
            x_neg_ids = torch.cat([x_ids, ref_neg_ids], dim=1)
            neg_pred = transformer(
                x=x_neg, x_ids=x_neg_ids, timesteps=t_vec,
                ctx=neutral_txt, ctx_ids=neutral_txt_ids,
                guidance=None,
            )
            neg_pred = neg_pred[:, :packed_x_t.shape[1]]

            # Pos reference pass
            x_pos = torch.cat([packed_x_t, ref_pos], dim=1)
            x_pos_ids = torch.cat([x_ids, ref_pos_ids], dim=1)
            pos_pred = transformer(
                x=x_pos, x_ids=x_pos_ids, timesteps=t_vec,
                ctx=neutral_txt, ctx_ids=neutral_txt_ids,
                guidance=None,
            )
            pos_pred = pos_pred[:, :packed_x_t.shape[1]]

        # Ground truth: directional target
        gt = neg_pred + cfg.eta * (pos_pred - neg_pred)
        gt = (gt / gt.norm()) * pos_pred.norm()

        # ----- One forward pass WITH LoRA (using neg ref as baseline) -----
        with ExitStack() as stack:
            stack.enter_context(network)
            lora_pred = transformer(
                x=x_neg, x_ids=x_neg_ids, timesteps=t_vec,
                ctx=neutral_txt, ctx_ids=neutral_txt_ids,
                guidance=None,
            )
            lora_pred = lora_pred[:, :packed_x_t.shape[1]]

        # MSE loss
        loss = torch.mean(
            ((lora_pred.float() - gt.float()) ** 2).reshape(gt.shape[0], -1), 1
        ).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        # Periodic save and sample
        if step > 0 and step % cfg.sample_every == 0:
            network.save_weights(
                str(weight_dir / f"slider_{step:06d}.safetensors"), dtype=dtype
            )
            try:
                sample_images = generate_sample(
                    transformer, vae, network, sample_img_pil,
                    neutral_txt, neutral_txt_ids,
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
    # Save final outputs
    # -------------------------------------------------------------------------
    network.save_weights(str(weight_dir / "slider_latest.safetensors"), dtype=dtype)
    plot_loss(losses, output_dir / "loss.png")

    # Final sample
    try:
        sample_images = generate_sample(
            transformer, vae, network, sample_img_pil,
            neutral_txt, neutral_txt_ids,
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
