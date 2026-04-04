"""
LivePortrait + LoRA gaze pipeline.

Steps:
  1. LivePortrait warps input face gaze left/right (eyeball_direction_x)
  2. Warped face passed to Klein + LoRA slider for refinement

Usage:
  python pipeline_lora_gaze.py \
    --config config/eye_gaze_v7b.yaml \
    --lora_path outputs/eye_gaze_v7b/weights/slider_latest.safetensors \
    --source data/ffhq_source/face_0100.png \
    --gaze_strength 15 \
    --scales -5 -2.5 0 2.5 5 \
    --strength 0.4 \
    --output outputs/pipeline_lora_gaze/result.png
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
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

LIVEPORTRAIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../LivePortrait")
sys.path.insert(0, LIVEPORTRAIT_DIR)


def resolve_runtime_size(image_path: str, requested_size: int) -> tuple[int, int]:
    with Image.open(image_path).convert("RGB") as src:
        if requested_size > 0:
            return requested_size, requested_size
        width, height = src.size
    width = max(16, (width // 16) * 16)
    height = max(16, (height // 16) * 16)
    return width, height


def load_transformer(path, device, dtype=torch.bfloat16):
    params = Klein9BParams()
    transformer = Flux2(params)
    sd = load_file(path)
    transformer.load_state_dict(sd)
    transformer.to(device, dtype=dtype)
    transformer.requires_grad_(False)
    transformer.eval()
    return transformer


def load_vae(path, device, dtype=torch.bfloat16):
    params = AutoEncoderParams()
    vae = AutoEncoder(params)
    sd = load_file(path)
    vae.load_state_dict(sd)
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def build_liveportrait(device_id: int):
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.gradio_pipeline import GradioPipeline

    a = ArgumentConfig()
    a.device_id = device_id
    a.flag_use_half_precision = True
    a.flag_pasteback = True
    a.flag_do_crop = True
    inf = InferenceConfig()
    inf.device_id = device_id
    inf.flag_use_half_precision = True
    return GradioPipeline(inf, CropConfig(), a)


def warp_gaze(
    pipeline,
    img_path: str,
    eyeball_x: float,
    output_size: tuple[int, int],
) -> np.ndarray:
    eye_ratio, lip_ratio = pipeline.init_retargeting_image(
        retargeting_source_scale=2.3,
        source_eye_ratio=0.4,
        source_lip_ratio=0.0,
        input_image=img_path,
    )
    _, out = pipeline.execute_image_retargeting(
        input_eye_ratio=eye_ratio, input_lip_ratio=lip_ratio,
        input_head_pitch_variation=0.0, input_head_yaw_variation=0.0,
        input_head_roll_variation=0.0,
        mov_x=0.0, mov_y=0.0, mov_z=1.0,
        lip_variation_zero=0.0, lip_variation_one=0.0,
        lip_variation_two=0.0, lip_variation_three=0.0,
        smile=0.0, wink=0.0, eyebrow=0.0,
        eyeball_direction_x=float(eyeball_x),
        eyeball_direction_y=0.0,
        input_image=img_path,
        retargeting_source_scale=2.3,
        flag_stitching_retargeting_input=True,
        flag_do_crop_input_retargeting_image=True,
    )
    out = cv2.resize(out, output_size, interpolation=cv2.INTER_LANCZOS4)
    return out


@torch.no_grad()
def denoise_with_lora(
    transformer, network, vae, txt, txt_ids,
    ref_pil, device, dtype,
    height, width,
    lora_scale: float,
    num_steps: int = 28,
    strength: float = 0.4,
    seed: int = 42,
) -> Image.Image:
    """Klein img2img with LoRA — uses warped image as both reference and start latent."""
    ref_tokens, ref_ids = encode_image_refs(vae, [ref_pil])
    ref_tokens = ref_tokens.to(device, dtype=dtype)
    ref_ids = ref_ids.to(device)

    source_tensor = default_prep(ref_pil, limit_pixels=height * width)
    if isinstance(source_tensor, list):
        source_tensor = source_tensor[0]
    source_latent = vae.encode(source_tensor.unsqueeze(0).to(device, dtype=dtype))[0]
    packed_source, _ = batched_prc_img(source_latent.unsqueeze(0))

    height_latent = height // 16
    width_latent = width // 16
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(1, 128, height_latent, width_latent,
                        generator=generator, device=device, dtype=dtype)
    packed_noise, noise_ids = batched_prc_img(noise)
    timesteps = get_schedule(num_steps, packed_noise.shape[1])

    if strength >= 1.0:
        packed_start = packed_noise
        timesteps_use = timesteps
    else:
        start_idx = int((1.0 - strength) * (len(timesteps) - 1))
        t_start = timesteps[start_idx]
        packed_start = (1.0 - t_start) * packed_source + t_start * packed_noise
        timesteps_use = timesteps[start_idx:]

    network.set_lora_slider(lora_scale)

    img = packed_start
    for t_curr, t_prev in tqdm(zip(timesteps_use[:-1], timesteps_use[1:]),
                                total=len(timesteps_use) - 1,
                                desc=f"  LoRA scale={lora_scale}", leave=False):
        t_vec = torch.full((1,), t_curr, device=device, dtype=dtype)
        img_input = torch.cat([img, ref_tokens], dim=1)
        img_input_ids = torch.cat([noise_ids, ref_ids], dim=1)
        with network:
            pred = transformer(x=img_input, x_ids=img_input_ids, timesteps=t_vec,
                               ctx=txt, ctx_ids=txt_ids, guidance=None)
        pred = pred[:, :img.shape[1]]
        img = img + (t_prev - t_curr) * pred

    output_latent = torch.cat(scatter_ids(img, noise_ids)).squeeze(2)
    output_img = vae.decode(output_latent.to(dtype)).float()
    output_img = (output_img.clamp(-1, 1) + 1) / 2
    output_np = output_img[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((output_np * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--gaze_strength", type=float, default=15)
    parser.add_argument("--left_gaze", type=float, default=-18.0)
    parser.add_argument("--right_gaze", type=float, default=18.0)
    parser.add_argument("--left_scale", type=float, default=-8.0)
    parser.add_argument("--right_scale", type=float, default=8.0)
    parser.add_argument("--size", type=int, default=0,
                        help="Square output size. Use 0 to preserve source aspect ratio and near-full resolution.")
    parser.add_argument("--preview_size", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.4)
    parser.add_argument("--prompt", default="a person")
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/pipeline_lora_gaze/result.png")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = cfg.device
    dtype = torch.bfloat16
    runtime_width, runtime_height = resolve_runtime_size(args.source, args.size)
    output_size = (runtime_width, runtime_height)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_pil = Image.open(args.source).convert("RGB").resize(output_size, Image.LANCZOS)

    # ------------------------------------------------------------------
    # Step 1: LivePortrait warp
    # ------------------------------------------------------------------
    print("Running LivePortrait warp...")
    device_id = int(device.split(":")[-1])
    lp = build_liveportrait(device_id)
    left_gaze = args.left_gaze if args.left_gaze is not None else -abs(args.gaze_strength)
    right_gaze = args.right_gaze if args.right_gaze is not None else abs(args.gaze_strength)
    right_np = warp_gaze(lp, args.source, right_gaze, output_size)
    left_np = warp_gaze(lp, args.source, left_gaze, output_size)
    right_pil = Image.fromarray(right_np)
    left_pil  = Image.fromarray(left_np)
    right_pil.save(str(output_path.parent / "lp_right.png"))
    left_pil.save(str(output_path.parent / "lp_left.png"))
    del lp
    torch.cuda.empty_cache()
    print("  LivePortrait done")

    # ------------------------------------------------------------------
    # Step 2: Load Klein + LoRA
    # ------------------------------------------------------------------
    print("Loading Klein transformer...")
    transformer = load_transformer(cfg.transformer_path, device, dtype)

    print("Loading VAE...")
    vae = load_vae(cfg.vae_path, device, dtype)

    print("Loading text encoder...")
    text_encoder, tokenizer = load_text_encoder(cfg.te_path, device, dtype)
    prompt_embeds = encode_prompt(text_encoder, tokenizer, args.prompt, device)
    txt, txt_ids = batched_prc_txt(prompt_embeds)
    text_encoder.to("cpu")
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    print("Loading LoRA weights...")
    lora_sd = load_file(args.lora_path)
    lora_rank = cfg.rank
    for k, v in lora_sd.items():
        if "lora_A" in k or "lora_down" in k:
            lora_rank = v.shape[0]
            break
    network = LoRANetwork(
        transformer, rank=lora_rank, multiplier=1.0,
        alpha=cfg.alpha, train_method=cfg.train_method, save_dir=".",
    ).to(device, dtype=dtype)
    # Remap ai-toolkit key format if needed
    if any(k.startswith("diffusion_model.") for k in lora_sd):
        remapped = {}
        for k, v in lora_sd.items():
            k2 = k.replace("diffusion_model.", "")
            parts = k2.split(".")
            lora_suffix = parts[-2]
            weight_suffix = parts[-1]
            module_key = "_".join(parts[:-2])
            lora_dir = "lora_down" if lora_suffix == "lora_A" else "lora_up"
            remapped[f"{module_key}.{lora_dir}.{weight_suffix}"] = v
        alpha_val = torch.tensor(float(cfg.alpha))
        for k in list(remapped.keys()):
            if k.endswith(".lora_down.weight"):
                remapped[k.replace(".lora_down.weight", ".alpha")] = alpha_val
        network_keys = set(network.state_dict().keys())
        lora_sd = {k: v for k, v in remapped.items() if k in network_keys}
    network.load_state_dict(lora_sd, strict=False)

    # ------------------------------------------------------------------
    # Step 3: Run LoRA at each scale on the LP-warped right image
    # ------------------------------------------------------------------
    print(
        f"\nRunning LoRA refinement:"
        f" left_scale={args.left_scale:+.1f}, right_scale={args.right_scale:+.1f},"
        f" size={runtime_width}x{runtime_height}"
    )
    left_out = denoise_with_lora(
        transformer, network, vae, txt, txt_ids,
        left_pil, device, dtype,
        runtime_height, runtime_width,
        lora_scale=args.left_scale,
        num_steps=args.num_steps,
        strength=args.strength,
        seed=args.seed,
    )
    right_out = denoise_with_lora(
        transformer, network, vae, txt, txt_ids,
        right_pil, device, dtype,
        runtime_height, runtime_width,
        lora_scale=args.right_scale,
        num_steps=args.num_steps,
        strength=args.strength,
        seed=args.seed + 1,
    )
    left_out.save(str(output_path.parent / "lora_left_full.png"))
    right_out.save(str(output_path.parent / "lora_right_full.png"))
    left_out.save(str(output_path.parent / f"scale_{args.left_scale:+.1f}.png"))
    right_out.save(str(output_path.parent / f"scale_{args.right_scale:+.1f}.png"))

    # ------------------------------------------------------------------
    # Step 4: Visualization strip
    # ------------------------------------------------------------------
    preview_panels = [
        source_pil.resize((args.preview_size, args.preview_size), Image.LANCZOS),
        left_pil.resize((args.preview_size, args.preview_size), Image.LANCZOS),
        left_out.resize((args.preview_size, args.preview_size), Image.LANCZOS),
        right_pil.resize((args.preview_size, args.preview_size), Image.LANCZOS),
        right_out.resize((args.preview_size, args.preview_size), Image.LANCZOS),
    ]
    titles = [
        "Source",
        f"LP Left\n(gaze={left_gaze:+.1f})",
        f"LoRA Left\n(scale={args.left_scale:+.1f})",
        f"LP Right\n(gaze={right_gaze:+.1f})",
        f"LoRA Right\n(scale={args.right_scale:+.1f})",
    ]

    n = len(preview_panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    for ax, img, title in zip(axes, preview_panels, titles):
        ax.imshow(np.array(img))
        ax.axis("off")
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()

    source_pil.save(str(output_path.parent / "source_full.png"))
    print(f"\nSaved to {output_path.parent}/")
    print("  source_full.png")
    print("  lp_left.png")
    print("  lp_right.png")
    print("  lora_left_full.png")
    print("  lora_right_full.png")
    print("  result.png")


if __name__ == "__main__":
    main()
