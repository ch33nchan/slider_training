"""
Gaze direction pipeline: LivePortrait warp + Klein 9B refinement.

Steps:
  1. LivePortrait warps input face gaze left/right (eyeball_direction_x)
  2. Warped face passed as Klein 9B reference -> generates high-quality output

No LoRA needed. Klein acts as a quality enhancer for the LivePortrait warp.

Usage:
  python pipeline_gaze.py \
    --config config/eye_gaze_v8.yaml \
    --source data/ffhq_source/face_0100.png \
    --gaze_strength 15 \
    --strength 0.4 \
    --output outputs/pipeline_gaze/result.png
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

# LivePortrait lives one level up
LIVEPORTRAIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../LivePortrait")
sys.path.insert(0, LIVEPORTRAIT_DIR)


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


def warp_gaze(pipeline, img_path: str, eyeball_x: float, size: int) -> np.ndarray:
    """Run LivePortrait gaze warp, return RGB numpy array."""
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
    # out is RGB numpy
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return out  # RGB uint8


@torch.no_grad()
def klein_refine(
    transformer, vae, txt, txt_ids,
    ref_pil: Image.Image,
    device, dtype,
    height: int, width: int,
    num_steps: int = 28,
    strength: float = 0.4,
    seed: int = 42,
) -> Image.Image:
    """
    Run Klein img2img with ref_pil as reference.
    strength: 0=copy reference, 1=full generation from noise.
    """
    # Encode reference
    ref_tokens, ref_ids = encode_image_refs(vae, [ref_pil])
    ref_tokens = ref_tokens.to(device, dtype=dtype)
    ref_ids = ref_ids.to(device)

    # Encode source latent (same image as reference for img2img start)
    source_tensor = default_prep(ref_pil, limit_pixels=height * width)
    if isinstance(source_tensor, list):
        source_tensor = source_tensor[0]
    source_latent = vae.encode(source_tensor.unsqueeze(0).to(device, dtype=dtype))[0]
    packed_source, _ = batched_prc_img(source_latent.unsqueeze(0))

    # Noise + schedule
    height_latent = height // 16
    width_latent = width // 16
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        1, 128, height_latent, width_latent,
        generator=generator, device=device, dtype=dtype,
    )
    packed_noise, noise_ids = batched_prc_img(noise)
    timesteps = get_schedule(num_steps, packed_noise.shape[1])

    # img2img: blend at t_start
    if strength >= 1.0:
        packed_start = packed_noise
        timesteps_use = timesteps
    else:
        start_idx = int((1.0 - strength) * (len(timesteps) - 1))
        t_start = timesteps[start_idx]
        packed_start = (1.0 - t_start) * packed_source + t_start * packed_noise
        timesteps_use = timesteps[start_idx:]

    # Denoise
    img = packed_start
    for t_curr, t_prev in tqdm(
        zip(timesteps_use[:-1], timesteps_use[1:]),
        total=len(timesteps_use) - 1,
        desc="  Klein denoising",
        leave=False,
    ):
        t_vec = torch.full((1,), t_curr, device=device, dtype=dtype)
        img_input = torch.cat([img, ref_tokens], dim=1)
        img_input_ids = torch.cat([noise_ids, ref_ids], dim=1)
        pred = transformer(
            x=img_input, x_ids=img_input_ids, timesteps=t_vec,
            ctx=txt, ctx_ids=txt_ids, guidance=None,
        )
        pred = pred[:, :img.shape[1]]
        img = img + (t_prev - t_curr) * pred

    # Decode
    output_latent = torch.cat(scatter_ids(img, noise_ids)).squeeze(2)
    output_img = vae.decode(output_latent.to(dtype)).float()
    output_img = (output_img.clamp(-1, 1) + 1) / 2
    output_np = output_img[0].permute(1, 2, 0).cpu().numpy()
    output_np = (output_np * 255).astype(np.uint8)
    return Image.fromarray(output_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True, help="Input face image")
    parser.add_argument("--gaze_strength", type=float, default=15,
                        help="LivePortrait eyeball_direction_x (8=subtle, 15=clear, 20=strong)")
    parser.add_argument("--strength", type=float, default=0.4,
                        help="Klein img2img strength (0=copy warp, 1=full regen)")
    parser.add_argument("--prompt", default="a person")
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/pipeline_gaze/result.png")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = cfg.device
    dtype = torch.bfloat16
    size = cfg.height  # assume square

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: LivePortrait gaze warp
    # -------------------------------------------------------------------------
    print("Building LivePortrait pipeline...")
    device_id = int(device.split(":")[-1])
    lp_pipeline = build_liveportrait(device_id)

    source_path = args.source
    print(f"Warping gaze (strength={args.gaze_strength})...")

    left_np  = warp_gaze(lp_pipeline, source_path, -args.gaze_strength, size)
    right_np = warp_gaze(lp_pipeline, source_path, +args.gaze_strength, size)

    left_pil  = Image.fromarray(left_np)
    right_pil = Image.fromarray(right_np)
    source_pil = Image.open(source_path).convert("RGB").resize((size, size))

    # Save raw LivePortrait outputs
    left_pil.save(str(output_path.parent / "lp_left.png"))
    right_pil.save(str(output_path.parent / "lp_right.png"))
    print("  Saved LivePortrait warps")

    # Free LivePortrait GPU memory
    del lp_pipeline
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Step 2: Klein refinement
    # -------------------------------------------------------------------------
    print("Loading Klein models...")
    transformer = load_transformer(cfg.transformer_path, device, dtype)
    vae = load_vae(cfg.vae_path, device, dtype)

    text_encoder, tokenizer = load_text_encoder(cfg.te_path, device, dtype)
    prompt_embeds = encode_prompt(text_encoder, tokenizer, args.prompt, device)
    txt, txt_ids = batched_prc_txt(prompt_embeds)
    text_encoder.to("cpu")
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

    print("Refining with Klein (left gaze)...")
    left_refined = klein_refine(
        transformer, vae, txt, txt_ids,
        left_pil, device, dtype,
        cfg.height, cfg.width,
        num_steps=args.num_steps,
        strength=args.strength,
        seed=args.seed,
    )

    print("Refining with Klein (right gaze)...")
    right_refined = klein_refine(
        transformer, vae, txt, txt_ids,
        right_pil, device, dtype,
        cfg.height, cfg.width,
        num_steps=args.num_steps,
        strength=args.strength,
        seed=args.seed,
    )

    # -------------------------------------------------------------------------
    # Step 3: Visualization
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    titles = [
        "Source",
        f"LP Left (s={args.gaze_strength})",
        f"Klein Left",
        f"LP Right (s={args.gaze_strength})",
        f"Klein Right",
    ]
    imgs = [source_pil, left_pil, left_refined, right_pil, right_refined]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(np.array(img))
        ax.axis("off")
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()

    # Save individual outputs
    left_refined.save(str(output_path.parent / "klein_left.png"))
    right_refined.save(str(output_path.parent / "klein_right.png"))

    print(f"\nDone! Saved to {output_path.parent}/")
    print(f"  result.png     — full comparison grid")
    print(f"  lp_left.png    — LivePortrait warp only")
    print(f"  lp_right.png   — LivePortrait warp only")
    print(f"  klein_left.png — Klein refined left gaze")
    print(f"  klein_right.png— Klein refined right gaze")


if __name__ == "__main__":
    main()
