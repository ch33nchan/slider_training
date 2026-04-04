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
from PIL import Image, ImageFilter
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

try:
    import cv2
except ImportError:
    cv2 = None


def _largest_box(boxes):
    if len(boxes) == 0:
        return None
    return max(boxes, key=lambda b: int(b[2]) * int(b[3]))


def build_eye_mask(
    source_rgb: np.ndarray,
    eye_mask_scale: float = 1.8,
    eye_mask_blur: int = 31,
) -> np.ndarray:
    h, w = source_rgb.shape[:2]
    eye_boxes = []
    if cv2 is not None:
        gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(max(48, w // 8), max(48, h // 8)),
        )
        face = _largest_box(faces)
        if face is not None:
            fx, fy, fw, fh = map(int, face)
            roi_y_end = fy + int(fh * 0.65)
            roi = gray[fy:roi_y_end, fx:fx + fw]
            detected = eye_detector.detectMultiScale(
                roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(max(12, fw // 12), max(12, fh // 12)),
            )
            for ex, ey, ew, eh in detected:
                eye_boxes.append((fx + int(ex), fy + int(ey), int(ew), int(eh)))

    if len(eye_boxes) >= 2:
        eye_boxes = sorted(
            eye_boxes,
            key=lambda b: int(b[2]) * int(b[3]),
            reverse=True,
        )[:4]
        eye_boxes = sorted(eye_boxes, key=lambda b: b[0])[:2]
    else:
        cx = w // 2
        cy = int(h * 0.38)
        rx = int(w * 0.09)
        ry = int(h * 0.05)
        offset = int(w * 0.13)
        eye_boxes = [
            (cx - offset - rx, cy - ry, 2 * rx, 2 * ry),
            (cx + offset - rx, cy - ry, 2 * rx, 2 * ry),
        ]

    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=np.float32)
    for ex, ey, ew, eh in eye_boxes:
        cx = float(ex + ew * 0.5)
        cy = float(ey + eh * 0.5)
        rx = float(max(8, int(ew * 0.5 * eye_mask_scale)))
        ry = float(max(6, int(eh * 0.6 * eye_mask_scale)))
        ellipse = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
        mask[ellipse] = 1.0

    blur_radius = max(0.1, float(eye_mask_blur) / 6.0)
    mask_img = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.array(mask_img, dtype=np.float32) / 255.0


def blend_eyes_only(
    source_rgb: np.ndarray,
    edited_rgb: np.ndarray,
    eye_mask: np.ndarray,
    blend_strength: float,
) -> np.ndarray:
    alpha = np.clip(eye_mask * float(blend_strength), 0.0, 1.0)[..., None].astype(np.float32)
    source_f = source_rgb.astype(np.float32)
    edited_f = edited_rgb.astype(np.float32)
    blended = source_f * (1.0 - alpha) + edited_f * alpha
    return np.clip(blended + 0.5, 0, 255).astype(np.uint8)


def apply_eye_delta(
    source_rgb: np.ndarray,
    baseline_rgb: np.ndarray,
    target_rgb: np.ndarray,
    eye_mask: np.ndarray,
    delta_gain: float,
    blend_strength: float,
) -> np.ndarray:
    alpha = np.clip(eye_mask * float(blend_strength), 0.0, 1.0)[..., None].astype(np.float32)
    source_f = source_rgb.astype(np.float32)
    baseline_f = baseline_rgb.astype(np.float32)
    target_f = target_rgb.astype(np.float32)
    delta = (target_f - baseline_f) * float(delta_gain)
    edited = source_f + alpha * delta
    return np.clip(edited + 0.5, 0, 255).astype(np.uint8)


def resolve_sizes(image_path: str, requested_size: int) -> tuple[tuple[int, int], tuple[int, int]]:
    with Image.open(image_path).convert("RGB") as src:
        src_width, src_height = src.size

    if requested_size > 0:
        final_width, final_height = requested_size, requested_size
    else:
        final_width, final_height = src_width, src_height

    model_width = max(16, (final_width // 16) * 16)
    model_height = max(16, (final_height // 16) * 16)
    return (final_width, final_height), (model_width, model_height)


def resolve_model_size(width: int, height: int) -> tuple[int, int]:
    model_width = max(16, (width // 16) * 16)
    model_height = max(16, (height // 16) * 16)
    return model_width, model_height


def mask_bbox(mask: np.ndarray, threshold: float = 0.08) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > threshold)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    new_w = max(64, int(round(bw * padding)))
    new_h = max(64, int(round(bh * padding)))
    nx0 = max(0, int(round(cx - new_w / 2)))
    ny0 = max(0, int(round(cy - new_h / 2)))
    nx1 = min(image_width, nx0 + new_w)
    ny1 = min(image_height, ny0 + new_h)
    nx0 = max(0, nx1 - new_w)
    ny0 = max(0, ny1 - new_h)
    return nx0, ny0, nx1, ny1


def feather_mask(height: int, width: int, feather_ratio: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    dx = np.minimum(xx, width - 1 - xx).astype(np.float32)
    dy = np.minimum(yy, height - 1 - yy).astype(np.float32)
    d = np.minimum(dx, dy)
    feather = max(1.0, feather_ratio * float(min(width, height)))
    alpha = np.clip(d / feather, 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return alpha.astype(np.float32)


def paste_crop(
    base_rgb: np.ndarray,
    crop_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    alpha_mask: np.ndarray,
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    out = base_rgb.copy()
    base_crop = out[y0:y1, x0:x1].astype(np.float32)
    crop_f = crop_rgb.astype(np.float32)
    alpha = alpha_mask[..., None].astype(np.float32)
    blended = base_crop * (1.0 - alpha) + crop_f * alpha
    out[y0:y1, x0:x1] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    return out


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
    packed_source,
    noise_ids,
    txt,
    txt_ids,
    timesteps,
    ref_tokens,
    ref_ids,
    device,
    dtype,
    source_lock,
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

        # Keep identity fixed by conditioning on source-image reference tokens.
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
        if source_lock > 0:
            img = (1.0 - source_lock) * img + source_lock * packed_source

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
    parser.add_argument("--strength", type=float, default=0.25,
                        help="img2img strength 0-1: 0=no change, 1=full regen from noise")
    parser.add_argument("--scale_multiplier", type=float, default=1.8,
                        help="Multiply each requested slider scale by this factor for stronger gaze edits.")
    parser.add_argument("--flip_direction", action="store_true",
                        help="Flip sign of slider scales if learned direction is reversed.")
    parser.add_argument("--source_lock", type=float, default=0.06,
                        help="Per-step latent pull toward source (0-0.2 recommended) to preserve identity.")
    parser.add_argument("--eye_blend_mode", type=str, default="adaptive", choices=["none", "fixed", "adaptive"],
                        help="Blend generated eyes onto original source to preserve face identity.")
    parser.add_argument("--eye_blend_strength", type=float, default=0.9,
                        help="Base blend strength for eye-only compositing.")
    parser.add_argument("--eye_edit_mode", type=str, default="delta", choices=["absolute", "delta"],
                        help="absolute blends target eyes directly; delta applies target-minus-baseline eye change to the source.")
    parser.add_argument("--delta_gain", type=float, default=2.4,
                        help="Amplification for eye delta edits when eye_edit_mode=delta.")
    parser.add_argument("--eye_mask_scale", type=float, default=1.8)
    parser.add_argument("--eye_mask_blur", type=int, default=31)
    parser.add_argument("--keep_source_at_zero", action="store_true",
                        help="Use exact source image for scale=0 output.")
    parser.add_argument("--save_eye_mask", action="store_true")
    parser.add_argument("--size", type=int, default=0,
                        help="Square output size. Use 0 to preserve exact input resolution in final saves.")
    parser.add_argument("--crop_mode", type=str, default="none", choices=["none", "eyes"],
                        help="Run LoRA on the full image or only an eye-centered crop.")
    parser.add_argument("--crop_padding", type=float, default=4.0,
                        help="Expansion factor around the detected eye box when crop_mode=eyes.")
    parser.add_argument("--crop_threshold", type=float, default=0.08,
                        help="Mask threshold for deriving the eye crop.")
    parser.add_argument("--crop_feather", type=float, default=0.18,
                        help="Edge feather ratio when pasting crop edits back into the full image.")
    parser.add_argument("--left_scale", type=float, default=None,
                        help="Optional explicit left output scale. If set with right_scale, also saves left_full/right_full.")
    parser.add_argument("--right_scale", type=float, default=None,
                        help="Optional explicit right output scale. If set with left_scale, also saves left_full/right_full.")
    parser.add_argument("--output", type=str, default="outputs/inference_result.png")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = cfg.device
    dtype = torch.bfloat16
    final_size, _ = resolve_sizes(args.source_image, args.size)
    final_width, final_height = final_size

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
    source_full = Image.open(args.source_image).convert("RGB").resize(final_size, Image.LANCZOS)
    source_rgb_full = np.array(source_full)
    eye_mask_full = build_eye_mask(
        source_rgb_full,
        eye_mask_scale=args.eye_mask_scale,
        eye_mask_blur=args.eye_mask_blur,
    )

    crop_bbox = None
    if args.crop_mode == "eyes":
        raw_bbox = mask_bbox(eye_mask_full, threshold=args.crop_threshold)
        crop_bbox = expand_bbox(raw_bbox, final_width, final_height, args.crop_padding)
        x0, y0, x1, y1 = crop_bbox
        work_source = source_full.crop((x0, y0, x1, y1))
        work_mask = eye_mask_full[y0:y1, x0:x1]
        crop_alpha = feather_mask(y1 - y0, x1 - x0, args.crop_feather)
    else:
        work_source = source_full
        work_mask = eye_mask_full
        crop_alpha = None

    work_width, work_height = work_source.size
    model_width, model_height = resolve_model_size(work_width, work_height)
    model_size = (model_width, model_height)

    source_img = work_source.resize(model_size, Image.LANCZOS)
    work_source_rgb = np.array(work_source)
    ref_tokens, ref_ids = encode_image_refs(vae, [source_img], limit_pixels=model_width * model_height)
    ref_tokens = ref_tokens.to(device, dtype=dtype)
    ref_ids = ref_ids.to(device)

    # Encode source as starting latent for img2img
    source_tensor = default_prep(source_img, limit_pixels=model_width * model_height)
    if isinstance(source_tensor, list):
        source_tensor = source_tensor[0]
    with torch.no_grad():
        source_latent = vae.encode(source_tensor.unsqueeze(0).to(device, dtype=dtype))[0]
    packed_source, _ = batched_prc_img(source_latent.unsqueeze(0))

    max_abs_scale = max([abs(s) for s in args.scales]) if args.scales else 1.0
    eye_mask = work_mask if args.eye_blend_mode != "none" else None

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
    height_latent = model_height // 16
    width_latent = model_width // 16

    slider_images = []
    raw_slider_images = []
    print(f"\nGenerating images at scales: {args.scales}")

    for scale in args.scales:
        print(f"\nScale: {scale}")
        effective_scale = scale * args.scale_multiplier
        if args.flip_direction:
            effective_scale = -effective_scale
        network.set_lora_slider(effective_scale)

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
            packed_start, packed_source, noise_ids,
            txt, txt_ids,
            timesteps_use,
            ref_tokens, ref_ids,
            device, dtype, args.source_lock,
        )

        # Decode to PIL
        raw_output_pil = latent_to_pil(vae, packed_output, noise_ids, dtype).resize((work_width, work_height), Image.LANCZOS)
        raw_slider_images.append(raw_output_pil)
        slider_images.append(raw_output_pil)

    baseline_index = min(range(len(args.scales)), key=lambda i: abs(args.scales[i]))
    baseline_rgb = np.array(raw_slider_images[baseline_index]) if raw_slider_images else None

    for idx, scale in enumerate(args.scales):
        output_pil = raw_slider_images[idx]

        if args.keep_source_at_zero and abs(scale) < 1e-8:
            output_pil = source_full.copy()
        elif eye_mask is not None:
            if args.eye_blend_mode == "fixed":
                blend_strength = float(args.eye_blend_strength)
            else:
                scale_ratio = abs(scale) / max(max_abs_scale, 1e-6)
                blend_strength = float(args.eye_blend_strength) * (0.45 + 0.55 * scale_ratio)
            blend_strength = max(0.0, min(1.0, blend_strength))

            if args.eye_edit_mode == "delta" and baseline_rgb is not None:
                edited = apply_eye_delta(
                    source_rgb=work_source_rgb,
                    baseline_rgb=baseline_rgb,
                    target_rgb=np.array(raw_slider_images[idx]),
                    eye_mask=eye_mask,
                    delta_gain=args.delta_gain,
                    blend_strength=blend_strength,
                )
            else:
                edited = blend_eyes_only(
                    work_source_rgb,
                    np.array(raw_slider_images[idx]),
                    eye_mask,
                    blend_strength,
                )
            if args.crop_mode == "eyes" and crop_bbox is not None and crop_alpha is not None:
                pasted = paste_crop(
                    base_rgb=source_rgb_full,
                    crop_rgb=edited,
                    bbox=crop_bbox,
                    alpha_mask=crop_alpha,
                )
                output_pil = Image.fromarray(pasted)
            else:
                output_pil = Image.fromarray(edited)
        else:
            if args.crop_mode == "eyes" and crop_bbox is not None and crop_alpha is not None:
                pasted = paste_crop(
                    base_rgb=source_rgb_full,
                    crop_rgb=np.array(raw_slider_images[idx]),
                    bbox=crop_bbox,
                    alpha_mask=crop_alpha,
                )
                output_pil = Image.fromarray(pasted)
            else:
                output_pil = raw_slider_images[idx]

        slider_images[idx] = output_pil

    # -------------------------------------------------------------------------
    # Create stacked visualization
    # -------------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Include source image in visualization
    n = len(slider_images) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    # Source image
    axes[0].imshow(np.array(source_full))
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

    if eye_mask is not None and args.save_eye_mask:
        mask_path = output_path.parent / "eye_mask.png"
        mask_to_save = eye_mask_full if args.crop_mode == "eyes" else eye_mask
        mask_u8 = np.clip(mask_to_save * 255.0 + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(mask_u8, mode="L").save(str(mask_path))
        print(f"  Saved {mask_path}")

    # Save individual images
    for img, scale in zip(slider_images, args.scales):
        individual_path = output_path.parent / f"scale_{scale:+.1f}.png"
        img.save(str(individual_path))
        print(f"  Saved {individual_path}")

    if args.left_scale is not None and args.right_scale is not None:
        scale_to_img = {float(scale): img for scale, img in zip(args.scales, slider_images)}
        left_img = scale_to_img.get(float(args.left_scale))
        right_img = scale_to_img.get(float(args.right_scale))
        if left_img is not None:
            left_path = output_path.parent / "left_full.png"
            left_img.save(str(left_path))
            print(f"  Saved {left_path}")
        if right_img is not None:
            right_path = output_path.parent / "right_full.png"
            right_img.save(str(right_path))
            print(f"  Saved {right_path}")


if __name__ == "__main__":
    main()
