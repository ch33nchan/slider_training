from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from PIL import Image, ImageDraw, ImageFilter
from safetensors.torch import load_file
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

try:
    import cv2
except ImportError:
    cv2 = None


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--flux-repo", default=None)
    parser.add_argument("--input-dir", default=str(repo_root / "characters"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-path", default="/mnt/data1/models/base-models/black-forest-labs/FLUX.1-dev")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="portrait of a person")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--strength", type=float, default=0.22)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--scales", nargs="+", type=float, default=[-3.0, 0.0, 3.0])
    parser.add_argument("--eye-blend-mode", default="adaptive", choices=["none", "fixed", "adaptive"])
    parser.add_argument("--eye-blend-strength", type=float, default=0.72)
    parser.add_argument("--eye-edit-mode", default="delta", choices=["absolute", "delta"])
    parser.add_argument("--delta-gain", type=float, default=1.35)
    parser.add_argument("--eye-mask-scale", type=float, default=1.8)
    parser.add_argument("--eye-mask-blur", type=int, default=31)
    parser.add_argument("--crop-mode", default="eyes", choices=["none", "eyes"])
    parser.add_argument("--crop-padding", type=float, default=4.0)
    parser.add_argument("--crop-threshold", type=float, default=0.08)
    parser.add_argument("--crop-feather", type=float, default=0.18)
    parser.add_argument("--training-scale-max", type=float, default=2.0)
    parser.add_argument("--user-scale-max", type=float, default=5.0)
    parser.add_argument("--skip-slider-timestep-till", type=int, default=0)
    parser.add_argument("--keep-source-at-zero", action="store_true")
    parser.add_argument("--save-eye-mask", action="store_true")
    return parser.parse_args()


def list_images(input_dir: Path) -> List[Path]:
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def snap_to_model_size(width: int, height: int) -> tuple[int, int]:
    return max(16, (width // 16) * 16), max(16, (height // 16) * 16)


def _largest_box(boxes: Sequence[Sequence[int]]):
    if len(boxes) == 0:
        return None
    return max(boxes, key=lambda box: int(box[2]) * int(box[3]))


def _box_center(box: Sequence[int]) -> tuple[float, float]:
    x, y, w, h = map(float, box)
    return x + 0.5 * w, y + 0.5 * h


def _box_area(box: Sequence[int]) -> float:
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def _box_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    ax0, ay0, aw, ah = map(float, box_a)
    bx0, by0, bw, bh = map(float, box_b)
    ax1 = ax0 + aw
    ay1 = ay0 + ah
    bx1 = bx0 + bw
    by1 = by0 + bh
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = _box_area(box_a) + _box_area(box_b) - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def _dedupe_boxes(boxes: Sequence[Sequence[int]], iou_threshold: float = 0.18) -> list[tuple[int, int, int, int]]:
    deduped: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=_box_area, reverse=True):
        if any(_box_iou(box, kept) >= iou_threshold for kept in deduped):
            continue
        deduped.append(tuple(map(int, box)))
    return deduped


def _default_eye_boxes(image_width: int, image_height: int, face_box: Sequence[int] | None = None) -> list[tuple[int, int, int, int]]:
    if face_box is not None:
        fx, fy, fw, fh = map(int, face_box)
        eye_w = max(12, int(fw * 0.18))
        eye_h = max(8, int(fh * 0.10))
        eye_cy = fy + int(fh * 0.39)
        left_cx = fx + int(fw * 0.34)
        right_cx = fx + int(fw * 0.66)
        return [
            (left_cx - eye_w // 2, eye_cy - eye_h // 2, eye_w, eye_h),
            (right_cx - eye_w // 2, eye_cy - eye_h // 2, eye_w, eye_h),
        ]

    cx = image_width // 2
    cy = int(image_height * 0.38)
    rx = int(image_width * 0.09)
    ry = int(image_height * 0.05)
    offset = int(image_width * 0.13)
    return [
        (cx - offset - rx, cy - ry, 2 * rx, 2 * ry),
        (cx + offset - rx, cy - ry, 2 * rx, 2 * ry),
    ]


def _filter_eye_boxes(eye_boxes: Sequence[Sequence[int]], face_box: Sequence[int] | None) -> list[tuple[int, int, int, int]]:
    if face_box is None:
        return [tuple(map(int, box)) for box in eye_boxes]

    fx, fy, fw, fh = map(float, face_box)
    filtered: list[tuple[int, int, int, int]] = []
    for box in eye_boxes:
        ex, ey, ew, eh = map(float, box)
        cx, cy = _box_center(box)
        aspect = ew / max(eh, 1.0)
        if ew < fw * 0.07 or ew > fw * 0.32:
            continue
        if eh < fh * 0.05 or eh > fh * 0.20:
            continue
        if cx < fx + fw * 0.10 or cx > fx + fw * 0.90:
            continue
        if cy < fy + fh * 0.18 or cy > fy + fh * 0.52:
            continue
        if aspect < 0.6 or aspect > 3.5:
            continue
        filtered.append((int(ex), int(ey), int(ew), int(eh)))
    return _dedupe_boxes(filtered)


def _select_eye_pair(eye_boxes: Sequence[Sequence[int]], face_box: Sequence[int] | None):
    if len(eye_boxes) < 2:
        return None

    if face_box is None:
        face_width = max(float(max(box[0] + box[2] for box in eye_boxes) - min(box[0] for box in eye_boxes)), 1.0)
        face_height = max(float(max(box[1] + box[3] for box in eye_boxes) - min(box[1] for box in eye_boxes)), 1.0)
        face_center_x = sum(_box_center(box)[0] for box in eye_boxes) / float(len(eye_boxes))
    else:
        fx, _, fw, fh = map(float, face_box)
        face_width = max(fw, 1.0)
        face_height = max(fh, 1.0)
        face_center_x = fx + 0.5 * fw

    best_pair = None
    best_score = -1e9
    for i in range(len(eye_boxes)):
        for j in range(i + 1, len(eye_boxes)):
            left, right = sorted((eye_boxes[i], eye_boxes[j]), key=lambda box: box[0])
            left_cx, left_cy = _box_center(left)
            right_cx, right_cy = _box_center(right)
            separation = (right_cx - left_cx) / face_width
            vertical_gap = abs(right_cy - left_cy) / face_height
            overlap = _box_iou(left, right)
            area_ratio = min(_box_area(left), _box_area(right)) / max(_box_area(left), _box_area(right), 1.0)

            score = area_ratio * 3.0
            score -= vertical_gap * 6.0
            score -= abs(separation - 0.32) * 5.0
            score -= overlap * 8.0

            if left_cx >= face_center_x or right_cx <= face_center_x:
                score -= 2.5
            if separation < 0.16 or separation > 0.72:
                score -= 2.0
            if vertical_gap > 0.14:
                score -= 2.0

            if score > best_score:
                best_score = score
                best_pair = [left, right]

    if best_score < -0.25:
        return None
    return best_pair


def build_eye_mask(source_rgb: np.ndarray, eye_mask_scale: float = 1.8, eye_mask_blur: int = 31) -> np.ndarray:
    height, width = source_rgb.shape[:2]
    eye_boxes: list[tuple[int, int, int, int]] = []
    face = None
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
            minSize=(max(48, width // 8), max(48, height // 8)),
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

    eye_boxes = _filter_eye_boxes(eye_boxes, face)
    eye_pair = _select_eye_pair(eye_boxes[:6], face)
    if eye_pair is not None:
        eye_boxes = eye_pair
    else:
        eye_boxes = _default_eye_boxes(width, height, face)

    yy, xx = np.mgrid[0:height, 0:width]
    mask = np.zeros((height, width), dtype=np.float32)
    face_width = float(face[2]) if face is not None else float(width)
    face_height = float(face[3]) if face is not None else float(height)
    for ex, ey, ew, eh in eye_boxes:
        cx = float(ex + ew * 0.5)
        cy = float(ey + eh * 0.5)
        rx = float(max(8, int(ew * 0.5 * eye_mask_scale)))
        ry = float(max(6, int(eh * 0.6 * eye_mask_scale)))
        rx = min(rx, max(10.0, face_width * 0.16))
        ry = min(ry, max(8.0, face_height * 0.11))
        ellipse = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
        mask[ellipse] = 1.0

    blur_radius = max(0.1, float(eye_mask_blur) / 6.0)
    mask_img = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.array(mask_img, dtype=np.float32) / 255.0


def blend_eyes_only(source_rgb: np.ndarray, edited_rgb: np.ndarray, eye_mask: np.ndarray, blend_strength: float) -> np.ndarray:
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
    delta = target_f - baseline_f

    if cv2 is not None:
        sigma = max(2.0, 0.04 * float(min(delta.shape[0], delta.shape[1])))
        low_freq = cv2.GaussianBlur(
            delta,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        delta = delta - low_freq

    mask = np.clip(eye_mask.astype(np.float32), 0.0, 1.0)
    mask_sum = float(mask.sum())
    if mask_sum > 1e-6:
        masked_mean = (delta * mask[..., None]).sum(axis=(0, 1)) / mask_sum
        delta = delta - masked_mean
        masked_delta = delta[mask > 0.05]
        if masked_delta.size > 0:
            clip_value = float(np.quantile(np.abs(masked_delta), 0.98))
            clip_value = max(12.0, min(32.0, clip_value * 1.15))
            delta = np.clip(delta, -clip_value, clip_value)

    delta = delta * float(delta_gain)
    edited = source_f + alpha * delta
    return np.clip(edited + 0.5, 0, 255).astype(np.uint8)


def mask_bbox(mask: np.ndarray, threshold: float = 0.08) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > threshold)
    if len(xs) == 0 or len(ys) == 0:
        height, width = mask.shape[:2]
        return 0, 0, width, height
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
    distance = np.minimum(dx, dy)
    feather = max(1.0, feather_ratio * float(min(width, height)))
    alpha = np.clip(distance / feather, 0.0, 1.0)
    return (alpha * alpha * (3.0 - 2.0 * alpha)).astype(np.float32)


def paste_crop(
    base_rgb: np.ndarray,
    crop_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    alpha_mask: np.ndarray,
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    output = base_rgb.copy()
    base_crop = output[y0:y1, x0:x1].astype(np.float32)
    crop_f = crop_rgb.astype(np.float32)
    alpha = alpha_mask[..., None].astype(np.float32)
    blended = base_crop * (1.0 - alpha) + crop_f * alpha
    output[y0:y1, x0:x1] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    return output


def load_input_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    model_width, model_height = snap_to_model_size(width, height)
    return image.resize((model_width, model_height), Image.LANCZOS)


def user_scale_to_lora(scale: float, training_scale_max: float, user_scale_max: float) -> float:
    if user_scale_max <= 0:
        raise ValueError("user_scale_max must be positive.")
    clipped = max(-user_scale_max, min(user_scale_max, scale))
    return clipped * (training_scale_max / user_scale_max)


def resolve_flux_repo(flux_repo: str | None, project_root: Path) -> Path:
    candidates: list[Path] = []
    if flux_repo:
        candidates.append(Path(flux_repo))
    env_flux_repo = os.environ.get("FLUX_REPO")
    if env_flux_repo:
        candidates.append(Path(env_flux_repo))
    workspace_root = project_root.parent
    candidates.extend(
        [
            workspace_root / "flux-sliders-upstream",
            workspace_root / "flux-sliders",
            project_root / "flux-sliders-upstream",
            project_root / "flux-sliders",
        ]
    )
    for candidate in candidates:
        if (candidate / "flux_sliders" / "__init__.py").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not resolve an importable flux-sliders repo.")


def import_flux_components(flux_repo: Path):
    if str(flux_repo) not in sys.path:
        sys.path.insert(0, str(flux_repo))
    from flux_sliders.utils.custom_flux_pipeline import FluxPipeline, calculate_shift, retrieve_timesteps  # type: ignore
    from flux_sliders.utils.lora import LoRANetwork  # type: ignore

    return FluxPipeline, LoRANetwork, calculate_shift, retrieve_timesteps


def load_text_encoders(model_path: str, device: str, dtype: torch.dtype):
    text_encoder_config = PretrainedConfig.from_pretrained(model_path, subfolder="text_encoder", device_map=device)
    text_encoder_2_config = PretrainedConfig.from_pretrained(model_path, subfolder="text_encoder_2", device_map=device)

    from transformers import CLIPTextModel, T5EncoderModel

    if text_encoder_config.architectures[0] != "CLIPTextModel":
        raise ValueError(f"Unsupported text encoder: {text_encoder_config.architectures[0]}")
    if text_encoder_2_config.architectures[0] != "T5EncoderModel":
        raise ValueError(f"Unsupported text encoder 2: {text_encoder_2_config.architectures[0]}")

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=dtype)
    text_encoder.to(device)
    text_encoder_2.to(device)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    text_encoder.eval()
    text_encoder_2.eval()
    return tokenizer, tokenizer_2, text_encoder, text_encoder_2


def infer_lora_hyperparams(state_dict: dict[str, torch.Tensor]) -> tuple[int, float]:
    rank = None
    alpha = None
    for key, value in state_dict.items():
        if key.endswith(".lora_down.weight") or key.endswith(".lora_A.weight"):
            rank = int(value.shape[0])
            break
        if key.endswith(".lora_up.weight") or key.endswith(".lora_B.weight"):
            rank = int(value.shape[1])
            break
    for key, value in state_dict.items():
        if key.endswith(".alpha"):
            alpha = float(value.item())
            break
    if rank is None:
        raise ValueError("Could not infer LoRA rank from checkpoint.")
    if alpha is None:
        alpha = float(rank)
    return rank, alpha


def remap_peft_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    network_state_keys: set[str],
) -> dict[str, torch.Tensor]:
    if any(key in network_state_keys for key in state_dict):
        return state_dict

    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".alpha"):
            continue

        if key.endswith(".lora_A.weight"):
            prefix = key[: -len(".lora_A.weight")]
            suffix = ".lora_down.weight"
        elif key.endswith(".lora_B.weight"):
            prefix = key[: -len(".lora_B.weight")]
            suffix = ".lora_up.weight"
        else:
            continue

        if prefix.startswith("transformer."):
            prefix = prefix[len("transformer.") :]
        module_key = prefix.replace(".", "_")
        new_key = f"lora_unet_{module_key}{suffix}"
        if new_key in network_state_keys:
            remapped[new_key] = value

    if not remapped:
        return state_dict
    return remapped


def prepare_packed_source_latents(pipe, source_image: Image.Image, device: str, dtype: torch.dtype):
    source_tensor = pipe.image_processor.preprocess(source_image).to(device=device, dtype=dtype)
    latents = pipe.vae.encode(source_tensor).latent_dist.sample()
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    packed = pipe._pack_latents(
        latents,
        latents.shape[0],
        latents.shape[1],
        latents.shape[2],
        latents.shape[3],
    )
    return latents, packed


def generate_crop(
    pipe,
    network,
    calculate_shift_fn,
    retrieve_timesteps_fn,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    source_image: Image.Image,
    scale: float,
    training_scale_max: float,
    user_scale_max: float,
    guidance: float,
    steps: int,
    strength: float,
    seed: int,
    device: str,
    dtype: torch.dtype,
    skip_slider_timestep_till: int,
) -> Image.Image:
    model_source = load_input_image(source_image)
    model_width, model_height = model_source.size
    source_latents, packed_source = prepare_packed_source_latents(pipe, model_source, device=device, dtype=dtype)

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        source_latents.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    packed_noise = pipe._pack_latents(
        noise,
        noise.shape[0],
        noise.shape[1],
        noise.shape[2],
        noise.shape[3],
    )

    sigmas = np.linspace(1.0, 1 / steps, steps)
    image_seq_len = packed_noise.shape[1]
    mu = calculate_shift_fn(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps_fn(
        pipe.scheduler,
        steps,
        device,
        None,
        sigmas,
        mu=mu,
    )

    if strength >= 1.0:
        start_idx = 0
        packed_start = packed_noise
    else:
        start_idx = int((1.0 - strength) * (len(timesteps) - 1))
        start_idx = max(0, min(start_idx, len(timesteps) - 1))
        t_start = timesteps[start_idx].to(device=device, dtype=dtype)
        packed_start = (1.0 - t_start) * packed_source + t_start * packed_noise

    network.set_lora_slider(user_scale_to_lora(scale, training_scale_max, user_scale_max))
    with torch.no_grad():
        output = pipe(
            prompt=None,
            prompt_2=None,
            height=model_height,
            width=model_width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            latents=packed_start,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            output_type="pil",
            from_timestep=start_idx,
            till_timestep=None,
            network=network,
            skip_slider_timestep_till=skip_slider_timestep_till,
        )
    return output.images[0]


def save_strip(images: Iterable[Image.Image], labels: Iterable[str], output_path: Path) -> None:
    images = list(images)
    labels = list(labels)
    width, height = images[0].size
    label_height = 40
    canvas = Image.new("RGB", (width * len(images), height + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * width
        canvas.paste(image, (x, label_height))
        draw.text((x + 16, 10), label, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_root).resolve()
    lora_path = Path(args.lora_path).resolve()
    project_root = Path(__file__).resolve().parent.parent
    flux_repo_path = resolve_flux_repo(args.flux_repo, project_root)
    FluxPipeline, LoRANetwork, calculate_shift_fn, retrieve_timesteps_fn = import_flux_components(flux_repo_path)
    dtype = torch.bfloat16

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")
    if not lora_path.exists():
        raise FileNotFoundError(f"Missing LoRA path: {lora_path}")

    image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    tokenizer, tokenizer_2, text_encoder, text_encoder_2 = load_text_encoders(args.model_path, args.device, dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_path,
        subfolder="scheduler",
        torch_dtype=dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(args.device)
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(args.device)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.eval()
    transformer.eval()

    pipe = FluxPipeline(
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
    )
    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=args.prompt,
        prompt_2=args.prompt,
        device=args.device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    lora_state = load_file(str(lora_path))
    lora_rank, lora_alpha = infer_lora_hyperparams(lora_state)
    network = LoRANetwork(
        transformer,
        rank=lora_rank,
        multiplier=1.0,
        alpha=lora_alpha,
        train_method="xattn",
        save_dir=str(output_root),
    ).to(args.device, dtype=dtype)
    lora_state = remap_peft_lora_state_dict(lora_state, set(network.state_dict().keys()))
    network.load_state_dict(lora_state, strict=False)

    max_abs_scale = max(abs(scale) for scale in args.scales) if args.scales else 1.0

    for image_path in image_paths:
        source_full = Image.open(image_path).convert("RGB")
        source_rgb_full = np.array(source_full)
        sample_dir = output_root / image_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        source_full.save(sample_dir / "input.png")

        eye_mask_full = build_eye_mask(
            source_rgb_full,
            eye_mask_scale=args.eye_mask_scale,
            eye_mask_blur=args.eye_mask_blur,
        )

        crop_bbox = None
        crop_alpha = None
        if args.crop_mode == "eyes":
            raw_bbox = mask_bbox(eye_mask_full, threshold=args.crop_threshold)
            crop_bbox = expand_bbox(raw_bbox, source_full.width, source_full.height, args.crop_padding)
            x0, y0, x1, y1 = crop_bbox
            work_source = source_full.crop((x0, y0, x1, y1))
            work_mask = eye_mask_full[y0:y1, x0:x1]
            crop_alpha = feather_mask(y1 - y0, x1 - x0, args.crop_feather)
        else:
            work_source = source_full
            work_mask = eye_mask_full

        source_crop_rgb = np.array(work_source)
        model_source = load_input_image(work_source)
        baseline_crop = None
        result_images = [source_full]
        result_labels = ["input"]

        if args.save_eye_mask:
            Image.fromarray((eye_mask_full * 255.0).astype(np.uint8), mode="L").save(sample_dir / "eye_mask.png")

        for scale in args.scales:
            generated = generate_crop(
                pipe=pipe,
                network=network,
                calculate_shift_fn=calculate_shift_fn,
                retrieve_timesteps_fn=retrieve_timesteps_fn,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                source_image=model_source,
                scale=scale,
                training_scale_max=args.training_scale_max,
                user_scale_max=args.user_scale_max,
                guidance=args.guidance,
                steps=args.steps,
                strength=args.strength,
                seed=args.seed,
                device=args.device,
                dtype=dtype,
                skip_slider_timestep_till=args.skip_slider_timestep_till,
            )
            generated = generated.resize(work_source.size, Image.LANCZOS)
            generated_rgb = np.array(generated)

            if scale == 0.0:
                baseline_crop = generated_rgb.copy()

            if scale == 0.0 and args.keep_source_at_zero:
                final_crop_rgb = source_crop_rgb.copy()
            elif args.eye_blend_mode == "none":
                final_crop_rgb = generated_rgb
            else:
                if args.eye_blend_mode == "fixed":
                    blend_strength = float(args.eye_blend_strength)
                else:
                    scale_ratio = abs(scale) / max(max_abs_scale, 1e-6)
                    blend_strength = float(args.eye_blend_strength) * (0.45 + 0.55 * scale_ratio)

                if args.eye_edit_mode == "delta" and baseline_crop is not None:
                    final_crop_rgb = apply_eye_delta(
                        source_rgb=source_crop_rgb,
                        baseline_rgb=baseline_crop,
                        target_rgb=generated_rgb,
                        eye_mask=work_mask,
                        delta_gain=args.delta_gain,
                        blend_strength=blend_strength,
                    )
                else:
                    final_crop_rgb = blend_eyes_only(
                        source_rgb=source_crop_rgb,
                        edited_rgb=generated_rgb,
                        eye_mask=work_mask,
                        blend_strength=blend_strength,
                    )

            if args.crop_mode == "eyes" and crop_bbox is not None and crop_alpha is not None:
                final_full_rgb = paste_crop(source_rgb_full, final_crop_rgb, crop_bbox, crop_alpha)
            else:
                final_full_rgb = final_crop_rgb

            final_image = Image.fromarray(final_full_rgb)
            label = f"{scale:+.1f}"
            final_image.save(sample_dir / f"scale_{label}.png")
            result_images.append(final_image)
            result_labels.append(label)

        save_strip(result_images, result_labels, sample_dir / "result.png")


if __name__ == "__main__":
    main()
