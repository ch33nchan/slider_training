from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.training_utils import compute_density_for_timestep_sampling
from omegaconf import OmegaConf
from PIL import Image
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast


@dataclass
class ImagePair:
    stem: str
    neg_path: str
    pos_path: str
    neutral_path: str
    mask_path: Optional[str]


@dataclass(frozen=True)
class GazeImageRecord:
    subject_id: str
    level: int
    file_path: str
    yaw_rad: float
    pitch_rad: float
    face_bbox: tuple[int, int, int, int]
    eye_bbox: tuple[int, int, int, int]
    image_width: int
    image_height: int


def resolve_flux_repo(flux_repo: Optional[str], project_root: Path) -> Path:
    candidates = []
    if flux_repo:
        candidates.append(Path(flux_repo))
    env_flux_repo = os.environ.get("FLUX_REPO")
    if env_flux_repo:
        candidates.append(Path(env_flux_repo))
    candidates.extend(
        [
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
    from flux_sliders.utils.custom_flux_pipeline import FluxPipeline  # type: ignore
    from flux_sliders.utils.lora import LoRANetwork  # type: ignore

    return FluxPipeline, LoRANetwork


def list_image_map(directory: str) -> dict[str, str]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return {
        Path(name).stem: str(Path(directory) / name)
        for name in os.listdir(directory)
        if Path(name).suffix.lower() in exts
    }


def load_pairs(cfg) -> list[ImagePair]:
    neg_map = list_image_map(cfg.neg_image_dir)
    pos_map = list_image_map(cfg.pos_image_dir)
    neutral_map = list_image_map(cfg.neutral_image_dir)
    mask_map = list_image_map(cfg.mask_dir) if cfg.get("mask_dir") else {}
    stems = sorted(set(neg_map) & set(pos_map) & set(neutral_map))
    return [
        ImagePair(
            stem=stem,
            neg_path=neg_map[stem],
            pos_path=pos_map[stem],
            neutral_path=neutral_map[stem],
            mask_path=mask_map.get(stem),
        )
        for stem in stems
    ]


def parse_xywh_bbox(value: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if isinstance(value, dict):
        x = value.get("x", value.get("left", 0))
        y = value.get("y", value.get("top", 0))
        w = value.get("w", value.get("width", width))
        h = value.get("h", value.get("height", height))
        value = [x, y, w, h]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        value = [0, 0, width, height]
    x, y, w, h = [int(round(float(v))) for v in value]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def load_gaze_records(cfg) -> list[GazeImageRecord]:
    metadata_path = Path(str(cfg.get("metadata_path", Path(str(cfg.data_root)) / "metadata.json"))).resolve()
    data_root = Path(str(cfg.data_root)).resolve()
    payload = json.loads(metadata_path.read_text())
    items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    records: list[GazeImageRecord] = []
    for item in items:
        image_width = int(item.get("image_width", cfg.width))
        image_height = int(item.get("image_height", cfg.height))
        records.append(
            GazeImageRecord(
                subject_id=str(item["subject_id"]),
                level=int(item["level"]),
                file_path=str((data_root / item["file_path"]).resolve()),
                yaw_rad=float(item["yaw_rad"]),
                pitch_rad=float(item.get("pitch_rad", 0.0)),
                face_bbox=parse_xywh_bbox(item.get("face_bbox"), image_width, image_height),
                eye_bbox=parse_xywh_bbox(item.get("eye_bbox"), image_width, image_height),
                image_width=image_width,
                image_height=image_height,
            )
        )
    return records


def group_gaze_records(records: list[GazeImageRecord]) -> dict[str, dict[int, list[GazeImageRecord]]]:
    grouped: dict[str, dict[int, list[GazeImageRecord]]] = {}
    for record in records:
        grouped.setdefault(record.subject_id, {}).setdefault(record.level, []).append(record)
    return grouped


def choose_source_record(level_map: dict[int, list[GazeImageRecord]], neutral_level: int = 0) -> Optional[GazeImageRecord]:
    if neutral_level in level_map:
        return random.choice(level_map[neutral_level])
    if not level_map:
        return None
    closest_level = min(level_map.keys(), key=lambda level: abs(level - neutral_level))
    return random.choice(level_map[closest_level])


def choose_level_pair(
    grouped_records: dict[str, dict[int, list[GazeImageRecord]]],
    training_levels: list[int],
    neutral_level: int = 0,
) -> tuple[GazeImageRecord, GazeImageRecord, int]:
    valid_levels = [level for level in training_levels if level != 0]
    if not valid_levels:
        raise ValueError("training_scale_levels must include at least one non-zero level.")

    candidate_subjects = [
        subject_id
        for subject_id, level_map in grouped_records.items()
        if choose_source_record(level_map, neutral_level) is not None and any(level in level_map for level in valid_levels)
    ]
    if not candidate_subjects:
        raise ValueError("No subjects have both a source level and a target level for metadata-driven gaze training.")

    subject_id = random.choice(candidate_subjects)
    level_map = grouped_records[subject_id]
    source_record = choose_source_record(level_map, neutral_level)
    available_levels = [level for level in valid_levels if level in level_map]
    target_level = random.choice(available_levels)
    target_record = random.choice(level_map[target_level])
    if source_record is None:
        raise ValueError("Failed to choose a source record.")
    return source_record, target_record, target_level


def load_rgb_tensor(path: str, width: int, height: int) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1) * 2.0 - 1.0
    return tensor, image


def load_mask_tensor(path: Optional[str], width: int, height: int) -> Optional[torch.Tensor]:
    if not path:
        return None
    mask = Image.open(path).convert("L").resize((width, height), Image.LANCZOS)
    array = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None, ...]


def mask_bbox(mask_tensor: torch.Tensor, threshold: float) -> tuple[int, int, int, int]:
    mask_2d = mask_tensor[0]
    ys, xs = torch.where(mask_2d > threshold)
    height, width = mask_2d.shape
    if xs.numel() == 0 or ys.numel() == 0:
        return 0, 0, width, height
    x0 = int(xs.min().item())
    y0 = int(ys.min().item())
    x1 = int(xs.max().item()) + 1
    y1 = int(ys.max().item()) + 1
    return x0, y0, x1, y1


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding: float,
    min_size: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    box_width = x1 - x0
    box_height = y1 - y0
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    new_width = max(min_size, int(round(box_width * padding)))
    new_height = max(min_size, int(round(box_height * padding)))
    nx0 = max(0, int(round(center_x - new_width / 2)))
    ny0 = max(0, int(round(center_y - new_height / 2)))
    nx1 = min(image_width, nx0 + new_width)
    ny1 = min(image_height, ny0 + new_height)
    nx0 = max(0, nx1 - new_width)
    ny0 = max(0, ny1 - new_height)
    return nx0, ny0, nx1, ny1


def jitter_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    shift_ratio: float,
    scale_ratio: float,
) -> tuple[int, int, int, int]:
    if shift_ratio <= 0.0 and scale_ratio <= 0.0:
        return bbox
    x0, y0, x1, y1 = bbox
    box_width = x1 - x0
    box_height = y1 - y0
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    center_x += random.uniform(-shift_ratio, shift_ratio) * box_width
    center_y += random.uniform(-shift_ratio, shift_ratio) * box_height
    scale = 1.0 + random.uniform(-scale_ratio, scale_ratio)
    new_width = max(32, int(round(box_width * scale)))
    new_height = max(32, int(round(box_height * scale)))
    nx0 = max(0, int(round(center_x - new_width / 2)))
    ny0 = max(0, int(round(center_y - new_height / 2)))
    nx1 = min(image_width, nx0 + new_width)
    ny1 = min(image_height, ny0 + new_height)
    nx0 = max(0, nx1 - new_width)
    ny0 = max(0, ny1 - new_height)
    return nx0, ny0, nx1, ny1


def crop_and_resize_tensor(
    tensor: torch.Tensor,
    bbox: tuple[int, int, int, int],
    out_height: int,
    out_width: int,
) -> torch.Tensor:
    x0, y0, x1, y1 = bbox
    cropped = tensor[:, y0:y1, x0:x1]
    if cropped.shape[-2:] == (out_height, out_width):
        return cropped
    resized = F.interpolate(
        cropped.unsqueeze(0),
        size=(out_height, out_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def xywh_to_xyxy(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    x0 = min(ax0, bx0)
    y0 = min(ay0, by0)
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    return x0, y0, x1 - x0, y1 - y0


def bbox_mask_tensor(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> torch.Tensor:
    x, y, w, h = bbox
    mask = torch.zeros((1, height, width), dtype=torch.float32)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    mask[:, y:y1, x:x1] = 1.0
    return mask


def scale_bbox_to_tensor(
    bbox: tuple[int, int, int, int],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    sx = dst_width / float(src_width)
    sy = dst_height / float(src_height)
    scaled = (
        int(round(x * sx)),
        int(round(y * sy)),
        max(1, int(round(w * sx))),
        max(1, int(round(h * sy))),
    )
    return parse_xywh_bbox(scaled, dst_width, dst_height)


def apply_saved_crop(
    image_tensor: torch.Tensor,
    bbox: tuple[int, int, int, int],
    src_width: int,
    src_height: int,
) -> torch.Tensor:
    _, _, dst_height, dst_width = image_tensor.shape
    x, y, w, h = scale_bbox_to_tensor(bbox, src_width, src_height, dst_width, dst_height)
    x1 = min(dst_width, x + w)
    y1 = min(dst_height, y + h)
    return image_tensor[:, :, y:y1, x:x1]


def maybe_crop_to_eye_region(
    neg_tensor: torch.Tensor,
    pos_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    mask_tensor: Optional[torch.Tensor],
    cfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if not bool(cfg.get("train_eye_crop_only", False)) or mask_tensor is None:
        return neg_tensor, pos_tensor, src_tensor, mask_tensor

    image_height, image_width = src_tensor.shape[-2:]
    bbox = mask_bbox(mask_tensor, threshold=float(cfg.get("eye_crop_threshold", 0.08)))
    bbox = expand_bbox(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        padding=float(cfg.get("eye_crop_padding", 4.0)),
        min_size=int(cfg.get("eye_crop_min_size", 192)),
    )
    bbox = jitter_bbox(
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        shift_ratio=float(cfg.get("eye_crop_shift_jitter", 0.03)),
        scale_ratio=float(cfg.get("eye_crop_scale_jitter", 0.08)),
    )

    out_height = int(cfg.height)
    out_width = int(cfg.width)
    neg_cropped = crop_and_resize_tensor(neg_tensor, bbox, out_height, out_width)
    pos_cropped = crop_and_resize_tensor(pos_tensor, bbox, out_height, out_width)
    src_cropped = crop_and_resize_tensor(src_tensor, bbox, out_height, out_width)
    mask_cropped = crop_and_resize_tensor(mask_tensor, bbox, out_height, out_width).clamp_(0.0, 1.0)
    return neg_cropped, pos_cropped, src_cropped, mask_cropped


def build_mask_weights(
    mask_tensor: Optional[torch.Tensor],
    latent_height: int,
    latent_width: int,
    eye_weight: float,
    non_eye_weight: float,
    device: str,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if mask_tensor is None:
        return None, None
    mask = mask_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    mask_latent = F.interpolate(mask, size=(latent_height, latent_width), mode="bilinear", align_corners=False).clamp_(0.0, 1.0)
    token_weights = non_eye_weight + (eye_weight - non_eye_weight) * mask_latent
    bg_weights = 1.0 - mask_latent
    return token_weights, bg_weights


def weighted_latent_mse(pred: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    diff = (pred.float() - target.float()) ** 2
    if weights is None:
        return diff.mean()
    weight_tensor = weights.to(device=diff.device, dtype=diff.dtype)
    weighted = diff * weight_tensor
    denom = (weight_tensor.sum() * diff.shape[1]).clamp_min(1e-6)
    return weighted.sum() / denom


def predict_clean_latent(noisy_latent: torch.Tensor, model_output: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return noisy_latent - sigma.to(device=model_output.device, dtype=model_output.dtype) * model_output


def decode_vae_latents(vae: AutoencoderKL, latent: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    scaled = latent / vae.config.scaling_factor + vae.config.shift_factor
    image = vae.decode(scaled.to(device=vae.device, dtype=dtype), return_dict=False)[0]
    return (image.float() / 2.0 + 0.5).clamp_(0.0, 1.0)


def grayscale_image(image_tensor: torch.Tensor) -> torch.Tensor:
    return (
        0.299 * image_tensor[:, 0:1] +
        0.587 * image_tensor[:, 1:2] +
        0.114 * image_tensor[:, 2:3]
    )


def split_eye_masks(mask_tensor: torch.Tensor, threshold: float) -> list[torch.Tensor]:
    mask_2d = mask_tensor[0]
    ys, xs = torch.where(mask_2d > threshold)
    if xs.numel() == 0 or ys.numel() == 0:
        return [mask_tensor]
    split_x = 0.5 * float(xs.min().item() + xs.max().item())
    width = mask_2d.shape[1]
    x_coords = torch.arange(width, device=mask_tensor.device, dtype=mask_tensor.dtype)[None, :]
    left_mask = mask_2d * (x_coords <= split_x)
    right_mask = mask_2d * (x_coords > split_x)
    masks: list[torch.Tensor] = []
    for candidate in (left_mask, right_mask):
        if float(candidate.sum().item()) > 1e-4:
            masks.append(candidate.unsqueeze(0))
    return masks if masks else [mask_tensor]


def horizontal_darkness_center(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    darkness_gamma: float,
) -> torch.Tensor:
    gray = grayscale_image(image_tensor)
    mask = mask_tensor.unsqueeze(0).to(device=image_tensor.device, dtype=image_tensor.dtype)
    darkness = (1.0 - gray).clamp_(0.0, 1.0).pow(darkness_gamma)
    weights = darkness * mask
    width = image_tensor.shape[-1]
    x_coords = torch.linspace(-1.0, 1.0, width, device=image_tensor.device, dtype=image_tensor.dtype).view(1, 1, 1, width)
    denom = weights.sum(dim=(2, 3)).clamp_min(1e-6)
    return (weights * x_coords).sum(dim=(2, 3)) / denom


def compute_gaze_geometry_loss(
    pred_image: torch.Tensor,
    target_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    mask_tensor: Optional[torch.Tensor],
    cfg,
) -> torch.Tensor:
    if mask_tensor is None:
        return pred_image.new_tensor(0.0)

    target_image = ((target_tensor.unsqueeze(0).to(device=pred_image.device, dtype=pred_image.dtype) / 2.0) + 0.5).clamp_(0.0, 1.0)
    src_image = ((src_tensor.unsqueeze(0).to(device=pred_image.device, dtype=pred_image.dtype) / 2.0) + 0.5).clamp_(0.0, 1.0)
    mask_device = mask_tensor.to(device=pred_image.device, dtype=pred_image.dtype)
    eye_masks = split_eye_masks(mask_device, threshold=float(cfg.get("gaze_center_threshold", 0.05)))
    darkness_gamma = float(cfg.get("gaze_darkness_gamma", 3.5))
    min_fraction = float(cfg.get("gaze_center_min_fraction", 0.65))
    direction_weight = float(cfg.get("gaze_direction_weight", 0.5))

    total_loss = pred_image.new_tensor(0.0)
    active_masks = 0
    for eye_mask in eye_masks:
        pred_center = horizontal_darkness_center(pred_image, eye_mask, darkness_gamma)
        target_center = horizontal_darkness_center(target_image, eye_mask, darkness_gamma)
        src_center = horizontal_darkness_center(src_image, eye_mask, darkness_gamma)
        pred_shift = pred_center - src_center
        target_shift = target_center - src_center
        shift_loss = F.l1_loss(pred_shift, target_shift)

        if direction_weight > 0.0:
            target_sign = torch.sign(target_shift.detach())
            min_signed_shift = target_shift.detach().abs() * min_fraction
            direction_loss = F.relu(min_signed_shift - pred_shift * target_sign).mean()
            shift_loss = shift_loss + direction_weight * direction_loss

        total_loss = total_loss + shift_loss
        active_masks += 1

    if active_masks == 0:
        return pred_image.new_tensor(0.0)
    return total_loss / float(active_masks)


def normalize_to(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ref_norm = reference.float().norm().clamp_min(1e-6)
    target_norm = target.float().norm().clamp_min(1e-6)
    return target * (ref_norm / target_norm)


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


def encode_vae_latents(vae: AutoencoderKL, image_tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    latent = vae.encode(image_tensor.unsqueeze(0).to(device=vae.device, dtype=dtype)).latent_dist.sample()
    latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
    return latent


def unpack_prediction(pred: torch.Tensor, flux_pipeline, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
    return flux_pipeline._unpack_latents(pred, height=height, width=width, vae_scale_factor=vae_scale_factor)


def build_guidance(transformer: FluxTransformer2DModel, batch_size: int, guidance_scale: float, device: str):
    if transformer.config.guidance_embeds:
        return torch.full([batch_size], guidance_scale, device=device, dtype=torch.float32)
    return None


def plot_loss(losses: list[float], save_path: Path) -> None:
    if not losses:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_pair_slider(
    config_path: str,
    flux_repo: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> None:
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    config_dir = Path(config_path).resolve().parent
    project_root = config_dir.parent.parent
    if not (project_root / "utils" / "l2cs_loss.py").exists():
        raise FileNotFoundError(f"Could not locate project root from config path: {config_path}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    flux_repo_path = resolve_flux_repo(flux_repo, project_root)
    FluxPipeline, LoRANetwork = import_flux_components(flux_repo_path)
    from utils.l2cs_loss import DifferentiableGazeLoss

    device = cfg.device
    dtype = torch.bfloat16
    metadata_mode = bool(cfg.get("data_root"))

    output_dir = Path(cfg.output_dir).resolve()
    save_dir = output_dir / f"flux-{cfg.slider_name}"
    weight_dir = save_dir / "weights"
    for directory in (save_dir, weight_dir):
        directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_dir / "config.yaml")

    tokenizer, tokenizer_2, text_encoder, text_encoder_2 = load_text_encoders(
        cfg.pretrained_model_name_or_path, device, dtype
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="scheduler",
        torch_dtype=dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)
    transformer = FluxTransformer2DModel.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(device)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.eval()
    transformer.eval()

    flux_pipeline = FluxPipeline(
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
    )

    prompt_embeds, pooled_prompt_embeds, text_ids = flux_pipeline.encode_prompt(
        prompt=cfg.prompt,
        prompt_2=cfg.prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )
    text_encoder.to("cpu")
    text_encoder_2.to("cpu")
    del text_encoder, text_encoder_2, tokenizer, tokenizer_2
    torch.cuda.empty_cache()

    grouped_records: Optional[dict[str, dict[int, list[GazeImageRecord]]]] = None
    training_levels = [int(level) for level in cfg.get("training_scale_levels", [-2, -1, 0, 1, 2])]
    neutral_level = int(cfg.get("neutral_level", 0))
    if metadata_mode:
        grouped_records = group_gaze_records(load_gaze_records(cfg))
        if not grouped_records:
            raise ValueError("No metadata-driven gaze records loaded from metadata.json.")
    else:
        pairs = load_pairs(cfg)
        if not pairs:
            raise ValueError("No matched neg/pos/neutral image triplets found.")

    network = LoRANetwork(
        transformer,
        rank=int(cfg.rank),
        multiplier=1.0,
        alpha=float(cfg.alpha),
        train_method=str(cfg.train_method),
        save_dir=save_dir,
    ).to(device, dtype=dtype)
    params = network.prepare_optimizer_params()
    optimizer = AdamW(params, lr=float(cfg.lr))
    gaze_loss_module = None
    if metadata_mode and float(cfg.get("gaze_geometry_weight", 0.0)) > 0.0:
        gaze_loss_module = DifferentiableGazeLoss(
            model_path=str(cfg.l2cs_model_path),
            device=device,
            num_bins=int(cfg.get("l2cs_num_bins", 90)),
            angle_min_deg=float(cfg.get("l2cs_angle_min_deg", -42.0)),
            angle_max_deg=float(cfg.get("l2cs_angle_max_deg", 42.0)),
        )

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
    guidance = build_guidance(transformer, 1, 3.5, device)
    losses: list[float] = []
    progress = tqdm(range(int(cfg.max_train_steps)), desc="FLUX Pair Training")

    for step in progress:
        latent_display = 0.0
        geometry_display = 0.0

        if metadata_mode:
            assert grouped_records is not None
            source_record, target_record, target_level = choose_level_pair(
                grouped_records=grouped_records,
                training_levels=training_levels,
                neutral_level=neutral_level,
            )
            src_tensor, _ = load_rgb_tensor(source_record.file_path, int(cfg.width), int(cfg.height))
            target_tensor, _ = load_rgb_tensor(target_record.file_path, int(cfg.width), int(cfg.height))
            mask_tensor = bbox_mask_tensor(
                union_bbox(source_record.eye_bbox, target_record.eye_bbox),
                int(cfg.width),
                int(cfg.height),
            )
            target_tensor, _, src_tensor, mask_tensor = maybe_crop_to_eye_region(
                neg_tensor=target_tensor,
                pos_tensor=target_tensor,
                src_tensor=src_tensor,
                mask_tensor=mask_tensor,
                cfg=cfg,
            )

            with torch.no_grad():
                src_latent = encode_vae_latents(vae, src_tensor, dtype)
                target_latent = encode_vae_latents(vae, target_tensor, dtype)
        else:
            pair = random.choice(pairs)
            neg_tensor, _ = load_rgb_tensor(pair.neg_path, int(cfg.width), int(cfg.height))
            pos_tensor, _ = load_rgb_tensor(pair.pos_path, int(cfg.width), int(cfg.height))
            src_tensor, _ = load_rgb_tensor(pair.neutral_path, int(cfg.width), int(cfg.height))
            mask_tensor = load_mask_tensor(pair.mask_path, int(cfg.width), int(cfg.height))
            neg_tensor, pos_tensor, src_tensor, mask_tensor = maybe_crop_to_eye_region(
                neg_tensor=neg_tensor,
                pos_tensor=pos_tensor,
                src_tensor=src_tensor,
                mask_tensor=mask_tensor,
                cfg=cfg,
            )

            with torch.no_grad():
                neg_latent = encode_vae_latents(vae, neg_tensor, dtype)
                pos_latent = encode_vae_latents(vae, pos_tensor, dtype)
                src_latent = encode_vae_latents(vae, src_tensor, dtype)

        u = compute_density_for_timestep_sampling(
            weighting_scheme=str(cfg.get("weighting_scheme", "none")),
            batch_size=1,
            logit_mean=float(cfg.get("logit_mean", 0.0)),
            logit_std=float(cfg.get("logit_std", 1.0)),
            mode_scale=float(cfg.get("mode_scale", 1.29)),
        )
        indices = (u * scheduler.config.num_train_timesteps).long()
        timesteps = scheduler.timesteps[indices].to(device=device)
        sigma = (timesteps.float() / float(scheduler.config.num_train_timesteps)).view(1, 1, 1, 1).to(device=device, dtype=dtype)

        noise = torch.randn_like(src_latent)
        x_src = (1.0 - sigma) * src_latent + sigma * noise
        packed_src = flux_pipeline._pack_latents(x_src, 1, x_src.shape[1], x_src.shape[2], x_src.shape[3])
        img_ids = flux_pipeline._prepare_latent_image_ids(1, x_src.shape[2], x_src.shape[3], device, dtype)

        eye_weights, bg_weights = build_mask_weights(
            mask_tensor,
            latent_height=x_src.shape[2],
            latent_width=x_src.shape[3],
            eye_weight=1.0,
            non_eye_weight=0.0,
            device=device,
        )

        if metadata_mode:
            x_target = (1.0 - sigma) * target_latent + sigma * noise
            packed_target = flux_pipeline._pack_latents(x_target, 1, x_target.shape[1], x_target.shape[2], x_target.shape[3])

            with torch.no_grad():
                src_pred_base = transformer(
                    hidden_states=packed_src,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]
                target_pred = transformer(
                    hidden_states=packed_target,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]
                src_pred_base = unpack_prediction(src_pred_base, flux_pipeline, int(cfg.height), int(cfg.width), vae_scale_factor)
                target_pred = unpack_prediction(target_pred, flux_pipeline, int(cfg.height), int(cfg.width), vae_scale_factor)

            optimizer.zero_grad(set_to_none=True)
            network.set_lora_slider(float(target_level))
            with network:
                pred = transformer(
                    hidden_states=packed_src,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]
            pred = unpack_prediction(pred, flux_pipeline, int(cfg.height), int(cfg.width), vae_scale_factor)

            target_delta = target_pred - src_pred_base
            pred_delta = pred - src_pred_base

            latent_target_loss = weighted_latent_mse(pred_delta, target_delta, None)
            eye_region_recon_loss = pred.new_tensor(0.0)
            if eye_weights is not None:
                eye_region_recon_loss = float(cfg.get("eye_region_weight", 0.0)) * weighted_latent_mse(pred_delta, target_delta, eye_weights)
            non_eye_preserve_loss = pred.new_tensor(0.0)
            if bg_weights is not None:
                non_eye_preserve_loss = float(cfg.get("non_eye_preserve_weight", cfg.get("bg_preserve_weight", 0.0))) * weighted_latent_mse(pred, src_pred_base, bg_weights)

            total_loss = latent_target_loss + eye_region_recon_loss + non_eye_preserve_loss
            geometry_loss = pred.new_tensor(0.0)
            if gaze_loss_module is not None:
                pred_clean = predict_clean_latent(x_src, pred, sigma)
                pred_image = decode_vae_latents(vae, pred_clean, dtype)
                face_crop = apply_saved_crop(
                    pred_image,
                    target_record.face_bbox,
                    target_record.image_width,
                    target_record.image_height,
                )
                target_yaw = torch.tensor([target_record.yaw_rad], device=device, dtype=pred.dtype)
                sigma_scalar = float(sigma.detach().float().mean().item())
                geometry_weight = float(cfg.get("gaze_geometry_weight", 0.0)) * max(0.0, 1.0 - sigma_scalar)
                geometry_loss = gaze_loss_module(face_crop, target_yaw)
                total_loss = total_loss + geometry_weight * geometry_loss

                direction_weight = float(cfg.get("gaze_direction_weight", 0.0))
                if direction_weight > 0.0:
                    pred_yaw = gaze_loss_module.predict_yaw(face_crop).to(device=target_yaw.device, dtype=target_yaw.dtype)
                    target_sign = torch.sign(target_yaw)
                    direction_margin = target_yaw.abs() * 0.5
                    direction_loss = F.relu(direction_margin - pred_yaw * target_sign).mean()
                    total_loss = total_loss + geometry_weight * direction_weight * direction_loss

            total_loss.backward()
            optimizer.step()

            loss_value = float(total_loss.detach().item())
            latent_display = float(latent_target_loss.detach().item())
            geometry_display = float(geometry_loss.detach().item())
        else:
            x_neg = (1.0 - sigma) * neg_latent + sigma * noise
            x_pos = (1.0 - sigma) * pos_latent + sigma * noise
            packed_neg = flux_pipeline._pack_latents(x_neg, 1, x_neg.shape[1], x_neg.shape[2], x_neg.shape[3])
            packed_pos = flux_pipeline._pack_latents(x_pos, 1, x_pos.shape[1], x_pos.shape[2], x_pos.shape[3])

            with torch.no_grad():
                src_pred_base = transformer(
                    hidden_states=packed_src,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]
                neg_pred = transformer(
                    hidden_states=packed_neg,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]
                pos_pred = transformer(
                    hidden_states=packed_pos,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]
                src_pred_base = unpack_prediction(src_pred_base, flux_pipeline, int(cfg.height), int(cfg.width), vae_scale_factor)
                neg_pred = unpack_prediction(neg_pred, flux_pipeline, int(cfg.height), int(cfg.width), vae_scale_factor)
                pos_pred = unpack_prediction(pos_pred, flux_pipeline, int(cfg.height), int(cfg.width), vae_scale_factor)

            direction = 0.5 * (pos_pred - neg_pred)
            gt_fwd = normalize_to(src_pred_base, src_pred_base + float(cfg.eta) * direction)
            gt_bwd = normalize_to(src_pred_base, src_pred_base - float(cfg.eta) * direction)
            target_fwd_delta = gt_fwd - src_pred_base
            target_bwd_delta = gt_bwd - src_pred_base

            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            bg_weight = float(cfg.get("bg_preserve_weight", 0.0))
            geometry_weight = float(cfg.get("gaze_geometry_weight", 0.0))

            network.set_lora_slider(1.0)
            with network:
                pred_fwd = transformer(
                    hidden_states=packed_src,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]
            pred_fwd = unpack_prediction(pred_fwd, flux_pipeline, int(cfg.height), int(cfg.width), vae_scale_factor)
            pred_fwd_delta = pred_fwd - src_pred_base
            loss_fwd = weighted_latent_mse(pred_fwd_delta, target_fwd_delta, eye_weights)
            if bg_weight > 0 and bg_weights is not None:
                loss_fwd = loss_fwd + bg_weight * weighted_latent_mse(pred_fwd, src_pred_base, bg_weights)
            geometry_fwd_value = 0.0
            if geometry_weight > 0.0 and mask_tensor is not None:
                pred_fwd_clean = predict_clean_latent(x_src, pred_fwd, sigma)
                pred_fwd_image = decode_vae_latents(vae, pred_fwd_clean, dtype)
                geometry_fwd = compute_gaze_geometry_loss(
                    pred_image=pred_fwd_image,
                    target_tensor=pos_tensor,
                    src_tensor=src_tensor,
                    mask_tensor=mask_tensor,
                    cfg=cfg,
                )
                loss_fwd = loss_fwd + geometry_weight * geometry_fwd
                geometry_fwd_value = float(geometry_fwd.detach().item())
            (0.5 * loss_fwd).backward()
            loss_value += 0.5 * float(loss_fwd.detach().item())

            network.set_lora_slider(-1.0)
            with network:
                pred_bwd = transformer(
                    hidden_states=packed_src,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]
            pred_bwd = unpack_prediction(pred_bwd, flux_pipeline, int(cfg.height), int(cfg.width), vae_scale_factor)
            pred_bwd_delta = pred_bwd - src_pred_base
            loss_bwd = weighted_latent_mse(pred_bwd_delta, target_bwd_delta, eye_weights)
            if bg_weight > 0 and bg_weights is not None:
                loss_bwd = loss_bwd + bg_weight * weighted_latent_mse(pred_bwd, src_pred_base, bg_weights)
            geometry_bwd_value = 0.0
            if geometry_weight > 0.0 and mask_tensor is not None:
                pred_bwd_clean = predict_clean_latent(x_src, pred_bwd, sigma)
                pred_bwd_image = decode_vae_latents(vae, pred_bwd_clean, dtype)
                geometry_bwd = compute_gaze_geometry_loss(
                    pred_image=pred_bwd_image,
                    target_tensor=neg_tensor,
                    src_tensor=src_tensor,
                    mask_tensor=mask_tensor,
                    cfg=cfg,
                )
                loss_bwd = loss_bwd + geometry_weight * geometry_bwd
                geometry_bwd_value = float(geometry_bwd.detach().item())
            (0.5 * loss_bwd).backward()
            loss_value += 0.5 * float(loss_bwd.detach().item())

            optimizer.step()
            latent_display = loss_value
            geometry_display = 0.5 * (geometry_fwd_value + geometry_bwd_value)

        losses.append(loss_value)
        progress.set_postfix(
            loss=f"{loss_value:.6f}",
            latent=f"{latent_display:.4f}",
            geo=f"{geometry_display:.4f}",
        )

        if (step + 1) % int(cfg.sample_every) == 0:
            weight_path = weight_dir / f"flux-{cfg.slider_name}_{step + 1:06d}.safetensors"
            network.save_weights(str(weight_path), dtype=dtype)
            plot_loss(losses, save_dir / "loss.png")

    latest_path = weight_dir / f"flux-{cfg.slider_name}_latest.safetensors"
    network.save_weights(str(latest_path), dtype=dtype)
    plot_loss(losses, save_dir / "loss.png")
