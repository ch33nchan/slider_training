from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    project_root = config_dir.parent.parent.parent
    flux_repo_path = resolve_flux_repo(flux_repo, project_root)
    FluxPipeline, LoRANetwork = import_flux_components(flux_repo_path)

    device = cfg.device
    dtype = torch.bfloat16

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

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
    guidance = build_guidance(transformer, 1, 3.5, device)
    losses: list[float] = []
    progress = tqdm(range(int(cfg.max_train_steps)), desc="FLUX Pair Training")

    for step in progress:
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
        x_neg = (1.0 - sigma) * neg_latent + sigma * noise
        x_pos = (1.0 - sigma) * pos_latent + sigma * noise
        x_src = (1.0 - sigma) * src_latent + sigma * noise

        packed_neg = flux_pipeline._pack_latents(x_neg, 1, x_neg.shape[1], x_neg.shape[2], x_neg.shape[3])
        packed_pos = flux_pipeline._pack_latents(x_pos, 1, x_pos.shape[1], x_pos.shape[2], x_pos.shape[3])
        packed_src = flux_pipeline._pack_latents(x_src, 1, x_src.shape[1], x_src.shape[2], x_src.shape[3])
        img_ids = flux_pipeline._prepare_latent_image_ids(1, x_src.shape[2], x_src.shape[3], device, dtype)

        token_weights, bg_weights = build_mask_weights(
            mask_tensor,
            latent_height=x_src.shape[2],
            latent_width=x_src.shape[3],
            eye_weight=float(cfg.get("eye_region_weight", 1.0)),
            non_eye_weight=float(cfg.get("non_eye_region_weight", 1.0)),
            device=device,
        )

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
        loss_fwd = weighted_latent_mse(pred_fwd_delta, target_fwd_delta, token_weights)
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
        loss_bwd = weighted_latent_mse(pred_bwd_delta, target_bwd_delta, token_weights)
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

        losses.append(loss_value)
        progress.set_postfix(
            loss=f"{loss_value:.6f}",
            geo=f"{0.5 * (geometry_fwd_value + geometry_bwd_value):.4f}",
        )

        if (step + 1) % int(cfg.sample_every) == 0:
            weight_path = weight_dir / f"flux-{cfg.slider_name}_{step + 1:06d}.safetensors"
            network.save_weights(str(weight_path), dtype=dtype)
            plot_loss(losses, save_dir / "loss.png")

    latest_path = weight_dir / f"flux-{cfg.slider_name}_latest.safetensors"
    network.save_weights(str(latest_path), dtype=dtype)
    plot_loss(losses, save_dir / "loss.png")
