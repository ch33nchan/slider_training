#!/usr/bin/env python3
"""
train_leco.py
-------------
Image-conditioned LECO concept slider training for FLUX.2-klein-9B.

Unlike the original text-based LECO (which failed on distilled models), this
version uses REAL image pairs from LivePortrait as the training signal.

LECO target at each step:
  v_target = v_neutral + eta * (v_positive - v_negative)

Where v_neutral / v_positive / v_negative are velocity predictions from the
FROZEN base model on noisy latents of real portrait images.

Two LoRAs trained:
  gaze_horizontal : (gaze_x+N, gaze_x-N) pairs  →  left/right joystick
  gaze_vertical   : (gaze_y+N, gaze_y-N) pairs  →  up/down joystick

Launch one per GPU:
  CUDA_VISIBLE_DEVICES=2 python train_leco.py --axis horizontal --device cuda
  CUDA_VISIBLE_DEVICES=3 python train_leco.py --axis vertical   --device cuda
"""

import os
import sys
import math
import random
import argparse
import logging
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import numpy as np
from tqdm import tqdm
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Paths — reuse lora.py from flux-sliders
# ---------------------------------------------------------------------------
HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR  = HERE / "data" / "gaze_pairs"
sys.path.insert(0, str(REPO_ROOT / "flux-sliders"))

from utils.lora import LoRANetwork  # noqa: E402

try:
    from diffusers import Flux2KleinPipeline as FluxPipeline
except ImportError:
    from diffusers import FluxPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image-conditioned LECO gaze LoRA")

    p.add_argument("--axis", required=True, choices=["horizontal", "vertical"],
                   help="Which gaze axis to train")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-9B")
    p.add_argument("--data_dir", type=Path, default=DATA_DIR)
    p.add_argument("--output_dir", type=Path, default=HERE / "models")
    p.add_argument("--name", default="")

    # LoRA
    p.add_argument("--rank",         type=int,   default=8)
    p.add_argument("--alpha",        type=float, default=4.0)
    p.add_argument("--train_method", default="noxattn",
                   choices=["noxattn", "xattn", "full", "selfattn"])

    # Training
    p.add_argument("--steps",      type=int,   default=1500)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--eta",        type=float, default=5.0,
                   help="LECO guidance scale (how strongly to steer toward positive)")
    p.add_argument("--resolution", type=int,   default=512)
    p.add_argument("--save_every", type=int,   default=250)
    p.add_argument("--seed",       type=int,   default=42)

    # Hardware
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype",  default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    return p.parse_args()


# ===========================================================================
# Data loading
# ===========================================================================

# Pairs per axis: (positive_folder, negative_folder, weight)
AXIS_PAIRS = {
    "horizontal": [
        ("gaze_x+0.5", "gaze_x-0.5", 1.0),
        ("gaze_x+0.9", "gaze_x-0.9", 1.5),   # stronger pairs weighted more
        ("gaze_ul+0.7", "gaze_ur+0.7", 0.5),  # diagonals (H component)
        ("gaze_dl+0.7", "gaze_dr+0.7", 0.5),
    ],
    "vertical": [
        ("gaze_y+0.5", "gaze_y-0.5", 1.0),
        ("gaze_y+0.9", "gaze_y-0.9", 1.5),
        ("gaze_ul+0.7", "gaze_dl+0.7", 0.5),  # diagonals (V component)
        ("gaze_ur+0.7", "gaze_dr+0.7", 0.5),
    ],
}


def load_triplets(axis: str, data_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """
    Build list of (neutral, positive, negative) image triplets for the axis.
    One entry per image per pair config.
    """
    neutral_dir = data_dir / "neutral"
    stems = {p.stem: p for p in sorted(neutral_dir.glob("*.png")) + sorted(neutral_dir.glob("*.jpg"))}

    triplets = []
    for pos_folder, neg_folder, _ in AXIS_PAIRS[axis]:
        pos_dir = data_dir / pos_folder
        neg_dir = data_dir / neg_folder
        for stem, neutral_path in stems.items():
            pos_path = next(pos_dir.glob(f"{stem}.*"), None)
            neg_path = next(neg_dir.glob(f"{stem}.*"), None)
            if pos_path and neg_path:
                triplets.append((neutral_path, pos_path, neg_path))

    log.info(f"[{axis}] Loaded {len(triplets)} training triplets from {data_dir}")
    return triplets


def build_weighted_sampler(axis: str, data_dir: Path, triplets: List):
    """
    Build sampling weights matching AXIS_PAIRS weights.
    Triplets are ordered (pair_config × stems), so weight assignment is clean.
    """
    neutral_dir = data_dir / "neutral"
    n_stems = len(list(neutral_dir.glob("*.png")) + list(neutral_dir.glob("*.jpg")))
    weights = []
    for _, _, w in AXIS_PAIRS[axis]:
        weights.extend([w] * n_stems)
    # Trim to actual triplets found (some may be missing)
    weights = weights[:len(triplets)]
    total   = sum(weights)
    return [w / total for w in weights]


# ===========================================================================
# Image → latent encoding
# ===========================================================================

def preprocess_image(img_path: Path, resolution: int) -> torch.Tensor:
    """Load image, resize to square, return [1, 3, H, W] tensor in [-1, 1]."""
    img = Image.open(img_path).convert("RGB").resize(
        (resolution, resolution), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0   # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]


@torch.no_grad()
def encode_image(img_path: Path, vae, resolution: int,
                 device, dtype) -> torch.Tensor:
    """
    Encode image to FLUX.2-klein latent format: [1, 128, H//16, W//16]

    Pipeline:
      PIL → [1,3,H,W] → VAE.encode → [1,32,H//8,W//8]
           → scale → pixel_unshuffle(2) → [1,128,H//16,W//16]
    """
    pixel_values = preprocess_image(img_path, resolution).to(device, dtype)
    latents = vae.encode(pixel_values).latent_dist.sample()
    scale_factor = vae.config.get("scaling_factor", 0.18215)
    latents = latents * scale_factor
    # pixel_unshuffle: [B, C, H, W] → [B, C*4, H/2, W/2]
    latents = F.pixel_unshuffle(latents, 2)
    return latents   # [1, 128, res//16, res//16]


# ===========================================================================
# Latent packing + positional IDs (same as train_eye_gaze.py)
# ===========================================================================

def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] → [B, H*W, C]"""
    B, C, H, W = latents.shape
    return latents.reshape(B, C, H * W).permute(0, 2, 1)


def prepare_img_ids(batch_size: int, H: int, W: int,
                    device, dtype) -> torch.Tensor:
    h_idx = torch.arange(H, device=device, dtype=dtype)
    w_idx = torch.arange(W, device=device, dtype=dtype)
    grid_h, grid_w = torch.meshgrid(h_idx, w_idx, indexing="ij")
    ids = torch.zeros(batch_size, H * W, 4, device=device, dtype=dtype)
    ids[:, :, 2] = grid_h.reshape(-1).unsqueeze(0).expand(batch_size, -1)
    ids[:, :, 3] = grid_w.reshape(-1).unsqueeze(0).expand(batch_size, -1)
    return ids


# ===========================================================================
# Text encoding (neutral prompt — same for all images)
# ===========================================================================

@torch.no_grad()
def encode_text(prompt: str, pipe, device, dtype):
    import inspect
    ep_sig = inspect.signature(pipe.encode_prompt).parameters
    kwargs  = dict(prompt=prompt, device=device, num_images_per_prompt=1)
    if "prompt_2" in ep_sig:
        kwargs["prompt_2"] = None
    result = pipe.encode_prompt(**kwargs)
    if len(result) == 3:
        seq_emb, pooled_emb, txt_ids = result
    else:
        seq_emb, pooled_emb = result
        txt_ids = None
    seq_emb = seq_emb.to(dtype)
    if pooled_emb is not None:
        pooled_emb = pooled_emb.to(dtype)
    if txt_ids is None:
        txt_ids = torch.zeros(seq_emb.shape[0], seq_emb.shape[1], 4,
                              device=device, dtype=dtype)
    else:
        txt_ids = txt_ids.to(dtype)
    return seq_emb, pooled_emb, txt_ids


# ===========================================================================
# Transformer forward pass
# ===========================================================================

def forward_transformer(transformer, x_packed, seq_emb, pooled_emb,
                        timestep_norm, img_ids, txt_ids) -> torch.Tensor:
    import inspect
    sig = inspect.signature(transformer.forward).parameters
    kwargs: dict = {"hidden_states": x_packed, "return_dict": False}
    if "encoder_hidden_states" in sig:
        kwargs["encoder_hidden_states"] = seq_emb
    if "timestep" in sig:
        kwargs["timestep"] = timestep_norm
    if pooled_emb is not None and "pooled_projections" in sig:
        kwargs["pooled_projections"] = pooled_emb
    if img_ids is not None and "img_ids" in sig:
        kwargs["img_ids"] = img_ids
    if txt_ids is not None and "txt_ids" in sig:
        kwargs["txt_ids"] = txt_ids
    return transformer(**kwargs)[0]


# ===========================================================================
# Training loop
# ===========================================================================

def train(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device   = torch.device(args.device)
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype    = dtype_map[args.dtype]

    # ── 1. Load triplets ────────────────────────────────────────────────────
    triplets = load_triplets(args.axis, args.data_dir)
    weights  = build_weighted_sampler(args.axis, args.data_dir, triplets)
    if not triplets:
        raise RuntimeError(f"No triplets found in {args.data_dir}. "
                           "Run generate_gaze_pairs.py first.")

    # ── 2. Load pipeline ────────────────────────────────────────────────────
    log.info(f"Loading {args.model_id} …")
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    transformer = pipe.transformer
    transformer.requires_grad_(False)
    vae = pipe.vae
    vae.requires_grad_(False)
    vae.eval()
    log.info("Pipeline loaded — VAE kept for image encoding.")

    # ── 3. LoRA network ─────────────────────────────────────────────────────
    name = args.name or f"gaze_{args.axis}"
    log.info(f"Building LoRANetwork rank={args.rank} alpha={args.alpha} method={args.train_method}")
    network = LoRANetwork(
        transformer,
        rank=args.rank,
        multiplier=0.0,
        alpha=args.alpha,
        train_method=args.train_method,
    ).to(device=device, dtype=dtype)

    if len(network.unet_loras) == 0:
        raise RuntimeError(
            "0 LoRA modules created. Check UNET_TARGET_REPLACE_MODULE_TRANSFORMER "
            "in flux-sliders/utils/lora.py includes the Flux2 attention class names."
        )
    log.info(f"Created {len(network.unet_loras)} LoRA modules")

    trainable = [p for p in network.parameters() if p.requires_grad]
    log.info(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    # ── 4. Static geometry ──────────────────────────────────────────────────
    latent_ch = transformer.x_embedder.weight.shape[1]   # 128
    spatial   = args.resolution // 16                     # 512 → 32
    log.info(f"latent_ch={latent_ch}  spatial={spatial}×{spatial}")

    img_ids = prepare_img_ids(1, spatial, spatial, device, dtype)

    # Neutral text conditioning (same for all images)
    NEUTRAL_PROMPT = "portrait photograph"
    seq_emb, pooled_emb, txt_ids = encode_text(NEUTRAL_PROMPT, pipe, device, dtype)
    log.info(f"Text embedding shape: {seq_emb.shape}")

    # ── 5. Output dir ───────────────────────────────────────────────────────
    out_dir = args.output_dir / f"{name}_rank{args.rank}_alpha{args.alpha}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoints → {out_dir}")

    # ── 6. Training loop ─────────────────────────────────────────────────────
    log.info(f"Training {args.axis} LoRA — {args.steps} steps  eta={args.eta}")
    pbar         = tqdm(range(args.steps), desc=f"gaze_{args.axis}", dynamic_ncols=True)
    running_loss = 0.0

    for step in pbar:

        # Sample a triplet (weighted toward stronger gaze magnitudes)
        idx = random.choices(range(len(triplets)), weights=weights, k=1)[0]
        neutral_path, pos_path, neg_path = triplets[idx]

        # Encode images to latents [1, 128, 32, 32]
        with torch.no_grad():
            z_n   = encode_image(neutral_path, vae, args.resolution, device, dtype)
            z_pos = encode_image(pos_path,     vae, args.resolution, device, dtype)
            z_neg = encode_image(neg_path,     vae, args.resolution, device, dtype)

        # Flow-matching forward process: x_t = (1-t)*x_0 + t*eps
        t_norm = torch.tensor([random.uniform(0.02, 0.98)], device=device, dtype=dtype)
        eps    = torch.randn_like(z_n)

        z_n_t   = (1 - t_norm) * z_n   + t_norm * eps
        z_pos_t = (1 - t_norm) * z_pos + t_norm * eps   # same noise
        z_neg_t = (1 - t_norm) * z_neg + t_norm * eps

        # Pack for transformer
        x_n_packed   = pack_latents(z_n_t)
        x_pos_packed = pack_latents(z_pos_t)
        x_neg_packed = pack_latents(z_neg_t)

        # Base velocity predictions (frozen)
        with torch.no_grad():
            v_n   = forward_transformer(transformer, x_n_packed,   seq_emb, pooled_emb, t_norm, img_ids, txt_ids)
            v_pos = forward_transformer(transformer, x_pos_packed, seq_emb, pooled_emb, t_norm, img_ids, txt_ids)
            v_neg = forward_transformer(transformer, x_neg_packed, seq_emb, pooled_emb, t_norm, img_ids, txt_ids)

        # LECO target: steer neutral velocity toward positive gaze direction
        # target ≈ (z_n - eps) + eta*(z_pos - z_neg)  [analytically]
        v_target = v_n + args.eta * (v_pos - v_neg)

        # Slider velocity (LoRA active, gradients flow through lora_down)
        network.set_lora_slider(scale=1.0)
        with network:
            v_slider = forward_transformer(
                transformer, x_n_packed, seq_emb, pooled_emb, t_norm, img_ids, txt_ids)

        # LECO loss
        loss = F.mse_loss(v_slider.float(), v_target.detach().float())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if (step + 1) % 10 == 0:
            avg = running_loss / 10
            running_loss = 0.0
            pbar.set_postfix(
                loss=f"{avg:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                t=f"{t_norm.item():.2f}",
            )

        if (step + 1) % args.save_every == 0:
            ckpt = out_dir / f"step_{step + 1}.safetensors"
            network.save_weights(ckpt, dtype=torch.float16)
            log.info(f"✓ Checkpoint → {ckpt}")

    # Final save
    final = out_dir / "last.safetensors"
    network.save_weights(final, dtype=torch.float16)
    log.info(f"\nDone. Final weights → {final}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
