#!/usr/bin/env python3
"""
train_eye_gaze.py
-----------------
LECO-style concept slider training for eye gaze direction on FLUX.2-klein-9B.

Usage:
  # Train horizontal (left <-> right) LoRA
  python train_eye_gaze.py \
      --prompts_file configs/eye_gaze_horizontal.yaml \
      --name eye_gaze_horizontal \
      --steps 500 --rank 4 --alpha 1.0

  # Train vertical (up <-> down) LoRA
  python train_eye_gaze.py \
      --prompts_file configs/eye_gaze_vertical.yaml \
      --name eye_gaze_vertical \
      --steps 500 --rank 4 --alpha 1.0

Outputs are saved to:  ./models/<name>_rank<R>_alpha<A>/
  - step_<N>.safetensors   (checkpoints every --save_every steps)
  - last.safetensors       (final weights)
"""

import os
import sys
import math
import random
import argparse
import logging
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Path — point to existing flux-sliders/utils so we reuse lora.py as-is
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
FLUX_UTILS = REPO_ROOT / "flux-sliders" / "utils"
sys.path.insert(0, str(FLUX_UTILS.parent))   # adds  sliders/flux-sliders  to path

from utils.lora import LoRANetwork  # noqa: E402  (flux-sliders/utils/lora.py)

try:
    from diffusers import Flux2Transformer2DModel as FluxTransformerModel
except ImportError:
    from diffusers import FluxTransformer2DModel as FluxTransformerModel  # older diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

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
    p = argparse.ArgumentParser(description="Train Flux eye-gaze concept slider LoRA")

    # Core
    p.add_argument("--prompts_file", required=True,
                   help="YAML file with 30 prompt pairs (target / positive / unconditional)")
    p.add_argument("--model_id", default="black-forest-labs/FLUX.2-klein-9B",
                   help="HuggingFace model ID for FLUX.2-klein or any FLUX variant")
    p.add_argument("--name", default="eye_gaze",
                   help="Identifier used in the output directory name")
    p.add_argument("--output_dir", default="./models",
                   help="Root directory for saved checkpoints")

    # LoRA hyper-params
    p.add_argument("--rank", type=int, default=4,
                   help="LoRA rank (start with 4; increase to 8 if concept is weak)")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="LoRA alpha (effective scale = alpha/rank)")
    p.add_argument("--train_method", default="noxattn",
                   choices=["noxattn", "xattn", "full", "selfattn"],
                   help="Which layers to train. 'noxattn' trains all self-attn layers in Flux.")

    # Training hyper-params
    p.add_argument("--steps", type=int, default=500,
                   help="Total gradient update steps")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="AdamW learning rate")
    p.add_argument("--eta", type=float, default=1.0,
                   help="Guidance strength in the LECO loss target")
    p.add_argument("--resolution", type=int, default=512,
                   help="Spatial resolution for training latents (must be divisible by 16)")
    p.add_argument("--save_every", type=int, default=100,
                   help="Save a checkpoint every N steps")
    p.add_argument("--seed", type=int, default=42)

    # Hardware
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])

    return p.parse_args()


# ===========================================================================
# Helpers — latent packing / unpacking for Flux's sequence format
# ===========================================================================

def pack_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """
    [B, C, H, W]  →  [B, (H/p)*(W/p), C*p*p]

    Flux's MMDiT treats each 2×2 patch of the latent grid as one token.
    C=16 channels × patch 2×2 → 64-dim token.
    """
    B, C, H, W = latents.shape
    ph, pw = H // patch_size, W // patch_size
    x = latents.view(B, C, ph, patch_size, pw, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)           # B, ph, pw, C, p, p
    x = x.reshape(B, ph * pw, C * patch_size * patch_size)
    return x


def prepare_img_ids(latent_h: int, latent_w: int,
                    device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build the [seq_len, 3] positional-ID tensor that Flux's RoPE uses.
    Each row is (batch_idx=0, row, col).  latent_h/w are already /2 (packed).
    """
    ids = torch.zeros(latent_h, latent_w, 3, device=device, dtype=dtype)
    ids[..., 1] = ids[..., 1] + torch.arange(latent_h, device=device, dtype=dtype).unsqueeze(1)
    ids[..., 2] = ids[..., 2] + torch.arange(latent_w, device=device, dtype=dtype).unsqueeze(0)
    return ids.reshape(latent_h * latent_w, 3)


# ===========================================================================
# Text encoding — CLIP (pooled) + T5 (sequence)
# ===========================================================================

@torch.no_grad()
def encode_text(
    prompt: str,
    tokenizer_clip: CLIPTokenizer,
    encoder_clip: CLIPTextModel,
    tokenizer_t5: T5TokenizerFast,
    encoder_t5: T5EncoderModel,
    device: torch.device,
    dtype: torch.dtype,
    max_len_clip: int = 77,
    max_len_t5: int = 256,
):
    """
    Returns
    -------
    t5_emb   : [1, max_len_t5, 4096]  — T5 sequence embeddings (encoder_hidden_states)
    clip_emb : [1, 768]               — CLIP pooled embedding  (pooled_projections)
    txt_ids  : [max_len_t5, 3]        — all-zero positional IDs for text tokens
    """
    # --- CLIP ---
    tok_c = tokenizer_clip(
        [prompt],
        padding="max_length",
        max_length=max_len_clip,
        truncation=True,
        return_tensors="pt",
    )
    clip_out = encoder_clip(tok_c.input_ids.to(device))
    clip_emb = clip_out.pooler_output.to(dtype)           # [1, 768]

    # --- T5 ---
    tok_t = tokenizer_t5(
        [prompt],
        padding="max_length",
        max_length=max_len_t5,
        truncation=True,
        return_tensors="pt",
    )
    t5_out = encoder_t5(
        input_ids=tok_t.input_ids.to(device),
        attention_mask=tok_t.attention_mask.to(device),
    )
    t5_emb = t5_out.last_hidden_state.to(dtype)           # [1, max_len_t5, 4096]

    txt_ids = torch.zeros(t5_emb.shape[1], 3, device=device, dtype=dtype)
    return t5_emb, clip_emb, txt_ids


# ===========================================================================
# One transformer forward pass (with or without LoRA context)
# ===========================================================================

def forward_transformer(
    transformer,
    x_packed: torch.Tensor,       # [1, seq_len, 64]
    t5_emb: torch.Tensor,         # [1, 256, 4096]
    clip_emb: torch.Tensor,       # [1, 768]
    timestep_norm: torch.Tensor,  # [1]  in [0, 1]
    img_ids: torch.Tensor,        # [seq_len, 3]
    txt_ids: torch.Tensor,        # [256, 3]
) -> torch.Tensor:
    """
    Returns the velocity prediction from the Flux transformer.
    Call this inside a `with network:` block to enable the LoRA,
    or inside `torch.no_grad()` for baseline predictions.
    """
    out = transformer(
        hidden_states=x_packed,
        encoder_hidden_states=t5_emb,
        pooled_projections=clip_emb,
        timestep=timestep_norm,
        img_ids=img_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )
    return out[0]


# ===========================================================================
# Main training loop
# ===========================================================================

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    # ------------------------------------------------------------------
    # 1.  Load prompt pairs
    # ------------------------------------------------------------------
    with open(args.prompts_file) as f:
        prompts = yaml.safe_load(f)
    log.info(f"Loaded {len(prompts)} prompt pairs from {args.prompts_file}")

    # ------------------------------------------------------------------
    # 2.  Load models (text encoders + transformer only — no VAE needed)
    # ------------------------------------------------------------------
    log.info(f"Loading model components from  {args.model_id}  …")

    tokenizer_clip = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    encoder_clip = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device).eval()

    tokenizer_t5 = T5TokenizerFast.from_pretrained(args.model_id, subfolder="tokenizer_2")
    encoder_t5 = T5EncoderModel.from_pretrained(
        args.model_id, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device).eval()

    transformer = FluxTransformerModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=dtype
    ).to(device)
    transformer.requires_grad_(False)   # freeze all base weights

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )

    # ------------------------------------------------------------------
    # 3.  Build LoRA network on the transformer
    # ------------------------------------------------------------------
    log.info(f"Building LoRANetwork  rank={args.rank}  alpha={args.alpha}  method={args.train_method}")
    network = LoRANetwork(
        transformer,
        rank=args.rank,
        multiplier=0.0,       # start DISABLED — __enter__ turns it on
        alpha=args.alpha,
        train_method=args.train_method,
    ).to(device)

    if len(network.unet_loras) == 0:
        log.error(
            "LoRA found 0 modules! The transformer may use a different Attention class name.\n"
            "Open flux-sliders/utils/lora.py and add the Flux attention class name to\n"
            "UNET_TARGET_REPLACE_MODULE_TRANSFORMER (e.g. 'FluxAttention')."
        )
        raise RuntimeError("No LoRA modules created — see log above.")
    log.info(f"Created {len(network.unet_loras)} LoRA modules")

    # Only the lora_down weights are trainable (lora_up is frozen orthogonal)
    trainable_params = [p for p in network.parameters() if p.requires_grad]
    log.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    lr_sched  = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    # ------------------------------------------------------------------
    # 4.  Pre-compute latent geometry
    # ------------------------------------------------------------------
    latent_h = args.resolution // 8     # e.g. 512 → 64
    latent_w = args.resolution // 8
    packed_h = latent_h // 2            # after 2×2 spatial packing
    packed_w = latent_w // 2
    img_ids  = prepare_img_ids(packed_h, packed_w, device, dtype)  # [seq, 3]

    # Pre-fetch scheduler sigmas for noise level sampling
    scheduler.set_timesteps(1000, device=device)
    all_timesteps = scheduler.timesteps   # shape [1000], values in [0, 1000)

    # ------------------------------------------------------------------
    # 5.  Output directory
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir) / f"{args.name}_rank{args.rank}_alpha{args.alpha}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving checkpoints to  {out_dir}")

    # ------------------------------------------------------------------
    # 6.  Training loop
    # ------------------------------------------------------------------
    log.info(f"Starting training — {args.steps} steps")
    pbar = tqdm(range(args.steps), desc="training", dynamic_ncols=True)
    running_loss = 0.0

    for step in pbar:

        # --- 6a.  Sample one prompt config ---
        pc = random.choice(prompts)
        target_prompt       = pc["target"]
        positive_prompt     = pc["positive"]
        unconditional_prompt = pc["unconditional"]
        guidance_scale      = float(pc.get("guidance_scale", args.eta))

        # --- 6b.  Encode all three prompts ---
        t5_tgt,  clip_tgt,  txt_ids = encode_text(
            target_prompt, tokenizer_clip, encoder_clip,
            tokenizer_t5, encoder_t5, device, dtype,
        )
        t5_pos,  clip_pos,  _       = encode_text(
            positive_prompt, tokenizer_clip, encoder_clip,
            tokenizer_t5, encoder_t5, device, dtype,
        )
        t5_neg,  clip_neg,  _       = encode_text(
            unconditional_prompt, tokenizer_clip, encoder_clip,
            tokenizer_t5, encoder_t5, device, dtype,
        )

        # --- 6c.  Random timestep + random noise latent ---
        t_idx   = random.randint(0, len(all_timesteps) - 1)
        t_raw   = all_timesteps[t_idx]                        # scalar, e.g. 750
        t_norm  = (t_raw / 1000.0).unsqueeze(0).to(device, dtype)  # [1] in [0, 1]

        # Random noise as "input" — concept sliders train on the noise distribution
        x_noise = torch.randn(1, 16, latent_h, latent_w, device=device, dtype=dtype)
        x_packed = pack_latents(x_noise)   # [1, packed_h*packed_w, 64]

        # --- 6d.  Baseline velocity predictions — NO LoRA, NO grad ---
        with torch.no_grad():
            vel_tgt = forward_transformer(
                transformer, x_packed, t5_tgt, clip_tgt, t_norm, img_ids, txt_ids
            )
            vel_pos = forward_transformer(
                transformer, x_packed, t5_pos, clip_pos, t_norm, img_ids, txt_ids
            )
            vel_neg = forward_transformer(
                transformer, x_packed, t5_neg, clip_neg, t_norm, img_ids, txt_ids
            )

        # Ground-truth direction the slider should learn:
        # shift velocity from neutral toward (positive − negative)
        gt_vel = vel_tgt + guidance_scale * (vel_pos - vel_neg)

        # --- 6e.  Slider prediction — WITH LoRA, grads THROUGH lora_down ---
        network.set_lora_slider(scale=1.0)
        with network:
            vel_slider = forward_transformer(
                transformer, x_packed, t5_tgt, clip_tgt, t_norm, img_ids, txt_ids
            )

        # --- 6f.  LECO loss ---
        loss = F.mse_loss(vel_slider.float(), gt_vel.detach().float())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        lr_sched.step()

        running_loss += loss.item()

        # Progress display
        if step % 10 == 0:
            avg = running_loss / (10 if step > 0 else 1)
            running_loss = 0.0
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_sched.get_last_lr()[0]:.2e}")

        # --- 6g.  Checkpoint ---
        if (step + 1) % args.save_every == 0:
            ckpt = out_dir / f"step_{step + 1}.safetensors"
            network.save_weights(ckpt, dtype=torch.float16)
            log.info(f"  ✓ Checkpoint saved → {ckpt}")

    # ------------------------------------------------------------------
    # 7.  Final save
    # ------------------------------------------------------------------
    final = out_dir / "last.safetensors"
    network.save_weights(final, dtype=torch.float16)
    log.info(f"\nTraining complete.  Final weights → {final}")


# ===========================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)
