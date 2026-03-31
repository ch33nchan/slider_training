"""
Compute the average gaze direction in VAE reference token space.

For each training pair (neg=left gaze, pos=right gaze), encodes both images
as Klein reference tokens via the VAE, then computes:
    delta = mean(ref_pos_tokens - ref_neg_tokens)  over all pairs

This "mean delta" is a direction vector in reference token space that points
from left-gaze toward right-gaze. Saved as gaze_delta.pt for use by
inference_ref_edit.py.

Usage:
    python compute_gaze_delta.py --device cuda:1
"""

import argparse
import os

import torch
from PIL import Image
from safetensors.torch import load_file

from models.autoencoder import AutoEncoder, AutoEncoderParams
from models.sampling import encode_image_refs


def load_vae(path: str, device: str, dtype=torch.bfloat16):
    params = AutoEncoderParams()
    vae = AutoEncoder(params)
    sd = load_file(path)
    vae.load_state_dict(sd)
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_path", default="models/klein-9b/ae.safetensors")
    parser.add_argument("--neg_dir", default="data/eye_gaze_v3/neg/")
    parser.add_argument("--pos_dir", default="data/eye_gaze_v3/pos/")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", default="gaze_delta.pt")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16

    print("Loading VAE...")
    vae = load_vae(args.vae_path, device, dtype)

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    neg_files = {
        os.path.splitext(f)[0]: os.path.join(args.neg_dir, f)
        for f in os.listdir(args.neg_dir)
        if os.path.splitext(f)[1].lower() in exts
    }
    pos_files = {
        os.path.splitext(f)[0]: os.path.join(args.pos_dir, f)
        for f in os.listdir(args.pos_dir)
        if os.path.splitext(f)[1].lower() in exts
    }
    common = sorted(set(neg_files) & set(pos_files))
    print(f"Found {len(common)} pairs")

    deltas = []
    for i, stem in enumerate(common):
        neg_pil = Image.open(neg_files[stem]).convert("RGB").resize((args.width, args.height))
        pos_pil = Image.open(pos_files[stem]).convert("RGB").resize((args.width, args.height))

        with torch.no_grad():
            ref_neg, _ = encode_image_refs(vae, [neg_pil])  # (1, N, C)
            ref_pos, _ = encode_image_refs(vae, [pos_pil])  # (1, N, C)

        delta = (ref_pos - ref_neg).float().cpu()
        deltas.append(delta)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(common)}")

    mean_delta = torch.stack(deltas).mean(dim=0)  # (1, N, C)
    print(f"Mean delta shape: {mean_delta.shape}")
    print(f"Mean delta norm:  {mean_delta.norm():.4f}")
    print(f"Mean delta max:   {mean_delta.abs().max():.4f}")

    torch.save(mean_delta, args.output)
    print(f"Saved to {args.output}")

    # Also save a masked version: keep only top-20% tokens by delta magnitude.
    # These are the tokens that change most between left and right gaze — i.e.
    # the eye/iris region — while zeroing out background/identity tokens.
    token_norms = mean_delta[0].norm(dim=-1)  # (N,)
    threshold = torch.quantile(token_norms, 0.80)
    mask = (token_norms >= threshold).float().unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
    masked_delta = mean_delta * mask
    masked_path = args.output.replace(".pt", "_masked.pt")
    torch.save(masked_delta, masked_path)
    n_kept = int(mask.sum().item())
    print(f"Masked delta: kept {n_kept}/{mean_delta.shape[1]} tokens (top 20% by magnitude)")
    print(f"Saved masked delta to {masked_path}")


if __name__ == "__main__":
    main()
