"""
Compute the mean gaze direction in transformer OUTPUT prediction space.

For each training pair, runs two frozen forward passes (neg ref, pos ref) with
the same noisy latent x_t, then accumulates:
    delta = mean(pos_pred - neg_pred)  over all pairs and timesteps

This gives a direction in the model's output prediction space (shape: 1024×128)
that points from left-gaze toward right-gaze — independent of identity.

At inference (see inference_output_steer.py), we add scale × mean_output_delta
to every denoising step's prediction, steering the trajectory toward the
desired gaze direction without any LoRA or weight modification.

Usage:
    python compute_output_delta.py --device cuda:1
"""

import argparse
import os
import random

import torch
from PIL import Image
from safetensors.torch import load_file

from models.autoencoder import AutoEncoder, AutoEncoderParams
from models.sampling import (
    batched_prc_img,
    batched_prc_txt,
    encode_image_refs,
    get_schedule,
)
from models.transformer import Flux2, Klein9BParams
from utils.text_encoder import encode_prompt, load_text_encoder
import torchvision.transforms as transforms


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_path", default="models/klein-9b/flux-2-klein-9b.safetensors")
    parser.add_argument("--vae_path", default="models/klein-9b/ae.safetensors")
    parser.add_argument("--te_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--neg_dir", default="data/eye_gaze_v3/neg/")
    parser.add_argument("--pos_dir", default="data/eye_gaze_v3/pos/")
    parser.add_argument("--neutral_dir", default="data/eye_gaze_v3/neutral/")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", default="output_delta.pt")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--samples_per_pair", type=int, default=3,
                        help="Number of random timesteps to sample per pair")
    parser.add_argument("--max_pairs", type=int, default=200)
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16

    print("Loading models...")
    transformer = load_transformer(args.transformer_path, device, dtype)
    vae = load_vae(args.vae_path, device, dtype)

    print("Loading text encoder...")
    text_encoder, tokenizer = load_text_encoder(args.te_path, device, dtype)
    prompt_embeds = encode_prompt(text_encoder, tokenizer, "a person", device)
    neutral_txt, neutral_txt_ids = batched_prc_txt(prompt_embeds)
    text_encoder.to("cpu")
    del text_encoder, tokenizer
    torch.cuda.empty_cache()

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
    neutral_files = {}
    if os.path.isdir(args.neutral_dir):
        neutral_files = {
            os.path.splitext(f)[0]: os.path.join(args.neutral_dir, f)
            for f in os.listdir(args.neutral_dir)
            if os.path.splitext(f)[1].lower() in exts
        }
    common = sorted(set(neg_files) & set(pos_files))[:args.max_pairs]
    print(f"Using {len(common)} pairs × {args.samples_per_pair} timesteps = "
          f"{len(common) * args.samples_per_pair} forward-pass pairs")

    deltas = []

    for i, stem in enumerate(common):
        neg_pil = Image.open(neg_files[stem]).convert("RGB").resize((args.width, args.height))
        pos_pil = Image.open(pos_files[stem]).convert("RGB").resize((args.width, args.height))

        # Use neutral as source for x_t if available, else neg
        src_path = neutral_files.get(stem, neg_files[stem])
        src_pil = Image.open(src_path).convert("RGB").resize((args.width, args.height))
        src_tensor = (transforms.ToTensor()(src_pil) * 2 - 1).unsqueeze(0).to(device, dtype=dtype)

        with torch.no_grad():
            src_latent = vae.encode(src_tensor)
            ref_neg, ref_neg_ids = encode_image_refs(vae, [neg_pil])
            ref_pos, ref_pos_ids = encode_image_refs(vae, [pos_pil])

        ref_neg = ref_neg.to(device, dtype=dtype)
        ref_neg_ids = ref_neg_ids.to(device)
        ref_pos = ref_pos.to(device, dtype=dtype)
        ref_pos_ids = ref_pos_ids.to(device)

        for _ in range(args.samples_per_pair):
            t = torch.rand(1, device=device, dtype=dtype)
            noise = torch.randn_like(src_latent)
            t_expand = t.view(1, 1, 1, 1)
            x_t = (1 - t_expand) * src_latent + t_expand * noise

            packed_x_t, x_ids = batched_prc_img(x_t)
            t_vec = torch.full((1,), t.item(), device=device, dtype=dtype)

            with torch.no_grad():
                x_neg = torch.cat([packed_x_t, ref_neg], dim=1)
                x_neg_ids = torch.cat([x_ids, ref_neg_ids], dim=1)
                neg_pred = transformer(
                    x=x_neg, x_ids=x_neg_ids, timesteps=t_vec,
                    ctx=neutral_txt, ctx_ids=neutral_txt_ids, guidance=None,
                )
                neg_pred = neg_pred[:, :packed_x_t.shape[1]]

                x_pos = torch.cat([packed_x_t, ref_pos], dim=1)
                x_pos_ids = torch.cat([x_ids, ref_pos_ids], dim=1)
                pos_pred = transformer(
                    x=x_pos, x_ids=x_pos_ids, timesteps=t_vec,
                    ctx=neutral_txt, ctx_ids=neutral_txt_ids, guidance=None,
                )
                pos_pred = pos_pred[:, :packed_x_t.shape[1]]

            delta = (pos_pred - neg_pred).float().cpu()
            deltas.append(delta)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(common)} pairs")

    mean_delta = torch.stack(deltas).mean(dim=0)  # (1, N, 128)
    print(f"Output delta shape: {mean_delta.shape}")
    print(f"Output delta norm:  {mean_delta.norm():.6f}")
    print(f"Output delta max:   {mean_delta.abs().max():.6f}")

    torch.save(mean_delta, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
