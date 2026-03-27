#!/usr/bin/env python3
"""
test_gaze_cli.py — direct CLI test for eye gaze LoRA.

Generates 3 side-by-side outputs for the same input image:
  1. scale=0   (no LoRA  — baseline)
  2. scale=-15 (hard LEFT)
  3. scale=+15 (hard RIGHT)

Usage:
  python test_gaze_cli.py --input path/to/portrait.jpg
  python test_gaze_cli.py   # uses a synthetically generated face
"""

import sys, argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# ── resolve paths ────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
LORA_UTILS = ROOT.parent / "flux-sliders"
sys.path.insert(0, str(LORA_UTILS))

from utils.lora import LoRANetwork          # noqa
from safetensors.torch import load_file     # noqa

try:
    from diffusers import Flux2KleinPipeline
except ImportError:
    raise SystemExit("diffusers Flux2KleinPipeline not found — update diffusers")

# ── helpers ──────────────────────────────────────────────────────────────────

def peek_rank(path: Path) -> int:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            if "lora_down" in k:
                return f.get_tensor(k).shape[0]
    return 4


def load_lora(network: LoRANetwork, path: Path, device, dtype):
    state = load_file(str(path), device=str(device))
    miss, unexp = network.load_state_dict(state, strict=False)
    if miss:   print(f"  ⚠ {len(miss)} missing keys")
    if unexp:  print(f"  ⚠ {len(unexp)} unexpected keys")
    return network


def run(pipe, network, image, prompt, scale, steps, device, dtype, seed=42):
    """Run pipeline once with a given LoRA scale. scale=0 = no LoRA."""
    generator = torch.Generator(device=device).manual_seed(seed)
    network.set_lora_slider(scale=scale)

    # Sanity: measure LoRA effect on first to_q
    blk = pipe.transformer.transformer_blocks[0].attn.to_q
    x   = torch.randn(1, blk.weight.shape[1], device=device, dtype=dtype)
    base = blk(x).abs().sum().item()
    with network:
        lora_out = blk(x).abs().sum().item()
    diff = abs(lora_out - base)
    print(f"  [LoRA-check] scale={scale:+.1f}  base={base:.4f}  "
          f"with_lora={lora_out:.4f}  diff={diff:.6f}"
          + ("  ← LoRA ACTIVE" if diff > 1e-6 else "  ← LoRA NOT ACTIVE ❌"))

    with network:
        out = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
            output_type="pil",
        ).images[0]
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="black-forest-labs/FLUX.2-klein-9B")
    p.add_argument("--lora_h",  default="models/eye_gaze_horizontal_v2_rank8_alpha4.0/last.safetensors")
    p.add_argument("--input",   default=None, help="Input portrait path (optional)")
    p.add_argument("--prompt",  default="professional portrait photograph, studio lighting, photorealistic, sharp focus, high quality")
    p.add_argument("--steps",   type=int,   default=8)
    p.add_argument("--scale",   type=float, default=15.0, help="Max |scale| for left/right")
    p.add_argument("--out",     default="gaze_test_result.png")
    p.add_argument("--device",  default="cuda")
    p.add_argument("--seed",    type=int,   default=42)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype  = torch.bfloat16

    # ── load pipeline ─────────────────────────────────────────────────────
    print(f"Loading pipeline …")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)

    # ── input image ───────────────────────────────────────────────────────
    if args.input:
        img = Image.open(args.input).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        print(f"Input image: {args.input}")
    else:
        # Generate a synthetic face first (text-only, no image conditioning)
        print("No --input provided; generating synthetic face from text …")
        # Use first call without image conditioning — pipe needs image, so
        # provide a neutral grey image and let text drive the generation
        grey = Image.new("RGB", (1024, 1024), (128, 128, 128))
        gen  = torch.Generator(device=device).manual_seed(args.seed)
        img  = pipe(
            image=grey,
            prompt="photo of a young man, neutral expression, looking straight at camera, studio portrait",
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            generator=gen,
            output_type="pil",
        ).images[0]
        img.save("gaze_test_input.png")
        print("Saved synthetic face → gaze_test_input.png")

    # ── load LoRA ─────────────────────────────────────────────────────────
    lora_h_path = Path(args.lora_h)
    rank = peek_rank(lora_h_path)
    print(f"Loading horizontal LoRA  rank={rank}  from {lora_h_path.name} …")
    network_h = LoRANetwork(
        pipe.transformer, rank=rank, multiplier=0.0, alpha=4.0, train_method="noxattn"
    ).to(device).to(dtype)
    load_lora(network_h, lora_h_path, device, dtype)

    # ── run 3 passes ──────────────────────────────────────────────────────
    results = {}
    for label, scale in [("center (scale=0)", 0.0),
                         (f"LEFT   (scale=-{args.scale})", -args.scale),
                         (f"RIGHT  (scale=+{args.scale})", +args.scale)]:
        print(f"\n→ {label}")
        results[label] = run(pipe, network_h, img, args.prompt,
                             scale, args.steps, device, dtype, args.seed)

    # ── stitch and save ───────────────────────────────────────────────────
    W, H = img.size
    strip = Image.new("RGB", (W * 4, H))
    strip.paste(img.resize((W, H)),           (0,       0))
    strip.paste(results["center (scale=0)"],  (W,       0))
    strip.paste(results[f"LEFT   (scale=-{args.scale})"],  (W * 2, 0))
    strip.paste(results[f"RIGHT  (scale=+{args.scale})"],  (W * 3, 0))

    strip.save(args.out)
    print(f"\n✓ Saved 4-panel strip (input | center | left | right) → {args.out}")


if __name__ == "__main__":
    main()
