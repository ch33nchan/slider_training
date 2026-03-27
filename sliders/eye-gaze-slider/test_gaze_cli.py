#!/usr/bin/env python3
"""
test_gaze_cli.py — MediaPipe iris warp batch test.

Produces two strips saved to disk:

  batch_extreme.png   — input | LEFT | RIGHT | UP | DOWN  (gaze=1.0, scale=10)
  batch_moderate.png  — input | left | right | center | down  (gaze=0.4, scale=5)

Usage:
  python test_gaze_cli.py --input portrait.jpg
  python test_gaze_cli.py          # uses gaze_test_input.png if it exists
"""

import sys, argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference_eye_gaze import GazeSliderInference


def label_image(img: Image.Image, text: str, font_size: int = 28) -> Image.Image:
    """Paste a white label bar under the image."""
    bar_h = font_size + 10
    out   = Image.new("RGB", (img.width, img.height + bar_h), (30, 30, 30))
    out.paste(img, (0, 0))
    draw  = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, img.height + 4), text, fill=(220, 220, 220), font=font)
    return out


def make_strip(images, labels) -> Image.Image:
    labeled = [label_image(im, lb) for im, lb in zip(images, labels)]
    W, H    = labeled[0].size
    strip   = Image.new("RGB", (W * len(labeled), H), (20, 20, 20))
    for i, img in enumerate(labeled):
        strip.paste(img, (i * W, 0))
    return strip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="black-forest-labs/FLUX.2-klein-9B")
    p.add_argument("--input",  default="gaze_test_input.png")
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", default="/tmp")
    args = p.parse_args()

    # ── load engine (no LoRA needed) ──────────────────────────────────────
    print("Loading engine …")
    engine = GazeSliderInference(model_id=args.model, device=args.device)

    # ── load input image ──────────────────────────────────────────────────
    inp_path = Path(args.input)
    if not inp_path.exists():
        raise SystemExit(f"Input image not found: {inp_path}\n"
                         "Run  python test_gaze_cli.py  once without --input to generate one.")
    img = Image.open(inp_path).convert("RGB")
    print(f"Input: {inp_path}  {img.size}")

    # ── batch 1 — EXTREME (gaze=1.0, scale=10) ───────────────────────────
    print("\n=== Batch 1: EXTREME ===")
    extreme_cases = [
        ("INPUT",  0.0,  0.0, 10.0),
        ("LEFT",  +1.0,  0.0, 10.0),
        ("RIGHT", -1.0,  0.0, 10.0),
        ("UP",     0.0, +1.0, 10.0),
        ("DOWN",   0.0, -1.0, 10.0),
    ]
    extreme_imgs = []
    for label, gx, gy, scale in extreme_cases:
        print(f"  → {label:6s}  gaze=({gx:+.1f},{gy:+.1f})  scale={scale}")
        if label == "INPUT":
            extreme_imgs.append(img.resize((512, 512), Image.LANCZOS))
        else:
            out = engine.apply_gaze(img, gaze_x=gx, gaze_y=gy,
                                    max_scale=scale, strength=0.0)
            extreme_imgs.append(out.resize((512, 512), Image.LANCZOS))

    strip1 = make_strip(extreme_imgs, [c[0] for c in extreme_cases])
    out1   = Path(args.outdir) / "batch_extreme.png"
    strip1.save(str(out1))
    print(f"\n✓ Saved → {out1}")

    # ── batch 2 — MODERATE (gaze=0.4, scale=5) ───────────────────────────
    print("\n=== Batch 2: MODERATE ===")
    moderate_cases = [
        ("INPUT",   0.0,   0.0,  5.0),
        ("left",   +0.4,   0.0,  5.0),
        ("right",  -0.4,   0.0,  5.0),
        ("center",  0.0,   0.0,  5.0),
        ("down",    0.0,  -0.4,  5.0),
    ]
    moderate_imgs = []
    for label, gx, gy, scale in moderate_cases:
        print(f"  → {label:7s}  gaze=({gx:+.2f},{gy:+.2f})  scale={scale}")
        if label in ("INPUT", "center"):
            moderate_imgs.append(img.resize((512, 512), Image.LANCZOS))
        else:
            out = engine.apply_gaze(img, gaze_x=gx, gaze_y=gy,
                                    max_scale=scale, strength=0.0)
            moderate_imgs.append(out.resize((512, 512), Image.LANCZOS))

    strip2 = make_strip(moderate_imgs, [c[0] for c in moderate_cases])
    out2   = Path(args.outdir) / "batch_moderate.png"
    strip2.save(str(out2))
    print(f"✓ Saved → {out2}")

    print("\nDone. Copy results with:")
    print(f"  scp charizard:{out1} ~/Desktop/")
    print(f"  scp charizard:{out2} ~/Desktop/")


if __name__ == "__main__":
    main()
