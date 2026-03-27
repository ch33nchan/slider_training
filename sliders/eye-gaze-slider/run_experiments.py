#!/usr/bin/env python3
"""
run_experiments.py
------------------
Systematic experiment runner: 5 portraits × N cases → comparison grid.

Each row = one portrait.
Each column = one experimental case.
Direction tested: LEFT gaze (gaze_x=+1.0).

Cases:
  A  — LivePortrait only          scale=12  no FLUX
  B  — LivePortrait + FLUX refine scale=12  strength=0.15  no LoRA
  C  — LivePortrait + C1 LoRA    scale=12  strength=0.15
  D  — LivePortrait + C1 LoRA    scale=12  strength=0.25
  E  — LivePortrait + C1 LoRA    scale=20  strength=0.15

Usage:
  python run_experiments.py \
      --portraits /tmp/test_portrait.jpg p2.jpg p3.jpg p4.jpg p5.jpg \
      --lora_h models/C1/gaze_horizontal_C1.safetensors \
      --lora_v models/C1/gaze_vertical_C1.safetensors \
      --outdir /tmp/experiments \
      --device cuda
"""

import sys
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── helpers ──────────────────────────────────────────────────────────────────

def label_image(img: Image.Image, text: str, font_size: int = 24) -> Image.Image:
    bar_h = font_size + 10
    out   = Image.new("RGB", (img.width, img.height + bar_h), (20, 20, 20))
    out.paste(img, (0, 0))
    draw  = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bb = draw.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    draw.text(((img.width - tw) // 2, img.height + 4),
              text, fill=(220, 220, 220), font=font)
    return out


def make_grid(rows: list[list[Image.Image]],
              row_labels: list[str],
              col_labels: list[str],
              cell_size: tuple[int, int] = (384, 384),
              font_size: int = 22) -> Image.Image:
    """rows[i][j] = image for portrait i, case j."""
    n_rows = len(rows)
    n_cols = len(rows[0])
    cw, ch = cell_size
    bar_h  = font_size + 10
    label_w = 140   # row label column

    grid_w = label_w + n_cols * cw
    grid_h = bar_h + n_rows * (ch + bar_h)   # col header + rows

    grid = Image.new("RGB", (grid_w, grid_h), (15, 15, 15))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size - 4)
    except Exception:
        font = font_sm = ImageFont.load_default()

    # Column headers
    for j, clabel in enumerate(col_labels):
        x = label_w + j * cw + cw // 2
        bb = draw.textbbox((0, 0), clabel, font=font)
        tw = bb[2] - bb[0]
        draw.text((x - tw // 2, 4), clabel, fill=(255, 220, 80), font=font)

    # Rows
    for i, (row_imgs, rlabel) in enumerate(zip(rows, row_labels)):
        y_off = bar_h + i * (ch + bar_h)

        # Row label
        bb  = draw.textbbox((0, 0), rlabel, font=font_sm)
        th  = bb[3] - bb[1]
        draw.text((4, y_off + ch // 2 - th // 2),
                  rlabel, fill=(180, 180, 180), font=font_sm)

        for j, img in enumerate(row_imgs):
            cell = img.resize(cell_size, Image.LANCZOS)
            x_off = label_w + j * cw
            grid.paste(cell, (x_off, y_off))

        # Row separator
        draw.line([(0, y_off + ch + bar_h - 1),
                   (grid_w, y_off + ch + bar_h - 1)],
                  fill=(40, 40, 40), width=1)

    return grid


# ── experiment cases ──────────────────────────────────────────────────────────

CASES = [
    # (label,  lp_scale, refine_strength, use_lora)
    ("A  LP only\nscale=12",            12,  0.00,  False),
    ("B  FLUX refine\nno LoRA s=0.15",  12,  0.15,  False),
    ("C  C1 LoRA\nscale=12 s=0.15",     12,  0.15,  True),
    ("D  C1 LoRA\nscale=12 s=0.25",     12,  0.25,  True),
    ("E  C1 LoRA\nscale=20 s=0.15",     20,  0.15,  True),
]

GAZE_X = +1.0   # LEFT gaze for all cases


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--portraits", nargs="+", required=True,
                   help="5 portrait image paths")
    p.add_argument("--lora_h",   required=True)
    p.add_argument("--lora_v",   required=True)
    p.add_argument("--outdir",   default="/tmp/experiments")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--rank",     type=int,   default=8)
    p.add_argument("--alpha",    type=float, default=4.0)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    portraits = [Path(pp) for pp in args.portraits[:5]]
    for pp in portraits:
        if not pp.exists():
            raise SystemExit(f"Portrait not found: {pp}")

    print(f"Portraits : {[p.name for p in portraits]}")
    print(f"Cases     : {len(CASES)}")
    print(f"Output    : {outdir}")

    # ── Load engines ─────────────────────────────────────────────────────────
    # We need two engines:
    #   engine_no_lora  — LivePortrait + FLUX (no LoRA)
    #   engine_lora     — LivePortrait + FLUX + C1 LoRAs
    # Engine A uses LivePortrait only (refine_strength=0), reuses engine_no_lora.

    from inference_eye_gaze import GazeSliderInference

    print("\n[1/2] Loading engine WITHOUT LoRA …")
    engine_no_lora = GazeSliderInference(device=args.device)

    print("\n[2/2] Loading engine WITH C1 LoRA …")
    engine_lora = GazeSliderInference(
        lora_h=args.lora_h,
        lora_v=args.lora_v,
        rank=args.rank,
        alpha=args.alpha,
        device=args.device,
    )

    # ── Run experiments ───────────────────────────────────────────────────────
    col_labels = [c[0].replace("\n", " ") for c in CASES]
    row_labels = [f"P{i+1}\n{p.stem[:10]}" for i, p in enumerate(portraits)]
    grid_rows  = []

    for i, port_path in enumerate(portraits):
        img = Image.open(port_path).convert("RGB")
        print(f"\n── Portrait {i+1}/{len(portraits)}: {port_path.name} ──")
        row_imgs = []

        for label, lp_scale, strength, use_lora in CASES:
            case_tag = label.split()[0]
            print(f"  Case {case_tag}  scale={lp_scale}  strength={strength}"
                  f"  lora={use_lora}")

            engine = engine_lora if use_lora else engine_no_lora

            out = engine.apply_gaze(
                img,
                gaze_x=GAZE_X,
                gaze_y=0.0,
                max_scale=lp_scale,
                strength=strength,
            )
            row_imgs.append(out)

            # Save individual result
            fname = outdir / f"p{i+1}_{case_tag}.png"
            out.save(str(fname))
            print(f"    ✓ {fname.name}")

        grid_rows.append(row_imgs)

    # ── Save comparison grid ──────────────────────────────────────────────────
    print("\nBuilding comparison grid …")
    grid = make_grid(grid_rows, row_labels, col_labels, cell_size=(384, 384))
    grid_path = outdir / "comparison_grid.png"
    grid.save(str(grid_path))
    print(f"✓ Grid → {grid_path}")
    print(f"\nSCP command:")
    print(f"  scp charizard:{outdir}/* ~/Desktop/exp/")


if __name__ == "__main__":
    main()
