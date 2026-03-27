"""
generate_gaze_pairs.py
----------------------
Generates gaze-variant training pairs from 100 base portraits using
LivePortrait via GazeSliderInference.apply_gaze().

Covers the FULL 2D joystick space:
  - Cardinal:  left, right, up, down at two strengths (0.5 and 0.9)
  - Diagonals: upper-left, upper-right, lower-left, lower-right at 0.7

Output structure (900+ images):
  data/gaze_pairs/
      neutral/               ← 100 originals
      gaze_x+0.5/            ← look left  (mild)
      gaze_x-0.5/            ← look right (mild)
      gaze_x+0.9/            ← look left  (strong)
      gaze_x-0.9/            ← look right (strong)
      gaze_y+0.5/            ← look up    (mild)
      gaze_y-0.5/            ← look down  (mild)
      gaze_y+0.9/            ← look up    (strong)
      gaze_y-0.9/            ← look down  (strong)
      gaze_ul+0.7/           ← upper-left diagonal
      gaze_ur+0.7/           ← upper-right diagonal
      gaze_dl+0.7/           ← lower-left diagonal
      gaze_dr+0.7/           ← lower-right diagonal

LECO training pairs built from this:
  H slider: (gaze_x+N, gaze_x-N) for each portrait
  V slider: (gaze_y+N, gaze_y-N) for each portrait
  Diagonals supplement both sliders.

Usage:
    python sliders/eye-gaze-slider/generate_gaze_pairs.py
"""

import os
import sys
import shutil
import multiprocessing as mp
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent
BASE_DIR = HERE / "data" / "portrait_base"
OUT_DIR  = HERE / "data" / "gaze_pairs"
GPUS     = [2, 3]

# Full 2D gaze grid: (folder_name, gaze_x, gaze_y)
GAZE_GRID = [
    # Cardinals — mild
    ("gaze_x+0.5",  0.5,  0.0),
    ("gaze_x-0.5", -0.5,  0.0),
    ("gaze_y+0.5",  0.0,  0.5),
    ("gaze_y-0.5",  0.0, -0.5),
    # Cardinals — strong
    ("gaze_x+0.9",  0.9,  0.0),
    ("gaze_x-0.9", -0.9,  0.0),
    ("gaze_y+0.9",  0.0,  0.9),
    ("gaze_y-0.9",  0.0, -0.9),
    # Diagonals
    ("gaze_ul+0.7",  0.7,  0.7),
    ("gaze_ur+0.7", -0.7,  0.7),
    ("gaze_dl+0.7",  0.7, -0.7),
    ("gaze_dr+0.7", -0.7, -0.7),
]


# ── Worker (runs in a subprocess with its own CUDA_VISIBLE_DEVICES) ───────────
def worker(gpu_id: int, image_paths: list):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Add sliders dir to path so we can import inference_eye_gaze
    sliders_dir = str(Path(__file__).resolve().parent)
    if sliders_dir not in sys.path:
        sys.path.insert(0, sliders_dir)

    from inference_eye_gaze import GazeSliderInference
    from PIL import Image

    engine = GazeSliderInference()
    total  = len(image_paths) * len(GAZE_GRID)
    done   = 0

    for img_path in image_paths:
        img_path = Path(img_path)
        img_pil  = Image.open(img_path).convert("RGB")

        # Copy neutral
        dst = OUT_DIR / "neutral" / img_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst)

        for folder, gaze_x, gaze_y in GAZE_GRID:
            dst = OUT_DIR / folder / img_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists():
                done += 1
                continue

            try:
                out = engine.apply_gaze(img_pil, gaze_x=gaze_x, gaze_y=gaze_y)
                out.save(dst)
                done += 1
                print(f"[GPU {gpu_id}] {done}/{total}  {img_path.stem} → {folder}")
            except Exception as e:
                print(f"[GPU {gpu_id}] SKIP {img_path.stem} → {folder}: {e}")
                import traceback; traceback.print_exc()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_images = sorted(BASE_DIR.glob("*.png")) + sorted(BASE_DIR.glob("*.jpg"))
    if not all_images:
        raise RuntimeError(f"No images found in {BASE_DIR}")

    print(f"Found {len(all_images)} base portraits")
    print(f"Generating {len(GAZE_GRID)} gaze variants each = "
          f"{len(all_images) * len(GAZE_GRID)} images total")
    print(f"Output → {OUT_DIR}")

    mid    = len(all_images) // 2
    chunks = [all_images[:mid], all_images[mid:]]

    procs = []
    for gpu_id, chunk in zip(GPUS, chunks):
        p = mp.Process(target=worker, args=(gpu_id, [str(p) for p in chunk]))
        p.start()
        procs.append(p)
        print(f"[GPU {gpu_id}] Started — {len(chunk)} portraits × "
              f"{len(GAZE_GRID)} directions")

    for p in procs:
        p.join()

    # Summary
    print("\n── Results ──────────────────────────────")
    total = 0
    for folder, gx, gy in [("neutral", 0, 0)] + [(f, x, y) for f, x, y in GAZE_GRID]:
        d = OUT_DIR / folder
        n = len(list(d.glob("*"))) if d.exists() else 0
        total += n
        print(f"  {folder:20s}: {n:3d} images")
    print(f"  {'TOTAL':20s}: {total:3d} images")
    print("\nDone. Run train_leco.py next.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
