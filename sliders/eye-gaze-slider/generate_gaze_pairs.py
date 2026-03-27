"""
generate_gaze_pairs.py
----------------------
Takes the 100 base portraits and generates gaze-variant pairs using
LivePortrait. Runs in parallel across two GPUs.

Output structure:
    data/gaze_pairs/
        horizontal/
            positive/   ← look_left  (gaze_x=+0.8)
            negative/   ← look_right (gaze_x=-0.8)
            neutral/    ← original copy
        vertical/
            positive/   ← look_up    (gaze_y=+0.8)
            negative/   ← look_down  (gaze_y=-0.8)
            neutral/    ← original copy

Usage:
    python sliders/eye-gaze-slider/generate_gaze_pairs.py
"""

import os
import sys
import shutil
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
BASE_DIR  = HERE / "data" / "portrait_base"
OUT_DIR   = HERE / "data" / "gaze_pairs"
GPUS      = [2, 3]
GAZE_VAL  = 0.8   # gaze magnitude for training pairs

# ── LivePortrait path ─────────────────────────────────────────────────────────
def _find_liveportrait() -> Path:
    candidates = [
        HERE.parents[1] / "LivePortrait",
        HERE.parents[2] / "LivePortrait",
        Path("/mnt/data1/srini/eyegaze/LivePortrait"),
    ]
    for c in candidates:
        if (c / "src" / "gradio_pipeline.py").exists():
            return c
    raise RuntimeError("Cannot find LivePortrait directory.")


# ── Worker ────────────────────────────────────────────────────────────────────
def worker(gpu_id: int, image_paths: list[Path]):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    lp_root = _find_liveportrait()
    if str(lp_root) not in sys.path:
        sys.path.insert(0, str(lp_root))

    os.chdir(lp_root)   # LivePortrait needs cwd = its own root

    from src.gradio_pipeline import GradioPipeline
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig

    args         = ArgumentConfig()
    infer_cfg    = InferenceConfig()
    crop_cfg     = CropConfig()
    pipeline     = GradioPipeline(args, infer_cfg, crop_cfg)

    # Directions: (axis, direction_name, gaze_x, gaze_y, split, role)
    DIRECTIONS = [
        ("horizontal", "positive", GAZE_VAL,  0.0),   # look left
        ("horizontal", "negative", -GAZE_VAL, 0.0),   # look right
        ("vertical",   "positive", 0.0,  GAZE_VAL),   # look up
        ("vertical",   "negative", 0.0, -GAZE_VAL),   # look down
    ]

    MAX_SCALE = 12.0

    for img_path in image_paths:
        img_pil = Image.open(img_path).convert("RGB")
        stem    = img_path.stem

        # Copy neutral to both splits
        for axis in ("horizontal", "vertical"):
            dst = OUT_DIR / axis / "neutral" / img_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst)

        for axis, role, gaze_x, gaze_y in DIRECTIONS:
            eye_x = float(-gaze_x) * MAX_SCALE   # sign convention
            eye_y = float(gaze_y)  * MAX_SCALE

            try:
                result = pipeline.execute_image_retargeting(
                    img_pil,
                    eyeball_direction_x              = eye_x,
                    eyeball_direction_y              = eye_y,
                    input_eye_ratio                  = None,
                    input_lip_ratio                  = None,
                    input_head_pitch_variation       = 0.0,
                    input_head_yaw_variation         = 0.0,
                    input_head_roll_variation        = 0.0,
                    flag_do_crop_input_retargeting_image = True,
                    flag_stitching_retargeting_input = True,
                )
                # execute_image_retargeting returns (out_img, ...) or just img
                out_img = result[0] if isinstance(result, (list, tuple)) else result
                if not isinstance(out_img, Image.Image):
                    out_img = Image.fromarray(np.uint8(out_img))

                dst = OUT_DIR / axis / role / img_path.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                out_img.save(dst)
                print(f"[GPU {gpu_id}] {stem} → {axis}/{role}")

            except Exception as e:
                print(f"[GPU {gpu_id}] SKIP {stem} ({axis}/{role}): {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_images = sorted(BASE_DIR.glob("*.png")) + sorted(BASE_DIR.glob("*.jpg"))
    if not all_images:
        raise RuntimeError(f"No images found in {BASE_DIR}")

    print(f"Found {len(all_images)} base portraits")
    print(f"Output → {OUT_DIR}")

    # Split across 2 GPUs
    mid     = len(all_images) // 2
    chunks  = [all_images[:mid], all_images[mid:]]

    procs = []
    for gpu_id, chunk in zip(GPUS, chunks):
        p = mp.Process(target=worker, args=(gpu_id, chunk))
        p.start()
        procs.append(p)
        print(f"[GPU {gpu_id}] Started — {len(chunk)} images")

    for p in procs:
        p.join()

    # Summary
    for axis in ("horizontal", "vertical"):
        for role in ("positive", "negative", "neutral"):
            d   = OUT_DIR / axis / role
            n   = len(list(d.glob("*"))) if d.exists() else 0
            print(f"  {axis}/{role}: {n} images")

    print("\nDone. Ready for LoRA training.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
