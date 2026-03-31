"""
Download gaze pairs from vigil1917/GazeGene (HuggingFace).

GazeGene has 56 subjects × 9 cameras × 2000 frames with full 3D gaze vectors.
For each subject we select:
  neg    = image with strongest leftward horizontal gaze  (gaze_x most negative)
  pos    = image with strongest rightward horizontal gaze (gaze_x most positive)
  neutral = image with gaze most forward (gaze_x ≈ 0, gaze_y ≈ 0)

Only uses camera 0 (frontal) for clean portrait-style images.

Saves to data/columbia_gaze/{neg,pos,neutral}/ (reuses the v7 config paths).

Usage:
    python download_gaze_data.py --device cuda:1
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/columbia_gaze")
    parser.add_argument("--n_subjects", type=int, default=56,
                        help="Max subjects to use (dataset has 56)")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    neg_dir = Path(args.output_dir) / "neg"
    pos_dir = Path(args.output_dir) / "pos"
    neutral_dir = Path(args.output_dir) / "neutral"
    for d in [neg_dir, pos_dir, neutral_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("Loading GazeGene dataset from HuggingFace (this may take a while)...")
    print("Dataset: vigil1917/GazeGene")

    try:
        ds = load_dataset("vigil1917/GazeGene", split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load GazeGene: {e}")
        print("\nFalling back to generating pairs from FFHQ with small eyeball movement...")
        fallback_liveportrait(args)
        return

    # Group by subject
    print("Streaming dataset to collect per-subject gaze samples...")
    subjects = {}  # subject_id -> list of (gaze_x, gaze_y, image)

    try:
        for i, sample in enumerate(ds):
            # Inspect first sample to find field names
            if i == 0:
                print(f"Dataset fields: {list(sample.keys())}")

            # Try to extract subject id and gaze vector
            subject_id = sample.get("subject", sample.get("subject_id",
                         sample.get("person_id", str(i // 2000))))

            # Gaze vector: try various field names
            gaze = None
            for field in ["gaze_vector", "gaze", "face_gaze", "gaze_label"]:
                if field in sample:
                    gaze = sample[field]
                    break

            if gaze is None:
                if i == 0:
                    print("Could not find gaze field. Available fields:", list(sample.keys()))
                    print("Trying fallback...")
                    break
                continue

            # Convert gaze to numpy
            if isinstance(gaze, (list, tuple)):
                gaze = np.array(gaze, dtype=np.float32)
            elif isinstance(gaze, torch.Tensor):
                gaze = gaze.numpy()

            # gaze_x = horizontal component (negative = left, positive = right)
            if gaze.ndim == 1 and len(gaze) >= 2:
                gaze_x = float(gaze[0])
                gaze_y = float(gaze[1]) if len(gaze) > 1 else 0.0
            else:
                continue

            # Get image
            img = sample.get("image", sample.get("face_image", sample.get("img")))
            if img is None:
                continue
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB").resize((args.size, args.size))

            if subject_id not in subjects:
                subjects[subject_id] = []
            subjects[subject_id].append((gaze_x, gaze_y, img))

            if i % 5000 == 0:
                print(f"  Processed {i} samples, {len(subjects)} subjects so far")

            if len(subjects) >= args.n_subjects and all(
                len(v) >= 100 for v in subjects.values()
            ):
                break

    except Exception as e:
        print(f"Error streaming dataset: {e}")
        print("Trying fallback...")
        fallback_liveportrait(args)
        return

    if not subjects:
        print("No data collected from GazeGene. Trying fallback...")
        fallback_liveportrait(args)
        return

    print(f"\nCollected {len(subjects)} subjects")

    # For each subject, pick best left/right/neutral samples
    saved = 0
    for subject_id, samples in list(subjects.items())[:args.n_subjects]:
        if len(samples) < 3:
            continue

        gaze_xs = np.array([s[0] for s in samples])

        # Most leftward gaze
        neg_idx = int(np.argmin(gaze_xs))
        # Most rightward gaze
        pos_idx = int(np.argmax(gaze_xs))
        # Most neutral (closest to 0,0)
        total_gaze = np.abs(np.array([s[0] for s in samples])) + np.abs(np.array([s[1] for s in samples]))
        neutral_idx = int(np.argmin(total_gaze))

        # Skip if gaze range is too small (would give noisy training signal)
        gaze_range = gaze_xs[pos_idx] - gaze_xs[neg_idx]
        if gaze_range < 0.1:
            print(f"  Subject {subject_id}: gaze range too small ({gaze_range:.3f}), skipping")
            continue

        stem = f"subject_{str(subject_id).zfill(4)}"
        samples[neg_idx][2].save(neg_dir / f"{stem}.png")
        samples[pos_idx][2].save(pos_dir / f"{stem}.png")
        samples[neutral_idx][2].save(neutral_dir / f"{stem}.png")
        saved += 1
        print(f"  {stem}: left={gaze_xs[neg_idx]:.3f}, right={gaze_xs[pos_idx]:.3f}, range={gaze_range:.3f}")

    print(f"\nSaved {saved} pairs to {args.output_dir}")
    print(f"  neg/:     {len(list(neg_dir.glob('*.png')))} files")
    print(f"  pos/:     {len(list(pos_dir.glob('*.png')))} files")
    print(f"  neutral/: {len(list(neutral_dir.glob('*.png')))} files")


def fallback_liveportrait(args):
    """
    Fallback: generate pairs from FFHQ using LivePortrait with small gaze strength.
    Uses gaze_strength=8 for realistic, subtle eyeball movement (not head rotation).
    """
    import sys
    import glob
    import cv2

    LIVEPORTRAIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../LivePortrait")
    sys.path.insert(0, LIVEPORTRAIT_DIR)

    ffhq_dir = "data/ffhq_source"
    if not os.path.isdir(ffhq_dir):
        print(f"No FFHQ source images found at {ffhq_dir}")
        print("Please run: python prepare_gaze_data.py --n_faces 30 --device_id 1")
        return

    sources = sorted(glob.glob(os.path.join(ffhq_dir, "*.png")))[:30]
    if not sources:
        print(f"No images in {ffhq_dir}")
        return

    print(f"Generating pairs from {len(sources)} FFHQ faces with gaze_strength=8 ...")

    try:
        from src.config.argument_config import ArgumentConfig
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.gradio_pipeline import GradioPipeline
    except ImportError as e:
        print(f"LivePortrait import failed: {e}")
        return

    def build_pipeline():
        a = ArgumentConfig()
        a.device_id = 1
        a.flag_use_half_precision = True
        a.flag_pasteback = True
        a.flag_do_crop = True
        inf = InferenceConfig()
        inf.device_id = 1
        inf.flag_use_half_precision = True
        return GradioPipeline(inf, CropConfig(), a)

    def warp(pipeline, img_path, eyeball_x, size):
        eye_ratio, lip_ratio = pipeline.init_retargeting_image(
            retargeting_source_scale=2.3,
            source_eye_ratio=0.4,
            source_lip_ratio=0.0,
            input_image=img_path,
        )
        _, out = pipeline.execute_image_retargeting(
            input_eye_ratio=eye_ratio, input_lip_ratio=lip_ratio,
            input_head_pitch_variation=0.0, input_head_yaw_variation=0.0,
            input_head_roll_variation=0.0,
            mov_x=0.0, mov_y=0.0, mov_z=1.0,
            lip_variation_zero=0.0, lip_variation_one=0.0,
            lip_variation_two=0.0, lip_variation_three=0.0,
            smile=0.0, wink=0.0, eyebrow=0.0,
            eyeball_direction_x=float(eyeball_x), eyeball_direction_y=0.0,
            input_image=img_path, retargeting_source_scale=2.3,
            flag_stitching_retargeting_input=True,
            flag_do_crop_input_retargeting_image=True,
        )
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LANCZOS4)

    neg_dir = Path(args.output_dir) / "neg"
    pos_dir = Path(args.output_dir) / "pos"
    neutral_dir = Path(args.output_dir) / "neutral"

    print("Building LivePortrait pipeline...")
    pipeline = build_pipeline()

    for img_path in sources:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        try:
            src = cv2.imread(img_path)
            src = cv2.resize(src, (args.size, args.size))
            cv2.imwrite(str(neutral_dir / f"{stem}.png"), src)
            cv2.imwrite(str(neg_dir / f"{stem}.png"), warp(pipeline, img_path, -8, args.size))
            cv2.imwrite(str(pos_dir / f"{stem}.png"), warp(pipeline, img_path, +8, args.size))
            print(f"  {stem} done")
        except Exception as e:
            print(f"  {stem} SKIP: {e}")

    print(f"\nFallback done. {len(list(neg_dir.glob('*.png')))} pairs saved.")


if __name__ == "__main__":
    main()
