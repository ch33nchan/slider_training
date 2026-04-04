"""
Download FFHQ faces from HuggingFace and generate left/right/neutral gaze pairs
using LivePortrait. Saves organized dataset to data/eye_gaze_v7/{neg,pos,neutral}/.

Usage:
    cd SliderTraining-Klein
    python prepare_gaze_data.py --n_faces 1000 --device_id 1

This will:
    1. Download 1000 faces from bitmind/ffhq-256 (uses HF cache if available)
    2. Run LivePortrait to warp each face left and right (gaze_strength=30)
    3. Optionally blend only eye regions from warped outputs (default)
    4. Organize into data/eye_gaze_v7/neg/, pos/, neutral/, masks/
    4. Skip any face that already exists in the output dirs
"""

import argparse
import os
import sys
import cv2
import glob
from pathlib import Path
import numpy as np

# ── LivePortrait is one level up ──────────────────────────────────────────────
LIVEPORTRAIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../LivePortrait")
sys.path.insert(0, LIVEPORTRAIT_DIR)

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.gradio_pipeline import GradioPipeline


def build_pipeline(device_id: int):
    args = ArgumentConfig()
    args.device_id = device_id
    args.flag_use_half_precision = True
    args.flag_pasteback = True
    args.flag_do_crop = True
    inference_cfg = InferenceConfig()
    inference_cfg.device_id = device_id
    inference_cfg.flag_use_half_precision = True
    crop_cfg = CropConfig()
    return GradioPipeline(inference_cfg, crop_cfg, args)


def warp_gaze(pipeline, img_path: str, eyeball_x: float, output_size: int):
    eye_ratio, lip_ratio = pipeline.init_retargeting_image(
        retargeting_source_scale=2.3,
        source_eye_ratio=0.4,
        source_lip_ratio=0.0,
        input_image=img_path,
    )
    _, out_blended = pipeline.execute_image_retargeting(
        input_eye_ratio=eye_ratio,
        input_lip_ratio=lip_ratio,
        input_head_pitch_variation=0.0,
        input_head_yaw_variation=0.0,
        input_head_roll_variation=0.0,
        mov_x=0.0, mov_y=0.0, mov_z=1.0,
        lip_variation_zero=0.0, lip_variation_one=0.0,
        lip_variation_two=0.0, lip_variation_three=0.0,
        smile=0.0, wink=0.0, eyebrow=0.0,
        eyeball_direction_x=float(eyeball_x),
        eyeball_direction_y=0.0,
        input_image=img_path,
        retargeting_source_scale=2.3,
        flag_stitching_retargeting_input=True,
        flag_do_crop_input_retargeting_image=True,
    )
    import cv2 as _cv2
    out_bgr = _cv2.cvtColor(out_blended, _cv2.COLOR_RGB2BGR)
    out_bgr = _cv2.resize(out_bgr, (output_size, output_size),
                          interpolation=_cv2.INTER_LANCZOS4)
    return out_bgr


def _largest_box(boxes):
    if len(boxes) == 0:
        return None
    return max(boxes, key=lambda b: int(b[2]) * int(b[3]))


def build_eye_mask(
    source_bgr: np.ndarray,
    eye_mask_scale: float,
    eye_mask_blur: int,
) -> np.ndarray:
    h, w = source_bgr.shape[:2]
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(48, w // 8), max(48, h // 8)),
    )
    face = _largest_box(faces)

    eye_boxes = []
    if face is not None:
        fx, fy, fw, fh = map(int, face)
        roi_y_end = fy + int(fh * 0.65)
        roi = gray[fy:roi_y_end, fx:fx + fw]
        detected = eye_detector.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(max(12, fw // 12), max(12, fh // 12)),
        )
        for ex, ey, ew, eh in detected:
            eye_boxes.append((fx + int(ex), fy + int(ey), int(ew), int(eh)))

    if len(eye_boxes) >= 2:
        eye_boxes = sorted(
            eye_boxes,
            key=lambda b: int(b[2]) * int(b[3]),
            reverse=True,
        )[:4]
        eye_boxes = sorted(eye_boxes, key=lambda b: b[0])[:2]
    else:
        cx = w // 2
        cy = int(h * 0.38)
        rx = int(w * 0.09)
        ry = int(h * 0.05)
        offset = int(w * 0.13)
        eye_boxes = [
            (cx - offset - rx, cy - ry, 2 * rx, 2 * ry),
            (cx + offset - rx, cy - ry, 2 * rx, 2 * ry),
        ]

    mask = np.zeros((h, w), dtype=np.float32)
    for ex, ey, ew, eh in eye_boxes:
        cx = int(ex + ew * 0.5)
        cy = int(ey + eh * 0.5)
        rx = max(8, int(ew * 0.5 * eye_mask_scale))
        ry = max(6, int(eh * 0.6 * eye_mask_scale))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)

    if eye_mask_blur <= 0:
        eye_mask_blur = 1
    if eye_mask_blur % 2 == 0:
        eye_mask_blur += 1
    mask = cv2.GaussianBlur(mask, (eye_mask_blur, eye_mask_blur), 0)
    return np.clip(mask, 0.0, 1.0)


def blend_eyes_only(neutral_bgr: np.ndarray, warped_bgr: np.ndarray, eye_mask: np.ndarray) -> np.ndarray:
    alpha = eye_mask[..., None].astype(np.float32)
    neutral_f = neutral_bgr.astype(np.float32)
    warped_f = warped_bgr.astype(np.float32)
    out = neutral_f * (1.0 - alpha) + warped_f * alpha
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def download_ffhq_faces(n_faces: int, tmp_dir: str):
    """Download n_faces from bitmind/ffhq-256 and save as PNGs."""
    import torch
    from datasets import load_dataset

    os.makedirs(tmp_dir, exist_ok=True)
    existing = set(os.path.splitext(f)[0]
                   for f in os.listdir(tmp_dir) if f.endswith(".png"))

    print(f"Loading bitmind/ffhq-256 dataset (streaming)...")
    ds = load_dataset("bitmind/ffhq-256", split="train", streaming=True)

    saved = len(existing)
    print(f"  {saved} faces already downloaded, need {n_faces} total")

    for i, sample in enumerate(ds):
        stem = f"face_{i:04d}"
        if stem in existing:
            continue
        img = sample["image"]  # PIL Image
        out_path = os.path.join(tmp_dir, f"{stem}.png")
        img.save(out_path)
        saved += 1
        if saved % 100 == 0:
            print(f"  Downloaded {saved}/{n_faces}")
        if saved >= n_faces:
            break

    print(f"Downloaded {saved} faces to {tmp_dir}")
    return sorted(glob.glob(os.path.join(tmp_dir, "*.png")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_faces", type=int, default=1000)
    parser.add_argument("--gaze_strength", type=float, default=30.0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device_id", type=int, default=1)
    parser.add_argument("--output_dir", default="data/eye_gaze_v7")
    parser.add_argument("--tmp_dir", default="data/ffhq_source")
    parser.add_argument(
        "--blend_mode",
        type=str,
        default="eyes_only",
        choices=["eyes_only", "full"],
        help="eyes_only keeps source skin texture by blending only eye regions.",
    )
    parser.add_argument("--eye_mask_scale", type=float, default=1.8)
    parser.add_argument("--eye_mask_blur", type=int, default=31)
    parser.add_argument("--skip_mask_write", action="store_true")
    args = parser.parse_args()

    neg_dir = os.path.join(args.output_dir, "neg")
    pos_dir = os.path.join(args.output_dir, "pos")
    neutral_dir = os.path.join(args.output_dir, "neutral")
    mask_dir = os.path.join(args.output_dir, "masks")
    for d in [neg_dir, pos_dir, neutral_dir, mask_dir]:
        os.makedirs(d, exist_ok=True)

    # Download source faces
    face_paths = download_ffhq_faces(args.n_faces, args.tmp_dir)
    print(f"\nBuilding LivePortrait pipeline on cuda:{args.device_id} ...")
    pipeline = build_pipeline(args.device_id)
    print(f"Blend mode: {args.blend_mode}")

    done = 0
    skipped = 0
    failed = 0

    for img_path in face_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        neu_out = os.path.join(neutral_dir, f"{stem}.png")
        neg_out = os.path.join(neg_dir, f"{stem}.png")
        pos_out = os.path.join(pos_dir, f"{stem}.png")
        mask_out = os.path.join(mask_dir, f"{stem}.png")

        if (
            os.path.exists(neu_out)
            and os.path.exists(neg_out)
            and os.path.exists(pos_out)
            and (args.skip_mask_write or os.path.exists(mask_out))
        ):
            skipped += 1
            continue

        try:
            # Neutral: just resize source
            src_bgr = cv2.imread(img_path)
            src_bgr = cv2.resize(src_bgr, (args.size, args.size),
                                 interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(neu_out, src_bgr)

            eye_mask = build_eye_mask(
                source_bgr=src_bgr,
                eye_mask_scale=args.eye_mask_scale,
                eye_mask_blur=args.eye_mask_blur,
            )
            if not args.skip_mask_write:
                mask_u8 = np.clip(eye_mask * 255.0 + 0.5, 0, 255).astype(np.uint8)
                cv2.imwrite(mask_out, mask_u8)

            # Right gaze (pos)
            right_bgr = warp_gaze(pipeline, img_path, +args.gaze_strength, args.size)

            # Left gaze (neg)
            left_bgr = warp_gaze(pipeline, img_path, -args.gaze_strength, args.size)

            if args.blend_mode == "eyes_only":
                right_bgr = blend_eyes_only(src_bgr, right_bgr, eye_mask)
                left_bgr = blend_eyes_only(src_bgr, left_bgr, eye_mask)

            cv2.imwrite(pos_out, right_bgr)
            cv2.imwrite(neg_out, left_bgr)

            done += 1
            if done % 50 == 0:
                print(f"  Generated {done} pairs (skipped {skipped}, failed {failed})")

        except Exception as e:
            print(f"  SKIP {stem}: {e}")
            failed += 1

    total = done + skipped
    print(f"\nDone! {total} pairs in {args.output_dir}")
    print(f"  Generated: {done}, Skipped (already existed): {skipped}, Failed: {failed}")
    print(f"  neg/: {len(os.listdir(neg_dir))} files")
    print(f"  pos/: {len(os.listdir(pos_dir))} files")
    print(f"  neutral/: {len(os.listdir(neutral_dir))} files")
    if not args.skip_mask_write:
        print(f"  masks/: {len(os.listdir(mask_dir))} files")


if __name__ == "__main__":
    main()
