"""
Generate vertical gaze pairs using LivePortrait with optional eye-only blending.

Writes:
  neg/<stem>.png      gaze down
  pos/<stem>.png      gaze up
  neutral/<stem>.png  resized source
  masks/<stem>.png    soft eye mask (when blend_mode=eyes_only)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LIVEPORTRAIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../LivePortrait")
sys.path.insert(0, LIVEPORTRAIT_DIR)


def build_liveportrait(device_id):
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.gradio_pipeline import GradioPipeline

    a = ArgumentConfig()
    a.device_id = device_id
    a.flag_use_half_precision = True
    a.flag_pasteback = True
    a.flag_do_crop = True
    inf = InferenceConfig()
    inf.device_id = device_id
    inf.flag_use_half_precision = True
    return GradioPipeline(inf, CropConfig(), a)


def warp_gaze_vertical(pipeline, img_path, eyeball_y, size):
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
        eyeball_direction_x=0.0,
        eyeball_direction_y=float(eyeball_y),
        input_image=img_path,
        retargeting_source_scale=2.3,
        flag_stitching_retargeting_input=True,
        flag_do_crop_input_retargeting_image=True,
    )
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_LANCZOS4)


def _largest_box(boxes):
    if len(boxes) == 0:
        return None
    return max(boxes, key=lambda b: int(b[2]) * int(b[3]))


def build_eye_mask(
    source_rgb: np.ndarray,
    eye_mask_scale: float = 1.8,
    eye_mask_blur: int = 31,
) -> np.ndarray:
    h, w = source_rgb.shape[:2]
    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)

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


def blend_eyes_only(neutral_rgb: np.ndarray, warped_rgb: np.ndarray, eye_mask: np.ndarray) -> np.ndarray:
    alpha = eye_mask[..., None].astype(np.float32)
    neutral_f = neutral_rgb.astype(np.float32)
    warped_f = warped_rgb.astype(np.float32)
    blended = neutral_f * (1.0 - alpha) + warped_f * alpha
    return np.clip(blended + 0.5, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="../LivePortrait/source_faces")
    parser.add_argument("--output_dir", default="data/gaze_vertical_s15")
    parser.add_argument("--num_faces", type=int, default=30)
    parser.add_argument("--gaze_strength", type=float, default=15)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device_id", type=int, default=3)
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

    out = Path(args.output_dir)
    for split in ["neg", "pos", "neutral", "masks"]:
        (out / split).mkdir(parents=True, exist_ok=True)

    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    all_sources = []
    for ext in exts:
        all_sources.extend(Path(args.input_dir).glob(ext))
    sources = sorted(all_sources)[:args.num_faces]

    print(f"Using {len(sources)} source faces from {args.input_dir}")
    print(f"Output: {args.output_dir} at {args.size}x{args.size}")
    print(f"Gaze strength: +/-{args.gaze_strength} (neg=down, pos=up)\n")
    print(f"Blend mode: {args.blend_mode}")

    print("Building LivePortrait...")
    pipeline = build_liveportrait(args.device_id)

    for i, src in enumerate(sources):
        stem = src.stem
        print(f"[{i+1}/{len(sources)}] {stem}")

        neutral_pil = Image.open(src).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
        neutral_rgb = np.array(neutral_pil)
        neutral_pil.save(str(out / "neutral" / f"{stem}.png"))

        eye_mask = build_eye_mask(
            source_rgb=neutral_rgb,
            eye_mask_scale=args.eye_mask_scale,
            eye_mask_blur=args.eye_mask_blur,
        )
        if not args.skip_mask_write:
            mask_u8 = np.clip(eye_mask * 255.0 + 0.5, 0, 255).astype(np.uint8)
            Image.fromarray(mask_u8, mode="L").save(str(out / "masks" / f"{stem}.png"))

        # neg = gaze DOWN (positive eyeball_direction_y)
        neg_rgb = warp_gaze_vertical(pipeline, str(src), +args.gaze_strength, args.size)

        # pos = gaze UP (negative eyeball_direction_y)
        pos_rgb = warp_gaze_vertical(pipeline, str(src), -args.gaze_strength, args.size)

        if args.blend_mode == "eyes_only":
            neg_rgb = blend_eyes_only(neutral_rgb, neg_rgb, eye_mask)
            pos_rgb = blend_eyes_only(neutral_rgb, pos_rgb, eye_mask)

        Image.fromarray(neg_rgb).save(str(out / "neg" / f"{stem}.png"))
        Image.fromarray(pos_rgb).save(str(out / "pos" / f"{stem}.png"))

        print(f"  saved neg(down)/neutral/pos(up)")

    print(f"\nDone. Dataset at {args.output_dir}/")
    print(f"  neg/     — {len(sources)} gaze-DOWN images")
    print(f"  pos/     — {len(sources)} gaze-UP images")
    print(f"  neutral/ — {len(sources)} neutral images")
    if not args.skip_mask_write:
        print(f"  masks/   — {len(sources)} eye masks")


if __name__ == "__main__":
    main()
