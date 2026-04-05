"""
Generate horizontal gaze pairs using LivePortrait with optional eye-only blending.

For each source image this script writes:
  neg/<stem>.png      gaze left
  pos/<stem>.png      gaze right
  neutral/<stem>.png  resized source
  masks/<stem>.png    soft eye mask (when blend_mode=eyes_only)

The default blend mode is `eyes_only` to avoid the soft-skin artifact from
full-frame LivePortrait warping. Only the eye regions are transferred from the
warped image back onto the neutral source image.
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


def warp_gaze(pipeline, img_path, eyeball_x, size):
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
        eyeball_direction_x=float(eyeball_x),
        eyeball_direction_y=0.0,
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


def _box_center(box):
    x, y, w, h = map(float, box)
    return x + 0.5 * w, y + 0.5 * h


def _box_area(box):
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def _box_iou(box_a, box_b):
    ax0, ay0, aw, ah = map(float, box_a)
    bx0, by0, bw, bh = map(float, box_b)
    ax1 = ax0 + aw
    ay1 = ay0 + ah
    bx1 = bx0 + bw
    by1 = by0 + bh

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    inter = (ix1 - ix0) * (iy1 - iy0)
    union = _box_area(box_a) + _box_area(box_b) - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def _dedupe_boxes(boxes, iou_threshold=0.18):
    deduped = []
    for box in sorted(boxes, key=_box_area, reverse=True):
        if any(_box_iou(box, kept) >= iou_threshold for kept in deduped):
            continue
        deduped.append(tuple(map(int, box)))
    return deduped


def _default_eye_boxes(image_width, image_height, face_box=None):
    if face_box is not None:
        fx, fy, fw, fh = map(int, face_box)
        eye_w = max(12, int(fw * 0.18))
        eye_h = max(8, int(fh * 0.10))
        eye_cy = fy + int(fh * 0.39)
        left_cx = fx + int(fw * 0.34)
        right_cx = fx + int(fw * 0.66)
        return [
            (left_cx - eye_w // 2, eye_cy - eye_h // 2, eye_w, eye_h),
            (right_cx - eye_w // 2, eye_cy - eye_h // 2, eye_w, eye_h),
        ]

    cx = image_width // 2
    cy = int(image_height * 0.38)
    rx = int(image_width * 0.09)
    ry = int(image_height * 0.05)
    offset = int(image_width * 0.13)
    return [
        (cx - offset - rx, cy - ry, 2 * rx, 2 * ry),
        (cx + offset - rx, cy - ry, 2 * rx, 2 * ry),
    ]


def _filter_eye_boxes(eye_boxes, face_box):
    if face_box is None:
        return [tuple(map(int, box)) for box in eye_boxes]

    fx, fy, fw, fh = map(float, face_box)
    filtered = []
    for box in eye_boxes:
        ex, ey, ew, eh = map(float, box)
        cx, cy = _box_center(box)
        aspect = ew / max(eh, 1.0)
        if ew < fw * 0.07 or ew > fw * 0.32:
            continue
        if eh < fh * 0.05 or eh > fh * 0.20:
            continue
        if cx < fx + fw * 0.10 or cx > fx + fw * 0.90:
            continue
        if cy < fy + fh * 0.18 or cy > fy + fh * 0.52:
            continue
        if aspect < 0.6 or aspect > 3.5:
            continue
        filtered.append((int(ex), int(ey), int(ew), int(eh)))
    return _dedupe_boxes(filtered)


def _select_eye_pair(eye_boxes, face_box):
    if len(eye_boxes) < 2:
        return None

    if face_box is None:
        face_width = max(float(max(box[0] + box[2] for box in eye_boxes) - min(box[0] for box in eye_boxes)), 1.0)
        face_height = max(float(max(box[1] + box[3] for box in eye_boxes) - min(box[1] for box in eye_boxes)), 1.0)
        face_center_x = sum(_box_center(box)[0] for box in eye_boxes) / float(len(eye_boxes))
    else:
        fx, fy, fw, fh = map(float, face_box)
        face_width = max(fw, 1.0)
        face_height = max(fh, 1.0)
        face_center_x = fx + 0.5 * fw

    best_pair = None
    best_score = -1e9
    for i in range(len(eye_boxes)):
        for j in range(i + 1, len(eye_boxes)):
            left, right = sorted((eye_boxes[i], eye_boxes[j]), key=lambda box: box[0])
            left_cx, left_cy = _box_center(left)
            right_cx, right_cy = _box_center(right)
            separation = (right_cx - left_cx) / face_width
            vertical_gap = abs(right_cy - left_cy) / face_height
            overlap = _box_iou(left, right)
            area_ratio = min(_box_area(left), _box_area(right)) / max(_box_area(left), _box_area(right), 1.0)

            score = area_ratio * 3.0
            score -= vertical_gap * 6.0
            score -= abs(separation - 0.32) * 5.0
            score -= overlap * 8.0

            if left_cx >= face_center_x or right_cx <= face_center_x:
                score -= 2.5
            if separation < 0.16 or separation > 0.72:
                score -= 2.0
            if vertical_gap > 0.14:
                score -= 2.0

            if score > best_score:
                best_score = score
                best_pair = [left, right]

    if best_score < -0.25:
        return None
    return best_pair


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

    eye_boxes = _filter_eye_boxes(eye_boxes, face)
    eye_pair = _select_eye_pair(eye_boxes[:6], face)
    if eye_pair is not None:
        eye_boxes = eye_pair
    else:
        eye_boxes = _default_eye_boxes(w, h, face)

    mask = np.zeros((h, w), dtype=np.float32)
    face_width = float(face[2]) if face is not None else float(w)
    face_height = float(face[3]) if face is not None else float(h)
    for ex, ey, ew, eh in eye_boxes:
        cx = int(ex + ew * 0.5)
        cy = int(ey + eh * 0.5)
        rx = max(8, int(ew * 0.5 * eye_mask_scale))
        ry = max(6, int(eh * 0.6 * eye_mask_scale))
        rx = min(rx, int(max(10.0, face_width * 0.16)))
        ry = min(ry, int(max(8.0, face_height * 0.11)))
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


def collect_source_images(input_dir: Path, num_faces: int) -> list[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    all_sources: list[Path] = []
    for ext in exts:
        all_sources.extend(input_dir.glob(ext))

    all_sources = sorted(p for p in all_sources if p.is_file())
    if not all_sources:
        raise SystemExit(f"No source images found in {input_dir}")

    neutral_sources = [p for p in all_sources if p.stem.endswith("_neutral")]
    selected_pool = neutral_sources if neutral_sources else all_sources
    return selected_pool[:num_faces]


def resolve_existing_triplet(src: Path) -> tuple[Path, Path] | None:
    if not src.stem.endswith("_neutral"):
        return None

    prefix = src.stem[: -len("_neutral")]
    left = src.with_name(f"{prefix}_left{src.suffix}")
    right = src.with_name(f"{prefix}_right{src.suffix}")
    if left.exists() and right.exists():
        return left, right
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/ffhq_source")
    parser.add_argument("--output_dir", default="data/gaze_v2_1024")
    parser.add_argument("--num_faces", type=int, default=30)
    parser.add_argument("--gaze_strength", type=float, default=5)
    parser.add_argument("--size", type=int, default=1024)
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

    input_dir = Path(args.input_dir)
    sources = collect_source_images(input_dir, args.num_faces)

    print(f"Using {len(sources)} source faces from {args.input_dir}")
    print(f"Output: {args.output_dir} at {args.size}x{args.size}")
    print(f"Gaze strength: +/-{args.gaze_strength}\n")
    print(f"Blend mode: {args.blend_mode}")
    if len(sources) < args.num_faces:
        print(f"Requested {args.num_faces} faces, found only {len(sources)} usable images.")
    if any(src.stem.endswith("_neutral") for src in sources):
        print("Using *_neutral source images only.")

    triplets = [resolve_existing_triplet(src) for src in sources]
    use_existing_triplets = all(t is not None for t in triplets)
    if use_existing_triplets:
        print("Using existing *_left/*_neutral/*_right triplets from input_dir.")
        pipeline = None
    else:
        print("Building LivePortrait...")
        pipeline = build_liveportrait(args.device_id)

    for i, src in enumerate(sources):
        stem = src.stem
        if stem.endswith("_neutral"):
            stem = stem[: -len("_neutral")]
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

        triplet = resolve_existing_triplet(src)
        if triplet is not None:
            left_path, right_path = triplet
            neg_rgb = np.array(
                Image.open(left_path).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
            )
            pos_rgb = np.array(
                Image.open(right_path).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
            )
        else:
            neg_rgb = warp_gaze(pipeline, str(src), -args.gaze_strength, args.size)
            pos_rgb = warp_gaze(pipeline, str(src), +args.gaze_strength, args.size)

        if args.blend_mode == "eyes_only":
            neg_rgb = blend_eyes_only(neutral_rgb, neg_rgb, eye_mask)
            pos_rgb = blend_eyes_only(neutral_rgb, pos_rgb, eye_mask)

        Image.fromarray(neg_rgb).save(str(out / "neg" / f"{stem}.png"))
        Image.fromarray(pos_rgb).save(str(out / "pos" / f"{stem}.png"))
        print("  saved neg/neutral/pos")

    print(f"\nDone. Dataset at {args.output_dir}/")
    print(f"  neg/     — {len(sources)} left-gaze images")
    print(f"  pos/     — {len(sources)} right-gaze images")
    print(f"  neutral/ — {len(sources)} neutral images")
    if not args.skip_mask_write:
        print(f"  masks/   — {len(sources)} eye masks")


if __name__ == "__main__":
    main()
