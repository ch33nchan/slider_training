"""
Eye gaze control using LivePortrait retargeting.
Shifts gaze left/right while perfectly preserving identity.

Usage:
  python infer_gaze_liveportrait.py \
    --input_image /path/to/face.png \
    --output_dir outputs/gaze_liveportrait \
    --liveportrait_dir /mnt/data1/srini/eyegaze/slider_training/LivePortrait
"""

import argparse
import os
import sys
import cv2
from PIL import Image, ImageDraw
import numpy as np


def resolve_output_size(image_path: str, requested_size: int) -> tuple[int, int]:
    src = Image.open(image_path).convert("RGB")
    if requested_size <= 0:
        return src.size
    return requested_size, requested_size


def save_rgb_image(rgb_image: np.ndarray, output_path: str, size: tuple[int, int]) -> Image.Image:
    if tuple(rgb_image.shape[1::-1]) != size:
        rgb_image = cv2.resize(rgb_image, size, interpolation=cv2.INTER_LANCZOS4)
    image = Image.fromarray(rgb_image)
    image.save(output_path)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--output_dir", default="outputs/gaze_liveportrait")
    parser.add_argument("--liveportrait_dir",
                        default="/mnt/data1/srini/eyegaze/slider_training/LivePortrait")
    parser.add_argument("--max_gaze", type=float, default=35.0,
                        help="Max eyeball_direction_x value (positive=right, negative=left)")
    parser.add_argument("--steps", type=int, default=7,
                        help="Number of steps across the gaze range")
    parser.add_argument("--size", type=int, default=0,
                        help="Square output size. Use 0 to keep original full resolution.")
    parser.add_argument("--left_gaze", type=float, default=-18.0,
                        help="Explicit left gaze output.")
    parser.add_argument("--right_gaze", type=float, default=18.0,
                        help="Explicit right gaze output.")
    parser.add_argument("--preview_size", type=int, default=512,
                        help="Preview tile size for strip.png.")
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_w, output_h = resolve_output_size(args.input_image, args.size)

    sys.path.insert(0, args.liveportrait_dir)
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.gradio_pipeline import GradioPipeline

    print("Building LivePortrait pipeline...")
    lp_args = ArgumentConfig()
    lp_args.device_id = args.device_id
    lp_args.flag_use_half_precision = True
    lp_args.flag_pasteback = True
    lp_args.flag_do_crop = True
    inference_cfg = InferenceConfig()
    inference_cfg.device_id = args.device_id
    inference_cfg.flag_use_half_precision = True
    crop_cfg = CropConfig()
    pipeline = GradioPipeline(inference_cfg, crop_cfg, lp_args)

    # Gaze values: evenly spaced from -max to +max
    gaze_values = np.linspace(-args.max_gaze, args.max_gaze, args.steps).tolist()

    print(f"Processing {len(gaze_values)} gaze directions: {[f'{g:+.1f}' for g in gaze_values]}")
    print(f"Output size: {output_w}x{output_h}")

    input_path = args.input_image
    eye_ratio, lip_ratio = pipeline.init_retargeting_image(
        retargeting_source_scale=2.3,
        source_eye_ratio=0.4,
        source_lip_ratio=0.0,
        input_image=input_path,
    )

    frames = []
    for gaze_x in gaze_values:
        print(f"  gaze_x={gaze_x:+.1f}", end="", flush=True)
        _, out_blended = pipeline.execute_image_retargeting(
            input_eye_ratio=eye_ratio,
            input_lip_ratio=lip_ratio,
            input_head_pitch_variation=0.0,
            input_head_yaw_variation=0.0,
            input_head_roll_variation=0.0,
            mov_x=0.0,
            mov_y=0.0,
            mov_z=1.0,
            lip_variation_zero=0.0,
            lip_variation_one=0.0,
            lip_variation_two=0.0,
            lip_variation_three=0.0,
            smile=0.0,
            wink=0.0,
            eyebrow=0.0,
            eyeball_direction_x=float(gaze_x),
            eyeball_direction_y=0.0,
            input_image=input_path,
            retargeting_source_scale=2.3,
            flag_stitching_retargeting_input=True,
            flag_do_crop_input_retargeting_image=True,
        )
        out_rgb = np.asarray(out_blended, dtype=np.uint8)
        fname = os.path.join(args.output_dir, f"gaze_{gaze_x:+.1f}.png")
        frames.append(save_rgb_image(out_rgb, fname, (output_w, output_h)))
        print(f" -> saved", flush=True)

    explicit_outputs = [
        ("left_full.png", args.left_gaze),
        ("right_full.png", args.right_gaze),
    ]
    for filename, gaze_x in explicit_outputs:
        print(f"  explicit {filename}: gaze_x={gaze_x:+.1f}", end="", flush=True)
        _, out_blended = pipeline.execute_image_retargeting(
            input_eye_ratio=eye_ratio,
            input_lip_ratio=lip_ratio,
            input_head_pitch_variation=0.0,
            input_head_yaw_variation=0.0,
            input_head_roll_variation=0.0,
            mov_x=0.0,
            mov_y=0.0,
            mov_z=1.0,
            lip_variation_zero=0.0,
            lip_variation_one=0.0,
            lip_variation_two=0.0,
            lip_variation_three=0.0,
            smile=0.0,
            wink=0.0,
            eyebrow=0.0,
            eyeball_direction_x=float(gaze_x),
            eyeball_direction_y=0.0,
            input_image=input_path,
            retargeting_source_scale=2.3,
            flag_stitching_retargeting_input=True,
            flag_do_crop_input_retargeting_image=True,
        )
        save_rgb_image(
            np.asarray(out_blended, dtype=np.uint8),
            os.path.join(args.output_dir, filename),
            (output_w, output_h),
        )
        print(" -> saved", flush=True)

    # Load input for strip
    input_img = Image.open(input_path).convert("RGB")
    strip_input = input_img.resize((args.preview_size, args.preview_size), Image.LANCZOS)

    # Build strip: INPUT + all gaze frames
    strip_frames = [frame.resize((args.preview_size, args.preview_size), Image.LANCZOS) for frame in frames]
    all_frames = [strip_input] + strip_frames
    label_h = 40
    w, h = args.preview_size, args.preview_size
    strip = Image.new("RGB", (w * len(all_frames), h + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(strip)

    labels = ["INPUT"] + [f"{g:+.1f}" for g in gaze_values]
    for i, (frame, label) in enumerate(zip(all_frames, labels)):
        strip.paste(frame, (i * w, label_h))
        draw.text((i * w + w // 2 - 20, 10), label, fill=(0, 0, 0))

    strip_path = os.path.join(args.output_dir, "strip.png")
    strip.save(strip_path)
    print(f"\nStrip saved: {strip_path}")
    print(f"Full-resolution outputs:")
    print(f"  {os.path.join(args.output_dir, 'left_full.png')}")
    print(f"  {os.path.join(args.output_dir, 'right_full.png')}")
    print(f"Done! Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
