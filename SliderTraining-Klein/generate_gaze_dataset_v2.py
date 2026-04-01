"""
Generate 30-pair gaze dataset at 1024x1024 using LivePortrait.

For each source face produces:
  neg/<stem>.png   — gaze left  (eyeball_direction_x = -gaze_strength)
  pos/<stem>.png   — gaze right (eyeball_direction_x = +gaze_strength)
  neutral/<stem>.png — no warp (source resized)

Usage (run from SliderTraining-Klein/):
  python generate_gaze_dataset_v2.py \
    --input_dir data/ffhq_source \
    --output_dir data/gaze_v2_1024 \
    --num_faces 30 \
    --gaze_strength 5 \
    --size 1024 \
    --device_id 3
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
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
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/ffhq_source")
    parser.add_argument("--output_dir", default="data/gaze_v2_1024")
    parser.add_argument("--num_faces", type=int, default=30)
    parser.add_argument("--gaze_strength", type=float, default=5)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--device_id", type=int, default=3)
    args = parser.parse_args()

    out = Path(args.output_dir)
    for split in ["neg", "pos", "neutral"]:
        (out / split).mkdir(parents=True, exist_ok=True)

    sources = sorted(Path(args.input_dir).glob("*.png"))[:args.num_faces]
    print(f"Using {len(sources)} source faces from {args.input_dir}")
    print(f"Output: {args.output_dir} at {args.size}x{args.size}")
    print(f"Gaze strength: +/-{args.gaze_strength}\n")

    print("Building LivePortrait...")
    pipeline = build_liveportrait(args.device_id)

    for i, src in enumerate(sources):
        stem = src.stem
        print(f"[{i+1}/{len(sources)}] {stem}")

        # neutral — just resize source
        neutral = Image.open(src).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
        neutral.save(str(out / "neutral" / f"{stem}.png"))

        # neg (left gaze)
        neg = warp_gaze(pipeline, str(src), -args.gaze_strength, args.size)
        neg.save(str(out / "neg" / f"{stem}.png"))

        # pos (right gaze)
        pos = warp_gaze(pipeline, str(src), +args.gaze_strength, args.size)
        pos.save(str(out / "pos" / f"{stem}.png"))

        print(f"  saved neg/neutral/pos")

    print(f"\nDone. Dataset at {args.output_dir}/")
    print(f"  neg/     — {len(sources)} left-gaze images")
    print(f"  pos/     — {len(sources)} right-gaze images")
    print(f"  neutral/ — {len(sources)} neutral images")


if __name__ == "__main__":
    main()
