#!/bin/bash
# Generate vertical gaze dataset using LivePortrait.
# Run this BEFORE run_all_training.sh
#
# Usage:
#   source /mnt/data1/srini/eyegaze/slider_training/venv/bin/activate
#   cd /mnt/data1/srini/eyegaze/slider_training/SliderTraining-Klein
#   bash run_generate_vertical.sh

set -e

VENV=/mnt/data1/srini/eyegaze/slider_training/venv/bin/activate
source $VENV

cd /mnt/data1/srini/eyegaze/slider_training/SliderTraining-Klein

echo "=== Generating vertical gaze dataset (strength=15, size=512) ==="
CUDA_VISIBLE_DEVICES=3 python generate_gaze_dataset_vertical.py \
  --input_dir ../LivePortrait/source_faces \
  --output_dir data/gaze_vertical_s15 \
  --gaze_strength 15 \
  --size 512 \
  --device_id 3

echo "=== Done! Vertical gaze dataset at data/gaze_vertical_s15/ ==="
