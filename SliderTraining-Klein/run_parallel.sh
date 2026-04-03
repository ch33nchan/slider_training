#!/bin/bash
set -e

BASE=/mnt/data1/srini/eyegaze/slider_training/SliderTraining-Klein
VENV=$BASE/ai-toolkit/venv/bin/activate

source $VENV

echo "=== Starting text slider training on GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 python $BASE/ai-toolkit/run.py \
  $BASE/ai-toolkit/config/eye_gaze_flux_dev_v1.yaml \
  2>&1 | tee /tmp/gaze_text_train.log &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

echo "=== Starting inference test on GPU 1 ==="
CUDA_VISIBLE_DEVICES=1 python $BASE/infer_gaze_img2img.py \
  --lora_path $BASE/ai-toolkit/output/eye_gaze_flux_dev_lp/eye_gaze_flux_dev_lp.safetensors \
  --input_image /mnt/data1/srini/eyegaze/slider_training/LivePortrait/source_faces/face_07.png \
  --output_dir $BASE/outputs/gaze_img2img_lp_fixed \
  --strength 0.5 \
  2>&1 | tee /tmp/gaze_infer_fixed.log &
INFER_PID=$!
echo "Inference PID: $INFER_PID"

echo "=== Both running. Waiting... ==="
wait $INFER_PID
echo "=== Inference done ==="
wait $TRAIN_PID
echo "=== Training done ==="
