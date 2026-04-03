#!/bin/bash
# Launch all 4 gaze slider training runs in parallel across 4 GPUs.
# Prerequisites:
#   - gaze_s15 dataset already exists for horizontal Klein training
#   - gaze_vertical_s15 dataset must be generated first (run generate_gaze_dataset_vertical.py)
#
# Usage:
#   source /mnt/data1/srini/eyegaze/slider_training/venv/bin/activate
#   cd /mnt/data1/srini/eyegaze/slider_training/SliderTraining-Klein
#   bash run_all_training.sh

set -e

BASE=/mnt/data1/srini/eyegaze/slider_training/SliderTraining-Klein
FLUX_SLIDERS=/mnt/data1/srini/eyegaze/slider_training/sliders/flux-sliders
VENV=/mnt/data1/srini/eyegaze/slider_training/venv/bin/activate

source $VENV
cd $BASE

# --- GPU 0: Klein Horizontal Gaze ---
echo "=== [GPU 0] Klein Horizontal Gaze (4000 steps) ==="
CUDA_VISIBLE_DEVICES=0 python train_slider.py \
  --config config/eye_gaze_horizontal_v1.yaml \
  2>&1 | tee /tmp/klein_horizontal.log &
PID0=$!
echo "PID: $PID0"

# --- GPU 1: Klein Vertical Gaze ---
echo "=== [GPU 1] Klein Vertical Gaze (4000 steps) ==="
CUDA_VISIBLE_DEVICES=1 python train_slider.py \
  --config config/eye_gaze_vertical_v1.yaml \
  2>&1 | tee /tmp/klein_vertical.log &
PID1=$!
echo "PID: $PID1"

# --- GPU 2: FLUX Horizontal Gaze ---
echo "=== [GPU 2] FLUX.1-dev Horizontal Gaze (4000 steps) ==="
cd $FLUX_SLIDERS
CUDA_VISIBLE_DEVICES=2 python -c "
from flux_sliders.text_sliders import FLUXTextSliders
model = FLUXTextSliders('config/gaze_horizontal_flux.yaml')
model.train()
" 2>&1 | tee /tmp/flux_horizontal.log &
PID2=$!
echo "PID: $PID2"

# --- GPU 3: FLUX Vertical Gaze ---
echo "=== [GPU 3] FLUX.1-dev Vertical Gaze (4000 steps) ==="
CUDA_VISIBLE_DEVICES=3 python -c "
from flux_sliders.text_sliders import FLUXTextSliders
model = FLUXTextSliders('config/gaze_vertical_flux.yaml')
model.train()
" 2>&1 | tee /tmp/flux_vertical.log &
PID3=$!
echo "PID: $PID3"

echo ""
echo "=== All 4 training runs launched ==="
echo "  GPU 0: Klein horizontal  -> /tmp/klein_horizontal.log"
echo "  GPU 1: Klein vertical    -> /tmp/klein_vertical.log"
echo "  GPU 2: FLUX horizontal   -> /tmp/flux_horizontal.log"
echo "  GPU 3: FLUX vertical     -> /tmp/flux_vertical.log"
echo ""
echo "Monitor with: tail -f /tmp/klein_horizontal.log"
echo "           or: tail -f /tmp/flux_horizontal.log"
echo ""

wait $PID0 && echo "=== Klein Horizontal DONE ===" || echo "=== Klein Horizontal FAILED ==="
wait $PID1 && echo "=== Klein Vertical DONE ===" || echo "=== Klein Vertical FAILED ==="
wait $PID2 && echo "=== FLUX Horizontal DONE ===" || echo "=== FLUX Horizontal FAILED ==="
wait $PID3 && echo "=== FLUX Vertical DONE ===" || echo "=== FLUX Vertical FAILED ==="

echo ""
echo "=== All training complete ==="
echo "  Klein horizontal LoRA: $BASE/outputs/gaze_horizontal_klein/"
echo "  Klein vertical LoRA:   $BASE/outputs/gaze_vertical_klein/"
echo "  FLUX horizontal LoRA:  $FLUX_SLIDERS/outputs/gaze_horizontal_flux/"
echo "  FLUX vertical LoRA:    $FLUX_SLIDERS/outputs/gaze_vertical_flux/"
