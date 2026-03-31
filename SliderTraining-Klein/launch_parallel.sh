#!/bin/bash
# Launch 3 Klein slider experiments in parallel on GPUs 1, 2, 3.
# GPU 0 is left untouched (main.py is running there).
#
# Usage: bash launch_parallel.sh
# Logs go to: outputs/eye_gaze_v{4,5,6}/train.log

set -e
cd "$(dirname "$0")"

mkdir -p outputs/eye_gaze_v4 outputs/eye_gaze_v5 outputs/eye_gaze_v6

echo "=== Launching eye_gaze_v4 (GPU 1, rank=32, train_lora_up=True, eta=6) ==="
python train_slider.py --config config/eye_gaze_v4.yaml \
    > outputs/eye_gaze_v4/train.log 2>&1 &
PID4=$!
echo "  PID: $PID4"

echo "=== Launching eye_gaze_v5 (GPU 2, rank=64, eta=8, 5000 steps) ==="
python train_slider.py --config config/eye_gaze_v5.yaml \
    > outputs/eye_gaze_v5/train.log 2>&1 &
PID5=$!
echo "  PID: $PID5"

echo "=== Launching eye_gaze_v6 (GPU 3, rank=32, bidirectional=True, eta=6) ==="
python train_slider.py --config config/eye_gaze_v6.yaml \
    > outputs/eye_gaze_v6/train.log 2>&1 &
PID6=$!
echo "  PID: $PID6"

echo ""
echo "All 3 experiments launched. Monitor with:"
echo "  tail -f outputs/eye_gaze_v4/train.log"
echo "  tail -f outputs/eye_gaze_v5/train.log"
echo "  tail -f outputs/eye_gaze_v6/train.log"
echo ""
echo "Wait for all to finish:"
wait $PID4 && echo "v4 done" || echo "v4 FAILED"
wait $PID5 && echo "v5 done" || echo "v5 FAILED"
wait $PID6 && echo "v6 done" || echo "v6 FAILED"
echo "All experiments finished."
