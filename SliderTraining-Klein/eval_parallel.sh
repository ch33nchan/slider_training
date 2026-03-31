#!/bin/bash
# Run inference on all 3 parallel experiments using a test face.
# Produces comparison grids in outputs/eval/
#
# Usage: bash eval_parallel.sh /path/to/test_face.jpg
# If no test face given, uses the first image from data/eye_gaze_v3/neutral/

set -e
cd "$(dirname "$0")"

SOURCE="${1:-}"
if [ -z "$SOURCE" ]; then
    SOURCE=$(ls data/eye_gaze_v3/neutral/*.png 2>/dev/null | head -1)
    if [ -z "$SOURCE" ]; then
        echo "Error: no test face found. Pass a path as first arg."
        exit 1
    fi
fi
echo "Test face: $SOURCE"

mkdir -p outputs/eval

run_inference() {
    local VERSION=$1
    local LORA_PATH="outputs/eye_gaze_${VERSION}/weights/slider_latest.safetensors"
    local CONFIG="config/eye_gaze_${VERSION}.yaml"
    local OUT="outputs/eval/result_${VERSION}.png"

    if [ ! -f "$LORA_PATH" ]; then
        echo "[$VERSION] SKIP: weights not found at $LORA_PATH"
        return
    fi

    echo "[$VERSION] Running inference (device from config)..."
    python inference_slider.py \
        --config "$CONFIG" \
        --lora_path "$LORA_PATH" \
        --source_image "$SOURCE" \
        --prompt "a person" \
        --scales -5 -2.5 0 2.5 5 \
        --output "$OUT" \
        > "outputs/eval/infer_${VERSION}.log" 2>&1
    echo "[$VERSION] Done -> $OUT"
}

# Run sequentially to avoid VRAM conflicts (each needs ~60GB)
run_inference "v4"
run_inference "v5"
run_inference "v6"

echo ""
echo "All inference done. Results in outputs/eval/"
ls -la outputs/eval/result_*.png 2>/dev/null || true
