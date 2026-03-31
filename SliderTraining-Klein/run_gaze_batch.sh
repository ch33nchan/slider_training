#!/bin/bash
# Run gaze pipeline on 10 diverse FFHQ faces
# Usage: bash run_gaze_batch.sh

CONFIG="config/eye_gaze_v8.yaml"
GAZE_STRENGTH=15
STRENGTH=0.4
SEED=42

FACES=(
    "data/ffhq_source/face_0000.png"
    "data/ffhq_source/face_0010.png"
    "data/ffhq_source/face_0020.png"
    "data/ffhq_source/face_0030.png"
    "data/ffhq_source/face_0040.png"
    "data/ffhq_source/face_0050.png"
    "data/ffhq_source/face_0060.png"
    "data/ffhq_source/face_0070.png"
    "data/ffhq_source/face_0080.png"
    "data/ffhq_source/face_0090.png"
)

for FACE in "${FACES[@]}"; do
    STEM=$(basename "$FACE" .png)
    OUT_DIR="outputs/pipeline_gaze_batch/$STEM"
    echo "Processing $STEM ..."
    python pipeline_gaze.py \
        --config "$CONFIG" \
        --source "$FACE" \
        --gaze_strength "$GAZE_STRENGTH" \
        --strength "$STRENGTH" \
        --seed "$SEED" \
        --output "$OUT_DIR/result.png"
    echo "  Done: $OUT_DIR/result.png"
done

echo ""
echo "All 10 done. Results in outputs/pipeline_gaze_batch/"
