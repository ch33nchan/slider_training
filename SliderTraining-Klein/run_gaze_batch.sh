#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/characters}"
MODE="${MODE:-PURE_LORA}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/character_lora}"
CONFIG="${CONFIG:-config/eye_gaze_horizontal_texture_v1.yaml}"
LORA_PATH="${LORA_PATH:-$(ls -1 outputs/gaze_horizontal_klein_texture/weights/slider_*.safetensors 2>/dev/null | sort | tail -n 1)}"
GPU_ID="${GPU_ID:-0}"

LEFT_GAZE="${LEFT_GAZE:--18}"
RIGHT_GAZE="${RIGHT_GAZE:-18}"
LEFT_SCALE="${LEFT_SCALE:--8}"
RIGHT_SCALE="${RIGHT_SCALE:-8}"
STRENGTH="${STRENGTH:-0.18}"
SIZE="${SIZE:-0}"
PREVIEW_SIZE="${PREVIEW_SIZE:-512}"
PROMPT="${PROMPT:-a person}"
SEED="${SEED:-42}"
SCALES="${SCALES:--8 0 8}"
SCALE_MULTIPLIER="${SCALE_MULTIPLIER:-1.6}"
SOURCE_LOCK="${SOURCE_LOCK:-0.08}"
EYE_BLEND_MODE="${EYE_BLEND_MODE:-adaptive}"
EYE_BLEND_STRENGTH="${EYE_BLEND_STRENGTH:-0.72}"
EYE_EDIT_MODE="${EYE_EDIT_MODE:-delta}"
DELTA_GAIN="${DELTA_GAIN:-2.4}"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "Missing input directory: ${INPUT_DIR}" >&2
    exit 1
fi

if [ -z "${LORA_PATH}" ] || [ ! -f "${LORA_PATH}" ]; then
    echo "Could not resolve a valid LoRA checkpoint. Set LORA_PATH explicitly." >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

mapfile -t FACES < <(find "${INPUT_DIR}" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | sort)

if [ "${#FACES[@]}" -eq 0 ]; then
    echo "No images found in ${INPUT_DIR}" >&2
    exit 1
fi

echo "Input dir: ${INPUT_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "LoRA: ${LORA_PATH}"
echo "Images: ${#FACES[@]}"
echo "Mode: ${MODE}"

for FACE in "${FACES[@]}"; do
    STEM="$(basename "${FACE}")"
    STEM="${STEM%.*}"
    OUT_DIR="${OUTPUT_ROOT}/${STEM}"
    mkdir -p "${OUT_DIR}"
    echo "Processing ${STEM} ..."
    if [ "${MODE}" = "PURE_LORA" ]; then
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 inference_slider.py \
            --config "${CONFIG}" \
            --lora_path "${LORA_PATH}" \
            --source_image "${FACE}" \
            --prompt "${PROMPT}" \
            --scales ${SCALES} \
            --left_scale "${LEFT_SCALE}" \
            --right_scale "${RIGHT_SCALE}" \
            --strength "${STRENGTH}" \
            --size "${SIZE}" \
            --scale_multiplier "${SCALE_MULTIPLIER}" \
            --source_lock "${SOURCE_LOCK}" \
            --eye_blend_mode "${EYE_BLEND_MODE}" \
            --eye_blend_strength "${EYE_BLEND_STRENGTH}" \
            --eye_edit_mode "${EYE_EDIT_MODE}" \
            --delta_gain "${DELTA_GAIN}" \
            --keep_source_at_zero \
            --save_eye_mask \
            --seed "${SEED}" \
            --output "${OUT_DIR}/result.png"
    else
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 pipeline_lora_gaze.py \
            --config "${CONFIG}" \
            --lora_path "${LORA_PATH}" \
            --source "${FACE}" \
            --left_gaze "${LEFT_GAZE}" \
            --right_gaze "${RIGHT_GAZE}" \
            --left_scale "${LEFT_SCALE}" \
            --right_scale "${RIGHT_SCALE}" \
            --size "${SIZE}" \
            --preview_size "${PREVIEW_SIZE}" \
            --strength "${STRENGTH}" \
            --prompt "${PROMPT}" \
            --seed "${SEED}" \
            --output "${OUT_DIR}/result.png"
    fi
    echo "  Done: ${OUT_DIR}/result.png"
done

echo ""
echo "All done. Results in ${OUTPUT_ROOT}/"
