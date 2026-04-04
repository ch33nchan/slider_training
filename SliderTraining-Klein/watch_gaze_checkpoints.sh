#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

INPUT_DIR="${INPUT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)/characters}"
CONFIG="${CONFIG:-config/eye_gaze_horizontal_texture_v1.yaml}"
WEIGHTS_DIR="${WEIGHTS_DIR:-outputs/gaze_horizontal_klein_texture/weights}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-outputs/character_lora_checkpoints}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT_BASE}/${RUN_NAME}"
GPU_ID="${GPU_ID:-2}"
MODE="${MODE:-PURE_LORA}"
POLL_SECONDS="${POLL_SECONDS:-30}"
PROCESS_EXISTING="${PROCESS_EXISTING:-0}"
STOP_AFTER_IDLE_POLLS="${STOP_AFTER_IDLE_POLLS:-0}"

LEFT_SCALE="${LEFT_SCALE:--8}"
RIGHT_SCALE="${RIGHT_SCALE:-8}"
STRENGTH="${STRENGTH:-0.10}"
SIZE="${SIZE:-0}"
SCALES="${SCALES:--8 0 8}"
SCALE_MULTIPLIER="${SCALE_MULTIPLIER:-1.5}"
SOURCE_LOCK="${SOURCE_LOCK:-0.10}"
EYE_BLEND_MODE="${EYE_BLEND_MODE:-adaptive}"
EYE_BLEND_STRENGTH="${EYE_BLEND_STRENGTH:-0.72}"
EYE_EDIT_MODE="${EYE_EDIT_MODE:-delta}"
DELTA_GAIN="${DELTA_GAIN:-2.2}"
CROP_MODE="${CROP_MODE:-eyes}"
CROP_PADDING="${CROP_PADDING:-3.0}"
CROP_THRESHOLD="${CROP_THRESHOLD:-0.08}"
CROP_FEATHER="${CROP_FEATHER:-0.12}"
PROMPT="${PROMPT:-a person}"
SEED="${SEED:-42}"

mkdir -p "${OUTPUT_ROOT}"
START_TS="$(date +%s)"
IDLE_POLLS=0

mtime_seconds() {
    if stat -c %Y "$1" >/dev/null 2>&1; then
        stat -c %Y "$1"
    else
        stat -f %m "$1"
    fi
}

process_checkpoint() {
    local checkpoint_path="$1"
    local checkpoint_name
    local checkpoint_stem
    local checkpoint_output

    checkpoint_name="$(basename "${checkpoint_path}")"
    checkpoint_stem="${checkpoint_name%.safetensors}"
    checkpoint_output="${OUTPUT_ROOT}/${checkpoint_stem}"

    if [ -f "${checkpoint_output}/.done" ]; then
        return 1
    fi

    echo "=== Processing ${checkpoint_name} ==="
    mkdir -p "${checkpoint_output}"

    INPUT_DIR="${INPUT_DIR}" \
    MODE="${MODE}" \
    OUTPUT_ROOT="${checkpoint_output}" \
    CONFIG="${CONFIG}" \
    LORA_PATH="${checkpoint_path}" \
    GPU_ID="${GPU_ID}" \
    LEFT_SCALE="${LEFT_SCALE}" \
    RIGHT_SCALE="${RIGHT_SCALE}" \
    STRENGTH="${STRENGTH}" \
    SIZE="${SIZE}" \
    SCALES="${SCALES}" \
    SCALE_MULTIPLIER="${SCALE_MULTIPLIER}" \
    SOURCE_LOCK="${SOURCE_LOCK}" \
    EYE_BLEND_MODE="${EYE_BLEND_MODE}" \
    EYE_BLEND_STRENGTH="${EYE_BLEND_STRENGTH}" \
    EYE_EDIT_MODE="${EYE_EDIT_MODE}" \
    DELTA_GAIN="${DELTA_GAIN}" \
    CROP_MODE="${CROP_MODE}" \
    CROP_PADDING="${CROP_PADDING}" \
    CROP_THRESHOLD="${CROP_THRESHOLD}" \
    CROP_FEATHER="${CROP_FEATHER}" \
    PROMPT="${PROMPT}" \
    SEED="${SEED}" \
    bash "${SCRIPT_DIR}/run_gaze_batch.sh" | tee "${checkpoint_output}/watch.log"

    date -Is > "${checkpoint_output}/.done"
    ln -sfn "${checkpoint_stem}" "${OUTPUT_ROOT}/latest"
    echo "=== Done ${checkpoint_name} -> ${checkpoint_output} ==="
    return 0
}

echo "Input dir: ${INPUT_DIR}"
echo "Weights dir: ${WEIGHTS_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run name: ${RUN_NAME}"
echo "Mode: ${MODE}"
echo "GPU: ${GPU_ID}"

while true; do
    found_new=0

    while IFS= read -r checkpoint_path; do
        checkpoint_name="$(basename "${checkpoint_path}")"
        [ "${checkpoint_name}" = "slider_latest.safetensors" ] && continue

        checkpoint_mtime="$(mtime_seconds "${checkpoint_path}")"
        if [ "${PROCESS_EXISTING}" != "1" ] && [ "${checkpoint_mtime}" -lt "${START_TS}" ]; then
            continue
        fi

        if process_checkpoint "${checkpoint_path}"; then
            found_new=1
        fi
    done < <(find "${WEIGHTS_DIR}" -maxdepth 1 -type f -name 'slider_*.safetensors' | sort)

    if [ "${found_new}" -eq 1 ]; then
        IDLE_POLLS=0
    else
        IDLE_POLLS=$((IDLE_POLLS + 1))
    fi

    if [ "${STOP_AFTER_IDLE_POLLS}" -gt 0 ] && [ "${IDLE_POLLS}" -ge "${STOP_AFTER_IDLE_POLLS}" ]; then
        echo "No new checkpoints after ${IDLE_POLLS} polls. Stopping."
        break
    fi

    sleep "${POLL_SECONDS}"
done
