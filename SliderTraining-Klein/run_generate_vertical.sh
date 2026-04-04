#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_ROOT}/LivePortrait/source_faces}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/data/gaze_vertical_texture}"
NUM_FACES="${NUM_FACES:-80}"
SIZE="${SIZE:-1024}"
GAZE_STRENGTH="${GAZE_STRENGTH:-12}"
GPU_ID="${GPU_ID:-0}"

if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

cd "${SCRIPT_DIR}"

echo "=== Generating vertical gaze dataset with eye-only blending ==="
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" generate_gaze_dataset_vertical.py \
  --input_dir "${SOURCE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_faces "${NUM_FACES}" \
  --gaze_strength "${GAZE_STRENGTH}" \
  --size "${SIZE}" \
  --device_id 0 \
  --blend_mode eyes_only

echo "=== Done: ${OUTPUT_DIR} ==="
