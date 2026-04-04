#!/usr/bin/env bash
set -euo pipefail

# End-to-end eye-gaze training pipeline:
# 1) Generate LivePortrait-based paired datasets with eye-only blending
# 2) Train FLUX Klein sliders (horizontal + vertical) with non-eye preservation
# 3) Train FLUX.1-dev text sliders (horizontal + vertical) via flux-sliders
#
# Override defaults with env vars as needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FLUX_SLIDERS_DIR="${FLUX_SLIDERS_DIR:-${PROJECT_ROOT}/flux-sliders}"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_DIR="${SOURCE_DIR:-${PROJECT_ROOT}/LivePortrait/source_faces}"

H_DATASET_OUT="${H_DATASET_OUT:-${SCRIPT_DIR}/data/gaze_horizontal_texture}"
V_DATASET_OUT="${V_DATASET_OUT:-${SCRIPT_DIR}/data/gaze_vertical_texture}"

H_GAZE_STRENGTH="${H_GAZE_STRENGTH:-8}"
V_GAZE_STRENGTH="${V_GAZE_STRENGTH:-12}"
DATASET_SIZE="${DATASET_SIZE:-1024}"
NUM_FACES="${NUM_FACES:-80}"

GPU_DATA_H="${GPU_DATA_H:-0}"
GPU_DATA_V="${GPU_DATA_V:-1}"
GPU_KLEIN_H="${GPU_KLEIN_H:-0}"
GPU_KLEIN_V="${GPU_KLEIN_V:-1}"
GPU_FLUX_H="${GPU_FLUX_H:-2}"
GPU_FLUX_V="${GPU_FLUX_V:-3}"

LOG_DIR="${LOG_DIR:-/tmp/eye_gaze_pipeline_logs}"
mkdir -p "${LOG_DIR}"

echo "=== Stage 1/3: Generate horizontal dataset (eye-only blend) ==="
CUDA_VISIBLE_DEVICES="${GPU_DATA_H}" "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_gaze_dataset_v2.py" \
  --input_dir "${SOURCE_DIR}" \
  --output_dir "${H_DATASET_OUT}" \
  --num_faces "${NUM_FACES}" \
  --gaze_strength "${H_GAZE_STRENGTH}" \
  --size "${DATASET_SIZE}" \
  --device_id 0 \
  --blend_mode eyes_only

echo "=== Stage 1/3: Generate vertical dataset (eye-only blend) ==="
CUDA_VISIBLE_DEVICES="${GPU_DATA_V}" "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_gaze_dataset_vertical.py" \
  --input_dir "${SOURCE_DIR}" \
  --output_dir "${V_DATASET_OUT}" \
  --num_faces "${NUM_FACES}" \
  --gaze_strength "${V_GAZE_STRENGTH}" \
  --size "${DATASET_SIZE}" \
  --device_id 0 \
  --blend_mode eyes_only

echo "=== Stage 2/3: Launch Klein training jobs ==="
CUDA_VISIBLE_DEVICES="${GPU_KLEIN_H}" "${PYTHON_BIN}" "${SCRIPT_DIR}/train_slider.py" \
  --config "${SCRIPT_DIR}/config/eye_gaze_horizontal_texture_v1.yaml" \
  > "${LOG_DIR}/klein_horizontal.log" 2>&1 &
PID_KLEIN_H=$!
echo "Klein horizontal PID: ${PID_KLEIN_H}"

CUDA_VISIBLE_DEVICES="${GPU_KLEIN_V}" "${PYTHON_BIN}" "${SCRIPT_DIR}/train_slider.py" \
  --config "${SCRIPT_DIR}/config/eye_gaze_vertical_texture_v1.yaml" \
  > "${LOG_DIR}/klein_vertical.log" 2>&1 &
PID_KLEIN_V=$!
echo "Klein vertical PID: ${PID_KLEIN_V}"

echo "=== Stage 3/3: Launch FLUX.1-dev slider training jobs ==="
pushd "${FLUX_SLIDERS_DIR}" > /dev/null

CUDA_VISIBLE_DEVICES="${GPU_FLUX_H}" "${PYTHON_BIN}" -c \
"from flux_sliders.text_sliders import FLUXTextSliders; FLUXTextSliders('config/gaze_horizontal_flux.yaml').train()" \
  > "${LOG_DIR}/flux_horizontal.log" 2>&1 &
PID_FLUX_H=$!
echo "FLUX horizontal PID: ${PID_FLUX_H}"

CUDA_VISIBLE_DEVICES="${GPU_FLUX_V}" "${PYTHON_BIN}" -c \
"from flux_sliders.text_sliders import FLUXTextSliders; FLUXTextSliders('config/gaze_vertical_flux.yaml').train()" \
  > "${LOG_DIR}/flux_vertical.log" 2>&1 &
PID_FLUX_V=$!
echo "FLUX vertical PID: ${PID_FLUX_V}"

popd > /dev/null

echo "=== All jobs launched ==="
echo "Logs:"
echo "  ${LOG_DIR}/klein_horizontal.log"
echo "  ${LOG_DIR}/klein_vertical.log"
echo "  ${LOG_DIR}/flux_horizontal.log"
echo "  ${LOG_DIR}/flux_vertical.log"

wait "${PID_KLEIN_H}" && echo "Klein horizontal complete"
wait "${PID_KLEIN_V}" && echo "Klein vertical complete"
wait "${PID_FLUX_H}" && echo "FLUX horizontal complete"
wait "${PID_FLUX_V}" && echo "FLUX vertical complete"

echo "=== Pipeline complete ==="
