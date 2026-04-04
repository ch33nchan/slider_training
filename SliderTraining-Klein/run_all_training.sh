#!/usr/bin/env bash
set -euo pipefail

# Portable launcher for 4 concurrent runs:
# 1) Klein horizontal eye-gaze slider (texture-preserving config)
# 2) Klein vertical eye-gaze slider (texture-preserving config)
# 3) FLUX.1-dev horizontal slider (flux-sliders)
# 4) FLUX.1-dev vertical slider (flux-sliders)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FLUX_SLIDERS_DIR="${FLUX_SLIDERS_DIR:-${PROJECT_ROOT}/flux-sliders}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${LOG_DIR:-/tmp/eye_gaze_training_logs}"
KLEIN_CFG_H="${KLEIN_CFG_H:-config/eye_gaze_horizontal_texture_v1.yaml}"
KLEIN_CFG_V="${KLEIN_CFG_V:-config/eye_gaze_vertical_texture_v1.yaml}"
FLUX_CFG_H="${FLUX_CFG_H:-config/gaze_horizontal_flux.yaml}"
FLUX_CFG_V="${FLUX_CFG_V:-config/gaze_vertical_flux.yaml}"
mkdir -p "${LOG_DIR}"

# Optional venv activation:
#   export VENV_PATH=/path/to/venv
if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

cd "${SCRIPT_DIR}"

echo "=== [GPU ${GPU_KLEIN_H:-0}] Klein Horizontal Gaze ==="
CUDA_VISIBLE_DEVICES="${GPU_KLEIN_H:-0}" "${PYTHON_BIN}" train_slider.py \
  --config "${KLEIN_CFG_H}" \
  > "${LOG_DIR}/klein_horizontal.log" 2>&1 &
PID0=$!
echo "PID: ${PID0}"

echo "=== [GPU ${GPU_KLEIN_V:-1}] Klein Vertical Gaze ==="
CUDA_VISIBLE_DEVICES="${GPU_KLEIN_V:-1}" "${PYTHON_BIN}" train_slider.py \
  --config "${KLEIN_CFG_V}" \
  > "${LOG_DIR}/klein_vertical.log" 2>&1 &
PID1=$!
echo "PID: ${PID1}"

echo "=== [GPU ${GPU_FLUX_H:-2}] FLUX.1-dev Horizontal Gaze ==="
(
  cd "${FLUX_SLIDERS_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_FLUX_H:-2}" "${PYTHON_BIN}" -c \
  "from flux_sliders.text_sliders import FLUXTextSliders; FLUXTextSliders('${FLUX_CFG_H}').train()"
) > "${LOG_DIR}/flux_horizontal.log" 2>&1 &
PID2=$!
echo "PID: ${PID2}"

echo "=== [GPU ${GPU_FLUX_V:-3}] FLUX.1-dev Vertical Gaze ==="
(
  cd "${FLUX_SLIDERS_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_FLUX_V:-3}" "${PYTHON_BIN}" -c \
  "from flux_sliders.text_sliders import FLUXTextSliders('${FLUX_CFG_V}').train()"
) > "${LOG_DIR}/flux_vertical.log" 2>&1 &
PID3=$!
echo "PID: ${PID3}"

echo ""
echo "=== All 4 training runs launched ==="
echo "  Klein horizontal  -> ${LOG_DIR}/klein_horizontal.log"
echo "  Klein vertical    -> ${LOG_DIR}/klein_vertical.log"
echo "  FLUX horizontal   -> ${LOG_DIR}/flux_horizontal.log"
echo "  FLUX vertical     -> ${LOG_DIR}/flux_vertical.log"
echo ""

wait "${PID0}" && echo "=== Klein Horizontal DONE ===" || echo "=== Klein Horizontal FAILED ==="
wait "${PID1}" && echo "=== Klein Vertical DONE ===" || echo "=== Klein Vertical FAILED ==="
wait "${PID2}" && echo "=== FLUX Horizontal DONE ===" || echo "=== FLUX Horizontal FAILED ==="
wait "${PID3}" && echo "=== FLUX Vertical DONE ===" || echo "=== FLUX Vertical FAILED ==="

echo ""
echo "=== All training complete ==="
echo "  Klein horizontal: ${SCRIPT_DIR}/outputs/gaze_horizontal_klein_texture/"
echo "  Klein vertical:   ${SCRIPT_DIR}/outputs/gaze_vertical_klein_texture/"
echo "  FLUX outputs:     ${FLUX_SLIDERS_DIR}/outputs/"
