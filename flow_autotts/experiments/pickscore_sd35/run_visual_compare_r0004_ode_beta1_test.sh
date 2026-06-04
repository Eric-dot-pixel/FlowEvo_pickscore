#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[pickscore-visual-compare-beta1] python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

DEVICES="${FLOW_TTS_EVAL_DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3}"
TEXT_ENCODER_DEVICES="${FLOW_TTS_EVAL_TEXT_ENCODER_DEVICES:-${DEVICES}}"
SCORE_DEVICES="${FLOW_TTS_EVAL_SCORE_DEVICES:-${DEVICES}}"
MODEL="${FLOW_TTS_MODEL:-${REPO_ROOT}/SD_3.5_med}"
PICKSCORE_MODEL="${FLOW_TTS_PICKSCORE_MODEL:-${REPO_ROOT}/PickScore_v1}"
PICKSCORE_PROCESSOR="${FLOW_TTS_PICKSCORE_PROCESSOR:-}"
DTYPE="${FLOW_TTS_DTYPE:-bfloat16}"
SCORE_DTYPE="${FLOW_TTS_SCORE_DTYPE:-float32}"
RESOLUTION="${FLOW_TTS_RESOLUTION:-512}"
GUIDANCE_SCALE="${FLOW_TTS_GUIDANCE_SCALE:-4.5}"
NOISE_LEVEL="${FLOW_TTS_NOISE_LEVEL:-0.7}"
SDE_TYPE="${FLOW_TTS_SDE_TYPE:-sde}"
SAMPLE_SEED="${FLOW_TTS_SAMPLE_SEED:-42}"
SAMPLE_SIZE="${FLOW_TTS_SAMPLE_SIZE:-2048}"
CONTROLLER_NUM_STEPS="${FLOW_TTS_NUM_STEPS:-10}"
DATASET="${FLOW_TTS_DATASET:-${REPO_ROOT}/flow_grpo/dataset/pickscore}"
CONTROLLER_PATH="${FLOW_TTS_CONTROLLER_PATH:-${REPO_ROOT}/logs/flow_autotts/pickscore_sd35/history_autotts_b64_fixed_target_reference_20260527_160759/r0004_20260527_160800_ffd4e330/flow_autotts/controllers/optimal.py}"
CONTROLLER_KEY="${FLOW_TTS_CONTROLLER_KEY:-r0004_20260527_160800_ffd4e330}"
BETA="${FLOW_TTS_BETA:-1.0}"
BUDGET="${FLOW_TTS_BUDGET:-64}"
BASELINE_TOTAL_NFE="${FLOW_TTS_BASELINE_TOTAL_NFE:-64}"

RESULT_TAG="${RESULT_TAG:-r0004_vs_ode_b64_beta1_visual_compare_test}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/logs/flow_autotts/pickscore_sd35/${RESULT_TAG}}"
IMAGE_ROOT="${IMAGE_ROOT:-${OUTPUT_DIR}/samples}"
SHARD_OUTPUT_DIR="${SHARD_OUTPUT_DIR:-${OUTPUT_DIR}/shards}"
WORKFLOW_LOG_DIR="${WORKFLOW_LOG_DIR:-${OUTPUT_DIR}/workflow_logs}"

BOOTSTRAP_LOG="${WORKFLOW_LOG_DIR}/visual_compare.bootstrap.log"
STDOUT_LOG="${WORKFLOW_LOG_DIR}/visual_compare.stdout.log"
STDERR_LOG="${WORKFLOW_LOG_DIR}/visual_compare.stderr.log"
ENV_LOG="${WORKFLOW_LOG_DIR}/visual_compare.env.log"

mkdir -p "${WORKFLOW_LOG_DIR}" "${OUTPUT_DIR}" "${IMAGE_ROOT}" "${SHARD_OUTPUT_DIR}"
exec >>"${BOOTSTRAP_LOG}" 2>&1

if [[ ! -f "${CONTROLLER_PATH}" ]]; then
  echo "[pickscore-visual-compare-beta1] controller not found: ${CONTROLLER_PATH}" >&2
  exit 2
fi

COMMON_ARGS=(
  --devices "${DEVICES}"
  --text-encoder-devices "${TEXT_ENCODER_DEVICES}"
  --score-devices "${SCORE_DEVICES}"
  --dataset "${DATASET}"
  --split test
  --sample-size "${SAMPLE_SIZE}"
  --sample-seed "${SAMPLE_SEED}"
  --beta "${BETA}"
  --budget "${BUDGET}"
  --baseline-total-nfe "${BASELINE_TOTAL_NFE}"
  --controller-num-steps "${CONTROLLER_NUM_STEPS}"
  --output-dir "${OUTPUT_DIR}"
  --image-root "${IMAGE_ROOT}"
  --shard-output-dir "${SHARD_OUTPUT_DIR}"
  --model "${MODEL}"
  --pickscore-model "${PICKSCORE_MODEL}"
  --controller-path "${CONTROLLER_PATH}"
  --controller-key "${CONTROLLER_KEY}"
  --resolution "${RESOLUTION}"
  --guidance-scale "${GUIDANCE_SCALE}"
  --noise-level "${NOISE_LEVEL}"
  --sde-type "${SDE_TYPE}"
  --score-dtype "${SCORE_DTYPE}"
  --dtype "${DTYPE}"
)

if [[ -n "${PICKSCORE_PROCESSOR}" ]]; then
  COMMON_ARGS+=(--pickscore-processor "${PICKSCORE_PROCESSOR}")
fi

{
  echo "[pickscore-visual-compare-beta1] REPO_ROOT=${REPO_ROOT}"
  echo "[pickscore-visual-compare-beta1] PYTHON_BIN=${PYTHON_BIN}"
  echo "[pickscore-visual-compare-beta1] CONTROLLER_PATH=${CONTROLLER_PATH}"
  echo "[pickscore-visual-compare-beta1] CONTROLLER_KEY=${CONTROLLER_KEY}"
  echo "[pickscore-visual-compare-beta1] OUTPUT_DIR=${OUTPUT_DIR}"
  echo "[pickscore-visual-compare-beta1] IMAGE_ROOT=${IMAGE_ROOT}"
  echo "[pickscore-visual-compare-beta1] SHARD_OUTPUT_DIR=${SHARD_OUTPUT_DIR}"
  echo "[pickscore-visual-compare-beta1] DATASET=${DATASET}"
  echo "[pickscore-visual-compare-beta1] SAMPLE_SIZE=${SAMPLE_SIZE}"
  echo "[pickscore-visual-compare-beta1] SAMPLE_SEED=${SAMPLE_SEED}"
  echo "[pickscore-visual-compare-beta1] BETA=${BETA}"
  echo "[pickscore-visual-compare-beta1] BUDGET=${BUDGET}"
  echo "[pickscore-visual-compare-beta1] BASELINE_TOTAL_NFE=${BASELINE_TOTAL_NFE}"
  echo "[pickscore-visual-compare-beta1] CONTROLLER_NUM_STEPS=${CONTROLLER_NUM_STEPS}"
  echo "[pickscore-visual-compare-beta1] DEVICES=${DEVICES}"
  echo "[pickscore-visual-compare-beta1] PICKSCORE_MODEL=${PICKSCORE_MODEL}"
  echo "[pickscore-visual-compare-beta1] STDOUT_LOG=${STDOUT_LOG}"
  echo "[pickscore-visual-compare-beta1] STDERR_LOG=${STDERR_LOG}"
} >"${ENV_LOG}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m flow_autotts.experiments.pickscore_sd35.visual_compare_beta1 \
  "${COMMON_ARGS[@]}" \
  >"${STDOUT_LOG}" 2>"${STDERR_LOG}"
