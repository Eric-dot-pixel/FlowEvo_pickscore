#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN_DEFAULT="${REPO_ROOT}/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[flow_autotts workflow] python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

BESTOF4_BASELINE_SOURCE="${BESTOF4_BASELINE_SOURCE:-/inspire/hdd/global_user/gongjingjing-25039/sywang/jkcai/FlowEvo_10nobase/logs/flow_autotts/pickscore_sd35/train_bestof4_ode_retry2_clean_b128_20260517_052333}"
BESTOF4_BASELINE_LOCAL_DIR="${BESTOF4_BASELINE_LOCAL_DIR:-${REPO_ROOT}/logs/flow_autotts/pickscore_sd35/train_bestof4_ode_retry2_clean_b128_compact_baseline}"

FLOW_TTS_SAMPLE_SIZE="${FLOW_TTS_SAMPLE_SIZE:-500}"
FLOW_TTS_SAMPLE_SEED="${FLOW_TTS_SAMPLE_SEED:-42}"
FLOW_TTS_SPLIT="${FLOW_TTS_SPLIT:-train}"
FLOW_TTS_BETAS="${FLOW_TTS_BETAS:-0 0.25 0.5 0.75 1.0}"
FLOW_TTS_BUDGET="${FLOW_TTS_BUDGET:-128}"
FLOW_TTS_NUM_STEPS="${FLOW_TTS_NUM_STEPS:-10}"
FLOW_TTS_RESOLUTION="${FLOW_TTS_RESOLUTION:-512}"
FLOW_TTS_GUIDANCE_SCALE="${FLOW_TTS_GUIDANCE_SCALE:-4.5}"
FLOW_TTS_NOISE_LEVEL="${FLOW_TTS_NOISE_LEVEL:-0.7}"
FLOW_TTS_SDE_TYPE="${FLOW_TTS_SDE_TYPE:-sde}"
FLOW_TTS_DATASET="${FLOW_TTS_DATASET:-${REPO_ROOT}/flow_grpo/dataset/pickscore}"
FLOW_TTS_MODEL="${FLOW_TTS_MODEL:-${REPO_ROOT}/SD_3.5_med}"
FLOW_TTS_PICKSCORE_MODEL="${FLOW_TTS_PICKSCORE_MODEL:-${REPO_ROOT}/PickScore_v1}"
FLOW_TTS_DTYPE="${FLOW_TTS_DTYPE:-bfloat16}"
FLOW_TTS_SCORE_DTYPE="${FLOW_TTS_SCORE_DTYPE:-float32}"
FLOW_TTS_EVAL_DEVICES="${FLOW_TTS_EVAL_DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3}"
FLOW_TTS_EVAL_TEXT_ENCODER_DEVICES="${FLOW_TTS_EVAL_TEXT_ENCODER_DEVICES:-${FLOW_TTS_EVAL_DEVICES}}"
FLOW_TTS_EVAL_SCORE_DEVICES="${FLOW_TTS_EVAL_SCORE_DEVICES:-${FLOW_TTS_EVAL_DEVICES}}"
FLOW_TTS_PROMPT_PROFILE="${FLOW_TTS_PROMPT_PROFILE:-autotts}"

RUN_NAME="${RUN_NAME:-train_codex_bestof4_ode_clean_b128}"
WORKFLOW_HISTORY_DIR_DEFAULT="logs/flow_autotts/pickscore_sd35/${RUN_NAME}/history"
WORKFLOW_CODEX_LOG_PARENT_DEFAULT="${REPO_ROOT}/logs/flow_autotts/pickscore_sd35/${RUN_NAME}/codex_logs"
WORKFLOW_RESULT_DIR_DEFAULT="${REPO_ROOT}/logs/flow_autotts/pickscore_sd35/${RUN_NAME}/training_results"
WORKFLOW_SHARD_OUTPUT_DIR_DEFAULT="${WORKFLOW_RESULT_DIR_DEFAULT}/shards"
WORKFLOW_PRECHECK_DIR_DEFAULT="${WORKFLOW_RESULT_DIR_DEFAULT}/precheck"
WORKFLOW_LOG_DIR_DEFAULT="${REPO_ROOT}/logs/flow_autotts/pickscore_sd35/${RUN_NAME}/workflow_logs"
WORKFLOW_LOG_DIR="${WORKFLOW_LOG_DIR:-${WORKFLOW_LOG_DIR_DEFAULT}}"

mkdir -p "${WORKFLOW_LOG_DIR}"
WORKFLOW_RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
WORKFLOW_BOOTSTRAP_LOG="${WORKFLOW_LOG_DIR}/workflow_${WORKFLOW_RUN_TS}.bootstrap.log"
WORKFLOW_STDOUT_LOG="${WORKFLOW_LOG_DIR}/workflow_${WORKFLOW_RUN_TS}.stdout.log"
WORKFLOW_STDERR_LOG="${WORKFLOW_LOG_DIR}/workflow_${WORKFLOW_RUN_TS}.stderr.log"
WORKFLOW_ENV_LOG="${WORKFLOW_LOG_DIR}/workflow_${WORKFLOW_RUN_TS}.env.log"
exec >> "${WORKFLOW_BOOTSTRAP_LOG}" 2>&1

PRECHECK_CMD_DEFAULT="${PYTHON_BIN} -m flow_autotts.experiments.pickscore_sd35.harness --dataset ${FLOW_TTS_DATASET} --split ${FLOW_TTS_SPLIT} --sample-size 8 --sample-seed ${FLOW_TTS_SAMPLE_SEED} --num-shards 1 --shard-index 0 --rounds 1 --controllers optimal --betas 0.5 --budget ${FLOW_TTS_BUDGET} --output ${WORKFLOW_PRECHECK_DIR_DEFAULT}/history.json --summary-output ${WORKFLOW_PRECHECK_DIR_DEFAULT}/summary.json --model ${FLOW_TTS_MODEL} --pickscore-model ${FLOW_TTS_PICKSCORE_MODEL} --num-steps ${FLOW_TTS_NUM_STEPS} --resolution ${FLOW_TTS_RESOLUTION} --guidance-scale ${FLOW_TTS_GUIDANCE_SCALE} --noise-level ${FLOW_TTS_NOISE_LEVEL} --sde-type ${FLOW_TTS_SDE_TYPE} --score-dtype ${FLOW_TTS_SCORE_DTYPE} --dtype ${FLOW_TTS_DTYPE} --device cuda:0 --text-encoder-device cuda:0 --score-device cuda:0 --controller-file ${REPO_ROOT}/flow_autotts/controllers/optimal.py --controller-class-name OptimalController"

WORKFLOW_EVAL_CMD_DEFAULT="${PYTHON_BIN} -m flow_autotts.experiments.pickscore_sd35.parallel_eval --devices '${FLOW_TTS_EVAL_DEVICES}' --text-encoder-devices '${FLOW_TTS_EVAL_TEXT_ENCODER_DEVICES}' --score-devices '${FLOW_TTS_EVAL_SCORE_DEVICES}' --shard-output-dir ${WORKFLOW_SHARD_OUTPUT_DIR_DEFAULT} --dataset ${FLOW_TTS_DATASET} --split ${FLOW_TTS_SPLIT} --sample-size ${FLOW_TTS_SAMPLE_SIZE} --sample-seed ${FLOW_TTS_SAMPLE_SEED} --rounds 1 --controllers optimal --betas ${FLOW_TTS_BETAS} --budget ${FLOW_TTS_BUDGET} --output ${WORKFLOW_RESULT_DIR_DEFAULT}/history.json --summary-output ${WORKFLOW_RESULT_DIR_DEFAULT}/summary.json --model ${FLOW_TTS_MODEL} --pickscore-model ${FLOW_TTS_PICKSCORE_MODEL} --num-steps ${FLOW_TTS_NUM_STEPS} --resolution ${FLOW_TTS_RESOLUTION} --guidance-scale ${FLOW_TTS_GUIDANCE_SCALE} --noise-level ${FLOW_TTS_NOISE_LEVEL} --sde-type ${FLOW_TTS_SDE_TYPE} --score-dtype ${FLOW_TTS_SCORE_DTYPE} --dtype ${FLOW_TTS_DTYPE} --controller-file ${REPO_ROOT}/flow_autotts/controllers/optimal.py --controller-class-name OptimalController"

export WORKFLOW_RESUME="${WORKFLOW_RESUME:-1}"
export WORKFLOW_WORKDIR="${WORKFLOW_WORKDIR:-${REPO_ROOT}}"
export WORKFLOW_METHOD_FILE="${WORKFLOW_METHOD_FILE:-flow_autotts/controllers/optimal.py}"
export WORKFLOW_TEMPLATE_FILE="${WORKFLOW_TEMPLATE_FILE:-flow_autotts/controllers/optimal.template.py}"
export WORKFLOW_HISTORY_DIR="${WORKFLOW_HISTORY_DIR:-${WORKFLOW_HISTORY_DIR_DEFAULT}}"
export WORKFLOW_CODEX_LOG_PARENT="${WORKFLOW_CODEX_LOG_PARENT:-${WORKFLOW_CODEX_LOG_PARENT_DEFAULT}}"
export WORKFLOW_RESULT_DIR="${WORKFLOW_RESULT_DIR:-${WORKFLOW_RESULT_DIR_DEFAULT}}"
export WORKFLOW_PROMPT_PATH="${WORKFLOW_PROMPT_PATH:-${SCRIPT_DIR}/prompts/proposer_prompt_autotts.txt}"
export WORKFLOW_ROUNDS="${WORKFLOW_ROUNDS:-5}"
export WORKFLOW_CONTEXT_HISTORY_ROUNDS="${WORKFLOW_CONTEXT_HISTORY_ROUNDS:-5}"
export WORKFLOW_EVAL_CWD="${WORKFLOW_EVAL_CWD:-${REPO_ROOT}}"
export WORKFLOW_EVAL_TIMEOUT_SEC="${WORKFLOW_EVAL_TIMEOUT_SEC:-21600}"
export WORKFLOW_EVAL_CMD="${WORKFLOW_EVAL_CMD:-${WORKFLOW_EVAL_CMD_DEFAULT}}"
export WORKFLOW_BASELINE_DIR="${WORKFLOW_BASELINE_DIR:-${BESTOF4_BASELINE_LOCAL_DIR}}"
export WORKFLOW_PRECHECK_CMD="${WORKFLOW_PRECHECK_CMD:-${PRECHECK_CMD_DEFAULT}}"
export WORKFLOW_PRECHECK_TIMEOUT_SEC="${WORKFLOW_PRECHECK_TIMEOUT_SEC:-3600}"
export WORKFLOW_REPAIR_MAX_ATTEMPTS="${WORKFLOW_REPAIR_MAX_ATTEMPTS:-2}"
export CODEX_BIN="${CODEX_BIN:-codex}"
export CODEX_EXEC_ARGS="${CODEX_EXEC_ARGS:---dangerously-bypass-approvals-and-sandbox}"

{
  echo "[flow_autotts workflow] REPO_ROOT=${REPO_ROOT}"
  echo "[flow_autotts workflow] PYTHON_BIN=${PYTHON_BIN}"
  echo "[flow_autotts workflow] RUN_NAME=${RUN_NAME}"
  echo "[flow_autotts workflow] WORKFLOW_HISTORY_DIR=${WORKFLOW_HISTORY_DIR}"
  echo "[flow_autotts workflow] WORKFLOW_CODEX_LOG_PARENT=${WORKFLOW_CODEX_LOG_PARENT}"
  echo "[flow_autotts workflow] WORKFLOW_RESULT_DIR=${WORKFLOW_RESULT_DIR}"
  echo "[flow_autotts workflow] WORKFLOW_BASELINE_DIR=${WORKFLOW_BASELINE_DIR}"
  echo "[flow_autotts workflow] WORKFLOW_PRECHECK_CMD=${WORKFLOW_PRECHECK_CMD}"
  echo "[flow_autotts workflow] WORKFLOW_REPAIR_MAX_ATTEMPTS=${WORKFLOW_REPAIR_MAX_ATTEMPTS}"
  echo "[flow_autotts workflow] REPAIR_POLICY=bugfix-only"
  echo "[flow_autotts workflow] WORKFLOW_BOOTSTRAP_LOG=${WORKFLOW_BOOTSTRAP_LOG}"
  echo "[flow_autotts workflow] WORKFLOW_EVAL_CMD=${WORKFLOW_EVAL_CMD}"
  echo "[flow_autotts workflow] WORKFLOW_STDOUT_LOG=${WORKFLOW_STDOUT_LOG}"
  echo "[flow_autotts workflow] WORKFLOW_STDERR_LOG=${WORKFLOW_STDERR_LOG}"
} > "${WORKFLOW_ENV_LOG}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m flow_autotts.workflow > "${WORKFLOW_STDOUT_LOG}" 2> "${WORKFLOW_STDERR_LOG}"
