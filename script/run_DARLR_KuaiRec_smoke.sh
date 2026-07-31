#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export SEED="${SEED:-2023}"
export CUDA="${CUDA:-0}"
export EPOCH="${EPOCH:-1}"
export STEP_PER_EPOCH="${STEP_PER_EPOCH:-1}"
export EPISODE_PER_COLLECT="${EPISODE_PER_COLLECT:-1}"
export TRAINING_NUM="${TRAINING_NUM:-1}"
export TEST_NUM="${TEST_NUM:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export MAX_TURN="${MAX_TURN:-2}"
export FORCE_LENGTH="${FORCE_LENGTH:-1}"
export SELECTOR_K="${SELECTOR_K:-1}"
export SELECTOR_CANDIDATE_SIZE="${SELECTOR_CANDIDATE_SIZE:-2}"
export SELECTOR_CANDIDATE_MODE="${SELECTOR_CANDIDATE_MODE:-random}"
export LAMBDA_ENTROPY="${LAMBDA_ENTROPY:-0.0}"
export SAVE_MODEL="${SAVE_MODEL:-0}"
export MESSAGE="${MESSAGE:-DARLR_smoke}"

exec bash script/run_DARLR_KuaiRec.sh --cpu --entropy_window "$@"
