#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "${SWANLAB_MODE:-}" ]]; then
  export SWANLAB_MODE
fi
PYTHON_BIN="${PYTHON_BIN:-/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python}"

DATASET="${DATASET:-KuaiEnv-v0}"
SEED="${SEED:-2023}"
CUDA="${CUDA:-2}"
EPOCH="${EPOCH:-100}"
STEP_PER_EPOCH="${STEP_PER_EPOCH:-100000}"
EPISODE_PER_COLLECT="${EPISODE_PER_COLLECT:-100}"
TRAINING_NUM="${TRAINING_NUM:-100}"
TEST_NUM="${TEST_NUM:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BUFFER_SIZE="${BUFFER_SIZE:-100000}"
MAX_TURN="${MAX_TURN:-30}"
FORCE_LENGTH="${FORCE_LENGTH:-10}"

WHICH_TRACKER="${WHICH_TRACKER:-sasrec}"
REWARD_HANDLE="${REWARD_HANDLE:-cat}"
WINDOW_SIZE="${WINDOW_SIZE:-3}"
NUM_HEADS="${NUM_HEADS:-1}"

SELECTOR_CANDIDATE_MODE="${SELECTOR_CANDIDATE_MODE:-embedding_topk}"
SELECTOR_PREF_DIM="${SELECTOR_PREF_DIM:-64}"
SELECTOR_NUM_HEADS="${SELECTOR_NUM_HEADS:-1}"
SELECTOR_NUM_LAYERS="${SELECTOR_NUM_LAYERS:-1}"
SELECTOR_DROPOUT_RATE="${SELECTOR_DROPOUT_RATE:-0.1}"
SELECTOR_LAMBDA_S="${SELECTOR_LAMBDA_S:-1.0}"
SELECTOR_LAMBDA_D="${SELECTOR_LAMBDA_D:-0.05}"
LAMBDA_UNCERTAINTY="${LAMBDA_UNCERTAINTY:-0.05}"
LAMBDA_ENTROPY="${LAMBDA_ENTROPY:-0.05}"
DARLR_EPS="${DARLR_EPS:-1.0e-8}"
LR="${LR-}"
ENT_COEF="${ENT_COEF-}"
DEFAULT_LR="0.001"
DEFAULT_ENT_COEF="0.0"

DATASET_ARGS=()
DEFAULT_READ_MESSAGE="pointneg"
case "${DATASET}" in
  KuaiEnv-v0)
    SELECTOR_K="${SELECTOR_K:-10}"
    SELECTOR_CANDIDATE_SIZE="${SELECTOR_CANDIDATE_SIZE:-512}"
    DATASET_ARGS+=(--is_feature_level)
    ;;
  KuaiRand-v0)
    SELECTOR_K="${SELECTOR_K:-50}"
    SELECTOR_CANDIDATE_SIZE="${SELECTOR_CANDIDATE_SIZE:-512}"
    DATASET_ARGS+=(--is_feature_level)
    ;;
  CoatEnv-v0)
    SELECTOR_K="${SELECTOR_K:-10}"
    SELECTOR_CANDIDATE_SIZE="${SELECTOR_CANDIDATE_SIZE:-512}"
    DEFAULT_LR="0.005"
    DEFAULT_ENT_COEF="0.01"
    DATASET_ARGS+=(--no_feature_level --no_exploration_noise)
    ;;
  YahooEnv-v0)
    SELECTOR_K="${SELECTOR_K:-25}"
    SELECTOR_CANDIDATE_SIZE="${SELECTOR_CANDIDATE_SIZE:-512}"
    DEFAULT_LR="0.005"
    DEFAULT_ENT_COEF="0.01"
    DATASET_ARGS+=(--no_feature_level --no_exploration_noise)
    ;;
  *)
    echo "Unsupported DATASET='${DATASET}'. Expected KuaiEnv-v0, KuaiRand-v0, CoatEnv-v0, or YahooEnv-v0." >&2
    exit 2
    ;;
esac

LR="${LR:-${DEFAULT_LR}}"
ENT_COEF="${ENT_COEF:-${DEFAULT_ENT_COEF}}"
READ_MESSAGE="${READ_MESSAGE:-${DEFAULT_READ_MESSAGE}}"
MESSAGE="${MESSAGE:-DARLR_${DATASET}_all_modules}"

"${PYTHON_BIN}" examples/advance/run_DARLR.py \
  --model_name "DARLR" \
  --message "${MESSAGE}" \
  --env "${DATASET}" \
  --seed "${SEED}" \
  --cuda "${CUDA}" \
  --epoch "${EPOCH}" \
  --step-per-epoch "${STEP_PER_EPOCH}" \
  --episode-per-collect "${EPISODE_PER_COLLECT}" \
  --training-num "${TRAINING_NUM}" \
  --test-num "${TEST_NUM}" \
  --batch-size "${BATCH_SIZE}" \
  --buffer-size "${BUFFER_SIZE}" \
  --max_turn "${MAX_TURN}" \
  --force_length "${FORCE_LENGTH}" \
  --which_tracker "${WHICH_TRACKER}" \
  --reward_handle "${REWARD_HANDLE}" \
  --window_size "${WINDOW_SIZE}" \
  --num_heads "${NUM_HEADS}" \
  --is_sorted \
  --no_exposure_intervention \
  --version "v1" \
  --tau 0.0 \
  --gamma_exposure 10.0 \
  --selector_k "${SELECTOR_K}" \
  --selector_candidate_size "${SELECTOR_CANDIDATE_SIZE}" \
  --selector_candidate_mode "${SELECTOR_CANDIDATE_MODE}" \
  --selector_pref_dim "${SELECTOR_PREF_DIM}" \
  --selector_num_heads "${SELECTOR_NUM_HEADS}" \
  --selector_num_layers "${SELECTOR_NUM_LAYERS}" \
  --selector_dropout_rate "${SELECTOR_DROPOUT_RATE}" \
  --selector_lambda_s "${SELECTOR_LAMBDA_S}" \
  --selector_lambda_d "${SELECTOR_LAMBDA_D}" \
  --selector_reward_mode full \
  --dynamic_reward_mode reference_mean \
  --dynamic_uncertainty_mode dynamic \
  --lambda_uncertainty "${LAMBDA_UNCERTAINTY}" \
  --darlr_eps "${DARLR_EPS}" \
  --lambda_entropy "${LAMBDA_ENTROPY}" \
  --entropy_window 1 2 \
  --vf-coef 0.5 \
  --ent-coef "${ENT_COEF}" \
  --gae-lambda 1.0 \
  --gamma 0.9 \
  --lr "${LR}" \
  --read_message "${READ_MESSAGE}" \
  "${DATASET_ARGS[@]}"
