#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

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
MAX_TURN="${MAX_TURN:-100}"
FORCE_LENGTH="${FORCE_LENGTH:-10}"

WHICH_TRACKER="${WHICH_TRACKER:-sasrec}"
REWARD_HANDLE="${REWARD_HANDLE:-cat}"
WINDOW_SIZE="${WINDOW_SIZE:-3}"
NUM_HEADS="${NUM_HEADS:-1}"

SELECTOR_CANDIDATE_MODE="${SELECTOR_CANDIDATE_MODE:-embedding_topk}"
SELECTOR_POLICY_MODE="${SELECTOR_POLICY_MODE:-learned}"
SELECTOR_PREF_DIM="${SELECTOR_PREF_DIM:-64}"
SELECTOR_NUM_HEADS="${SELECTOR_NUM_HEADS:-1}"
SELECTOR_NUM_LAYERS="${SELECTOR_NUM_LAYERS:-1}"
SELECTOR_DROPOUT_RATE="${SELECTOR_DROPOUT_RATE:-0.1}"
SELECTOR_LAMBDA_S="${SELECTOR_LAMBDA_S:-1.0}"
SELECTOR_LAMBDA_D="${SELECTOR_LAMBDA_D:-0.05}"
SELECTOR_GAIN_MODE="${SELECTOR_GAIN_MODE:-paper_core}"
SELECTOR_LOSS_COEF="${SELECTOR_LOSS_COEF:-1.0}"
SELECTOR_ENT_COEF="${SELECTOR_ENT_COEF:-0.0}"
SELECTOR_ADVANTAGE_NORMALIZATION="${SELECTOR_ADVANTAGE_NORMALIZATION:-0}"
SELECTOR_REWARD_NORMALIZATION="${SELECTOR_REWARD_NORMALIZATION:-0}"
SELECTOR_NORMALIZATION_EPS="${SELECTOR_NORMALIZATION_EPS:-1.0e-8}"
SELECTOR_LR="${SELECTOR_LR-}"

SELECTOR_REWARD_MODE="${SELECTOR_REWARD_MODE:-full}"
DYNAMIC_REWARD_MODE="${DYNAMIC_REWARD_MODE:-reference_mean}"
DYNAMIC_UNCERTAINTY_MODE="${DYNAMIC_UNCERTAINTY_MODE:-dynamic}"
LAMBDA_UNCERTAINTY="${LAMBDA_UNCERTAINTY:-0.05}"
LAMBDA_ENTROPY="${LAMBDA_ENTROPY:-0.05}"
LAMBDA_VARIANCE="${LAMBDA_VARIANCE:-0.0}"
DARLR_EPS="${DARLR_EPS:-1.0e-8}"

LR="${LR-}"
ENT_COEF="${ENT_COEF-}"
MAX_GRAD_NORM="${MAX_GRAD_NORM-}"
REW_NORM="${REW_NORM:-0}"
BEST_METRIC="${BEST_METRIC:-NX_0}"
SAVE_MODEL="${SAVE_MODEL:-1}"
SAVE_BEST_ONLY="${SAVE_BEST_ONLY:-1}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_LR="0.001"
DEFAULT_ENT_COEF="0.0"
DEFAULT_READ_MESSAGE="pointneg"
DATASET_ARGS=()

case "${DATASET}" in
  KuaiEnv-v0)
    SELECTOR_K="${SELECTOR_K:-10}"
    SELECTOR_CANDIDATE_SIZE="${SELECTOR_CANDIDATE_SIZE:-512}"
    # KuaiRec 论文 N=4, M=0；当前环境检查前 3 个动作与当前动作。
    NUM_LEAVE_COMPUTE="${NUM_LEAVE_COMPUTE:-1}"
    LEAVE_THRESHOLD="${LEAVE_THRESHOLD:-0}"
    DATASET_ARGS+=(--is_feature_level)
    ;;
  KuaiRand-v0)
    SELECTOR_K="${SELECTOR_K:-50}"
    SELECTOR_CANDIDATE_SIZE="${SELECTOR_CANDIDATE_SIZE:-512}"
    NUM_LEAVE_COMPUTE="${NUM_LEAVE_COMPUTE:-1}"
    LEAVE_THRESHOLD="${LEAVE_THRESHOLD:-0}"
    DATASET_ARGS+=(--is_feature_level)
    ;;
  CoatEnv-v0)
    SELECTOR_K="${SELECTOR_K:-10}"
    SELECTOR_CANDIDATE_SIZE="${SELECTOR_CANDIDATE_SIZE:-512}"
    NUM_LEAVE_COMPUTE="${NUM_LEAVE_COMPUTE:-7}"
    LEAVE_THRESHOLD="${LEAVE_THRESHOLD:-6}"
    DEFAULT_LR="0.005"
    DEFAULT_ENT_COEF="0.01"
    DATASET_ARGS+=(--no_feature_level --no_exploration_noise)
    ;;
  YahooEnv-v0)
    SELECTOR_K="${SELECTOR_K:-25}"
    SELECTOR_CANDIDATE_SIZE="${SELECTOR_CANDIDATE_SIZE:-512}"
    NUM_LEAVE_COMPUTE="${NUM_LEAVE_COMPUTE:-3}"
    LEAVE_THRESHOLD="${LEAVE_THRESHOLD:-120}"
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
MESSAGE="${MESSAGE:-DARLR_${DATASET}_seed${SEED}}"

append_boolean_pair() {
  local value="$1"
  local enabled_flag="$2"
  local disabled_flag="$3"
  case "${value,,}" in
    1|true|yes|on)
      COMMAND+=("${enabled_flag}")
      ;;
    0|false|no|off)
      COMMAND+=("${disabled_flag}")
      ;;
    *)
      echo "Expected boolean value for ${enabled_flag}: got '${value}'." >&2
      exit 2
      ;;
  esac
}

COMMAND=(
  "${PYTHON_BIN}"
  examples/advance/run_DARLR.py
  --model_name DARLR
  --message "${MESSAGE}"
  --env "${DATASET}"
  --seed "${SEED}"
  --cuda "${CUDA}"
  --epoch "${EPOCH}"
  --step-per-epoch "${STEP_PER_EPOCH}"
  --episode-per-collect "${EPISODE_PER_COLLECT}"
  --training-num "${TRAINING_NUM}"
  --test-num "${TEST_NUM}"
  --batch-size "${BATCH_SIZE}"
  --buffer-size "${BUFFER_SIZE}"
  --max_turn "${MAX_TURN}"
  --force_length "${FORCE_LENGTH}"
  --num_leave_compute "${NUM_LEAVE_COMPUTE}"
  --leave_threshold "${LEAVE_THRESHOLD}"
  --which_tracker "${WHICH_TRACKER}"
  --reward_handle "${REWARD_HANDLE}"
  --window_size "${WINDOW_SIZE}"
  --num_heads "${NUM_HEADS}"
  --is_sorted
  --no_exposure_intervention
  --version v1
  --tau 0.0
  --gamma_exposure 10.0
  --selector_k "${SELECTOR_K}"
  --selector_candidate_size "${SELECTOR_CANDIDATE_SIZE}"
  --selector_candidate_mode "${SELECTOR_CANDIDATE_MODE}"
  --selector_policy_mode "${SELECTOR_POLICY_MODE}"
  --selector_pref_dim "${SELECTOR_PREF_DIM}"
  --selector_num_heads "${SELECTOR_NUM_HEADS}"
  --selector_num_layers "${SELECTOR_NUM_LAYERS}"
  --selector_dropout_rate "${SELECTOR_DROPOUT_RATE}"
  --selector_lambda_s "${SELECTOR_LAMBDA_S}"
  --selector_lambda_d "${SELECTOR_LAMBDA_D}"
  --selector_gain_mode "${SELECTOR_GAIN_MODE}"
  --selector_loss_coef "${SELECTOR_LOSS_COEF}"
  --selector_ent_coef "${SELECTOR_ENT_COEF}"
  --selector_normalization_eps "${SELECTOR_NORMALIZATION_EPS}"
  --selector_reward_mode "${SELECTOR_REWARD_MODE}"
  --dynamic_reward_mode "${DYNAMIC_REWARD_MODE}"
  --dynamic_uncertainty_mode "${DYNAMIC_UNCERTAINTY_MODE}"
  --lambda_uncertainty "${LAMBDA_UNCERTAINTY}"
  --lambda_variance "${LAMBDA_VARIANCE}"
  --darlr_eps "${DARLR_EPS}"
  --lambda_entropy "${LAMBDA_ENTROPY}"
  --entropy_window 1 2
  --vf-coef 0.5
  --ent-coef "${ENT_COEF}"
  --gae-lambda 1.0
  --gamma 0.9
  --lr "${LR}"
  --best-metric "${BEST_METRIC}"
  --read_message "${READ_MESSAGE}"
  "${DATASET_ARGS[@]}"
)

append_boolean_pair \
  "${SELECTOR_ADVANTAGE_NORMALIZATION}" \
  --selector_advantage_normalization \
  --no_selector_advantage_normalization
append_boolean_pair \
  "${SELECTOR_REWARD_NORMALIZATION}" \
  --selector_reward_normalization \
  --no_selector_reward_normalization
case "${SAVE_MODEL,,}" in
  1|true|yes|on)
    COMMAND+=(--is_save)
    append_boolean_pair "${SAVE_BEST_ONLY}" --save-best-only --save-every-epoch
    ;;
  0|false|no|off)
    COMMAND+=(--no_save)
    ;;
  *)
    echo "Expected boolean SAVE_MODEL, got '${SAVE_MODEL}'." >&2
    exit 2
    ;;
esac

case "${REW_NORM,,}" in
  1|true|yes|on)
    COMMAND+=(--rew-norm)
    ;;
  0|false|no|off)
    ;;
  *)
    echo "Expected boolean REW_NORM, got '${REW_NORM}'." >&2
    exit 2
    ;;
esac

if [[ -n "${SELECTOR_LR}" ]]; then
  COMMAND+=(--selector_lr "${SELECTOR_LR}")
fi
if [[ -n "${MAX_GRAD_NORM}" ]]; then
  COMMAND+=(--max-grad-norm "${MAX_GRAD_NORM}")
fi

# Caller-supplied CLI flags are deliberately last so they can override defaults.
COMMAND+=("$@")

case "${DRY_RUN,,}" in
  1|true|yes|on)
    printf 'DRY_RUN:'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
    ;;
  0|false|no|off)
    ;;
  *)
    echo "Expected boolean DRY_RUN, got '${DRY_RUN}'." >&2
    exit 2
    ;;
esac

printf 'RUN:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
