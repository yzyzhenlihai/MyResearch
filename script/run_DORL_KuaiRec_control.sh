#!/usr/bin/env bash
# 与 DARLR KuaiRec 完全同协议的 DORL 对照入口。
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python}"
SEED="${SEED:-2023}"
CUDA="${CUDA:-1}"
EPOCH="${EPOCH:-100}"
STEP_PER_EPOCH="${STEP_PER_EPOCH:-100000}"
EPISODE_PER_COLLECT="${EPISODE_PER_COLLECT:-100}"
TRAINING_NUM="${TRAINING_NUM:-100}"
TEST_NUM="${TEST_NUM:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BUFFER_SIZE="${BUFFER_SIZE:-100000}"
MAX_TURN="${MAX_TURN:-30}"
FORCE_LENGTH="${FORCE_LENGTH:-10}"
NUM_LEAVE_COMPUTE="${NUM_LEAVE_COMPUTE:-3}"
LEAVE_THRESHOLD="${LEAVE_THRESHOLD:-0}"

WHICH_TRACKER="${WHICH_TRACKER:-sasrec}"
REWARD_HANDLE="${REWARD_HANDLE:-cat}"
WINDOW_SIZE="${WINDOW_SIZE:-3}"
NUM_HEADS="${NUM_HEADS:-1}"
LR="${LR:-0.001}"
ENT_COEF="${ENT_COEF:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM-5.0}"
REW_NORM="${REW_NORM:-0}"

LAMBDA_VARIANCE="${LAMBDA_VARIANCE:-0.05}"
LAMBDA_ENTROPY="${LAMBDA_ENTROPY:-0.05}"
BEST_METRIC="${BEST_METRIC:-NX_0}"
SAVE_MODEL="${SAVE_MODEL:-1}"
SAVE_BEST_ONLY="${SAVE_BEST_ONLY:-1}"
DRY_RUN="${DRY_RUN:-0}"
READ_MESSAGE="${READ_MESSAGE:-pointneg}"
MESSAGE="${MESSAGE:-DORL_KuaiRec_control_seed${SEED}}"

COMMAND=(
  "${PYTHON_BIN}"
  examples/advance/run_DORL.py
  --model_name DORL
  --message "${MESSAGE}"
  --env KuaiEnv-v0
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
  --is_feature_level
  --no_exposure_intervention
  --no_exploration_noise
  --version v1
  --tau 0.0
  --gamma_exposure 10.0
  --lambda_variance "${LAMBDA_VARIANCE}"
  --lambda_entropy "${LAMBDA_ENTROPY}"
  --entropy_window 1 2
  --vf-coef 0.5
  --ent-coef "${ENT_COEF}"
  --gae-lambda 1.0
  --gamma 0.9
  --lr "${LR}"
  --best-metric "${BEST_METRIC}"
  --read_message "${READ_MESSAGE}"
)

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

if [[ -n "${MAX_GRAD_NORM}" ]]; then
  COMMAND+=(--max-grad-norm "${MAX_GRAD_NORM}")
fi

case "${SAVE_MODEL,,}" in
  1|true|yes|on)
    COMMAND+=(--is_save)
    case "${SAVE_BEST_ONLY,,}" in
      1|true|yes|on)
        COMMAND+=(--save-best-only)
        ;;
      0|false|no|off)
        COMMAND+=(--save-every-epoch)
        ;;
      *)
        echo "Expected boolean SAVE_BEST_ONLY, got '${SAVE_BEST_ONLY}'." >&2
        exit 2
        ;;
    esac
    ;;
  0|false|no|off)
    COMMAND+=(--no_save)
    ;;
  *)
    echo "Expected boolean SAVE_MODEL, got '${SAVE_MODEL}'." >&2
    exit 2
    ;;
esac

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
