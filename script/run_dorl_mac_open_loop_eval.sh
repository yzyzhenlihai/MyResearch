#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# 该脚本只复用同一 checkpoint 做 H 消融，不重新训练 actor/critic。
export SWANLAB_MODE="${SWANLAB_MODE:-offline}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-easyrl4rec}"
export PYTHONPATH="${PWD}:${PWD}/src:${PWD}/src/DeepCTR-Torch:${PWD}/src/tianshou:${PWD}/examples/policy:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python}"
CHUNK_SIZE="${CHUNK_SIZE:-7}"
SEED="${SEED:-2023}"
H_VALUES="${H_VALUES:-1 2 3 4 5 6 7}"
COMPLETION_WINDOW="${COMPLETION_WINDOW:-${CHUNK_SIZE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-saved_models/KuaiEnv-v0/MAC_origin/open_loop_K${CHUNK_SIZE}_seed${SEED}}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-DORL-MAC-OpenLoop}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
TEST_NUM="${TEST_NUM:-2}"

# ---- 评估用公共参数（与训练脚本保持一致，可用环境变量覆盖）----
ENV_NAME="${ENV_NAME:-KuaiEnv-v0}"
USER_MODEL_NAME="${USER_MODEL_NAME:-DeepFM}"
READ_MESSAGE="${READ_MESSAGE:-pointneg}"
DATASET_PATH="${DATASET_PATH:-data/KuaiRec/data_processed/DM_KuaiEnv-v0_small_data.pkl}"
WHICH_TRACKER="${WHICH_TRACKER:-none}"
REWARD_HANDLE="${REWARD_HANDLE:-cat}"
WINDOW_SIZE="${WINDOW_SIZE:-3}"
GAMMA="${GAMMA:-0.9}"
DEVICE="${DEVICE:-cuda:0}"
CUDA="${CUDA:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_TURN="${MAX_TURN:-200}"
FORCE_LENGTH="${FORCE_LENGTH:-${MAX_TURN}}"
NUM_LEAVE_COMPUTE="${NUM_LEAVE_COMPUTE:-1}"
LEAVE_THRESHOLD="${LEAVE_THRESHOLD:-0}"
INVALID_ACTION_PENALTY="${INVALID_ACTION_PENALTY:-0.0}"
PREDICTED_MAT_NORMALIZE="${PREDICTED_MAT_NORMALIZE:-per_user_max}"
LAMBDA_ENTROPY="${LAMBDA_ENTROPY:-0.5}"
LAMBDA_VARIANCE="${LAMBDA_VARIANCE:-1}"
ENTROPY_WINDOW="${ENTROPY_WINDOW:-1 2}"
DYNAMICS_LOSS_WEIGHT="${DYNAMICS_LOSS_WEIGHT:-1.0}"
# 若设置 MAC_CKPT，则自动作为 --mac_ckpt 传入（命令行显式 --mac_ckpt 仍优先）。
MAC_CKPT="${MAC_CKPT:-saved_models/KuaiEnv-v0/MAC_origin/dorl-mac-kuai-catbc-qv-K${CHUNK_SIZE}-seed${SEED}/mac_agent/latest.pt}"

read -r -a ENTROPY_WINDOW_ARGS <<< "${ENTROPY_WINDOW}"

COMMON_ARGS=(
  --env "${ENV_NAME}"
  --user_model_name "${USER_MODEL_NAME}"
  --read_message "${READ_MESSAGE}"
  --dataset_path "${DATASET_PATH}"
  --which_tracker "${WHICH_TRACKER}"
  --reward_handle "${REWARD_HANDLE}"
  --window_size "${WINDOW_SIZE}"
  --chunk_size "${CHUNK_SIZE}"
  --completion_window "${COMPLETION_WINDOW}"
  --gamma "${GAMMA}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --cuda "${CUDA}"
  --batch_size "${BATCH_SIZE}"
  --num_leave_compute "${NUM_LEAVE_COMPUTE}"
  --leave_threshold "${LEAVE_THRESHOLD}"
  --max_turn "${MAX_TURN}"
  --force_length "${FORCE_LENGTH}"
  --invalid_action_penalty "${INVALID_ACTION_PENALTY}"
  --predicted_mat_normalize "${PREDICTED_MAT_NORMALIZE}"
  --lambda_entropy "${LAMBDA_ENTROPY}"
  --lambda_variance "${LAMBDA_VARIANCE}"
  --entropy_window "${ENTROPY_WINDOW_ARGS[@]}"
  --dynamics_loss_weight "${DYNAMICS_LOSS_WEIGHT}"
)

if [[ -z "${MAC_CKPT}" && $# == 0 ]]; then
  printf 'Usage: MAC_CKPT=PATH CHUNK_SIZE=5 H_VALUES="1 2 3 5" %s [eval args...]\n' "$0" >&2
  exit 2
fi

read -r -a EXECUTION_HORIZONS <<< "${H_VALUES}"
for execution_horizon in "${EXECUTION_HORIZONS[@]}"; do
  if (( execution_horizon < 1 || execution_horizon > CHUNK_SIZE )); then
    printf '[run_dorl_mac_open_loop_eval] ERROR: H=%s must satisfy 1 <= H <= K=%s\n' \
      "${execution_horizon}" "${CHUNK_SIZE}" >&2
    exit 2
  fi
done

printf '\n[run_dorl_mac_open_loop_eval] single-process sweep: K=%s, H=(%s), output=%s\n' \
  "${CHUNK_SIZE}" "${H_VALUES}" "${OUTPUT_ROOT}"
printf '[run_dorl_mac_open_loop_eval] trajectories per branch=%s, parallel test envs=%s\n' \
  "${EVAL_EPISODES}" "${TEST_NUM}"
EXTRA_ARGS=()
if [[ -n "${MAC_CKPT}" ]]; then
  EXTRA_ARGS+=(--mac_ckpt "${MAC_CKPT}")
fi
"${PYTHON_BIN}" examples/our_model/runners/eval_dorl_mac.py \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$@" \
  --eval_episodes "${EVAL_EPISODES}" \
  --test_num "${TEST_NUM}" \
  --execution_horizons "${EXECUTION_HORIZONS[@]}" \
  --enable_open_loop_diagnostics \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --run_name "open-loop-K${CHUNK_SIZE}-seed${SEED}" \
  --eval_save_dir "${OUTPUT_ROOT}"

printf '\n[run_dorl_mac_open_loop_eval] done. outputs are under %s\n' "${OUTPUT_ROOT}"
