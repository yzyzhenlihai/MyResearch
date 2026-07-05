#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# 默认使用 SwanLab cloud 在线记录；如需本地记录可启动时设置 SWANLAB_MODE=offline。
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-easyrl4rec}"
export PYTHONPATH="${PWD}:${PWD}/src:${PWD}/src/DeepCTR-Torch:${PWD}/src/tianshou:${PWD}/examples/policy:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
RESET_METRICS="${RESET_METRICS:-1}"

ENV_NAME="${ENV_NAME:-KuaiEnv-v0}"
USER_MODEL_NAME="${USER_MODEL_NAME:-DeepFM}"
READ_MESSAGE="${READ_MESSAGE:-pointneg}"
DATASET_PATH="${DATASET_PATH:-data/KuaiRec/data_processed/DM_KuaiEnv-v0_small_data.pkl}"
WHICH_TRACKER="${WHICH_TRACKER:-avg}"
REWARD_HANDLE="${REWARD_HANDLE:-cat}"
WINDOW_SIZE="${WINDOW_SIZE:-3}"
CHUNK_SIZE="${CHUNK_SIZE:-3}"
GAMMA="${GAMMA:-0.9}"
SEED="${SEED:-2023}"
DEVICE="${DEVICE:-cuda:2}"
CUDA="${CUDA:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_TURN="${MAX_TURN:-30}"
FORCE_LENGTH="${FORCE_LENGTH:-${MAX_TURN}}"
NUM_LEAVE_COMPUTE="${NUM_LEAVE_COMPUTE:-9}"
LEAVE_THRESHOLD="${LEAVE_THRESHOLD:-1.0}"
INVALID_ACTION_PENALTY="${INVALID_ACTION_PENALTY:--1.0}"

# 默认为空表示使用全量离线轨迹和全量 action chunks。
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-}"
MAX_CHUNKS="${MAX_CHUNKS:-}"
ITEM_EMBEDDING_PATH="${ITEM_EMBEDDING_PATH:-}"
PREDICTED_MAT_PATH="${PREDICTED_MAT_PATH:-}"
MAXVAR_MAT_PATH="${MAXVAR_MAT_PATH:-}"

USE_ENTROPY_REWARD="${USE_ENTROPY_REWARD:-1}"
USE_UNCERTAINTY_PENALTY="${USE_UNCERTAINTY_PENALTY:-1}"
LAMBDA_ENTROPY="${LAMBDA_ENTROPY:-5.0}"
LAMBDA_VARIANCE="${LAMBDA_VARIANCE:-0.05}"
ENTROPY_WINDOW="${ENTROPY_WINDOW:-1 2}"
FEATURE_LEVEL="${FEATURE_LEVEL:-1}"
IS_SORTED="${IS_SORTED:-1}"
DYNAMICS_LOSS_WEIGHT="${DYNAMICS_LOSS_WEIGHT:-1.0}"

ACTOR_BACKEND="${ACTOR_BACKEND:-flow}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-100000}"
FLOW_STEPS="${FLOW_STEPS:-16}"
ACTOR_LR="${ACTOR_LR:-0.0003}"
BC_WEIGHT="${BC_WEIGHT:-1.0}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-100}"

EPOCH="${EPOCH:-100}"
STEP_PER_EPOCH="${STEP_PER_EPOCH:-1000}"
QV_LR="${QV_LR:-0.0003}"
TARGET_TAU="${TARGET_TAU:-0.005}"
NUM_SAMPLES_TRAIN="${NUM_SAMPLES_TRAIN:-8}"
NUM_SAMPLES_TEST="${NUM_SAMPLES_TEST:-32}"
REPEAT_POLICY="${REPEAT_POLICY:-truncate}"
TEST_NUM="${TEST_NUM:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-0}"
BUFFER_SIZE="${BUFFER_SIZE:-0}"
QV_LOG_INTERVAL="${QV_LOG_INTERVAL:-100}"

SWANLAB_PROJECT="${SWANLAB_PROJECT:-DORL-MAC}"
FLOW_SWANLAB_PROJECT="${FLOW_SWANLAB_PROJECT:-${SWANLAB_PROJECT}-FlowBC}"
QV_SWANLAB_PROJECT="${QV_SWANLAB_PROJECT:-${SWANLAB_PROJECT}-QV}"
EVAL_SWANLAB_PROJECT="${EVAL_SWANLAB_PROJECT:-${SWANLAB_PROJECT}-Eval}"
RUN_NAME="${RUN_NAME:-dorl-mac-kuai-flow-qv}"
FLOW_RUN_NAME="${FLOW_RUN_NAME:-${RUN_NAME}-flow-bc}"
QV_RUN_NAME="${QV_RUN_NAME:-${RUN_NAME}-qv}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-${RUN_NAME}-eval}"
RUN_DIR="${RUN_DIR:-saved_models/${ENV_NAME}/DORL_MAC/${RUN_NAME}}"
FLOW_SAVE_DIR="${FLOW_SAVE_DIR:-${RUN_DIR}/flow_bc}"
MAC_SAVE_DIR="${MAC_SAVE_DIR:-${RUN_DIR}/mac_agent}"
EVAL_SAVE_DIR="${EVAL_SAVE_DIR:-${RUN_DIR}/eval}"
FLOW_ACTOR_CKPT="${FLOW_ACTOR_CKPT:-}"

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
  --lambda_entropy "${LAMBDA_ENTROPY}"
  --lambda_variance "${LAMBDA_VARIANCE}"
  --entropy_window "${ENTROPY_WINDOW_ARGS[@]}"
  --dynamics_loss_weight "${DYNAMICS_LOSS_WEIGHT}"
)

if [[ -n "${MAX_TRAJECTORIES}" ]]; then
  COMMON_ARGS+=(--max_trajectories "${MAX_TRAJECTORIES}")
fi
if [[ -n "${MAX_CHUNKS}" ]]; then
  COMMON_ARGS+=(--max_chunks "${MAX_CHUNKS}")
fi
if [[ -n "${ITEM_EMBEDDING_PATH}" ]]; then
  COMMON_ARGS+=(--item_embedding_path "${ITEM_EMBEDDING_PATH}")
fi
if [[ -n "${PREDICTED_MAT_PATH}" ]]; then
  COMMON_ARGS+=(--predicted_mat_path "${PREDICTED_MAT_PATH}")
fi
if [[ -n "${MAXVAR_MAT_PATH}" ]]; then
  COMMON_ARGS+=(--maxvar_mat_path "${MAXVAR_MAT_PATH}")
fi
if [[ "${USE_ENTROPY_REWARD}" == "1" || "${USE_ENTROPY_REWARD}" == "true" || "${USE_ENTROPY_REWARD}" == "True" ]]; then
  COMMON_ARGS+=(--use_entropy_reward)
else
  COMMON_ARGS+=(--no_entropy_reward)
fi
if [[ "${USE_UNCERTAINTY_PENALTY}" == "1" || "${USE_UNCERTAINTY_PENALTY}" == "true" || "${USE_UNCERTAINTY_PENALTY}" == "True" ]]; then
  COMMON_ARGS+=(--use_uncertainty_penalty)
else
  COMMON_ARGS+=(--no_uncertainty_penalty)
fi
if [[ "${FEATURE_LEVEL}" == "1" || "${FEATURE_LEVEL}" == "true" || "${FEATURE_LEVEL}" == "True" ]]; then
  COMMON_ARGS+=(--feature_level)
else
  COMMON_ARGS+=(--no_feature_level)
fi
if [[ "${IS_SORTED}" == "1" || "${IS_SORTED}" == "true" || "${IS_SORTED}" == "True" ]]; then
  COMMON_ARGS+=(--is_sorted)
else
  COMMON_ARGS+=(--no_sorted)
fi

run_command() {
  local cmd=("$@")
  printf '\n[run_dorl_mac_kuai_train] %s\n' "${cmd[*]}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}"
  fi
}

is_truthy() {
  local raw_value="$1"
  [[ "${raw_value}" == "1" || "${raw_value}" == "true" || "${raw_value}" == "True" || "${raw_value}" == "yes" || "${raw_value}" == "YES" ]]
}

reset_metrics_log() {
  local metrics_path="$1"
  local stage_name="$2"

  if ! is_truthy "${RESET_METRICS}"; then
    printf '[run_dorl_mac_kuai_train] keep existing %s metrics: %s\n' \
      "${stage_name}" "${metrics_path}"
    return 0
  fi

  printf '[run_dorl_mac_kuai_train] reset %s metrics: %s\n' \
    "${stage_name}" "${metrics_path}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "$(dirname "${metrics_path}")"
    : > "${metrics_path}"
  fi
}

find_existing_flow_ckpt() {
  local checkpoint_dir="$1"
  local checkpoints=()

  if [[ ! -d "${checkpoint_dir}" ]]; then
    return 1
  fi

  # 优先复用 runner 默认保存的 latest.pt，避免目录中有多个历史 checkpoint 时选错。
  if [[ -f "${checkpoint_dir}/latest.pt" ]]; then
    printf '%s\n' "${checkpoint_dir}/latest.pt"
    return 0
  fi

  shopt -s nullglob
  checkpoints=("${checkpoint_dir}"/*.pt)
  shopt -u nullglob
  if (( ${#checkpoints[@]} == 0 )); then
    return 1
  fi
  printf '%s\n' "${checkpoints[0]}"
  return 0
}

printf '[run_dorl_mac_kuai_train] run_dir=%s\n' "${RUN_DIR}"
printf '[run_dorl_mac_kuai_train] actor_backend=%s, pretrain_steps=%s, flow_steps=%s\n' \
  "${ACTOR_BACKEND}" "${PRETRAIN_STEPS}" "${FLOW_STEPS}"
printf '[run_dorl_mac_kuai_train] epoch=%s, step_per_epoch=%s, test_num=%s\n' \
  "${EPOCH}" "${STEP_PER_EPOCH}" "${TEST_NUM}"
printf '[run_dorl_mac_kuai_train] reward: entropy=%s(lambda=%s, window=%s), uncertainty=%s(lambda=%s), invalid_penalty=%s\n' \
  "${USE_ENTROPY_REWARD}" "${LAMBDA_ENTROPY}" "${ENTROPY_WINDOW}" \
  "${USE_UNCERTAINTY_PENALTY}" "${LAMBDA_VARIANCE}" "${INVALID_ACTION_PENALTY}"
printf '[run_dorl_mac_kuai_train] swanlab projects: flow=%s, qv=%s, eval=%s\n' \
  "${FLOW_SWANLAB_PROJECT}" "${QV_SWANLAB_PROJECT}" "${EVAL_SWANLAB_PROJECT}"

if [[ -z "${FLOW_ACTOR_CKPT}" ]]; then
  if FLOW_ACTOR_CKPT="$(find_existing_flow_ckpt "${FLOW_SAVE_DIR}")"; then
    printf '[run_dorl_mac_kuai_train] found existing flow checkpoint, skip flow training: %s\n' \
      "${FLOW_ACTOR_CKPT}"
  else
    run_command \
      "${PYTHON_BIN}" examples/our_model/runners/pretrain_flow_bc.py \
      "${COMMON_ARGS[@]}" \
      --swanlab_project "${FLOW_SWANLAB_PROJECT}" \
      --run_name "${FLOW_RUN_NAME}" \
      --actor_backend "${ACTOR_BACKEND}" \
      --pretrain_steps "${PRETRAIN_STEPS}" \
      --flow_steps "${FLOW_STEPS}" \
      --actor_lr "${ACTOR_LR}" \
      --bc_weight "${BC_WEIGHT}" \
      --save_dir "${FLOW_SAVE_DIR}" \
      --log_interval "${PRETRAIN_LOG_INTERVAL}"
    FLOW_ACTOR_CKPT="${FLOW_SAVE_DIR}/latest.pt"
  fi
else
  printf '[run_dorl_mac_kuai_train] use user provided flow checkpoint: %s\n' "${FLOW_ACTOR_CKPT}"
fi

if [[ "${DRY_RUN}" != "1" && ! -f "${FLOW_ACTOR_CKPT}" ]]; then
  printf '[run_dorl_mac_kuai_train] ERROR: flow checkpoint does not exist: %s\n' "${FLOW_ACTOR_CKPT}" >&2
  exit 1
fi

reset_metrics_log "${MAC_SAVE_DIR}/metrics.jsonl" "qv"
run_command \
  "${PYTHON_BIN}" examples/our_model/runners/train_dorl_mac_qv.py \
  "${COMMON_ARGS[@]}" \
  --swanlab_project "${QV_SWANLAB_PROJECT}" \
  --run_name "${QV_RUN_NAME}" \
  --flow_actor_ckpt "${FLOW_ACTOR_CKPT}" \
  --epoch "${EPOCH}" \
  --step-per-epoch "${STEP_PER_EPOCH}" \
  --qv_lr "${QV_LR}" \
  --target_tau "${TARGET_TAU}" \
  --num_samples_train "${NUM_SAMPLES_TRAIN}" \
  --num_samples_test "${NUM_SAMPLES_TEST}" \
  --repeat_policy "${REPEAT_POLICY}" \
  --test-num "${TEST_NUM}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --buffer-size "${BUFFER_SIZE}" \
  --save_dir "${MAC_SAVE_DIR}" \
  --eval_save_dir "${MAC_SAVE_DIR}/eval_during_train" \
  --log_interval "${QV_LOG_INTERVAL}"

reset_metrics_log "${EVAL_SAVE_DIR}/metrics.jsonl" "eval"
run_command \
  "${PYTHON_BIN}" examples/our_model/runners/eval_dorl_mac.py \
  "${COMMON_ARGS[@]}" \
  --swanlab_project "${EVAL_SWANLAB_PROJECT}" \
  --run_name "${EVAL_RUN_NAME}" \
  --mac_ckpt "${MAC_SAVE_DIR}/latest.pt" \
  --num_samples_test "${NUM_SAMPLES_TEST}" \
  --test-num "${TEST_NUM}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --buffer-size "${BUFFER_SIZE}" \
  --eval_save_dir "${EVAL_SAVE_DIR}"

printf '\n[run_dorl_mac_kuai_train] done. outputs are under %s\n' "${RUN_DIR}"
