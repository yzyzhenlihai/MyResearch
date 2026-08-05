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

ENV_NAME="${ENV_NAME:-KuaiEnv-v0}" # KuaiRand-v0  KuaiEnv-v0
USER_MODEL_NAME="${USER_MODEL_NAME:-DeepFM}"
READ_MESSAGE="${READ_MESSAGE:-pointneg}"

case "${ENV_NAME}" in
  KuaiEnv-v0)
    DEFAULT_DATASET_PATH="data/KuaiRec/data_processed/DM_KuaiEnv-v0_small_data.pkl"
    DEFAULT_MAX_TURN=100
    DEFAULT_NUM_LEAVE_COMPUTE=1
    DEFAULT_LEAVE_THRESHOLD=0
    DEFAULT_RUN_NAME="dorl-mac-kuai-catbc-qv"
    ;;
  KuaiRand-v0)
    DEFAULT_DATASET_PATH="data/KuaiRand_Pure/data_processed/DM_KuaiRand-v0_test_data.pkl"
    DEFAULT_MAX_TURN=100
    DEFAULT_NUM_LEAVE_COMPUTE=1
    DEFAULT_LEAVE_THRESHOLD=0
    DEFAULT_RUN_NAME="dorl-mac-kuairand-catbc-qv"
    ;;
  *)
    printf '[run_dorl_mac_kuai_train] ERROR: unsupported ENV_NAME=%s; choose KuaiEnv-v0 or KuaiRand-v0.\n' \
      "${ENV_NAME}" >&2
    exit 2
    ;;
esac

DATASET_PATH="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
WHICH_TRACKER="${WHICH_TRACKER:-avg}"
REWARD_HANDLE="${REWARD_HANDLE:-cat}"
WINDOW_SIZE="${WINDOW_SIZE:-3}"
CHUNK_SIZE="${CHUNK_SIZE:-7}"
# H 仅控制评估时每次规划后连续执行的前缀长度；默认 H=K 保持原开环语义。
EXECUTION_HORIZON="${EXECUTION_HORIZON:-${CHUNK_SIZE}}"
COMPLETION_WINDOW="${COMPLETION_WINDOW:-${CHUNK_SIZE}}"
ENABLE_OPEN_LOOP_DIAGNOSTICS="${ENABLE_OPEN_LOOP_DIAGNOSTICS:-0}"
GAMMA="${GAMMA:-0.9}"
SEED="${SEED:-2023}"
DEVICE="${DEVICE:-cuda:7}"
CUDA="${CUDA:-7}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_TURN="${MAX_TURN:-${DEFAULT_MAX_TURN}}"
FORCE_LENGTH="${FORCE_LENGTH:-${MAX_TURN}}"
NUM_LEAVE_COMPUTE="${NUM_LEAVE_COMPUTE:-${DEFAULT_NUM_LEAVE_COMPUTE}}"
LEAVE_THRESHOLD="${LEAVE_THRESHOLD:-${DEFAULT_LEAVE_THRESHOLD}}"
INVALID_ACTION_PENALTY="${INVALID_ACTION_PENALTY:-0.0}"
# 默认为空表示使用全量离线轨迹和全量 action chunks。
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-}"
MAX_CHUNKS="${MAX_CHUNKS:-}"
USER_MODEL_ROOT="saved_models/${ENV_NAME}/${USER_MODEL_NAME}"
ITEM_EMBEDDING_PATH="${ITEM_EMBEDDING_PATH:-${USER_MODEL_ROOT}/embeddings/[${READ_MESSAGE}]_emb_item_val_M0.pt}"
USER_EMBEDDING_PATH="${USER_EMBEDDING_PATH:-${USER_MODEL_ROOT}/embeddings/[${READ_MESSAGE}]_emb_user_val_M0.pt}"
PREDICTED_MAT_PATH="${PREDICTED_MAT_PATH:-${USER_MODEL_ROOT}/matsPre/[${READ_MESSAGE}]_matPre.pickle}"
MAXVAR_MAT_PATH="${MAXVAR_MAT_PATH:-${USER_MODEL_ROOT}/matsVar/[${READ_MESSAGE}]_matVar.pickle}"

USE_ENTROPY_REWARD="${USE_ENTROPY_REWARD:-1}"
USE_UNCERTAINTY_PENALTY="${USE_UNCERTAINTY_PENALTY:-1}"
LAMBDA_ENTROPY="${LAMBDA_ENTROPY:-0.5}"
LAMBDA_VARIANCE="${LAMBDA_VARIANCE:-1}"
# predicted_mat 归一化模式（决定 pred_reward 的数值尺度）：
#   none            → 保留 DeepFM 原始输出（KuaiRec 上均值 ~1e-4，与 λ_entropy × entropy 严重错配）
#   global_minmax   → 整表线性缩放到 [0, 1]
#   per_user_max    → 每行除以该用户最大 |pred|，把 per-user top-1 拉到 1.0（推荐）
#   per_user_minmax → 每行 min-max 归一化到 [0, 1]
#   sigmoid         → 套 sigmoid 映射到 (0, 1) 概率域
PREDICTED_MAT_NORMALIZE="${PREDICTED_MAT_NORMALIZE:-per_user_max}"
ENTROPY_WINDOW="${ENTROPY_WINDOW:-1 2}"
FEATURE_LEVEL="${FEATURE_LEVEL:-1}"
IS_SORTED="${IS_SORTED:-1}"
DYNAMICS_LOSS_WEIGHT="${DYNAMICS_LOSS_WEIGHT:-1.0}"
# 阶段 ①：Categorical BC 预训练
PRETRAIN_STEPS="${PRETRAIN_STEPS:-100000}"
ACTOR_LR="${ACTOR_LR:-0.0003}"
PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-100}"

# 阶段 ②：Q/V 训练
EPOCH="${EPOCH:-100}"
STEP_PER_EPOCH="${STEP_PER_EPOCH:-1000}"
QV_LR="${QV_LR:-0.0003}"
TARGET_TAU="${TARGET_TAU:-0.005}"
NUM_SAMPLES_TRAIN="${NUM_SAMPLES_TRAIN:-8}"
NUM_SAMPLES_TEST="${NUM_SAMPLES_TEST:-32}"
REPEAT_POLICY="${REPEAT_POLICY:-mask}" # mask,采样阶段屏蔽已推荐 item
# MAC chunk-level value expansion 与退出规则控制：
#   ROLLOUT_DEPTH=1  → 单 chunk 一步 TD（默认，历史行为）
#   ROLLOUT_DEPTH>1  → 启用 MAC 完整版 imagined rollout + chunk-level TD(λ)
#   LAMBDA_CHUNK ∈ [0, 1]：0 近似单 chunk 一步 TD，1 为 chunk-level 蒙特卡洛
#   LEAVE_POLICY=penalty  → 违规只惩罚不截断
#   LEAVE_POLICY=terminate→ 与原版 DORL 一致，触发退出即终止
ROLLOUT_DEPTH="${ROLLOUT_DEPTH:-3}"
LAMBDA_CHUNK="${LAMBDA_CHUNK:-1.0}"
LEAVE_POLICY="${LEAVE_POLICY:-terminate}"
TEST_NUM="${TEST_NUM:-10}"
EVAL_EPISODES="${EVAL_EPISODES:-0}"
EVAL_EVERY_N_EPOCHS="${EVAL_EVERY_N_EPOCHS:-1}"
BUFFER_SIZE="${BUFFER_SIZE:-0}"
QV_LOG_INTERVAL="${QV_LOG_INTERVAL:-100}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
ENABLE_STEP_PROFILER="${ENABLE_STEP_PROFILER:-0}"

# SwanLab 项目命名（离散 Categorical MAC 版）
SWANLAB_PROJECT="${SWANLAB_PROJECT:-DORL-MAC}"
BC_SWANLAB_PROJECT="${BC_SWANLAB_PROJECT:-${SWANLAB_PROJECT}-CategoricalBC}"
QV_SWANLAB_PROJECT="${QV_SWANLAB_PROJECT:-${SWANLAB_PROJECT}-QV}"
EVAL_SWANLAB_PROJECT="${EVAL_SWANLAB_PROJECT:-${SWANLAB_PROJECT}-Eval}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
# 默认在 run_name / run_dir 里拼上 K=${CHUNK_SIZE}，让不同 K 的实验自动落到不同目录：
#   - 相同 K 再次运行：BC 目录已有 latest.pt → 自动短路复用；
#   - 不同 K 运行：BC / QV / eval 全部落到新目录，自动触发 BC 重训。
# 用户仍可显式覆盖 RUN_DIR / BC_SAVE_DIR / *_RUN_NAME 以复用旧路径。
RUN_NAME_WITH_K="${RUN_NAME}-K${CHUNK_SIZE}"
BC_RUN_NAME="${BC_RUN_NAME:-${RUN_NAME_WITH_K}-bc}"
QV_RUN_NAME="${QV_RUN_NAME:-${RUN_NAME_WITH_K}-qv}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-${RUN_NAME_WITH_K}-H${EXECUTION_HORIZON}-eval}"
RUN_DIR="${RUN_DIR:-saved_models/${ENV_NAME}/DORL_MAC/${RUN_NAME_WITH_K}}"
BC_SAVE_DIR="${BC_SAVE_DIR:-${RUN_DIR}/categorical_bc}"
MAC_SAVE_DIR="${MAC_SAVE_DIR:-${RUN_DIR}/mac_agent}"
EVAL_SAVE_DIR="${EVAL_SAVE_DIR:-${RUN_DIR}/eval_H${EXECUTION_HORIZON}}"
BC_ACTOR_CKPT="${BC_ACTOR_CKPT:-}"

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
  --execution_horizon "${EXECUTION_HORIZON}"
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

if [[ -n "${MAX_TRAJECTORIES}" ]]; then
  COMMON_ARGS+=(--max_trajectories "${MAX_TRAJECTORIES}")
fi
if [[ -n "${MAX_CHUNKS}" ]]; then
  COMMON_ARGS+=(--max_chunks "${MAX_CHUNKS}")
fi
COMMON_ARGS+=(--item_embedding_path "${ITEM_EMBEDDING_PATH}")
COMMON_ARGS+=(--user_embedding_path "${USER_EMBEDDING_PATH}")
COMMON_ARGS+=(--predicted_mat_path "${PREDICTED_MAT_PATH}")
COMMON_ARGS+=(--maxvar_mat_path "${MAXVAR_MAT_PATH}")
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
if [[ "${ENABLE_OPEN_LOOP_DIAGNOSTICS}" == "1" || "${ENABLE_OPEN_LOOP_DIAGNOSTICS}" == "true" || "${ENABLE_OPEN_LOOP_DIAGNOSTICS}" == "True" ]]; then
  COMMON_ARGS+=(--enable_open_loop_diagnostics)
else
  COMMON_ARGS+=(--disable_open_loop_diagnostics)
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

find_existing_bc_ckpt() {
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

printf '[run_dorl_mac_kuai_train] run_dir=%s (chunk_size=K=%s, execution_horizon=H=%s)\n' \
  "${RUN_DIR}" "${CHUNK_SIZE}" "${EXECUTION_HORIZON}"
printf '[run_dorl_mac_kuai_train] dataset: env=%s, trajectories=%s\n' \
  "${ENV_NAME}" "${DATASET_PATH}"
printf '[run_dorl_mac_kuai_train] user model assets: item=%s, user=%s, prediction=%s, variance=%s\n' \
  "${ITEM_EMBEDDING_PATH}" "${USER_EMBEDDING_PATH}" "${PREDICTED_MAT_PATH}" "${MAXVAR_MAT_PATH}"
printf '[run_dorl_mac_kuai_train] open-loop diagnostics: CCR_window=%s, ADR=%s\n' \
  "${COMPLETION_WINDOW}" "${ENABLE_OPEN_LOOP_DIAGNOSTICS}"
printf '[run_dorl_mac_kuai_train] pretrain (categorical BC): steps=%s, actor_lr=%s\n' \
  "${PRETRAIN_STEPS}" "${ACTOR_LR}"
printf '[run_dorl_mac_kuai_train] qv: epoch=%s, step_per_epoch=%s, test_num=%s, rollout_depth=%s, lambda_chunk=%s, leave_policy=%s, repeat_policy=%s\n' \
  "${EPOCH}" "${STEP_PER_EPOCH}" "${TEST_NUM}" \
  "${ROLLOUT_DEPTH}" "${LAMBDA_CHUNK}" "${LEAVE_POLICY}" "${REPEAT_POLICY}"
printf '[run_dorl_mac_kuai_train] reward: entropy=%s(lambda=%s, window=%s), uncertainty=%s(lambda=%s), invalid_penalty=%s, pred_mat_normalize=%s\n' \
  "${USE_ENTROPY_REWARD}" "${LAMBDA_ENTROPY}" "${ENTROPY_WINDOW}" \
  "${USE_UNCERTAINTY_PENALTY}" "${LAMBDA_VARIANCE}" "${INVALID_ACTION_PENALTY}" \
  "${PREDICTED_MAT_NORMALIZE}"
printf '[run_dorl_mac_kuai_train] swanlab projects: bc=%s, qv=%s, eval=%s\n' \
  "${BC_SWANLAB_PROJECT}" "${QV_SWANLAB_PROJECT}" "${EVAL_SWANLAB_PROJECT}"

# 阶段 ①：Categorical BC 预训练；若目录里已有 latest.pt/其他 *.pt，短路跳过。
if [[ -z "${BC_ACTOR_CKPT}" ]]; then
  if BC_ACTOR_CKPT="$(find_existing_bc_ckpt "${BC_SAVE_DIR}")"; then
    printf '[run_dorl_mac_kuai_train] found existing BC checkpoint, skip BC training: %s\n' \
      "${BC_ACTOR_CKPT}"
  else
    reset_metrics_log "${BC_SAVE_DIR}/metrics.jsonl" "bc"
    run_command \
      "${PYTHON_BIN}" examples/our_model/runners/pretrain_categorical_bc.py \
      "${COMMON_ARGS[@]}" \
      --swanlab_project "${BC_SWANLAB_PROJECT}" \
      --run_name "${BC_RUN_NAME}" \
      --pretrain_steps "${PRETRAIN_STEPS}" \
      --actor_lr "${ACTOR_LR}" \
      --save_dir "${BC_SAVE_DIR}" \
      --log_interval "${PRETRAIN_LOG_INTERVAL}"
    BC_ACTOR_CKPT="${BC_SAVE_DIR}/latest.pt"
  fi
else
  printf '[run_dorl_mac_kuai_train] use user provided BC checkpoint: %s\n' "${BC_ACTOR_CKPT}"
fi

if [[ "${DRY_RUN}" != "1" && ! -f "${BC_ACTOR_CKPT}" ]]; then
  printf '[run_dorl_mac_kuai_train] ERROR: BC checkpoint does not exist: %s\n' "${BC_ACTOR_CKPT}" >&2
  exit 1
fi

# 阶段 ②：Q/V 训练
reset_metrics_log "${MAC_SAVE_DIR}/metrics.jsonl" "qv"
run_command \
  "${PYTHON_BIN}" examples/our_model/runners/train_dorl_mac_qv.py \
  "${COMMON_ARGS[@]}" \
  --swanlab_project "${QV_SWANLAB_PROJECT}" \
  --run_name "${QV_RUN_NAME}" \
  --bc_actor_ckpt "${BC_ACTOR_CKPT}" \
  --epoch "${EPOCH}" \
  --step-per-epoch "${STEP_PER_EPOCH}" \
  --qv_lr "${QV_LR}" \
  --target_tau "${TARGET_TAU}" \
  --num_samples_train "${NUM_SAMPLES_TRAIN}" \
  --num_samples_test "${NUM_SAMPLES_TEST}" \
  --repeat_policy "${REPEAT_POLICY}" \
  --rollout_depth "${ROLLOUT_DEPTH}" \
  --lambda_chunk "${LAMBDA_CHUNK}" \
  --leave_policy "${LEAVE_POLICY}" \
  --test-num "${TEST_NUM}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --eval_every_n_epochs "${EVAL_EVERY_N_EPOCHS}" \
  --buffer-size "${BUFFER_SIZE}" \
  --save_dir "${MAC_SAVE_DIR}" \
  --eval_save_dir "${MAC_SAVE_DIR}/eval_during_train_H${EXECUTION_HORIZON}" \
  --log_interval "${QV_LOG_INTERVAL}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  $( [ "${ENABLE_STEP_PROFILER}" = "1" ] && echo "--enable_step_profiler" )

# 阶段 ③：最终评估
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
