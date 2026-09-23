#!/usr/bin/env bash
# RAVE-H 主 TD 训练/评估；MAC checkpoint 必须具有预编码状态与 dynamics。
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-easyrl4rec}"
export PYTHONPATH="${PWD}:${PWD}/src:${PWD}/src/DeepCTR-Torch:${PWD}/src/tianshou:${PWD}/examples/policy:${PYTHONPATH:-}"
# 默认上传 SwanLab；设为 offline 或 disabled 时仅保留 saved_models 下的本地日志。
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

PYTHON_BIN="${PYTHON_BIN:-/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python}"
MODE="${MODE:-train}"
ENV_NAME="${ENV_NAME:-KuaiEnv-v0}"
DEVICE="${DEVICE:-cpu}"
SAVE_ROOT="${SAVE_ROOT:-saved_models}"
CHUNK_SIZE="${CHUNK_SIZE:-3}"
if [[ -n "${SWANLAB_PROJECT:-}" ]]; then
  SWANLAB_PROJECT="${SWANLAB_PROJECT}"
elif [[ "${MODE}" == "eval" ]]; then
  SWANLAB_PROJECT="RAVE-H-Eval"
else
  SWANLAB_PROJECT="RAVE-H"
fi
RUN_NAME="${RUN_NAME:-}"
QV_TYPE="${QV_TYPE:-}"
ETA_T="${ETA_T:-}"
ETA_D="${ETA_D:-}"
ETA_C="${ETA_C:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
ARGS=(--mode "${MODE}" --device "${DEVICE}" --save_root "${SAVE_ROOT}"
  --swanlab_project "${SWANLAB_PROJECT}")
if [[ -n "${OUTPUT_ROOT}" ]]; then ARGS+=(--output_root "${OUTPUT_ROOT}"); fi
if [[ -n "${RUN_NAME}" ]]; then ARGS+=(--run_name "${RUN_NAME}"); fi
case "${MODE}" in
  train)
    : "${MAC_CHECKPOINT:?请设置 MAC_CHECKPOINT 为已训练的共享预训练 checkpoint}"
    case "${ENV_NAME}" in
      KuaiEnv-v0) DEFAULT_LEAVE_WINDOW=4 ;;
      KuaiRand-v0) DEFAULT_LEAVE_WINDOW=1 ;;
      *) printf '不支持的环境：%s\n' "${ENV_NAME}" >&2; exit 2 ;;
    esac
    ARGS+=(--env "${ENV_NAME}" --mac_checkpoint "${MAC_CHECKPOINT}"
      --chunk_size "${CHUNK_SIZE}"
      --max_turn "${MAX_TURN:-100}" --num_leave_compute "${NUM_LEAVE_COMPUTE:-${DEFAULT_LEAVE_WINDOW}}"
      --leave_threshold "${LEAVE_THRESHOLD:-0}" --num_candidates "${NUM_CANDIDATES:-32}"
      --force_length "${FORCE_LENGTH:-${MAX_TURN:-100}}"
      --completion_window "${COMPLETION_WINDOW:-${CHUNK_SIZE}}"
      --test_num "${TEST_NUM:-10}" --buffer_size "${BUFFER_SIZE:-0}"
      --steps "${STEPS:-10000}" --epochs "${EPOCHS:-100}"
      --eval_every_epochs "${EVAL_EVERY_EPOCHS:-1}"
      --eval_episodes "${TRAIN_EVAL_EPISODES:-20}"
      --eval_swanlab_project "${EVAL_SWANLAB_PROJECT:-RAVE-H-Eval}"
      --batch_size "${BATCH_SIZE:-32}"
      --rollout_candidates "${ROLLOUT_CANDIDATES:-4}"
      --episodes_per_collect "${EPISODES_PER_COLLECT:-32}"
      --ordinary_rollout_depth "${ORDINARY_ROLLOUT_DEPTH:-3}"
      --lambda_chunk "${LAMBDA_CHUNK:-1.0}"
      --gamma "${GAMMA:-0.9}"
      --seed "${SEED:-2023}")
    # 该配置只控制部署/评估选长；训练行为收集始终固定执行完整 K。
    if [[ -n "${FIXED_HORIZON:-}" ]]; then
      ARGS+=(--fixed_horizon "${FIXED_HORIZON}")
    fi
    # 与现有 MAC 主训练脚本保持相同历史屏蔽；chunk 内仍允许重复。
    if [[ "${MASK_HISTORY:-1}" == "1" ]]; then ARGS+=(--mask_history); fi
    if [[ -n "${QV_TYPE}" ]]; then ARGS+=(--qv_type "${QV_TYPE}"); fi
    # 辅助目标权重只属于四头模型；单头 Transformer 始终严格只学习 R。
    if [[ "${QV_TYPE}" == "multi_prefix_transformer" ]]; then
      if [[ -n "${ETA_T}" ]]; then ARGS+=(--eta_t "${ETA_T}"); fi
      if [[ -n "${ETA_D}" ]]; then ARGS+=(--eta_d "${ETA_D}"); fi
      if [[ -n "${ETA_C}" ]]; then ARGS+=(--eta_c "${ETA_C}"); fi
    fi
    ;;
  eval)
    : "${RAVE_CHECKPOINT:?请设置 RAVE_CHECKPOINT 为 RAVE 的 latest.pt}"
    ARGS+=(--rave_checkpoint "${RAVE_CHECKPOINT}" --eval_split "${EVAL_SPLIT:-validation}"
      --eval_episodes "${EVAL_EPISODES:-20}")
    if [[ "${DORL_EVAL_PARITY:-0}" == "1" ]]; then
      ARGS+=(--dorl_eval_parity --env "${ENV_NAME}" --max_turn "${MAX_TURN:-100}"
        --num_leave_compute "${NUM_LEAVE_COMPUTE:-1}"
        --leave_threshold "${LEAVE_THRESHOLD:-0}"
        --force_length "${FORCE_LENGTH:-${MAX_TURN:-100}}"
        --completion_window "${COMPLETION_WINDOW:-${CHUNK_SIZE}}"
        --test_num "${TEST_NUM:-10}" --buffer_size "${BUFFER_SIZE:-0}"
        --num_candidates "${NUM_CANDIDATES:-32}")
      # 独立评估默认沿用自适应 H；固定 H 仅用于显式消融。
      if [[ -n "${FIXED_HORIZON:-}" ]]; then
        ARGS+=(--fixed_horizon "${FIXED_HORIZON}")
      fi
    fi
    ;;
  *) printf 'MODE 必须为 train 或 eval。\n' >&2; exit 2 ;;
esac
if [[ -n "${OUTPUT_DIR:-}" ]]; then ARGS+=(--output_dir "${OUTPUT_DIR}"); fi
exec "${PYTHON_BIN}" -m examples.our_model.runners.run_rave "${ARGS[@]}" "$@"
