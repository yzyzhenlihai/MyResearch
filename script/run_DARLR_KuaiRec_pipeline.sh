#!/usr/bin/env bash
# DARLR KuaiRec 分阶段实验流水线。
# 默认串行；例如 CUDA_DEVICES="0 1" PARALLEL_JOBS=2 可在阶段内两卡并行。
set -euo pipefail

cd "$(dirname "$0")/.."

STAGE="${STAGE:-stability}"
SEED="${SEED:-2023}"
CUDA="${CUDA:-4 5 6 7}"
CUDA_DEVICES="${CUDA_DEVICES:-${CUDA}}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
DRY_RUN="${DRY_RUN:-0}"
SAVE_MODEL="${SAVE_MODEL:-1}"
SAVE_BEST_ONLY="${SAVE_BEST_ONLY:-1}"
BEST_METRIC="${BEST_METRIC:-NX_0}"
RUN_PREFIX="${RUN_PREFIX:-DARLR_KuaiRec}"
LAMBDA_VARIANCE="${LAMBDA_VARIANCE:-0.05}"
PAPER_TARGETS="${PAPER_TARGETS:-k lambda_s lambda_d lambda_u lambda_e layers heads window}"

case "${STAGE}" in
  baseline|causal|stability|paper|all)
    ;;
  *)
    echo "Unsupported STAGE='${STAGE}'. Expected baseline, causal, stability, paper, or all." >&2
    exit 2
    ;;
esac

if ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_JOBS must be a positive integer, got '${PARALLEL_JOBS}'." >&2
  exit 2
fi

read -r -a CUDA_DEVICE_POOL <<< "${CUDA_DEVICES}"
if (( ${#CUDA_DEVICE_POOL[@]} == 0 )); then
  echo "CUDA_DEVICES must contain at least one CUDA index." >&2
  exit 2
fi
for cuda_device in "${CUDA_DEVICE_POOL[@]}"; do
  if ! [[ "${cuda_device}" =~ ^[0-9]+$ ]]; then
    echo "CUDA device must be a non-negative integer, got '${cuda_device}'." >&2
    exit 2
  fi
done
if (( PARALLEL_JOBS > ${#CUDA_DEVICE_POOL[@]} )); then
  echo "PARALLEL_JOBS=${PARALLEL_JOBS} exceeds CUDA device count=${#CUDA_DEVICE_POOL[@]}." >&2
  echo "Provide one distinct GPU per concurrent training process." >&2
  exit 2
fi

ACTIVE_PIDS=()
ACTIVE_LABELS=()
NEXT_CUDA_INDEX=0

# 等待当前批次全部结束；任意子任务失败都会停止后续阶段。
wait_for_jobs() {
  local failed_jobs=0
  local job_index
  local job_status
  for job_index in "${!ACTIVE_PIDS[@]}"; do
    if wait "${ACTIVE_PIDS[${job_index}]}"; then
      printf 'DONE: %s\n' "${ACTIVE_LABELS[${job_index}]}"
    else
      job_status=$?
      printf 'FAILED(%s): %s\n' \
        "${job_status}" "${ACTIVE_LABELS[${job_index}]}" >&2
      failed_jobs=1
    fi
  done
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  NEXT_CUDA_INDEX=0
  if (( failed_jobs != 0 )); then
    return 1
  fi
}

# 按 GPU 池启动任务；达到并发上限后先等待当前整批任务。
launch_job() {
  local job_label="$1"
  shift
  if (( ${#ACTIVE_PIDS[@]} >= PARALLEL_JOBS )); then
    wait_for_jobs
  fi

  local job_cuda="${CUDA_DEVICE_POOL[${NEXT_CUDA_INDEX}]}"
  NEXT_CUDA_INDEX=$((NEXT_CUDA_INDEX + 1))
  printf 'LAUNCH: %s on CUDA=%s\n' "${job_label}" "${job_cuda}"
  (
    export CUDA="${job_cuda}"
    "$@"
  ) &
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("${job_label}")
}

# 收到终止信号时停止仍在运行的 launcher 子进程。
terminate_active_jobs() {
  local active_pid
  for active_pid in "${ACTIVE_PIDS[@]}"; do
    kill "${active_pid}" 2>/dev/null || true
  done
}

trap terminate_active_jobs INT TERM

run_dorl() {
  local variant="$1"
  shift
  env \
    SEED="${SEED}" \
    CUDA="${CUDA}" \
    DRY_RUN="${DRY_RUN}" \
    SAVE_MODEL="${SAVE_MODEL}" \
    SAVE_BEST_ONLY="${SAVE_BEST_ONLY}" \
    BEST_METRIC="${BEST_METRIC}" \
    MESSAGE="${RUN_PREFIX}_baseline_${variant}_seed${SEED}" \
    "$@" \
    bash script/run_DORL_KuaiRec_control.sh
}

run_darlr() {
  local phase="$1"
  local variant="$2"
  shift 2
  env \
    SEED="${SEED}" \
    CUDA="${CUDA}" \
    DRY_RUN="${DRY_RUN}" \
    SAVE_MODEL="${SAVE_MODEL}" \
    SAVE_BEST_ONLY="${SAVE_BEST_ONLY}" \
    BEST_METRIC="${BEST_METRIC}" \
    MESSAGE="${RUN_PREFIX}_${phase}_${variant}_seed${SEED}" \
    "$@" \
    bash script/run_DARLR_KuaiRec.sh
}

run_baseline_stage() {
  launch_job baseline/dorl_control run_dorl control \
    "MAX_GRAD_NORM=" \
    "LAMBDA_VARIANCE=${LAMBDA_VARIANCE}"
  launch_job baseline/darlr_static_dorl run_darlr baseline static_dorl \
    "SELECTOR_POLICY_MODE=fixed" \
    "SELECTOR_LOSS_COEF=0" \
    "SELECTOR_LR=0.001" \
    "MAX_GRAD_NORM=" \
    "SELECTOR_ENT_COEF=0" \
    "SELECTOR_ADVANTAGE_NORMALIZATION=0" \
    "SELECTOR_REWARD_NORMALIZATION=0" \
    "DYNAMIC_REWARD_MODE=static_dorl" \
    "DYNAMIC_UNCERTAINTY_MODE=off" \
    "LAMBDA_VARIANCE=${LAMBDA_VARIANCE}"
  wait_for_jobs
}

run_causal_stage() {
  launch_job causal/static_dorl run_darlr causal static_dorl \
    "SELECTOR_POLICY_MODE=fixed" \
    "SELECTOR_LOSS_COEF=0" \
    "DYNAMIC_REWARD_MODE=static_dorl" \
    "DYNAMIC_UNCERTAINTY_MODE=off" \
    "LAMBDA_VARIANCE=${LAMBDA_VARIANCE}"
  launch_job causal/random_selector_no_uncertainty \
    run_darlr causal random_selector_no_uncertainty \
    "SELECTOR_POLICY_MODE=random" \
    "SELECTOR_LOSS_COEF=0" \
    "SELECTOR_LR=0.001" \
    "MAX_GRAD_NORM=" \
    "SELECTOR_ENT_COEF=0" \
    "SELECTOR_ADVANTAGE_NORMALIZATION=0" \
    "SELECTOR_REWARD_NORMALIZATION=0" \
    "DYNAMIC_REWARD_MODE=reference_mean" \
    "DYNAMIC_UNCERTAINTY_MODE=off"
  launch_job causal/learned_selector_no_uncertainty \
    run_darlr causal learned_selector_no_uncertainty \
    "SELECTOR_POLICY_MODE=learned" \
    "SELECTOR_LOSS_COEF=1" \
    "SELECTOR_LR=0.001" \
    "MAX_GRAD_NORM=" \
    "SELECTOR_ENT_COEF=0" \
    "SELECTOR_ADVANTAGE_NORMALIZATION=0" \
    "SELECTOR_REWARD_NORMALIZATION=0" \
    "DYNAMIC_REWARD_MODE=reference_mean" \
    "DYNAMIC_UNCERTAINTY_MODE=off"
  launch_job causal/full run_darlr causal full \
    "SELECTOR_POLICY_MODE=learned" \
    "SELECTOR_LOSS_COEF=1" \
    "SELECTOR_LR=0.001" \
    "MAX_GRAD_NORM=" \
    "SELECTOR_ENT_COEF=0" \
    "SELECTOR_ADVANTAGE_NORMALIZATION=0" \
    "SELECTOR_REWARD_NORMALIZATION=0" \
    "DYNAMIC_REWARD_MODE=reference_mean" \
    "DYNAMIC_UNCERTAINTY_MODE=dynamic"
  wait_for_jobs
}

run_stability_stage() {
  launch_job stability/original run_darlr stability original \
    "SELECTOR_LR=0.001" \
    "MAX_GRAD_NORM=" \
    "SELECTOR_ENT_COEF=0" \
    "SELECTOR_ADVANTAGE_NORMALIZATION=0" \
    "SELECTOR_REWARD_NORMALIZATION=0"
  launch_job stability/lr_clip run_darlr stability lr_clip \
    "SELECTOR_LR=0.0003" \
    "MAX_GRAD_NORM=5.0" \
    "SELECTOR_ENT_COEF=0" \
    "SELECTOR_ADVANTAGE_NORMALIZATION=0" \
    "SELECTOR_REWARD_NORMALIZATION=0"
  launch_job stability/entropy run_darlr stability entropy \
    "SELECTOR_LR=0.0003" \
    "MAX_GRAD_NORM=5.0" \
    "SELECTOR_ENT_COEF=0.001" \
    "SELECTOR_ADVANTAGE_NORMALIZATION=0" \
    "SELECTOR_REWARD_NORMALIZATION=0"
  launch_job stability/normalized run_darlr stability normalized \
    "SELECTOR_LR=0.0003" \
    "MAX_GRAD_NORM=5.0" \
    "SELECTOR_ENT_COEF=0.001" \
    "SELECTOR_ADVANTAGE_NORMALIZATION=1" \
    "SELECTOR_REWARD_NORMALIZATION=1"
  wait_for_jobs
}

run_paper_stage() {
  local target
  for target in ${PAPER_TARGETS}; do
    launch_job "paper/${target}" env \
      TUNE_TARGET="${target}" \
      SEEDS="${SEEDS:-${SEED}}" \
      DRY_RUN="${DRY_RUN}" \
      SAVE_MODEL="${SAVE_MODEL}" \
      SAVE_BEST_ONLY="${SAVE_BEST_ONLY}" \
      BEST_METRIC="${BEST_METRIC}" \
      RUN_PREFIX="${RUN_PREFIX}" \
      bash script/run_DARLR_KuaiRec_tune.sh
  done
  wait_for_jobs
}

case "${STAGE}" in
  baseline)
    run_baseline_stage
    ;;
  causal)
    run_causal_stage
    ;;
  stability)
    run_stability_stage
    ;;
  paper)
    run_paper_stage
    ;;
  all)
    run_baseline_stage
    run_causal_stage
    run_stability_stage
    run_paper_stage
    ;;
esac
