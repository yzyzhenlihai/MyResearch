#!/usr/bin/env bash
# DARLR KuaiRec 分阶段实验流水线。
# 默认每次启动任务前按空闲显存自动选卡，且每卡最多一个任务。
# AUTO_PARALLEL_MAX_JOBS 可设定 auto 并发硬上限；整数 PARALLEL_JOBS 可切换为手动模式。
set -euo pipefail

cd "$(dirname "$0")/.."

STAGE="${STAGE:-paper}"
SEED="${SEED:-2023}"
CUDA_WAS_EXPLICIT=0
CUDA_DEVICES_WAS_EXPLICIT=0
if [[ -n "${CUDA+x}" ]]; then
  CUDA_WAS_EXPLICIT=1
fi
if [[ -n "${CUDA_DEVICES+x}" ]]; then
  CUDA_DEVICES_WAS_EXPLICIT=1
fi
CUDA="${CUDA:-0 1 2 3 4 5 6 7}"
if (( CUDA_DEVICES_WAS_EXPLICIT != 0 )); then
  CUDA_DEVICES="${CUDA_DEVICES}"
elif (( CUDA_WAS_EXPLICIT != 0 )); then
  CUDA_DEVICES="${CUDA}"
else
  CUDA_DEVICES=""
fi
PARALLEL_JOBS="${PARALLEL_JOBS:-auto}"
GPU_MIN_FREE_MEMORY_MB="${GPU_MIN_FREE_MEMORY_MB:-15000}"
GPU_MAX_UTILIZATION_PERCENT="${GPU_MAX_UTILIZATION_PERCENT:-100}"
GPU_UTILIZATION_WARNING_PERCENT="${GPU_UTILIZATION_WARNING_PERCENT:-90}"
AUTO_PARALLEL_MAX_JOBS="${AUTO_PARALLEL_MAX_JOBS:-0}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-/tmp/easyrl4rec-darlr-gpu-locks-${UID}}"
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

read -r -a PAPER_TARGET_LIST <<< "${PAPER_TARGETS}"
case "${STAGE}" in
  baseline)
    AUTO_STAGE_TASK_LIMIT=2
    ;;
  causal|stability)
    AUTO_STAGE_TASK_LIMIT=4
    ;;
  paper)
    AUTO_STAGE_TASK_LIMIT="${#PAPER_TARGET_LIST[@]}"
    ;;
  all)
    AUTO_STAGE_TASK_LIMIT="${#PAPER_TARGET_LIST[@]}"
    if (( AUTO_STAGE_TASK_LIMIT < 4 )); then
      AUTO_STAGE_TASK_LIMIT=4
    fi
    ;;
esac
if (( AUTO_STAGE_TASK_LIMIT == 0 )); then
  echo "PAPER_TARGETS must contain at least one target for STAGE='${STAGE}'." >&2
  exit 2
fi

CUDA_DEVICE_POOL=()
GPU_SNAPSHOT_INDEXES=()
AUTO_PARALLELISM=0
ACQUIRED_GPU=""
ACQUIRED_LOCK_FD=""
declare -A GPU_TOTAL_MEMORY_BY_INDEX=()
declare -A GPU_FREE_MEMORY_BY_INDEX=()
declare -A GPU_UTILIZATION_BY_INDEX=()
declare -A GPU_LOCK_FD_BY_INDEX=()

# 解析并校验用户指定的 CUDA 设备池。
parse_cuda_device_pool() {
  local raw_devices="$1"
  local cuda_device
  local -A cuda_device_seen=()

  read -r -a CUDA_DEVICE_POOL <<< "${raw_devices}"
  if (( ${#CUDA_DEVICE_POOL[@]} == 0 )); then
    echo "CUDA_DEVICES must contain at least one CUDA index." >&2
    return 2
  fi
  for cuda_device in "${CUDA_DEVICE_POOL[@]}"; do
    if ! [[ "${cuda_device}" =~ ^(0|[1-9][0-9]*)$ ]]; then
      echo "CUDA device must be a canonical non-negative integer, got '${cuda_device}'." >&2
      return 2
    fi
    if [[ -n "${cuda_device_seen[${cuda_device}]:-}" ]]; then
      echo "CUDA_DEVICES contains duplicate GPU index '${cuda_device}'." >&2
      return 2
    fi
    cuda_device_seen["${cuda_device}"]=1
  done
}

if [[ -n "${CUDA_DEVICES}" ]]; then
  parse_cuda_device_pool "${CUDA_DEVICES}"
elif (( CUDA_DEVICES_WAS_EXPLICIT != 0 )); then
  echo "CUDA_DEVICES was explicitly set but contains no CUDA index." >&2
  exit 2
elif [[ "${PARALLEL_JOBS}" != "auto" ]]; then
  CUDA_DEVICES="${CUDA}"
  parse_cuda_device_pool "${CUDA_DEVICES}"
fi

# 去除 nvidia-smi CSV 字段两侧的空白。
trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

# 查询并严格解析当前 GPU 显存与利用率快照。
query_gpu_snapshot() {
  local gpu_snapshot
  local gpu_index
  local total_memory_mb
  local free_memory_mb
  local utilization_percent
  local extra_field

  if ! gpu_snapshot="$("${NVIDIA_SMI_BIN}" \
      --query-gpu=index,memory.total,memory.free,utilization.gpu \
      --format=csv,noheader,nounits)"; then
    echo "Failed to query GPU memory with '${NVIDIA_SMI_BIN}'." >&2
    return 2
  fi
  if [[ -z "${gpu_snapshot//[[:space:]]/}" ]]; then
    echo "nvidia-smi returned an empty GPU snapshot." >&2
    return 2
  fi

  GPU_SNAPSHOT_INDEXES=()
  GPU_TOTAL_MEMORY_BY_INDEX=()
  GPU_FREE_MEMORY_BY_INDEX=()
  GPU_UTILIZATION_BY_INDEX=()
  while IFS=',' read -r gpu_index total_memory_mb free_memory_mb \
      utilization_percent extra_field; do
    gpu_index="$(trim_whitespace "${gpu_index}")"
    total_memory_mb="$(trim_whitespace "${total_memory_mb}")"
    free_memory_mb="$(trim_whitespace "${free_memory_mb}")"
    utilization_percent="$(trim_whitespace "${utilization_percent}")"
    extra_field="$(trim_whitespace "${extra_field:-}")"
    if ! [[ "${gpu_index}" =~ ^(0|[1-9][0-9]*)$ ]] || \
        ! [[ "${total_memory_mb}" =~ ^[1-9][0-9]*$ ]] || \
        ! [[ "${free_memory_mb}" =~ ^(0|[1-9][0-9]*)$ ]] || \
        ! [[ "${utilization_percent}" =~ ^(0|[1-9][0-9]*)$ ]] || \
        [[ -n "${extra_field}" ]]; then
      echo "Invalid nvidia-smi row: '${gpu_index},${total_memory_mb},${free_memory_mb},${utilization_percent}${extra_field:+,${extra_field}}'." >&2
      return 2
    fi
    if (( free_memory_mb > total_memory_mb || utilization_percent > 100 )); then
      echo "Out-of-range nvidia-smi row for GPU ${gpu_index}." >&2
      return 2
    fi
    if [[ -n "${GPU_TOTAL_MEMORY_BY_INDEX[${gpu_index}]:-}" ]]; then
      echo "nvidia-smi returned duplicate GPU index '${gpu_index}'." >&2
      return 2
    fi
    GPU_SNAPSHOT_INDEXES+=("${gpu_index}")
    GPU_TOTAL_MEMORY_BY_INDEX["${gpu_index}"]="${total_memory_mb}"
    GPU_FREE_MEMORY_BY_INDEX["${gpu_index}"]="${free_memory_mb}"
    GPU_UTILIZATION_BY_INDEX["${gpu_index}"]="${utilization_percent}"
  done <<< "${gpu_snapshot}"
}

# 为单张 GPU 获取跨 pipeline 的非阻塞文件锁。
open_gpu_lock() {
  local gpu_index="$1"
  local lock_file="${GPU_LOCK_DIR}/gpu-${gpu_index}.lock"
  local allocated_fd

  if ! exec {allocated_fd}>>"${lock_file}"; then
    echo "Unable to open GPU lock file '${lock_file}'." >&2
    return 2
  fi
  if ! flock -n "${allocated_fd}"; then
    exec {allocated_fd}>&-
    return 1
  fi
  ACQUIRED_LOCK_FD="${allocated_fd}"
}

# 关闭单个 GPU 锁的文件描述符。
close_gpu_lock_fd() {
  local lock_fd
  lock_fd="$1"
  if [[ -n "${lock_fd}" ]]; then
    exec {lock_fd}>&- || true
  fi
}

# 释放一个已分配 GPU 的文件锁。
release_gpu_lock() {
  local gpu_index="$1"
  local lock_fd="$2"

  close_gpu_lock_fd "${lock_fd}"
  unset "GPU_LOCK_FD_BY_INDEX[${gpu_index}]"
}

# 释放当前 pipeline 父进程持有的全部 GPU 文件锁。
release_all_gpu_locks() {
  local gpu_index
  local -a locked_gpu_indexes=("${!GPU_LOCK_FD_BY_INDEX[@]}")

  for gpu_index in "${locked_gpu_indexes[@]}"; do
    release_gpu_lock \
      "${gpu_index}" "${GPU_LOCK_FD_BY_INDEX[${gpu_index}]}"
  done
}

# 校验自动调度环境，并确定候选 GPU 池与并发上限。
configure_auto_parallelism() {
  local gpu_index
  local max_jobs
  local pool_summary
  local eligible_jobs=0

  if ! [[ "${GPU_MIN_FREE_MEMORY_MB}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_MIN_FREE_MEMORY_MB must be a positive integer, got '${GPU_MIN_FREE_MEMORY_MB}'." >&2
    return 2
  fi
  if ! [[ "${GPU_MAX_UTILIZATION_PERCENT}" =~ ^(0|[1-9][0-9]*)$ ]] || \
      (( GPU_MAX_UTILIZATION_PERCENT > 100 )); then
    echo "GPU_MAX_UTILIZATION_PERCENT must be an integer in [0, 100], got '${GPU_MAX_UTILIZATION_PERCENT}'." >&2
    return 2
  fi
  if ! [[ "${GPU_UTILIZATION_WARNING_PERCENT}" =~ ^(0|[1-9][0-9]*)$ ]] || \
      (( GPU_UTILIZATION_WARNING_PERCENT > 100 )); then
    echo "GPU_UTILIZATION_WARNING_PERCENT must be an integer in [0, 100], got '${GPU_UTILIZATION_WARNING_PERCENT}'." >&2
    return 2
  fi
  if ! [[ "${AUTO_PARALLEL_MAX_JOBS}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "AUTO_PARALLEL_MAX_JOBS must be a non-negative integer, got '${AUTO_PARALLEL_MAX_JOBS}'." >&2
    return 2
  fi
  if ! command -v "${NVIDIA_SMI_BIN}" >/dev/null 2>&1; then
    echo "PARALLEL_JOBS=auto requires an executable nvidia-smi command: '${NVIDIA_SMI_BIN}'." >&2
    echo "Install/enable nvidia-smi or set an explicit integer PARALLEL_JOBS." >&2
    return 2
  fi
  if ! command -v flock >/dev/null 2>&1; then
    echo "PARALLEL_JOBS=auto requires 'flock' for cross-pipeline GPU locking." >&2
    return 2
  fi
  if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    echo "PARALLEL_JOBS=auto cannot safely map physical GPU indexes while CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'." >&2
    echo "Unset CUDA_VISIBLE_DEVICES or use an explicit integer PARALLEL_JOBS and logical CUDA_DEVICES." >&2
    return 2
  fi
  if ! mkdir -p "${GPU_LOCK_DIR}"; then
    echo "Unable to create GPU lock directory '${GPU_LOCK_DIR}'." >&2
    return 2
  fi

  query_gpu_snapshot || return 2
  if (( ${#CUDA_DEVICE_POOL[@]} == 0 )); then
    CUDA_DEVICE_POOL=("${GPU_SNAPSHOT_INDEXES[@]}")
    CUDA_DEVICES="${CUDA_DEVICE_POOL[*]}"
  fi
  for gpu_index in "${CUDA_DEVICE_POOL[@]}"; do
    if [[ -z "${GPU_TOTAL_MEMORY_BY_INDEX[${gpu_index}]:-}" ]]; then
      echo "CUDA_DEVICES requests GPU ${gpu_index}, but nvidia-smi did not report it." >&2
      return 2
    fi
    if (( GPU_TOTAL_MEMORY_BY_INDEX[${gpu_index}] >= GPU_MIN_FREE_MEMORY_MB && \
          GPU_FREE_MEMORY_BY_INDEX[${gpu_index}] >= GPU_MIN_FREE_MEMORY_MB && \
          GPU_UTILIZATION_BY_INDEX[${gpu_index}] <= GPU_MAX_UTILIZATION_PERCENT )); then
      eligible_jobs=$((eligible_jobs + 1))
    fi
  done
  if (( eligible_jobs == 0 )); then
    echo "No GPU in CUDA_DEVICES='${CUDA_DEVICES}' initially satisfies free memory >= ${GPU_MIN_FREE_MEMORY_MB} MiB and utilization <= ${GPU_MAX_UTILIZATION_PERCENT}%." >&2
    return 2
  fi

  max_jobs="${AUTO_PARALLEL_MAX_JOBS}"
  if (( max_jobs == 0 || max_jobs > ${#CUDA_DEVICE_POOL[@]} )); then
    max_jobs="${#CUDA_DEVICE_POOL[@]}"
  fi
  if (( max_jobs > eligible_jobs )); then
    max_jobs="${eligible_jobs}"
  fi
  if (( max_jobs > AUTO_STAGE_TASK_LIMIT )); then
    max_jobs="${AUTO_STAGE_TASK_LIMIT}"
  fi

  PARALLEL_JOBS="${max_jobs}"
  AUTO_PARALLELISM=1
  pool_summary="$(IFS=,; printf '%s' "${CUDA_DEVICE_POOL[*]}")"
  printf 'AUTO_GPU: candidate CUDA devices [%s], initially_eligible=%s, PARALLEL_JOBS=%s, minimum_free_memory=%s MiB.\n' \
    "${pool_summary}" "${eligible_jobs}" "${PARALLEL_JOBS}" \
    "${GPU_MIN_FREE_MEMORY_MB}"
}

# 每次启动任务前重新探测并锁定一张当前可用的 GPU。
acquire_auto_gpu() {
  local gpu_index
  local total_memory_mb
  local free_memory_mb
  local utilization_percent
  local lock_status
  local lock_fd
  local snapshot_entry
  local -a candidate_rows=()
  local -a snapshot_summary=()

  ACQUIRED_GPU=""
  ACQUIRED_LOCK_FD=""
  query_gpu_snapshot || return 2
  for gpu_index in "${CUDA_DEVICE_POOL[@]}"; do
    if [[ -z "${GPU_TOTAL_MEMORY_BY_INDEX[${gpu_index}]:-}" ]]; then
      echo "CUDA device ${gpu_index} disappeared from the nvidia-smi snapshot." >&2
      return 2
    fi
    total_memory_mb="${GPU_TOTAL_MEMORY_BY_INDEX[${gpu_index}]}"
    free_memory_mb="${GPU_FREE_MEMORY_BY_INDEX[${gpu_index}]}"
    utilization_percent="${GPU_UTILIZATION_BY_INDEX[${gpu_index}]}"
    snapshot_summary+=("GPU${gpu_index}:free=${free_memory_mb}MiB,total=${total_memory_mb}MiB,util=${utilization_percent}%")
    if [[ -z "${GPU_LOCK_FD_BY_INDEX[${gpu_index}]:-}" ]] && \
        (( total_memory_mb >= GPU_MIN_FREE_MEMORY_MB && \
           free_memory_mb >= GPU_MIN_FREE_MEMORY_MB && \
           utilization_percent <= GPU_MAX_UTILIZATION_PERCENT )); then
      candidate_rows+=("${free_memory_mb} ${utilization_percent} ${gpu_index}")
    fi
  done

  while read -r free_memory_mb utilization_percent gpu_index; do
    [[ -n "${gpu_index:-}" ]] || continue
    if open_gpu_lock "${gpu_index}"; then
      lock_fd="${ACQUIRED_LOCK_FD}"
    else
      lock_status=$?
      if (( lock_status == 2 )); then
        return 2
      fi
      printf 'AUTO_GPU_SKIP: GPU%s is locked by another DARLR pipeline.\n' \
        "${gpu_index}" >&2
      continue
    fi

    if ! query_gpu_snapshot; then
      close_gpu_lock_fd "${lock_fd}"
      return 2
    fi
    if [[ -n "${GPU_TOTAL_MEMORY_BY_INDEX[${gpu_index}]:-}" ]]; then
      total_memory_mb="${GPU_TOTAL_MEMORY_BY_INDEX[${gpu_index}]}"
      free_memory_mb="${GPU_FREE_MEMORY_BY_INDEX[${gpu_index}]}"
      utilization_percent="${GPU_UTILIZATION_BY_INDEX[${gpu_index}]}"
      if (( total_memory_mb >= GPU_MIN_FREE_MEMORY_MB && \
            free_memory_mb >= GPU_MIN_FREE_MEMORY_MB && \
            utilization_percent <= GPU_MAX_UTILIZATION_PERCENT )); then
        GPU_LOCK_FD_BY_INDEX["${gpu_index}"]="${lock_fd}"
        ACQUIRED_GPU="${gpu_index}"
        ACQUIRED_LOCK_FD="${lock_fd}"
        printf 'AUTO_GPU_ASSIGN: GPU%s free=%s MiB, utilization=%s%%.\n' \
          "${gpu_index}" "${free_memory_mb}" "${utilization_percent}"
        if (( utilization_percent >= GPU_UTILIZATION_WARNING_PERCENT )); then
          printf 'AUTO_GPU_WARN: GPU%s utilization is %s%%; memory is sufficient but compute contention may slow training.\n' \
            "${gpu_index}" "${utilization_percent}" >&2
        fi
        return 0
      fi
    fi
    close_gpu_lock_fd "${lock_fd}"
    printf 'AUTO_GPU_RETRY: GPU%s changed after locking and is no longer eligible.\n' \
      "${gpu_index}" >&2
  done < <(printf '%s\n' "${candidate_rows[@]}" | \
    sort -k1,1nr -k2,2n -k3,3n)

  echo "No unlocked GPU in CUDA_DEVICES='${CUDA_DEVICES}' satisfies free memory >= ${GPU_MIN_FREE_MEMORY_MB} MiB and utilization <= ${GPU_MAX_UTILIZATION_PERCENT}%." >&2
  for snapshot_entry in "${snapshot_summary[@]}"; do
    printf 'AUTO_GPU_SNAPSHOT: %s\n' "${snapshot_entry}" >&2
  done
  return 1
}

if [[ "${PARALLEL_JOBS}" == "auto" ]]; then
  configure_auto_parallelism
elif ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_JOBS must be 'auto' or a positive integer, got '${PARALLEL_JOBS}'." >&2
  exit 2
fi

if (( PARALLEL_JOBS > ${#CUDA_DEVICE_POOL[@]} )); then
  echo "PARALLEL_JOBS=${PARALLEL_JOBS} exceeds CUDA device count=${#CUDA_DEVICE_POOL[@]}." >&2
  echo "Provide one distinct GPU per concurrent training process." >&2
  exit 2
fi

ACTIVE_PIDS=()
ACTIVE_LABELS=()
ACTIVE_CUDA_DEVICES=()
ACTIVE_LOCK_FDS=()
NEXT_CUDA_INDEX=0
JOB_TERMINATION_GRACE_SECONDS=1
# TERM 后给训练进程组的优雅退出时间。

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
    if [[ -n "${ACTIVE_LOCK_FDS[${job_index}]}" ]]; then
      release_gpu_lock \
        "${ACTIVE_CUDA_DEVICES[${job_index}]}" \
        "${ACTIVE_LOCK_FDS[${job_index}]}"
    fi
  done
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  ACTIVE_CUDA_DEVICES=()
  ACTIVE_LOCK_FDS=()
  NEXT_CUDA_INDEX=0
  if (( failed_jobs != 0 )); then
    return 1
  fi
}

# 按 GPU 池启动任务；达到并发上限后先等待当前整批任务。
launch_job() {
  local job_label="$1"
  local job_cuda
  local job_lock_fd=""
  local acquisition_status
  local inherited_gpu
  local inherited_lock_fd
  local job_pid
  shift

  while true; do
    if (( ${#ACTIVE_PIDS[@]} >= PARALLEL_JOBS )); then
      wait_for_jobs
    fi
    if (( AUTO_PARALLELISM == 0 )); then
      job_cuda="${CUDA_DEVICE_POOL[${NEXT_CUDA_INDEX}]}"
      NEXT_CUDA_INDEX=$((NEXT_CUDA_INDEX + 1))
      break
    fi
    if acquire_auto_gpu; then
      job_cuda="${ACQUIRED_GPU}"
      job_lock_fd="${ACQUIRED_LOCK_FD}"
      break
    else
      acquisition_status=$?
    fi
    if (( acquisition_status == 1 && ${#ACTIVE_PIDS[@]} > 0 )); then
      printf 'AUTO_GPU_WAIT: waiting for the current job batch before retrying %s.\n' \
        "${job_label}"
      wait_for_jobs
      continue
    fi
    return "${acquisition_status}"
  done

  printf 'LAUNCH: %s on CUDA=%s\n' "${job_label}" "${job_cuda}"
  # 短暂启用 job control，使每个后台任务获得独立进程组。
  set -m
  (
    if (( AUTO_PARALLELISM != 0 )); then
      for inherited_gpu in "${!GPU_LOCK_FD_BY_INDEX[@]}"; do
        inherited_lock_fd="${GPU_LOCK_FD_BY_INDEX[${inherited_gpu}]}"
        if [[ "${inherited_lock_fd}" != "${job_lock_fd}" ]]; then
          close_gpu_lock_fd "${inherited_lock_fd}"
        fi
      done
    fi
    export CUDA="${job_cuda}"
    "$@"
  ) &
  job_pid="$!"
  ACTIVE_PIDS+=("${job_pid}")
  ACTIVE_LABELS+=("${job_label}")
  ACTIVE_CUDA_DEVICES+=("${job_cuda}")
  ACTIVE_LOCK_FDS+=("${job_lock_fd}")
  set +m
}

# 停止并回收所有活动 launcher 的完整进程组。
terminate_active_jobs() {
  local active_pid

  for active_pid in "${ACTIVE_PIDS[@]}"; do
    kill -TERM -- "-${active_pid}" 2>/dev/null || true
  done
  if (( ${#ACTIVE_PIDS[@]} > 0 )); then
    sleep "${JOB_TERMINATION_GRACE_SECONDS}"
  fi
  for active_pid in "${ACTIVE_PIDS[@]}"; do
    if kill -0 -- "-${active_pid}" 2>/dev/null; then
      kill -KILL -- "-${active_pid}" 2>/dev/null || true
    fi
  done
  for active_pid in "${ACTIVE_PIDS[@]}"; do
    wait "${active_pid}" 2>/dev/null || true
  done
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  ACTIVE_CUDA_DEVICES=()
  ACTIVE_LOCK_FDS=()
}

# 退出时终止子任务并释放自动调度持有的 GPU 锁。
cleanup_pipeline() {
  local exit_status=$?
  trap - EXIT INT TERM
  terminate_active_jobs
  release_all_gpu_locks
  return "${exit_status}"
}

# 将中断信号转换为明确的非零退出码，由 EXIT trap 统一清理。
handle_interrupt() {
  exit 130
}

# 将终止信号转换为明确的非零退出码，由 EXIT trap 统一清理。
handle_termination() {
  exit 143
}

trap cleanup_pipeline EXIT
trap handle_interrupt INT
trap handle_termination TERM

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
  for target in "${PAPER_TARGET_LIST[@]}"; do
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
