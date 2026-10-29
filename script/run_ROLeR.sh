#!/usr/bin/env bash

set -euo pipefail

DEFAULT_ENV_NAME="KuaiEnv-v0"
DEFAULT_SEEDS="0"
DEFAULT_GPUS="5"
DEFAULT_MAX_TURN=30
DEFAULT_EPOCHS=100
DEFAULT_STEPS_PER_EPOCH=100000


show_usage() {
  # 打印单种子兼容调用和多种子、多 GPU 调度方式。
  cat <<'EOF'
用法：
  bash script/run_ROLeR.sh [ENV_NAME] [SEEDS] [GPUS] [MAX_TURN]

位置参数：
  ENV_NAME   环境名：KuaiEnv-v0、KuaiRand-v0、CoatEnv-v0、YahooEnv-v0
  SEEDS      随机种子列表，支持逗号或空格分隔，默认 0
  GPUS       物理 GPU 列表，支持逗号或空格分隔，默认 5
  MAX_TURN   单条轨迹最大交互轮数，默认 30

环境变量：
  SEEDS                 覆盖第 2 个位置参数
  GPUS                  覆盖第 3 个位置参数
  MAX_TURN              覆盖第 4 个位置参数
  EPOCHS                训练 epoch 数，默认 100
  STEPS_PER_EPOCH       每个 epoch 的环境步数，默认 100000
  PYTHON_BIN            Python 可执行文件，默认 python
  PREPARE_CACHE         多种子运行前是否单独生成共享缓存，默认 1
  CPU_THREADS           可选；限制 OMP/MKL/OpenBLAS/NumExpr 线程数

示例：
  # 保持旧用法：seed 0 在物理 GPU 5 上运行，max_turn=30
  bash script/run_ROLeR.sh KuaiEnv-v0 0 5 30

  # seeds 0-4 在 GPU 4-7 上分批运行，每张 GPU 同时最多一个进程
  bash script/run_ROLeR.sh KuaiEnv-v0 "0,1,2,3,4" "4,5,6,7" 30

  # 使用环境变量设置多种子、多 GPU 和 max_turn
  SEEDS="0 1 2 3" GPUS="4 5 6 7" MAX_TURN=100 \
    bash script/run_ROLeR.sh KuaiEnv-v0
EOF
}


parse_list() {
  # 将逗号或空格分隔的字符串解析为 Bash 数组，并通过全局 PARSED_LIST 返回。
  local raw_value="$1"
  local normalized_value="${raw_value//,/ }"
  read -r -a PARSED_LIST <<< "${normalized_value}"
}


validate_nonnegative_integer() {
  # 校验 seed 或 GPU 编号是否为非负整数。
  local value="$1"
  local label="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${label} 必须是非负整数，收到：${value}" >&2
    exit 2
  fi
}


validate_unique_list() {
  # 禁止重复 seed 或 GPU，避免覆盖日志或在同一 GPU 同时启动两个任务。
  local label="$1"
  shift
  local -A seen_values=()
  local value
  for value in "$@"; do
    if [[ -n "${seen_values[${value}]+exists}" ]]; then
      echo "${label} 列表包含重复值：${value}" >&2
      exit 2
    fi
    seen_values["${value}"]=1
  done
}


run_with_resources() {
  # 为子进程设置物理 GPU 隔离和可选 CPU 线程限制后执行命令。
  local physical_gpu="$1"
  shift
  local -a resource_env=("CUDA_VISIBLE_DEVICES=${physical_gpu}")

  if [[ -n "${CPU_THREADS}" ]]; then
    resource_env+=(
      "OMP_NUM_THREADS=${CPU_THREADS}"
      "MKL_NUM_THREADS=${CPU_THREADS}"
      "OPENBLAS_NUM_THREADS=${CPU_THREADS}"
      "NUMEXPR_NUM_THREADS=${CPU_THREADS}"
    )
  fi

  env "${resource_env[@]}" "$@"
}


run_experiment() {
  # 在指定物理 GPU 上运行一个 seed；进程内部 GPU 编号固定映射为 cuda:0。
  local seed="$1"
  local physical_gpu="$2"
  local -a command=(
    "${PYTHON_BIN}"
    "${COMMON_ARGS[@]}"
    --seed "${seed}"
    --cuda 0
    --message "ROLeR-${ENV_NAME}-seed${seed}"
  )

  echo "启动训练：env=${ENV_NAME}, seed=${seed}, physical_gpu=${physical_gpu}, max_turn=${MAX_TURN_VALUE}"
  run_with_resources "${physical_gpu}" "${command[@]}"
}


prepare_shared_cache() {
  # 多种子并行前只生成一次奖励缓存，防止多个进程竞争同一缓存文件。
  local first_seed="${SEED_LIST[0]}"
  local -a command=(
    "${PYTHON_BIN}"
    "${COMMON_ARGS[@]}"
    --seed "${first_seed}"
    --cuda 0
    --message "ROLeR-${ENV_NAME}-prepare-cache"
    --prepare_only
    --cpu
  )

  echo "准备共享缓存：env=${ENV_NAME}, max_turn=${MAX_TURN_VALUE}"
  run_with_resources "" "${command[@]}"
}


terminate_active_jobs() {
  # 收到中断信号时终止当前批次子进程，避免训练进程脱离启动脚本继续运行。
  local process_id
  for process_id in "${ACTIVE_PIDS[@]:-}"; do
    kill "${process_id}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit 130
}


if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_usage
  exit 0
fi

ENV_NAME="${1:-${DEFAULT_ENV_NAME}}"
SEED_SPEC="${SEEDS:-${2:-${DEFAULT_SEEDS}}}"
GPU_SPEC="${GPUS:-${3:-${DEFAULT_GPUS}}}"
MAX_TURN_VALUE="${MAX_TURN:-${4:-${DEFAULT_MAX_TURN}}}"
EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-${DEFAULT_STEPS_PER_EPOCH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARE_CACHE="${PREPARE_CACHE:-1}"
CPU_THREADS="${CPU_THREADS:-}"

parse_list "${SEED_SPEC}"
SEED_LIST=("${PARSED_LIST[@]}")
parse_list "${GPU_SPEC}"
GPU_LIST=("${PARSED_LIST[@]}")

if ((${#SEED_LIST[@]} == 0)); then
  echo "SEEDS 不允许为空。" >&2
  exit 2
fi
if ((${#GPU_LIST[@]} == 0)); then
  echo "GPUS 不允许为空。" >&2
  exit 2
fi

for seed in "${SEED_LIST[@]}"; do
  validate_nonnegative_integer "${seed}" "seed"
done
for gpu in "${GPU_LIST[@]}"; do
  validate_nonnegative_integer "${gpu}" "GPU 编号"
done
validate_unique_list "SEEDS" "${SEED_LIST[@]}"
validate_unique_list "GPUS" "${GPU_LIST[@]}"

if [[ ! "${MAX_TURN_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TURN 必须是正整数，收到：${MAX_TURN_VALUE}" >&2
  exit 2
fi
if [[ ! "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS 必须是正整数，收到：${EPOCHS}" >&2
  exit 2
fi
if [[ ! "${STEPS_PER_EPOCH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEPS_PER_EPOCH 必须是正整数，收到：${STEPS_PER_EPOCH}" >&2
  exit 2
fi
if [[ "${PREPARE_CACHE}" != "0" && "${PREPARE_CACHE}" != "1" ]]; then
  echo "PREPARE_CACHE 只能为 0 或 1，收到：${PREPARE_CACHE}" >&2
  exit 2
fi
if [[ -n "${CPU_THREADS}" && ! "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPU_THREADS 必须为空或正整数，收到：${CPU_THREADS}" >&2
  exit 2
fi

case "${ENV_NAME}" in
  KuaiEnv-v0)
    TRACKER="att"
    WINDOW_SIZE=3
    ;;
  KuaiRand-v0)
    TRACKER="avg"
    WINDOW_SIZE=3
    ;;
  CoatEnv-v0)
    TRACKER="att"
    WINDOW_SIZE=3
    ;;
  YahooEnv-v0)
    TRACKER="att"
    WINDOW_SIZE=5
    ;;
  *)
    echo "不支持的 ROLeR 环境：${ENV_NAME}" >&2
    exit 2
    ;;
esac

COMMON_ARGS=(
  examples/advance/run_ROLeR.py
  --env "${ENV_NAME}"
  --epoch "${EPOCHS}"
  --step-per-epoch "${STEPS_PER_EPOCH}"
  --training-num 100
  --test-num 100
  --which_tracker "${TRACKER}"
  --window_size "${WINDOW_SIZE}"
  --max_turn "${MAX_TURN_VALUE}"
  --num_leave_compute 1
  --leave_threshold 0
  --reward_handle cat
  --read_message pointneg
  --reward_formula official-minus
  --is_scratch
)

echo "ROLeR 调度配置："
echo "  env=${ENV_NAME}"
echo "  seeds=${SEED_LIST[*]}"
echo "  physical_gpus=${GPU_LIST[*]}"
echo "  max_turn=${MAX_TURN_VALUE}"
echo "  epochs=${EPOCHS}, steps_per_epoch=${STEPS_PER_EPOCH}"
if [[ -n "${CPU_THREADS}" ]]; then
  echo "  cpu_threads_per_job=${CPU_THREADS}"
fi

if ((${#SEED_LIST[@]} == 1)); then
  run_experiment "${SEED_LIST[0]}" "${GPU_LIST[0]}"
  exit 0
fi

if [[ "${PREPARE_CACHE}" == "1" ]]; then
  prepare_shared_cache
fi

LAUNCH_TIME="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_DIR="saved_models/${ENV_NAME}/ROLeR/launcher_logs/${LAUNCH_TIME}"
mkdir -p "${LAUNCH_LOG_DIR}"

ACTIVE_PIDS=()
trap terminate_active_jobs INT TERM

# 按 GPU 数量分批调度，保证同一批次中每张物理 GPU 最多运行一个 seed。
for ((batch_start = 0; batch_start < ${#SEED_LIST[@]}; batch_start += ${#GPU_LIST[@]})); do
  ACTIVE_PIDS=()
  ACTIVE_SEEDS=()

  for ((slot = 0; slot < ${#GPU_LIST[@]}; slot++)); do
    seed_index=$((batch_start + slot))
    if ((seed_index >= ${#SEED_LIST[@]})); then
      break
    fi

    seed="${SEED_LIST[${seed_index}]}"
    gpu="${GPU_LIST[${slot}]}"
    launcher_log="${LAUNCH_LOG_DIR}/seed${seed}_gpu${gpu}.log"

    run_experiment "${seed}" "${gpu}" >"${launcher_log}" 2>&1 &
    ACTIVE_PIDS+=("$!")
    ACTIVE_SEEDS+=("${seed}")
    echo "已提交：seed=${seed}, physical_gpu=${gpu}, pid=$!, launcher_log=${launcher_log}"
  done

  batch_failed=0
  for job_index in "${!ACTIVE_PIDS[@]}"; do
    process_id="${ACTIVE_PIDS[${job_index}]}"
    seed="${ACTIVE_SEEDS[${job_index}]}"
    if wait "${process_id}"; then
      echo "训练完成：seed=${seed}, pid=${process_id}"
    else
      echo "训练失败：seed=${seed}, pid=${process_id}" >&2
      batch_failed=1
    fi
  done

  if ((batch_failed != 0)); then
    exit 1
  fi
done

echo "全部 ROLeR 训练任务完成。"
