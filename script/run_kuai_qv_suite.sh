#!/usr/bin/env bash
set -euo pipefail

# 统一训练五个随机种子的三类 Q/V：MAC MLP Q/V、ACSAC 因果 Transformer Q
# 和 R/T/D/C 多目标前缀 Transformer Q/V。ordinary 与 acsac_transformer
# 共享当前 Actor/Dynamics 和 MAC 模型化训练流程。

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# 命令行参数优先；未传命令行时允许用同名环境变量配置批量任务。

ENV_NAME="${ENV_NAME:-KuaiEnv-v0}"
# ordinary：固定当前 Actor/Dynamics，使用 MAC 离线任意起点、MLP Q/V 与 chunk TD(lambda)。
# acsac_transformer：同 ordinary 流程，但用因果前缀 Q 联合选择候选与 H，且无独立 V。
# multi_prefix_transformer：R/T/D/C 四头前缀 Transformer Q/V，联合选择候选与 H。
QV_TYPES="${QV_TYPES:-acsac_transformer}"
SEEDS="${SEEDS:-2023 2024 2025 2026 2027}"
GPU_IDS="${GPU_IDS:-1 2 3 4 5}"
CHUNK_SIZE="${CHUNK_SIZE:-7}"
#pretrain：只预训练共享 BC Actor，并保存各 seed 的 BC checkpoint。
#qv：使用已有 BC checkpoint 训练指定的 Q/V 分支。
  #- ordinary 每步从离线任意起点重生成 fixed-depth imagined rollout，batch 起点接受 MAC Q/V 监督。
  #- ordinary 的 Actor/Dynamics 从共享 checkpoint 恢复并冻结，R 与原 MAC/DORL 保持一致。
  #- 两个 Transformer 仍使用完整 episode replay 和 ACSAC 全前缀 Q。
  #- 加 --no-eval-after-train 可关闭训练后的独立评估。
#eval：不训练，单独加载已有 checkpoint，评估 --qv-type 指定的分支。三个分支都支持。
#all：依次执行 pretrain 和 qv，并保留 qv 阶段默认的训练后评估行为。
STAGE="${STAGE:-all}"
PARALLEL_SEEDS="${PARALLEL_SEEDS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
RAVE_STEPS="${RAVE_STEPS:-100000}"
RAVE_EPOCHS="${RAVE_EPOCHS:-100}"
RAVE_EVAL_EVERY_EPOCHS="${RAVE_EVAL_EVERY_EPOCHS:-1}"
QV_BATCH_SIZE="${QV_BATCH_SIZE:-256}"
RAVE_ROLLOUT_CANDIDATES="${RAVE_ROLLOUT_CANDIDATES:-8}"
RAVE_EPISODES_PER_COLLECT="${RAVE_EPISODES_PER_COLLECT:-32}"
ORDINARY_ROLLOUT_DEPTH="${ORDINARY_ROLLOUT_DEPTH:-3}"
LAMBDA_CHUNK="${LAMBDA_CHUNK:-1.0}"
ALL_PREFIX_AUX_WEIGHT="${ALL_PREFIX_AUX_WEIGHT:-1.0}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-100000}"
DYNAMICS_PRETRAIN_STEPS="${DYNAMICS_PRETRAIN_STEPS:-10000}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
EVAL_SWANLAB_PROJECT="${EVAL_SWANLAB_PROJECT:-RAVE-H-Eval}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_EPISODES="${EVAL_EPISODES:-${RAVE_TRAIN_EVAL_EPISODES:-100}}"
RAVE_TRAIN_EVAL_EPISODES="${EVAL_EPISODES}"
TEST_NUM="${TEST_NUM:-10}"
NUM_SAMPLES_TEST="${NUM_SAMPLES_TEST:-32}"
BUFFER_SIZE="${BUFFER_SIZE:-0}"
RUN_EVAL_AFTER_TRAIN="${RUN_EVAL_AFTER_TRAIN:-1}"
ACTOR_TYPE="${ACTOR_TYPE:-categorical_flow}"
# 参数解析完成后再根据最终 env/K 计算默认根目录。
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
SHARED_PRETRAIN_ROOT="${SHARED_PRETRAIN_ROOT:-}"
NUM_LEAVE_COMPUTE="${NUM_LEAVE_COMPUTE:-1}"
LEAVE_THRESHOLD="${LEAVE_THRESHOLD:-0}"
MAX_TURN="${MAX_TURN:-100}"
LEAVE_POLICY="${LEAVE_POLICY:-terminate}"
ETA_T="${ETA_T:-0.2}"
ETA_D="${ETA_D:-0.2}"
ETA_C="${ETA_C:-0.05}"
GAMMA="${GAMMA:-0.9}"

usage() {
  cat <<'EOF'
用法：bash script/run_kuai_qv_suite.sh [参数]

参数：
  --env NAME              KuaiEnv-v0 或 KuaiRand-v0
  --qv-type TYPES         ordinary、acsac_transformer、multi_prefix_transformer、all；可传逗号分隔列表
  --seeds "S1 ... S5"     必须恰好提供五个随机种子
  --gpus "G0 ..."         外部 GPU 编号；五个种子按顺序循环分配
  --chunk-size K          actor chunk 长度
  --num-leave-compute N  退出规则观察的最近 N 个 item；两个环境默认均为 1
  --leave-threshold X    类别历史计数严格大于 X 时，匹配候选会触发退出
  --max-turn N           会话最大展示步数，默认 100
  --leave-policy MODE    ordinary 与 Transformer 均仅支持 terminate；保留该参数用于显式校验
  --eta-t X              四头模型的展示长度权重；默认 0.2
  --eta-d X              四头模型的 MCD 改善权重；默认 0.2
  --eta-c X              四头模型的规划成本权重；默认 0.05
  --gamma X              MAC/ACSAC 每环境步折扣率；默认 0.9
  --stage STAGE           all、pretrain、qv 或 eval
  --output-root PATH      默认 saved_models/<env>/QV_suite/K<K>
  --shared-pretrain-root PATH 复用其他实验根目录下的 shared_pretrain
  --rave-steps N          三种 Q/V 的统一更新步数，默认 100000
  --rave-epochs N         三种 Q/V 的统一 epoch 数，默认 100
  --rave-eval-every N     每 N 个 epoch 记录一次 RAVE-H-Eval，默认 1
  --batch-size N          三个 Q/V 分支统一训练 batch size；MAC 系分支是离线起点数，默认 256
  --train-candidates N    三个 Q/V 分支每状态训练候选数，默认 8
  --episodes-per-collect N 仅用于 Transformer 每次收集的完整 episode 数，默认 32
  --ordinary-rollout-depth N ordinary 每轮 imagined rollout 的 chunk 深度，默认 3
  --lambda-chunk X        ordinary 的 MAC chunk-level TD(lambda) 系数，默认 1.0
  --all-prefix-aux-weight X ACSAC 全部合法 H 的一步 target-Q 辅助损失权重，默认 1.0
  --train-eval-episodes N 与 --eval-episodes 等价；三个 Q/V 分支统一使用该轨迹数
  --pretrain-steps N       Flow/CE BC 更新步数
  --dynamics-steps N      chunk dynamics 更新步数
  --mac-epochs N          --rave-epochs 的兼容别名
  --eval-split SPLIT      独立评估输出标签；评估用户来自完整环境
  --eval-episodes N       每个 checkpoint 的独立评估轨迹数，默认 100
  --no-eval-after-train   Q/V 训练结束后不自动启动独立评估
  --serial                五个种子串行执行
  --no-skip-existing      不跳过已有 latest.pt
  --dry-run               只打印命令，不启动训练
  -h, --help              显示帮助

示例：
  bash script/run_kuai_qv_suite.sh --env KuaiRand-v0 \
    --qv-type all --seeds "2023 2024 2025 2026 2027" --gpus "0 1 2 3 4"
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --qv-type|--qv-types) QV_TYPES="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --gpus) GPU_IDS="$2"; shift 2 ;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
    --num-leave-compute) NUM_LEAVE_COMPUTE="$2"; shift 2 ;;
    --leave-threshold) LEAVE_THRESHOLD="$2"; shift 2 ;;
    --max-turn) MAX_TURN="$2"; shift 2 ;;
    --leave-policy) LEAVE_POLICY="$2"; shift 2 ;;
    --eta-t) ETA_T="$2"; shift 2 ;;
    --eta-d) ETA_D="$2"; shift 2 ;;
    --eta-c) ETA_C="$2"; shift 2 ;;
    --gamma) GAMMA="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --shared-pretrain-root) SHARED_PRETRAIN_ROOT="$2"; shift 2 ;;
    --rave-steps) RAVE_STEPS="$2"; shift 2 ;;
    --rave-epochs) RAVE_EPOCHS="$2"; shift 2 ;;
    --rave-eval-every) RAVE_EVAL_EVERY_EPOCHS="$2"; shift 2 ;;
    --batch-size) QV_BATCH_SIZE="$2"; shift 2 ;;
    --train-candidates) RAVE_ROLLOUT_CANDIDATES="$2"; shift 2 ;;
    --episodes-per-collect) RAVE_EPISODES_PER_COLLECT="$2"; shift 2 ;;
    --ordinary-rollout-depth) ORDINARY_ROLLOUT_DEPTH="$2"; shift 2 ;;
    --lambda-chunk) LAMBDA_CHUNK="$2"; shift 2 ;;
    --all-prefix-aux-weight) ALL_PREFIX_AUX_WEIGHT="$2"; shift 2 ;;
    --train-eval-episodes) EVAL_EPISODES="$2"; RAVE_TRAIN_EVAL_EPISODES="$2"; shift 2 ;;
    --pretrain-steps) PRETRAIN_STEPS="$2"; shift 2 ;;
    --dynamics-steps) DYNAMICS_PRETRAIN_STEPS="$2"; shift 2 ;;
    --mac-epochs) RAVE_EPOCHS="$2"; shift 2 ;;
    --mac-steps)
      printf -- '--mac-steps 已停用；请用 --rave-steps 设置三个分支统一的总更新数。\n' >&2
      exit 2
      ;;
    --eval-split) EVAL_SPLIT="$2"; shift 2 ;;
    --eval-episodes) EVAL_EPISODES="$2"; RAVE_TRAIN_EVAL_EPISODES="$2"; shift 2 ;;
    --no-eval-after-train) RUN_EVAL_AFTER_TRAIN=0; shift ;;
    --serial) PARALLEL_SEEDS=0; shift ;;
    --no-skip-existing) SKIP_EXISTING=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf '未知参数：%s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

# 长任务开始前完整解析所有入口，避免训练结束后才暴露 Shell 引号等语法错误。
for script_path in "$0" script/run_dorl_mac_kuai_train.sh script/run_rave_kuai.sh; do
  if ! bash -n "${script_path}"; then
    printf '脚本语法预检失败：%s\n' "${script_path}" >&2
    exit 2
  fi
done

case "${ENV_NAME}" in
  KuaiEnv-v0) DATASET_TAG="kuairec"; DEFAULT_LEAVE_WINDOW=1 ;;
  KuaiRand-v0) DATASET_TAG="kuairand"; DEFAULT_LEAVE_WINDOW=1 ;;
  *) printf '不支持的环境：%s\n' "${ENV_NAME}" >&2; exit 2 ;;
esac
NUM_LEAVE_COMPUTE="${NUM_LEAVE_COMPUTE:-${DEFAULT_LEAVE_WINDOW}}"
case "${ACTOR_TYPE}" in
  categorical_flow) ACTOR_TAG="categorical-flow"; BC_SUBDIR="categorical_flow_bc" ;;
  categorical) ACTOR_TAG="categorical-ce"; BC_SUBDIR="categorical_bc" ;;
  *) printf 'ACTOR_TYPE 必须为 categorical_flow 或 categorical。\n' >&2; exit 2 ;;
esac
case "${STAGE}" in
  all|pretrain|qv|eval) ;;
  *) printf -- '--stage 必须为 all、pretrain、qv 或 eval。\n' >&2; exit 2 ;;
esac
case "${EVAL_SPLIT}" in
  validation|test) ;;
  *) printf -- '--eval-split 必须为 validation 或 test。\n' >&2; exit 2 ;;
esac
case "${LEAVE_POLICY}" in
  terminate|penalty) ;;
  *) printf -- '--leave-policy 必须为 terminate 或 penalty。\n' >&2; exit 2 ;;
esac

read -r -a SEED_VALUES <<< "${SEEDS}"
read -r -a GPU_VALUES <<< "${GPU_IDS}"
QV_TYPES="${QV_TYPES//,/ }"
read -r -a QV_TYPE_VALUES <<< "${QV_TYPES}"
if [[ "${#SEED_VALUES[@]}" -ne 5 ]]; then
  printf -- '--seeds 必须恰好包含五个随机种子；当前为 %s 个：%s\n' \
    "${#SEED_VALUES[@]}" "${SEEDS}" >&2
  exit 2
fi
if [[ "${#GPU_VALUES[@]}" -lt 1 ]]; then
  printf -- '--gpus 至少需要一个 GPU 编号。\n' >&2
  exit 2
fi
if [[ "${#QV_TYPE_VALUES[@]}" -eq 1 && "${QV_TYPE_VALUES[0]}" == "all" ]]; then
  QV_TYPE_VALUES=(ordinary acsac_transformer multi_prefix_transformer)
fi
ORDINARY_ENABLED=0
for qv_type in "${QV_TYPE_VALUES[@]}"; do
  case "${qv_type}" in
    ordinary|acsac_transformer) ORDINARY_ENABLED=1 ;;
    multi_prefix_transformer) ;;
    *) printf '不支持的 Q/V 类型：%s\n' "${qv_type}" >&2; exit 2 ;;
  esac
done
for numeric_value in "${CHUNK_SIZE}" "${RAVE_STEPS}" "${RAVE_EPOCHS}" \
                     "${RAVE_EVAL_EVERY_EPOCHS}" "${RAVE_TRAIN_EVAL_EPISODES}" \
                     "${PRETRAIN_STEPS}" \
                     "${DYNAMICS_PRETRAIN_STEPS}" \
                     "${QV_BATCH_SIZE}" "${RAVE_ROLLOUT_CANDIDATES}" \
                     "${RAVE_EPISODES_PER_COLLECT}" \
                     "${ORDINARY_ROLLOUT_DEPTH}" \
                     "${NUM_LEAVE_COMPUTE}" "${MAX_TURN}" "${EVAL_EPISODES}" \
                     "${TEST_NUM}" "${NUM_SAMPLES_TEST}"; do
  if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    printf '训练计数和 chunk-size 必须为正整数：%s\n' "${numeric_value}" >&2
    exit 2
  fi
done
if [[ ! "${LEAVE_THRESHOLD}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  printf -- '--leave-threshold 必须为非负数：%s\n' "${LEAVE_THRESHOLD}" >&2
  exit 2
fi
for eta_name in ETA_T ETA_D ETA_C; do
  eta_value="${!eta_name}"
  if [[ -n "${eta_value}" && ! "${eta_value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf '%s 必须为非负数：%s\n' "${eta_name}" "${eta_value}" >&2
    exit 2
  fi
done
if ! awk -v value="${GAMMA}" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0 && value <= 1) }'; then
  printf -- '--gamma 必须位于 [0,1]：%s\n' "${GAMMA}" >&2
  exit 2
fi
if ! awk -v value="${LAMBDA_CHUNK}" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0 && value <= 1) }'; then
  printf -- '--lambda-chunk 必须位于 [0,1]：%s\n' "${LAMBDA_CHUNK}" >&2
  exit 2
fi
if ! awk -v value="${ALL_PREFIX_AUX_WEIGHT}" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0) }'; then
  printf -- '--all-prefix-aux-weight 必须为非负数：%s\n' "${ALL_PREFIX_AUX_WEIGHT}" >&2
  exit 2
fi
if (( ORDINARY_ENABLED == 1 && RAVE_STEPS % RAVE_EPOCHS != 0 )); then
  printf -- '--rave-steps 必须能被 --rave-epochs 整除，才能映射为 MAC step_per_epoch：%s / %s\n' \
    "${RAVE_STEPS}" "${RAVE_EPOCHS}" >&2
  exit 2
fi
STEP_PER_EPOCH=$((RAVE_STEPS / RAVE_EPOCHS))
if [[ "${LEAVE_POLICY}" == "penalty" ]]; then
  printf -- '统一 Q/V trainer 始终按真实环境终止；--leave-policy penalty 已不再支持。\n' >&2
  exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-saved_models/${ENV_NAME}/QV_suite/K${CHUNK_SIZE}}"
SHARED_ROOT="${SHARED_PRETRAIN_ROOT:-${OUTPUT_ROOT}/shared_pretrain}"
LOG_ROOT="${OUTPUT_ROOT}/logs"
MANIFEST="${OUTPUT_ROOT}/manifest.tsv"
mkdir -p "${LOG_ROOT}"
if [[ ! -f "${MANIFEST}" ]]; then
  printf 'qv_type\tseed\tenv\tchunk_size\tbase_checkpoint\toutput_dir\tstatus\n' > "${MANIFEST}"
fi

shared_seed_dir() {
  local seed="$1"
  printf '%s/seed-%s' "${SHARED_ROOT}" "${seed}"
}

bc_checkpoint() {
  local seed="$1"
  printf '%s/%s/latest.pt' "$(shared_seed_dir "${seed}")" "${BC_SUBDIR}"
}

print_command() {
  local -a command=("$@")
  printf '[dry-run] '
  printf '%q ' "${command[@]}"
  printf '\n'
}

run_pretrain() {
  local seed="$1"
  local gpu="$2"
  local shared_dir checkpoint log_file
  shared_dir="$(shared_seed_dir "${seed}")"
  checkpoint="$(bc_checkpoint "${seed}")"
  log_file="${LOG_ROOT}/pretrain-seed${seed}.log"
  # 始终进入基础脚本，由它校验 actor 类型、蒸馏权重和采样模式后决定复用。
  local -a command=(env
    "CUDA_VISIBLE_DEVICES=${gpu}" "ENV_NAME=${ENV_NAME}" "ACTOR_TYPE=${ACTOR_TYPE}"
    "SEED=${seed}" "CUDA=0" "DEVICE=cuda:0" "CHUNK_SIZE=${CHUNK_SIZE}"
    "NUM_LEAVE_COMPUTE=${NUM_LEAVE_COMPUTE}" "LEAVE_THRESHOLD=${LEAVE_THRESHOLD}"
    "MAX_TURN=${MAX_TURN}" "FORCE_LENGTH=${MAX_TURN}" "LEAVE_POLICY=${LEAVE_POLICY}"
    "OUTPUT_ROOT=${OUTPUT_ROOT}"
    "RUN_NAME=qv-suite-${DATASET_TAG}-${ACTOR_TAG}" "RUN_DIR=${shared_dir}"
    "BC_SAVE_DIR=${shared_dir}/${BC_SUBDIR}" "TRAIN_STAGES=bc"
    "PRETRAIN_STEPS=${PRETRAIN_STEPS}" "DYNAMICS_PRETRAIN_STEPS=${DYNAMICS_PRETRAIN_STEPS}"
    "BC_SWANLAB_PROJECT=RAVE-H-QV-Suite-Pretrain"
    "SWANLAB_MODE=${SWANLAB_MODE}" bash script/run_dorl_mac_kuai_train.sh)
  if [[ "${DRY_RUN}" == "1" ]]; then print_command "${command[@]}"; return 0; fi
  if ! "${command[@]}" >"${log_file}" 2>&1; then
    printf '[failed] pretrain seed=%s；日志：%s\n' "${seed}" "${log_file}" >&2
    tail -n 20 "${log_file}" >&2 || true
    return 1
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    printf '[failed] pretrain seed=%s 未生成 checkpoint：%s\n' \
      "${seed}" "${checkpoint}" >&2
    return 1
  fi
}

run_qv() {
  local qv_type="$1"
  local seed="$2"
  local gpu="$3"
  local checkpoint output_dir latest log_file status="complete"
  local qv_eta_t="" qv_eta_d="" qv_eta_c=""
  checkpoint="$(bc_checkpoint "${seed}")"
  output_dir="${OUTPUT_ROOT}/${qv_type}/seed-${seed}"
  log_file="${LOG_ROOT}/${qv_type}-seed${seed}.log"
  if [[ "${DRY_RUN}" != "1" && ! -f "${checkpoint}" ]]; then
    printf '缺少共享 BC checkpoint：%s；请先运行 --stage pretrain 或 --stage all。\n' \
      "${checkpoint}" >&2
    return 1
  fi
  latest="${output_dir}/latest.pt"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${latest}" ]]; then
    printf '[skip] qv_type=%s seed=%s checkpoint=%s\n' "${qv_type}" "${seed}" "${latest}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${qv_type}" "${seed}" "${ENV_NAME}" "${CHUNK_SIZE}" "${checkpoint}" "${output_dir}" skipped >> "${MANIFEST}"
    return 0
  fi

  local -a command
  if [[ "${qv_type}" == "ordinary" || "${qv_type}" == "acsac_transformer" ]]; then
    local critic_backbone="mlp"
    if [[ "${qv_type}" == "acsac_transformer" ]]; then
      critic_backbone="acsac_transformer"
    fi
    command=(env
      "CUDA_VISIBLE_DEVICES=${gpu}" "ENV_NAME=${ENV_NAME}" "ACTOR_TYPE=${ACTOR_TYPE}"
      "SEED=${seed}" "CUDA=0" "DEVICE=cuda:0" "CHUNK_SIZE=${CHUNK_SIZE}"
      "NUM_LEAVE_COMPUTE=${NUM_LEAVE_COMPUTE}" "LEAVE_THRESHOLD=${LEAVE_THRESHOLD}"
      "MAX_TURN=${MAX_TURN}" "FORCE_LENGTH=${MAX_TURN}" "LEAVE_POLICY=${LEAVE_POLICY}"
      "COMPLETION_WINDOW=${CHUNK_SIZE}"
      "TEST_NUM=${TEST_NUM}" "NUM_SAMPLES_TEST=${NUM_SAMPLES_TEST}"
      "BATCH_SIZE=${QV_BATCH_SIZE}" "NUM_SAMPLES_TRAIN=${RAVE_ROLLOUT_CANDIDATES}"
      "ROLLOUT_DEPTH=${ORDINARY_ROLLOUT_DEPTH}" "LAMBDA_CHUNK=${LAMBDA_CHUNK}"
      "ALL_PREFIX_AUX_WEIGHT=${ALL_PREFIX_AUX_WEIGHT}"
      "GAMMA=${GAMMA}" "BUFFER_SIZE=${BUFFER_SIZE}"
      "OUTPUT_ROOT=${OUTPUT_ROOT}" "BC_ACTOR_CKPT=${checkpoint}"
      "BC_SAVE_DIR=$(dirname "${checkpoint}")" "MAC_SAVE_DIR=${output_dir}"
      "RUN_DIR=${output_dir}" "TRAIN_STAGES=qv" "CRITIC_BACKBONE=${critic_backbone}"
      "FREEZE_DYNAMICS_DURING_QV=1" "DYNAMICS_LOSS_WEIGHT=0" "EPOCH=${RAVE_EPOCHS}"
      "STEP_PER_EPOCH=${STEP_PER_EPOCH}" "EVAL_EVERY_N_EPOCHS=${RAVE_EVAL_EVERY_EPOCHS}"
      "EVAL_EPISODES=${EVAL_EPISODES}" "QV_SWANLAB_PROJECT=RAVE-H-QV-Suite"
      "QV_RUN_NAME=qv-suite-${DATASET_TAG}-K${CHUNK_SIZE}-${qv_type}-seed${seed}"
      "SWANLAB_MODE=${SWANLAB_MODE}" bash script/run_dorl_mac_kuai_train.sh)
  else
    if [[ "${qv_type}" == "multi_prefix_transformer" ]]; then
      qv_eta_t="${ETA_T}"
      qv_eta_d="${ETA_D}"
      qv_eta_c="${ETA_C}"
    fi
    command=(env
      "CUDA_VISIBLE_DEVICES=${gpu}" "ENV_NAME=${ENV_NAME}" "SEED=${seed}"
      "CUDA=0" "DEVICE=cuda:0" "CHUNK_SIZE=${CHUNK_SIZE}"
      "NUM_LEAVE_COMPUTE=${NUM_LEAVE_COMPUTE}" "LEAVE_THRESHOLD=${LEAVE_THRESHOLD}"
      "MAX_TURN=${MAX_TURN}" "FORCE_LENGTH=${MAX_TURN}"
      "COMPLETION_WINDOW=${CHUNK_SIZE}"
      "TEST_NUM=${TEST_NUM}" "NUM_CANDIDATES=${NUM_SAMPLES_TEST}"
      "BATCH_SIZE=${QV_BATCH_SIZE}" "ROLLOUT_CANDIDATES=${RAVE_ROLLOUT_CANDIDATES}"
      "EPISODES_PER_COLLECT=${RAVE_EPISODES_PER_COLLECT}"
      "LAMBDA_CHUNK=${LAMBDA_CHUNK}" "GAMMA=${GAMMA}"
      "BUFFER_SIZE=${BUFFER_SIZE}" "OUTPUT_ROOT=${OUTPUT_ROOT}"
      "MAC_CHECKPOINT=${checkpoint}" "QV_TYPE=${qv_type}"
      "ETA_T=${qv_eta_t}" "ETA_D=${qv_eta_d}" "ETA_C=${qv_eta_c}"
      "RUN_NAME=qv-suite-${DATASET_TAG}-K${CHUNK_SIZE}-${qv_type}-seed${seed}"
      "OUTPUT_DIR=${output_dir}" "STEPS=${RAVE_STEPS}" "EPOCHS=${RAVE_EPOCHS}"
      "EVAL_EVERY_EPOCHS=${RAVE_EVAL_EVERY_EPOCHS}"
      "TRAIN_EVAL_EPISODES=${EVAL_EPISODES}"
      "EVAL_SWANLAB_PROJECT=${EVAL_SWANLAB_PROJECT}"
      "SWANLAB_PROJECT=RAVE-H-QV-Suite" "SWANLAB_MODE=${SWANLAB_MODE}"
      "MODE=train" bash script/run_rave_kuai.sh)
  fi
  printf '[run] qv_type=%s seed=%s gpu=%s output=%s\n' \
    "${qv_type}" "${seed}" "${gpu}" "${output_dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then print_command "${command[@]}"; return 0; fi
  mkdir -p "${output_dir}"
  if ! "${command[@]}" >"${log_file}" 2>&1; then status="failed"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${qv_type}" "${seed}" "${ENV_NAME}" "${CHUNK_SIZE}" "${checkpoint}" "${output_dir}" "${status}" >> "${MANIFEST}"
  if [[ "${status}" == "failed" ]]; then
    printf '[failed] qv_type=%s seed=%s；日志：%s\n' "${qv_type}" "${seed}" "${log_file}" >&2
    return 1
  fi
}

run_rave_eval() {
  local qv_type="$1"
  local seed="$2"
  local gpu="$3"
  local output_dir checkpoint eval_dir result log_file run_name status="complete"
  output_dir="${OUTPUT_ROOT}/${qv_type}/seed-${seed}"
  checkpoint="${output_dir}/latest.pt"
  eval_dir="${output_dir}/eval_${EVAL_SPLIT}"
  result="${eval_dir}/summary_metrics.json"
  log_file="${LOG_ROOT}/${qv_type}-eval-${EVAL_SPLIT}-seed${seed}.log"
  run_name="qv-suite-${DATASET_TAG}-K${CHUNK_SIZE}-${qv_type}-eval-${EVAL_SPLIT}-seed${seed}"
  if [[ "${DRY_RUN}" != "1" && ! -f "${checkpoint}" ]]; then
    printf '[failed] RAVE-H-Eval 缺少 checkpoint：%s\n' "${checkpoint}" >&2
    return 1
  fi
  if [[ "${SKIP_EXISTING}" == "1" && -f "${result}" ]]; then
    printf '[skip] RAVE-H-Eval qv_type=%s seed=%s result=%s\n' \
      "${qv_type}" "${seed}" "${result}"
    return 0
  fi
  local -a command
  if [[ "${qv_type}" == "ordinary" || "${qv_type}" == "acsac_transformer" ]]; then
    local critic_backbone="mlp"
    if [[ "${qv_type}" == "acsac_transformer" ]]; then
      critic_backbone="acsac_transformer"
    fi
    command=(env
      "CUDA_VISIBLE_DEVICES=${gpu}" "ENV_NAME=${ENV_NAME}" "ACTOR_TYPE=${ACTOR_TYPE}"
      "SEED=${seed}" "CUDA=0" "DEVICE=cuda:0" "CHUNK_SIZE=${CHUNK_SIZE}"
      "NUM_LEAVE_COMPUTE=${NUM_LEAVE_COMPUTE}" "LEAVE_THRESHOLD=${LEAVE_THRESHOLD}"
      "MAX_TURN=${MAX_TURN}" "FORCE_LENGTH=${MAX_TURN}" "LEAVE_POLICY=${LEAVE_POLICY}"
      "COMPLETION_WINDOW=${CHUNK_SIZE}" "TEST_NUM=${TEST_NUM}"
      "NUM_SAMPLES_TEST=${NUM_SAMPLES_TEST}" "BUFFER_SIZE=${BUFFER_SIZE}"
      "OUTPUT_ROOT=${OUTPUT_ROOT}" "RUN_DIR=${output_dir}" "MAC_SAVE_DIR=${output_dir}"
      "EVAL_SAVE_DIR=${eval_dir}" "TRAIN_STAGES=eval" "CRITIC_BACKBONE=${critic_backbone}"
      "EVAL_EPISODES=${EVAL_EPISODES}" "EVAL_RUN_NAME=${run_name}"
      "EVAL_SWANLAB_PROJECT=${EVAL_SWANLAB_PROJECT}" "SWANLAB_MODE=${SWANLAB_MODE}"
      bash script/run_dorl_mac_kuai_train.sh)
  else
    command=(env
      "CUDA_VISIBLE_DEVICES=${gpu}" "DEVICE=cuda:0" "SEED=${seed}"
      "ENV_NAME=${ENV_NAME}" "CHUNK_SIZE=${CHUNK_SIZE}" "OUTPUT_ROOT=${OUTPUT_ROOT}"
      "NUM_LEAVE_COMPUTE=${NUM_LEAVE_COMPUTE}" "LEAVE_THRESHOLD=${LEAVE_THRESHOLD}"
      "MAX_TURN=${MAX_TURN}" "FORCE_LENGTH=${MAX_TURN}"
      "COMPLETION_WINDOW=${CHUNK_SIZE}"
      "TEST_NUM=${TEST_NUM}" "NUM_CANDIDATES=${NUM_SAMPLES_TEST}"
      "BUFFER_SIZE=${BUFFER_SIZE}" "DORL_EVAL_PARITY=1"
      "RAVE_CHECKPOINT=${checkpoint}" "RUN_NAME=${run_name}"
      "OUTPUT_DIR=${eval_dir}" "SWANLAB_PROJECT=${EVAL_SWANLAB_PROJECT}"
      "SWANLAB_MODE=${SWANLAB_MODE}" "EVAL_SPLIT=${EVAL_SPLIT}"
      "EVAL_EPISODES=${EVAL_EPISODES}" "MODE=eval" bash script/run_rave_kuai.sh)
  fi
  printf '[run] eval qv_type=%s seed=%s gpu=%s output=%s\n' \
    "${qv_type}" "${seed}" "${gpu}" "${eval_dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then print_command "${command[@]}"; return 0; fi
  mkdir -p "${eval_dir}"
  if ! "${command[@]}" >"${log_file}" 2>&1; then status="failed"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${qv_type}-eval-${EVAL_SPLIT}" "${seed}" "${ENV_NAME}" "${CHUNK_SIZE}" \
    "${checkpoint}" "${eval_dir}" "${status}" >> "${MANIFEST}"
  if [[ "${status}" == "failed" ]]; then
    printf '[failed] eval qv_type=%s seed=%s；日志：%s\n' \
      "${qv_type}" "${seed}" "${log_file}" >&2
    return 1
  fi
}

run_seed_group() {
  local task="$1"
  local qv_type="${2:-}"
  local failures=0
  local -a pids=()
  for index in "${!SEED_VALUES[@]}"; do
    local seed="${SEED_VALUES[${index}]}"
    local gpu="${GPU_VALUES[$((index % ${#GPU_VALUES[@]}))]}"
    if [[ "${PARALLEL_SEEDS}" == "1" ]]; then
      case "${task}" in
        pretrain) run_pretrain "${seed}" "${gpu}" & ;;
        qv) run_qv "${qv_type}" "${seed}" "${gpu}" & ;;
        eval) run_rave_eval "${qv_type}" "${seed}" "${gpu}" & ;;
      esac
      pids+=("$!")
    else
      case "${task}" in
        pretrain) run_pretrain "${seed}" "${gpu}" || failures=$((failures + 1)) ;;
        qv) run_qv "${qv_type}" "${seed}" "${gpu}" || failures=$((failures + 1)) ;;
        eval) run_rave_eval "${qv_type}" "${seed}" "${gpu}" || failures=$((failures + 1)) ;;
      esac
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || failures=$((failures + 1))
  done
  if (( failures > 0 )); then
    printf '%s 阶段有 %s 个运行失败，请检查 %s。\n' "${task}" "${failures}" "${LOG_ROOT}" >&2
    return 1
  fi
}

if [[ "${STAGE}" == "all" || "${STAGE}" == "pretrain" ]]; then
  run_seed_group pretrain
fi
if [[ "${STAGE}" == "all" || "${STAGE}" == "qv" ]]; then
  for qv_type in "${QV_TYPE_VALUES[@]}"; do
    run_seed_group qv "${qv_type}"
    if [[ "${RUN_EVAL_AFTER_TRAIN}" == "1" ]]; then
      run_seed_group eval "${qv_type}"
    fi
  done
fi
if [[ "${STAGE}" == "eval" ]]; then
  for qv_type in "${QV_TYPE_VALUES[@]}"; do
    run_seed_group eval "${qv_type}"
  done
fi

printf '五种子 Q/V 阶段完成。输出目录：%s\n' "${OUTPUT_ROOT}"
