#!/usr/bin/env bash
set -euo pipefail

# K5 ordinary 多训练 seed 的 action chunk 续执行价值差一键实验。
# 所有参数均可用同名环境变量覆盖；默认自动发现 seed-*/latest.pt。

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
MODEL_ROOT="${MODEL_ROOT:-saved_models/KuaiEnv-v0/QV_suite/K${CHUNK_SIZE}/ordinary}"
EVAL_EPISODES="${EVAL_EPISODES:-300}"
NUM_SAMPLES_TEST="${NUM_SAMPLES_TEST:-32}"
MAX_CHUNKS_PER_EPISODE="${MAX_CHUNKS_PER_EPISODE:-0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-5000}"
EVAL_SEED="${EVAL_SEED:-2023}"
OUTPUT_DIR="${OUTPUT_DIR:-results_analysis/replanning_motivation_k5_ordinary}"
DOMINANT_TOP_RATE="${DOMINANT_TOP_RATE:-0.8}"
RETURN_NONINFERIORITY_MARGIN="${RETURN_NONINFERIORITY_MARGIN:-0.0}"
DIVERSITY_IMPROVEMENT_MARGIN="${DIVERSITY_IMPROVEMENT_MARGIN:-0.0}"
RETURN_SCALE="${RETURN_SCALE:-1.0}"
DIVERSITY_SCALE="${DIVERSITY_SCALE:-0.01}"
DIVERSITY_WEIGHT="${DIVERSITY_WEIGHT:-1.0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf '[run_replanning_motivation] ERROR: Python 不可执行：%s\n' "${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! "${CHUNK_SIZE}" =~ ^[2-9][0-9]*$ ]]; then
  printf '[run_replanning_motivation] ERROR: CHUNK_SIZE 必须为大于 1 的整数。\n' >&2
  exit 2
fi
if [[ ! -d "${MODEL_ROOT}" ]]; then
  printf '[run_replanning_motivation] ERROR: 模型目录不存在：%s\n' "${MODEL_ROOT}" >&2
  exit 2
fi

shopt -s nullglob
CHECKPOINTS=("${MODEL_ROOT}"/seed-*/latest.pt)
shopt -u nullglob
if (( ${#CHECKPOINTS[@]} == 0 )); then
  printf '[run_replanning_motivation] ERROR: 未找到 %s/seed-*/latest.pt。\n' \
    "${MODEL_ROOT}" >&2
  exit 2
fi

CHECKPOINT_ARGS=()
for checkpoint_path in "${CHECKPOINTS[@]}"; do
  CHECKPOINT_ARGS+=("${CHUNK_SIZE}=${checkpoint_path}")
done

COMMAND=(
  "${PYTHON_BIN}"
  examples/our_model/runners/eval_replanning_motivation.py
  --mac-ckpts "${CHECKPOINT_ARGS[@]}"
  --device "${DEVICE}"
  --eval-episodes "${EVAL_EPISODES}"
  --num-samples-test "${NUM_SAMPLES_TEST}"
  --max-chunks-per-episode "${MAX_CHUNKS_PER_EPISODE}"
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
  --seed "${EVAL_SEED}"
  --output-dir "${OUTPUT_DIR}"
  --dominant-top-rate "${DOMINANT_TOP_RATE}"
  --return-noninferiority-margin "${RETURN_NONINFERIORITY_MARGIN}"
  --diversity-improvement-margin "${DIVERSITY_IMPROVEMENT_MARGIN}"
  --return-scale "${RETURN_SCALE}"
  --diversity-scale "${DIVERSITY_SCALE}"
  --diversity-weight "${DIVERSITY_WEIGHT}"
)

printf '[run_replanning_motivation] checkpoints=%s, K=%s, episodes=%s, device=%s\n' \
  "${#CHECKPOINTS[@]}" "${CHUNK_SIZE}" "${EVAL_EPISODES}" "${DEVICE}"
printf '[run_replanning_motivation] output=%s\n' "${OUTPUT_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

"${COMMAND[@]}"
printf '[run_replanning_motivation] 完成：%s\n' "${OUTPUT_DIR}"
