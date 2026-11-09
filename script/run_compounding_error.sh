#!/usr/bin/env bash
set -euo pipefail

# 使用 K=1 ordinary 的五个训练 seed，在固定日志动作下评估长程递归状态误差。

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
MAX_HORIZON="${MAX_HORIZON:-100}"
STARTS_PER_USER="${STARTS_PER_USER:-10}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
OUTPUT_DIR="${OUTPUT_DIR:-saved_models/KuaiEnv-v0/QV_suite/K1/compounding_error}"

exec "${PYTHON_BIN}" -m examples.our_model.runners.eval_compounding_error \
  --device "${DEVICE}" \
  --max-horizon "${MAX_HORIZON}" \
  --starts-per-user "${STARTS_PER_USER}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
