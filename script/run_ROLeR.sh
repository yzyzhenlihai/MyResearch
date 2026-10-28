#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${1:-KuaiEnv-v0}"
SEED="${2:-0}"
CUDA_ID="${3:-0}"
EPOCHS="${EPOCHS:-200}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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
    echo "Unsupported ROLeR environment: ${ENV_NAME}" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" examples/advance/run_ROLeR.py \
  --env "${ENV_NAME}" \
  --seed "${SEED}" \
  --cuda "${CUDA_ID}" \
  --epoch "${EPOCHS}" \
  --step-per-epoch "${STEPS_PER_EPOCH}" \
  --training-num 100 \
  --test-num 100 \
  --which_tracker "${TRACKER}" \
  --window_size "${WINDOW_SIZE}" \
  --num_leave_compute 1 \
  --leave_threshold 0 \
  --reward_handle cat \
  --read_message pointneg \
  --message "ROLeR-${ENV_NAME}-seed${SEED}" \
  --reward_formula official-minus \
  --is_scratch

