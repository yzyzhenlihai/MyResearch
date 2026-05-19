#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "${SWANLAB_MODE:-}" ]]; then
  export SWANLAB_MODE
fi
PYTHON_BIN="${PYTHON_BIN:-/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python}"

"${PYTHON_BIN}" examples/advance/run_DARLR.py \
  --model_name "DARLR" \
  --env KuaiEnv-v0 \
  --seed 2023 \
  --cuda 0 \
  --cpu \
  --epoch 1 \
  --step-per-epoch 1 \
  --episode-per-collect 1 \
  --training-num 1 \
  --test-num 1 \
  --batch-size 1 \
  --max_turn 2 \
  --force_length 1 \
  --which_tracker sasrec \
  --reward_handle "cat" \
  --window_size 3 \
  --num_heads 1 \
  --selector_k 1 \
  --selector_candidate_size 2 \
  --selector_candidate_mode random \
  --selector_lambda_s 1.0 \
  --selector_lambda_d 0.05 \
  --lambda_uncertainty 0.05 \
  --lambda_entropy 0.0 \
  --entropy_window \
  --read_message "pointneg" \
  --message "DARLR_smoke"
