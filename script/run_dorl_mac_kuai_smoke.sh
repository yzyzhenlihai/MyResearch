#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export SWANLAB_MODE="${SWANLAB_MODE:-offline}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-easyrl4rec}"

PYTHON_BIN="${PYTHON_BIN:-/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python}"

"${PYTHON_BIN}" examples/our_model/runners/run_dorl_mac.py \
  --env KuaiEnv-v0 \
  --user_model_name DeepFM \
  --read_message pointneg \
  --dataset_path data/KuaiRec/data_processed/DM_KuaiEnv-v0_small_data.pkl \
  --which_tracker avg \
  --reward_handle cat \
  --window_size 3 \
  --chunk_size 3 \
  --gamma 0.9 \
  --seed 2023 \
  --device cpu \
  --cuda 0 \
  --batch_size 4 \
  --max_trajectories 4 \
  --max_chunks 64 \
  --pretrain_steps 2 \
  --epoch 2 \
  --step-per-epoch 1 \
  --test-num 1 \
  --num_samples_train 2 \
  --num_samples_test 2 \
  --max_turn 5 \
  --force_length 5 \
  --swanlab_project DORL-MAC \
  --run_name dorl-mac
