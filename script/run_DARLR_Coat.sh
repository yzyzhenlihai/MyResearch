#!/usr/bin/env bash
# ============================================================================
# DARLR on Coat (CoatEnv-v0) — 训练入口
#
# 参考: docs/DARLR_DORL融合复现方案.md §5.4 / §5.5
# 依赖: examples/advance/run_DARLR.py, script/run_DARLR_reproduce.sh
#
# 注意: Coat 数据稀疏，默认使用 no_feature_level、no_exploration_noise、
# 更大的 learning rate 与 entropy 系数，与 DORL 现有惯例保持一致。
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

export DATASET="CoatEnv-v0"

export SEED="${SEED:-2023}"
export CUDA="${CUDA:-0}"
export EPOCH="${EPOCH:-100}"
export STEP_PER_EPOCH="${STEP_PER_EPOCH:-100000}"
export TRAINING_NUM="${TRAINING_NUM:-100}"
export TEST_NUM="${TEST_NUM:-100}"
export BATCH_SIZE="${BATCH_SIZE:-256}"

export WHICH_TRACKER="${WHICH_TRACKER:-sasrec}"
export WINDOW_SIZE="${WINDOW_SIZE:-3}"
export NUM_HEADS="${NUM_HEADS:-1}"

# Episode 退出机制 (对齐 src/core/util/data.py::CoatEnv-v0 默认)
export MAX_TURN="${MAX_TURN:-30}"
export FORCE_LENGTH="${FORCE_LENGTH:-10}"
export NUM_LEAVE_COMPUTE="${NUM_LEAVE_COMPUTE:-7}"
export LEAVE_THRESHOLD="${LEAVE_THRESHOLD:-6}"

# selector 超参
export SELECTOR_K="${SELECTOR_K:-10}"                   # 搜索范围: 10, 20, 30, 40
export SELECTOR_CANDIDATE_SIZE="${SELECTOR_CANDIDATE_SIZE:-512}"
export SELECTOR_CANDIDATE_MODE="${SELECTOR_CANDIDATE_MODE:-embedding_topk}"
export SELECTOR_PREF_DIM="${SELECTOR_PREF_DIM:-64}"
export SELECTOR_NUM_HEADS="${SELECTOR_NUM_HEADS:-1}"
export SELECTOR_NUM_LAYERS="${SELECTOR_NUM_LAYERS:-1}"
export SELECTOR_LAMBDA_S="${SELECTOR_LAMBDA_S:-1.0}"
export SELECTOR_LAMBDA_D="${SELECTOR_LAMBDA_D:-0.05}"

export LAMBDA_UNCERTAINTY="${LAMBDA_UNCERTAINTY:-0.05}"
export LAMBDA_ENTROPY="${LAMBDA_ENTROPY:-0.05}"

# Coat / Yahoo 训练惯例
export LR="${LR:-0.005}"
export ENT_COEF="${ENT_COEF:-0.01}"

export MESSAGE="${MESSAGE:-DARLR_Coat}"
export READ_MESSAGE="${READ_MESSAGE:-pointneg}"

exec bash "$(dirname "$0")/run_DARLR_reproduce.sh" "$@"
