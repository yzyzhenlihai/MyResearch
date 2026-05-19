#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-.:./src:./src/DeepCTR-Torch:./src/tianshou}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

PYTHON_BIN="${PYTHON_BIN:-/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python}"

DATASET="${DATASET:-KuaiEnv-v0}"
SEED="${SEED:-2023}"
CUDA="${CUDA:-0}"
CPU_FLAG="${CPU_FLAG:-0}"
EPOCH="${EPOCH:-100}"
STEP_PER_EPOCH="${STEP_PER_EPOCH:-100000}"
STEP_PER_COLLECT="${STEP_PER_COLLECT:-100}"
EPISODE_PER_COLLECT="${EPISODE_PER_COLLECT:-100}"
REPEAT_PER_COLLECT="${REPEAT_PER_COLLECT:-1}"
TRAINING_NUM="${TRAINING_NUM:-100}"
TEST_NUM="${TEST_NUM:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BUFFER_SIZE="${BUFFER_SIZE:-100000}"
MAX_TURN="${MAX_TURN:-30}"
FORCE_LENGTH="${FORCE_LENGTH:-10}"

WHICH_TRACKER="${WHICH_TRACKER:-avg}"
REWARD_HANDLE="${REWARD_HANDLE:-cat}"
WINDOW_SIZE="${WINDOW_SIZE:-3}"
NUM_HEADS="${NUM_HEADS:-1}"
READ_MESSAGE="${READ_MESSAGE:-pointneg}"
MESSAGE="${MESSAGE:-DORL_DOSER_ONPOLICY_${DATASET}}"

DIFFUSION_SAVE_ROOT="${DIFFUSION_SAVE_ROOT:-saved_models}"
DIFFUSION_SAMPLE_STEPS="${DIFFUSION_SAMPLE_STEPS:-20}"

# 若需要优先使用新格式 flat artifact，可显式设置：
#   DIFFUSION_ARTIFACT_NAME="" bash script/run_DORL_DOSER_onpolicy.sh
if [[ "${DIFFUSION_ARTIFACT_NAME+x}" != "x" ]]; then
  DIFFUSION_ARTIFACT_NAME="DM_KuaiEnv-v0_small_data"
fi

DOSER_BETA="${DOSER_BETA:-0.001}"
DOSER_LAM="${DOSER_LAM:-0.001}"
DOSER_ETA="${DOSER_ETA:-0.9}"
DOSER_EXPECTILE="${DOSER_EXPECTILE:-0.9}"
DOSER_ACTION_SAMPLES="${DOSER_ACTION_SAMPLES:-10}"
DOSER_Q_MIN="${DOSER_Q_MIN:-0.0}"
DOSER_AUX_CRITIC_COEF="${DOSER_AUX_CRITIC_COEF:-1.0}"
DOSER_LOG_INTERVAL="${DOSER_LOG_INTERVAL:-100}"

LR="${LR:-0.001}"
GAMMA="${GAMMA:-0.9}"
VF_COEF="${VF_COEF:-0.5}"
ENT_COEF="${ENT_COEF:-0.0}"
GAE_LAMBDA="${GAE_LAMBDA:-1.0}"
LAMBDA_ENTROPY="${LAMBDA_ENTROPY:-0.05}"
ENTROPY_WINDOW="${ENTROPY_WINDOW:-1 2}"

EXTRA_ARGS=()
DATASET_ARGS=()

case "${DATASET}" in
  KuaiEnv-v0|KuaiRand-v0)
    DATASET_ARGS+=(--is_feature_level)
    ;;
  YahooEnv-v0|CoatEnv-v0)
    DATASET_ARGS+=(--no_feature_level --no_exploration_noise)
    ;;
  *)
    echo "Unsupported DATASET='${DATASET}'. Expected KuaiEnv-v0, KuaiRand-v0, YahooEnv-v0, or CoatEnv-v0." >&2
    exit 2
    ;;
esac

if [[ "${CPU_FLAG}" == "1" ]]; then
  EXTRA_ARGS+=(--cpu)
fi

if [[ -n "${DIFFUSION_ARTIFACT_NAME}" ]]; then
  EXTRA_ARGS+=(--diffusion_artifact_name "${DIFFUSION_ARTIFACT_NAME}")
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCH=1
  STEP_PER_EPOCH="${SMOKE_STEP_PER_EPOCH:-2}"
  STEP_PER_COLLECT="${SMOKE_STEP_PER_COLLECT:-2}"
  EPISODE_PER_COLLECT=1
  TRAINING_NUM=1
  TEST_NUM=1
  BATCH_SIZE=2
  BUFFER_SIZE=10
  DOSER_ACTION_SAMPLES=1
  DIFFUSION_SAMPLE_STEPS=1
  MAX_TURN="${SMOKE_MAX_TURN:-2}"
  FORCE_LENGTH="${SMOKE_FORCE_LENGTH:-1}"
fi

"${PYTHON_BIN}" examples/our_model/dorl_doser_onpolicy.py \
  --model_name "DORL_DOSER_ONPOLICY" \
  --message "${MESSAGE}" \
  --env "${DATASET}" \
  --seed "${SEED}" \
  --cuda "${CUDA}" \
  --epoch "${EPOCH}" \
  --step-per-epoch "${STEP_PER_EPOCH}" \
  --step-per-collect "${STEP_PER_COLLECT}" \
  --episode-per-collect "${EPISODE_PER_COLLECT}" \
  --repeat-per-collect "${REPEAT_PER_COLLECT}" \
  --training-num "${TRAINING_NUM}" \
  --test-num "${TEST_NUM}" \
  --batch-size "${BATCH_SIZE}" \
  --buffer-size "${BUFFER_SIZE}" \
  --max_turn "${MAX_TURN}" \
  --force_length "${FORCE_LENGTH}" \
  --which_tracker "${WHICH_TRACKER}" \
  --reward_handle "${REWARD_HANDLE}" \
  --window_size "${WINDOW_SIZE}" \
  --num_heads "${NUM_HEADS}" \
  --read_message "${READ_MESSAGE}" \
  --diffusion_save_root "${DIFFUSION_SAVE_ROOT}" \
  --diffusion_sample_steps "${DIFFUSION_SAMPLE_STEPS}" \
  --doser_beta "${DOSER_BETA}" \
  --doser_lam "${DOSER_LAM}" \
  --doser_eta "${DOSER_ETA}" \
  --doser_expectile "${DOSER_EXPECTILE}" \
  --doser_action_samples "${DOSER_ACTION_SAMPLES}" \
  --doser_q_min "${DOSER_Q_MIN}" \
  --doser_aux_critic_coef "${DOSER_AUX_CRITIC_COEF}" \
  --doser_detach_aux_state \
  --doser_log_interval "${DOSER_LOG_INTERVAL}" \
  --vf-coef "${VF_COEF}" \
  --ent-coef "${ENT_COEF}" \
  --gae-lambda "${GAE_LAMBDA}" \
  --gamma "${GAMMA}" \
  --lr "${LR}" \
  --lambda_entropy "${LAMBDA_ENTROPY}" \
  --entropy_window ${ENTROPY_WINDOW} \
  --wandb_mode "${WANDB_MODE}" \
  "${DATASET_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
