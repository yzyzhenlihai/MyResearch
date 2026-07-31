#!/usr/bin/env bash
# 一次只搜索一个 DARLR 论文超参数；不同 seed 串行运行。
set -euo pipefail

cd "$(dirname "$0")/.."

TUNE_TARGET="${TUNE_TARGET:-k}"
SEEDS="${SEEDS:-2023}"
CUDA="${CUDA:-1}"
DRY_RUN="${DRY_RUN:-0}"
SAVE_MODEL="${SAVE_MODEL:-1}"
SAVE_BEST_ONLY="${SAVE_BEST_ONLY:-1}"
BEST_METRIC="${BEST_METRIC:-NX_0}"
RUN_PREFIX="${RUN_PREFIX:-DARLR_KuaiRec}"

BASE_SELECTOR_K="${BASE_SELECTOR_K:-40}"
BASE_LAMBDA_S="${BASE_LAMBDA_S:-2}"
BASE_LAMBDA_D="${BASE_LAMBDA_D:-0.1}"
BASE_LAMBDA_U="${BASE_LAMBDA_U:-0.1}"
BASE_LAMBDA_E="${BASE_LAMBDA_E:-0.1}"
BASE_SELECTOR_LAYERS="${BASE_SELECTOR_LAYERS:-1}"
BASE_SELECTOR_HEADS="${BASE_SELECTOR_HEADS:-1}"
BASE_SELECTOR_PREF_DIM="${BASE_SELECTOR_PREF_DIM:-64}"
HEAD_SEARCH_PREF_DIM="${HEAD_SEARCH_PREF_DIM:-60}"
BASE_WINDOW_SIZE="${BASE_WINDOW_SIZE:-3}"

case "${TUNE_TARGET}" in
  k)
    read -r -a VALUES <<< "${K_VALUES:-10 20 30 40}"
    ;;
  lambda_s)
    read -r -a VALUES <<< "${LAMBDA_S_VALUES:-0.5 1 2 5}"
    ;;
  lambda_d)
    read -r -a VALUES <<< "${LAMBDA_D_VALUES:-0.01 0.05 0.1 0.5}"
    ;;
  lambda_u)
    read -r -a VALUES <<< "${LAMBDA_U_VALUES:-0.01 0.05 0.1 0.5 1}"
    ;;
  lambda_e)
    read -r -a VALUES <<< "${LAMBDA_E_VALUES:-0.01 0.05 0.1 0.5 1}"
    ;;
  layers)
    read -r -a VALUES <<< "${LAYER_VALUES:-1 2 3}"
    ;;
  heads)
    read -r -a VALUES <<< "${HEAD_VALUES:-1 2 3}"
    ;;
  window)
    read -r -a VALUES <<< "${WINDOW_VALUES:-3 5 10}"
    ;;
  *)
    echo "Unsupported TUNE_TARGET='${TUNE_TARGET}'. Expected k, lambda_s, lambda_d, lambda_u, lambda_e, layers, heads, or window." >&2
    exit 2
    ;;
esac

for seed in ${SEEDS}; do
  for value in "${VALUES[@]}"; do
    selector_k="${BASE_SELECTOR_K}"
    lambda_s="${BASE_LAMBDA_S}"
    lambda_d="${BASE_LAMBDA_D}"
    lambda_u="${BASE_LAMBDA_U}"
    lambda_e="${BASE_LAMBDA_E}"
    selector_layers="${BASE_SELECTOR_LAYERS}"
    selector_heads="${BASE_SELECTOR_HEADS}"
    selector_pref_dim="${BASE_SELECTOR_PREF_DIM}"
    window_size="${BASE_WINDOW_SIZE}"

    case "${TUNE_TARGET}" in
      k)
        selector_k="${value}"
        ;;
      lambda_s)
        lambda_s="${value}"
        ;;
      lambda_d)
        lambda_d="${value}"
        ;;
      lambda_u)
        lambda_u="${value}"
        ;;
      lambda_e)
        lambda_e="${value}"
        ;;
      layers)
        selector_layers="${value}"
        ;;
      heads)
        selector_heads="${value}"
        # 论文搜索包含 3 heads；统一使用 60 维保证每个 head 配置可整除。
        selector_pref_dim="${HEAD_SEARCH_PREF_DIM}"
        ;;
      window)
        window_size="${value}"
        ;;
    esac

    safe_value="${value//./p}"
    safe_value="${safe_value//-/m}"
    message="${RUN_PREFIX}_paper_${TUNE_TARGET}_${safe_value}_seed${seed}"

    env \
      SEED="${seed}" \
      CUDA="${CUDA}" \
      DRY_RUN="${DRY_RUN}" \
      SAVE_MODEL="${SAVE_MODEL}" \
      SAVE_BEST_ONLY="${SAVE_BEST_ONLY}" \
      BEST_METRIC="${BEST_METRIC}" \
      MESSAGE="${message}" \
      SELECTOR_POLICY_MODE=learned \
      SELECTOR_K="${selector_k}" \
      SELECTOR_LAMBDA_S="${lambda_s}" \
      SELECTOR_LAMBDA_D="${lambda_d}" \
      LAMBDA_UNCERTAINTY="${lambda_u}" \
      LAMBDA_ENTROPY="${lambda_e}" \
      SELECTOR_NUM_LAYERS="${selector_layers}" \
      SELECTOR_NUM_HEADS="${selector_heads}" \
      SELECTOR_PREF_DIM="${selector_pref_dim}" \
      WINDOW_SIZE="${window_size}" \
      DYNAMIC_REWARD_MODE=reference_mean \
      DYNAMIC_UNCERTAINTY_MODE=dynamic \
      bash script/run_DARLR_KuaiRec.sh
  done
done
