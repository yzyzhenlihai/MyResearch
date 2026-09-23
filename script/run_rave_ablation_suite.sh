#!/usr/bin/env bash
set -euo pipefail

# Five-seed RAVE-H ablation launcher. All paths and seed/GPU assignments may be
# overridden with environment variables; extra CLI arguments are forwarded to
# script/run_rave_kuai.sh.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-KuaiEnv-v0}"
CHUNK_SIZE="${CHUNK_SIZE:-3}"
SEEDS="${SEEDS:-2023 2024 2025 2026 2027}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4}"
METHODS="${METHODS:-rave_multiobjective_adaptive single_objective_adaptive}"
FIXED_HORIZONS="${FIXED_HORIZONS:-all}"
SUITE_NAME="${SUITE_NAME:-rave-h-ablation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-saved_models/${ENV_NAME}/QV_suite/K${CHUNK_SIZE}}"
SUITE_ROOT="${SUITE_ROOT:-${OUTPUT_ROOT}/ablations/${SUITE_NAME}}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-RAVE-H-Ablation}"
EVAL_SWANLAB_PROJECT="${EVAL_SWANLAB_PROJECT:-RAVE-H-Eval}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
PARALLEL_SEEDS="${PARALLEL_SEEDS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_TEST="${RUN_TEST:-0}"
DRY_RUN="${DRY_RUN:-0}"
ACTOR_TYPE="${ACTOR_TYPE:-categorical_flow}"
case "${ACTOR_TYPE}" in
  categorical_flow) BC_SUBDIR="categorical_flow_bc" ;;
  categorical) BC_SUBDIR="categorical_bc" ;;
  *) echo "ACTOR_TYPE must be categorical_flow or categorical: ${ACTOR_TYPE}" >&2; exit 2 ;;
esac
case "${ENV_NAME}" in
  KuaiEnv-v0|KuaiRand-v0) ;;
  *) echo "Unsupported ENV_NAME: ${ENV_NAME}" >&2; exit 2 ;;
esac
# 只有显式设置旧 MAC_RUN_NAME 时才回退到历史 MAC_origin 布局。
if [[ -z "${MAC_CHECKPOINT_TEMPLATE:-}" ]]; then
  if [[ -n "${MAC_RUN_NAME:-}" ]]; then
    MAC_CHECKPOINT_TEMPLATE="saved_models/${ENV_NAME}/MAC_origin/${MAC_RUN_NAME}-K{K}-seed{seed}/${BC_SUBDIR}/latest.pt"
  else
    MAC_CHECKPOINT_TEMPLATE="${OUTPUT_ROOT}/shared_pretrain/seed-{seed}/${BC_SUBDIR}/latest.pt"
  fi
fi
SHARED_MAC_CHECKPOINT="${SHARED_MAC_CHECKPOINT:-}"

read -r -a SEED_VALUES <<< "$SEEDS"
read -r -a GPU_VALUES <<< "$GPU_IDS"
read -r -a METHOD_VALUES <<< "$METHODS"

if [[ "${#SEED_VALUES[@]}" -ne 5 ]]; then
  echo "SEEDS must contain exactly five integers; got ${#SEED_VALUES[@]}: $SEEDS" >&2
  exit 2
fi
if [[ "${#GPU_VALUES[@]}" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 2
fi
if [[ "$FIXED_HORIZONS" == "all" ]]; then
  FIXED_HORIZONS="$(seq -s ' ' 1 "$CHUNK_SIZE")"
fi
read -r -a FIXED_HORIZON_VALUES <<< "$FIXED_HORIZONS"

mkdir -p "$SUITE_ROOT/logs"
MANIFEST="$SUITE_ROOT/manifest.tsv"
if [[ ! -f "$MANIFEST" ]]; then
  printf 'condition\tseed\tchunk_size\tfixed_horizon\tmac_checkpoint\toutput_dir\trun_name\tstatus\n' > "$MANIFEST"
fi

resolve_checkpoint() {
  local seed="$1"
  if [[ -n "$SHARED_MAC_CHECKPOINT" ]]; then
    printf '%s' "$SHARED_MAC_CHECKPOINT"
    return
  fi
  local path="$MAC_CHECKPOINT_TEMPLATE"
  local seed_token='{seed}'
  local chunk_token='{K}'
  path="${path//${seed_token}/${seed}}"
  path="${path//${chunk_token}/${CHUNK_SIZE}}"
  printf '%s' "$path"
}

set_condition_args() {
  local condition="$1"
  CONDITION_ARGS=()
  case "$condition" in
    rave_multiobjective_adaptive)
      CONDITION_ARGS+=(--qv_type multi_prefix_transformer)
      ;;
    single_objective_adaptive)
      # 单头 Transformer 的训练和选择目标固定为原始反馈 R。
      CONDITION_ARGS+=(--qv_type transformer)
      ;;
    fixed_h*)
      local h="${condition#fixed_h}"
      CONDITION_ARGS+=(--qv_type multi_prefix_transformer --fixed_horizon "$h")
      ;;
    *)
      echo "Unknown condition: $condition" >&2
      return 2
      ;;
  esac
}

run_one() {
  local condition="$1"
  local seed="$2"
  local gpu="$3"
  shift 3
  local -a extra_args=("$@")
  local checkpoint output_dir run_name log_file fixed_horizon=""
  checkpoint="$(resolve_checkpoint "$seed")"
  output_dir="$SUITE_ROOT/$condition/seed-$seed"
  run_name="${SUITE_NAME}-K${CHUNK_SIZE}-${condition}-seed${seed}"
  log_file="$SUITE_ROOT/logs/${condition}-seed${seed}.log"
  [[ "$condition" == fixed_h* ]] && fixed_horizon="${condition#fixed_h}"

  if [[ "$DRY_RUN" != "1" && ! -f "$checkpoint" ]]; then
    echo "Missing MAC checkpoint for seed $seed: $checkpoint" >&2
    return 3
  fi
  if [[ "$SKIP_EXISTING" == "1" && -f "$output_dir/swanlab_metrics.jsonl" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$condition" "$seed" "$CHUNK_SIZE" "$fixed_horizon" "$checkpoint" "$output_dir" "$run_name" "skipped" >> "$MANIFEST"
    echo "[skip] $condition seed=$seed"
    return
  fi

  local -a condition_args=()
  set_condition_args "$condition"
  condition_args=("${CONDITION_ARGS[@]}")
  local -a command=(bash script/run_rave_kuai.sh "${condition_args[@]}" "${extra_args[@]}")
  echo "[run] condition=$condition seed=$seed gpu=$gpu output=$output_dir"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q SEED=%q RUN_NAME=%q OUTPUT_DIR=%q MAC_CHECKPOINT=%q ' \
      "$gpu" "$seed" "$run_name" "$output_dir" "$checkpoint"
    printf '%q ' "${command[@]}"
    printf '\n'
    return
  fi

  mkdir -p "$output_dir"
  local status="complete"
  if ! CUDA_VISIBLE_DEVICES="$gpu" \
       CUDA="$gpu" \
       DEVICE="cuda:0" \
       SEED="$seed" \
       CHUNK_SIZE="$CHUNK_SIZE" \
       OUTPUT_ROOT="$OUTPUT_ROOT" \
       MAC_CHECKPOINT="$checkpoint" \
       RUN_NAME="$run_name" \
       OUTPUT_DIR="$output_dir" \
       SWANLAB_PROJECT="$SWANLAB_PROJECT" \
       SWANLAB_MODE="$SWANLAB_MODE" \
       EVAL_SPLIT="validation" \
       MODE="train" \
       "${command[@]}" >"$log_file" 2>&1; then
    status="failed"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$condition" "$seed" "$CHUNK_SIZE" "$fixed_horizon" "$checkpoint" "$output_dir" "$run_name" "$status" >> "$MANIFEST"
  if [[ "$status" == "failed" ]]; then
    echo "[failed] $condition seed=$seed; see $log_file" >&2
    return 1
  fi

  if [[ "$RUN_TEST" == "1" ]]; then
    local test_run_name="${run_name}-test"
    CUDA_VISIBLE_DEVICES="$gpu" \
    CUDA="$gpu" \
    DEVICE="cuda:0" \
    SEED="$seed" \
    CHUNK_SIZE="$CHUNK_SIZE" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    MAC_CHECKPOINT="$checkpoint" \
    RAVE_CHECKPOINT="$output_dir/latest.pt" \
    RUN_NAME="$test_run_name" \
    OUTPUT_DIR="$output_dir/test" \
    SWANLAB_PROJECT="$EVAL_SWANLAB_PROJECT" \
    SWANLAB_MODE="$SWANLAB_MODE" \
    EVAL_SPLIT="test" \
    MODE="eval" \
    "${command[@]}" >>"$log_file" 2>&1
  fi
}

conditions=()
for method in "${METHOD_VALUES[@]}"; do
  if [[ "$method" == "fixed" ]]; then
    for h in "${FIXED_HORIZON_VALUES[@]}"; do
      if (( h < 1 || h > CHUNK_SIZE )); then
        echo "Fixed horizon $h is outside [1, $CHUNK_SIZE]" >&2
        exit 2
      fi
      conditions+=("fixed_h${h}")
    done
  else
    conditions+=("$method")
  fi
done

failures=0
for condition in "${conditions[@]}"; do
  pids=()
  for index in "${!SEED_VALUES[@]}"; do
    seed="${SEED_VALUES[$index]}"
    gpu="${GPU_VALUES[$((index % ${#GPU_VALUES[@]}))]}"
    if [[ "$PARALLEL_SEEDS" == "1" ]]; then
      run_one "$condition" "$seed" "$gpu" "$@" &
      pids+=("$!")
    else
      run_one "$condition" "$seed" "$gpu" "$@" || failures=$((failures + 1))
    fi
  done
  if [[ "$PARALLEL_SEEDS" == "1" ]]; then
    for pid in "${pids[@]}"; do
      wait "$pid" || failures=$((failures + 1))
    done
  fi
done

if (( failures > 0 )); then
  echo "$failures run(s) failed; inspect $SUITE_ROOT/logs and $MANIFEST" >&2
  exit 1
fi
echo "Suite complete. Manifest: $MANIFEST"
