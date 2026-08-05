#!/usr/bin/env bash

: <<'COMMENT'
| 参数 | 默认值 | 含义 | 建议 |
|---|---:|---|---|
| `DEVICE` | `cuda:0` | 使用的计算设备 | 指定空闲 GPU，如 `cuda:2` |
| `EVAL_EPISODES` | 100 | 每个 K 评估的独立轨迹数 | 调试 5～20；正式 200～500 |
| `NUM_SAMPLES_TEST` | 32 | MAC 每次规划采样的候选 chunk 数 | 必须与原结果评估设置一致，当前建议 32 |
| `MAX_CHUNKS_PER_EPISODE` | 0 | 每条主轨迹最多诊断多少个 chunk；0 为全部 | 调试 1～5；正式用 0 |
| `BOOTSTRAP_SAMPLES` | 2000 | 置信区间 bootstrap 次数 | 调试 100；正式 2000～5000 |
| `SEED` | 2023 | 环境采样、主策略与配对公共随机数种子 | 多 seed 正式实验逐次修改 |
| `OUTPUT_DIR` | `results_analysis/replanning_motivation_long_term` | 结果输出目录 | 不要与旧的局部回报结果混用 |
| `K3_CKPT` | 默认 K3 checkpoint | K=3 模型路径 | 使用生成表格结果的对应 checkpoint |
| `K5_CKPT` | 默认 K5 checkpoint | K=5 模型路径 | 同上 |
| `K7_CKPT` | 默认 K7 checkpoint | K=7 模型路径 | 同上 |
| `PYTHON_BIN` | easyrl4rec 环境 | Python 解释器 | 通常不需要修改 |
COMMENT


export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-easyrl4rec}"
export PYTHONPATH="${PWD}:${PWD}/src:${PWD}/src/DeepCTR-Torch:${PWD}/src/tianshou:${PWD}/examples/policy:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python}"
DEVICE="${DEVICE:-cuda:6}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
NUM_SAMPLES_TEST="${NUM_SAMPLES_TEST:-32}"
MAX_CHUNKS_PER_EPISODE="${MAX_CHUNKS_PER_EPISODE:-0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
SEED="${SEED:-2023}"
OUTPUT_DIR="${OUTPUT_DIR:-results_analysis/replanning_motivation_long_term}"

K3_CKPT="${K3_CKPT:-saved_models/KuaiEnv-v0/DORL_MAC/dorl-mac-kuai-catbc-qv-K3/mac_agent/latest.pt}"
K5_CKPT="${K5_CKPT:-saved_models/KuaiEnv-v0/DORL_MAC/dorl-mac-kuai-catbc-qv-K5/mac_agent/latest.pt}"
K7_CKPT="${K7_CKPT:-saved_models/KuaiEnv-v0/DORL_MAC/dorl-mac-kuai-catbc-qv-K7/mac_agent/latest.pt}"

printf '[run_replanning_motivation] K=(3,5,7), episodes=%s, device=%s\n' \
  "${EVAL_EPISODES}" "${DEVICE}"
printf '[run_replanning_motivation] output=%s\n' "${OUTPUT_DIR}"

"${PYTHON_BIN}" examples/our_model/runners/eval_replanning_motivation.py \
  --mac-ckpts \
  "3=${K3_CKPT}" \
  "5=${K5_CKPT}" \
  "7=${K7_CKPT}" \
  --device "${DEVICE}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --num-samples-test "${NUM_SAMPLES_TEST}" \
  --max-chunks-per-episode "${MAX_CHUNKS_PER_EPISODE}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}"

printf '[run_replanning_motivation] done: %s\n' "${OUTPUT_DIR}"
