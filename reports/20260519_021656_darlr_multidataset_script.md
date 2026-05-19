# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
用户要求将 `run_DARLR_reproduce.sh` 改写成类似 `run_DARLR_KuaiRec_smoke.sh` 的直接运行脚本，不需要包含消融，同时需要“加上所有模块”，并进一步明确不同数据集也要支持。

### 1.2 任务分解
1. 保留脚本直接调用 `examples/advance/run_DARLR.py` 的形式。
2. 删除 `MODE`、`ABLATION`、`DRY_RUN` 等调度和消融逻辑。
3. 增加轻量 `DATASET` 开关，支持 `KuaiEnv-v0`、`KuaiRand-v0`、`CoatEnv-v0`、`YahooEnv-v0`。
4. 显式写入 DARLR v1 的 selector、动态奖励、动态不确定性、entropy、SASRec tracker 等模块参数。
5. 做 Bash 语法检查和多数据集命令展开验证。

### 1.3 技术方案
脚本采用“固定主命令 + 数据集参数分支”的 Bash 写法。主体保持 smoke 脚本的一条命令风格；数据集差异只在前置 `case "${DATASET}"` 中设置 `selector_k`、`selector_candidate_size`、`feature_level`、`lr` 和 `ent-coef`，避免把脚本重新变成复杂实验调度器。

## 2. 任务完成过程记录
### 2.1 代码结构
- 修改文件：`script/run_DARLR_reproduce.sh`
- 新增报告：`reports/20260519_021656_darlr_multidataset_script.md`
- 新增交接日志：`reports/agent_context/20260519_021656_darlr_multidataset_script.md`
- 新增 latest 交接入口：`reports/agent_context/latest_darlr_multidataset_script.md`

### 2.2 模块详细说明
#### 2.2.1 多数据集 DARLR 复现实验脚本
- 文件路径：`script/run_DARLR_reproduce.sh`
- 核心功能：以固定命令运行 DARLR 完整模块实验，同时通过 `DATASET` 环境变量切换数据集。
- 关键区域：
  - 第 9-20 行：训练规模、数据集和评估长度默认值。
  - 第 22-40 行：tracker、selector、动态奖励和优化器超参数默认值。
  - 第 42-73 行：四个数据集的参数分支。
  - 第 80-126 行：直接调用 `examples/advance/run_DARLR.py`，显式启用所有 DARLR v1 主要模块。
- 核心算法：本任务未新增算法，只调整复现实验命令入口。

### 2.3 既有代码修改说明
- 修改文件：`script/run_DARLR_reproduce.sh`
- 修改区域：全文件重写。
- 修改原因：旧版本是通用调度脚本，包含 `MODE`、`ABLATION`、`DRY_RUN` 和命令数组组装；用户希望改为接近 smoke 脚本的直接执行形式，但仍支持不同数据集。
- 修改前：支持 smoke/full 和多种 ablation，脚本结构偏调度器。
- 修改后：不包含消融逻辑，固定运行 DARLR all-modules 配置；通过 `DATASET=...` 切换数据集。

## 3. 数据处理说明
### 3.1 数据来源
本任务未新增数据处理。脚本使用仓库已有环境数据：KuaiRec、KuaiRand、Coat、Yahoo。

### 3.2 数据预处理步骤
- 本任务未涉及。

### 3.3 数据统计信息
本任务未涉及。

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：未启动训练，未统计。
- 软件环境：脚本默认 Python 为 `/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python`，可由 `PYTHON_BIN` 覆盖。
- 超参数设置：默认 `epoch=100`、`step_per_epoch=100000`、`training_num=100`、`test_num=100`、`selector_candidate_size=512`、`selector_candidate_mode=embedding_topk`。

### 4.2 图表结果分析
#### 4.2.1 本任务未涉及图表
- 图表类型：本任务未涉及。
- 横坐标含义：本任务未涉及。
- 纵坐标含义：本任务未涉及。
- 图例说明：本任务未涉及。
- 数据计算方式：本任务未涉及。
- 结果分析：本轮只验证脚本语法和参数展开，未产生实验曲线或论文指标。

## 5. 最终结论
`script/run_DARLR_reproduce.sh` 已改为直接运行式脚本，不再包含消融分支。脚本支持 `DATASET=KuaiEnv-v0`、`DATASET=KuaiRand-v0`、`DATASET=CoatEnv-v0`、`DATASET=YahooEnv-v0`，并显式传入 selector、候选用户子集、selector reward full、dynamic reward、dynamic uncertainty、DORL entropy、SASRec state tracker 等 DARLR v1 主要模块参数。

已完成验证：
- `bash -n script/run_DARLR_reproduce.sh` 通过。
- `PYTHON_BIN=echo DATASET=KuaiEnv-v0 bash script/run_DARLR_reproduce.sh` 参数展开通过。
- `PYTHON_BIN=echo DATASET=KuaiRand-v0 bash script/run_DARLR_reproduce.sh` 参数展开通过。
- `PYTHON_BIN=echo DATASET=CoatEnv-v0 bash script/run_DARLR_reproduce.sh` 参数展开通过。
- `PYTHON_BIN=echo DATASET=YahooEnv-v0 bash script/run_DARLR_reproduce.sh` 参数展开通过。

## 6. 后续建议
1. 先用已有 `script/run_DARLR_KuaiRec_smoke.sh` 验证最小链路。
2. 再用 `DATASET=KuaiEnv-v0 bash script/run_DARLR_reproduce.sh` 跑主实验。
3. 对 KuaiRand/Yahoo 可根据显存与速度调整 `SELECTOR_CANDIDATE_SIZE`，但初版不建议全用户 softmax。
4. 若后续要做论文表格，再单独新增批量消融脚本，避免污染主复现脚本。

## 7. 代码使用说明
```bash
# KuaiRec 主实验
DATASET=KuaiEnv-v0 bash script/run_DARLR_reproduce.sh

# KuaiRand 主实验
DATASET=KuaiRand-v0 bash script/run_DARLR_reproduce.sh

# Coat 主实验
DATASET=CoatEnv-v0 bash script/run_DARLR_reproduce.sh

# Yahoo 主实验
DATASET=YahooEnv-v0 bash script/run_DARLR_reproduce.sh

# 覆盖候选池大小或训练轮数
DATASET=YahooEnv-v0 SELECTOR_CANDIDATE_SIZE=1024 EPOCH=50 bash script/run_DARLR_reproduce.sh
```
