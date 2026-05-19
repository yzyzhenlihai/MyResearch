# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
用户要求给出执行训练的运行脚本，并将脚本保存到 `script` 文件夹下。

### 1.2 任务分解
1. 查看仓库中 `script/` 与 `scripts/` 目录现有脚本风格。
2. 新增 DORL-DOSER on-policy 训练 shell 脚本。
3. 支持常用训练参数通过环境变量覆盖，并保留 smoke 模式。
4. 给脚本添加可执行权限并执行 bash 语法检查。

### 1.3 技术方案
采用 Bash 脚本封装 `examples/our_model/dorl_doser_onpolicy.py` 训练入口。默认使用 `easyrl4rec` 环境 Python，默认关闭 wandb，默认运行 KuaiEnv-v0，并读取当前本地已有的 `pointneg` 用户模型和 `DM_KuaiEnv-v0_small_data` diffusion artifact。通过环境变量支持切换数据集、GPU、训练轮数、diffusion artifact、DOSER 超参数等。

## 2. 任务完成过程记录
### 2.1 代码结构
- 新增文件：`script/run_DORL_DOSER_onpolicy.sh`

### 2.2 模块详细说明
#### 2.2.1 DORL-DOSER on-policy 训练脚本
- 文件路径：`script/run_DORL_DOSER_onpolicy.sh`
- 核心功能：封装 DORL-DOSER on-policy 训练命令，便于直接执行或通过环境变量配置实验。
- 关键配置：
  - `DATASET`：默认 `KuaiEnv-v0`，支持 `KuaiRand-v0`、`YahooEnv-v0`、`CoatEnv-v0`。
  - `PYTHON_BIN`：默认 `/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python`。
  - `DIFFUSION_ARTIFACT_NAME`：默认 `DM_KuaiEnv-v0_small_data`；显式设置为空字符串时使用 flat artifact 默认目录。
  - `SMOKE=1`：启用极小配置，仅用于接口冒烟验证。
  - `CPU_FLAG=1`：追加 `--cpu`。
- 核心算法：本任务仅新增运行脚本，不涉及算法实现变化。

### 2.3 既有代码修改说明
本任务未修改既有代码，仅新增脚本文件。

## 3. 数据处理说明
### 3.1 数据来源
本任务未新增数据处理流程。脚本默认训练环境为 `KuaiEnv-v0`，依赖项目已有数据、用户模型参数和 diffusion artifact。

### 3.2 数据预处理步骤
本任务未涉及。

### 3.3 数据统计信息
本任务未涉及。

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：当前环境未知。
- 软件环境：脚本默认使用 `easyrl4rec` conda 环境中的 Python。
- 超参数设置：脚本默认 `EPOCH=100`、`STEP_PER_EPOCH=100000`、`TRAINING_NUM=100`、`TEST_NUM=100`、`BATCH_SIZE=256`、`DOSER_BETA=0.001`、`DOSER_LAM=0.001`、`DOSER_ETA=0.9`。

### 4.2 图表结果分析
本任务未涉及。

### 4.3 验证记录
- 已执行：`bash -n script/run_DORL_DOSER_onpolicy.sh`
- 结果：语法检查通过。
- 已执行：`chmod +x script/run_DORL_DOSER_onpolicy.sh`
- 结果：脚本已设置为可执行。
- 未执行完整训练：避免在当前交互中触发长时间训练任务。

## 5. 最终结论
已在 `script/` 文件夹下新增 DORL-DOSER on-policy 训练运行脚本 `run_DORL_DOSER_onpolicy.sh`，脚本可直接执行，也可通过环境变量调整数据集、设备、训练规模和 DOSER 超参数。脚本语法检查通过。

## 6. 后续建议
1. 若只验证链路，可先执行 `SMOKE=1 CPU_FLAG=1 bash script/run_DORL_DOSER_onpolicy.sh`。
2. 若使用新格式 diffusion flat artifact，可执行 `DIFFUSION_ARTIFACT_NAME="" bash script/run_DORL_DOSER_onpolicy.sh`。
3. 正式训练前确认对应环境的用户模型参数和 diffusion artifact 已存在。

## 7. 代码使用说明
- 默认训练：
  ```bash
  bash script/run_DORL_DOSER_onpolicy.sh
  ```
- CPU 冒烟测试：
  ```bash
  SMOKE=1 CPU_FLAG=1 bash script/run_DORL_DOSER_onpolicy.sh
  ```
- 指定 GPU 和训练轮数：
  ```bash
  CUDA=1 EPOCH=100 STEP_PER_EPOCH=100000 bash script/run_DORL_DOSER_onpolicy.sh
  ```
- 使用新格式 flat diffusion artifact：
  ```bash
  DIFFUSION_ARTIFACT_NAME="" bash script/run_DORL_DOSER_onpolicy.sh
  ```
