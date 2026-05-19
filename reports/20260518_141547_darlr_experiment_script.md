# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
用户询问“复现DQRLR实验结果的脚本怎么写”。经本地检索，当前仓库没有 `DQRLR` 命名的入口、模块或脚本，已有实现入口为 `DARLR`。因此本轮按当前仓库已复现的 `examples/advance/run_DARLR.py` 生成 DARLR 实验复现脚本。

### 1.2 任务分解
1. 检索仓库中 `DQRLR` 与 `DARLR` 的已有入口，确认可运行对象。
2. 新增一个统一 DARLR 复现实验脚本，覆盖 smoke、full、数据集切换和常用消融。
3. 对脚本做 Bash 语法检查与 dry-run 命令拼装验证。
4. 生成科研代码任务报告与 agent 交接日志。

### 1.3 技术方案
采用 Bash 环境变量驱动的统一脚本 `script/run_DARLR_reproduce.sh`。脚本默认调用 `examples/advance/run_DARLR.py`，保留 `sasrec` tracker、selector 候选用户子集、动态 reward、动态 uncertainty、selector intrinsic reward 等 DARLR v1 参数。为了适配 KuaiRand/Yahoo 等大用户数数据集，默认使用 `selector_candidate_size=512` 且 `selector_candidate_mode=embedding_topk`，避免 selector actor 在全用户空间上 softmax。

## 2. 任务完成过程记录
### 2.1 代码结构
- 新增文件：`script/run_DARLR_reproduce.sh`
- 新增报告：`reports/20260518_141547_darlr_experiment_script.md`
- 新增交接日志：`reports/agent_context/20260518_141547_darlr_experiment_script.md`
- 更新交接日志入口：`reports/agent_context/latest_darlr_experiment_script.md`

### 2.2 模块详细说明
#### 2.2.1 DARLR 复现实验脚本
- 文件路径：`script/run_DARLR_reproduce.sh`
- 核心功能：提供统一实验入口，支持最小 smoke 验证、完整训练配置、四个数据集默认参数和常用消融实验。
- 关键区域：
  - 第 16-36 行：定义 Python 解释器、数据集、模式、selector 与动态奖励相关默认参数。
  - 第 38-70 行：根据 `MODE=smoke/full` 设置训练规模。
  - 第 72-100 行：按 `DATASET` 设置 `selector_k`、`selector_candidate_size` 与 Coat/Yahoo 的 DORL 风格参数。
  - 第 102-130 行：按 `ABLATION` 切换动态 reward、动态 uncertainty 或 selector reward。
  - 第 134-189 行：组装 `run_DARLR.py` 命令数组，避免 shell 字符串拼接导致参数错位。
  - 第 197-200 行：支持 `DRY_RUN=1`，只打印命令不启动训练。
- 核心算法：本脚本不新增算法实现，只把 DARLR 复现实验所需的命令行参数进行可复用封装。

### 2.3 既有代码修改说明
本轮未修改既有 Python 算法代码。新增脚本复用已有 `examples/advance/run_DARLR.py`、`src/core/policy/darlr.py` 和 `src/core/envs/Simulated_Env/darlr_dynamic_reward.py`。

## 3. 数据处理说明
### 3.1 数据来源
本任务未涉及新增数据处理。脚本目标数据集为当前仓库已有环境：`KuaiEnv-v0`、`KuaiRand-v0`、`CoatEnv-v0`、`YahooEnv-v0`。

### 3.2 数据预处理步骤
- 本任务未涉及。

### 3.3 数据统计信息
本任务未涉及。

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：当前未执行完整训练，未统计硬件信息。
- 软件环境：脚本默认使用 `/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python`，可通过 `PYTHON_BIN` 覆盖。
- 超参数设置：`MODE=full` 默认 `epoch=100`、`step_per_epoch=100000`、`training_num=100`、`test_num=100`、`batch_size=1024`、`selector_candidate_size=512`、`lambda_uncertainty=0.05`、`lambda_entropy=0.05`。

### 4.2 图表结果分析
#### 4.2.1 本任务未涉及图表
- 图表类型：本任务未涉及。
- 横坐标含义：本任务未涉及。
- 纵坐标含义：本任务未涉及。
- 图例说明：本任务未涉及。
- 数据计算方式：本任务未涉及。
- 结果分析：本任务只验证脚本命令拼装，不产生论文指标或图表。

## 5. 最终结论
已新增 `script/run_DARLR_reproduce.sh` 作为 DARLR 实验复现脚本。由于仓库中没有 `DQRLR` 命名实现，本轮按现有 DARLR 代码入口处理。脚本支持 smoke/full 两种运行规模，支持 KuaiRec、KuaiRand、Coat、Yahoo 四个数据集，支持 `full`、`static_reward`、`no_uncertainty`、`selector_base`、`selector_sim`、`selector_div` 等对照消融。

已完成的验证：
- `bash -n script/run_DARLR_reproduce.sh` 通过。
- `DRY_RUN=1 MODE=smoke DATASET=KuaiEnv-v0 bash script/run_DARLR_reproduce.sh` 成功打印 KuaiRec smoke 命令。
- `DRY_RUN=1 MODE=full DATASET=YahooEnv-v0 ABLATION=no_uncertainty bash script/run_DARLR_reproduce.sh` 成功打印 Yahoo full 消融命令。

未执行完整训练，因此本报告不声称已复现论文数值结果。

## 6. 后续建议
1. 先运行 `MODE=smoke DATASET=KuaiEnv-v0 bash script/run_DARLR_reproduce.sh`，确认本机数据、DeepFM user model 和 DARLR 训练链路可用。
2. 再运行 `MODE=full DATASET=KuaiEnv-v0 CUDA=0 bash script/run_DARLR_reproduce.sh` 做 KuaiRec 主实验。
3. 对 Yahoo/KuaiRand 保持 `selector_candidate_size`，不要直接全用户 softmax；必要时把 `SELECTOR_CANDIDATE_SIZE` 从 512 调到 1024 或 2048 做敏感性实验。
4. 论文级复现需要追加多随机种子、DORL baseline、dynamic reward/uncertainty/selector reward 消融，并汇总累计奖励、单步奖励、轨迹长度和 MCD。

## 7. 代码使用说明
常用命令如下：

```bash
# 只打印命令，不启动训练
DRY_RUN=1 MODE=smoke DATASET=KuaiEnv-v0 bash script/run_DARLR_reproduce.sh

# 最小链路验证
MODE=smoke DATASET=KuaiEnv-v0 bash script/run_DARLR_reproduce.sh

# KuaiRec 完整 DARLR 主实验
MODE=full DATASET=KuaiEnv-v0 CUDA=0 bash script/run_DARLR_reproduce.sh

# Yahoo 动态 uncertainty 消融
MODE=full DATASET=YahooEnv-v0 ABLATION=no_uncertainty CUDA=0 bash script/run_DARLR_reproduce.sh

# KuaiRand selector 候选池调大
MODE=full DATASET=KuaiRand-v0 SELECTOR_CANDIDATE_SIZE=1024 CUDA=0 bash script/run_DARLR_reproduce.sh
```

可覆盖的关键环境变量包括：`PYTHON_BIN`、`DATASET`、`MODE`、`ABLATION`、`SEED`、`CUDA`、`EPOCH`、`STEP_PER_EPOCH`、`TRAINING_NUM`、`TEST_NUM`、`SELECTOR_K`、`SELECTOR_CANDIDATE_SIZE`、`SELECTOR_CANDIDATE_MODE`、`LAMBDA_UNCERTAINTY`、`LAMBDA_ENTROPY`、`MESSAGE`。
