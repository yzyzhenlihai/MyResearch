# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
用户要求将 DARLR 实验监控从 `swanlab` 改成使用 `wandb`，用于监看实验过程。

### 1.2 任务分解
1. 检查 `examples/advance/run_DARLR.py` 当前实验记录实现。
2. 将 `swanlab` 导入和登录逻辑替换为项目已有 `load_wandb()` 工具。
3. 为 DARLR 入口补充 wandb project、mode、entity、group、tags 等命令行参数。
4. 修改 DARLR 运行脚本，默认启用 `WANDB_MODE=online`。
5. 执行静态检查和命令展开验证。

### 1.3 技术方案
复用 `src/core/util/wandb_utils.py::load_wandb()`，避免仓库本地 `wandb/` 日志目录遮蔽真实 W&B 包。`run_DARLR.py` 初始化真实 wandb run 后，现有 `src/tianshou/tianshou/trainer/base.py` 中的 epoch/test 指标日志和 `examples/policy/policy_utils.py` 中的最终结果汇总会写入同一个 active run。

## 2. 任务完成过程记录
### 2.1 代码结构
- 修改文件：`examples/advance/run_DARLR.py`
- 修改文件：`script/run_DARLR_reproduce.sh`
- 修改文件：`script/run_DARLR_KuaiRec_smoke.sh`
- 新增报告：`reports/20260519_024040_darlr_wandb_migration.md`
- 新增交接日志：`reports/agent_context/20260519_024040_darlr_wandb_migration.md`
- 更新 latest 交接入口：`reports/agent_context/latest_darlr_wandb_migration.md`

### 2.2 模块详细说明
#### 2.2.1 DARLR 入口 wandb 初始化
- 文件路径：`examples/advance/run_DARLR.py`
- 核心功能：将实验记录后端从 `swanlab` 迁移到真实 `wandb`。
- 关键函数：
  - `_get_wandb_mode()`：读取 `WANDB_MODE`。
  - `_is_wandb_disabled()`：支持 `WANDB_DISABLED=true` 或 `WANDB_MODE=disabled` 关闭记录。
  - `_to_wandb_serializable(value)`：将 `torch.device`、`Path`、列表、字典等配置转成 wandb 可接受格式。
  - `_build_default_wandb_run_name(args)`：生成包含数据集、随机种子和 DARLR 关键超参数的 run 名。
  - `set_wandb(args)`：初始化 wandb run。
  - `finish_wandb()`：安全结束 wandb run。

#### 2.2.2 DARLR 运行脚本
- 文件路径：`script/run_DARLR_reproduce.sh`
- 核心功能：默认设置 `WANDB_PROJECT=DARLR`、`WANDB_MODE=online`，并向 `run_DARLR.py` 传入 `--wandb_project` 与 `--wandb_mode`。
- 关键区域：第 6-7 行设置 W&B 环境变量，第 82-84 行传入 W&B 参数。

#### 2.2.3 DARLR smoke 脚本
- 文件路径：`script/run_DARLR_KuaiRec_smoke.sh`
- 核心功能：smoke 运行同样默认使用 W&B，便于验证最小链路时也能看到 run。
- 关键区域：第 6-7 行设置 W&B 环境变量，第 11-13 行传入 W&B 参数。

### 2.3 既有代码修改说明
- 修改文件：`examples/advance/run_DARLR.py`
- 修改区域：第 10、44、50、62-189、227-234 行附近。
- 修改原因：原代码 `import swanlab as wandb`，且包含硬编码 swanlab api key 和 `SWANLAB_MODE` 逻辑，不符合用户希望使用 W&B 的要求。
- 修改前：依赖 swanlab，`SWANLAB_MODE=disabled/offline` 控制跳过。
- 修改后：依赖真实 `wandb`，通过 `WANDB_MODE`、`WANDB_DISABLED` 和新增 CLI 参数控制。

## 3. 数据处理说明
### 3.1 数据来源
本任务未涉及新增数据处理。

### 3.2 数据预处理步骤
- 本任务未涉及。

### 3.3 数据统计信息
本任务未涉及。

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：未启动完整训练，未统计。
- 软件环境：使用 `/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python` 进行编译和导入检查。
- 超参数设置：脚本默认 `WANDB_PROJECT=DARLR`、`WANDB_MODE=online`。

### 4.2 图表结果分析
#### 4.2.1 本任务未涉及图表
- 图表类型：本任务未涉及。
- 横坐标含义：本任务未涉及。
- 纵坐标含义：本任务未涉及。
- 图例说明：本任务未涉及。
- 数据计算方式：本任务未涉及。
- 结果分析：本轮只验证监控后端迁移，不产生实验曲线。

## 5. 最终结论
已将 DARLR 实验监控从 swanlab 迁移到真实 wandb。DARLR 入口不再导入 swanlab，也不再使用硬编码 swanlab key。完整复现实验脚本和 KuaiRec smoke 脚本默认创建 W&B online run；如需关闭，可设置 `WANDB_MODE=disabled` 或 `WANDB_DISABLED=true`。

已完成验证：
- `/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python -m py_compile examples/advance/run_DARLR.py` 通过。
- `bash -n script/run_DARLR_reproduce.sh` 通过。
- `bash -n script/run_DARLR_KuaiRec_smoke.sh` 通过。
- `PYTHON_BIN=echo DATASET=KuaiEnv-v0 bash script/run_DARLR_reproduce.sh` 展开命令包含 `--wandb_project DARLR --wandb_mode online`。
- `PYTHON_BIN=echo bash script/run_DARLR_KuaiRec_smoke.sh` 展开命令包含 `--wandb_project DARLR --wandb_mode online`。
- `WANDB_MODE=disabled` 下调用 `set_wandb()` 和 `finish_wandb()` 成功返回。
- `examples/advance/run_DARLR.py` 与 `script/run_DARLR*.sh` 中不再残留 `swanlab` 或 `SWANLAB`。

## 6. 后续建议
1. 首次 online 使用前运行 `wandb login`，或设置 `WANDB_API_KEY`。
2. 如果服务器不能联网，可用 `WANDB_MODE=offline bash script/run_DARLR_reproduce.sh` 先本地记录，再用 `wandb sync` 上传。
3. 若后续希望更细粒度监控 selector loss、recommender loss 和动态 reward，可进一步在 trainer update 阶段补充更高频的 `wandb.log`。

## 7. 代码使用说明
```bash
# 在线监控，默认 project 为 DARLR
DATASET=KuaiEnv-v0 bash script/run_DARLR_reproduce.sh

# 指定 project
WANDB_PROJECT=EasyRL4Rec-DARLR DATASET=YahooEnv-v0 bash script/run_DARLR_reproduce.sh

# 离线记录
WANDB_MODE=offline DATASET=KuaiEnv-v0 bash script/run_DARLR_reproduce.sh

# 关闭 wandb
WANDB_MODE=disabled DATASET=KuaiEnv-v0 bash script/run_DARLR_reproduce.sh
```
