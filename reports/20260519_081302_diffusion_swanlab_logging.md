# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
用户要求将 `examples/diffusion/pretrain_diffusion.py` 中扩散模型训练日志展示修改为 SwanLab，使行为扩散模型和状态扩散模型训练过程中的 loss 等指标可以通过 SwanLab 记录和展示。

### 1.2 任务分解
1. 检查仓库中已有 SwanLab 使用方式，确认项目约定。
2. 修改 diffusion 预训练脚本中的实验日志后端，从动态导入 `wandb` 切换为 `swanlab`。
3. 保留旧命令行参数的兼容别名，避免已有脚本直接失效。
4. 运行语法检查、参数兼容检查和小样本 smoke test。
5. 生成科研代码任务报告与 agent 交接日志。

### 1.3 技术方案
沿用当前脚本的 `ExperimentTracker` 抽象，只替换其内部后端实现。新增 `--enable_swanlab`、`--swanlab_project`、`--swanlab_run_name`、`--swanlab_mode` 参数；保留 `--enable_wandb` 和 `--wandb_project` 作为兼容别名，实际映射到 SwanLab 参数。训练循环仍通过 `tracker.log(...)` 记录 `state_diffusion/loss`、`behavior_diffusion/loss` 等指标，因此不改变扩散训练数学逻辑。

## 2. 任务完成过程记录
### 2.1 代码结构
- 修改文件：`examples/diffusion/pretrain_diffusion.py`
- 新增报告文件：`reports/20260519_081302_diffusion_swanlab_logging.md`
- 新增交接日志：`reports/agent_context/20260519_081302_diffusion_swanlab_logging.md` 与 `reports/agent_context/latest_diffusion_swanlab_logging.md`

### 2.2 模块详细说明
#### 2.2.1 SwanLab 日志跟踪器
- 文件路径：`examples/diffusion/pretrain_diffusion.py`
- 修改区域：约第 59 行、第 130-207 行
- 核心功能：将可选实验日志后端从 `wandb` 改为 `swanlab`。
- 关键函数/类：
  - `ExperimentTracker.__init__(...)`：按需导入 `swanlab` 并调用 `swanlab.init(project=..., name=..., config=...)`。
  - `ExperimentTracker.log(...)`：调用 `swanlab.log(metrics)` 上传训练标量。
  - `ExperimentTracker.finish()`：调用 `swanlab.finish()` 安全结束运行。
- 核心算法：本模块不涉及算法变更，只替换训练指标上报后端。

#### 2.2.2 CLI 与训练入口适配
- 文件路径：`examples/diffusion/pretrain_diffusion.py`
- 修改区域：约第 1607-1611 行、第 1738-1761 行
- 核心功能：训练入口使用 SwanLab 参数初始化 tracker，并保留旧参数兼容。
- 关键参数：
  - `--enable_swanlab`：启用 SwanLab 日志。
  - `--swanlab_project`：设置 SwanLab project，默认 `DOSER-Diffusion`。
  - `--swanlab_run_name`：自定义 SwanLab run 名称。
  - `--swanlab_mode`：可选设置 `SWANLAB_MODE`，例如 `disabled` 或 `offline`。
  - `--enable_wandb` / `--wandb_project`：兼容旧参数名，实际映射到 SwanLab。

### 2.3 既有代码修改说明
- 修改前：`ExperimentTracker` 在启用时尝试 `import wandb`，参数为 `--enable_wandb` 和 `--wandb_project`。
- 修改后：`ExperimentTracker` 在启用时尝试 `import swanlab`，参数为 `--enable_swanlab` 和 `--swanlab_project`，旧 wandb 参数仅作为别名保留。
- 修改原因：用户要求 diffusion 训练日志展示切换为 SwanLab，与仓库中 DARLR/DORL/MOPO 等训练入口的日志后端保持一致。
- 额外说明：如果传入 `--swanlab_mode disabled` 或环境变量 `SWANLAB_MODE=disabled`，脚本会跳过 SwanLab 初始化并只输出控制台日志，便于 smoke test 或无网络环境运行。

## 3. 数据处理说明
### 3.1 数据来源
本任务未修改数据读取逻辑。smoke test 仍使用 `data/KuaiRec/data_processed/DM_KuaiEnv-v0_small_data.pkl` 的小样本切片。

### 3.2 数据预处理步骤
本任务未新增数据预处理步骤。验证时使用 `--max_trajectories 2 --max_transitions 8` 限制样本量。

### 3.3 数据统计信息
smoke test 加载日志显示：`users=2`、`behavior_transitions=8`、`state_transitions=8`、`state_dim=42`、`action_dim=41`。

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：CPU smoke test。
- 软件环境：`conda run -n easyrl4rec`；已确认 `swanlab 0.7.2` 可用。
- 验证命令：
  - `conda run -n easyrl4rec python -m py_compile examples/diffusion/pretrain_diffusion.py`
  - `conda run -n easyrl4rec python -c "from examples.diffusion.pretrain_diffusion import parse_args; ..."`
  - `conda run -n easyrl4rec python examples/diffusion/pretrain_diffusion.py --device cpu --pretrain_epochs 1 --batch_size 4 --max_trajectories 2 --max_transitions 8 --save_root /tmp/easyrl4rec_diffusion_swanlab_smoke --overwrite --enable_swanlab --swanlab_mode disabled`

### 4.2 图表结果分析
#### 4.2.1 本任务未涉及图表
- 图表类型：本任务未生成图表。
- 横坐标含义：本任务未涉及。
- 纵坐标含义：本任务未涉及。
- 图例说明：本任务未涉及。
- 数据计算方式：本任务未涉及。
- 结果分析：本任务只验证日志后端切换和训练流程可运行，未分析完整训练曲线。

## 5. 最终结论
已将 diffusion 预训练脚本的实验日志后端从 wandb 切换为 SwanLab。启用 `--enable_swanlab` 后，训练循环会通过 SwanLab 记录扩散模型 loss 和 step 指标；旧的 `--enable_wandb`、`--wandb_project` 参数仍可作为兼容别名使用。语法检查、旧参数兼容检查和小样本 smoke test 均已通过。

## 6. 后续建议
1. 全量训练时直接添加 `--enable_swanlab`，并按需指定 `--swanlab_project` 与 `--swanlab_run_name`。
2. 在无网络或调试环境中使用 `--swanlab_mode disabled` 跳过 SwanLab 初始化。
3. 若希望本地离线记录，可尝试 `--swanlab_mode offline`，但本次未在仓库内生成离线 SwanLab run，以避免引入额外运行目录。

## 7. 代码使用说明
启用 SwanLab 的运行示例：

```bash
conda run -n easyrl4rec python examples/diffusion/pretrain_diffusion.py \
  --env_name KuaiEnv-v0 \
  --dataset_path data/KuaiRec/data_processed/DM_KuaiEnv-v0_small_data.pkl \
  --save_root saved_models \
  --device 0 \
  --pretrain_epochs 100000 \
  --batch_size 256 \
  --enable_swanlab \
  --swanlab_project DOSER-Diffusion
```

兼容旧参数的运行示例：

```bash
conda run -n easyrl4rec python examples/diffusion/pretrain_diffusion.py \
  --enable_wandb \
  --wandb_project DOSER-Diffusion
```
