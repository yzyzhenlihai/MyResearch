# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
用户发现 `DORL_DOSER_ONPOLICY` 训练过程没有记录到 SwanLab，要求将日志逻辑从 `wandb` 迁移成 `swanlab`。

### 1.2 任务分解
1. 检查当前训练日志和入口代码，确认未记录 SwanLab 的原因。
2. 将 `examples/our_model/dorl_doser.py` 中的日志初始化、关闭和配置序列化从 `wandb` 迁移为 `swanlab`。
3. 更新 `examples/our_model/dorl_doser_onpolicy.py` 的导入、CLI 参数和主流程调用。
4. 更新 `script/run_DORL_DOSER_onpolicy.sh`，默认使用 `SWANLAB_MODE=online` 并传递 `--swanlab_mode`。
5. 保留 `--wandb_*` 旧参数别名，避免旧脚本无法解析。
6. 执行静态编译、参数解析和 disabled 模式初始化验证。

### 1.3 技术方案
采用直接迁移而非新增并行日志系统：`DORL_DOSER` 入口现在直接 `import swanlab`，用 `set_swanlab()` / `finish_swanlab()` 管理实验记录。由于仓库的 `policy_utils` 和 Tianshou trainer 历史上通过名为 `wandb` 的模块变量写指标，新增 `_activate_swanlab_metric_logger()` 在 SwanLab 初始化后将这些模块变量绑定到 `swanlab`，确保 epoch/test 指标和 final 指标也能进入 SwanLab。

## 2. 任务完成过程记录
### 2.1 代码结构
- 修改文件：`examples/our_model/dorl_doser.py`
- 修改文件：`examples/our_model/dorl_doser_onpolicy.py`
- 修改文件：`script/run_DORL_DOSER_onpolicy.sh`

### 2.2 模块详细说明
#### 2.2.1 SwanLab 初始化与兼容层
- 文件路径：`examples/our_model/dorl_doser.py`
- 核心功能：将 DORL-DOSER 的在线日志后端迁移为 SwanLab。
- 关键函数：
  - `_get_swanlab_mode()`：读取 `SWANLAB_MODE`，兼容旧 `WANDB_MODE`。
  - `_is_swanlab_disabled()`：读取 `SWANLAB_DISABLED`，兼容旧 `WANDB_DISABLED`。
  - `_to_swanlab_serializable(value)`：把配置和 metadata 转成 SwanLab 可接受的 JSON 风格对象。
  - `_activate_swanlab_metric_logger()`：把 `policy_utils`、`tianshou.trainer.base` 和 `src.core.policy.doser` 中历史名为 `wandb` 的模块变量绑定到 `swanlab`。
  - `set_swanlab(args, run_metadata)`：初始化 SwanLab run，并写入训练配置和 diffusion metadata。
  - `finish_swanlab()`：安全结束 SwanLab run。
  - `set_wandb()` / `finish_wandb()`：兼容旧调用名，内部转调 SwanLab。

#### 2.2.2 On-policy 训练入口迁移
- 文件路径：`examples/our_model/dorl_doser_onpolicy.py`
- 核心功能：从共享模块导入 `DEFAULT_SWANLAB_PROJECT`、`set_swanlab` 和 `finish_swanlab`，主训练流程改为 SwanLab 初始化和关闭。
- 参数兼容：新增 `--swanlab_project`、`--swanlab_run_name`、`--swanlab_mode` 等参数，同时保留 `--wandb_project`、`--wandb_run_name`、`--wandb_mode` 等旧参数别名，统一写入 `args.swanlab_*`。

#### 2.2.3 训练运行脚本迁移
- 文件路径：`script/run_DORL_DOSER_onpolicy.sh`
- 核心功能：默认导出 `SWANLAB_MODE=online`，并将训练入口参数从 `--wandb_mode` 改为 `--swanlab_mode`。
- 注意事项：脚本中 `CUDA=1` 是当前工作树已有改动，本轮未将其改回默认值。

### 2.3 既有代码修改说明
- 修改文件：`examples/our_model/dorl_doser.py`
- 修改区域：日志初始化区域，约第 48-231 行；CLI 日志参数区域，约第 261-291 行；主流程调用区域，约第 668-684 行。
- 修改原因：训练过程需要记录到 SwanLab，而不是继续使用 `wandb` 兼容层。
- 修改前：通过 `load_wandb()` 加载 W&B，`WANDB_MODE=disabled` 时跳过初始化，trainer 指标不会进入 SwanLab。
- 修改后：直接导入 `swanlab`，通过 `set_swanlab()` 初始化，并绑定 trainer/policy_utils 指标后端。

- 修改文件：`examples/our_model/dorl_doser_onpolicy.py`
- 修改区域：导入区域，约第 24-31 行；参数区域，约第 95-122 行；主流程调用区域，约第 300-316 行。
- 修改原因：on-policy 主入口需要复用 SwanLab 初始化/关闭逻辑。
- 修改前：导入和调用 `set_wandb()` / `finish_wandb()`。
- 修改后：导入和调用 `set_swanlab()` / `finish_swanlab()`。

- 修改文件：`script/run_DORL_DOSER_onpolicy.sh`
- 修改区域：环境变量和最终训练参数，约第 6-8 行、第 138 行。
- 修改原因：默认运行时不再通过 `WANDB_MODE` 控制日志，而是通过 `SWANLAB_MODE` 控制 SwanLab。
- 修改前：默认导出 `WANDB_MODE`，并传入 `--wandb_mode`。
- 修改后：默认导出 `SWANLAB_MODE=online`，并传入 `--swanlab_mode`。

### 2.4 未触碰的工作树变更
当前工作树中已有 `results/ood_granularity_experiment_smoke/*` 删除记录，这些文件与本任务无关，本轮未修改或恢复。

## 3. 数据处理说明
### 3.1 数据来源
本任务未涉及数据处理。

### 3.2 数据预处理步骤
本任务未涉及。

### 3.3 数据统计信息
本任务未涉及。

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：当前环境未知。
- 软件环境：使用 `conda run -n easyrl4rec` 执行静态编译和导入验证；环境中已安装 `swanlab`。
- 超参数设置：本任务未启动完整训练，仅验证日志初始化和参数解析。

### 4.2 图表结果分析
本任务未涉及图表生成。

### 4.3 验证记录
- `bash -n script/run_DORL_DOSER_onpolicy.sh`：通过。
- `conda run -n easyrl4rec python -m py_compile examples/our_model/dorl_doser.py examples/our_model/dorl_doser_onpolicy.py`：通过。
- 导入验证：成功导入 `set_swanlab`、`finish_swanlab`、`DEFAULT_SWANLAB_PROJECT` 和 on-policy 参数解析函数。
- 新参数解析验证：`--swanlab_mode disabled --swanlab_project P1` 解析为 `args.swanlab_mode=disabled`、`args.swanlab_project=P1`。
- 旧别名解析验证：`--wandb_mode disabled --wandb_project P2` 解析为 `args.swanlab_mode=disabled`、`args.swanlab_project=P2`。
- disabled 初始化 smoke：`set_swanlab()` 在 `swanlab_mode=disabled` 下安全跳过，输出 `swanlab disabled smoke ok`。
- 未执行完整训练：避免在当前交互中触发长时间训练；SwanLab online 实际上传需下次训练启动后确认。

## 5. 最终结论
已将 DORL-DOSER / DORL-DOSER-ONPOLICY 的日志初始化从 `wandb` 迁移到 `swanlab`。新的训练脚本默认 `SWANLAB_MODE=online`，并传入 `--swanlab_mode`。同时保留旧 `--wandb_*` 参数别名和 `set_wandb()` / `finish_wandb()` 兼容函数。静态编译、脚本语法、参数解析和 disabled 模式初始化均已通过验证。

## 6. 后续建议
1. 需要重启训练进程，当前已经启动的训练不会 retroactively 记录到 SwanLab。
2. 正式运行前确认 SwanLab 已登录，或设置可用的 SwanLab API 配置。
3. 如需关闭记录，可使用 `SWANLAB_MODE=disabled bash script/run_DORL_DOSER_onpolicy.sh`。
4. 若训练启动后只有 run 没有指标曲线，优先检查 trainer 模块是否在 `set_swanlab()` 初始化之后才被调用；本轮已加入模块变量绑定以降低该风险。

## 7. 代码使用说明
- 默认开启 SwanLab online：
  ```bash
  bash script/run_DORL_DOSER_onpolicy.sh
  ```
- 显式开启 SwanLab online：
  ```bash
  SWANLAB_MODE=online bash script/run_DORL_DOSER_onpolicy.sh
  ```
- 关闭 SwanLab：
  ```bash
  SWANLAB_MODE=disabled bash script/run_DORL_DOSER_onpolicy.sh
  ```
- 兼容旧参数：
  ```bash
  python examples/our_model/dorl_doser_onpolicy.py --env KuaiEnv-v0 --wandb_mode disabled
  ```
