# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
用户要求直接修改 `examples/diffusion/pretrain_diffusion.py`，使 DOSER 原论文中的扩散模型预训练代码适配 `DM_KuaiEnv-v0_small_data.pkl`。数据以 `user_id` 为单位组织，每条轨迹包含 `actions`、`terminals`、`rewards`、`observations`、`next_observations`、`user_id`。本阶段只预训练行为扩散模型和状态扩散模型，不训练 dynamic model；行为扩散学习 `p(a | s)`，状态扩散学习 `p(s_next | s, a)`；训练产物默认保存到 `saved_models/KuaiEnv-v0/`，并预留后续数据集接口。

### 1.2 任务分解
1. 检查现有 `pretrain_diffusion.py` 的数据加载、模型构建、训练和保存逻辑。
2. 将数据集样本从 `(state, action)` 扩展为 `(state, action, next_state)`，并兼容 terminal 过滤。
3. 将状态扩散模型改为条件下一状态扩散模型，训练目标为 `next_observations`。
4. 更新保存路径、文件名、配置与指标输出。
5. 使用 `easyrl4rec` conda 环境进行语法检查、全量数据加载检查和小样本 smoke test。

### 1.3 技术方案
采用项目现有 `DiffusionModel` 与 `ScoreNetwork`，不新增外部依赖。行为模型保持条件扩散形式，输入目标为 `actions`、条件为 `observations`；状态模型改为条件扩散形式，输入目标为 `next_observations`、条件为 `concat(observations, actions)`。轨迹数据通过统一 `DatasetBundle` 暴露行为数据集与状态数据集，其中状态数据集支持跳过 terminal transition。默认保存目录为 `saved_models/<env_name>/`，可通过 `--artifact_name` 额外创建子目录。

## 2. 任务完成过程记录
### 2.1 代码结构
- 修改文件：`examples/diffusion/pretrain_diffusion.py`
- 新增报告文件：`reports/20260519_080226_kuai_diffusion_pretrain.md`
- 新增交接日志：`reports/agent_context/20260519_080226_kuai_diffusion_pretrain.md` 与 `reports/agent_context/latest_kuai_diffusion_pretrain.md`

### 2.2 模块详细说明
#### 2.2.1 数据集适配模块
- 文件路径：`examples/diffusion/pretrain_diffusion.py`
- 修改区域：约第 89-126 行、第 202-493 行、第 739-899 行
- 核心功能：将 KuaiEnv 轨迹 pickle 解析为 transition 级样本，并同时提供行为扩散和状态扩散训练视图。
- 关键函数/类：
  - `DatasetBundle`：新增 `state_dataset` 与 `num_state_transitions`，区分行为扩散和状态扩散样本数。
  - `ArrayTransitionDataset`：支持 `(state, action, next_state)`，兼容 D4RL 后端。
  - `TrajectoryTransitionDataset`：读取用户级轨迹，使用前缀和索引展开 transition，并支持 terminal 过滤。
  - `load_trajectory_dataset_bundle(...)`：构造行为数据集和状态数据集。
- 核心算法：对每条用户轨迹校验字段、长度和维度；若动作是一维离散 id，则 reshape 为 `(N, 1)`；状态扩散数据集可跳过 terminal transition；全量轨迹使用前缀和索引，避免为 467 万条 transition 构造巨大的 Python tuple 列表。

#### 2.2.2 扩散模型训练模块
- 文件路径：`examples/diffusion/pretrain_diffusion.py`
- 修改区域：约第 919-1494 行、第 1514-1693 行
- 核心功能：训练行为扩散模型和条件状态扩散模型，并分别计算 OOD 阈值。
- 关键函数：
  - `build_state_diffusion_model(state_dim, action_dim, device)`：构造 `p(s_next | s, a)` 条件扩散网络。
  - `prepare_diffusion_batch(batch, model_kind, device)`：按模型类型生成扩散目标和条件。
  - `train_score_model(...)`：统一训练行为扩散和状态扩散。
  - `compute_state_error(...)`：计算条件下一状态重构误差。
  - `get_state_threshold(...)` / `get_action_threshold(...)`：基于训练集误差分位数标定阈值。
- 核心算法：沿用 Karras-style diffusion loss；行为扩散 denoise `actions`，条件为 `states`；状态扩散 denoise `next_states`，条件为 `states` 与 `actions` 拼接。

#### 2.2.3 产物保存与 CLI 模块
- 文件路径：`examples/diffusion/pretrain_diffusion.py`
- 修改区域：约第 608-637 行、第 1446-1509 行、第 1698-1734 行
- 核心功能：将训练产物保存到 `saved_models/<env_name>/`，并提供可复用数据集接口。
- 关键函数：
  - `resolve_artifact_paths(...)`：生成 `behavior_diffusion.pt`、`state_diffusion.pt`、`pretrain_config.json`、`training_metrics.json`。
  - `save_pretrain_metadata(...)`：拆分保存配置快照和训练指标。
  - `parse_args()`：新增 `--train_behavior/--no-train_behavior` 与 `--train_state/--no-train_state`。
- 核心算法：默认不使用 dynamic model；配置中明确记录 `dynamic_model: disabled_for_recommender_state_tracker_setting`。

### 2.3 既有代码修改说明
- 修改前：状态扩散模型是无条件 `state_distribution`，训练目标为当前 `observations`，保存路径为 `saved_models/<env_name>/DOSER/diffusion/<artifact_name>/`，保存文件为 `behavior_model.pth`、`state_distribution.pth`、`pretrain_meta.json`。
- 修改后：状态扩散模型为条件 `state_diffusion`，训练目标为 `next_observations`，条件为 `(observations, actions)`；默认保存路径为 `saved_models/<env_name>/`，保存文件为 `behavior_diffusion.pt`、`state_diffusion.pt`、`pretrain_config.json`、`training_metrics.json`。
- 修改原因：推荐系统中下一状态由用户交互历史和 state tracker 表示，不需要额外 dynamic model；预训练阶段需要为后续 DORL 接入 OOD 识别提供行为和下一状态扩散模型。
- 备注：由于当前工具层普通 sandbox 启动器缺失，标准 `apply_patch` 工具无法访问仓库文件；本次使用 Python 精确文本替换完成写入，并通过 diff、语法检查和 smoke test 验证。

## 3. 数据处理说明
### 3.1 数据来源
- 数据文件：`data/KuaiRec/data_processed/DM_KuaiEnv-v0_small_data.pkl`
- 顶层格式：`list[dict]`
- 字段：`actions`、`terminals`、`rewards`、`observations`、`next_observations`、`user_id`
- 本次检查结果：用户轨迹数 1411，状态维度 42，动作维度 41。

### 3.2 数据预处理步骤
- 操作 1：读取 pickle 后校验顶层类型和每条轨迹的必需字段。
- 操作 2：校验 `observations`、`actions`、`next_observations`、`rewards`、`terminals` 的长度一致。
- 操作 3：将一维动作自动 reshape 为 `(N, 1)`，兼容未来离散 action id 数据。
- 操作 4：行为扩散保留所有 transition；状态扩散默认跳过 terminal transition。
- 操作 5：`trajectory_pkl` 后端默认不做状态归一化，保持 KuaiEnv 已处理状态向量。

### 3.3 数据统计信息
- 全量加载检查输出：`state_dim=42, action_dim=41, users=1411, behavior_transitions=4676570, state_transitions=4676570`
- terminal transition 数：`0`
- smoke test 使用：`max_trajectories=2, max_transitions=8, batch_size=4, pretrain_epochs=1`

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：smoke test 使用 CPU；完整硬件资源未做额外探测。
- 软件环境：`conda run -n easyrl4rec`，Python 3.11.13，PyTorch 2.8.0+cu128，NumPy 2.3.5。
- 超参数设置：smoke test 使用 `--device cpu --pretrain_epochs 1 --batch_size 4 --max_trajectories 2 --max_transitions 8 --save_root /tmp/easyrl4rec_diffusion_smoke_both_v2 --overwrite`。
- 语法检查：`conda run -n easyrl4rec python -m py_compile examples/diffusion/pretrain_diffusion.py` 通过。

### 4.2 图表结果分析
#### 4.2.1 本任务未涉及图表
- 图表类型：本任务未生成图表。
- 横坐标含义：本任务未涉及。
- 纵坐标含义：本任务未涉及。
- 图例说明：本任务未涉及。
- 数据计算方式：本任务未涉及。
- 结果分析：本任务只做代码适配与 smoke test，未做完整训练曲线分析。

## 5. 最终结论
已完成 `examples/diffusion/pretrain_diffusion.py` 对 KuaiEnv-v0 轨迹 pickle 的适配：脚本现在可以读取用户轨迹数据，展开 transition，同时训练行为扩散模型和条件状态扩散模型，并跳过 dynamic model。小样本 smoke test 已验证 forward、loss、backward、阈值计算和文件保存均可运行，产物文件名符合计划要求。

### 5.1 未完成事项说明
- 已完成部分：代码适配、语法检查、全量数据加载检查、小样本训练与保存验证。
- 未完成部分：未执行 4676570 条 transition 的全量长时间预训练，也未修改后续 DORL 加载器以消费新的扁平保存格式。
- 原因：当前任务目标是修改预训练脚本并完成 smoke test；全量训练耗时较长，DORL 接入属于下一阶段。
- 已尝试方案：使用 `/tmp/easyrl4rec_diffusion_smoke_both_v2/KuaiEnv-v0` 做隔离 smoke test，避免覆盖正式 `saved_models/KuaiEnv-v0/` 目录。
- 下一步：运行全量预训练后，再更新 DORL/DOSER artifact loader，使其识别 `behavior_diffusion.pt`、`state_diffusion.pt` 和新 metadata。

## 6. 后续建议
1. 全量训练建议从 CPU 切换到可用 GPU，并明确设置 `--device <gpu_id>`、`--pretrain_epochs`、`--batch_size`。
2. 若要保持旧 DORL-DOSER 脚本兼容，需要同步修改其 diffusion artifact 加载逻辑；当前仓库中已有旧日志显示旧 loader 读取 `saved_models/KuaiEnv-v0/DOSER/diffusion/DM_KuaiEnv-v0_small_data/`。
3. 后续接入 DORL 时，需要明确在线阶段状态扩散 OOD score 的输入来源，即 `(当前 state tracker state, candidate action, next state/state tracker update)`。
4. 如未来数据集存在 terminal transition，当前状态扩散训练会自动过滤这些样本；行为扩散仍保留 terminal transition 的行为样本。

## 7. 代码使用说明
默认全量训练命令示例：

```bash
conda run -n easyrl4rec python examples/diffusion/pretrain_diffusion.py \
  --env_name KuaiEnv-v0 \
  --dataset_path data/KuaiRec/data_processed/DM_KuaiEnv-v0_small_data.pkl \
  --save_root saved_models \
  --device 0 \
  --pretrain_epochs 100000 \
  --batch_size 256 \
  --overwrite
```

默认输出目录：`saved_models/KuaiEnv-v0/`

默认输出文件：
- `behavior_diffusion.pt`
- `state_diffusion.pt`
- `pretrain_config.json`
- `training_metrics.json`

可选控制：
- 只训练状态扩散：添加 `--no-train_behavior --train_state`
- 只训练行为扩散：添加 `--train_behavior --no-train_state`
- 指定子目录：添加 `--artifact_name <name>`，输出到 `saved_models/<env_name>/<name>/`
