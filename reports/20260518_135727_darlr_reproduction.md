# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
用户要求根据 `docs/DARLR_reproduction_description.md` 中的复现文档，对当前 EasyRL4Rec 代码进行 DARLR 复现，并提供可运行的执行脚本。

### 1.2 任务分解
1. 阅读 DARLR 复现文档和 DORL 基线代码，确认新增组件与现有训练链路的接口。
2. 新增 DARLR 策略模块，实现 recommender A2C 与 selector A2C 的联合策略。
3. 新增动态奖励环境，实现 reference users reward shaping 与 dynamic uncertainty penalty。
4. 轻改 collector，使 selector context 能在 `env.step()` 前注入环境。
5. 新增 `run_DARLR.py` 入口和 KuaiRec smoke shell 脚本。
6. 执行静态、导入和单元级验证，并记录未完成的真实数据 smoke 风险。

### 1.3 技术方案
采用最小侵入式实现：保留 DORL 的训练外壳、DeepFM `predicted_mat`、entropy penalty、state tracker 和评估流程；新增 `DARLRPolicy` 在训练采样阶段选择 reference users，并通过 `Batch.policy.darlr_context` 将 selector 结果传给环境。动态奖励环境继承 DORL 的 `PenaltyEntExpSimulatedEnv`，没有 selector context 时回退到 DORL reward，确保兼容原有测试 collector 和消融实验。

## 2. 任务完成过程记录
### 2.1 代码结构
生成或修改的文件：

- `examples/advance/run_DARLR.py`
- `src/core/policy/darlr.py`
- `src/core/envs/Simulated_Env/darlr_dynamic_reward.py`
- `src/core/collector/collector.py`
- `script/run_DARLR_KuaiRec_smoke.sh`
- `reports/20260518_135727_darlr_reproduction.md`
- `reports/agent_context/20260518_135727_darlr_reproduction.md`
- `reports/agent_context/latest_darlr_reproduction.md`

### 2.2 模块详细说明
#### 2.2.1 DARLR 训练入口
- 文件路径：`examples/advance/run_DARLR.py`
- 核心功能：基于 DORL 训练流程组装 DARLR 环境、策略、collector 和 trainer。
- 关键函数：
  - `get_args_DARLR()`：解析 selector、dynamic reward、dynamic uncertainty 相关超参数。
  - `prepare_train_envs()`：构造 `DARLRDynamicRewardEnv` 并加载 `predicted_mat`。
  - `setup_policy_model()`：创建 recommender actor/critic、selector actor/critic、偏好编码器、selector Transformer 状态编码器和 `DARLRPolicy`。
  - `main()`：执行完整训练流程。
- 核心算法：沿用 DORL 的 A2C recommender，并在 policy 层新增 selector 参考用户选择。

#### 2.2.2 DARLR 策略模块
- 文件路径：`src/core/policy/darlr.py`
- 核心功能：实现 DARLR 双智能体策略。
- 关键类：
  - `PreferenceEncoder`：将 `predicted_mat[user, :]` 投影到 selector 偏好空间。
  - `SelectorStateEncoder`：使用 Transformer 编码已选 reference users。
  - `SelectorActor`：在候选用户子集内输出选择 logits。
  - `SelectorCritic`：估计 selector 选择步骤的状态价值。
  - `DARLRPolicy`：兼容 Tianshou A2C 接口，联合训练 recommender 与 selector。
- 核心算法：
  1. recommender 基于 state tracker 输出 item 分布并采样推荐物品。
  2. selector 生成候选用户池，默认使用 `embedding_topk`，缺失时可使用 `random`。
  3. selector 连续选择 `selector_k` 个 reference users，并计算 intrinsic reward。
  4. `forward()` 将选择结果保存到 `Batch.policy.darlr_context`。
  5. `learn()` 同时计算 recommender A2C loss 和 selector A2C loss。

#### 2.2.3 DARLR 动态奖励环境
- 文件路径：`src/core/envs/Simulated_Env/darlr_dynamic_reward.py`
- 核心功能：在 DORL entropy/exposure reward 链路中接入动态 reward shaping 和 dynamic uncertainty penalty。
- 关键函数：
  - `_compute_pred_reward(action)`：读取 selector context，计算最终训练 reward。
  - `_compute_entropy_bonus(action_id)`：复用 DORL entropy bonus。
  - `_compute_dynamic_uncertainty(...)`：计算动态不确定性。
  - `step(action)`：把 DARLR 诊断指标写入 `info`。
- 核心算法：`dynamic_reward = mean(predicted_mat[selected_users, item])`；`dynamic_uncertainty = abs(dynamic_reward - previous_reward) / (similarity_gain + diversity_gain + eps)`；最终 reward 为 dynamic reward、uncertainty penalty 与 entropy bonus 的组合。

#### 2.2.4 Collector context 注入
- 文件路径：`src/core/collector/collector.py`
- 核心功能：在 `env.step()` 前将 `Batch.policy.darlr_context` 拆分为单环境 context，并通过 `set_env_attr("darlr_context", ...)` 注入底层环境。
- 关键函数：
  - `_inject_darlr_context(ready_env_ids)`：检查并注入 DARLR context。
  - `_slice_darlr_context(context_batch, local_index)`：从 batch context 中切出单个环境的上下文。
- 修改原因：原 collector 只向环境传递 action，DARLR 动态奖励还需要 selector 结果。
- 修改前：`action_remap = self.policy.map_action(self.data)` 后直接 `env.step()`。
- 修改后：在 `env.step()` 前调用 `_inject_darlr_context()`；非 DARLR 策略没有 `darlr_context`，因此不影响原路径。

#### 2.2.5 可运行脚本
- 文件路径：`script/run_DARLR_KuaiRec_smoke.sh`
- 核心功能：提供最小 KuaiRec DARLR smoke 命令。
- 关键配置：`SWANLAB_MODE=disabled`、`--cpu`、`epoch=1`、`step-per-epoch=1`、`training-num=1`、`test-num=1`、`selector_k=1`、`selector_candidate_size=2`。

### 2.3 既有代码修改说明
- 修改文件：`src/core/collector/collector.py`
- 修改区域：`Collector` 类中 `_reset_env_with_ids()` 后新增两个 DARLR context 辅助函数；`collect()` 中 `env.step()` 前新增 context 注入调用。
- 修改原因：DARLR 的动态奖励需要 selector 选择出的参考用户、相似性和多样性统计。
- 修改前：collector 只将 action 发送给环境。
- 修改后：当 policy 生成 `darlr_context` 时，collector 将其按环境 id 写入对应 env；普通策略无额外副作用。

## 3. 数据处理说明
### 3.1 数据来源
本任务未新增数据集。代码复用仓库已有数据路径、DeepFM 用户模型保存目录和 `predicted_mat`。

### 3.2 数据预处理步骤
- 本任务未新增离线数据预处理。
- DARLR 策略运行时会读取 `predicted_mat`，并在 `PreferenceEncoder` 中即时投影用户预测偏好向量。

### 3.3 数据统计信息
本任务未新增数据统计。真实 KuaiRec smoke 在进入配置日志后运行时间较长，本轮未完成完整真实数据训练验证。

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：当前环境未知；验证中观察到真实 KuaiRec tiny 命令占用 CPU 接近 100%。
- 软件环境：使用 `/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python`；该环境包含 `numpy 2.3.5`、`torch 2.8.0+cu128`。
- 超参数设置：单元测试使用 dummy 数据；shell smoke 使用 `selector_k=1`、`selector_candidate_size=2`、`lambda_uncertainty=0.05`、`lambda_entropy=0.0`。

### 4.2 图表结果分析
#### 4.2.1 本任务未涉及
- 图表类型：本任务未生成图表。
- 横坐标含义：本任务未涉及。
- 纵坐标含义：本任务未涉及。
- 图例说明：本任务未涉及。
- 数据计算方式：本任务未涉及。
- 结果分析：本任务未涉及。

### 4.3 验证结果
- 通过：`/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python -m py_compile examples/advance/run_DARLR.py src/core/policy/darlr.py src/core/envs/Simulated_Env/darlr_dynamic_reward.py src/core/collector/collector.py`
- 通过：最小导入检查 `DARLRPolicy` 与 `DARLRDynamicRewardEnv`。
- 通过：dummy `DARLRPolicy.forward()` 生成 `darlr_context`。
- 通过：dummy `DARLRPolicy.learn()` 可产生 recommender loss 和 selector loss，并完成反向传播。
- 通过：dummy `DARLRDynamicRewardEnv.step()` 返回 DARLR 诊断指标。
- 通过：`bash -n script/run_DARLR_KuaiRec_smoke.sh`。
- 未完成：真实 KuaiRec tiny/smoke 命令进入配置日志后长时间占用 CPU，本轮为避免持续占用资源已手动停止，尚未完成真实数据 1 epoch 验证。

## 5. 最终结论
已完成 DARLR v1 的代码复现骨架：新增训练入口、双智能体策略、动态奖励环境、collector context 注入和 KuaiRec smoke 脚本。静态检查、导入检查和不依赖真实数据的关键接口验证均已通过。真实 KuaiRec 端到端 smoke 尚未完成，当前主要风险在真实数据/DeepFM 模型加载或完整 trainer 运行耗时，而不是新增模块的基础接口。

### 5.1 未完成事项说明
- 已完成部分：DARLR 策略、环境、入口、脚本与单元级验证。
- 未完成部分：真实 KuaiRec smoke 脚本完整跑完 1 个 epoch。
- 原因：真实命令在输出配置日志后长时间 CPU 满载，当前回合为避免资源占用已停止。
- 已尝试方案：将脚本缩小到 `training_num=1`、`test_num=1`、`step_per_epoch=1`、`selector_k=1`，仍未在合理等待时间内完成。
- 下一步：定位 `prepare_user_model()` / DeepFM 模型加载耗时，或添加一个 bypass user-model 的本地 synthetic smoke。

## 6. 后续建议
1. 为 `run_DARLR.py` 增加 `--debug_synthetic`，用小型 synthetic `predicted_mat` 和 dummy env 快速跑 trainer。
2. 对真实 KuaiRec 运行做分段计时，定位耗时在用户模型加载、数据加载、collector 还是 trainer。
3. 若要更贴近论文，可将 `selector_candidate_mode=embedding_topk` 扩展为聚类候选池。
4. 完整实验前恢复 `--entropy_window 1 2` 和更大的 `selector_k`，并记录 DARLR 动态指标。

## 7. 代码使用说明
最小 smoke 脚本：

```bash
bash script/run_DARLR_KuaiRec_smoke.sh
```

手动运行示例：

```bash
/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python examples/advance/run_DARLR.py \
  --env KuaiEnv-v0 \
  --seed 2023 \
  --cpu \
  --epoch 1 \
  --step-per-epoch 1 \
  --episode-per-collect 1 \
  --training-num 1 \
  --test-num 1 \
  --batch-size 1 \
  --max_turn 2 \
  --force_length 1 \
  --which_tracker sasrec \
  --reward_handle "cat" \
  --window_size 3 \
  --selector_k 1 \
  --selector_candidate_size 2 \
  --selector_candidate_mode random \
  --lambda_entropy 0.0 \
  --entropy_window \
  --read_message "pointneg" \
  --message "DARLR_smoke"
```

完整实验可逐步放大 `selector_k`、`selector_candidate_size`、`training_num`、`test_num`、`step_per_epoch`，并切换 `selector_candidate_mode=embedding_topk`。
