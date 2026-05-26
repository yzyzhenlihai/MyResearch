# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
用户反馈 DORL-DOSER on-policy 在 `test_episode -> collector.collect -> buffer.add` 阶段报错：`ValueError: shape mismatch: value array of shape (100,145) could not be broadcast to indexing result of shape (100,144)`，要求判断问题原因并处理。

### 1.2 任务分解
1. 定位 replay buffer 写入的 shape mismatch 字段。
2. 检查 rerank 逻辑是否向 `policy` 写入了变长字段。
3. 修复 `doser_candidates` 在 collector 阶段写入 buffer 导致的 144/145 列不一致问题。
4. 避免 collector 中上一轮 `batch.act` 被误当作 learn 阶段 required action。
5. 运行静态编译和最小行为验证。

### 1.3 技术方案
问题来自 `policy.doser_candidates`：第一步 collector 前向时没有 required action，候选列数为 `64 + 64 + 16 = 144`；后续 collector 前向中 `self.data.act` 保存了上一轮动作，旧逻辑把它当作 required action 又加入候选，候选列数变为 `145`。Tianshou replay buffer 初始化后要求同一字段形状固定，因此写入 `(100,145)` 到已分配 `(100,144)` 的字段时广播失败。

修复方案：
1. 不再把 rerank 候选动作写入 `Batch.policy.doser_candidates`，避免 replay buffer 保存变长候选。
2. learn 阶段重新生成候选，并通过 `batch.act` 强制加入真实动作，保证 `log_prob(minibatch.act)` 非零。
3. 新增 `_is_learning_minibatch()`，只有 batch 同时包含 `adv` 和 `returns` 时才使用 `_required_action_ids()`。

## 2. 任务完成过程记录
### 2.1 代码结构
修改文件：

1. `src/core/policy/dorl_doser_onpolicy_impl.py`

新增报告文件：

1. `reports/20260526_015340_dorl_doser_candidate_shape_fix.md`
2. `reports/agent_context/20260526_015340_dorl_doser_candidate_shape_fix.md`
3. `reports/agent_context/latest_dorl_doser_candidate_shape_fix.md`

### 2.2 模块详细说明
#### 2.2.1 Rerank 候选不再写入 replay buffer
- 文件路径：`src/core/policy/dorl_doser_onpolicy_impl.py`
- 修改区域：`_build_rerank_distribution()` 和 `forward()`。
- 核心功能：`_build_rerank_distribution()` 现在只返回 rerank 概率和日志指标，不再返回 `stored_candidates`；`forward()` 不再设置 `policy_batch.doser_candidates`。
- 核心算法：collector 阶段只用候选动作计算当前采样分布，候选本身不进入 buffer；learn 阶段重新构造候选集合。

#### 2.2.2 Required action 只在 learn 阶段启用
- 文件路径：`src/core/policy/dorl_doser_onpolicy_impl.py`
- 修改区域：`_required_action_ids()` 后新增 `_is_learning_minibatch()`；`_build_rerank_distribution()` 中 required action 获取条件。
- 关键函数：
  - `_is_learning_minibatch(batch)`：当 batch 同时包含 `adv` 和 `returns` 时返回 `True`。
  - `_required_action_ids(batch, device)`：仅在 learn minibatch 中用于保留真实动作。
- 修改目的：collector 中的 `batch.act` 是上一轮动作，不是当前 rerank 必须保留的真实动作，因此不能参与当前候选集。

### 2.3 既有代码修改说明
- 修改文件：`src/core/policy/dorl_doser_onpolicy_impl.py`
- 修改原因：`policy.doser_candidates` 形状会在 collector 的不同时间步从 144 变为 145，导致 replay buffer 字段 shape mismatch。
- 修改前：rerank 候选写入 `policy_batch.doser_candidates`，并且任意 `batch.act` 都会作为 required action。
- 修改后：rerank 候选不写入 buffer；只有 learn minibatch 的 `batch.act` 才作为 required action。

## 3. 数据处理说明
### 3.1 数据来源
本任务未涉及。

### 3.2 数据预处理步骤
- 本任务未涉及。

### 3.3 数据统计信息
本任务未涉及。

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：当前环境未知。
- 软件环境：`conda run -n easyrl4rec`。
- 超参数设置：本任务未修改训练超参数。

验证命令：

```bash
conda run -n easyrl4rec python -m py_compile \
  src/core/policy/dorl_doser_onpolicy_impl.py \
  examples/our_model/dorl_doser_onpolicy.py
```

结果：通过。

最小行为验证：

```bash
PYTHONPATH=.:./src:./src/DeepCTR-Torch:./src/tianshou:./examples/our_model \
conda run -n easyrl4rec python -c "..."
```

输出：

```text
collector_is_learning False
learn_is_learning True
empty_required None
```

说明 collector batch 不会触发 required action，learn batch 会触发 required action，空 `Batch()` 动作仍安全返回 `None`。

### 4.2 图表结果分析
#### 4.2.1 本任务未涉及
- 图表类型：本任务未生成图表。
- 横坐标含义：本任务未涉及。
- 纵坐标含义：本任务未涉及。
- 图例说明：本任务未涉及。
- 数据计算方式：本任务未涉及。
- 结果分析：本任务只修复训练流程中的 buffer shape 错误。

#### 4.2.2 本任务未涉及
- 图表类型：本任务未生成图表。
- 横坐标含义：本任务未涉及。
- 纵坐标含义：本任务未涉及。
- 图例说明：本任务未涉及。
- 数据计算方式：本任务未涉及。
- 结果分析：本轮未运行完整训练。

## 5. 最终结论
该错误是 rerank 候选动作被保存到 replay buffer 后列数不固定造成的，不是 `training_metrics.json`、扩散模型权重或环境 reward 的问题。已修复为：候选动作仅用于当前 rerank 采样，不再进入 buffer；learn 阶段重新生成候选并只在 `adv/returns` 存在时强制加入真实动作。

## 6. 后续建议
1. 重新运行 `bash script/run_DORL_DOSER_onpolicy.sh`，确认 `buffer.add` 阶段不再出现 `(100,145)` 与 `(100,144)` 的 shape mismatch。
2. 若继续报 shape mismatch，优先检查 traceback 中具体字段，确认是否还有其他 `policy.*` 字段形状变化。
3. 后续若想复用 collection 时的候选集，需要改成固定宽度 padding，并额外保存 candidate mask；当前为稳定训练，选择不保存候选。

## 7. 代码使用说明
原训练命令保持不变：

```bash
bash script/run_DORL_DOSER_onpolicy.sh
```

smoke 模式：

```bash
SWANLAB_MODE=disabled SMOKE=1 CPU_FLAG=1 bash script/run_DORL_DOSER_onpolicy.sh
```
