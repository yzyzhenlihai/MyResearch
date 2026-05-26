# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
用户要求在当前 `dorl_doser_onpolicy` 表现较差的背景下，执行架构修改计划：在保留 DORL/A2C on-policy 更新语义的前提下，让 DOSER 的扩散 OOD 识别真正影响最终 actor 动作。具体采用“扩散 rerank”路径：actor 仍输出原始离散动作概率，但最终采样分布由 actor top-k、行为扩散候选、随机候选、critic auxiliary Q、action OOD penalty 和 counterfactual reward prior 共同重排。

### 1.2 任务分解
1. 在 `OnPolicyDORLDOSERPolicy` 中实现扩散引导的离散动作 rerank 出口。
2. 保留 A2C actor/value/entropy loss 和 on-policy learn 流程，同时保证 replay 中真实动作在学习阶段仍有非零概率。
3. 将 OOD threshold scale、rerank 候选数量、rerank 权重等参数暴露到训练入口和脚本。
4. 默认拆分 actor 与 critic backbone，降低 DOSER auxiliary critic 梯度污染 actor 表征的风险。
5. 补充 SwanLab/本地日志指标中的 rerank 与细分 OOD 指标。
6. 完成静态检查、导入检查、最小 forward/learn smoke test、脚本语法检查，并生成本报告与 agent 交接记录。

### 1.3 技术方案
采用“actor proposal + DOSER critic/OOD rerank”的保守方案。actor 原始概率仍参与最终 score，因此 actor 的 on-policy 梯度仍来自 `dist.log_prob(minibatch.act) * advantage`；critic Q、扩散 OOD error 和 reward prior 在 rerank 时均使用 `torch.no_grad()` 计算，只改变最终采样分布，不直接把 OOD scorer 梯度传回 actor。

候选动作集合由四部分组成：actor top-k、行为扩散模型采样后最近邻 item、随机合法动作、学习阶段必须保留的历史动作。最终 score 为：

```text
actor_log_prob / temperature
+ alpha_q * zscore(aux_q)
+ gamma_reward * zscore(reward_prior)
- beta_action_ood * relu(action_error / action_threshold - 1)
```

该设计使 DOSER 可以通过最终 `Categorical(probs=rerank_probs)` 影响 actor 实际采样动作，同时保留 A2C 的主更新语义。

## 2. 任务完成过程记录
### 2.1 代码结构
本轮修改文件：

1. `src/core/policy/dorl_doser_onpolicy_impl.py`
2. `examples/our_model/dorl_doser_onpolicy.py`
3. `script/run_DORL_DOSER_onpolicy.sh`

本轮新增报告文件：

1. `reports/20260524_053239_dorl_doser_diffusion_rerank.md`
2. `reports/agent_context/20260524_053239_dorl_doser_diffusion_rerank.md`
3. `reports/agent_context/latest_dorl_doser_diffusion_rerank.md`

### 2.2 模块详细说明
#### 2.2.1 DORL-DOSER On-Policy 策略实现
- 文件路径：`src/core/policy/dorl_doser_onpolicy_impl.py`
- 修改区域：常量区约第 71-102 行；策略初始化约第 301-415 行；rerank helper 与 `forward()` 约第 521-986 行；OOD threshold 与指标约第 1104-1160 行；`learn()` 指标返回约第 1341-1438 行。
- 核心功能：为 `OnPolicyDORLDOSERPolicy` 增加扩散引导 rerank 动作出口，使 DOSER 不再只作为 critic auxiliary loss，而是能影响最终动作分布。
- 关键函数：
  - `_normalize_actor_probs(actor_probs, action_mask)`：归一化 actor 概率，并处理合法动作 mask 全空的边界情况。
  - `_actor_topk_candidate_ids(actor_probs, action_mask)`：提取 actor top-k 候选动作。
  - `_diffusion_candidate_ids(obs_emb)`：调用行为扩散模型采样连续动作，并映射到最近邻 item id。
  - `_random_candidate_ids(batch_size, action_dim, action_mask, device)`：补充随机候选，保留探索。
  - `_merge_candidate_ids(...)`：合并候选、应用合法动作 mask，并保证学习阶段真实执行动作不被移除。
  - `_build_rerank_distribution(...)`：计算 Q score、reward prior、action OOD penalty，并生成 rerank 后的离散动作概率。
  - `forward(...)`：替换默认 actor 分布出口，返回 `dist=Categorical(rerank_probs)`，同时把候选动作写入 `Batch.policy.doser_candidates`。
  - `learn(...)`：继续使用 A2C actor/value/entropy loss，补充 `rerank/candidate_size`、`ood/action_ood_ratio`、`ood/state_ood_ratio` 等日志返回。
- 核心算法：候选集合先通过 actor、扩散模型和随机采样生成；然后在候选集合内使用 actor log-prob、critic auxiliary Q、reward prior 和 action OOD penalty 组合打分；最后对完整 action 维度做 masked softmax，得到最终采样分布。

#### 2.2.2 On-Policy 主训练入口
- 文件路径：`examples/our_model/dorl_doser_onpolicy.py`
- 修改区域：CLI 参数约第 82-104 行；policy/model 初始化约第 214-264 行。
- 核心功能：暴露 rerank 与 OOD threshold scale 参数，并默认拆分 actor/critic backbone。
- 关键变化：
  - 新增 `--doser_enable_rerank` / `--no_doser_enable_rerank`，默认启用 rerank。
  - 新增 `--doser_actor_topk`、`--doser_diffusion_candidates`、`--doser_random_candidates`。
  - 新增 `--doser_rerank_alpha_q`、`--doser_rerank_beta_action_ood`、`--doser_rerank_gamma_reward`、`--doser_rerank_temperature`。
  - 新增 `--doser_action_threshold_scale`、`--doser_state_threshold_scale`。
  - 新增 `--doser_share_actor_critic_backbone` / `--no_doser_share_actor_critic_backbone`，默认不共享 backbone。

#### 2.2.3 训练脚本
- 文件路径：`script/run_DORL_DOSER_onpolicy.sh`
- 修改区域：rerank 环境变量约第 50-60 行；flag 拼装约第 90-99 行；smoke 配置约第 106-119 行；Python 参数传入约第 156-164 行。
- 核心功能：提供可直接运行的 DORL-DOSER on-policy 训练脚本，并允许通过环境变量调节 rerank。
- 关键参数：
  - `DOSER_ENABLE_RERANK=1`：启用扩散 rerank。
  - `DOSER_ACTOR_TOPK=64`、`DOSER_DIFFUSION_CANDIDATES=64`、`DOSER_RANDOM_CANDIDATES=16`：控制候选集大小。
  - `DOSER_RERANK_ALPHA_Q=1.0`、`DOSER_RERANK_BETA_ACTION_OOD=0.5`、`DOSER_RERANK_GAMMA_REWARD=0.2`：控制 rerank score 组成。
  - `DOSER_ACTION_THRESHOLD_SCALE=1.0`、`DOSER_STATE_THRESHOLD_SCALE=2.0`：控制 OOD 判定阈值缩放。
  - `DOSER_SHARE_ACTOR_CRITIC_BACKBONE=0`：默认拆分 actor 与 critic backbone。

### 2.3 既有代码修改说明
- 修改文件：`src/core/policy/dorl_doser_onpolicy_impl.py`
- 修改原因：此前 DOSER 只通过 critic auxiliary loss 间接影响训练，最终 actor 动作仍主要由原始 A2C 分布决定，导致 OOD 模块对在线动作选择影响弱。
- 修改前：`forward()` 使用 actor 原始概率构造 `Categorical`，`learn()` 只返回 A2C 与 DOSER auxiliary loss。
- 修改后：`forward()` 在离散动作场景下默认启用 rerank；`learn()` 使用 rerank 后的分布计算 log-prob，并记录 rerank 与细分 OOD 指标。

- 修改文件：`examples/our_model/dorl_doser_onpolicy.py`
- 修改原因：训练入口缺少 rerank 相关参数，也默认共享 actor/critic backbone。
- 修改前：actor 与 critic 共用同一个 `Net`；DOSER 权重参数主要限制在 auxiliary critic loss。
- 修改后：默认拆分 actor/critic backbone，并将 rerank 候选数量、score 权重、阈值缩放等参数传入 policy。

- 修改文件：`script/run_DORL_DOSER_onpolicy.sh`
- 修改原因：需要给 KuaiEnv-v0 训练提供可复现实验入口和 rerank 参数控制。
- 修改前：脚本不传入 rerank 参数。
- 修改后：脚本支持启用/关闭 rerank、调节候选集、调节 OOD threshold scale，并在 smoke 模式下降低候选数量以缩短验证时间。
- 备注：该脚本中 `SWANLAB_MODE` 和 `CUDA` 默认值在本轮开始前已处于工作区未提交状态；本轮重点新增 rerank 参数和传参逻辑。

## 3. 数据处理说明
### 3.1 数据来源
本任务未新增或修改数据集。运行期仍依赖项目已有 KuaiEnv-v0 数据、离线预测矩阵、state tracker 和已预训练扩散产物。

### 3.2 数据预处理步骤
- 本任务未新增离线预处理步骤。
- rerank 阶段仅在训练/采样时读取当前 batch 的状态表示、动作 mask、用户 id，并基于行为扩散模型和 item embedding 构造候选动作。

### 3.3 数据统计信息
本任务未新增数据统计。完整脚本 smoke 尝试时进入 KuaiEnv 现有 entropy 统计构建流程，该流程会遍历约 `12530806` 行数据，未在本轮等待其完成。

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：当前环境未完整记录；本轮验证主要在 CPU/本地 Python 进程下完成。
- 软件环境：使用 `conda run -n easyrl4rec`；关键依赖包括 PyTorch、Tianshou、SwanLab、DeepCTR-Torch。具体版本本轮未单独查询。
- 超参数设置：默认启用 rerank；`doser_actor_topk=64`、`doser_diffusion_candidates=64`、`doser_random_candidates=16`、`doser_rerank_alpha_q=1.0`、`doser_rerank_beta_action_ood=0.5`、`doser_rerank_gamma_reward=0.2`、`doser_action_threshold_scale=1.0`、`doser_state_threshold_scale=2.0`。

已执行验证：

1. 静态编译检查：
   ```bash
   conda run -n easyrl4rec python -m py_compile \
     examples/our_model/dorl_doser_onpolicy.py \
     src/core/policy/dorl_doser_onpolicy_impl.py \
     src/core/policy/dorl_doser_impl.py \
     src/core/policy/doser.py
   ```
   结果：通过。

2. 导入 smoke test：
   ```bash
   PYTHONPATH=.:./src:./src/DeepCTR-Torch:./src/tianshou:./examples/our_model \
   conda run -n easyrl4rec python -c "from src.core.policy.doser import ..."
   ```
   结果：通过，确认 `A2CDOSERAugmentedCritic`、`OnPolicyDORLDOSERPolicy`、`load_diffusion_artifact` 可导入，默认参数包含 `doser_enable_rerank=True`。

3. 最小 forward/learn smoke test：
   - 使用 dummy state tracker、dummy actor、dummy critic、dummy diffusion artifact 和 dummy reward model。
   - 验证 `forward()` 输出 shape 为 `(3, 8)`。
   - 验证 `Batch.policy.doser_candidates` 可以写入。
   - 验证 `learn()` 可以完成一次 backward/update，并返回 `rerank/candidate_size`。
   - 输出摘要：`dummy_ok (3, 8) True 4.0`。

4. 脚本语法检查：
   ```bash
   bash -n script/run_DORL_DOSER_onpolicy.sh
   ```
   结果：通过。

5. Diff 空白检查：
   ```bash
   git diff --check -- \
     src/core/policy/dorl_doser_onpolicy_impl.py \
     examples/our_model/dorl_doser_onpolicy.py \
     script/run_DORL_DOSER_onpolicy.sh
   ```
   结果：通过。

6. 完整脚本 smoke 尝试：
   ```bash
   SWANLAB_MODE=disabled SMOKE=1 CPU_FLAG=1 \
   PYTHON_BIN=/data/yuzhengyang/miniconda3/envs/easyrl4rec/bin/python \
   bash script/run_DORL_DOSER_onpolicy.sh
   ```
   结果：脚本成功进入配置阶段并打印 rerank 参数，但随后进入 KuaiEnv entropy 统计构建流程。该流程需要遍历约 `12530806` 行数据，本轮未等待完成，已中止。因此本轮未完成真实 KuaiEnv collector/train smoke。

### 4.2 图表结果分析
#### 4.2.1 本任务未涉及
- 图表类型：本任务未生成图表。
- 横坐标含义：本任务未涉及。
- 纵坐标含义：本任务未涉及。
- 图例说明：本任务未涉及。
- 数据计算方式：本任务未涉及。
- 结果分析：本轮只完成代码接入和 smoke 验证，未进行完整训练曲线分析。

#### 4.2.2 本任务未涉及
- 图表类型：本任务未生成图表。
- 横坐标含义：本任务未涉及。
- 纵坐标含义：本任务未涉及。
- 图例说明：本任务未涉及。
- 数据计算方式：本任务未涉及。
- 结果分析：本任务未验证最终是否超过 DARLR baseline。

## 5. 最终结论
本轮已完成 DORL-DOSER on-policy 的扩散 rerank 架构接入。DOSER 现在不仅在 critic 端提供 auxiliary OOD penalty / compensation，还会通过 rerank 后的 `Categorical` 分布影响最终 actor 动作选择。A2C 的 actor loss、value loss、entropy loss 和 on-policy 更新形式仍保留。

已验证代码可编译、关键导入可用、最小 forward/learn 可完成一次更新、训练脚本语法正确。尚未完成 KuaiEnv-v0 全量训练，也尚未证明指标超过 DARLR，因此不能给出性能提升结论。

## 6. 后续建议
1. 优先运行完整 KuaiEnv-v0 训练，并在 SwanLab 中观察 `Trajectory Reward`、`CTR`、`Trajectory length`、`MCD` 以及新增的 `rerank/*`、`ood/action_ood_ratio`、`ood/state_ood_ratio`。
2. 做 ablation：`DOSER_ENABLE_RERANK=0` 对比纯 critic OOD；`DOSER_RERANK_ALPHA_Q=0` 对比去掉 Q；`DOSER_RERANK_BETA_ACTION_OOD=0` 对比去掉 action OOD penalty。
3. 若 entropy 仍快速坍缩，提高 `ENT_COEF`，或增大 `DOSER_RANDOM_CANDIDATES`、提高 `doser_rerank_temperature`。
4. 若 `ood/state_ood_ratio` 继续接近 1，可继续上调 `DOSER_STATE_THRESHOLD_SCALE`，或降低状态 OOD 在辅助损失中的影响。
5. 若完整 smoke 长时间卡在 entropy 统计，建议缓存 KuaiEnv entropy 统计或提供更小的 smoke 数据入口。

## 7. 代码使用说明
默认训练命令：

```bash
bash script/run_DORL_DOSER_onpolicy.sh
```

关闭 rerank 做消融：

```bash
DOSER_ENABLE_RERANK=0 bash script/run_DORL_DOSER_onpolicy.sh
```

调整 rerank 权重：

```bash
DOSER_RERANK_ALPHA_Q=1.5 \
DOSER_RERANK_BETA_ACTION_OOD=0.3 \
DOSER_RERANK_GAMMA_REWARD=0.3 \
bash script/run_DORL_DOSER_onpolicy.sh
```

使用新 flat 扩散产物目录 `saved_models/<env_name>/`：

```bash
DIFFUSION_ARTIFACT_NAME="" bash script/run_DORL_DOSER_onpolicy.sh
```

smoke 模式：

```bash
SWANLAB_MODE=disabled SMOKE=1 CPU_FLAG=1 bash script/run_DORL_DOSER_onpolicy.sh
```

注意：当前 smoke 仍可能触发 KuaiEnv 原有 entropy 统计流程，耗时取决于数据规模。
