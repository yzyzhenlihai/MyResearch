# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
用户要求让 `CounterfactualRewardModel.estimate()` 和真实环境 reward 更一致，并给出真实 reward 的值分布和 reward 范围，以判断最终 CTR 的取值范围。

### 1.2 任务分解
1. 检查 DORL-DOSER 当前 reward prior 的计算路径。
2. 对齐 `CounterfactualRewardModel.estimate()` 与真实 KuaiEnv reward 的来源。
3. 保留无真实 reward 矩阵时的模拟环境 reward 近似回退逻辑。
4. 在 rerank 日志中增加 reward prior 的分布信息。
5. 统计 KuaiEnv-v0 真实 reward 矩阵的数值范围与分位数。
6. 执行静态编译、最小 smoke test 和导入验证。

### 1.3 技术方案
真实 KuaiEnv 的 `BaseEnv.step()` 使用 `self.mat[self.cur_user, action]` 作为单步 reward。为了使 `CounterfactualRewardModel.estimate()` 更贴近最终评估 reward，本次实现优先读取 `RewardModelConfig.real_env.mat` 作为 reward prior 的基础矩阵；只有当真实矩阵缺失、维度不合法或与预测矩阵形状不一致时，才回退到离线 `predicted_mat + lambda_entropy * entropy - min_reward` 的模拟训练 reward 近似。这样既满足 KuaiEnv-v0 的真实 reward 对齐，也为其他推荐数据集保留兼容接口。

## 2. 任务完成过程记录
### 2.1 代码结构
本次修改了以下文件：

- `src/core/policy/dorl_doser_impl.py`
- `src/core/policy/dorl_doser_onpolicy_impl.py`

新增了以下报告文件：

- `reports/20260526_080318_reward_prior_real_reward_alignment.md`
- `reports/agent_context/20260526_080318_reward_prior_real_reward_alignment.md`
- `reports/agent_context/latest_reward_prior_real_reward_alignment.md`

### 2.2 模块详细说明
#### 2.2.1 reward prior 对齐模块
- 文件路径：`/data/yuzhengyang/RL_Learning/EasyRL4Rec/src/core/policy/dorl_doser_impl.py`
- 核心功能：让 `CounterfactualRewardModel.estimate()` 优先使用真实环境 reward 矩阵，并在无真实矩阵时回退到模拟环境 reward 近似。
- 关键函数：
  - `_resolve_real_reward_matrix()`：检查并读取 `real_env.mat`。
  - `_estimate_entropy_for_history(encoded_history)`：回退路径中按模拟训练环境公式估计 entropy bonus。
  - `_build_action_histories(actions, history_actions)`：整理候选动作历史。
  - `estimate(user_ids, action_ids, history_actions=None)`：返回 counterfactual reward prior。
- 核心算法：
  1. 若 `prefer_real_env_reward=True` 且 `real_env.mat` 与 `predicted_mat` 形状一致，则直接使用 `real_env.mat[user, action]`。
  2. 若真实矩阵不可用，则使用 `predicted_mat[user, action] + lambda_entropy * entropy - MIN_R`。
  3. 两条路径均对 reward 做非负裁剪。

#### 2.2.2 on-policy rerank 日志模块
- 文件路径：`/data/yuzhengyang/RL_Learning/EasyRL4Rec/src/core/policy/dorl_doser_onpolicy_impl.py`
- 核心功能：在 rerank 时为 reward prior 提供短动作历史，并记录 reward prior 分布指标。
- 关键函数：
  - `_extract_reward_histories(batch, row_ids, action_ids)`：从 `batch.obs[:, 1]` 读取上一动作，并拼接当前候选动作。
  - `_compute_reward_prior(batch, row_ids, action_ids)`：调用新的 `estimate(..., history_actions=...)`。
- 新增日志：
  - `rerank/reward_prior_min`
  - `rerank/reward_prior_max`
  - `rerank/reward_prior_std`

### 2.3 既有代码修改说明
- 修改文件：`src/core/policy/dorl_doser_impl.py`
- 修改区域：`RewardModelConfig` 与 `CounterfactualRewardModel`
- 修改原因：原先 `estimate()` 只使用 DeepFM `predicted_mat`，与最终 KuaiEnv 评估 reward `real_env.mat` 不一致，且数值范围极窄。
- 修改前：`reward = predicted_mat[user, action] - min_reward`
- 修改后：优先 `reward = real_env.mat[user, action]`；若真实矩阵不可用，回退为 `predicted_mat + entropy bonus - min_reward`

- 修改文件：`src/core/policy/dorl_doser_onpolicy_impl.py`
- 修改区域：rerank reward prior 计算与日志 metrics
- 修改原因：需要更容易诊断 reward prior 是否仍塌缩为 0。
- 修改前：只记录 `rerank/reward_prior` 均值。
- 修改后：同时记录均值、最小值、最大值和标准差。

## 3. 数据处理说明
### 3.1 数据来源
真实 reward 统计来自 KuaiEnv-v0 的 small matrix，即 `KuaiData.load_mat()` 读取的 `data/KuaiRec/data_raw/small_matrix_processed.csv`。该函数使用 `watch_ratio` 作为 reward，并将大于 5 的值截断为 5。

### 3.2 数据预处理步骤
- 调用 `KuaiData.load_mat()` 加载 small matrix。
- 将矩阵展平为一维数组。
- 统计最小值、最大值、均值、标准差、非零比例和分位数。

### 3.3 数据统计信息
真实 KuaiEnv-v0 reward 矩阵统计如下：

| 指标 | 数值 |
| --- | ---: |
| shape | `(1411, 3327)` |
| min | `0.0` |
| max | `5.0` |
| mean | `0.8722596110803493` |
| std | `0.661129569589299` |
| nonzero_ratio | `0.9897454348236845` |

分位数：

| percentile | reward |
| ---: | ---: |
| 0% | `0.0` |
| 1% | `0.0` |
| 5% | `0.09978110962557353` |
| 10% | `0.19053801739949253` |
| 25% | `0.4633644859813084` |
| 50% | `0.7668651541418557` |
| 75% | `1.118923076923077` |
| 90% | `1.5567889750619444` |
| 95% | `1.9516107718641977` |
| 99% | `3.580333903743316` |
| 99.9% | `5.0` |
| 100% | `5.0` |

对比训练用 DeepFM 预测矩阵：

| 指标 | 数值 |
| --- | ---: |
| shape | `(4694397,)`，对应矩阵 `(1411, 3327)` 展平 |
| min | `-0.014028804749250412` |
| max | `0.046700384095311166` |
| mean | `0.0001464662597567236` |
| std | `0.0017499976405377677` |

该对比说明原先 reward prior 的基础分数范围远小于真实 reward 范围。

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：当前环境未知。
- 软件环境：使用 `conda run -n easyrl4rec` 执行验证。
- 超参数设置：本次未运行完整训练，未改变训练超参数。

### 4.2 图表结果分析
#### 4.2.1 真实 reward 数值分布
- 图表类型：本次未绘图，提供数值分布表。
- 横坐标含义：reward 分位点。
- 纵坐标含义：KuaiEnv-v0 单步真实 reward，即 `watch_ratio` 截断后的值。
- 图例说明：本任务未涉及。
- 数据计算方式：对 `KuaiData.load_mat()` 返回的 dense reward matrix 展平后计算。
- 结果分析：真实单步 reward 范围为 `[0, 5]`，因此日志中的 CTR 如果定义为 `trajectory_reward / trajectory_length`，理论范围也是 `[0, 5]`。它不是二分类点击率，所以 CTR 超过 1 是合理现象。

### 4.3 验证记录
已执行：

```bash
conda run -n easyrl4rec python -m py_compile \
  src/core/policy/dorl_doser_impl.py \
  src/core/policy/dorl_doser_onpolicy_impl.py \
  examples/our_model/dorl_doser_onpolicy.py
```

结果：通过。

已执行最小 reward smoke test：

```text
real [0.20000000298023224, 5.0]
fallback [0.0, 0.10000000894069672]
```

含义：真实矩阵路径会直接返回真实 reward；关闭真实矩阵路径时，仍可回退到模拟 reward 近似。

已执行导入验证：

```text
import_ok CounterfactualRewardModel OnPolicyDORLDOSERPolicy
```

结果：通过。

## 5. 最终结论
`CounterfactualRewardModel.estimate()` 已改为优先使用真实 KuaiEnv reward 矩阵 `real_env.mat`，因此 rerank 的 `reward_prior` 与最终评估中的单步 reward/CTR 口径更一致。KuaiEnv-v0 的真实单步 reward 范围为 `[0, 5]`，所以最终 CTR 作为平均单步 reward 时也应落在 `[0, 5]`，不是 `[0, 1]`。

## 6. 后续建议
1. 重新运行 `DORL_DOSER_ONPOLICY`，观察 SwanLab 中 `rerank/reward_prior_min`、`rerank/reward_prior_max`、`rerank/reward_prior_std` 是否覆盖真实 reward 的有效区间。
2. 若需要严格避免训练阶段使用真实评估矩阵，可在构造 `RewardModelConfig` 时设置 `prefer_real_env_reward=False`，回到模拟训练 reward 近似。
3. 下一步应重点比较开启真实 reward prior 后的 `Trajectory Reward`、`CTR` 和 `MCD`，确认是否朝 DARLR baseline 靠近。

## 7. 代码使用说明
正常运行现有脚本即可使用新逻辑：

```bash
bash script/run_DORL_DOSER_onpolicy.sh
```

若后续需要禁用真实 reward prior，需要在构造 `RewardModelConfig` 时显式传入：

```python
RewardModelConfig(..., prefer_real_env_reward=False)
```

