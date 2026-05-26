# DORL-DOSER 当前改进方案总结

## 1. 改进目标

当前改进围绕 `DORL + DOSER OOD` 在推荐环境 `KuaiEnv-v0` 下的 on-policy 训练展开。原始 DORL-DOSER 版本的主要问题是：DOSER 的 OOD 识别主要作为 critic 端辅助正则存在，对 actor 最终推荐动作的影响较弱，导致模型在实际交互时仍可能沿用 A2C actor 的原始动作分布，OOD 模块难以直接改善推荐动作质量。

本轮改进目标是：

1. 保留 DORL/A2C 的 on-policy 更新逻辑。
2. 保留 DOSER 细粒度 OOD penalty / compensation。
3. 让 DOSER 的扩散模型、auxiliary critic 和 OOD 识别结果直接参与最终动作选择。
4. 适配 `KuaiEnv-v0` 的扩散预训练产物。
5. 保持接口可迁移到其他推荐数据集。

## 2. 当前总体架构

当前主入口为：

```text
examples/our_model/dorl_doser_onpolicy.py
```

核心策略实现为：

```text
src/core/policy/dorl_doser_onpolicy_impl.py
```

训练脚本为：

```text
script/run_DORL_DOSER_onpolicy.sh
```

当前策略可以概括为：

```text
A2C actor 原始概率
        |
        v
候选动作集合构造
        |
        v
DOSER rerank score
        |
        v
最终 Categorical 动作分布
        |
        v
环境交互 + A2C on-policy 更新
        |
        v
critic auxiliary OOD penalty / compensation
```

其中 actor 仍然是 A2C actor，critic 端增加 DOSER auxiliary Q/V 分支；额外新增的 rerank 机制负责让 DOSER 影响最终动作分布。

## 3. 与原始 DORL-DOSER 的主要差异

### 3.1 原始方案

原始接入方式主要是：

1. actor 输出动作分布。
2. collector 按 actor 分布采样动作。
3. critic 端计算 DOSER auxiliary loss。
4. OOD penalty / compensation 主要更新 critic auxiliary 分支。

这种方式的问题是：OOD 模块虽然能约束 critic，但 actor 最终动作分布不一定直接受 OOD 识别结果控制。

### 3.2 当前方案

当前方案增加了扩散引导 rerank：

1. actor 仍先输出原始动作概率。
2. 构造候选动作集合。
3. 对候选动作计算：
   - actor log-prob
   - auxiliary Q score
   - action OOD error
   - reward prior
4. 使用组合分数重新构造最终动作分布。
5. collector 使用 rerank 后的动作分布采样。

因此，DOSER 不再只是 critic 端正则，而是能通过 rerank 直接影响 actor 的最终动作出口。

## 4. Rerank 候选动作构造

当前候选动作由三类来源组成：

```text
候选动作 = actor top-k + 行为扩散候选 + 随机探索候选
```

默认参数为：

```text
doser_actor_topk = 64
doser_diffusion_candidates = 64
doser_random_candidates = 16
```

### 4.1 Actor Top-K 候选

从 A2C actor 原始动作概率中选出 top-k item。该部分保留 actor 已学习到的偏好，避免 rerank 完全脱离原策略。

### 4.2 行为扩散候选

使用预训练行为扩散模型从 `p(a | s)` 中采样连续动作 embedding，再映射到最近邻 item id。该部分引入 DOSER 学到的离线行为分布信息。

### 4.3 随机探索候选

随机采样部分合法动作，避免候选集过窄，保留一定探索空间。

### 4.4 Learn 阶段真实动作保护

在 `learn()` 阶段，当前 minibatch 的真实动作 `batch.act` 会被强制加入候选集，保证：

```text
log_prob(minibatch.act)
```

不会因为 rerank 候选遗漏真实动作而变成无效值。

注意：collector 阶段的 `batch.act` 可能是上一轮动作，不再作为当前 required action 使用。当前用 `_is_learning_minibatch()` 判断 batch 是否包含 `adv` 和 `returns`，只有 learn 阶段才启用真实动作保护。

## 5. Rerank 打分公式

当前 rerank score 为：

```text
score(s, a)
  = actor_log_prob(s, a) / temperature
  + alpha_q * zscore(Q_aux(s, a))
  + gamma_reward * zscore(reward_prior(s, a))
  - beta_action_ood * action_ood_penalty(s, a)
```

默认参数为：

```text
doser_rerank_temperature = 1.0
doser_rerank_alpha_q = 1.0
doser_rerank_beta_action_ood = 0.5
doser_rerank_gamma_reward = 0.2
```

各项含义如下：

| 项 | 含义 | 作用 |
|---|---|---|
| `actor_log_prob` | A2C actor 原始动作概率的 log 值 | 保留原 actor 策略偏好 |
| `Q_aux(s,a)` | DOSER auxiliary critic 对候选动作的价值估计 | 偏向长期价值更高的动作 |
| `reward_prior(s,a)` | 离线预测矩阵给出的即时 reward 先验 | 偏向单步点击/偏好更高的 item |
| `action_ood_penalty` | 行为扩散模型给出的 action OOD 惩罚 | 压低偏离离线行为分布过远的动作 |

其中 `Q_aux` 和 `reward_prior` 会做 row-wise z-score，避免不同 batch 或候选集合之间尺度差异过大。

## 6. Reward Prior 的定义

当前 `reward_prior` 是一个轻量级 counterfactual reward 近似值，来源于离线预测矩阵：

```text
reward_prior(u, a) = predicted_mat[u, a] - min_reward
```

并进行非负截断：

```text
reward_prior = max(reward_prior, MIN_REWARD_VALUE)
```

它不是真实环境 reward，也不是 critic 学到的长期 Q value，而是一个单步即时奖励先验，用于在候选 item 内辅助排序。

如果希望消融该项，可以设置：

```bash
DOSER_RERANK_GAMMA_REWARD=0
```

## 7. OOD 识别与惩罚/激励

当前并没有取消原有 OOD 细粒度识别。现在有两层 OOD 使用方式：

### 7.1 Critic 端细粒度 OOD 正则

critic 端仍保留：

```text
action_ood_mask = action_error > action_threshold
state_ood_mask = state_error > state_threshold
ood_mask = action_ood_mask OR state_ood_mask
```

并继续区分：

```text
positive_mask = action_ood 且非 state_ood 且 value_s_pi >= value_s_in
negative_mask = ood_mask 且非 positive_mask
```

对应损失为：

```text
ood_loss = beta * penalty_loss + lam * compensation_loss
```

含义如下：

| 损失 | 含义 |
|---|---|
| `loss/doser_penalty` | 对 negative OOD 动作进行 Q 值压制 |
| `loss/doser_compensation` | 对 positive OOD 动作进行价值补偿 |
| `loss/doser_aux` | 辅助 Q/V 主训练损失 |
| `loss/doser` | DOSER critic 端总损失 |

### 7.2 Actor 出口 rerank OOD 惩罚

rerank 阶段额外使用 action OOD penalty：

```text
action_ood_penalty = relu(action_error / action_threshold - 1)
```

该项直接进入最终动作 score，用于降低高 action OOD 候选动作的采样概率。

目前 rerank 阶段使用的是较轻量的 action OOD 惩罚；完整 positive/negative OOD 细粒度逻辑仍主要保留在 critic auxiliary loss 中。

## 8. Threshold Scale

当前引入了两个阈值缩放参数：

```text
doser_action_threshold_scale = 1.0
doser_state_threshold_scale = 2.0
```

原因是在线 counterfactual next state 的状态扩散误差容易偏高，如果直接使用预训练阶段的 `state_threshold`，可能导致 `state_ood_ratio` 长期接近 1，使 compensation 几乎无法触发。

因此当前默认将 state threshold 放宽为 2 倍，缓解状态 OOD 过度敏感问题。

## 9. Actor/Critic Backbone 拆分

当前默认不共享 actor 与 critic backbone：

```text
doser_share_actor_critic_backbone = False
```

这样做的原因是 DOSER auxiliary critic loss 较复杂，包含 Q/V expectile、OOD penalty、OOD compensation 等项。如果 actor 和 critic 共用 backbone，critic auxiliary 梯度可能污染 actor 表征，进一步加剧策略不稳定。

如需恢复共享 backbone，可以设置：

```bash
DOSER_SHARE_ACTOR_CRITIC_BACKBONE=1
```

## 10. SwanLab 与日志指标

当前新增或保留的关键日志包括：

```text
loss
loss/actor
loss/vf
loss/ent
loss/doser
loss/doser_aux
loss/doser_penalty
loss/doser_compensation
ood/ood_ratio
ood/action_ood_ratio
ood/state_ood_ratio
ood/positive_ratio
ood/negative_ratio
rerank/candidate_size
rerank/q_score
rerank/action_error
rerank/reward_prior
rerank/action_ood_penalty
```

这些指标用于判断：

1. A2C 主训练是否正常。
2. DOSER auxiliary critic 是否主导训练。
3. OOD 判断是否过严或过松。
4. rerank 候选规模是否符合预期。
5. action OOD penalty 是否真的参与动作出口。

## 11. 已修复的关键工程问题

### 11.1 空 `Batch()` 动作问题

报错：

```text
TypeError: Object Batch() has no len()
```

原因：

collector 前向阶段 `batch.act` 可能是 Tianshou 的空占位 `Batch()`，旧逻辑把它当作 tensor 转换。

修复：

`_required_action_ids()` 现在会识别空 `Batch()` 并返回 `None`。

### 11.2 `doser_candidates` 变长写入 buffer 问题

报错：

```text
ValueError: shape mismatch: value array of shape (100,145)
could not be broadcast to indexing result of shape (100,144)
```

原因：

旧逻辑把 rerank 候选写入 `policy.doser_candidates`。collector 不同 step 的候选列数可能不同：

```text
无 required action: 144
带 required action: 145
```

Tianshou replay buffer 要求同一字段 shape 固定，因此写入失败。

修复：

1. 不再把 rerank 候选写入 replay buffer。
2. collector 阶段只用候选计算当前动作分布。
3. learn 阶段重新生成候选，并强制加入真实动作。

## 12. 当前运行脚本

默认运行：

```bash
bash script/run_DORL_DOSER_onpolicy.sh
```

关闭 rerank 做消融：

```bash
DOSER_ENABLE_RERANK=0 bash script/run_DORL_DOSER_onpolicy.sh
```

调节 rerank 权重：

```bash
DOSER_RERANK_ALPHA_Q=1.5 \
DOSER_RERANK_BETA_ACTION_OOD=0.3 \
DOSER_RERANK_GAMMA_REWARD=0.3 \
bash script/run_DORL_DOSER_onpolicy.sh
```

调节候选数量：

```bash
DOSER_ACTOR_TOPK=64 \
DOSER_DIFFUSION_CANDIDATES=64 \
DOSER_RANDOM_CANDIDATES=16 \
bash script/run_DORL_DOSER_onpolicy.sh
```

调节 OOD 阈值：

```bash
DOSER_ACTION_THRESHOLD_SCALE=1.0 \
DOSER_STATE_THRESHOLD_SCALE=2.0 \
bash script/run_DORL_DOSER_onpolicy.sh
```

smoke 模式：

```bash
SWANLAB_MODE=disabled SMOKE=1 CPU_FLAG=1 bash script/run_DORL_DOSER_onpolicy.sh
```

## 13. 推荐消融实验

为了判断每个模块是否有效，建议至少做以下消融：

| 实验 | 参数 | 目的 |
|---|---|---|
| 无 rerank | `DOSER_ENABLE_RERANK=0` | 验证 DOSER 仅 critic 正则的效果 |
| 无 Q rerank | `DOSER_RERANK_ALPHA_Q=0` | 验证 auxiliary Q 对动作选择的贡献 |
| 无 reward prior | `DOSER_RERANK_GAMMA_REWARD=0` | 验证即时 reward 先验贡献 |
| 无 action OOD penalty | `DOSER_RERANK_BETA_ACTION_OOD=0` | 验证 action OOD 对最终动作的影响 |
| 更宽 state OOD | `DOSER_STATE_THRESHOLD_SCALE=3.0` | 检查 state OOD 是否过敏 |
| 更强探索 | `DOSER_RANDOM_CANDIDATES=32` | 检查策略是否过早收敛 |
| 更平滑 rerank | `DOSER_RERANK_TEMPERATURE=1.5` | 降低 rerank 过度尖锐的问题 |

## 14. 当前风险与后续方向

### 14.1 当前风险

1. rerank 会增加训练开销，主要来自扩散采样、action OOD error 和 auxiliary Q 计算。
2. learn 阶段重新生成候选，和 collection 阶段候选不完全一致；但 A2C 不依赖 old log-prob，且真实动作已强制加入候选。
3. state OOD threshold 仍需根据 SwanLab 日志调节。
4. 目前尚未完成全量训练验证，不能断言已经超过 DARLR baseline。

### 14.2 后续方向

1. 将 full positive/negative OOD 细粒度逻辑进一步引入 rerank。
2. 对候选动作构造 counterfactual next state，并在 rerank 阶段同时考虑 state OOD。
3. 缓存扩散候选和 OOD error，降低训练成本。
4. 针对 KuaiEnv-v0 调参，使 `Trajectory Reward`、`CTR`、`Trajectory length` 和 `MCD` 接近或超过 DARLR。

## 15. 一句话总结

当前改进方案是在 DORL/A2C on-policy 框架上保留 DOSER critic 端细粒度 OOD penalty / compensation，同时新增扩散引导 rerank 动作出口，使行为扩散模型、auxiliary Q、reward prior 和 action OOD penalty 能共同影响最终推荐动作。
