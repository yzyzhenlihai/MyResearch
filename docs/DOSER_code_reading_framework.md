# DOSER 代码阅读与 OOD 迁移框架

## 1. 文档目的

这份文档的目标不是逐行解释源码，而是把 DOSER 论文中的关键创新点，映射到当前仓库里的具体实现位置，并进一步抽象出一套可以迁移到其他任务的细粒度 OOD 识别框架。

阅读这份文档时，建议始终区分三层内容：

1. `论文方法层`
   这里关注论文第 3.1-3.4 节提出了什么机制，尤其是公式 `(4)-(14)` 对应的算法设计。
2. `代码实现层`
   这里关注当前仓库是如何把论文主干落到 `main.py`、`pretrain_diffusion.py`、`pretrain_dynamics.py` 和 `agents/doser.py` 中的。
3. `迁移抽象层`
   这里关注哪些设计可以直接复用，哪些必须适配，哪些本质上只在离线 RL 场景中成立。

论文主来源：

- OpenReview 正式版本：<https://openreview.net/pdf?id=a4DbIONcpb>

需要特别记住的一点是：当前仓库应视为分析对象本体。也就是说，即使论文和代码之间存在工程化近似，我们也优先按“代码的真实执行链”来理解方法。

## 2. 一张图看完整个仓库

如果只想先建立整体心智模型，可以把仓库理解成下面这 7 个角色：

| 模块 | 作用 | 在阅读链中的位置 |
| --- | --- | --- |
| `main.py` | 训练入口，负责组装所有子模块 | 第一站 |
| `pretrain_diffusion.py` | 预训练两个 diffusion model，并基于训练集误差计算 OOD 阈值 | 第二站 |
| `pretrain_dynamics.py` | 预训练动力学模型 `p(s'\|s,a)` | 第三站 |
| `agents/doser.py` | DOSER 的核心逻辑，尤其是 OOD 分类与选择性正则 | 第四站，最关键 |
| `agents/models.py` | Actor / Critic / Value 相关网络结构 | 与 `agents/doser.py` 配套阅读 |
| `diffusion/karras.py` | diffusion 训练与采样的底层实现 | 与 `pretrain_diffusion.py` 配套阅读 |
| `diffusion/mlps.py` | diffusion score network 的具体网络实现 | 与 `diffusion/karras.py` 配套阅读 |

一句话概括：

- `main.py` 负责“拼系统”
- `pretrain_diffusion.py` 负责“学分布 + 算阈值”
- `pretrain_dynamics.py` 负责“学后果预测器”
- `agents/doser.py` 负责“根据 OOD 类型做不同处理”

## 3. 从运行入口出发的主调用链

推荐先顺着 `main.py` 的执行顺序读一遍，这样后面看 `doser.py` 时不会迷路。

### 3.1 配置与超参数装配

入口从 [main.py](/data/yuzhengyang/RL_Learning/DOSER/main.py:112) 开始。

关键内容：

- 解析环境名、设备、训练步数等参数
- 从 `hyperparameters` 中按数据集加载 `beta`、`lam`、`eta`、`expectile`、`percentile`、`state_levels`、`action_levels`、`action_samples`、`Q_min`

关键代码位置：

- 超参数表：[main.py](/data/yuzhengyang/RL_Learning/DOSER/main.py:19)
- 参数注入与解释：[main.py](/data/yuzhengyang/RL_Learning/DOSER/main.py:131)

这里的重要结论是：论文里的很多方法超参数，在代码里是“按数据集预先配置”的，而不是统一默认值。

### 3.2 数据集加载与归一化

接下来 `main.py` 会创建环境、读取 D4RL 数据集，并把状态写入 `ReplayBuffer`。

关键代码位置：

- 数据集装载与归一化：[main.py](/data/yuzhengyang/RL_Learning/DOSER/main.py:155)
- `ReplayBuffer` 实现：[utils.py](/data/yuzhengyang/RL_Learning/DOSER/utils.py:5)

这里要注意：

- `normalize_states()` 会同时归一化 `state` 和 `next_state`
- 后续 diffusion 状态建模、动力学预测和 critic 训练都默认工作在这套状态表示上

### 3.3 载入 diffusion / dynamics 模型

`main.py` 的下一步是把三个预训练模块接入在线训练：

1. 行为扩散模型 `behavior_model`
2. 状态分布扩散模型 `state_distribution`
3. 动力学模型 `dynamics_model`

关键代码位置：

- diffusion 模型装配：[main.py](/data/yuzhengyang/RL_Learning/DOSER/main.py:178)
- 动力学模型装配：[main.py](/data/yuzhengyang/RL_Learning/DOSER/main.py:218)

这一段对应论文算法 1 的模型预训练结果载入阶段。

### 3.4 计算 OOD 阈值

在开始策略训练之前，`main.py` 会先基于离线数据集统计重构误差，并用分位数得到：

- `state_threshold`
- `action_threshold`

关键代码位置：

- 阈值计算入口：[main.py](/data/yuzhengyang/RL_Learning/DOSER/main.py:225)
- 阈值函数实现：[pretrain_diffusion.py](/data/yuzhengyang/RL_Learning/DOSER/pretrain_diffusion.py:44)

这一步非常关键，因为它把论文公式 `(8)` 中的 `τ_a` 和 `τ_s` 具体落成了代码里的阈值标定流程。

### 3.5 组装 DOSER 并进入训练循环

最后，`main.py` 把所有组件传给 `DOSER(**kwargs)`，再在训练循环里持续调用 `agent.train()`。

关键代码位置：

- DOSER 初始化：[main.py](/data/yuzhengyang/RL_Learning/DOSER/main.py:231)
- 训练循环：[main.py](/data/yuzhengyang/RL_Learning/DOSER/main.py:261)

到这里为止，你已经知道了系统的“外骨架”。真正体现论文创新的部分，从这里转到 `agents/doser.py`。

## 4. 论文创新点到代码位置的映射

这一节是本文档的核心。

### 4.1 创新点一：用 diffusion reconstruction error 做 OOD 检测

### 论文在说什么

论文第 3.1 和 3.2 节提出：

- 用条件 diffusion model 学习经验行为策略 $\hat{\pi}_\beta(a|s)$，对应公式 `(4)`
- 用无条件 diffusion model 学习状态分布，公式 $(5)$
- 对输入样本加噪并做单步去噪，利用重构误差作为 OOD score，公式 `(6)` 和 `(7)`
- 用训练集误差的 `p` 分位数定义阈值 $τ_a$ 和 $τ_s$，公式 `(8)`

OpenReview 中对应的位置：

- 3.1 行为与状态建模：第 3 节 3.1，小节起点约见论文 PDF 第 3 页
- 3.2 重构误差检测：OpenReview PDF 行 `233-255`

### 代码是怎么实现的

#### 1. 预训练两个 diffusion model

关键代码：

- 状态分布 diffusion 训练：[pretrain_diffusion.py](/data/yuzhengyang/RL_Learning/DOSER/pretrain_diffusion.py:141)
- 行为策略 diffusion 训练：[pretrain_diffusion.py](/data/yuzhengyang/RL_Learning/DOSER/pretrain_diffusion.py:154)

对应关系：

- `state_distribution` 对应论文中的状态分布 denoiser $\epsilon_{\theta_s}$
- `behavior_model` 对应论文中的条件动作 denoiser $\epsilon_{\theta_a}$

#### 2. 计算训练集上的 reconstruction error 分位数

关键代码：

- `get_state_threshold()`：[pretrain_diffusion.py](/data/yuzhengyang/RL_Learning/DOSER/pretrain_diffusion.py:44)
- `get_action_threshold()`：[pretrain_diffusion.py](/data/yuzhengyang/RL_Learning/DOSER/pretrain_diffusion.py:52)

这部分正是论文公式 `(8)` 的直接落地：

- 先在训练数据上计算 $E_s$ / $E_a$
- 再用 `np.percentile(..., percentile)` 取阈值

#### 3. 训练时计算当前策略动作和预测后继状态的 OOD 分数

关键代码：

- 状态误差：[agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:86)
- 动作误差：[agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:118)

这里有一个实现细节值得注意：

- 论文强调“跨多个随机 diffusion timestep 评估误差能更稳健”
- 代码通过 `state_levels` 和 `action_levels` 做多次随机噪声级别采样后取平均，正好对应了这个思想

#### 4. diffusion 底座是谁在做

关键代码：

- 训练损失与加噪去噪：[diffusion/karras.py](/data/yuzhengyang/RL_Learning/DOSER/diffusion/karras.py:50)
- 采样函数：[diffusion/karras.py](/data/yuzhengyang/RL_Learning/DOSER/diffusion/karras.py:181)
- score network：[diffusion/mlps.py](/data/yuzhengyang/RL_Learning/DOSER/diffusion/mlps.py:66)

### 这一创新点里哪些能迁移

#### 直接复用

- 条件分布模型 + 无条件分布模型的双模型结构
- 单步加噪去噪后的重构误差作为 OOD score
- 使用训练集误差分位数做阈值标定
- 在多个随机噪声级别上平均误差来提升鲁棒性

#### 需要适配

- 当前 `state_distribution` 建模的是“状态空间分布”，迁到别的任务时，应该替换成你的目标空间分布
- 如果目标任务不是连续动作空间，`behavior_model` 的条件生成对象也要换成新的样本类型

#### 仅 RL 语境下的解释

- 当前动作 OOD 的含义是“偏离行为策略支持集”
- 当前状态 OOD 的含义是“预测后的 next state 脱离离线数据状态分布”

## 4.2 创新点二：细粒度 OOD 识别，不止区分 ID / OOD

### 论文在说什么

论文第 3.3 节提出：一个动作被判为 OOD 之后，不应直接统一惩罚，而应进一步区分为：

- `beneficial OOD`
- `detrimental OOD`

判别依据有两层：

1. 该动作导致的预测后继状态是否仍在状态分布内
2. 如果后继状态仍可信，其价值是否优于“参考 ID 动作”导致的后继状态

OpenReview 中对应的位置：

- 定义与判别规则：OpenReview PDF 行 `302-364`
- 形式化定义：公式 `(9)`

### 代码是怎么实现的

#### 1. 先为每个状态生成一个“ID 参考动作”

关键代码：

- `select_best_id_action()`：[agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:160)

函数逻辑：

1. 从 diffusion 行为模型中为每个状态采样多个候选动作
2. 用 critic 对每个候选动作估值
3. 选出 Q 值最高的那个作为近似的 `\hat{a}^*_{id}`

这对应论文公式 `(11)` 的工程实现。

#### 2. 用动力学模型预测两个动作的后果

关键代码：

- 当前策略动作的后继状态：`pred_next_state`
- ID 参考动作的后继状态：`best_id_next_state`
- 对应实现：[agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:246)

动力学模型本体与训练：

- 模型定义：[pretrain_dynamics.py](/data/yuzhengyang/RL_Learning/DOSER/pretrain_dynamics.py:15)
- 训练流程：[pretrain_dynamics.py](/data/yuzhengyang/RL_Learning/DOSER/pretrain_dynamics.py:81)

#### 3. 结合状态 OOD 与价值比较，生成两类 mask

关键代码：

- `ood_action_mask`：动作是否 OOD
- `ood_next_state_mask`：后继状态是否 OOD
- `negative_value_mask` / `positive_value_mask`：后继价值是否优于 ID 参考
- `negative_ood_action_mask` / `positive_ood_action_mask`：最终分类结果

位置：

- [agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:259)

这段代码基本就是论文公式 `(9)` 的直接布尔化实现：

- 如果动作 OOD 且后继状态 OOD，或者后继价值更差，那么是 `negative`
- 如果动作 OOD 且后继状态仍在分布内，并且后继价值不低于 ID 参考，那么是 `positive`

### 把这部分抽象成通用四步框架

如果你想迁走的不是整套离线 RL，而是“细粒度 OOD 识别思想”，最值得带走的是下面这个四步框架：

#### Step A: 候选 OOD 检测

先用生成式或重构式模型筛出“偏离训练分布”的候选样本。

仓库对应：

- `compute_action_error()`
- `compute_state_error()`

#### Step B: 生成 ID 参考样本

为同一上下文生成或检索一个“代表性的 ID 参考候选”。

仓库对应：

- `select_best_id_action()`

#### Step C: 比较候选样本与参考样本的后果

不要只看样本本身是否越界，还要比较它们带来的后果。

仓库对应：

- `pred_next_state`
- `best_id_next_state`
- `value_s_pi`
- `value_s_in`

#### Step D: 按 OOD 子类型采取不同处理

同样是 OOD，负向 OOD 与正向 OOD 不能用同一种规则处理。

仓库对应：

- `negative_ood_action_mask`
- `positive_ood_action_mask`

### 这一创新点里哪些能迁移

#### 直接复用

- “先检测，再细分”的两阶段 OOD 处理思路
- “与 ID 参考样本比较”的判别方式
- 用“后果比较”而不是“距离比较”来划分 OOD 子类型

#### 需要适配

- `dynamics_model` 可以换成任何后果预测器、打分器、表征变化模型
- `V(s'_\pi) >= V(s'_{id})` 可以换成任何任务相关效用函数
- `best_id_action` 的产生方式可以换成近邻检索、prototype retrieval、memory bank、类条件生成

#### 仅 RL 语境下的解释

- 当前“后果”是 next state
- 当前“效用”是 value function
- 当前“ID 参考”是来自行为策略支持集中的高价值动作

## 4.3 创新点三：选择性正则，而不是统一惩罚

### 论文在说什么

论文第 3.3 节和公式 `(10)` 的核心观点是：

- 对 `detrimental OOD` 施加保守惩罚
- 对 `beneficial OOD` 提供补偿目标
- 避免把所有越界行为都统一压回数据集内部

这就是 DOSER 名字里 “Selective Regularization” 的来源。

### 代码是怎么实现的

关键位置集中在 `critic_loss()`：

- critic 主体损失：[agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:209)
- 负向 OOD 惩罚 `reg_loss`：[agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:266)
- 正向 OOD 补偿 `vc_loss`：[agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:272)

#### 负向 OOD 分支

```python
reg_loss = self.beta * (((pi_Q - qmin) ** 2) * negative_ood_action_mask).mean()
```

含义：

- 对被判为 `negative OOD` 的策略动作，把 Q 值向 `Q_min` 拉
- 这相当于把“危险越界动作”推回保守下界

#### 正向 OOD 分支

```python
value_diff = (value_s_pi - value_s_in).clamp(min=0.0)
q_comp_target = self.eta * (best_id_q + value_diff).detach()
vc_loss = self.lam * (((pi_Q - q_comp_target) ** 2) * positive_ood_action_mask).mean()
```

含义：

- 对被判为 `positive OOD` 的动作，不是简单放行，而是给它一个“相对于 ID 参考动作更高但受控”的目标
- 这部分体现了论文中的 compensation / bonus 逻辑

### 这一创新点里哪些能迁移

#### 直接复用

- 不同 OOD 子类使用不同训练信号
- 对负向 OOD 采用保守拉回
- 对正向 OOD 采用受控鼓励，而不是完全信任

#### 需要适配

- 负向目标未必一定是 `Q_min`，也可以是拒识分数、风险上界、低置信目标
- 正向目标未必一定是 `best_id_q + value_diff`，也可以是 margin、utility bonus、ranking target

#### 仅 RL 适用

- `Q_min`
- critic 上的 value compensation
- 基于 Bellman 误差的训练框架

## 5. 代码中的 RL 专属部分与可迁移边界

如果你只想借用 OOD 细粒度识别方案，不想把整个 DOSER 都搬走，那么下面这个边界一定要分清。

### 5.1 可以视为通用 OOD 框架的部分

可以抽象成以下接口：

```text
OODScore(sample, context) -> scalar
ThresholdCalibrator(train_scores) -> tau
ReferenceGenerator(context) -> id_reference
OutcomeComparator(candidate, reference, context) -> positive_ood | negative_ood
SelectiveHandler(type, candidate, reference) -> training_target_or_decision
```

在本仓库里的近似对应是：

- `OODScore` -> `compute_action_error()` / `compute_state_error()`
- `ThresholdCalibrator` -> `get_action_threshold()` / `get_state_threshold()`
- `ReferenceGenerator` -> `select_best_id_action()`
- `OutcomeComparator` -> `critic_loss()` 中的 mask 构造逻辑
- `SelectiveHandler` -> `reg_loss` 和 `vc_loss`

### 5.2 需要适配但思想可迁移的部分

这些模块不建议照抄实现，但建议保留其“角色”：

| DOSER 里的角色 | 当前实现 | 迁移时可替换成 |
| --- | --- | --- |
| 候选 OOD 检测器 | diffusion reconstruction error | 自编码器、扩散模型、能量模型、表征一致性模型 |
| ID 参考生成器 | diffusion action sampling + Q 选优 | prototype 检索、近邻支持集、条件生成器 |
| 后果预测器 | dynamics model | 表征演化模型、重排序器、风险估计器 |
| 效用比较器 | `V(s'_\pi)` vs `V(s'_{id})` | 置信提升、margin、校准收益、任务效用 |

### 5.3 基本只在离线 RL 场景里成立的部分

这些内容通常不该原样迁走：

- 四头 `Q` 网络结构：[agents/models.py](/data/yuzhengyang/RL_Learning/DOSER/agents/models.py:22)
- 双 `V` 头和 `expectile regression`：[agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:229)
- 熵正则 actor 更新：[agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:198)
- 自适应温度 `alpha`：[agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:191)
- target network 更新：[agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:282)
- `Q_min = R_min / (1 - gamma)` 的理论解释

## 6. 如果把 DOSER 的细粒度 OOD 识别迁到“识别/分类”任务

如果你的目标不是离线 RL，而是更一般的细粒度识别、OOD 识别或开放世界分类，那么建议按下面的方式改写 DOSER。

### 6.1 建议保留的主框架

保留这条逻辑链：

1. 先检测样本是否偏离训练分布
2. 如果偏离，不急着直接拒绝
3. 为当前样本找到一个可信的 ID 参考
4. 比较两者的“后果”或“任务效用”
5. 将 OOD 分成可利用与不可利用两类
6. 对两类样本施加不同训练策略

### 6.2 在识别任务里的替身关系

你可以把 DOSER 里的变量理解为：

| DOSER 概念 | 在识别任务中的可替身 |
| --- | --- |
| `action` | 当前待判别样本、局部 patch、候选特征、伪标签实例 |
| `state` | 样本上下文、图像特征、query 条件、类别上下文 |
| `behavior model` | 训练集内分布建模器、类条件生成器、特征重构器 |
| `state_distribution` | 特征空间或中间表征空间的 ID 分布模型 |
| `best_id_action` | 最相近 prototype、支持集近邻、历史高置信 ID 参考 |
| `dynamics_model` | 特征变化预测器、后续模块响应预测器、重排序器 |
| `value` | 分类收益、置信稳定性、margin、校准质量、下游效用 |

### 6.3 可以怎么落地

#### 只迁移 OOD detector

适合你只想得到“更稳的 OOD 分数”时使用：

- 保留 diffusion / reconstruction error
- 保留 percentile threshold
- 不引入参考样本和后果比较器

#### 迁移细粒度 OOD 分类

适合你想区分“应该拒绝的 OOD”和“可以吸收利用的边界样本”：

- 加入 `ReferenceGenerator`
- 加入 `OutcomeComparator`
- 用正向/负向 OOD 的二路处理代替单一路径过滤

#### 全量迁移 DOSER 思路

适合你想把它改造成一个完整训练策略：

- 有候选 OOD 检测器
- 有 ID 参考生成器
- 有后果预测器
- 有针对正负 OOD 的不同优化目标

## 7. 推荐阅读顺序

如果你是第一次读这个仓库，建议按下面顺序走。

### 第一轮：15 分钟，先看主干

1. [README.md](/data/yuzhengyang/RL_Learning/DOSER/README.md:1)
2. [main.py](/data/yuzhengyang/RL_Learning/DOSER/main.py:112)
3. [agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:209)

目标：

- 先知道系统是怎么跑起来的
- 知道 OOD 分类与正则大致发生在哪里

### 第二轮：40-60 分钟，把论文创新点逐个对上

1. `pretrain_diffusion.py`
2. `diffusion/karras.py`
3. `agents/doser.py` 中的 `compute_*_error`、`select_best_id_action()`、`critic_loss()`
4. `pretrain_dynamics.py`

目标：

- 把“扩散建模 -> 重构误差 -> 阈值 -> OOD 分类 -> 选择性正则”串起来

### 第三轮：迁移视角阅读

只读下面这些函数：

- [pretrain_diffusion.py](/data/yuzhengyang/RL_Learning/DOSER/pretrain_diffusion.py:44)
- [agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:86)
- [agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:160)
- [agents/doser.py](/data/yuzhengyang/RL_Learning/DOSER/agents/doser.py:209)

目标：

- 直接抽出最核心的 4 个角色：打分、校准、参考生成、后果比较

## 8. 论文与当前代码的几个关键对齐点

为了避免读代码时产生误判，这里列几个最重要的“论文-代码对齐关系”。

### 8.1 对齐点

- 论文的公式 `(4)`、`(5)` 对应 `pretrain_diffusion.py` 中两个 denoiser 的训练
- 论文的公式 `(6)`、`(7)` 对应 `compute_action_error()` / `compute_state_error()`
- 论文的公式 `(8)` 对应 `get_action_threshold()` / `get_state_threshold()`
- 论文的公式 `(9)` 对应 `positive_ood_action_mask` / `negative_ood_action_mask` 的构造
- 论文的公式 `(10)` 对应 `reg_loss` 与 `vc_loss`
- 论文的公式 `(11)` 对应 `select_best_id_action()`
- 论文的公式 `(13)` 对应 `pretrain_dynamics.py` 的监督回归
- 论文的公式 `(14)` 对应 actor 的最大熵更新

### 8.2 当前代码里的工程化实现

- 论文将 `Q` 与 `V` 作为概念上分开的网络，代码把它们收进同一个 `Critic` 类里
- 论文描述了 `a^*_{id}` 的近似采样实现，代码用 diffusion 采样 `N` 个候选动作再用 critic 选优
- 论文中“多噪声级别更稳健”的表述，在代码里通过 `state_levels` / `action_levels` 实现

### 8.3 论文里有、当前代码里没有显式实现的内容

论文附录提到过 `ensemble-guided gating mechanism`，即用动力学模型集成的不确定性过滤不可靠预测，再决定是否信任 OOD 分类结果。

OpenReview 对应位置：

- PDF 行 `2799-2929`

但当前仓库中没有对应的 ensemble dynamics 实现，因此阅读本代码时不要把这部分当作现有功能。

## 9. 最后给你的迁移建议

如果你的重点是“借用论文中的 OOD 细粒度识别方案”，我建议不要从“复现 DOSER 全部 RL 机制”开始，而是优先抽出下面这条最小可迁移链：

1. 用生成式或重构式模型定义 OOD score
2. 用训练集统计量定义阈值
3. 为每个候选样本生成一个 ID 参考样本
4. 定义候选样本与参考样本的后果比较函数
5. 把 OOD 分成正向与负向两类
6. 对两类样本施加不同训练信号

如果只从当前仓库里挑“最值得迁”的代码角色，优先级建议如下：

### 第一优先级

- `compute_action_error()`
- `compute_state_error()`
- `get_action_threshold()`
- `get_state_threshold()`

这些是 OOD detector 的核心。

### 第二优先级

- `select_best_id_action()`
- `positive_ood_action_mask`
- `negative_ood_action_mask`

这些是细粒度 OOD 分类的核心。

### 第三优先级

- `reg_loss`
- `vc_loss`

这些是“根据 OOD 子类型给不同训练信号”的核心。

### 最后再考虑

- actor / critic / target network / alpha / expectile / Q_min

这些更多是 DOSER 在离线 RL 中的承载壳，而不是你真正想迁移的 OOD 识别思想本身。

