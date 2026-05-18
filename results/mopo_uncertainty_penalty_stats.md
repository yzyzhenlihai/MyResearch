# MOPO 不确定性惩罚量级统计

生成时间：2026-03-31 UTC

## 说明范围

这份文档总结了当前代码库中 MOPO 所使用的 uncertainty penalty 的量级。

惩罚项实现在 `src/core/envs/Simulated_Env/penalty_var.py` 中：

```python
penalized_reward = pred_reward - self.lambda_variance * max_var - self.MIN_R
```

真正代表“不确定性惩罚强度”的部分是：

```python
lambda_variance * max_var
```

其中 `- self.MIN_R` 只是一个全局平移项，用来把 reward 调整到正区间，不应被理解为 uncertainty 本身的强度。

## 数据来源

下面的统计量是基于当前工作区中实际存在的矩阵文件计算得到的：

- `saved_models/KuaiEnv-v0/DeepFM/matsPre/[pointneg]_matPre.pickle`
- `saved_models/KuaiEnv-v0/DeepFM/matsVar/[pointneg]_matVar.pickle`

矩阵形状如下：

- `predicted_mat`: `(1411, 3327)`
- `maxvar_mat`: `(1411, 3327)`

需要注意：

- 这些数值对应的是当前仓库里已有的 `DeepFM + pointneg` 矩阵。
- 如果某次具体的 MOPO 运行使用了不同的 `read_message`、`user_model_name`，或者重新生成了矩阵，那么实际数值会发生变化。

## 核心结论

在 MOPO 默认设置 `lambda_variance = 0.05` 下，不确定性惩罚通常明显小于基础的预测 reward。

- 惩罚项的典型量级大约在 `1e-6` 到 `1e-5`
- `pred_reward` 的典型量级大约在 `1e-4` 到 `1e-3`
- 结论：默认惩罚通常比 reward 主体小 1 到 2 个数量级

这意味着默认惩罚整体上是比较温和的，但在 top 候选分数间隔非常小的时候，仍然可能改变排序结果。

## 分布统计

### 1. 预测 reward `pred_reward`

| 指标 | 数值 |
| --- | ---: |
| min | -0.014028804749250412 |
| mean | 0.0001464662597567236 |
| std | 0.0017499976405377677 |
| p50 | 1.6796961426734923e-05 |
| p95 | 0.002317509427666664 |
| p99 | 0.005930995717644693 |
| max | 0.046700384095311166 |

### 2. 不确定性 `max_var`

| 指标 | 数值 |
| --- | ---: |
| min | 4.539992369245738e-05 |
| mean | 6.325357535985755e-05 |
| std | 0.00022641157673233204 |
| p50 | 4.539992369245738e-05 |
| p95 | 5.610916196019391e-05 |
| p99 | 0.0003867062658537187 |
| max | 0.033911317586898804 |

### 3. 在 `lambda_variance = 0.05` 下的惩罚项

即：

```python
penalty = 0.05 * max_var
```

| 指标 | 数值 |
| --- | ---: |
| min | 2.269996184622869e-06 |
| mean | 3.162678767992877e-06 |
| std | 1.1320578836616601e-05 |
| p50 | 2.269996184622869e-06 |
| p95 | 2.8054580980096955e-06 |
| p99 | 1.9335313292685936e-05 |
| max | 0.0016955658793449402 |

## 相对量级比较

### 全局比较

| 比较项 | 数值 |
| --- | ---: |
| predicted reward range | 0.06072918884456158 |
| penalty range at `lambda=0.05` | 0.0016932958831603173 |
| mean penalty / mean predicted reward | 0.021593224086188852 |
| p95 penalty / p95 predicted reward | 0.001210548731546828 |
| max penalty / max predicted reward | 0.03630732192447169 |
| penalty std / predicted reward std | 0.0064689109141529075 |

解释：

- 从均值看，惩罚项约为平均预测 reward 的 `2.16%`
- 从 95 分位看，惩罚项约为预测 reward 的 `0.12%`
- 从标准差看，惩罚项约为 reward 波动的 `0.65%`

### 按用户的排序尺度比较

这里比较的是：对每个 user 而言，惩罚项在不同 item 之间的变化幅度，与原始预测 reward 在不同 item 之间的变化幅度相比有多大。

`penalty range / predicted reward range`：

| 指标 | 数值 |
| --- | ---: |
| mean | 0.008543556397322582 |
| median | 0.006853373049299568 |
| p95 | 0.019928612517020227 |
| max | 0.04688006601504775 |

`penalty std / predicted reward std`：

| 指标 | 数值 |
| --- | ---: |
| mean | 0.00464915272812824 |
| median | 0.0034251538471528334 |
| p95 | 0.01266999784893948 |
| max | 0.03057045225854903 |

解释：

- 对一个典型 user 来说，uncertainty penalty 的变化通常不到原始打分变化的 `1%`
- 即使到 95 分位，按 range 也只有约 `2%`，按标准差也只有约 `1.27%`
- 所以在 `lambda=0.05` 下，惩罚项通常不足以大幅重塑整体排序

## 它会改变 Top-1 推荐吗？

这里直接比较了：

- `argmax(pred_reward)`
- `argmax(pred_reward - lambda_variance * max_var)`

| lambda_variance | top-1 发生变化的用户数 | 比例 |
| --- | ---: | ---: |
| 0.01 | 4 / 1411 | 0.002834868887313962 |
| 0.05 | 24 / 1411 | 0.01700921332388377 |
| 0.10 | 45 / 1411 | 0.031892274982282066 |
| 0.50 | 219 / 1411 | 0.1552090715804394 |
| 1.00 | 400 / 1411 | 0.28348688873139616 |

解释：

- 在默认 `lambda=0.05` 下，大约只有 `1.70%` 的用户会发生 top-1 改变
- 所以这个惩罚项并不主导排序，但也不是完全没有作用
- 它主要影响那些本来 top 候选就非常接近的边界样本

## 为什么这个惩罚看起来偏弱

方差矩阵 `maxvar_mat` 大量集中在它的下界附近：

| 统计项 | 数值 |
| --- | ---: |
| min variance | 4.539992369245738e-05 |
| fraction exactly at min variance | 0.7490489193819782 |
| fraction within `1e-7` of min | 0.8016307525758899 |
| fraction within `1.1x` of min | 0.9294942460128532 |

解释：

- 大约 `74.9%` 的位置都正好处在最小方差上
- 大约 `92.9%` 的位置都不超过最小方差的 `1.1x`
- 因此，对大多数 item 而言，这个惩罚更像是一个接近常数的偏移
- 接近常数的偏移对排序影响天然有限

## 加上全局平移后的最终 reward 量级

如果使用环境里的完整公式，并令 `lambda_variance = 0.05`：

```python
MIN_R = predicted_mat.min() - lambda_variance * maxvar_mat.max()
reward = pred_reward - lambda_variance * max_var - MIN_R
```

那么最终传给 policy 的 reward 分布会变成：

| 指标 | 数值 |
| --- | ---: |
| min | 0.0016932958831603149 |
| mean | 0.01586767420958408 |
| std | 0.0017447127216350799 |
| p50 | 0.015738797079393407 |
| p95 | 0.018038944345462368 |
| p99 | 0.021627690032590182 |
| max | 0.06180702089332044 |

这里需要特别注意：

- 这个较大的正 reward 量级主要是由于减掉了 `MIN_R`
- 它并不意味着 uncertainty penalty 本身很大

## 最终结论

- 在当前这份可用矩阵上，MOPO 的 uncertainty penalty 通常明显小于基础预测 reward
- 在默认 `lambda_variance = 0.05` 下，它整体上比较温和，只会让约 `1.7%` 的用户改变 top-1 推荐
- 主要原因是 `maxvar_mat` 高度集中在最小值附近，因此惩罚项在大多数位置上都接近常数偏移
- 如果想让保守性更强，单纯依赖当前默认值往往不够，通常需要增大 `lambda_variance`，或者使用区分度更强的不确定性估计
