# OOD 识别粒度实验报告

## 1. 实验设置

- 环境：`KuaiEnv-v0`
- user model：`DeepFM`
- read_message：`pointneg`
- 轨迹文件：`data/KuaiRec/data_processed/DM_KuaiEnv-v0_small_data.pkl`
- 原始 CSV：`data/KuaiRec/data_raw/small_matrix_processed.csv`
- `anchor_states_per_user`：`4`
- `use_nx0_mask`：`True`
- `lambda_grid`：`[0.0, 0.05, 146.0, 486.0, 1460.0]`
- 样本表：`results/ood_granularity_experiment_smoke/sample_metrics.csv.gz` (csv_gzip)

## 2. 数据与对齐摘要

- 用户数：`2`
- 候选物品数：`3327`
- 锚点总数：`8`
- 有效锚点数：`8`
- 动作嵌入对齐检查：`{'checked_samples': 4096, 'mean_abs_diff': 0.0, 'max_abs_diff': 0.0, 'allclose': True, 'atol': 1e-05}`

## 3. 标签计数

- 主标签计数：`{'Other': 453, 'NH': 50, 'Near-Low': 48, 'DO': 31, 'Far-High': 14}`
- 四象限计数：`{'Other': 459, 'Near-Low': 54, 'Near-High': 49, 'Far-Low': 19, 'Far-High': 15}`

## 4. 区分能力指标

- `AUC_DO_vs_NH`：`0.7761290322580644`
- `PR-AUC_DO_vs_NH`：`0.7257528461967402`
- `KS(U_current | NH, DO)`：`0.5451612903225806`
- `Wasserstein(U_current | NH, DO)`：`0.00047443019937124897`
- `HighVar` 统计：`{'0.1': {'threshold': 6.173161818878725e-05, 'highvar_count': 15, 'contamination_nh': 0.06666666666666667, 'danger_capture': 0.25806451612903225}, '0.2': {'threshold': 4.677625111071393e-05, 'highvar_count': 29, 'contamination_nh': 0.13793103448275862, 'danger_capture': 0.45161290322580644}}`

## 5. λ 扫描指标

| lambda_value | topk | fsr | drr | gg | nh_loss | do_gain |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0000 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0000 | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0000 | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0500 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0500 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0500 | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0500 | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 146.0000 | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0071 | 0.0764 |
| 146.0000 | 5 | 0.0000 | 0.7500 | 0.7500 | 0.0071 | 0.0764 |
| 146.0000 | 10 | 0.0000 | 0.4286 | 0.4286 | 0.0071 | 0.0764 |
| 146.0000 | 20 | 0.1429 | 0.4000 | 0.2571 | 0.0071 | 0.0764 |
| 486.0000 | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0236 | 0.2542 |
| 486.0000 | 5 | 1.0000 | 0.7500 | -0.2500 | 0.0236 | 0.2542 |
| 486.0000 | 10 | 0.0000 | 0.4286 | 0.4286 | 0.0236 | 0.2542 |
| 486.0000 | 20 | 0.1429 | 0.4000 | 0.2571 | 0.0236 | 0.2542 |
| 1460.0000 | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0710 | 0.7636 |
| 1460.0000 | 5 | 1.0000 | 0.7500 | -0.2500 | 0.0710 | 0.7636 |
| 1460.0000 | 10 | 0.0000 | 0.4286 | 0.4286 | 0.0710 | 0.7636 |
| 1460.0000 | 20 | 0.1429 | 0.4000 | 0.2571 | 0.0710 | 0.7636 |

## 6. 图表

- `d_sa` vs `r_true`：`figures/fig_distance_vs_true_reward.png`
- uncertainty violin：`figures/fig_uncertainty_violin_quadrants.png`
- high-var composition：`figures/fig_highvar_composition.png`
- lambda response：`figures/fig_lambda_response_curves.png`

![distance_vs_true_reward](figures/fig_distance_vs_true_reward.png)

![uncertainty_violin](figures/fig_uncertainty_violin_quadrants.png)

![highvar_composition](figures/fig_highvar_composition.png)

![lambda_response](figures/fig_lambda_response_curves.png)

## 7. 结论草案

若 `AUC_DO_vs_NH` 接近随机、`HighVar` 集合仍混入较多 `NH`，同时随 λ 增大 `FSR` 与 `DRR` 一起上升而 `GG` 不明显为正，则支持“当前不确定性惩罚粒度过粗”的判断。
