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
- 样本表：`results/ood_granularity_experiment/sample_metrics.csv.gz` (csv_gzip)

## 2. 数据与对齐摘要

- 用户数：`1411`
- 候选物品数：`3327`
- 锚点总数：`5644`
- 有效锚点数：`5644`
- 动作嵌入对齐检查：`{'checked_samples': 4096, 'mean_abs_diff': 0.0, 'max_abs_diff': 0.0, 'allclose': True, 'atol': 1e-05}`

## 3. 标签计数

- 主标签计数：`{'Other': 846840, 'Near-Low': 80280, 'NH': 75518, 'DO': 70034, 'Far-High': 26191}`
- 四象限计数：`{'Other': 865588, 'Near-Low': 84301, 'Near-High': 80128, 'Far-Low': 37742, 'Far-High': 31104}`

## 4. 区分能力指标

- `AUC_DO_vs_NH`：`0.7621348210243007`
- `PR-AUC_DO_vs_NH`：`0.7652703257601103`
- `KS(U_current | NH, DO)`：`0.46578846518092937`
- `Wasserstein(U_current | NH, DO)`：`0.00016256982922584822`
- `HighVar` 统计：`{'0.1': {'threshold': 6.0520629631355405e-05, 'highvar_count': 25203, 'contamination_nh': 0.03194064198706503, 'danger_capture': 0.28377645143787306}, '0.2': {'threshold': 4.5779750507790595e-05, 'highvar_count': 50408, 'contamination_nh': 0.07536502142517061, 'danger_capture': 0.49247508353085645}}`

## 5. λ 扫描指标

| lambda_value | topk | fsr | drr | gg | nh_loss | do_gain |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0000 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0000 | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0000 | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.0500 | 1 | 0.0000 | 0.0023 | 0.0023 | 0.0000 | 0.0000 |
| 0.0500 | 5 | 0.0000 | 0.0025 | 0.0025 | 0.0000 | 0.0000 |
| 0.0500 | 10 | 0.0000 | 0.0014 | 0.0014 | 0.0000 | 0.0000 |
| 0.0500 | 20 | 0.0003 | 0.0018 | 0.0015 | 0.0000 | 0.0000 |
| 146.0000 | 1 | 0.1535 | 0.7295 | 0.5759 | 0.0072 | 0.0309 |
| 146.0000 | 5 | 0.0643 | 0.6227 | 0.5584 | 0.0072 | 0.0309 |
| 146.0000 | 10 | 0.0455 | 0.5864 | 0.5409 | 0.0072 | 0.0309 |
| 146.0000 | 20 | 0.0327 | 0.5889 | 0.5562 | 0.0072 | 0.0309 |
| 486.0000 | 1 | 0.1969 | 0.7694 | 0.5726 | 0.0240 | 0.1030 |
| 486.0000 | 5 | 0.0968 | 0.6417 | 0.5450 | 0.0240 | 0.1030 |
| 486.0000 | 10 | 0.0608 | 0.6048 | 0.5440 | 0.0240 | 0.1030 |
| 486.0000 | 20 | 0.0360 | 0.6096 | 0.5736 | 0.0240 | 0.1030 |
| 1460.0000 | 1 | 0.2126 | 0.7957 | 0.5831 | 0.0720 | 0.3093 |
| 1460.0000 | 5 | 0.1250 | 0.6657 | 0.5407 | 0.0720 | 0.3093 |
| 1460.0000 | 10 | 0.0784 | 0.6143 | 0.5358 | 0.0720 | 0.3093 |
| 1460.0000 | 20 | 0.0409 | 0.6146 | 0.5736 | 0.0720 | 0.3093 |

## 6. 图表

- `d_sa` vs `r_true`：`figures/fig_distance_vs_true_reward.png`
- `d_sa` vs `U_current`：`figures/fig_distance_vs_uncertainty.png`
- uncertainty violin (`NH/DO` only, clipped)：`figures/fig_uncertainty_violin_quadrants.png`
- uncertainty bucket mix：`figures/fig_uncertainty_bucket_mix.png`
- high-var composition：`figures/fig_highvar_composition.png`
- lambda response：`figures/fig_lambda_response_curves.png`
- lambda suppressed mix：`figures/fig_lambda_suppressed_mix.png`

![distance_vs_true_reward](figures/fig_distance_vs_true_reward.png)

![distance_vs_uncertainty](figures/fig_distance_vs_uncertainty.png)

![uncertainty_violin](figures/fig_uncertainty_violin_quadrants.png)

![uncertainty_bucket_mix](figures/fig_uncertainty_bucket_mix.png)

![highvar_composition](figures/fig_highvar_composition.png)

![lambda_response](figures/fig_lambda_response_curves.png)

![lambda_suppressed_mix](figures/fig_lambda_suppressed_mix.png)

## 7. 结论草案

若 `AUC_DO_vs_NH` 接近随机、`HighVar` 集合仍混入较多 `NH`，同时随 λ 增大 `FSR` 与 `DRR` 一起上升而 `GG` 不明显为正，则支持“当前不确定性惩罚粒度过粗”的判断。
