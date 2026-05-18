# MOPO `lambda_variance` 标定报告

## 摘要
- 参考 DORL 熵项口径：`lambda_entropy=0.05`，`entropy_window=[1, 2]`，`feature_level=True`，`is_sorted=True`。
- 训练分布：`big_matrix_processed.csv` 中可映射到 KuaiEnv small-space 的交互，实际回放源为 `data/KuaiRec/data_raw/small_matrix_processed.csv`，共 `4676570` 条，涉及 `1411` 个用户。
- 标定中心值：`lambda_star = 1458.48`。
- 官方 5 点单种子方案：`[0.0, 0.05, 146.0, 486.0, 1460.0]`。

## 数据与对齐
- `big_csv`: `data/KuaiRec/data_raw/big_matrix_processed.csv`
- `prediction_mat_path`: `saved_models/KuaiEnv-v0/DeepFM/matsPre/[pointneg]_matPre.pickle`
- `var_mat_path`: `saved_models/KuaiEnv-v0/DeepFM/matsVar/[pointneg]_matVar.pickle`
- `small_csv`: `data/KuaiRec/data_raw/small_matrix_processed.csv`
- `predicted_mat.shape`: `[1411, 3327]`
- `maxvar_mat.shape`: `[1411, 3327]`
- `KuaiEnv small-space shape`: `[1411, 3327]`
- `replay_csv_used`: `data/KuaiRec/data_raw/small_matrix_processed.csv`

## 惩罚量级统计

| 项 | count | mean | median | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| entropy_t | 4676570 | 1.58552 | 1.58563 | 1.63651 | 1.65565 | 1.83224 |
| 0.05 * entropy_t | 4676570 | 0.0792761 | 0.0792817 | 0.0818255 | 0.0827824 | 0.091612 |
| maxvar_t | 4676570 | 6.32576e-05 | 4.53999e-05 | 5.61032e-05 | 0.000386731 | 0.0339113 |
| 0.05 * maxvar_t | 4676570 | 3.16288e-06 | 2.27e-06 | 2.80516e-06 | 1.93365e-05 | 0.00169557 |

## 标定结果
- `lambda_mean = 1253.23`
- `lambda_median = 1746.3`
- `lambda_star = 1458.48`
- `official_mopo_sweep = [0.0, 0.05, 146.0, 486.0, 1460.0]`

## MOPO reward sanity check

| lambda_variance | reward mean | reward p95 | reward max |
| ---: | ---: | ---: | ---: |
| 0 | 0.0141753 | 0.0163464 | 0.0607292 |
| 0.05 | 0.0158678 | 0.018039 | 0.061807 |
| 146 | 4.95599 | 4.96038 | 4.97003 |
| 486 | 16.4643 | 16.4747 | 16.4834 |
| 1460 | 49.4323 | 49.4601 | 49.4672 |

## 历史 MOPO λ
- 现有日志里已发现的 `lambda_variance`：`[0.0, 0.01, 0.05]`
- 其中 `0.01` 已经存在，可作为附录对比点，不占这次 5 个正式点位。

## 手动训练命令
- `CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=.:./src:./src/DeepCTR-Torch SWANLAB_MODE=offline conda run -n easyrl4rec python examples/advance/run_MOPO.py --env KuaiEnv-v0 --user_model_name DeepFM --read_message pointneg --seed 2023 --cuda <cuda_id> --which_tracker avg --window_size 3 --epoch 100 --batch-size 1024 --hidden-sizes 64 64 --leave_threshold 0 --num_leave_compute 1 --max_turn 30 --force_length 10 --lambda_entropy 0.0 --lambda_variance 0 --message MOPO_lambda_0`
- `CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=.:./src:./src/DeepCTR-Torch SWANLAB_MODE=offline conda run -n easyrl4rec python examples/advance/run_MOPO.py --env KuaiEnv-v0 --user_model_name DeepFM --read_message pointneg --seed 2023 --cuda <cuda_id> --which_tracker avg --window_size 3 --epoch 100 --batch-size 1024 --hidden-sizes 64 64 --leave_threshold 0 --num_leave_compute 1 --max_turn 30 --force_length 10 --lambda_entropy 0.0 --lambda_variance 0.05 --message MOPO_lambda_0p05`
- `CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=.:./src:./src/DeepCTR-Torch SWANLAB_MODE=offline conda run -n easyrl4rec python examples/advance/run_MOPO.py --env KuaiEnv-v0 --user_model_name DeepFM --read_message pointneg --seed 2023 --cuda <cuda_id> --which_tracker avg --window_size 3 --epoch 100 --batch-size 1024 --hidden-sizes 64 64 --leave_threshold 0 --num_leave_compute 1 --max_turn 30 --force_length 10 --lambda_entropy 0.0 --lambda_variance 146 --message MOPO_lambda_146`
- `CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=.:./src:./src/DeepCTR-Torch SWANLAB_MODE=offline conda run -n easyrl4rec python examples/advance/run_MOPO.py --env KuaiEnv-v0 --user_model_name DeepFM --read_message pointneg --seed 2023 --cuda <cuda_id> --which_tracker avg --window_size 3 --epoch 100 --batch-size 1024 --hidden-sizes 64 64 --leave_threshold 0 --num_leave_compute 1 --max_turn 30 --force_length 10 --lambda_entropy 0.0 --lambda_variance 486 --message MOPO_lambda_486`
- `CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=.:./src:./src/DeepCTR-Torch SWANLAB_MODE=offline conda run -n easyrl4rec python examples/advance/run_MOPO.py --env KuaiEnv-v0 --user_model_name DeepFM --read_message pointneg --seed 2023 --cuda <cuda_id> --which_tracker avg --window_size 3 --epoch 100 --batch-size 1024 --hidden-sizes 64 64 --leave_threshold 0 --num_leave_compute 1 --max_turn 30 --force_length 10 --lambda_entropy 0.0 --lambda_variance 1460 --message MOPO_lambda_1460`

## 警告
- big_matrix_processed.csv 中不存在同时落在 KuaiEnv small-space user/item 的交互；已自动回退到 small_matrix_processed.csv 作为回放分布。
