# Agent Context Log

## 1. 任务概述
- 任务名称：`reward_prior_real_reward_alignment`
- 当前状态：已完成
- 本轮目标：让 `CounterfactualRewardModel.estimate()` 更贴近真实 KuaiEnv reward，并给出真实 reward/CTR 的范围和分布。

## 2. 当前进展
- 已完成工作 1：`CounterfactualRewardModel` 现在优先使用 `real_env.mat[user, action]` 作为 reward prior。
- 已完成工作 2：保留无真实矩阵时的 `predicted_mat + entropy bonus - min_reward` 回退路径。
- 已完成工作 3：on-policy rerank 新增 `rerank/reward_prior_min/max/std` 日志。
- 未完成工作：未运行完整 DORL-DOSER 训练。

## 3. 已确认结论
- KuaiEnv-v0 真实单步 reward 范围是 `[0, 5]`。
- 当前日志中的 CTR 若按 `trajectory_reward / trajectory_length` 计算，理论范围也是 `[0, 5]`，不是二分类 CTR 的 `[0, 1]`。
- 训练用 DeepFM 预测矩阵范围约为 `[-0.0140, 0.0467]`，远小于真实 reward 范围。

## 4. 关键证据或依据
- 代码依据：`src/core/envs/BaseEnv.py` 中 `reward = self.mat[self.cur_user, action]`。
- 统计依据：`KuaiData.load_mat()` 返回矩阵 shape `(1411, 3327)`，min `0.0`，max `5.0`，mean `0.8722596110803493`。
- 验证依据：`py_compile`、最小 reward smoke test 和导入验证均通过。

## 5. 未解决问题与风险
- 风险 1：使用 `real_env.mat` 作为训练期 reward prior 更贴近评估指标，但也更接近 oracle 信息；若要严格保持离线训练设定，应设置 `prefer_real_env_reward=False`。
- 风险 2：本轮没有验证完整训练曲线，实际性能提升仍需重新跑 `DORL_DOSER_ONPOLICY`。

## 6. 建议下一步
- 下一步 1：重新运行 `bash script/run_DORL_DOSER_onpolicy.sh`，检查 SwanLab 中 reward prior 分布是否不再塌缩。
- 下一步 2：对比开启真实 reward prior 前后的 `Trajectory Reward`、`CTR`、`MCD` 与 DARLR baseline。

## 7. 相关文件
- 代码文件：`src/core/policy/dorl_doser_impl.py`
- 代码文件：`src/core/policy/dorl_doser_onpolicy_impl.py`
- 详细报告：`reports/20260526_080318_reward_prior_real_reward_alignment.md`
- 关键日志：当前未生成完整训练日志

