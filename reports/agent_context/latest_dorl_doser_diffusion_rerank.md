# Agent Context Log

## 1. 任务概述
- 任务名称：`dorl_doser_diffusion_rerank`
- 当前状态：部分完成
- 本轮目标：在 DORL-DOSER on-policy 中加入扩散引导 rerank，使 DOSER 能影响最终 actor 动作，同时保留 A2C/on-policy 更新语义。

## 2. 当前进展
- 已完成工作 1：`src/core/policy/dorl_doser_onpolicy_impl.py` 已加入 actor top-k、行为扩散候选、随机候选、aux Q、reward prior、action OOD penalty 组成的 rerank 动作出口。
- 已完成工作 2：`examples/our_model/dorl_doser_onpolicy.py` 已暴露 rerank 参数，并默认拆分 actor/critic backbone。
- 已完成工作 3：`script/run_DORL_DOSER_onpolicy.sh` 已支持 rerank 相关环境变量和 smoke 候选缩减。
- 未完成工作：未完成 KuaiEnv-v0 全量训练，也未验证是否超过 DARLR baseline。

## 3. 已确认结论
- 结论 1：新增代码通过 `py_compile`、导入 smoke test、dummy forward/learn smoke test、脚本语法检查和 `git diff --check`。
- 结论 2：dummy smoke 中 `forward()` 输出 `(3, 8)` 概率，能写入 `Batch.policy.doser_candidates`，`learn()` 能完成一次更新并返回 `rerank/candidate_size`。
- 结论 3：DOSER rerank 现在会改变 `Categorical(probs=rerank_probs)`，因此可以影响最终采样动作。

## 4. 关键证据或依据
- 证据 1：`dummy_ok (3, 8) True 4.0`。
- 证据 2：`conda run -n easyrl4rec python -m py_compile ...` 通过。
- 证据 3：`bash -n script/run_DORL_DOSER_onpolicy.sh` 与 `git diff --check` 通过。

## 5. 未解决问题与风险
- 风险 1：完整脚本 smoke 进入 KuaiEnv entropy 统计构建流程，需要遍历约 `12530806` 行数据，本轮未等待完成。
- 风险 2：新增 rerank 会增加训练开销，候选数量和扩散采样步数需要根据实际训练速度调参。
- 风险 3：当前没有全量训练结果，不能断言指标已经优于 DARLR。

## 6. 建议下一步
- 下一步 1：运行完整 `bash script/run_DORL_DOSER_onpolicy.sh`，观察 SwanLab 中 `rerank/*`、`ood/*` 与主评估指标。
- 下一步 2：做 `DOSER_ENABLE_RERANK=0`、`DOSER_RERANK_ALPHA_Q=0`、`DOSER_RERANK_BETA_ACTION_OOD=0` 三组消融，定位收益来源。
- 下一步 3：若 smoke 太慢，优先缓存或跳过 KuaiEnv entropy 全量统计。

## 7. 相关文件
- 代码文件：`src/core/policy/dorl_doser_onpolicy_impl.py`
- 代码文件：`examples/our_model/dorl_doser_onpolicy.py`
- 代码文件：`script/run_DORL_DOSER_onpolicy.sh`
- 详细报告：`reports/20260524_053239_dorl_doser_diffusion_rerank.md`
- 关键日志：本轮未生成完整训练日志；验证输出来自命令行 smoke test。
