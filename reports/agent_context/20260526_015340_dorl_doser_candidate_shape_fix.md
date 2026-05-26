# Agent Context Log

## 1. 任务概述
- 任务名称：`dorl_doser_candidate_shape_fix`
- 当前状态：已完成
- 本轮目标：修复 DORL-DOSER on-policy 中 `policy.doser_candidates` 变长导致的 replay buffer shape mismatch。

## 2. 当前进展
- 已完成工作 1：确认 `(100,145)` vs `(100,144)` 来自 rerank 候选列数变化。
- 已完成工作 2：移除 `policy_batch.doser_candidates` 写入，候选不再进入 replay buffer。
- 已完成工作 3：新增 `_is_learning_minibatch()`，只有 learn batch 才使用 `batch.act` 作为 required action。
- 未完成工作：未重新运行完整 KuaiEnv-v0 训练。

## 3. 已确认结论
- 结论 1：shape mismatch 不是扩散 artifact 问题，而是 buffer 字段形状不固定。
- 结论 2：collector batch 不再触发 required action，learn batch 仍会触发 required action。

## 4. 关键证据或依据
- 证据 1：`py_compile` 通过。
- 证据 2：最小验证输出 `collector_is_learning False`、`learn_is_learning True`、`empty_required None`。

## 5. 未解决问题与风险
- 风险 1：完整训练尚未重新跑，需要确认实际 collector/buffer 流程通过。
- 风险 2：learn 阶段重新采样候选，与 collection 时候选不完全一致；但 A2C 本身不依赖 old log-prob，且真实动作会强制加入候选。

## 6. 建议下一步
- 下一步 1：重新运行 `bash script/run_DORL_DOSER_onpolicy.sh`。
- 下一步 2：若仍有 shape mismatch，检查是否存在其他变长 `policy.*` 字段。

## 7. 相关文件
- 代码文件：`src/core/policy/dorl_doser_onpolicy_impl.py`
- 详细报告：`reports/20260526_015340_dorl_doser_candidate_shape_fix.md`
- 关键日志：本轮未生成完整训练日志。
