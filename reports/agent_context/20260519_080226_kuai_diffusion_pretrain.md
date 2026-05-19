# Agent Handoff: Kuai Diffusion Pretraining

## 已完成
- 修改 `examples/diffusion/pretrain_diffusion.py`，支持 KuaiEnv 轨迹 pickle 的 `(state, action, next_state)` transition 训练。
- 行为扩散：训练 `p(a | s)`，保存 `behavior_diffusion.pt`。
- 状态扩散：训练 `p(s_next | s, a)`，保存 `state_diffusion.pt`。
- 默认保存到 `saved_models/<env_name>/`；配置和指标分别保存为 `pretrain_config.json`、`training_metrics.json`。
- dynamic model 已在预训练脚本中明确禁用，仅在 metadata 中记录该设定。

## 已验证
- `conda run -n easyrl4rec python -m py_compile examples/diffusion/pretrain_diffusion.py` 通过。
- 全量数据 adapter 可加载：`state_dim=42, action_dim=41, users=1411, behavior_transitions=4676570, state_transitions=4676570`。
- 小样本 smoke test 通过：`pretrain_epochs=1, batch_size=4, max_trajectories=2, max_transitions=8`。
- smoke test 产物包含 `behavior_diffusion.pt`、`state_diffusion.pt`、`pretrain_config.json`、`training_metrics.json`。

## 风险与下一步
- 尚未执行全量长时间预训练。
- 后续 DORL/DOSER loader 仍需适配新扁平产物格式和条件状态扩散语义。
- 当前 `trajectory_pkl` 后端不做归一化，沿用数据集中已有状态/动作向量。

## 报告
- 详细报告：`reports/20260519_080226_kuai_diffusion_pretrain.md`
