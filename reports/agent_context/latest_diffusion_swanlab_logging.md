# Agent Handoff: Diffusion SwanLab Logging

## 已完成
- 将 `examples/diffusion/pretrain_diffusion.py` 的可选实验日志后端从 `wandb` 切换为 `swanlab`。
- 新增正向参数：`--enable_swanlab`、`--swanlab_project`、`--swanlab_run_name`、`--swanlab_mode`。
- 保留兼容别名：`--enable_wandb` 映射到 `enable_swanlab`，`--wandb_project` 映射到 `swanlab_project`。
- 训练循环仍通过 `tracker.log(...)` 记录 `state_diffusion/loss`、`behavior_diffusion/loss` 等标量。

## 已验证
- `swanlab 0.7.2` 在 `easyrl4rec` 环境可导入，具备 `init/log/finish` 接口。
- `python -m py_compile examples/diffusion/pretrain_diffusion.py` 通过。
- 旧参数兼容检查通过：`--enable_wandb --wandb_project P` 解析为 `enable_swanlab=True, swanlab_project=P`。
- 小样本 smoke test 通过，使用 `--enable_swanlab --swanlab_mode disabled` 验证日志开关分支不影响训练。

## 风险与下一步
- 本次未真正创建在线 SwanLab run，避免在验证中触发网络/账号侧影响。
- 全量训练时添加 `--enable_swanlab` 即可上报训练指标。

## 报告
- 详细报告：`reports/20260519_081302_diffusion_swanlab_logging.md`
