# Agent Context Log

## 1. 任务概述
- 任务名称：dorl_doser_ood_critic
- 当前状态：部分完成
- 本轮目标：为 DORL on-policy/A2C 训练接入 DOSER OOD critic penalty / compensation，并适配 diffusion 预训练产物加载。

## 2. 当前进展
- 已完成工作 1：新增 `src/core/policy/dorl_doser_impl.py`，实现 diffusion artifact loader、`DiffusionArtifact`、`RewardModelConfig`、`CounterfactualRewardModel` 和 `DORLDOSEROODHelper`。
- 已完成工作 2：新增 `src/core/policy/dorl_doser_onpolicy_impl.py`，实现 `A2CDOSERAugmentedCritic` 和 `OnPolicyDORLDOSERPolicy`。
- 已完成工作 3：将 `examples/our_model/dorl_doser_onpolicy.py` 中 `doser_detach_aux_state` 默认值改为 `True`。
- 未完成工作：真实 KuaiEnv-v0 端到端训练 smoke 未完成，受默认用户模型参数缺失和环境初始化耗时限制。

## 3. 已确认结论
- `doser.py` 导出的 `A2CDOSERAugmentedCritic`、`OnPolicyDORLDOSERPolicy`、`load_diffusion_artifact` 可以正常导入。
- 旧格式 artifact `saved_models/KuaiEnv-v0/DOSER/diffusion/DM_KuaiEnv-v0_small_data` 可以加载，维度为 `state_dim=42`、`action_dim=41`。
- OOD helper 可以输出有限的 action/state reconstruction error。
- 合成 batch 下 `OnPolicyDORLDOSERPolicy.learn()` 可以完成 forward、loss、backward 和 optimizer step。

## 4. 关键证据或依据
- 静态编译通过：`conda run -n easyrl4rec python -m py_compile ...`
- 导入 smoke 输出：`import ok`
- loader smoke 输出：`42 41 unconditional 1.6304931640625 1.6925586462020874`
- OOD helper smoke 输出：`ood smoke (2,) (2,) True True`
- learn smoke 输出：`learn smoke ['loss', 'loss/actor', 'loss/doser', 'loss/doser_aux'] 1 True`

## 5. 未解决问题与风险
- 风险 1：新格式 `saved_models/KuaiEnv-v0/behavior_diffusion.pt` 和 `state_diffusion.pt` 本轮未在本地验证到，全量预训练仍需执行。
- 风险 2：真实训练默认需要 `saved_models/KuaiEnv-v0/DeepFM/params/[UM]_params.pickle`，本地仅发现 `[pointneg]_params.pickle`。
- 风险 3：旧格式状态扩散是 unconditional，仅作为兼容路径；正式实验应优先使用新格式条件状态扩散。

## 6. 建议下一步
- 下一步 1：补齐默认用户模型参数或明确训练命令使用 `--read_message pointneg`，然后重新跑极小 KuaiEnv-v0 on-policy smoke。
- 下一步 2：完成新格式 diffusion 全量预训练后，用不传 `--diffusion_artifact_name` 的默认路径验证新 loader。

## 7. 相关文件
- 代码文件：`src/core/policy/dorl_doser_impl.py`、`src/core/policy/dorl_doser_onpolicy_impl.py`、`examples/our_model/dorl_doser_onpolicy.py`
- 详细报告：`reports/20260519_102913_dorl_doser_ood_critic.md`
- 关键日志：本轮验证输出来自终端命令，未单独保存训练日志。
