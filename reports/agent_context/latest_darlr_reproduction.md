# Agent Context Log

## 1. 任务概述
- 任务名称：`darlr_reproduction`
- 当前状态：部分完成
- 本轮目标：根据 `docs/DARLR_reproduction_description.md` 实现 DARLR v1 代码复现，并提供可运行 smoke 脚本。

## 2. 当前进展
- 已完成工作 1：新增 `examples/advance/run_DARLR.py`，复用 DORL 训练流程并接入 DARLR 参数、动态奖励环境和联合策略。
- 已完成工作 2：新增 `src/core/policy/darlr.py`，实现 recommender A2C + selector A2C、候选用户选择、selector intrinsic reward 和 `darlr_context` 输出。
- 已完成工作 3：新增 `src/core/envs/Simulated_Env/darlr_dynamic_reward.py`，实现 reference mean reward、dynamic uncertainty penalty 和 DORL fallback。
- 已完成工作 4：修改 `src/core/collector/collector.py`，在 `env.step()` 前通过 `set_env_attr` 注入 DARLR context。
- 已完成工作 5：新增 `script/run_DARLR_KuaiRec_smoke.sh`。
- 未完成工作：真实 KuaiRec smoke 脚本尚未完整跑完 1 个 epoch。

## 3. 已确认结论
- 结论 1：DARLR 新增模块能在 `easyrl4rec` conda 环境中通过静态编译和导入检查。
- 结论 2：dummy `DARLRPolicy.forward()` 能生成 selector context。
- 结论 3：dummy `DARLRPolicy.learn()` 能同时产生 recommender loss 和 selector loss，并完成反向传播。
- 结论 4：dummy `DARLRDynamicRewardEnv.step()` 能读取 selector context 并输出 DARLR 诊断指标。

## 4. 关键证据或依据
- 证据 1：`py_compile` 通过：`examples/advance/run_DARLR.py`、`src/core/policy/darlr.py`、`src/core/envs/Simulated_Env/darlr_dynamic_reward.py`、`src/core/collector/collector.py`。
- 证据 2：导入检查输出 `ok`。
- 证据 3：dummy policy forward 输出 `policy_forward_ok`，dummy env 输出 `env_step_ok`。
- 证据 4：dummy learn 输出 `policy_learn_ok`，包含 `loss/selector`、`loss/selector_actor`、`loss/selector_vf`。

## 5. 未解决问题与风险
- 风险 1：真实 KuaiRec tiny/smoke 命令在配置日志后长时间 CPU 满载，本轮已停止，未确认端到端 1 epoch 完成。
- 风险 2：当前 selector 候选池实现支持 `embedding_topk` 和 `random`，尚未实现论文提到的聚类候选池。
- 风险 3：`previous_reward` 采用环境内 `(user, item)` cache，是工程近似，官方代码未公开无法逐行对齐。

## 6. 建议下一步
- 下一步 1：给 `run_DARLR.py` 增加分段计时日志，定位真实 KuaiRec smoke 的耗时点。
- 下一步 2：增加 synthetic smoke 模式，绕过 DeepFM 真实模型加载，验证 trainer 端到端完成。
- 下一步 3：真实实验前逐步恢复 `entropy_window`、更大的 `selector_k` 和 `embedding_topk` 候选池。

## 7. 相关文件
- 代码文件：`examples/advance/run_DARLR.py`、`src/core/policy/darlr.py`、`src/core/envs/Simulated_Env/darlr_dynamic_reward.py`、`src/core/collector/collector.py`、`script/run_DARLR_KuaiRec_smoke.sh`
- 详细报告：`reports/20260518_135727_darlr_reproduction.md`
- 关键日志：本轮没有保留完整训练日志；真实 smoke 已手动停止。
