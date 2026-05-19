# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
用户要求基于 DORL 算法，在 `examples/our_model/dorl_doser.py` / `examples/our_model/dorl_doser_onpolicy.py` 训练代码基础上加入 DOSER 的 OOD 识别模块，并能够面向 KuaiEnv-v0 数据集训练。实现要求保留 A2CPolicy 和 on-policy 更新逻辑，只在 critic 端附加 OOD penalty / compensation；同时适配前序扩散模型预训练产物 `behavior_diffusion.pt`、`state_diffusion.pt`、`pretrain_config.json`、`training_metrics.json`，并保留其他推荐数据集的扩展接口。

### 1.2 任务分解
1. 新建 DORL-DOSER 运行期通用模块，负责 diffusion artifact 加载、counterfactual reward 和 OOD error 计算。
2. 新建 on-policy DORL-DOSER 策略模块，保留 A2CPolicy 主更新并加入 critic auxiliary Q/V、OOD penalty 和 compensation。
3. 调整 on-policy 入口默认参数，使 OOD 辅助损失默认不反传到 actor/state tracker 分支。
4. 在 `easyrl4rec` conda 环境中执行静态编译、导入、diffusion loader、OOD helper 和合成 batch learn smoke test。
5. 记录完整实现结果、验证结果、未完成的真实环境训练 smoke 限制和后续建议。

### 1.3 技术方案
采用“新增源码模块 + 保持原导出接口”的方式实现。`src/core/policy/doser.py` 末尾已有对 `dorl_doser_impl` 和 `dorl_doser_onpolicy_impl` 的导出引用，因此本次直接重写这两个缺失源码模块，不依赖 `.pyc`。on-policy 策略继承项目内 Tianshou `A2CPolicy`，复用其 forward、process_fn、GAE 和 A2C 损失结构，仅重写 `learn()`，将 DOSER 相关项作为 critic auxiliary loss 加入总 loss。

## 2. 任务完成过程记录
### 2.1 代码结构
- 新增文件：`src/core/policy/dorl_doser_impl.py`
- 新增文件：`src/core/policy/dorl_doser_onpolicy_impl.py`
- 修改文件：`examples/our_model/dorl_doser_onpolicy.py`

### 2.2 模块详细说明
#### 2.2.1 DORL-DOSER diffusion 与 OOD 通用模块
- 文件路径：`src/core/policy/dorl_doser_impl.py`
- 核心功能：加载新旧 diffusion artifact，提供 `DiffusionArtifact`、`RewardModelConfig`、`CounterfactualRewardModel`、`DORLDOSEROODHelper`，并保留 off-policy 兼容类。
- 关键函数：
  - `resolve_diffusion_artifact_dir(save_root, env_name, artifact_name)`：按 `saved_models/<env_name>/`、`saved_models/<env_name>/<artifact_name>/`、旧格式 `saved_models/<env_name>/DOSER/diffusion/<artifact_name>/` 查找产物目录。
  - `load_diffusion_artifact(save_root, env_name, artifact_name, device)`：加载行为扩散模型和状态扩散模型，读取维度、阈值和 Karras diffusion 参数。
  - `DORLDOSEROODHelper.compute_action_error(actions, states)`：计算 `p(a | s)` 行为扩散重构误差。
  - `DORLDOSEROODHelper.compute_state_error(states, actions, next_states)`：计算条件 `p(s_next | s, a)` 或旧格式无条件状态扩散重构误差。
  - `DORLDOSEROODHelper.select_best_id_action(...)`：用行为扩散采样候选动作并通过 critic 选择最佳 ID 动作。
  - `DORLDOSEROODHelper.build_counterfactual_next_state(...)`：使用 state tracker 和 counterfactual reward 构造候选动作对应的下一状态表示。
- 核心算法：运行期 OOD error 与预训练阈值口径一致，均对 noisy target 做 Karras denoising，并用逐样本 L2 重构误差判断是否超过 `state_threshold` / `action_threshold`。

#### 2.2.2 On-policy DORL-DOSER 策略模块
- 文件路径：`src/core/policy/dorl_doser_onpolicy_impl.py`
- 核心功能：实现 `A2CDOSERAugmentedCritic` 和 `OnPolicyDORLDOSERPolicy`，在 A2C 主更新上增加 critic 端 DOSER 辅助损失。
- 关键函数：
  - `A2CDOSERAugmentedCritic.forward(obs)`：保持 A2C 主 value head 输出 `V(s)`。
  - `A2CDOSERAugmentedCritic.aux_q(obs, action_embeddings)`：输出双 Q 辅助估计。
  - `A2CDOSERAugmentedCritic.aux_v(obs)`：输出双 V 辅助估计。
  - `OnPolicyDORLDOSERPolicy._compute_auxiliary_critic_loss(...)`：用 A2C returns 监督辅助 Q/V 分支。
  - `OnPolicyDORLDOSERPolicy._compute_ood_regularization(...)`：计算 action/state OOD mask、negative penalty 和 positive compensation。
  - `OnPolicyDORLDOSERPolicy.learn(...)`：保留 actor loss、vf loss、entropy loss，并叠加 DOSER critic loss。
- 核心算法：actor 的 greedy action 仅用于 OOD 评估，动作选择、扩散采样、counterfactual next state 和 OOD mask 均在 `torch.no_grad()` 下完成；默认 `doser_detach_aux_state=True`，避免 DOSER 辅助损失反传到 actor/state tracker。

### 2.3 既有代码修改说明
- 修改文件：`examples/our_model/dorl_doser_onpolicy.py`
- 修改区域：`get_args_dorl_doser_onpolicy()` 中 `doser_detach_aux_state` 默认值，约第 82-92 行。
- 修改原因：用户要求 OOD penalty / compensation 只附加在 critic 端。
- 修改前：`parser.set_defaults(doser_detach_aux_state=False)`，辅助损失默认可能反传到 state tracker。
- 修改后：`parser.set_defaults(doser_detach_aux_state=True)`，默认阻断 DOSER 辅助损失到状态编码分支。

### 2.4 工具限制说明
本轮 `apply_patch` 成功创建了新增源码文件，但后续对已存在文件执行增量 patch 时返回 “No such file or directory”。为完成任务，剩余两个定向替换使用 Python 脚本完成：修改 `doser_detach_aux_state` 默认值，以及修正新增模块的导入和常量拼写。该限制已在验证后复查 diff，未引入额外文件改动。

## 3. 数据处理说明
### 3.1 数据来源
本任务未新增数据处理流程。实现依赖前序 diffusion 预训练产物，当前本地可用旧格式产物位于 `saved_models/KuaiEnv-v0/DOSER/diffusion/DM_KuaiEnv-v0_small_data/`，metadata 显示 `state_dim=42`、`action_dim=41`。

### 3.2 数据预处理步骤
- 新增运行期维度对齐：当 state tracker 输出维度与 diffusion `state_dim` 不一致时，`DORLDOSEROODHelper.normalize_obs_array()` 会进行截断或零填充并记录 warning。
- 新增动作维度对齐：当 action embedding 维度与 diffusion `action_dim` 不一致时，`normalize_action_array()` 会进行截断或零填充。

### 3.3 数据统计信息
- 已验证旧格式 artifact：`state_dim=42`、`action_dim=41`、`state_model_kind=unconditional`、`state_threshold=1.6304931640625`、`action_threshold=1.6925586462020874`。
- 新格式全量 diffusion 预训练产物本轮未发现于 `saved_models/KuaiEnv-v0/`，代码已支持但未做全量训练验证。

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：当前环境 CPU 可用；完整 GPU 信息本次未采集。
- 软件环境：使用 `conda run -n easyrl4rec` 执行验证；导入阶段出现 TensorFlow CPU instruction 提示、Gym 0.24.1 兼容性提示和 DeepCTR-PyTorch 版本提示，均非本次代码错误。
- 超参数设置：合成 smoke test 使用旧格式 diffusion artifact，`doser_action_samples=1`、`diffusion_sample_steps=1`、batch size 为 2。

### 4.2 图表结果分析
本任务未涉及图表生成。

### 4.3 验证记录
- 静态编译通过：
  - `conda run -n easyrl4rec python -m py_compile examples/our_model/dorl_doser.py examples/our_model/dorl_doser_onpolicy.py src/core/policy/doser.py src/core/policy/dorl_doser_impl.py src/core/policy/dorl_doser_onpolicy_impl.py`
- 导入 smoke test 通过：
  - `from src.core.policy.doser import A2CDOSERAugmentedCritic, OnPolicyDORLDOSERPolicy, load_diffusion_artifact`
- 旧格式 diffusion loader smoke test 通过：
  - 成功加载 `saved_models/KuaiEnv-v0/DOSER/diffusion/DM_KuaiEnv-v0_small_data`
  - 输出 `42 41 unconditional 1.6304931640625 1.6925586462020874`
- OOD helper smoke test 通过：
  - action/state error 输出形状均为 `(2,)`
  - `torch.isfinite(...)` 均为 `True`
- 合成 batch learn smoke test 通过：
  - `OnPolicyDORLDOSERPolicy.learn()` 完成 forward、loss、backward、optimizer step
  - 输出包含 `loss`、`loss/actor`、`loss/doser`、`loss/doser_aux`，loss 有限
- KuaiEnv-v0 极小真实训练 smoke 未完成：
  - 使用默认 `read_message=UM` 时缺少 `saved_models/KuaiEnv-v0/DeepFM/params/[UM]_params.pickle`
  - 改用本地存在的 `read_message=pointneg` 后，进程在初始化阶段 CPU 打满且长时间无输出，已手动停止以避免占用资源

## 5. 最终结论
本轮已完成 DORL + DOSER OOD critic 的 on-policy 代码接入：新增 diffusion artifact loader、OOD helper、counterfactual reward 近似器、augmented critic 和继承 A2CPolicy 的 on-policy 策略。代码保留 A2C actor/value/entropy 主更新语义，OOD penalty / compensation 默认只作用于 critic auxiliary 分支。静态编译、导入、旧格式 diffusion loader、OOD error 和合成 batch learn/backward 均已通过验证。真实 KuaiEnv-v0 端到端训练 smoke 受本地用户模型参数与环境初始化耗时限制，本轮未完成。

## 6. 后续建议
1. 先运行或补齐 KuaiEnv-v0 新格式 diffusion 全量预训练，生成 `saved_models/KuaiEnv-v0/behavior_diffusion.pt` 和 `state_diffusion.pt`。
2. 补齐或确认用户模型参数路径，尤其是默认 `read_message=UM` 对应的 DeepFM 参数文件。
3. 使用较小 `training_num`、`test_num`、`step_per_epoch` 再次执行真实 on-policy smoke，并观察 `loss/doser_*` 与 `ood/*` 指标。
4. 若旧格式 artifact 仅作为兼容加载使用，建议后续以新格式条件状态扩散模型为主进行正式实验。

## 7. 代码使用说明
- 编译验证：
  ```bash
  conda run -n easyrl4rec python -m py_compile examples/our_model/dorl_doser.py examples/our_model/dorl_doser_onpolicy.py src/core/policy/doser.py src/core/policy/dorl_doser_impl.py src/core/policy/dorl_doser_onpolicy_impl.py
  ```
- on-policy 训练入口：
  ```bash
  WANDB_MODE=disabled PYTHONPATH=.:./src:./src/DeepCTR-Torch:./src/tianshou   conda run -n easyrl4rec python examples/our_model/dorl_doser_onpolicy.py     --env KuaiEnv-v0     --diffusion_save_root saved_models     --diffusion_artifact_name DM_KuaiEnv-v0_small_data     --doser_detach_aux_state     --wandb_mode disabled
  ```
- 若已经完成新格式 diffusion 预训练，可省略 `--diffusion_artifact_name` 或指向新格式子目录，loader 会优先查找 `saved_models/<env_name>/`。
