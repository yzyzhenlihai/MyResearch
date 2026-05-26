# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
用户反馈 `DORL_DOSER_ONPOLICY` 训练时 SwanLab 没有记录以下训练指标，并要求修改代码进行日志打印，同时解释每个 loss 的含义：

- `loss`
- `loss/actor`
- `loss/vf`
- `loss/ent`
- `loss/doser`
- `loss/doser_aux`
- `loss/doser_penalty`
- `loss/doser_compensation`
- `ood/ood_ratio`
- `ood/positive_ratio`
- `ood/negative_ratio`

### 1.2 任务分解
1. 检查 `OnPolicyDORLDOSERPolicy.learn()` 中损失返回与 trainer 日志链路。
2. 定位 SwanLab 未记录 update loss 的原因。
3. 在策略内部增加安全 SwanLab 记录与本地日志打印。
4. 保持现有 A2C/on-policy 更新语义不变。
5. 运行静态编译与最小 smoke test。

### 1.3 技术方案
当前 `learn()` 已经返回了用户列出的指标，但 `examples/policy/policy_utils.py` 调用 `onpolicy_trainer()` 时未传入 Tianshou logger，默认使用 `LazyLogger()`，因此 trainer 不会自动把 update loss 写入 SwanLab。为避免大范围改动训练器，本次在 `src/core/policy/dorl_doser_onpolicy_impl.py` 的 `OnPolicyDORLDOSERPolicy` 内部增加主动日志记录：

1. 在每个 minibatch update 后构造标量指标字典。
2. 第 1 个 update 和之后每 `log_interval` 个 update 调用本地 `LOGGER.info()`。
3. 若 SwanLab 已初始化，则调用 `swanlab.log(metrics, step=learn_step)`。
4. 用 `swanlab.get_run()` 判断 active run，避免误把 `swanlab.run` 模块对象当作运行实例。
5. 保留 `SWANLAB_MODE=disabled`、`SWANLAB_DISABLED=true` 以及历史 `WANDB_*` 环境变量的关闭兼容。

## 2. 任务完成过程记录
### 2.1 代码结构
本次修改文件：

- `src/core/policy/dorl_doser_onpolicy_impl.py`

本次新增报告文件：

- `reports/20260523_092346_dorl_doser_swanlab_loss_logging.md`
- `reports/agent_context/20260523_092346_dorl_doser_swanlab_loss_logging.md`
- `reports/agent_context/latest_dorl_doser_swanlab_loss_logging.md`

### 2.2 模块详细说明
#### 2.2.1 `OnPolicyDORLDOSERPolicy` 日志模块
- 文件路径：`/data/yuzhengyang/RL_Learning/EasyRL4Rec/src/core/policy/dorl_doser_onpolicy_impl.py`
- 核心功能：在 on-policy A2C + DOSER critic 更新阶段主动记录训练损失与 OOD 比例。
- 关键函数：
  - `_is_swanlab_disabled()`：读取 `SWANLAB_MODE`、`SWANLAB_DISABLED` 和历史 `WANDB_*` 环境变量，判断是否跳过 SwanLab。
  - `_to_finite_float(value)`：把张量、NumPy 标量或 Python 数值转换为有限浮点数，避免非标量或 NaN/Inf 写入。
  - `_safe_swanlab_log(metrics)`：在 SwanLab 可用且 active run 存在时写入指标，异常时只打印一次 warning，不中断训练。
  - `_log_training_metrics(metrics)`：按 `learn_step == 1` 或 `learn_step % log_interval == 0` 写入本地日志和 SwanLab。
  - `learn(...)`：保留原训练流程，补充 `trainer/env_step`、`loss/doser`、`loss/doser_aux` 等显式指标。
- 核心算法：未改变 A2C 与 DOSER 损失计算，仅改变日志记录路径。

### 2.3 既有代码修改说明
- 修改文件：`src/core/policy/dorl_doser_onpolicy_impl.py`
- 修改区域：文件导入区、`OnPolicyDORLDOSERPolicy._log_training_metrics()`、`OnPolicyDORLDOSERPolicy.learn()`
- 修改原因：`learn()` 返回值会被 trainer 聚合，但当前 trainer 使用 `LazyLogger()`，不会自动上传 update loss；需要策略内部直接写 SwanLab。
- 修改前：
  - `_log_training_metrics()` 只打印少量本地日志。
  - 指标字典内部使用 `loss/doser_total`，最终返回时才映射为 `loss/doser`。
  - 短 smoke run 如果 update 数少于 `log_interval`，没有任何 loss 打印。
- 修改后：
  - 第 1 个 update 即打印和尝试上传 SwanLab，之后按 `--doser_log_interval` 控制频率。
  - SwanLab 记录使用用户关心的原始 key：`loss/doser`、`loss/doser_aux`、`loss/doser_penalty`、`loss/doser_compensation` 等。
  - 本地日志行补充 actor、vf、entropy、DOSER、OOD ratio、positive ratio、negative ratio。
  - 额外记录 `trainer/env_step`，SwanLab 的 step 使用 `learn_step`。

## 3. 数据处理说明
### 3.1 数据来源
本任务未涉及。

### 3.2 数据预处理步骤
本任务未涉及。

### 3.3 数据统计信息
本任务未涉及。

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：当前环境未知。
- 软件环境：使用 `conda run -n easyrl4rec` 执行验证。
- 超参数设置：本任务未运行完整训练；日志频率仍由 `--doser_log_interval` 控制。

### 4.2 图表结果分析
本任务未涉及。

### 4.3 验证记录
1. 静态编译通过：

```bash
conda run -n easyrl4rec python -m py_compile \
  src/core/policy/dorl_doser_onpolicy_impl.py \
  examples/our_model/dorl_doser.py \
  examples/our_model/dorl_doser_onpolicy.py
```

2. 导入 smoke test 通过：

```bash
conda run -n easyrl4rec python -c "import sys; sys.path.extend(['.', './src', './src/tianshou']); import src.core.policy.dorl_doser_onpolicy_impl as m; print('import_ok', m.OnPolicyDORLDOSERPolicy.__name__); print('get_run_before_init', m.swanlab.get_run() if m.swanlab else None)"
```

确认输出包含：

```text
import_ok OnPolicyDORLDOSERPolicy
get_run_before_init None
```

3. 伪 SwanLab smoke test 通过：构造 fake SwanLab 对象并调用 `_log_training_metrics()`，确认写出的 key 包含：

```text
loss, loss/actor, loss/vf, loss/ent, loss/doser,
loss/doser_aux, loss/doser_penalty, loss/doser_compensation,
ood/ood_ratio, ood/positive_ratio, ood/negative_ratio,
trainer/env_step
```

未运行完整 KuaiEnv-v0 训练，因此未验证线上 SwanLab 页面是否实时出现曲线。

## 5. 最终结论
本次已完成 SwanLab update loss 主动记录逻辑。此前没有记录这些 loss 的直接原因是 trainer 默认使用 `LazyLogger()`，`learn()` 返回的 update loss 没有自动上传到 SwanLab。现在 `OnPolicyDORLDOSERPolicy` 会在第 1 个 update 和每 `--doser_log_interval` 个 update 后主动写入本地日志与 SwanLab。

## 6. 后续建议
1. 重新启动训练进程，已有训练不会补录历史 loss。
2. 若希望更密集地观察曲线，可设置 `--doser_log_interval 1` 或较小数值。
3. 完整训练启动后，检查 SwanLab 中是否出现 `loss/*` 与 `ood/*` 面板。
4. 当前工作区中 `script/run_DORL_DOSER_onpolicy.sh` 仍有既有未提交改动，本任务未修改该脚本。

## 7. 代码使用说明
推荐运行：

```bash
SWANLAB_MODE=cloud bash script/run_DORL_DOSER_onpolicy.sh
```

调试时可降低日志间隔：

```bash
SWANLAB_MODE=cloud bash script/run_DORL_DOSER_onpolicy.sh --doser_log_interval 1
```

若需要关闭 SwanLab：

```bash
SWANLAB_MODE=disabled bash script/run_DORL_DOSER_onpolicy.sh
```
