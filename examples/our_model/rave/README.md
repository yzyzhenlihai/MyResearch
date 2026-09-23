# RAVE-H

依据 `docs/RAVE-H_推荐场景下生存与多样性感知的自适应前缀执行方案.md` 实现。冻结 MAC actor、ChunkDynamics 和 user model。`ordinary` 在每个优化步重新生成固定深度 imagined rollout，立即用 TD 标签训练 MAC chunk Q/V，不跨步复用轨迹；两个 Transformer 分支仍从完整 episode replay 训练 ACSAC 全有效前缀 Q，并在部署时联合选择候选和 H。

## 运行

使用现有 `easyrl4rec` 环境，SwanLab 依赖已经安装。必须从仓库根目录运行。

```bash
# 首次使用 Flow actor 时先训练 Flow BC 与 chunk dynamics。
conda run --no-capture-output -n easyrl4rec python \
  examples/our_model/runners/pretrain_categorical_bc.py \
  --env KuaiEnv-v0 --chunk_size 3 --actor_type categorical_flow \
  --flow_steps 10 --flow_sample_batch_size 4 --flow_temperature 0.05 \
  --flow_distill_weight 1.0 --flow_sampling_mode onestep \
  --run_name kuai-flow-k3-seed2023 \
  --save_dir saved_models/KuaiEnv-v0/QV_suite/K3/shared_pretrain/seed-2023/categorical_flow_bc

# K 和 actor 类型都由 checkpoint 恢复。N 默认 32，训练和部署保持一致。
MAC_CHECKPOINT=saved_models/KuaiEnv-v0/QV_suite/K3/shared_pretrain/seed-2023/categorical_flow_bc/latest.pt \
CHUNK_SIZE=3 DEVICE=cpu RUN_NAME=rave-k3-main \
bash script/run_rave_kuai.sh

# 小规模端到端测试；只验证流程，不能用来判断效果。
MAC_CHECKPOINT=saved_models/KuaiEnv-v0/QV_suite/K3/shared_pretrain/seed-2023/categorical_flow_bc/latest.pt \
STEPS=2 BATCH_SIZE=2 MAX_TURN=8 NUM_CANDIDATES=4 \
RUN_NAME=rave-smoke-new SWANLAB_MODE=disabled bash script/run_rave_kuai.sh \
  --rollout_candidates 2 --hidden_dim 16 --transformer_layers 1 \
  --scale_samples 64 --eval_episodes 2 --mc_samples 1

MODE=eval RAVE_CHECKPOINT=saved_models/KuaiEnv-v0/QV_suite/K3/multi_prefix_transformer/seed-2023/latest.pt \
bash script/run_rave_kuai.sh

# 权重与超参数在验证集冻结后，显式选择最终测试划分。
MODE=eval EVAL_SPLIT=test \
RAVE_CHECKPOINT=saved_models/KuaiEnv-v0/QV_suite/K3/multi_prefix_transformer/seed-2023/latest.pt \
bash script/run_rave_kuai.sh

conda run -n easyrl4rec python -m unittest examples.our_model.tests.test_rave -v
```

KuaiRand 使用 `ENV_NAME=KuaiRand-v0` 和对应环境的新版共享预训练 checkpoint。旧 `DORL_MAC` checkpoint 不兼容预编码状态契约，入口会拒绝加载。可通过 `PYTHON_BIN` 指定其他已安装依赖的解释器。

未设置 `OUTPUT_DIR` 时，训练默认写入
`saved_models/<env>/QV_suite/K<K>/<qv_type>/seed-<seed>/`。普通 Q/V、共享预训练、
Transformer Q/V 和消融分别使用同一根目录下的独立子目录。重复运行相同路径会拒绝
覆盖已有 `latest.pt`；调整 eta 等实验配置时应设置独立 `OUTPUT_DIR` 或消融
`SUITE_NAME`。独立评估默认写入 checkpoint 目录下带时间戳的
`eval_<split>_<timestamp>/`。

启动脚本默认 `SWANLAB_MODE=cloud`，项目名为 `RAVE-H`。可用
`SWANLAB_PROJECT` 和 `RUN_NAME` 修改面板项目与实验名；首次在线运行前需要完成
SwanLab 登录。设置 `SWANLAB_MODE=offline` 或 `disabled` 时不上传，仅保存本地
`training_metrics.jsonl` 和 `swanlab_metrics.jsonl`。训练 loss、分头/分 H 的
TD MAE、V MAE、梯度、fresh batch/replay 大小按 `--log_every` 上传；完整验证指标按
`--eval_every` 上传，MC 校准误差在训练结束时上传。

脚本默认屏蔽已展示历史，与现有 MAC 训练脚本一致；设置 `MASK_HISTORY=0` 可取消屏蔽。直接调用 Python CLI 时只有显式传入 `--mask_history` 才屏蔽。所有对照必须固定此设置。chunk 内按现有 actor 规则允许重复，不做退出、类别、相关性或未来目录筛选。缓存执行期间接收反馈但不重新选择 H。

## 核心模块与契约

| 文件 | 职责 |
| --- | --- |
| `assets.py` | 从训练用户交互重算并冻结主导类别，复用 BaseData 与原评估截断规则，保存来源与指纹 |
| `state.py` | 完整会话计数、有序历史、固定全局 K 锚点转移；训练与执行共用 |
| `model.py` | 默认因果 Transformer StateToken/动作编码、线性 Q/V 输出、固定尺度与共同前缀选择 |
| `training.py` | ordinary fresh imagined rollout/EMA-bootstrap TD、Transformer 完整 episode replay、ACSAC 全有效前缀 TD 与尾 chunk mask |
| `policy.py` | 基于稳定 env ID 的独立 H 缓存；`reset → act → observe` 接口 |
| `evaluation.py` | macro/pooled MCD、反馈、退出、覆盖、重复、规划与耗时指标，训练域 MC 校准 |
| `../runners/run_rave.py` | 冻结资产加载、用户划分、收集与训练、保存和独立评估 |

`gamma` 可配置，统一脚本默认 0.9。`ordinary` 的标量 R 与原 MAC/DORL 一致，直接
复用 `DORLRewardModel.reward`：predicted reward 加 entropy bonus、减 uncertainty
penalty、应用 reward shift 后截断为非负；开关、权重、entropy window 和
`predicted_mat_normalize` 均从共享 MAC checkpoint 恢复，不能被 RAVE 默认值静默替换。
两个 Transformer 分支的 R 仍是未加入这些 shaping 项的 `raw_preference`。D 为完整
会话非主导比例的增量，允许为负，跨 chunk 不重置；仅在 `gamma=1` 时，它的无折扣
和严格等于 `F_end-F_start`。C 为一次前缀首次展示记 1，多前缀训练统一重新记账。
终止步计入即时信号，终止后的 bootstrap 严格为零。

三种主对照共用训练用户划分、冻结 actor/dynamics/user model、更新预算和评估器，
但轨迹协议不同。`ordinary` 每个 optimizer step 都从训练用户初始状态重新生成
`--ordinary_rollout_depth` 个 chunk（默认 3）的 imagined rollout；自然终止尾部取零，
ordinary 热路径使用纯 tensor trajectory，不创建 `Session`、逐 item endpoint 或
Python 历史副本；同一 chunk 深度的全部 active trajectories 一起执行 MAC tensor
reward/leave，并在整个 chunk 结束时只调用一次 batch dynamics。仅训练结束后的少量
MC calibration 会按需恢复 Session 参考路径。
深度截断尾部用 EMA target V bootstrap，并按 `--lambda_chunk`（默认 1）反向构造
第一个 chunk 起点的 MAC value expansion 标签。本轮轨迹只参与本轮更新，不进入
replay；每条轨迹只有第一个 chunk 的 `Q(s0,a0)` 与 `V(s0)` 使用未缩放 MSE 直接
监督，中间 chunk 只进入 TD(lambda) target，不产生独立 Q/V loss。ordinary 使用
因果 MLP 和仅 DORL-shaped R 单头，Q 的第一个 chunk 末端使用 EMA target V
bootstrap，并固定 `H=K`；`transformer` 使用
因果 Transformer 和仅 R 单头，按 ACSAC 对每个 chunk 的所有有效前缀做 TD，部署时自适应选择 H；
`multi_prefix_transformer` 使用相同 Transformer 主干和所有前缀监督，将
`q_heads` 扩展为 R/T/D/C 四维输出。两个 Transformer 分支不训练独立 V；下一状态
从 Flow BC 重采样候选，使用在线 critic 的 stop-gradient 在候选与前缀上联合取最大值。
Transformer 的动作位置 h 只能读取 StateToken 和前 h 个动作。

共享状态在每实际展示 K 步提交全局锚点，与此前 H 分段无关。固定 H 对照也使用这套规范递推。`Session.pending` 与 `anchor` 是内部记忆；窗口历史和 mask 摘要是近似输入，不宣称完整 Markov 状态。

类别和 reward 尺度只用 RAVE 训练用户计算，validation/test 用户不参与 Q/V rollout。已有 MAC、user model 和离线初始状态的预训练划分由它们自身的训练流程决定；当前用户划分不等于验证了整个预训练链条的严格用户隔离。要做严格泛化实验，需要按预定划分重新训练这些资产。离线 pkl 仅用于 MAC 资产契约及每用户起始 observation，Q/V 标签从明确 reset 的训练模拟轨迹采集，不使用中途日志伪造新会话。

真实环境评估反馈与训练 user-model 分数分别记录来源。环境退出规则只在实际训练模拟 step/环境交互中生效，选择器没有退出预测过滤。KuaiRec/KuaiRand 当前规则确定性；评估读取实际 step 前的退出原因，以区分自然结束和 max_turn。

## 对照与输出

当前核心 Q/V 比较固定为三种：固定完整前缀的单 R MLP `ordinary`、自适应 H 的仅 R
单头 `transformer`、自适应 H 的四头 `multi_prefix_transformer`。四头模型默认使用
`eta_t=0.2, eta_d=0.2, eta_c=0.05`；需要比较其他权重时显式传参、重新训练并使用独立
输出目录。`transformer` 始终只学习 R，设置非零 eta 会被拒绝。

其他扩展对照应单独训练并使用独立输出目录：

- 新多头 fixed-H：`--fixed_horizon 1`（或合法 H），预算不足时执行剩余长度。
- 仅反馈单头：`--qv_type transformer`。
- 去掉单个目标：将相应 `--eta_t`、`--eta_d` 或 `--eta_c` 置零。
- 原 MAC 独立训练流程：直接调用 `run_dorl_mac_kuai_train.sh`，不混入统一三组主对照。
- 只选 H：`--only_h`，始终使用候选 0。
- K=3/5/7：分别加载对应 checkpoint。

`latest.pt` 包含在线/EMA 副本、优化器、冻结 MAC、业务权重、尺度、划分、类别映射和 reward 配置。评估恢复这些配置，不允许通过测试时改 eta/N 冒充重新训练的策略。当前 checkpoint 格式为 `rave_h_dorl_reward_qv_v6`，并在元数据中记录 collection/reward rule；评估仍兼容模型结构相同的 v4/v5 checkpoint。训练入口目前不支持中断续训，传 `--rave_checkpoint` 会报错。

输出包括 `resolved_config.json`、`category_metadata.json`、`objective.json`、逐步 `training_metrics.jsonl`、SwanLab 本地镜像 `swanlab_metrics.jsonl`、验证明细、`latest.pt` 和可选的 `mc_calibration.json`。`--mc_samples 0` 关闭完整训练域 rollout 校准。MC 按当前策略快照重新运行，记录中途基线；基础版不将 MC 加入优化 loss（alpha_MC=0）。TD+MC 辅助优化和非主导曝光次数 reward 对照属于后续可选扩展。

同一个 K、数据、embedding、状态表示与预训练配置下，已有
`mac_origin_precomputed_state_v2` checkpoint 可以直接复用。该 checkpoint 同时含
categorical actor 和 ChunkDynamics；RAVE-H 会按 checkpoint 中的 `actor_type` 恢复
交叉熵或 Flow actor，冻结二者，只训练新的 Q/V。旧 K3/K5/K7 checkpoint 没有
`actor_type` 字段，兼容层会将其识别为 `categorical` 交叉熵模型；它们不会自动变成
Flow actor。要使用 Flow 候选策略，需为对应 K 训练一次新的 `categorical_flow`
checkpoint，之后可在相同预训练资产配置下供多个 RAVE-H Q/V seed 复用。
相同 seed 本身不是充分条件：K 或上述资产配置变化时必须重新预训练；若实验目标是
比较完整训练流水线在不同 seed 下的方差，也应为每个 seed 重训 actor/dynamics。

Flow actor 在 `K × N_items` 的 one-hot 类别顶点空间回归条件速度，从高斯基分布经
Euler 积分得到终点 logits，再直接采样 item ID，不经过 embedding 最近邻。
训练时，一步 actor 使用同一份状态与高斯噪声回归当前多步 Flow 教师终点，教师目标
停止梯度；总损失为 `flow_loss + flow_distill_weight * distill_loss`。默认
`--flow_sampling_mode onestep`，DORL-MAC 与 RAVE-H 候选生成只需一次学生前向。
`--flow_sampling_mode teacher` 可用于教师积分对照。`--flow_sample_batch_size` 控制
一次生成的候选数，减小它可降低显存峰值。

BC/ChunkDynamics 预训练使用轻量数据行，只构造状态、动作块、item ID、有效位和
next-state 目标；Q/V 才构造历史与全目录推荐 mask。GPU 训练仍会占用 CPU 进行
离线轨迹读取、DataLoader 拼接和主机到设备传输。多组 K/seed 并行时建议限制
`OMP_NUM_THREADS`、`MKL_NUM_THREADS` 与 `OPENBLAS_NUM_THREADS`，并避免多个任务
共享同一张 GPU。

训练日志提供各 h 各头 TD MAE 与 V MAE；它们不是无偏真实回报误差。训练域完整 rollout 的 MC 误差单独报告。`inference_seconds` 包括决策和展示后的 dynamics 状态维护，另报两部分耗时。延迟分位数为 `act` 耗时；并行 adapter 以批次耗时均摊到每个动作，默认串行评估无此均摊问题。

现有验证限于合成语义测试与 KuaiRec 小规模端到端运行。完整收敛、KuaiRand 端到端、GPU 耗时及所有正式对照需要实际运行后才能形成研究结论。


## 统一五种子 Q/V 训练

`script/run_kuai_qv_suite.sh` 共享每个 seed 的 categorical Flow BC 与
ChunkDynamics，并用统一参数选择三种 Q/V：

- `ordinary`：仅原 MAC/DORL shaped R 的 MLP 完整前缀 Q/V，固定 `H=K`；
- `transformer`：仅原始反馈 R 的单头前缀 Transformer Q/V；
- `multi_prefix_transformer`：R/T/D/C 多目标前缀 Transformer Q/V。

三者使用相同的训练用户划分、冻结生成资产、更新次数、候选数、优化器和 DORL
三分支评估器。`ordinary` 使用逐更新 fresh imagined rollout（默认深度 3）且不使用
replay；两个 Transformer 使用完整 episode replay。因此三者的轨迹分布与有效 batch
大小不再声称严格相同；ordinary 与 Transformer 的 R 定义也不同。比较结果应把这些
训练协议与 reward 差异一并记录，不能只归因于网络结构。

```bash
bash script/run_kuai_qv_suite.sh \
  --env KuaiRand-v0 \
  --qv-type all \
  --seeds "2023 2024 2025 2026 2027" \
  --gpus "0 1 2 3 4" \
  --chunk-size 3
```

未设置 `QV_TYPES` 时默认只运行 `ordinary`；要比较三种核心模型需显式传入
`--qv-type all`。四头模型
需要改变决策权重时显式增加例如
`--eta-t 0.1 --eta-d 0.3 --eta-c 0.05`；这些参数不会改变仅 R 单头模型。

`--seeds` 必须恰好包含五个值。可以先用 `--stage pretrain` 只生成五份共享
BC/dynamics checkpoint，再用 `--stage qv` 训练选定 Q/V。命令行参数和同名环境变量
均可配置；已有预训练 checkpoint 时可以直接恢复 Q/V 阶段：

```bash
ENV_NAME=KuaiEnv-v0 QV_TYPES=transformer STAGE=qv CHUNK_SIZE=5 \
SEEDS="2023 2024 2025 2026 2027" GPU_IDS="0 5 6 7" \
bash script/run_kuai_qv_suite.sh
```

所有结果统一写入 `saved_models/<env>/QV_suite/K<K>/`：

```text
shared_pretrain/seed-<seed>/<bc_type>/
ordinary/seed-<seed>/
transformer/seed-<seed>/
multi_prefix_transformer/seed-<seed>/
ablations/<suite_name>/<condition>/seed-<seed>/
```

`run_rave_kuai.sh`、`run_rave_ablation_suite.sh` 与本统一入口均采用上述默认结构。
`run_dorl_mac_kuai_train.sh` 保留为原 MAC 独立复现实验，不再由三组主对照调用。
旧目录不会自动移动；
仍可通过 `RUN_DIR`、`BC_SAVE_DIR`、`OUTPUT_DIR`、`SUITE_ROOT` 或
`MAC_CHECKPOINT_TEMPLATE` 显式复用。三个脚本中的 `OUTPUT_ROOT` 都表示
`QV_suite/K<K>` 这一层；下级模型类型、seed 和消融条件目录由脚本自动创建。


退出机制在统一入口中必须显式保持一致：

```bash
bash script/run_kuai_qv_suite.sh \
  --env KuaiEnv-v0 \
  --qv-type all \
  --num-leave-compute 4 \
  --leave-threshold 0 \
  --max-turn 100 \
  --leave-policy terminate
```

`num-leave-compute` 是类别重复规则使用的最近历史窗口；候选 item 的任一类别在
该窗口中的已有计数严格大于 `leave-threshold` 时触发退出。默认 KuaiRec 窗口为
4、KuaiRand 窗口为 1，阈值均为 0。统一 trainer 始终按真实环境语义在触发动作
执行后终止，三个主对照均不接受 `penalty`，避免退出机制成为额外变量。
