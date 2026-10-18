"""RAVE-H 纯 TD 训练、训练域 MC 校准和独立真实环境评估入口。"""

import argparse
from collections import deque
from copy import copy
from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch

from examples.our_model.logging import SwanLabLogger
from examples.our_model.rave.assets import file_sha256, freeze_categories
from examples.our_model.rave.dorl_policy import RAVEDORLPolicyAdapter
from examples.our_model.rave.evaluation import monte_carlo_calibration
from examples.our_model.rave.model import PrefixQV, Utility
from examples.our_model.rave.state import StateManager
from examples.our_model.rave.training import (
    CandidatePolicy,
    RAVETrainer,
    TrainingSimulator,
    collect_complete_episodes,
    collect_imagined_rollouts,
    collect_tensor_imagined_rollouts,
)

LOGGER = logging.getLogger(__name__)
CHECKPOINT_FORMAT = "rave_h_mac_tensor_qv_v9"
"""v9 使用严格 MAC Q/V 目标和无 Session endpoint 的纯 tensor rollout。"""
COMPATIBLE_CHECKPOINT_FORMATS = {
    CHECKPOINT_FORMAT,
    "rave_h_mac_vectorized_qv_v8",
    "rave_h_mac_strict_qv_v7",
    "rave_h_dorl_reward_qv_v6",
    "rave_h_collection_mode_qv_v5",
    "rave_h_complete_episode_qv_v4",
}
"""评估可读取模型结构相同的 v4-v8 checkpoint；新训练只写 v9。"""
DEFAULT_SWANLAB_PROJECT = "RAVE-H"
"""RAVE-H 默认使用的 SwanLab 项目名。"""
DEFAULT_SWANLAB_EVAL_PROJECT = "RAVE-H-Eval"
"""RAVE-H 独立验证或测试默认使用的 SwanLab 项目名。"""
QV_TYPE_ORDINARY = "ordinary"
QV_TYPE_TRANSFORMER = "transformer"
QV_TYPE_MULTI_PREFIX_TRANSFORMER = "multi_prefix_transformer"
QV_TYPES = (
    QV_TYPE_ORDINARY,
    QV_TYPE_TRANSFORMER,
    QV_TYPE_MULTI_PREFIX_TRANSFORMER,
)
"""统一 trainer 支持的三种主 Q/V 对照。"""
DEFAULT_ETA_T = 0.2
"""四头模型默认展示长度权重。"""
DEFAULT_ETA_D = 0.2
"""四头模型默认 MCD 改善权重。"""
DEFAULT_ETA_C = 0.05
"""四头模型默认规划成本权重。"""
COLLECTION_SEED_STRIDE = 1_000_003
"""按训练 step 派生独立候选采样随机流时使用的质数跨度。"""
EVALUATION_SEED_OFFSET = 10_000_019
"""评估随机流相对训练收集随机流使用的固定偏移。"""
DEFAULT_EPISODES_PER_COLLECT = 32
"""每轮 fresh imagined 起点数或 Transformer 完整 episode 数。"""
DEFAULT_ORDINARY_ROLLOUT_DEPTH = 3
"""ordinary 每次 Q/V 更新重新生成的 imagined rollout chunk 深度。"""
DORL_REWARD_CONFIG_NAMES = (
    "use_entropy_reward",
    "use_uncertainty_penalty",
    "lambda_entropy",
    "lambda_variance",
    "predicted_mat_normalize",
    "entropy_window",
    "feature_level",
    "is_sorted",
)
"""ordinary 必须从共享 MAC checkpoint 继承的 DORL reward 配置字段。"""


def build_parser():
    """返回复用 MAC 资产参数的 RAVE 命令行解析器。"""
    from examples.our_model.runners.common import add_common_args

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(use_entropy_reward=False,
                        use_uncertainty_penalty=False, batch_size=32,
                        swanlab_project=DEFAULT_SWANLAB_PROJECT)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--mac_checkpoint", default="")
    parser.add_argument("--rave_checkpoint", default="")
    parser.add_argument(
        "--output_root",
        default="",
        help="统一 Q/V 实验根目录；为空时使用 saved_models/<env>/QV_suite/K<K>。",
    )
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_every_epochs", type=int, default=1)
    parser.add_argument(
        "--eval_swanlab_project",
        default=DEFAULT_SWANLAB_EVAL_PROJECT,
        help="训练期逐 epoch 验证使用的独立 SwanLab 项目名。",
    )
    parser.add_argument("--collect_every", type=int, default=10)
    parser.add_argument("--rollout_candidates", type=int, default=4)
    parser.add_argument(
        "--episodes_per_collect",
        type=int,
        default=DEFAULT_EPISODES_PER_COLLECT,
        help=(
            "ordinary 每轮 fresh rollout 起点数；Transformer 每次收集的完整 "
            "episode 数。"
        ),
    )
    parser.add_argument(
        "--ordinary_rollout_depth",
        type=int,
        default=DEFAULT_ORDINARY_ROLLOUT_DEPTH,
        help="ordinary 每个更新步新生成的 imagined rollout 最大 chunk 数，默认 3。",
    )
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--replay_capacity", type=int, default=4096)
    parser.add_argument("--history_window", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha_v", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument(
        "--test-num", "--test_num", dest="test_num", type=int, default=10,
        help="DORL 三分支评估使用的并行环境数量。",
    )
    parser.add_argument(
        "--buffer-size", "--buffer_size", dest="buffer_size", type=int, default=0,
        help="DORL 评估 replay buffer 容量；0 表示自动计算。",
    )
    parser.add_argument(
        "--eval_split",
        choices=["validation", "test"],
        default="validation",
        help=(
            "兼容性输出标签；共享 DORL 评估与 ordinary 一样从完整真实环境采样，"
            "不再按 RAVE 用户划分限制评估用户。"
        ),
    )
    parser.add_argument(
        "--dorl_eval_parity",
        action="store_true",
        help="独立评估显式采用 suite 传入的 ordinary 同口径环境与采样参数。",
    )
    parser.add_argument("--eval_every", type=int, default=1000,
                        help="兼容旧 checkpoint 的 step 间隔；新训练按 epoch 评估。")
    parser.add_argument("--mc_samples", type=int, default=2)
    parser.add_argument("--mask_history", action="store_true")
    parser.add_argument("--fixed_horizon", type=int, default=None)
    parser.add_argument("--only_h", action="store_true")
    parser.add_argument("--scalar_q", action="store_true")
    parser.add_argument("--backbone", choices=["transformer", "mlp"], default="transformer")
    parser.add_argument(
        "--qv_type",
        choices=QV_TYPES,
        default=None,
        help=(
            "统一 Q/V 类型：ordinary 为仅 R 的完整前缀 MLP Q/V；"
            "transformer 为仅 R 的单头前缀 Transformer Q/V；"
            "multi_prefix_transformer 为 R/T/D/C 多目标前缀 Transformer Q/V。"
            "未设置时保留 --scalar_q/--backbone 的旧接口。"
        ),
    )
    parser.add_argument("--matched_length", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--torch_threads", type=int, default=4)
    return parser


def resolve_qv_type(args):
    """解析统一 Q/V 类型、单头目标和四头决策权重。

    Args:
        args (argparse.Namespace): 包含 ``qv_type``、``scalar_q`` 和
            ``backbone`` 的运行参数。

    Returns:
        str: 解析后的稳定 Q/V 类型名称。

    Raises:
        ValueError: 类型与旧参数冲突，或单 R 头显式配置非零辅助权重时抛出。
    """
    if args.qv_type is None:
        if args.backbone == "transformer":
            args.qv_type = (
                QV_TYPE_TRANSFORMER
                if args.scalar_q
                else QV_TYPE_MULTI_PREFIX_TRANSFORMER
            )
        else:
            args.qv_type = "scalar_prefix_mlp" if args.scalar_q else "multi_prefix_mlp"
    else:
        expected_scalar = args.qv_type != QV_TYPE_MULTI_PREFIX_TRANSFORMER
        if args.scalar_q and not expected_scalar:
            raise ValueError("--scalar_q conflicts with --qv_type multi_prefix_transformer")
        args.scalar_q = expected_scalar
        if args.qv_type == QV_TYPE_ORDINARY:
            args.backbone = "mlp"
            if args.fixed_horizon not in (None, args.chunk_size):
                raise ValueError("ordinary requires fixed_horizon equal to chunk_size")
            args.fixed_horizon = args.chunk_size
        else:
            args.backbone = "transformer"

    auxiliary_weights = (args.eta_t, args.eta_d, args.eta_c)
    if args.scalar_q:
        if any(weight not in (None, 0, 0.0) for weight in auxiliary_weights):
            raise ValueError(
                "Single-head ordinary/transformer learns only R; auxiliary eta weights "
                "must be omitted or set to zero"
            )
        args.eta_t = args.eta_d = args.eta_c = 0.0
    else:
        defaults = (DEFAULT_ETA_T, DEFAULT_ETA_D, DEFAULT_ETA_C)
        args.eta_t, args.eta_d, args.eta_c = tuple(
            default if weight is None else weight
            for weight, default in zip(auxiliary_weights, defaults)
        )
        if any(weight < 0 or not np.isfinite(weight)
               for weight in (args.eta_t, args.eta_d, args.eta_c)):
            raise ValueError("Four-head eta weights must be finite and nonnegative")
    return args.qv_type


def apply_ordinary_dorl_reward_config(args, mac_checkpoint):
    """从共享 MAC checkpoint 恢复 ordinary 的 DORL reward 定义。

    Args:
        args (argparse.Namespace): 将被原地更新的 RAVE 运行参数。
        mac_checkpoint (dict): 包含原 MAC ``config`` 的共享预训练 checkpoint。

    Returns:
        dict[str, object]: 实际恢复并写入 ``args`` 的 reward 配置快照。

    Raises:
        ValueError: checkpoint 缺少任一决定 DORL reward 数值的字段时抛出。

    Example:
        ``reward_config = apply_ordinary_dorl_reward_config(args, checkpoint)``。
    """
    config = mac_checkpoint.get("config", {})
    missing = [name for name in DORL_REWARD_CONFIG_NAMES if name not in config]
    if missing:
        raise ValueError(
            "MAC checkpoint missing DORL reward configuration: "
            + ", ".join(missing)
        )
    restored = {name: config[name] for name in DORL_REWARD_CONFIG_NAMES}
    for name, value in restored.items():
        setattr(args, name, value)
    return restored


def session_training_reward(reward_model, raw_users, state, item, device,
                            use_dorl_reward):
    """计算 RAVE 模拟一步的 R，并保证 ordinary 与原 MAC/DORL 同口径。

    Args:
        reward_model (DORLRewardModel): 已按共享 MAC 配置构造的 reward 模型。
        raw_users (Sequence[int]): 环境内部用户索引到原始用户 ID 的映射。
        state (Session): 执行动作前的完整会话状态。
        item (int): 即将执行的环境内部 item ID。
        device (torch.device): reward 模型所在设备。
        use_dorl_reward (bool): 为真时使用含 entropy/uncertainty/shift 的 DORL
            shaped reward；为假时保留 Transformer 的 raw preference 定义。

    Returns:
        float: 当前动作的标量 R；DORL 模式下与原 MAC 的
        ``DORLRewardModel.reward`` 定义相同。

    Raises:
        IndexError: ``state.user_id`` 不在用户映射范围内时抛出。
        ValueError: reward 模型拒绝 item 或历史张量时向上抛出。

    Example:
        ``value = session_training_reward(model, users, state, item, device, True)``。
    """
    raw_user_id = raw_users[state.user_id]
    users = torch.tensor([raw_user_id], device=device)
    items = torch.tensor([item], device=device)
    if not use_dorl_reward:
        return float(reward_model.raw_preference(users, items)[0])
    # 原 MAC 在 reward_components() 前先把当前动作追加到历史；这里保持相同顺序。
    history = torch.tensor([state.items + [int(item)]], device=device)
    return float(reward_model.reward(users, items, history_item_ids=history)[0])


def batch_session_training_reward(reward_model, raw_users, states, item_ids,
                                  executable, device, use_dorl_reward):
    """一次计算完整 session batch 的训练 reward。

    Args:
        reward_model (DORLRewardModel): DORL reward 模型。
        raw_users (Sequence[int]): 内部用户索引到原始用户 ID 的映射。
        states (Sequence[Session]): 动作执行前的 B 个会话。
        item_ids (Tensor): 已将不可执行行替换为合法占位值的 `(B,)` item。
        executable (Tensor): `(B,)` 实际执行掩码。
        device (torch.device): reward 模型设备。
        use_dorl_reward (bool): 是否使用 shaped DORL reward。

    Returns:
        torch.Tensor: 设备上的 `(B,)` reward；不可执行行返回零。

    Raises:
        IndexError: 会话用户索引不在映射范围内时抛出。
        ValueError: batch 形状非法时抛出。

    Example:
        ``reward = batch_session_training_reward(model, users, sessions,
        items, mask, device, True)``。
    """
    batch_size = len(states)
    if item_ids.shape != (batch_size,) or executable.shape != (batch_size,):
        raise ValueError("Reward batch tensors must have shape (B,)")
    user_ids = torch.as_tensor(
        [raw_users[state.user_id] for state in states],
        dtype=torch.long,
        device=device,
    )
    safe_items = item_ids.to(device=device, dtype=torch.long)
    if use_dorl_reward:
        entropy_width = max(
            (int(window) for window in reward_model.entropy_window),
            default=1,
        )
        history_width = max(entropy_width, 1)
        history = torch.full(
            (batch_size, history_width),
            -1,
            dtype=torch.long,
            device=device,
        )
        execute_values = executable.detach().cpu().tolist()
        item_values = safe_items.detach().cpu().tolist()
        for row, state in enumerate(states):
            if not execute_values[row]:
                continue
            history_with_current = (
                state.items[-(history_width - 1):]
                if history_width > 1 else []
            ) + [int(item_values[row])]
            history[row, -len(history_with_current):] = torch.as_tensor(
                history_with_current, dtype=torch.long, device=device,
            )
        rewards = reward_model.reward(
            user_ids, safe_items, history_item_ids=history,
        )
    else:
        rewards = reward_model.raw_preference(user_ids, safe_items)
    return torch.where(
        executable.to(device=device), rewards, torch.zeros_like(rewards),
    )


def batch_session_natural_termination(leave_model, states, item_ids,
                                      executable, device):
    """用 RuleBasedLeaveModel 一次判断完整 session batch 的自然退出。

    Args:
        leave_model (RuleBasedLeaveModel): 与真实环境参数一致的退出规则模型。
        states (Sequence[Session]): 当前动作执行前的 B 个会话。
        item_ids (Tensor): 合法占位后的 `(B,)` item。
        executable (Tensor): `(B,)` 实际执行掩码。
        device (torch.device): 计算设备。

    Returns:
        torch.Tensor: 设备上的 `(B,)` bool 自然终止标记。

    Raises:
        ValueError: 输入 batch 形状不一致时抛出。
    """
    batch_size = len(states)
    if item_ids.shape != (batch_size,) or executable.shape != (batch_size,):
        raise ValueError("Leave batch tensors must have shape (B,)")
    history_width = leave_model.num_leave_compute
    history = torch.full(
        (batch_size, history_width),
        -1,
        dtype=torch.long,
        device=device,
    )
    for row, state in enumerate(states):
        recent = state.items[-history_width:]
        if recent:
            history[row, -len(recent):] = torch.as_tensor(
                recent, dtype=torch.long, device=device,
            )
    violations = leave_model.first_violation_steps(
        history,
        item_ids.to(device=device, dtype=torch.long).unsqueeze(1),
    ) == 0
    return violations & executable.to(device=device)


def write_json(path, payload):
    """将可序列化结果写为 UTF-8 JSON；拒绝 NaN，返回 None。"""
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def default_run_name(args):
    """生成同时包含环境、K 和 seed 的 RAVE-H 运行名。

    Args:
        args (argparse.Namespace): 已从 MAC checkpoint 恢复结构的完整配置。

    Returns:
        str: 用户显式名称或稳定的默认运行名。
    """
    if args.run_name:
        return args.run_name
    return f"rave-h-{args.env}-K{args.chunk_size}-seed{args.seed}"


def default_eval_run_name(args):
    """生成包含数据集、结构、划分和随机种子的独立评估运行名。

    Args:
        args (argparse.Namespace): 从 RAVE checkpoint 恢复后的完整评估配置。

    Returns:
        str: 用户显式名称或稳定且可区分的独立评估运行名。
    """
    if args.run_name:
        return args.run_name
    return (
        f"rave-h-eval-{args.env}-K{args.chunk_size}-{args.qv_type}-"
        f"{args.eval_split}-seed{args.seed}"
    )


def resolve_epoch_schedule(steps, epochs):
    """把总更新步数均匀划分为固定数量的训练 epoch。

    Args:
        steps (int): Q/V 总更新步数，必须为正整数。
        epochs (int): 总 epoch 数，必须为正整数且整除 `steps`。

    Returns:
        int: 每个 epoch 包含的 Q/V 更新步数。

    Raises:
        ValueError: 当参数非正或总步数不能被 epoch 数整除时抛出。
    """
    if steps <= 0 or epochs <= 0:
        raise ValueError("steps and epochs must be positive")
    if steps % epochs != 0:
        raise ValueError(
            f"steps ({steps}) must be divisible by epochs ({epochs})"
        )
    return steps // epochs


def resolve_output_dir(args, timestamp):
    """把训练和评估产物统一解析到 saved_models 下。

    Args:
        args (argparse.Namespace): 包含模式、save_root 与 checkpoint 路径的配置。
        timestamp (str): 用于隔离重复独立评估的时间戳。

    Returns:
        Path: 显式 output_dir 或默认模型/评估目录。
    """
    if args.output_dir:
        return Path(args.output_dir)
    if args.mode == "train":
        qv_type = getattr(args, "qv_type", None)
        if qv_type not in QV_TYPES:
            qv_type = ("transformer" if getattr(args, "scalar_q", False)
                       else QV_TYPE_MULTI_PREFIX_TRANSFORMER)
        root = (Path(args.output_root) if getattr(args, "output_root", "")
                else Path(args.save_root) / args.env / "QV_suite" /
                f"K{args.chunk_size}")
        return root / qv_type / f"seed-{args.seed}"
    checkpoint_dir = Path(args.rave_checkpoint).resolve().parent
    return checkpoint_dir / f"eval_{args.eval_split}_{timestamp}"


def evaluation_log_metrics(result, split):
    """提取可上传 SwanLab 的评估标量、旧推荐指标和 H 选择频数。

    Args:
        result (dict): `episode_metrics` 返回的完整结果，可能包含逐会话数组。
        split (str): 数据划分名称，只接受 `validation` 或 `test`。

    Returns:
        dict[str, float | int]: 带数据划分前缀的扁平指标，以及与旧
            DORL/MAC 实验表兼容的 `NX_0_*` 指标。

    Raises:
        ValueError: 当 `split` 不是受支持的数据划分时抛出。
        KeyError: 当评估结果缺少生成旧推荐指标所需的核心字段时抛出。
    """
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported evaluation split: {split}")
    metrics = {}
    for name, value in result.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            metrics[f"{split}/{name}"] = value
    for horizon, count in result.get("selected_horizons", {}).items():
        metrics[f"{split}/selected_horizon/h{horizon}"] = count

    # 旧评估器的 NX_0 表示不强制固定长度的正常闭环轨迹。RAVE 验证本身就是
    # 该执行方式，因此只做同口径字段投影，不对真实环境结果进行展示校准。
    legacy_metrics = {
        "NX_0_rew": result["raw_return_mean"],
        "NX_0_len": result["length_mean"],
        "NX_0_ctr": result["raw_feedback_per_display"],
        "NX_0_ifeat_feat": result["mcd_pooled"],
    }
    metrics.update(legacy_metrics)
    metrics["raw/NX_0_rew"] = result["raw_return_mean"]
    return metrics


def validation_log_metrics(result):
    """按训练期验证命名空间投影评估指标。

    Args:
        result (dict): `episode_metrics` 返回的完整结果。

    Returns:
        dict[str, float | int]: `validation/*` 与旧推荐指标组成的扁平字典。
    """
    return evaluation_log_metrics(result, "validation")


def calibration_log_metrics(result):
    """提取可上传 SwanLab 的训练域 MC 校准误差。

    Args:
        result (dict): `monte_carlo_calibration` 返回结果。

    Returns:
        dict[str, float]: 带 `calibration/` 前缀的各 H/分量 MAE。
    """
    return {
        f"calibration/{name}": value
        for name, value in result.get("errors", {}).items()
    }


def split_users(raw_users, seed, validation_fraction, test_fraction):
    """按 seed 划分互斥用户集合；返回 train/validation/test 内部用户 ID。"""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validation and test fractions must lie in (0,1)")
    order = np.random.default_rng(seed).permutation(len(raw_users))
    validation_count = max(1, int(len(order) * validation_fraction))
    test_count = max(1, int(len(order) * test_fraction))
    if validation_count + test_count >= len(order):
        raise ValueError("Not enough users for three disjoint splits")
    return {"validation": order[:validation_count].tolist(),
            "test": order[validation_count:validation_count + test_count].tolist(),
            "train": order[validation_count + test_count:].tolist()}


def set_collection_seed(seed, step):
    """为同 seed 的三种 Q/V 重置一致的候选与探索随机流。

    Args:
        seed (int): 实验随机种子。
        step (int): 当前训练更新步，必须为正数。

    Returns:
        int: 实际写入 PyTorch 的派生种子。

    Raises:
        ValueError: `step` 不是正数时抛出。
    """

    if step <= 0:
        raise ValueError("collection step must be positive")
    collection_seed = int(seed) * COLLECTION_SEED_STRIDE + int(step)
    torch.manual_seed(collection_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(collection_seed)
    return collection_seed


def build_assets(args, restored=None):
    """加载冻结 MAC 和原始偏好，建立仅训练域可用的模拟器及显式用户划分。

    恢复时校验数据、embedding 和 user-model 指纹，不静默更换模型或映射。
    返回运行所需的资产字典；文件缺失和 checkpoint 不兼容直接抛错。
    """
    from examples.our_model.runners.common import (
        apply_checkpoint_model_config, build_agent, build_dataset_and_mapper,
        build_env_assets, build_reward_and_leave, resolve_common_paths,
    )
    from examples.our_model.runners.evaluation_utils import build_initial_state_table

    base = restored["mac"] if restored else torch.load(args.mac_checkpoint, map_location="cpu", weights_only=False)
    if base.get("config", {}).get("env", args.env) != args.env:
        raise ValueError("MAC checkpoint environment differs from --env")
    apply_checkpoint_model_config(args, base)
    use_dorl_reward = args.qv_type == QV_TYPE_ORDINARY
    reward_config = None
    if use_dorl_reward:
        reward_config = apply_ordinary_dorl_reward_config(args, base)
        LOGGER.info("ordinary 已恢复原 MAC/DORL reward 配置：%s", reward_config)
    resolve_common_paths(args)
    device = torch.device(args.device)
    offline, mapper, embeddings = build_dataset_and_mapper(args, device)
    env, dataset, kwargs_um = build_env_assets(args)
    reward, leave = build_reward_and_leave(args, env, dataset, device)
    mac = build_agent(
        args,
        device,
        mapper,
        offline.state_dim,
        reward_model=reward,
        leave_model=leave,
    )
    mac.load_checkpoint_state(base)
    # suite ordinary 的 R 严格等于 DORL shaped reward，不叠加重复/退出罚项。
    mac.invalid_action_penalty = 0.0
    mac.eval().requires_grad_(False)
    raw_users = list(reward.raw_user_to_index)
    initial = build_initial_state_table(offline, env, device)
    fingerprints = {name: file_sha256(getattr(args, name)) for name in
                    ("dataset_path", "item_embedding_path", "predicted_mat_path")}
    if restored:
        if fingerprints != restored["fingerprints"]:
            raise ValueError("RAVE checkpoint assets have changed")
        splits, metadata = restored["splits"], restored["category_metadata"]
        categories, hits = restored["categories"], restored["hits"]
        utility = Utility(**restored["utility"])
    else:
        splits = split_users(raw_users, args.seed, args.validation_fraction, args.test_fraction)
        categories, hits, metadata = freeze_categories(
            dataset, env, args.env, [raw_users[index] for index in splits["train"]], args.top_rate)
        # 尺度只取训练用户的模型分数，并固定此样本的随机种子和数量。
        generator = np.random.default_rng(args.seed)
        user_rows = generator.choice(splits["train"], args.scale_samples)
        item_rows = generator.integers(0, len(embeddings), args.scale_samples)
        sampled_raw_users = torch.tensor(
            [raw_users[int(user_index)] for user_index in user_rows],
            device=device,
        )
        sampled_items = torch.as_tensor(item_rows, device=device)
        if use_dorl_reward:
            # 尺度样本也必须来自同一个 shaped R；单 item 历史包含当前动作，
            # 与原 MAC 调用 reward_components() 前先 append item 的顺序一致。
            samples = reward.reward(
                sampled_raw_users,
                sampled_items,
                history_item_ids=sampled_items[:, None],
            )
        else:
            samples = reward.raw_preference(sampled_raw_users, sampled_items)
        utility = Utility.from_training_rewards(samples.cpu(), args.max_turn,
                                                args.eta_t, args.eta_d, args.eta_c)
    network = PrefixQV(
        offline.state_dim, embeddings, categories, hits, args.chunk_size, args.max_turn,
        args.history_window, args.hidden_dim, args.transformer_layers, args.attention_heads,
        args.dropout, args.scalar_q, args.backbone,
    ).to(device)
    if restored:
        network.load_state_dict(restored["online"])
    policy = CandidatePolicy(mac, network, utility, args.num_candidates,
                             args.mask_history, args.fixed_horizon, args.only_h)
    manager = StateManager(mac.dynamics, mapper.item_embeddings, hits, args.max_turn)

    def training_reward(state, item):
        """返回当前分支的训练 R；ordinary 使用原 MAC/DORL shaped reward。"""
        return session_training_reward(
            reward_model=reward,
            raw_users=raw_users,
            state=state,
            item=item,
            device=device,
            use_dorl_reward=use_dorl_reward,
        )

    def natural_termination(state, item):
        """实际模拟 step 使用原环境终止语义，不读取环境 reward 矩阵。"""
        branch = copy(env)
        branch.cur_user = state.user_id
        branch.sequence_action = state.items.copy()
        branch.total_turn = len(state.items)
        return branch._determine_whether_to_leave(len(state.items), item)

    def batch_training_reward(states, item_ids, executable):
        """在完整 active batch 上一次计算当前分支的训练 R。"""

        return batch_session_training_reward(
            reward_model=reward,
            raw_users=raw_users,
            states=states,
            item_ids=item_ids,
            executable=executable,
            device=device,
            use_dorl_reward=use_dorl_reward,
        )

    def batch_natural_termination(states, item_ids, executable):
        """在完整 active batch 上一次执行与环境一致的规则退出判断。"""

        return batch_session_natural_termination(
            leave_model=leave,
            states=states,
            item_ids=item_ids,
            executable=executable,
            device=device,
        )

    simulator = TrainingSimulator(
        manager,
        training_reward,
        natural_termination,
        batch_reward_callback=batch_training_reward,
        batch_natural_termination=batch_natural_termination,
    )
    return {"mac": base, "env": env, "env_dataset": dataset,
            "env_kwargs": kwargs_um, "initial": initial, "splits": splits,
            "categories": categories, "hits": hits, "category_metadata": metadata,
            "fingerprints": fingerprints, "utility": utility, "policy": policy,
            "manager": manager, "simulator": simulator, "raw_users": raw_users,
            "reward_config": reward_config,
            "training_reward_rule": (
                "dorl_shaped_reward_v1" if use_dorl_reward else "raw_preference_v1"
            )}


def build_shared_dorl_evaluator(args, assets, save_dir):
    """为三种统一 Q/V 构造同一 DORL 三分支评估器。

    Args:
        args (argparse.Namespace): RAVE 训练或独立测试配置。
        assets (dict): `build_assets` 返回的 actor、Transformer Q、环境和状态资产。
        save_dir (Path): 当前评估 summary 的保存目录。

    Returns:
        DORLMACEvaluator: 使用 FB、NX_0、NX_force 和原 DORL callbacks 的
        可复用评估器。

    Raises:
        ValueError: 当公共评估参数或策略资产不合法时抛出。
    """

    from examples.our_model.runners.evaluation_utils import (
        build_dorl_policy_evaluator,
    )

    adapter = RAVEDORLPolicyAdapter(
        candidate_policy=assets["policy"],
        manager=assets["manager"],
        initial_states=assets["initial"],
        device=torch.device(args.device),
    )
    return build_dorl_policy_evaluator(
        args=args,
        env=assets["env"],
        dataset=assets["env_dataset"],
        kwargs_um=assets["env_kwargs"],
        policy=adapter,
        eval_episodes=args.eval_episodes,
        save_dir=Path(save_dir),
        buffer_size=args.buffer_size,
        completion_window=args.completion_window,
    )


def train(args, assets, output, tracker, evaluator):
    """收集训练策略访问状态并将过程指标同时写入本地与 SwanLab。

    Args:
        args (argparse.Namespace): 已解析且冻结的训练配置。
        assets (dict): 冻结模型、训练模拟器、用户划分和策略资产。
        output (Path): checkpoint、JSONL 和验证结果保存目录。
        tracker (SwanLabLogger): SwanLab 与本地镜像日志器。
        evaluator (DORLMACEvaluator): 与 ordinary 共用的 DORL 三分支评估器。

    Returns:
        Path: 最终 `latest.pt` checkpoint 路径。
    """
    policy, simulator = assets["policy"], assets["simulator"]
    trainer = RAVETrainer(
        policy,
        args.learning_rate,
        args.tau,
        args.alpha_v,
        optimize_all_prefixes=args.qv_type != QV_TYPE_ORDINARY,
        gamma=args.gamma,
        lambda_chunk=args.lambda_chunk,
    )
    steps_per_epoch = resolve_epoch_schedule(args.steps, args.epochs)
    ordinary = args.qv_type == QV_TYPE_ORDINARY
    replay = None if ordinary else deque(maxlen=args.replay_capacity)
    generator = random.Random(args.seed)
    collection_rule = (
        "fresh_imagined_pure_tensor_first_state_td_lambda_v4"
        if ordinary else "complete_episode_fixed_K_v1"
    )
    metadata = {"source": "training_user_model_rollout", "seed": args.seed,
                "model_version": assets["fingerprints"]["predicted_mat_path"],
                "state_rule": "global_K_anchor_v1",
                "collection_rule": collection_rule,
                "reward_rule": assets["training_reward_rule"]}
    latest_collection_metrics = {}
    latest_records = []
    latest_tensor_trajectory = None
    log_path = output / "training_metrics.jsonl"
    with log_path.open("w", encoding="utf-8") as stream:
        for step in range(1, args.steps + 1):
            current_epoch = (step - 1) // steps_per_epoch + 1
            should_collect = ordinary or step == 1 or (step - 1) % args.collect_every == 0
            if should_collect:
                # 网络结构会消耗不同数量的初始化/dropout 随机数。每次收集前按
                # seed 和 step 重置候选随机流，避免这些无关消耗改变 actor 候选池。
                set_collection_seed(args.seed, step)
                users = [
                    generator.choice(assets["splits"]["train"])
                    for _ in range(args.episodes_per_collect)
                ]
                if ordinary:
                    latest_tensor_trajectory = collect_tensor_imagined_rollouts(
                        policy=policy,
                        user_ids=users,
                        initial_states=assets["initial"],
                        raw_users=assets["raw_users"],
                        dominant_hits=assets["hits"],
                        max_turn=args.max_turn,
                        epsilon=args.epsilon,
                        num_candidates=args.rollout_candidates,
                        rollout_depth=args.ordinary_rollout_depth,
                        gamma=args.gamma,
                    )
                    completed_episodes = (
                        latest_tensor_trajectory.completed_episodes
                    )
                    episode_count = latest_tensor_trajectory.batch_size
                    collected_chunks = latest_tensor_trajectory.total_chunks
                    collected_steps = latest_tensor_trajectory.total_steps
                    tail_chunks = latest_tensor_trajectory.tail_chunks
                else:
                    collected, tail_states = collect_complete_episodes(
                        policy=policy,
                        simulator=simulator,
                        manager=assets["manager"],
                        user_ids=users,
                        initial_states=assets["initial"],
                        metadata=metadata,
                        epsilon=args.epsilon,
                        num_candidates=args.rollout_candidates,
                        gamma=args.gamma,
                    )
                    replay.extend(collected)
                    completed_episodes = len(tail_states)
                    latest_records = collected
                    episode_count = len(tail_states)
                    collected_chunks = len(collected)
                    collected_steps = sum(
                        len(record.endpoints) for record in collected
                    )
                    tail_chunks = sum(
                        len(record.endpoints) < args.chunk_size
                        for record in collected
                    )
                collection_metrics = {
                    "collection/source_step": step,
                    "collection/episodes": episode_count,
                    "collection/completed_episodes": completed_episodes,
                    "collection/chunks": collected_chunks,
                    "collection/steps": collected_steps,
                    "collection/tail_chunks": tail_chunks,
                    "collection/mean_chunk_length": (
                        collected_steps / collected_chunks
                    ),
                    "collection/rollout_depth": (
                        args.ordinary_rollout_depth if ordinary else 0
                    ),
                    "collection/batch_vectorized": int(ordinary),
                    "collection/pure_tensor": int(ordinary),
                }
                LOGGER.info(
                    "%s 收集完成：step=%s episodes=%s chunks=%s steps=%s tails=%s",
                    "fresh imagined rollout" if ordinary else "完整 episode",
                    step,
                    episode_count,
                    collected_chunks,
                    collected_steps,
                    tail_chunks,
                )
                latest_collection_metrics = collection_metrics
            if ordinary:
                metrics = trainer.update_tensor_trajectory(
                    latest_tensor_trajectory,
                )
                fresh_batch_size = latest_tensor_trajectory.batch_size
            else:
                source_records = list(replay)
                batch = generator.sample(
                    source_records, min(args.batch_size, len(source_records)),
                )
                metrics = trainer.update(batch)
                fresh_batch_size = 0
            metrics.update(latest_collection_metrics)
            metrics["fresh_batch_size"] = fresh_batch_size
            metrics["replay_size"] = 0 if ordinary else len(replay)
            metrics["epoch"] = current_epoch
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            do_log = step == 1 or step % args.log_every == 0 or step == args.steps
            if do_log:
                stream.flush()
                tracker.log({f"train/{key}": value for key, value in metrics.items()},
                            step=step)
                LOGGER.info(
                    "RAVE step=%s loss=%.6f Q=%.6f V=%.6f fresh=%s replay=%s",
                    step, metrics["loss"], metrics["q_loss"], metrics["v_loss"],
                    metrics["fresh_batch_size"], metrics["replay_size"],
                )
            epoch_finished = step % steps_per_epoch == 0
            should_evaluate = epoch_finished and (
                current_epoch % args.eval_every_epochs == 0
                or current_epoch == args.epochs
            )
            if should_evaluate:
                set_collection_seed(
                    args.seed + EVALUATION_SEED_OFFSET,
                    current_epoch,
                )
                result = evaluator.evaluate(
                    epoch=current_epoch,
                    global_step=step,
                )
                # 与 ordinary Q/V 相同：训练期评估直接写入当前 Q/V run，
                # 使用 epoch 作为横轴，便于在同一 SwanLab 项目中比较。
                tracker.log(result, step=current_epoch)
                payload = {key: assets[key] for key in (
                    "mac", "splits", "categories", "hits", "category_metadata", "fingerprints")}
                payload.update({"format": CHECKPOINT_FORMAT, "config": vars(args).copy(),
                                "utility": asdict(assets["utility"]),
                                "online": policy.network.state_dict(),
                                "target": trainer.target.state_dict(),
                                "optimizer": trainer.optimizer.state_dict(), "step": step,
                                "epoch": current_epoch,
                                "state_rule": "global_K_anchor_v1",
                                "collection_rule": collection_rule,
                                "rollout_engine": (
                                    "pure_tensor_trajectory_v2"
                                    if ordinary else "scalar_session_v1"
                                ),
                                "reward_rule": assets["training_reward_rule"],
                                "reward_config": assets["reward_config"],
                                "td_objective": (
                                    "mac_strict_first_state_td_lambda_mse"
                                    if ordinary
                                    else "acsac_all_valid_prefix_q"
                                )})
                torch.save(payload, output / "latest.pt")
                LOGGER.info(
                    "DORL 三分支验证及 checkpoint 已保存：epoch=%s/%s "
                    "step=%s NX_0_rew=%.4f NX_0_len=%.2f",
                    current_epoch, args.epochs, step,
                    float(result.get("NX_0_rew", float("nan"))),
                    float(result.get("NX_0_len", float("nan"))),
                )
    if args.mc_samples:
        if ordinary:
            calibration_users = [
                generator.choice(assets["splits"]["train"])
                for _ in range(args.mc_samples)
            ]
            calibration_episodes, _ = collect_imagined_rollouts(
                policy=policy,
                simulator=simulator,
                manager=assets["manager"],
                user_ids=calibration_users,
                initial_states=assets["initial"],
                metadata=metadata,
                epsilon=0.0,
                num_candidates=args.rollout_candidates,
                rollout_depth=args.ordinary_rollout_depth,
            )
            # 校准只在训练结束执行一次；按需恢复少量 Session 记录，不污染热路径。
            calibration_source = [episode[0] for episode in calibration_episodes]
        else:
            calibration_source = list(replay)
        records = generator.sample(
            calibration_source,
            min(args.mc_samples, len(calibration_source)),
        )
        calibration = monte_carlo_calibration(
            policy, simulator, records, f"update_{trainer.updates}")
        write_json(output / "mc_calibration.json", calibration)
        tracker.log(calibration_log_metrics(calibration), step=trainer.updates)
    # SwanLab 折线指标不接受字符串；checkpoint 路径已保存在 resolved config
    # 和固定目录中，这里只记录数值型完成标记，避免产生无效图表警告。
    tracker.log({"artifact/latest_checkpoint_saved": 1}, step=trainer.updates)
    return output / "latest.pt"


def main(argv=None):
    """解析配置并完成训练或评估；返回结果路径，非法参数及时抛 ValueError。"""
    args = build_parser().parse_args(argv)
    if args.mode == "train":
        resolve_qv_type(args)
    if args.mode == "train" and args.rave_checkpoint:
        raise ValueError("Training resume is not supported; use --mac_checkpoint for a fresh run")
    required_checkpoint = args.mac_checkpoint if args.mode == "train" else args.rave_checkpoint
    if not required_checkpoint or not Path(required_checkpoint).is_file():
        raise ValueError("Provide an existing --mac_checkpoint (train) or --rave_checkpoint (eval)")
    restored = None
    if args.mode == "eval":
        restored = torch.load(args.rave_checkpoint, map_location="cpu", weights_only=False)
        if restored.get("format") not in COMPATIBLE_CHECKPOINT_FORMATS:
            raise ValueError("Incompatible RAVE checkpoint")
        # 部署 N、业务权重、尺度及模型结构全部采用已训练策略配置。
        runtime_names = [
            "mode", "rave_checkpoint", "output_dir", "eval_episodes",
            "eval_split", "device", "cpu", "torch_threads",
            "swanlab_project", "run_name", "dorl_eval_parity",
        ]
        if args.dorl_eval_parity:
            runtime_names.extend([
                "env", "max_turn", "num_leave_compute", "leave_threshold",
                "force_length", "completion_window", "test_num", "buffer_size",
                "num_candidates", "fixed_horizon",
            ])
        runtime_keys = {key: getattr(args, key) for key in runtime_names}
        vars(args).update(restored["config"])
        vars(args).update(runtime_keys)
        if "qv_type" not in restored["config"]:
            args.qv_type = None
        resolve_qv_type(args)
    if not 0 <= args.gamma <= 1:
        raise ValueError("RAVE gamma must lie in [0,1]")
    if not 0 <= args.lambda_chunk <= 1:
        raise ValueError("RAVE lambda_chunk must lie in [0,1]")
    if args.execution_horizon is not None:
        raise ValueError("Use --fixed_horizon for RAVE; --execution_horizon is a legacy MAC option")
    if not 0 <= args.epsilon <= 1:
        raise ValueError("epsilon must lie in [0,1]")
    if args.cpu:
        args.device = "cpu"
    if args.enable_open_loop_diagnostics or args.use_exposure_intervention:
        raise ValueError("RAVE does not use shadow replan or exposure intervention")
    positive = ("steps", "epochs", "eval_every_epochs", "batch_size", "collect_every",
                "rollout_candidates", "episodes_per_collect", "num_candidates", "test_num",
                "replay_capacity", "scale_samples", "eval_every", "eval_episodes",
                "log_every", "max_turn", "matched_length", "torch_threads",
                "ordinary_rollout_depth")
    if any(getattr(args, name) <= 0 for name in positive) or args.mc_samples < 0:
        raise ValueError("Training and evaluation counts must be positive")
    resolve_epoch_schedule(args.steps, args.epochs)
    torch.set_num_threads(args.torch_threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    assets = build_assets(args, restored)
    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    output = resolve_output_dir(args, timestamp)
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "train" and (output / "latest.pt").exists():
        raise FileExistsError("Choose a fresh run_name/output_dir to preserve the existing run")
    args.output_dir = str(output)
    if args.mode == "train":
        args.run_name = default_run_name(args)
    else:
        if args.swanlab_project == DEFAULT_SWANLAB_PROJECT:
            args.swanlab_project = DEFAULT_SWANLAB_EVAL_PROJECT
        args.run_name = default_eval_run_name(args)
    write_json(output / "resolved_config.json", vars(args))
    write_json(output / "category_metadata.json", assets["category_metadata"])
    write_json(output / "objective.json", asdict(assets["utility"]))
    LOGGER.info("RAVE assets ready: K=%s N=%s scales=%s splits=%s",
                args.chunk_size, args.num_candidates, assets["utility"].scales,
                {key: len(value) for key, value in assets["splits"].items()})
    tracker = None
    evaluator = build_shared_dorl_evaluator(args, assets, output)
    try:
        tracker = SwanLabLogger(
            project=args.swanlab_project,
            run_name=args.run_name,
            config=vars(args),
            log_path=str(output / "swanlab_metrics.jsonl"),
        )
        if args.mode == "train":
            checkpoint_path = train(
                args, assets, output, tracker, evaluator,
            )
            return checkpoint_path
        set_collection_seed(args.seed + EVALUATION_SEED_OFFSET, 1)
        result = evaluator.evaluate(epoch=0, global_step=0)
        destination = output / "summary_metrics.json"
        write_json(destination, result)
        # 与 ordinary 独立评估相同，最终测试固定记录在 step=0。
        tracker.log(result, step=0)
        LOGGER.info("独立评估及 SwanLab 记录已完成：split=%s episodes=%s output=%s",
                    args.eval_split, args.eval_episodes, destination)
        return destination
    finally:
        evaluator.close()
        if tracker is not None:
            tracker.finish()
        assets["env"].close()


if __name__ == "__main__":
    main()
