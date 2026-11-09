"""评估 K=1 动力学在长程自回归 rollout 中的累计状态误差。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.our_model.models.chunk_dynamics import ChunkDynamics  # noqa: E402
from examples.our_model.runners.common import (  # noqa: E402
    configure_logging,
    ensure_dir,
    resolve_device,
    set_mpl_cache_to_tmp,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_ROOT = (
    "saved_models/KuaiEnv-v0/QV_suite/K1/ordinary"
)
"""K=1 ordinary 多随机种子 checkpoint 的默认根目录。"""

DEFAULT_DATASET_PATH = (
    "data/KuaiRec/data_processed/DM_KuaiEnv-v0_small_data.pkl"
)
"""KuaiRec 预编码真实轨迹的默认路径。"""

DEFAULT_OUTPUT_DIR = (
    "saved_models/KuaiEnv-v0/QV_suite/K1/compounding_error"
)
"""累计误差实验的默认输出目录。"""

DEFAULT_SEEDS = (2023, 2024, 2025, 2026, 2027)
"""与 QV_suite 一致的五个训练随机种子。"""

DEFAULT_MAX_HORIZON = 100
"""默认递归预测的最大底层交互步数。"""

DEFAULT_STARTS_PER_USER = 10
"""每个用户默认抽取的真实轨迹起点数。"""

DEFAULT_BATCH_SIZE = 1024
"""动力学前向计算的默认 batch size。"""

DEFAULT_BOOTSTRAP_SAMPLES = 2000
"""用户级配对 bootstrap 的默认重采样次数。"""

DEFAULT_SAMPLING_SEED = 2029
"""真实轨迹起点抽样使用的固定随机种子。"""

DEFAULT_BOOTSTRAP_SEED = 2030
"""bootstrap 使用的固定随机种子。"""

DEFAULT_MIN_START = 3
"""跳过 StateTrackerAvg 初始化过渡段的默认最小起点。"""

DEFAULT_SCALE_EPS = 1.0e-6
"""判断状态维度是否具有有效方差的阈值。"""

DEFAULT_CI_LEVEL = 0.95
"""主图与汇总表使用的双侧置信水平。"""

CHECKPOINT_FORMAT = "mac_origin_precomputed_state_v2"
"""当前 MAC_origin 预编码状态 checkpoint 格式。"""

SEED_DIR_PATTERN = re.compile(r"^seed-(?P<seed>[0-9]+)$")
"""从 checkpoint 父目录解析训练 seed 的目录格式。"""


@dataclass(frozen=True)
class CheckpointSpec:
    """保存一个 K=1 dynamics checkpoint 的已校验元数据。

    Attributes:
        model_seed (int): 模型训练随机种子。
        path (Path): checkpoint 文件路径。
        payload (dict[str, Any]): 已加载的 checkpoint 内容。
        state_dim (int): 状态维度。
        action_dim (int): 单步动作 embedding 维度。
        hidden_dims (tuple[int, ...]): dynamics 隐藏层宽度。
    """

    model_seed: int
    path: Path
    payload: dict[str, Any]
    state_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...]


@dataclass(frozen=True)
class SamplingPlan:
    """描述固定日志动作实验使用的用户与轨迹起点。

    Attributes:
        trajectory_indices (np.ndarray): 每个窗口对应的轨迹下标。
        user_indices (np.ndarray): 每个窗口对应的紧凑用户下标。
        starts (np.ndarray): 每个窗口在原轨迹中的起点。
        user_ids (np.ndarray): 紧凑用户下标对应的原始 user id。
        starts_per_user (np.ndarray): 每个用户实际抽取的窗口数。
    """

    trajectory_indices: np.ndarray
    user_indices: np.ndarray
    starts: np.ndarray
    user_ids: np.ndarray
    starts_per_user: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    """构造累计误差实验命令行参数。

    Returns:
        argparse.ArgumentParser: 可直接解析实验配置的参数解析器。
    """

    parser = argparse.ArgumentParser(
        description=(
            "固定真实日志动作，比较 K=1 dynamics 的 Teacher Forcing 与 "
            "Free-Running 长程状态误差。"
        )
    )
    parser.add_argument(
        "--checkpoint-root", type=str, default=DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument(
        "--dataset-path", type=str, default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--max-horizon", type=int, default=DEFAULT_MAX_HORIZON,
    )
    parser.add_argument(
        "--starts-per-user", type=int, default=DEFAULT_STARTS_PER_USER,
    )
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--min-start", type=int, default=DEFAULT_MIN_START)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--bootstrap-samples", type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument(
        "--sampling-seed", type=int, default=DEFAULT_SAMPLING_SEED,
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """校验累计误差实验参数的边界条件。

    Args:
        args (argparse.Namespace): 命令行解析后的实验参数。

    Raises:
        ValueError: 当 horizon、样本量、batch size 或 seed 配置非法时抛出。
    """

    if args.max_horizon <= 0:
        raise ValueError("max_horizon must be positive.")
    if args.starts_per_user <= 0:
        raise ValueError("starts_per_user must be positive.")
    if args.max_users is not None and args.max_users <= 0:
        raise ValueError("max_users must be positive when provided.")
    if args.min_start < 0:
        raise ValueError("min_start must be non-negative.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be non-empty and unique.")


def load_checkpoint_specs(
    checkpoint_root: Path,
    model_seeds: Sequence[int],
) -> list[CheckpointSpec]:
    """加载并校验多个 K=1 ordinary checkpoint。

    Args:
        checkpoint_root (Path): 包含 `seed-<seed>/latest.pt` 的根目录。
        model_seeds (Sequence[int]): 需要加载的训练随机种子。

    Returns:
        list[CheckpointSpec]: 按输入 seed 顺序排列的 checkpoint 元数据。

    Raises:
        FileNotFoundError: 当任一 checkpoint 不存在时抛出。
        ValueError: 当 checkpoint 格式、K 或维度不一致时抛出。
    """

    specs: list[CheckpointSpec] = []
    reference_dims: tuple[int, int] | None = None
    for model_seed in model_seeds:
        checkpoint_path = checkpoint_root / f"seed-{model_seed}" / "latest.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_path}"
            )
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported checkpoint format at {checkpoint_path}: "
                f"{payload.get('format')!r}."
            )
        config = payload.get("config", {})
        if int(config.get("chunk_size", -1)) != 1:
            raise ValueError(
                f"Compounding experiment requires chunk_size=1: {checkpoint_path}"
            )
        state_dim = int(config.get("state_dim", -1))
        action_dim = int(config.get("action_dim", -1))
        hidden_dims = tuple(
            int(value) for value in config.get("dynamics_hidden_dims", ())
        )
        if state_dim <= 0 or action_dim <= 0 or not hidden_dims:
            raise ValueError(
                f"Checkpoint lacks valid dynamics dimensions: {checkpoint_path}"
            )
        if payload.get("dynamics") is None:
            raise ValueError(f"Checkpoint lacks dynamics weights: {checkpoint_path}")
        parsed_seed_match = SEED_DIR_PATTERN.match(checkpoint_path.parent.name)
        if (
            parsed_seed_match is None
            or int(parsed_seed_match.group("seed")) != int(model_seed)
        ):
            raise ValueError(f"Checkpoint seed directory is invalid: {checkpoint_path}")
        current_dims = (state_dim, action_dim)
        if reference_dims is None:
            reference_dims = current_dims
        elif current_dims != reference_dims:
            raise ValueError(
                "All checkpoints must share state/action dimensions: "
                f"expected {reference_dims}, got {current_dims}."
            )
        specs.append(
            CheckpointSpec(
                model_seed=int(model_seed),
                path=checkpoint_path,
                payload=payload,
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
            )
        )
    return specs


def load_trajectories(dataset_path: Path) -> list[dict[str, Any]]:
    """加载预编码真实轨迹并验证顶层格式。

    Args:
        dataset_path (Path): 包含 `list[dict]` 的 pickle 文件。

    Returns:
        list[dict[str, Any]]: 原始用户轨迹列表。

    Raises:
        FileNotFoundError: 当数据文件不存在时抛出。
        ValueError: 当数据为空或顶层格式不正确时抛出。
    """

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    LOGGER.info("加载真实轨迹：%s", dataset_path)
    with dataset_path.open("rb") as file_obj:
        trajectories = pickle.load(file_obj)
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("Trajectory dataset must be a non-empty list.")
    return trajectories


def valid_start_indices(
    terminals: np.ndarray,
    trajectory_length: int,
    max_horizon: int,
    min_start: int,
) -> np.ndarray:
    """返回不会跨越 terminal 的合法长程预测起点。

    Args:
        terminals (np.ndarray): 长度为 T 的终止标记。
        trajectory_length (int): 轨迹长度 T。
        max_horizon (int): 每个起点需要覆盖的动作数。
        min_start (int): 允许采样的最小全局位置。

    Returns:
        np.ndarray: 按升序排列的合法起点。

    Raises:
        ValueError: 当输入长度或 horizon 非法时抛出。

    Example:
        >>> valid_start_indices(np.array([False, True, False, False]), 4, 2, 0)
        array([0, 2])
    """

    terminal_flags = np.asarray(terminals, dtype=bool).reshape(-1)
    if trajectory_length <= 0 or len(terminal_flags) != trajectory_length:
        raise ValueError("Invalid trajectory length or terminal array.")
    if max_horizon <= 0 or min_start < 0:
        raise ValueError("max_horizon must be positive and min_start non-negative.")

    valid_starts: list[np.ndarray] = []
    segment_start = 0
    terminal_positions = np.flatnonzero(terminal_flags)
    segment_ends = [int(position) + 1 for position in terminal_positions]
    if not segment_ends or segment_ends[-1] < trajectory_length:
        segment_ends.append(trajectory_length)
    for segment_end in segment_ends:
        first_start = max(segment_start, min_start)
        last_start = segment_end - max_horizon
        if first_start <= last_start:
            valid_starts.append(
                np.arange(first_start, last_start + 1, dtype=np.int64)
            )
        segment_start = segment_end
    if not valid_starts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(valid_starts)


def build_sampling_plan(
    trajectories: Sequence[dict[str, Any]],
    max_horizon: int,
    starts_per_user: int,
    min_start: int,
    sampling_seed: int,
    max_users: int | None,
) -> SamplingPlan:
    """按用户等权抽取长程日志窗口。

    Args:
        trajectories (Sequence[dict[str, Any]]): 用户轨迹列表。
        max_horizon (int): 每个窗口的预测长度。
        starts_per_user (int): 每个用户最多抽取的起点数。
        min_start (int): 跳过的初始化位置数。
        sampling_seed (int): 起点和用户抽样随机种子。
        max_users (int | None): 可选用户上限；None 表示使用全部用户。

    Returns:
        SamplingPlan: 固定后可复用于全部模型 seed 的抽样计划。

    Raises:
        ValueError: 当没有任何用户能提供完整窗口时抛出。
    """

    rng = np.random.default_rng(sampling_seed)
    candidate_indices = np.arange(len(trajectories), dtype=np.int64)
    if max_users is not None and max_users < len(candidate_indices):
        candidate_indices = np.sort(
            rng.choice(candidate_indices, size=max_users, replace=False)
        )

    trajectory_indices: list[int] = []
    compact_user_indices: list[int] = []
    sampled_starts: list[int] = []
    user_ids: list[int] = []
    starts_per_user_actual: list[int] = []

    for trajectory_index in candidate_indices:
        trajectory = trajectories[int(trajectory_index)]
        actions = np.asarray(trajectory["actions"])
        terminals = np.asarray(trajectory["terminals"], dtype=bool)
        valid_starts = valid_start_indices(
            terminals=terminals,
            trajectory_length=len(actions),
            max_horizon=max_horizon,
            min_start=min_start,
        )
        if not len(valid_starts):
            continue
        sample_count = min(starts_per_user, len(valid_starts))
        starts = np.sort(
            rng.choice(valid_starts, size=sample_count, replace=False)
        )
        compact_index = len(user_ids)
        trajectory_indices.extend([int(trajectory_index)] * sample_count)
        compact_user_indices.extend([compact_index] * sample_count)
        sampled_starts.extend(int(start) for start in starts)
        user_ids.append(int(trajectory["user_id"]))
        starts_per_user_actual.append(sample_count)

    if not user_ids:
        raise ValueError("No trajectory can provide the requested horizon.")
    return SamplingPlan(
        trajectory_indices=np.asarray(trajectory_indices, dtype=np.int64),
        user_indices=np.asarray(compact_user_indices, dtype=np.int64),
        starts=np.asarray(sampled_starts, dtype=np.int64),
        user_ids=np.asarray(user_ids, dtype=np.int64),
        starts_per_user=np.asarray(starts_per_user_actual, dtype=np.int64),
    )


def iter_window_batches(
    trajectories: Sequence[dict[str, Any]],
    sampling_plan: SamplingPlan,
    max_horizon: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """按 batch 读取 Teacher Forcing 输入、动作与真实下一状态。

    Args:
        trajectories (Sequence[dict[str, Any]]): 原始用户轨迹。
        sampling_plan (SamplingPlan): 已冻结的用户与起点抽样计划。
        max_horizon (int): 每个窗口长度。
        batch_size (int): 单次返回的窗口数量。

    Yields:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 分别为紧凑用户
        下标 `(B,)`、真实当前状态 `(B,H,D_s)`、固定日志动作
        `(B,H,D_a)` 和真实下一状态 `(B,H,D_s)`。
    """

    num_windows = len(sampling_plan.starts)
    for offset in range(0, num_windows, batch_size):
        end = min(offset + batch_size, num_windows)
        user_indices = sampling_plan.user_indices[offset:end]
        teacher_states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for trajectory_index, start in zip(
            sampling_plan.trajectory_indices[offset:end],
            sampling_plan.starts[offset:end],
        ):
            trajectory = trajectories[int(trajectory_index)]
            window_slice = slice(int(start), int(start) + max_horizon)
            teacher_states.append(
                np.asarray(trajectory["observations"][window_slice], dtype=np.float32)
            )
            actions.append(
                np.asarray(trajectory["actions"][window_slice], dtype=np.float32)
            )
            targets.append(
                np.asarray(
                    trajectory["next_observations"][window_slice],
                    dtype=np.float32,
                )
            )
        yield (
            user_indices,
            np.stack(teacher_states),
            np.stack(actions),
            np.stack(targets),
        )


def compute_state_scale(
    trajectories: Sequence[dict[str, Any]],
    sampling_plan: SamplingPlan,
    max_horizon: int,
    batch_size: int,
    scale_eps: float = DEFAULT_SCALE_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """从评估目标状态估计逐维标准差并筛除常量维度。

    Args:
        trajectories (Sequence[dict[str, Any]]): 原始用户轨迹。
        sampling_plan (SamplingPlan): 固定抽样计划。
        max_horizon (int): 最大预测时域。
        batch_size (int): 数据统计 batch size。
        scale_eps (float): 最小有效标准差阈值。

    Returns:
        tuple[np.ndarray, np.ndarray]: 状态逐维标准差和有效维度布尔标记。

    Raises:
        ValueError: 当没有目标状态或所有维度均为常量时抛出。
    """

    sum_values: np.ndarray | None = None
    sum_squares: np.ndarray | None = None
    count = 0
    for _, _, _, targets in iter_window_batches(
        trajectories, sampling_plan, max_horizon, batch_size,
    ):
        flattened = targets.reshape(-1, targets.shape[-1]).astype(np.float64)
        if sum_values is None:
            sum_values = np.zeros(flattened.shape[1], dtype=np.float64)
            sum_squares = np.zeros_like(sum_values)
        sum_values += flattened.sum(axis=0)
        sum_squares += np.square(flattened).sum(axis=0)
        count += len(flattened)
    if count <= 0 or sum_values is None or sum_squares is None:
        raise ValueError("No target states are available for normalization.")
    means = sum_values / count
    variances = np.maximum(sum_squares / count - np.square(means), 0.0)
    scales = np.sqrt(variances)
    valid_dimensions = scales > scale_eps
    if not np.any(valid_dimensions):
        raise ValueError("All state dimensions have near-zero variance.")
    return scales.astype(np.float32), valid_dimensions


def evaluate_dynamics_batch(
    dynamics: Any,
    teacher_states: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    state_scale: torch.Tensor,
    valid_dimensions: torch.Tensor,
) -> dict[str, np.ndarray]:
    """计算单个 batch 的 Free-Running 与 Teacher Forcing 误差曲线。

    Args:
        dynamics (Any): 接受 `(states, actions, valid)` 的 K=1 dynamics。
        teacher_states (torch.Tensor): 真实当前状态，形状 `(B,H,D_s)`。
        actions (torch.Tensor): 固定日志动作，形状 `(B,H,D_a)`。
        targets (torch.Tensor): 真实下一状态，形状 `(B,H,D_s)`。
        state_scale (torch.Tensor): 状态逐维标准差，形状 `(D_s,)`。
        valid_dimensions (torch.Tensor): 有效归一化维度标记 `(D_s,)`。

    Returns:
        dict[str, np.ndarray]: 每个误差指标对应 `(B,H)` NumPy 数组。

    Raises:
        ValueError: 当输入形状、状态尺度或有效维度非法时抛出。
        FloatingPointError: 当模型输出或误差包含非有限值时抛出。

    Example:
        真实状态每步加一、模型每步额外加 0.1 时，Free-Running 误差随
        horizon 线性增加，而 Teacher Forcing 误差保持为 0.1。
    """

    if teacher_states.ndim != 3 or actions.ndim != 3 or targets.ndim != 3:
        raise ValueError("States, actions and targets must be 3D tensors.")
    if teacher_states.shape != targets.shape:
        raise ValueError("teacher_states and targets must share shape.")
    if teacher_states.shape[:2] != actions.shape[:2]:
        raise ValueError("State and action batch/horizon shapes must match.")
    if state_scale.shape != (teacher_states.shape[-1],):
        raise ValueError("state_scale shape does not match state dimension.")
    if valid_dimensions.shape != state_scale.shape or not valid_dimensions.any():
        raise ValueError("valid_dimensions must select at least one state dimension.")

    batch_size, max_horizon, _ = teacher_states.shape
    valid = torch.ones((batch_size, 1), dtype=torch.bool, device=actions.device)
    free_state = teacher_states[:, 0]
    metric_names = (
        "free_nrmse",
        "teacher_nrmse",
        "free_rmse",
        "teacher_rmse",
        "free_cosine_distance",
        "teacher_cosine_distance",
    )
    metrics = {
        metric_name: np.empty((batch_size, max_horizon), dtype=np.float32)
        for metric_name in metric_names
    }

    with torch.no_grad():
        for horizon_index in range(max_horizon):
            action_step = actions[:, horizon_index:horizon_index + 1]
            target_step = targets[:, horizon_index]
            free_state = dynamics(free_state, action_step, valid)
            teacher_state = dynamics(
                teacher_states[:, horizon_index], action_step, valid,
            )
            if not torch.isfinite(free_state).all() or not torch.isfinite(teacher_state).all():
                raise FloatingPointError(
                    f"Dynamics produced non-finite state at horizon {horizon_index + 1}."
                )

            free_difference = free_state - target_step
            teacher_difference = teacher_state - target_step
            free_normalized = (
                free_difference[:, valid_dimensions]
                / state_scale[valid_dimensions]
            )
            teacher_normalized = (
                teacher_difference[:, valid_dimensions]
                / state_scale[valid_dimensions]
            )
            step_metrics = {
                "free_nrmse": torch.sqrt(
                    torch.mean(torch.square(free_normalized), dim=1)
                ),
                "teacher_nrmse": torch.sqrt(
                    torch.mean(torch.square(teacher_normalized), dim=1)
                ),
                "free_rmse": torch.sqrt(
                    torch.mean(torch.square(free_difference), dim=1)
                ),
                "teacher_rmse": torch.sqrt(
                    torch.mean(torch.square(teacher_difference), dim=1)
                ),
                "free_cosine_distance": 1.0 - F.cosine_similarity(
                    free_state, target_step, dim=1,
                ),
                "teacher_cosine_distance": 1.0 - F.cosine_similarity(
                    teacher_state, target_step, dim=1,
                ),
            }
            for metric_name, metric_values in step_metrics.items():
                if not torch.isfinite(metric_values).all():
                    raise FloatingPointError(
                        f"Metric {metric_name} is non-finite at horizon "
                        f"{horizon_index + 1}."
                    )
                metrics[metric_name][:, horizon_index] = (
                    metric_values.detach().cpu().numpy()
                )
    return metrics


def evaluate_checkpoint(
    checkpoint_spec: CheckpointSpec,
    trajectories: Sequence[dict[str, Any]],
    sampling_plan: SamplingPlan,
    max_horizon: int,
    batch_size: int,
    state_scale: np.ndarray,
    valid_dimensions: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """评估一个训练 seed 的用户级平均误差曲线。

    Args:
        checkpoint_spec (CheckpointSpec): 已校验 checkpoint。
        trajectories (Sequence[dict[str, Any]]): 原始真实轨迹。
        sampling_plan (SamplingPlan): 全 seed 共用的抽样计划。
        max_horizon (int): 最大预测时域。
        batch_size (int): 前向计算 batch size。
        state_scale (np.ndarray): 状态逐维标准差。
        valid_dimensions (np.ndarray): 有效状态维度标记。
        device (torch.device): 模型计算设备。

    Returns:
        dict[str, np.ndarray]: 每项指标形状为 `(num_users,H)`。
    """

    dynamics = ChunkDynamics(
        state_dim=checkpoint_spec.state_dim,
        action_dim=checkpoint_spec.action_dim,
        chunk_size=1,
        hidden_dims=checkpoint_spec.hidden_dims,
    ).to(device)
    dynamics.load_state_dict(checkpoint_spec.payload["dynamics"], strict=True)
    dynamics.eval()

    num_users = len(sampling_plan.user_ids)
    metric_sums: dict[str, np.ndarray] = {}
    state_scale_tensor = torch.as_tensor(
        state_scale, dtype=torch.float32, device=device,
    )
    valid_dimensions_tensor = torch.as_tensor(
        valid_dimensions, dtype=torch.bool, device=device,
    )
    LOGGER.info(
        "评估 dynamics seed=%s checkpoint=%s",
        checkpoint_spec.model_seed,
        checkpoint_spec.path,
    )
    for user_indices, teacher_states, actions, targets in iter_window_batches(
        trajectories, sampling_plan, max_horizon, batch_size,
    ):
        batch_metrics = evaluate_dynamics_batch(
            dynamics=dynamics,
            teacher_states=torch.as_tensor(
                teacher_states, dtype=torch.float32, device=device,
            ),
            actions=torch.as_tensor(actions, dtype=torch.float32, device=device),
            targets=torch.as_tensor(targets, dtype=torch.float32, device=device),
            state_scale=state_scale_tensor,
            valid_dimensions=valid_dimensions_tensor,
        )
        for metric_name, metric_values in batch_metrics.items():
            if metric_name not in metric_sums:
                metric_sums[metric_name] = np.zeros(
                    (num_users, max_horizon), dtype=np.float64,
                )
            np.add.at(metric_sums[metric_name], user_indices, metric_values)

    user_counts = sampling_plan.starts_per_user.astype(np.float64)[:, None]
    return {
        metric_name: (metric_values / user_counts).astype(np.float32)
        for metric_name, metric_values in metric_sums.items()
    }


def paired_user_bootstrap(
    free_curves: np.ndarray,
    teacher_curves: np.ndarray,
    bootstrap_samples: int,
    bootstrap_seed: int,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> dict[str, np.ndarray]:
    """对先跨模型 seed 平均的用户曲线执行配对 bootstrap。

    Args:
        free_curves (np.ndarray): `(S,U,H)` Free-Running 用户曲线。
        teacher_curves (np.ndarray): `(S,U,H)` Teacher Forcing 用户曲线。
        bootstrap_samples (int): 用户级重采样次数。
        bootstrap_seed (int): bootstrap 随机种子。
        ci_level (float): 双侧置信水平，必须位于 `(0,1)`。

    Returns:
        dict[str, np.ndarray]: 均值、逐 horizon CI 和增长统计的 bootstrap
        样本。所有重采样对两个分支使用相同用户索引。

    Raises:
        ValueError: 当输入形状或 bootstrap 配置非法时抛出。
    """

    if free_curves.shape != teacher_curves.shape or free_curves.ndim != 3:
        raise ValueError("Bootstrap curves must share shape (S, U, H).")
    if bootstrap_samples <= 0 or not 0.0 < ci_level < 1.0:
        raise ValueError("Invalid bootstrap_samples or ci_level.")

    mean_over_seeds = np.stack(
        [free_curves.mean(axis=0), teacher_curves.mean(axis=0)], axis=0,
    ).astype(np.float32)
    num_users = mean_over_seeds.shape[1]
    max_horizon = mean_over_seeds.shape[2]
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_curves = np.empty(
        (bootstrap_samples, 2, max_horizon), dtype=np.float32,
    )
    for bootstrap_index in range(bootstrap_samples):
        sampled_users = rng.integers(0, num_users, size=num_users)
        bootstrap_curves[bootstrap_index] = mean_over_seeds[
            :, sampled_users, :
        ].mean(axis=1)

    alpha = (1.0 - ci_level) / 2.0
    quantiles = [100.0 * alpha, 100.0 * (1.0 - alpha)]
    point_mean = mean_over_seeds.mean(axis=1)
    ci_low, ci_high = np.percentile(
        bootstrap_curves, quantiles, axis=0,
    )
    horizons = np.arange(1, max_horizon + 1, dtype=np.float64)
    centered_horizons = horizons - horizons.mean()
    slope_denominator = np.sum(np.square(centered_horizons))
    if slope_denominator > 0.0:
        bootstrap_slopes = np.einsum(
            "bch,h->bc", bootstrap_curves, centered_horizons,
        ) / slope_denominator
    else:
        bootstrap_slopes = np.zeros(
            (bootstrap_samples, 2), dtype=np.float32,
        )
    bootstrap_gap_curves = bootstrap_curves[:, 0] - bootstrap_curves[:, 1]
    if slope_denominator > 0.0:
        bootstrap_gap_slopes = np.einsum(
            "bh,h->b", bootstrap_gap_curves, centered_horizons,
        ) / slope_denominator
    else:
        bootstrap_gap_slopes = np.zeros(
            bootstrap_samples, dtype=np.float32,
        )
    return {
        "mean": point_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_curves": bootstrap_curves,
        "bootstrap_slopes": bootstrap_slopes,
        "bootstrap_gap_curves": bootstrap_gap_curves,
        "bootstrap_gap_slopes": bootstrap_gap_slopes,
    }


def interval(values: np.ndarray, ci_level: float = DEFAULT_CI_LEVEL) -> list[float]:
    """计算一维 bootstrap 样本的双侧分位数区间。

    Args:
        values (np.ndarray): 一维有限数值数组。
        ci_level (float): 双侧置信水平。

    Returns:
        list[float]: `[下界, 上界]`。

    Raises:
        ValueError: 当输入不是非空一维数组时抛出。
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("interval expects a non-empty finite 1D array.")
    alpha = (1.0 - ci_level) / 2.0
    low, high = np.percentile(array, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return [float(low), float(high)]


def write_per_user_csv(
    output_path: Path,
    checkpoint_specs: Sequence[CheckpointSpec],
    sampling_plan: SamplingPlan,
    seed_results: Sequence[dict[str, np.ndarray]],
) -> Path:
    """保存可复算汇总统计的 seed-user-horizon 明细。

    Args:
        output_path (Path): CSV 输出路径。
        checkpoint_specs (Sequence[CheckpointSpec]): checkpoint 元数据。
        sampling_plan (SamplingPlan): 用户与窗口数量信息。
        seed_results (Sequence[dict[str, np.ndarray]]): 每个 seed 的用户曲线。

    Returns:
        Path: 已写入的 CSV 路径。
    """

    fieldnames = (
        "model_seed",
        "user_index",
        "user_id",
        "start_count",
        "horizon",
        "free_nrmse",
        "teacher_nrmse",
        "compounding_gap_nrmse",
        "free_rmse",
        "teacher_rmse",
        "free_cosine_distance",
        "teacher_cosine_distance",
    )
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for checkpoint_spec, result in zip(checkpoint_specs, seed_results):
            max_horizon = result["free_nrmse"].shape[1]
            for user_index, user_id in enumerate(sampling_plan.user_ids):
                for horizon_index in range(max_horizon):
                    free_nrmse = float(result["free_nrmse"][user_index, horizon_index])
                    teacher_nrmse = float(
                        result["teacher_nrmse"][user_index, horizon_index]
                    )
                    writer.writerow({
                        "model_seed": checkpoint_spec.model_seed,
                        "user_index": user_index,
                        "user_id": int(user_id),
                        "start_count": int(
                            sampling_plan.starts_per_user[user_index]
                        ),
                        "horizon": horizon_index + 1,
                        "free_nrmse": free_nrmse,
                        "teacher_nrmse": teacher_nrmse,
                        "compounding_gap_nrmse": free_nrmse - teacher_nrmse,
                        "free_rmse": float(
                            result["free_rmse"][user_index, horizon_index]
                        ),
                        "teacher_rmse": float(
                            result["teacher_rmse"][user_index, horizon_index]
                        ),
                        "free_cosine_distance": float(
                            result["free_cosine_distance"][user_index, horizon_index]
                        ),
                        "teacher_cosine_distance": float(
                            result["teacher_cosine_distance"][user_index, horizon_index]
                        ),
                    })
    return output_path


def write_aggregate_csv(
    output_path: Path,
    checkpoint_specs: Sequence[CheckpointSpec],
    seed_results: Sequence[dict[str, np.ndarray]],
    bootstrap_results: dict[str, dict[str, np.ndarray]],
) -> Path:
    """保存每个 seed 与跨 seed 主曲线的逐 horizon 汇总。

    Args:
        output_path (Path): CSV 输出路径。
        checkpoint_specs (Sequence[CheckpointSpec]): checkpoint 元数据。
        seed_results (Sequence[dict[str, np.ndarray]]): seed 级用户曲线。
        bootstrap_results (dict[str, dict[str, np.ndarray]]): RMSE 与 NRMSE
        的配对用户 bootstrap 结果。

    Returns:
        Path: 已写入的 CSV 路径。
    """

    fieldnames = (
        "scope",
        "model_seed",
        "horizon",
        "free_rmse",
        "free_rmse_ci_low",
        "free_rmse_ci_high",
        "teacher_rmse",
        "teacher_rmse_ci_low",
        "teacher_rmse_ci_high",
        "compounding_gap_rmse",
        "gap_rmse_ci_low",
        "gap_rmse_ci_high",
        "free_nrmse",
        "free_nrmse_ci_low",
        "free_nrmse_ci_high",
        "teacher_nrmse",
        "teacher_nrmse_ci_low",
        "teacher_nrmse_ci_high",
        "compounding_gap_nrmse",
        "gap_nrmse_ci_low",
        "gap_nrmse_ci_high",
    )
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for checkpoint_spec, result in zip(checkpoint_specs, seed_results):
            free_rmse = result["free_rmse"].mean(axis=0)
            teacher_rmse = result["teacher_rmse"].mean(axis=0)
            free_nrmse = result["free_nrmse"].mean(axis=0)
            teacher_nrmse = result["teacher_nrmse"].mean(axis=0)
            for horizon_index in range(len(free_rmse)):
                writer.writerow({
                    "scope": "per_seed",
                    "model_seed": checkpoint_spec.model_seed,
                    "horizon": horizon_index + 1,
                    "free_rmse": float(free_rmse[horizon_index]),
                    "free_rmse_ci_low": "",
                    "free_rmse_ci_high": "",
                    "teacher_rmse": float(teacher_rmse[horizon_index]),
                    "teacher_rmse_ci_low": "",
                    "teacher_rmse_ci_high": "",
                    "compounding_gap_rmse": float(
                        free_rmse[horizon_index] - teacher_rmse[horizon_index]
                    ),
                    "gap_rmse_ci_low": "",
                    "gap_rmse_ci_high": "",
                    "free_nrmse": float(free_nrmse[horizon_index]),
                    "free_nrmse_ci_low": "",
                    "free_nrmse_ci_high": "",
                    "teacher_nrmse": float(teacher_nrmse[horizon_index]),
                    "teacher_nrmse_ci_low": "",
                    "teacher_nrmse_ci_high": "",
                    "compounding_gap_nrmse": float(
                        free_nrmse[horizon_index] - teacher_nrmse[horizon_index]
                    ),
                    "gap_nrmse_ci_low": "",
                    "gap_nrmse_ci_high": "",
                })
        rmse_result = bootstrap_results["state_rmse"]
        nrmse_result = bootstrap_results["normalized_state_rmse"]
        rmse_gap_low, rmse_gap_high = np.percentile(
            rmse_result["bootstrap_gap_curves"], [2.5, 97.5], axis=0,
        )
        nrmse_gap_low, nrmse_gap_high = np.percentile(
            nrmse_result["bootstrap_gap_curves"], [2.5, 97.5], axis=0,
        )
        for horizon_index in range(rmse_result["mean"].shape[1]):
            writer.writerow({
                "scope": "all_seeds_user_bootstrap",
                "model_seed": "all",
                "horizon": horizon_index + 1,
                "free_rmse": float(rmse_result["mean"][0, horizon_index]),
                "free_rmse_ci_low": float(
                    rmse_result["ci_low"][0, horizon_index]
                ),
                "free_rmse_ci_high": float(
                    rmse_result["ci_high"][0, horizon_index]
                ),
                "teacher_rmse": float(rmse_result["mean"][1, horizon_index]),
                "teacher_rmse_ci_low": float(
                    rmse_result["ci_low"][1, horizon_index]
                ),
                "teacher_rmse_ci_high": float(
                    rmse_result["ci_high"][1, horizon_index]
                ),
                "compounding_gap_rmse": float(
                    rmse_result["mean"][0, horizon_index]
                    - rmse_result["mean"][1, horizon_index]
                ),
                "gap_rmse_ci_low": float(rmse_gap_low[horizon_index]),
                "gap_rmse_ci_high": float(rmse_gap_high[horizon_index]),
                "free_nrmse": float(nrmse_result["mean"][0, horizon_index]),
                "free_nrmse_ci_low": float(
                    nrmse_result["ci_low"][0, horizon_index]
                ),
                "free_nrmse_ci_high": float(
                    nrmse_result["ci_high"][0, horizon_index]
                ),
                "teacher_nrmse": float(nrmse_result["mean"][1, horizon_index]),
                "teacher_nrmse_ci_low": float(
                    nrmse_result["ci_low"][1, horizon_index]
                ),
                "teacher_nrmse_ci_high": float(
                    nrmse_result["ci_high"][1, horizon_index]
                ),
                "compounding_gap_nrmse": float(
                    nrmse_result["mean"][0, horizon_index]
                    - nrmse_result["mean"][1, horizon_index]
                ),
                "gap_nrmse_ci_low": float(nrmse_gap_low[horizon_index]),
                "gap_nrmse_ci_high": float(nrmse_gap_high[horizon_index]),
            })
    return output_path


def plot_error_curves(
    output_dir: Path,
    free_curves: np.ndarray,
    teacher_curves: np.ndarray,
    bootstrap_result: dict[str, np.ndarray],
) -> list[Path]:
    """绘制主误差曲线与 Free-Running 传播增量。

    Args:
        output_dir (Path): 图像输出目录。
        free_curves (np.ndarray): `(S,U,H)` Free-Running 状态 RMSE。
        teacher_curves (np.ndarray): `(S,U,H)` Teacher Forcing 状态 RMSE。
        bootstrap_result (dict[str, np.ndarray]): 用户级配对 bootstrap 结果。

    Returns:
        list[Path]: PNG 与 PDF 图像路径。
    """

    set_mpl_cache_to_tmp()
    import matplotlib.pyplot as plt

    horizons = np.arange(1, free_curves.shape[-1] + 1)
    mean_curves = bootstrap_result["mean"]
    ci_low = bootstrap_result["ci_low"]
    ci_high = bootstrap_result["ci_high"]
    gap_curve = mean_curves[0] - mean_curves[1]
    gap_low, gap_high = np.percentile(
        bootstrap_result["bootstrap_gap_curves"], [2.5, 97.5], axis=0,
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    for seed_index in range(free_curves.shape[0]):
        axes[0].plot(
            horizons,
            free_curves[seed_index].mean(axis=0),
            color="#D97706",
            alpha=0.18,
            linewidth=0.8,
        )
    axes[0].plot(
        horizons, mean_curves[0], color="#D97706", linewidth=2.2,
        label="Free-Running",
    )
    axes[0].fill_between(
        horizons, ci_low[0], ci_high[0], color="#F59E0B", alpha=0.22,
    )
    axes[0].plot(
        horizons, mean_curves[1], color="#2563EB", linewidth=2.2,
        label="Teacher Forcing",
    )
    axes[0].fill_between(
        horizons, ci_low[1], ci_high[1], color="#60A5FA", alpha=0.18,
    )
    axes[0].set_xlabel("Rollout horizon (environment steps)")
    axes[0].set_ylabel("State RMSE")
    axes[0].set_title("K=1 dynamics under fixed logged actions")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].axhline(0.0, color="#64748B", linestyle="--", linewidth=1.0)
    axes[1].plot(
        horizons, gap_curve, color="#7C3AED", linewidth=2.2,
        label="Free-Running - Teacher Forcing",
    )
    axes[1].fill_between(
        horizons, gap_low, gap_high, color="#A78BFA", alpha=0.24,
    )
    axes[1].set_xlabel("Rollout horizon (environment steps)")
    axes[1].set_ylabel("Compounding error gap (RMSE)")
    axes[1].set_title("Additional error from recursive self-feeding")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    png_path = output_dir / "compounding_error_curve.png"
    pdf_path = output_dir / "compounding_error_curve.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return [png_path, pdf_path]


def summarize_bootstrap_metric(
    bootstrap_result: dict[str, np.ndarray],
    max_horizon: int,
) -> dict[str, Any]:
    """汇总单个误差指标的 horizon、增长、斜率与统计判定。

    Args:
        bootstrap_result (dict[str, np.ndarray]): 配对用户 bootstrap 结果。
        max_horizon (int): 最大预测时域。

    Returns:
        dict[str, Any]: 可序列化的指标级统计结果。
    """

    mean_curves = bootstrap_result["mean"]
    bootstrap_curves = bootstrap_result["bootstrap_curves"]
    bootstrap_gap_curves = bootstrap_result["bootstrap_gap_curves"]
    free_growth_samples = bootstrap_curves[:, 0, -1] - bootstrap_curves[:, 0, 0]
    teacher_growth_samples = (
        bootstrap_curves[:, 1, -1] - bootstrap_curves[:, 1, 0]
    )
    gap_growth_samples = bootstrap_gap_curves[:, -1] - bootstrap_gap_curves[:, 0]
    free_slope_samples = bootstrap_result["bootstrap_slopes"][:, 0]
    teacher_slope_samples = bootstrap_result["bootstrap_slopes"][:, 1]
    gap_slope_samples = bootstrap_result["bootstrap_gap_slopes"]
    free_growth_ci = interval(free_growth_samples)
    gap_growth_ci = interval(gap_growth_samples)
    free_slope_ci = interval(free_slope_samples)
    gap_slope_ci = interval(gap_slope_samples)
    selected_horizons = sorted({
        horizon for horizon in (1, 5, 10, 20, 50, 100)
        if horizon <= max_horizon
    })
    horizon_metrics = {}
    for horizon in selected_horizons:
        index = horizon - 1
        gap_samples = bootstrap_gap_curves[:, index]
        horizon_metrics[str(horizon)] = {
            "free": float(mean_curves[0, index]),
            "free_ci": interval(bootstrap_curves[:, 0, index]),
            "teacher": float(mean_curves[1, index]),
            "teacher_ci": interval(bootstrap_curves[:, 1, index]),
            "compounding_gap": float(
                mean_curves[0, index] - mean_curves[1, index]
            ),
            "compounding_gap_ci": interval(gap_samples),
        }

    return {
        "horizon_metrics": horizon_metrics,
        "growth_from_h1_to_hmax": {
            "free": float(mean_curves[0, -1] - mean_curves[0, 0]),
            "free_ci": free_growth_ci,
            "teacher": float(mean_curves[1, -1] - mean_curves[1, 0]),
            "teacher_ci": interval(teacher_growth_samples),
            "compounding_gap": float(
                (mean_curves[0, -1] - mean_curves[1, -1])
                - (mean_curves[0, 0] - mean_curves[1, 0])
            ),
            "compounding_gap_ci": gap_growth_ci,
        },
        "linear_slope_per_step": {
            "free": float(np.mean(free_slope_samples)),
            "free_ci": free_slope_ci,
            "teacher": float(np.mean(teacher_slope_samples)),
            "teacher_ci": interval(teacher_slope_samples),
            "compounding_gap": float(np.mean(gap_slope_samples)),
            "compounding_gap_ci": gap_slope_ci,
        },
        "diagnostic_flags": {
            "free_error_growth_ci_above_zero": bool(free_growth_ci[0] > 0.0),
            "free_slope_ci_above_zero": bool(free_slope_ci[0] > 0.0),
            "gap_growth_ci_above_zero": bool(gap_growth_ci[0] > 0.0),
            "gap_slope_ci_above_zero": bool(gap_slope_ci[0] > 0.0),
        },
    }


def build_summary(
    checkpoint_specs: Sequence[CheckpointSpec],
    sampling_plan: SamplingPlan,
    bootstrap_results: dict[str, dict[str, np.ndarray]],
    max_horizon: int,
    state_scale: np.ndarray,
    valid_dimensions: np.ndarray,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """构造机器可读的主结果、敏感性分析与判定信息。

    Args:
        checkpoint_specs (Sequence[CheckpointSpec]): 多 seed checkpoint。
        sampling_plan (SamplingPlan): 用户与窗口抽样信息。
        bootstrap_results (dict[str, dict[str, np.ndarray]]): 原始 RMSE 与
        标准化 NRMSE 的用户级 bootstrap 结果。
        max_horizon (int): 最大预测时域。
        state_scale (np.ndarray): 状态逐维标准差。
        valid_dimensions (np.ndarray): 有效归一化维度。
        bootstrap_samples (int): bootstrap 次数。

    Returns:
        dict[str, Any]: 可序列化实验汇总。
    """

    metric_summaries = {
        metric_name: summarize_bootstrap_metric(result, max_horizon)
        for metric_name, result in bootstrap_results.items()
    }
    primary_flags = metric_summaries["state_rmse"]["diagnostic_flags"]
    motivation_supported = bool(
        primary_flags["free_error_growth_ci_above_zero"]
        and primary_flags["gap_growth_ci_above_zero"]
    )
    valid_scales = state_scale[valid_dimensions]
    return {
        "status": "completed",
        "experiment": "k1_fixed_action_free_running_compounding_error",
        "primary_metric": (
            "state_rmse; matches the raw-state MSE objective used to train "
            "ChunkDynamics"
        ),
        "sensitivity_metric": "per_dimension_std_normalized_state_rmse",
        "checkpoint_seeds": [spec.model_seed for spec in checkpoint_specs],
        "num_model_seeds": len(checkpoint_specs),
        "num_users": int(len(sampling_plan.user_ids)),
        "num_windows": int(len(sampling_plan.starts)),
        "starts_per_user_min": int(sampling_plan.starts_per_user.min()),
        "starts_per_user_max": int(sampling_plan.starts_per_user.max()),
        "max_horizon": int(max_horizon),
        "bootstrap_samples": int(bootstrap_samples),
        "normalization": {
            "type": "per_dimension_std_over_sampled_true_target_states",
            "state_dim": int(len(state_scale)),
            "valid_state_dims": int(valid_dimensions.sum()),
            "scale_eps": DEFAULT_SCALE_EPS,
            "valid_scale_min": float(valid_scales.min()),
            "valid_scale_median": float(np.median(valid_scales)),
            "valid_scale_max": float(valid_scales.max()),
        },
        "data_scope": (
            "in_distribution_diagnostic; current checkpoints were pretrained "
            "with max_trajectories=None on this trajectory file"
        ),
        "metrics": metric_summaries,
        "diagnostic_flags": primary_flags,
        "motivation_supported_by_primary_metric": motivation_supported,
        "interpretation": (
            "Support requires both Free-Running h_max minus h_1 growth and the "
            "paired Free-Running minus Teacher-Forcing gap growth to have a "
            "95% bootstrap confidence interval above zero."
        ),
    }


def main() -> None:
    """运行多 seed K=1 累计状态误差动机实验。

    Raises:
        ValueError: 当输入配置、checkpoint 或轨迹维度不一致时抛出。
        FileNotFoundError: 当所需 checkpoint 或数据文件缺失时抛出。
    """

    configure_logging()
    args = build_parser().parse_args()
    validate_args(args)
    checkpoint_root = Path(args.checkpoint_root)
    dataset_path = Path(args.dataset_path)
    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device)

    checkpoint_specs = load_checkpoint_specs(checkpoint_root, args.seeds)
    trajectories = load_trajectories(dataset_path)
    state_dim = checkpoint_specs[0].state_dim
    action_dim = checkpoint_specs[0].action_dim
    first_observations = np.asarray(trajectories[0]["observations"])
    first_actions = np.asarray(trajectories[0]["actions"])
    if first_observations.shape[1] != state_dim or first_actions.shape[1] != action_dim:
        raise ValueError(
            "Dataset dimensions do not match checkpoints: "
            f"dataset={(first_observations.shape[1], first_actions.shape[1])}, "
            f"checkpoint={(state_dim, action_dim)}."
        )

    sampling_plan = build_sampling_plan(
        trajectories=trajectories,
        max_horizon=args.max_horizon,
        starts_per_user=args.starts_per_user,
        min_start=args.min_start,
        sampling_seed=args.sampling_seed,
        max_users=args.max_users,
    )
    LOGGER.info(
        "抽样完成：users=%s windows=%s horizon=%s",
        len(sampling_plan.user_ids),
        len(sampling_plan.starts),
        args.max_horizon,
    )
    state_scale, valid_dimensions = compute_state_scale(
        trajectories=trajectories,
        sampling_plan=sampling_plan,
        max_horizon=args.max_horizon,
        batch_size=args.batch_size,
    )
    LOGGER.info(
        "状态归一化：有效维度=%s/%s",
        int(valid_dimensions.sum()),
        len(valid_dimensions),
    )

    seed_results = [
        evaluate_checkpoint(
            checkpoint_spec=checkpoint_spec,
            trajectories=trajectories,
            sampling_plan=sampling_plan,
            max_horizon=args.max_horizon,
            batch_size=args.batch_size,
            state_scale=state_scale,
            valid_dimensions=valid_dimensions,
            device=device,
        )
        for checkpoint_spec in checkpoint_specs
    ]
    metric_keys = {
        "state_rmse": ("free_rmse", "teacher_rmse"),
        "normalized_state_rmse": ("free_nrmse", "teacher_nrmse"),
    }
    metric_curves = {
        metric_name: (
            np.stack([result[free_key] for result in seed_results], axis=0),
            np.stack([result[teacher_key] for result in seed_results], axis=0),
        )
        for metric_name, (free_key, teacher_key) in metric_keys.items()
    }
    bootstrap_results = {
        metric_name: paired_user_bootstrap(
            free_curves=curves[0],
            teacher_curves=curves[1],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for metric_name, curves in metric_curves.items()
    }

    per_user_path = write_per_user_csv(
        output_dir / "per_seed_user_horizon_metrics.csv",
        checkpoint_specs,
        sampling_plan,
        seed_results,
    )
    aggregate_path = write_aggregate_csv(
        output_dir / "aggregate_horizon_metrics.csv",
        checkpoint_specs,
        seed_results,
        bootstrap_results,
    )
    free_rmse_curves, teacher_rmse_curves = metric_curves["state_rmse"]
    figure_paths = plot_error_curves(
        output_dir=output_dir,
        free_curves=free_rmse_curves,
        teacher_curves=teacher_rmse_curves,
        bootstrap_result=bootstrap_results["state_rmse"],
    )
    summary = build_summary(
        checkpoint_specs=checkpoint_specs,
        sampling_plan=sampling_plan,
        bootstrap_results=bootstrap_results,
        max_horizon=args.max_horizon,
        state_scale=state_scale,
        valid_dimensions=valid_dimensions,
        bootstrap_samples=args.bootstrap_samples,
    )
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    resolved_config = {
        **vars(args),
        "device_resolved": str(device),
        "checkpoint_paths": [str(spec.path) for spec in checkpoint_specs],
        "dataset_path_resolved": str(dataset_path.resolve()),
        "output_dir_resolved": str(output_dir.resolve()),
    }
    config_path = output_dir / "resolved_config.json"
    with config_path.open("w", encoding="utf-8") as file_obj:
        json.dump(resolved_config, file_obj, ensure_ascii=False, indent=2)

    manifest = {
        "status": "completed",
        "summary": str(summary_path),
        "aggregate_metrics": str(aggregate_path),
        "per_user_metrics": str(per_user_path),
        "figures": [str(path) for path in figure_paths],
        "resolved_config": str(config_path),
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=False, indent=2)

    flags = summary["diagnostic_flags"]
    LOGGER.info("实验完成：summary=%s", summary_path)
    LOGGER.info("累计误差诊断标记：%s", flags)


if __name__ == "__main__":
    main()
