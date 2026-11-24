"""将快手交互日志转换为用户级离线强化学习轨迹。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

LOGGER = logging.getLogger(__name__)


DEFAULT_WINDOW_SIZE = 3
"""StateTrackerAvg 默认使用的历史窗口长度。"""

DEFAULT_PADDING_STD = 0.01
"""StateTrackerAvg 为不存在物品追加 padding embedding 时使用的标准差。"""

BASE_INPUT_COLUMNS = ("user_id", "item_id")
"""所有快手用户级轨迹共同需要的 ID 字段。"""

REQUIRED_TRAJECTORY_KEYS = (
    "actions",
    "terminals",
    "rewards",
    "observations",
    "next_observations",
    "user_id",
)
"""输出轨迹必须包含的字段。"""


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """计算文件的 SHA-256 摘要。

    Args:
        file_path (Path): 待读取的文件路径，必须存在且为普通文件。
        chunk_size (int): 分块读取大小，单位为字节，必须大于 0。

    Returns:
        str: 小写十六进制 SHA-256 摘要。

    Raises:
        FileNotFoundError: 输入文件不存在时抛出。
        ValueError: `chunk_size` 不为正整数时抛出。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0。")
    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        while True:
            block = file_obj.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_item_embeddings(embedding_path: Path) -> np.ndarray:
    """加载由 DeepFM user model 保存的 item embedding。

    Args:
        embedding_path (Path): `*_emb_item_M*.pt` 或
            `*_emb_item_val_M*.pt` 文件路径。文件内容应为二维 PyTorch
            tensor，行下标必须与输入日志的 `item_id` 一致。

    Returns:
        np.ndarray: `float32` item embedding 矩阵，形状为
        `(num_items, action_dim)`。

    Raises:
        FileNotFoundError: embedding 文件不存在时抛出。
        TypeError: 文件内容不是 PyTorch tensor 时抛出。
        ValueError: tensor 不是二维、为空或包含非有限值时抛出。
    """

    if not embedding_path.is_file():
        raise FileNotFoundError(f"Item embedding 文件不存在：{embedding_path}")

    embedding_tensor = torch.load(
        embedding_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(embedding_tensor, torch.Tensor):
        raise TypeError(
            "Item embedding 文件必须保存 PyTorch tensor，"
            f"实际类型为 {type(embedding_tensor)!r}。"
        )

    embeddings = (
        embedding_tensor.detach().cpu().numpy().astype(np.float32, copy=True)
    )
    if embeddings.ndim != 2:
        raise ValueError(
            f"Item embedding 必须为二维矩阵，实际 shape={embeddings.shape}。"
        )
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError(f"Item embedding 不能为空，实际 shape={embeddings.shape}。")
    if not np.isfinite(embeddings).all():
        raise ValueError("Item embedding 包含 NaN 或 Inf。")
    return embeddings


def load_interactions(
    input_path: Path,
    time_column: str,
    reward_column: str,
    max_users: Optional[int] = None,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """读取、校验并按用户时间顺序排列快手交互日志。

    Args:
        input_path (Path): 输入 CSV 路径。
        time_column (str): 用户内排序使用的时间字段名。
        reward_column (str): 写入轨迹 `rewards` 的反馈字段名。
        max_users (Optional[int]): 可选的用户数量上限。设置后按升序保留前
            `max_users` 个用户，主要用于 smoke test；必须大于 0。
        dataset_name (str): 日志和异常信息中使用的数据集名称。

    Returns:
        pd.DataFrame: 仅包含 ID、时间和 reward 字段的有序数据表。

    Raises:
        FileNotFoundError: 输入 CSV 不存在时抛出。
        ValueError: 数据为空、缺列、存在缺失值、ID 非整数、reward 非有限，
            或 `max_users` 非法时抛出。
    """

    if not input_path.is_file():
        raise FileNotFoundError(f"{dataset_name} 输入文件不存在：{input_path}")
    if not time_column:
        raise ValueError("time_column 不能为空。")
    if not reward_column:
        raise ValueError("reward_column 不能为空。")
    if max_users is not None and max_users <= 0:
        raise ValueError("max_users 必须大于 0。")

    required_columns = (*BASE_INPUT_COLUMNS, time_column, reward_column)
    try:
        interactions = pd.read_csv(
            input_path,
            usecols=list(required_columns),
        )
    except ValueError as exc:
        raise ValueError(
            f"输入文件缺少必要字段 {required_columns}：{input_path}"
        ) from exc

    if interactions.empty:
        raise ValueError(f"输入数据为空：{input_path}")
    if interactions.isna().any().any():
        missing_counts = interactions.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"输入数据包含缺失值：{missing_counts}")

    for id_column in ("user_id", "item_id"):
        numeric_values = pd.to_numeric(interactions[id_column], errors="raise")
        integer_values = numeric_values.astype(np.int64)
        if not np.array_equal(
            numeric_values.to_numpy(),
            integer_values.to_numpy(),
        ):
            raise ValueError(f"{id_column} 必须只包含整数。")
        if (integer_values < 0).any():
            raise ValueError(f"{id_column} 不允许包含负数。")
        interactions[id_column] = integer_values

    timestamps = pd.to_numeric(interactions[time_column], errors="raise")
    if not np.isfinite(timestamps.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{time_column} 包含 NaN 或 Inf。")
    interactions[time_column] = timestamps

    rewards = pd.to_numeric(interactions[reward_column], errors="raise")
    if not np.isfinite(rewards.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{reward_column} 包含 NaN 或 Inf。")
    interactions[reward_column] = rewards.astype(np.float32)

    interactions["_source_order"] = np.arange(len(interactions), dtype=np.int64)
    interactions.sort_values(
        ["user_id", time_column, "_source_order"],
        kind="mergesort",
        inplace=True,
    )

    if max_users is not None:
        selected_users = interactions["user_id"].drop_duplicates().iloc[:max_users]
        interactions = interactions[
            interactions["user_id"].isin(selected_users)
        ].copy()

    interactions.drop(columns="_source_order", inplace=True)
    interactions.reset_index(drop=True, inplace=True)
    return interactions


def load_kuairand_interactions(
    input_path: Path,
    max_users: Optional[int] = None,
) -> pd.DataFrame:
    """读取并整理 KuaiRand train/test 随机曝光日志。

    Args:
        input_path (Path): `train_processed.csv` 或 `test_processed.csv` 路径。
        max_users (Optional[int]): 可选用户数量上限，必须大于 0。

    Returns:
        pd.DataFrame: 按 `user_id`、`time_ms` 稳定排序的交互表。

    Raises:
        FileNotFoundError: 输入文件不存在时抛出。
        ValueError: 输入字段或数值不合法时抛出。
    """

    return load_interactions(
        input_path=input_path,
        time_column="time_ms",
        reward_column="is_click",
        max_users=max_users,
        dataset_name="KuaiRand",
    )


def create_padding_state(
    action_dim: int,
    seed: int,
    padding_std: float = DEFAULT_PADDING_STD,
) -> np.ndarray:
    """创建与 StateTrackerAvg 首步语义一致的 padding 状态。

    StateTrackerAvg 会为 `item_id=-1` 追加随机 item embedding，并在
    `reward_handle="cat"` 时拼接零 reward。本函数复现该表示。

    Args:
        action_dim (int): item embedding 维度，必须大于 0。
        seed (int): PyTorch 随机种子。
        padding_std (float): padding item embedding 的正态分布标准差，
            必须大于等于 0。

    Returns:
        np.ndarray: `float32` 初始状态，形状为 `(action_dim + 1,)`。

    Raises:
        ValueError: `action_dim` 或 `padding_std` 非法时抛出。
    """

    if action_dim <= 0:
        raise ValueError("action_dim 必须大于 0。")
    if padding_std < 0:
        raise ValueError("padding_std 不允许小于 0。")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    padding_embedding = torch.normal(
        mean=0.0,
        std=padding_std,
        size=(action_dim,),
        generator=generator,
        dtype=torch.float32,
    ).numpy()
    return np.concatenate(
        [padding_embedding, np.zeros(1, dtype=np.float32)],
        axis=0,
    )


def compute_avg_states(
    actions: np.ndarray,
    rewards: np.ndarray,
    padding_state: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """按 StateTrackerAvg 语义计算状态和下一状态。

    对每一步，将 item embedding 与当前 reward 拼接为交互向量。状态是当前
    动作执行前最近 `window_size` 个历史向量的均值，下一状态则额外纳入当前
    交互。首个历史向量为 `padding_state`。

    Args:
        actions (np.ndarray): 动作 embedding，形状为 `(T, action_dim)`。
        rewards (np.ndarray): 原始 reward，形状为 `(T,)`。
        padding_state (np.ndarray): 初始 padding 状态，形状为
            `(action_dim + 1,)`。
        window_size (int): 历史窗口长度，必须大于 0。

    Returns:
        Tuple[np.ndarray, np.ndarray]: `observations` 与
        `next_observations`，二者形状均为 `(T, action_dim + 1)`，
        dtype 为 `float32`。

    Raises:
        ValueError: 输入维度、长度、窗口或数值合法性不满足要求时抛出。

    Example:
        >>> actions = np.asarray([[1.0], [3.0]], dtype=np.float32)
        >>> rewards = np.asarray([0.0, 1.0], dtype=np.float32)
        >>> padding = np.zeros(2, dtype=np.float32)
        >>> obs, next_obs = compute_avg_states(actions, rewards, padding, 2)
        >>> obs.tolist()
        [[0.0, 0.0], [0.5, 0.0]]
        >>> next_obs.tolist()
        [[0.5, 0.0], [2.0, 0.5]]
    """

    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    padding_state = np.asarray(padding_state, dtype=np.float32)

    if window_size <= 0:
        raise ValueError("window_size 必须大于 0。")
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError(
            f"actions 必须是非空二维矩阵，实际 shape={actions.shape}。"
        )
    if rewards.ndim != 1 or rewards.shape[0] != actions.shape[0]:
        raise ValueError(
            "rewards 必须是一维数组且长度与 actions 相同，"
            f"实际 actions={actions.shape}, rewards={rewards.shape}。"
        )
    expected_state_dim = actions.shape[1] + 1
    if padding_state.shape != (expected_state_dim,):
        raise ValueError(
            "padding_state 维度不匹配，"
            f"期望 {(expected_state_dim,)}, 实际 {padding_state.shape}。"
        )
    if not (
        np.isfinite(actions).all()
        and np.isfinite(rewards).all()
        and np.isfinite(padding_state).all()
    ):
        raise ValueError("状态构造输入包含 NaN 或 Inf。")

    interaction_vectors = np.concatenate(
        [actions, rewards[:, np.newaxis]],
        axis=1,
    )
    history_vectors = np.concatenate(
        [padding_state[np.newaxis, :], interaction_vectors],
        axis=0,
    )

    # 前缀和首行补零，使任意历史窗口均可通过两次索引相减获得。
    prefix_sums = np.concatenate(
        [
            np.zeros((1, expected_state_dim), dtype=np.float64),
            np.cumsum(history_vectors, axis=0, dtype=np.float64),
        ],
        axis=0,
    )
    transition_count = actions.shape[0]
    observation_ends = np.arange(1, transition_count + 1, dtype=np.int64)
    next_observation_ends = observation_ends + 1

    def rolling_mean(end_indices: np.ndarray) -> np.ndarray:
        """计算以给定索引为右开端点的窗口均值。

        Args:
            end_indices (np.ndarray): 每个窗口的右开端点，一维整数数组。

        Returns:
            np.ndarray: 每个端点对应的历史窗口均值，形状为
            `(len(end_indices), state_dim)`。
        """

        start_indices = np.maximum(0, end_indices - window_size)
        window_sums = prefix_sums[end_indices] - prefix_sums[start_indices]
        window_lengths = (end_indices - start_indices)[:, np.newaxis]
        return (window_sums / window_lengths).astype(np.float32)

    observations = rolling_mean(observation_ends)
    next_observations = rolling_mean(next_observation_ends)
    return observations, next_observations


def build_user_trajectory(
    user_id: int,
    item_ids: np.ndarray,
    rewards: np.ndarray,
    item_embeddings: np.ndarray,
    padding_state: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> Dict[str, Any]:
    """构造单个用户的一条完整离线强化学习轨迹。

    Args:
        user_id (int): 用户 ID，必须为非负整数。
        item_ids (np.ndarray): 时间有序的 item ID，一维非空整数数组。
        rewards (np.ndarray): 与 item 一一对应的原始 reward。
        item_embeddings (np.ndarray): 完整 item embedding 表。
        padding_state (np.ndarray): StateTrackerAvg 初始 padding 状态。
        window_size (int): StateTrackerAvg 历史窗口长度。

    Returns:
        Dict[str, Any]: 包含 `actions`、`terminals`、`rewards`、
        `observations`、`next_observations`、`user_id` 的轨迹。

    Raises:
        ValueError: 用户、item ID 或数组长度非法时抛出。
        IndexError: item ID 超出 embedding 表范围时抛出。
    """

    if int(user_id) < 0:
        raise ValueError("user_id 不允许为负数。")

    item_ids = np.asarray(item_ids)
    rewards = np.asarray(rewards, dtype=np.float32)
    if item_ids.ndim != 1 or item_ids.shape[0] == 0:
        raise ValueError("item_ids 必须是一维非空数组。")
    if rewards.shape != item_ids.shape:
        raise ValueError(
            f"rewards 与 item_ids shape 不一致：{rewards.shape} != {item_ids.shape}。"
        )
    integer_item_ids = item_ids.astype(np.int64)
    if not np.array_equal(item_ids, integer_item_ids):
        raise ValueError("item_ids 必须只包含整数。")
    if integer_item_ids.min() < 0 or integer_item_ids.max() >= len(item_embeddings):
        raise IndexError(
            "item_id 超出 embedding 表范围："
            f"min={integer_item_ids.min()}, max={integer_item_ids.max()}, "
            f"num_embeddings={len(item_embeddings)}。"
        )

    actions = item_embeddings[integer_item_ids].astype(np.float32, copy=True)
    observations, next_observations = compute_avg_states(
        actions=actions,
        rewards=rewards,
        padding_state=padding_state,
        window_size=window_size,
    )
    terminals = np.zeros(len(integer_item_ids), dtype=np.bool_)
    terminals[-1] = True

    return {
        "actions": actions,
        "terminals": terminals,
        "rewards": rewards.astype(np.float32, copy=True),
        "observations": observations,
        "next_observations": next_observations,
        "user_id": int(user_id),
    }


def build_trajectories(
    interactions: pd.DataFrame,
    item_embeddings: np.ndarray,
    reward_column: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    seed: int = 2023,
    padding_std: float = DEFAULT_PADDING_STD,
) -> List[Dict[str, Any]]:
    """将快手交互表转换为按用户组织的轨迹列表。

    Args:
        interactions (pd.DataFrame): 已按用户与时间排序的交互表，至少包含
            `user_id`、`item_id` 和指定 reward 字段。
        item_embeddings (np.ndarray): 完整 item embedding 表。
        reward_column (str): 作为轨迹 reward 的交互表字段名。
        window_size (int): StateTrackerAvg 历史窗口长度。
        seed (int): padding embedding 随机种子。
        padding_std (float): padding embedding 正态分布标准差。

    Returns:
        List[Dict[str, Any]]: 每个元素对应一个用户的完整轨迹，按 user ID
        升序排列。

    Raises:
        ValueError: 输入数据为空、缺列或 embedding 非法时抛出。
    """

    if not reward_column:
        raise ValueError("reward_column 不能为空。")
    required_columns = {"user_id", "item_id", reward_column}
    missing_columns = sorted(required_columns.difference(interactions.columns))
    if missing_columns:
        raise ValueError(f"交互表缺少字段：{missing_columns}")
    if interactions.empty:
        raise ValueError("交互表不能为空。")

    item_embeddings = np.asarray(item_embeddings, dtype=np.float32)
    if item_embeddings.ndim != 2 or item_embeddings.size == 0:
        raise ValueError(
            f"item_embeddings 必须是非空二维矩阵，实际 {item_embeddings.shape}。"
        )

    padding_state = create_padding_state(
        action_dim=item_embeddings.shape[1],
        seed=seed,
        padding_std=padding_std,
    )
    trajectories: List[Dict[str, Any]] = []
    grouped = interactions.groupby("user_id", sort=True, observed=True)
    for user_id, user_frame in grouped:
        trajectory = build_user_trajectory(
            user_id=int(user_id),
            item_ids=user_frame["item_id"].to_numpy(),
            rewards=user_frame[reward_column].to_numpy(dtype=np.float32),
            item_embeddings=item_embeddings,
            padding_state=padding_state,
            window_size=window_size,
        )
        trajectories.append(trajectory)

    if not trajectories:
        raise ValueError("未构造出任何用户轨迹。")
    return trajectories


def build_kuairand_trajectories(
    interactions: pd.DataFrame,
    item_embeddings: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    seed: int = 2023,
    padding_std: float = DEFAULT_PADDING_STD,
) -> List[Dict[str, Any]]:
    """将 KuaiRand 交互表转换为按用户组织的轨迹列表。

    Args:
        interactions (pd.DataFrame): 已排序且包含 `is_click` 的交互表。
        item_embeddings (np.ndarray): KuaiRand 统一 item embedding 表。
        window_size (int): StateTrackerAvg 历史窗口长度。
        seed (int): padding embedding 随机种子。
        padding_std (float): padding embedding 正态分布标准差。

    Returns:
        List[Dict[str, Any]]: 按用户 ID 升序排列的轨迹列表。

    Raises:
        ValueError: 输入字段、embedding 或参数不合法时抛出。
        IndexError: item ID 超出 embedding 表范围时抛出。
    """

    return build_trajectories(
        interactions=interactions,
        item_embeddings=item_embeddings,
        reward_column="is_click",
        window_size=window_size,
        seed=seed,
        padding_std=padding_std,
    )


def validate_trajectories(
    trajectories: Sequence[Dict[str, Any]],
    expected_action_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """校验轨迹字段、shape、时序对齐和 terminal 语义。

    Args:
        trajectories (Sequence[Dict[str, Any]]): 待校验的用户轨迹。
        expected_action_dim (Optional[int]): 可选的期望动作维度。

    Returns:
        Dict[str, Any]: 包含轨迹数、transition 数、维度和长度统计的摘要。

    Raises:
        ValueError: 轨迹为空、字段缺失、shape 不一致、含非有限值、用户重复、
            时序错位或 terminal 设置错误时抛出。
    """

    if not trajectories:
        raise ValueError("轨迹列表不能为空。")

    user_ids = set()
    lengths = []
    action_dim = None
    state_dim = None
    for trajectory_index, trajectory in enumerate(trajectories):
        missing_keys = sorted(
            set(REQUIRED_TRAJECTORY_KEYS).difference(trajectory.keys())
        )
        if missing_keys:
            raise ValueError(
                f"Trajectory #{trajectory_index} 缺少字段：{missing_keys}"
            )

        user_id = int(trajectory["user_id"])
        if user_id in user_ids:
            raise ValueError(f"user_id={user_id} 出现重复轨迹。")
        user_ids.add(user_id)

        actions = np.asarray(trajectory["actions"])
        terminals = np.asarray(trajectory["terminals"])
        rewards = np.asarray(trajectory["rewards"])
        observations = np.asarray(trajectory["observations"])
        next_observations = np.asarray(trajectory["next_observations"])

        if actions.ndim != 2 or observations.ndim != 2:
            raise ValueError(
                f"Trajectory #{trajectory_index} 的 actions/observations 必须为二维。"
            )
        trajectory_length = actions.shape[0]
        if trajectory_length == 0:
            raise ValueError(f"Trajectory #{trajectory_index} 为空。")
        if (
            observations.shape[0] != trajectory_length
            or next_observations.shape != observations.shape
            or rewards.shape != (trajectory_length,)
            or terminals.shape != (trajectory_length,)
        ):
            raise ValueError(
                f"Trajectory #{trajectory_index} 字段长度或 shape 不一致。"
            )
        if observations.shape[1] != actions.shape[1] + 1:
            raise ValueError(
                f"Trajectory #{trajectory_index} state_dim 必须等于 action_dim + 1。"
            )
        if not terminals[-1] or terminals[:-1].any():
            raise ValueError(
                f"Trajectory #{trajectory_index} 必须仅在最后一步 terminal=True。"
            )
        if trajectory_length > 1 and not np.allclose(
            next_observations[:-1],
            observations[1:],
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(
                f"Trajectory #{trajectory_index} 的 next_observation 时序未对齐。"
            )
        if not (
            np.isfinite(actions).all()
            and np.isfinite(rewards).all()
            and np.isfinite(observations).all()
            and np.isfinite(next_observations).all()
        ):
            raise ValueError(f"Trajectory #{trajectory_index} 包含 NaN 或 Inf。")

        if action_dim is None:
            action_dim = int(actions.shape[1])
            state_dim = int(observations.shape[1])
        elif actions.shape[1] != action_dim or observations.shape[1] != state_dim:
            raise ValueError("不同轨迹的 action/state 维度不一致。")
        lengths.append(trajectory_length)

    if expected_action_dim is not None and action_dim != expected_action_dim:
        raise ValueError(
            f"action_dim={action_dim}，与期望值 {expected_action_dim} 不一致。"
        )

    length_array = np.asarray(lengths, dtype=np.int64)
    return {
        "num_trajectories": int(len(trajectories)),
        "num_transitions": int(length_array.sum()),
        "action_dim": int(action_dim),
        "state_dim": int(state_dim),
        "min_trajectory_length": int(length_array.min()),
        "max_trajectory_length": int(length_array.max()),
        "mean_trajectory_length": float(length_array.mean()),
    }


def save_trajectory_dataset(
    trajectories: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    """原子保存轨迹 pickle，避免中断后留下不完整目标文件。

    Args:
        trajectories (Sequence[Dict[str, Any]]): 已校验的轨迹列表。
        output_path (Path): 目标 `.pkl` 路径。

    Returns:
        None

    Raises:
        ValueError: 输出扩展名不是 `.pkl` 时抛出。
        OSError: 创建目录、写入或原子替换失败时抛出。
    """

    if output_path.suffix.lower() != ".pkl":
        raise ValueError(f"输出文件必须使用 .pkl 扩展名：{output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("wb") as file_obj:
            pickle.dump(
                list(trajectories),
                file_obj,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_summary(summary: Dict[str, Any], summary_path: Path) -> None:
    """保存 JSON 格式的数据集构造摘要。

    Args:
        summary (Dict[str, Any]): 可 JSON 序列化的构造参数与统计信息。
        summary_path (Path): JSON 输出路径。

    Returns:
        None

    Raises:
        OSError: 创建目录或写入文件失败时抛出。
        TypeError: `summary` 无法 JSON 序列化时抛出。
    """

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")


def resolve_summary_path(
    output_path: Path,
    summary_path: Optional[Path],
) -> Path:
    """解析轨迹构造摘要的输出路径。

    Args:
        output_path (Path): 轨迹 pickle 输出路径。
        summary_path (Optional[Path]): 用户显式指定的摘要路径；为 `None`
            时在 pickle 文件名后追加 `.summary.json`。

    Returns:
        Path: 最终摘要 JSON 路径。

    Raises:
        ValueError: 摘要路径与 pickle 输出路径相同时抛出。
    """

    resolved_path = (
        summary_path
        if summary_path is not None
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )
    if resolved_path.resolve() == output_path.resolve():
        raise ValueError("summary_path 不能与 output_path 相同。")
    return resolved_path


def build_and_save_trajectory_dataset(
    *,
    dataset_name: str,
    split: str,
    input_path: Path,
    embedding_path: Path,
    output_path: Path,
    summary_path: Optional[Path],
    time_column: str,
    reward_column: str,
    window_size: int,
    seed: int,
    padding_std: float,
    max_users: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """构造、校验并保存用户级离线强化学习轨迹。

    该函数统一 KuaiRec 与 KuaiRand 的构造语义：交互按用户和时间稳定
    排序，动作由 item embedding 查表得到，状态由 StateTrackerAvg 的
    固定窗口平均得到，每位用户仅在最后一步终止。

    Args:
        dataset_name (str): 数据集名称，用于日志与摘要。
        split (str): 数据划分名称，例如 `train` 或 `test`。
        input_path (Path): 原始交互 CSV 路径。
        embedding_path (Path): 与 CSV item ID 空间一致的 embedding 路径。
        output_path (Path): 目标轨迹 pickle 路径。
        summary_path (Optional[Path]): 可选摘要 JSON 路径。
        time_column (str): 用户内时间排序字段。
        reward_column (str): 作为轨迹 reward 的字段。
        window_size (int): StateTrackerAvg 历史窗口长度，必须大于 0。
        seed (int): padding embedding 随机种子。
        padding_std (float): padding embedding 标准差，必须非负。
        max_users (Optional[int]): 可选用户数量上限，供 smoke test 使用。
        overwrite (bool): 是否允许覆盖已有输出及摘要。

    Returns:
        Dict[str, Any]: 已写入摘要 JSON 的构建统计与复现信息。

    Raises:
        FileExistsError: 输出已存在且未启用 `overwrite` 时抛出。
        FileNotFoundError: 输入 CSV 或 embedding 不存在时抛出。
        ValueError: 路径、字段、参数或轨迹校验失败时抛出。
        OSError: 输出写入失败时抛出。
    """

    if not dataset_name:
        raise ValueError("dataset_name 不能为空。")
    if not split:
        raise ValueError("split 不能为空。")
    if window_size <= 0:
        raise ValueError("window_size 必须大于 0。")
    if padding_std < 0:
        raise ValueError("padding_std 不允许小于 0。")

    input_path = input_path.resolve()
    embedding_path = embedding_path.resolve()
    output_path = output_path.resolve()
    resolved_summary_path = resolve_summary_path(
        output_path=output_path,
        summary_path=summary_path,
    ).resolve()
    if not overwrite:
        existing_paths = [
            path for path in (output_path, resolved_summary_path) if path.exists()
        ]
        if existing_paths:
            formatted_paths = ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(
                "输出已存在；如需替换请显式传入 --overwrite："
                f"{formatted_paths}"
            )

    LOGGER.info("读取 %s/%s 交互：%s", dataset_name, split, input_path)
    interactions = load_interactions(
        input_path=input_path,
        time_column=time_column,
        reward_column=reward_column,
        max_users=max_users,
        dataset_name=dataset_name,
    )
    source_statistics = {
        "source_num_users": int(interactions["user_id"].nunique()),
        "source_num_transitions": int(len(interactions)),
        "source_num_items": int(interactions["item_id"].nunique()),
        "source_item_id_min": int(interactions["item_id"].min()),
        "source_item_id_max": int(interactions["item_id"].max()),
    }
    LOGGER.info("交互加载完成：%s", source_statistics)

    LOGGER.info("读取 user model item embedding：%s", embedding_path)
    item_embeddings = load_item_embeddings(embedding_path)
    LOGGER.info("Embedding shape=%s", item_embeddings.shape)

    LOGGER.info(
        "按 StateTrackerAvg 语义构造轨迹：window_size=%d, seed=%d",
        window_size,
        seed,
    )
    trajectories = build_trajectories(
        interactions=interactions,
        item_embeddings=item_embeddings,
        reward_column=reward_column,
        window_size=window_size,
        seed=seed,
        padding_std=padding_std,
    )
    statistics = validate_trajectories(
        trajectories,
        expected_action_dim=int(item_embeddings.shape[1]),
    )
    LOGGER.info("轨迹校验通过：%s", statistics)

    # 全量 big matrix 较大，保存前释放不再使用的 DataFrame 以降低峰值内存。
    del interactions
    save_trajectory_dataset(trajectories, output_path)
    summary = {
        **statistics,
        **source_statistics,
        "format_version": 1,
        "dataset_name": dataset_name,
        "split": split,
        "input_path": str(input_path),
        "input_sha256": compute_sha256(input_path),
        "embedding_path": str(embedding_path),
        "embedding_sha256": compute_sha256(embedding_path),
        "embedding_num_items": int(item_embeddings.shape[0]),
        "output_path": str(output_path),
        "output_sha256": compute_sha256(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "time_column": time_column,
        "reward_column": reward_column,
        "state_representation": "StateTrackerAvg(action_embedding,reward)",
        "window_size": int(window_size),
        "seed": int(seed),
        "padding_std": float(padding_std),
        "max_users": max_users,
        "terminal_rule": "only_last_transition_true",
    }
    save_summary(summary, resolved_summary_path)
    LOGGER.info("数据集已保存：%s", output_path)
    LOGGER.info("构造摘要已保存：%s", resolved_summary_path)
    return summary
