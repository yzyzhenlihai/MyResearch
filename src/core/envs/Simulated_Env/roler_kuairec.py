"""ROLeR 在 KuaiRec 上使用的训练/测试跨块近邻构造。"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.core.envs.Simulated_Env.roler_reward import (
    KNNRewardResult,
    compute_knn_reward,
)


DEFAULT_CHUNK_SIZE = 500_000
WATCH_RATIO_CAP = 5.0


def _load_id_column(path: Path, column: str) -> np.ndarray:
    """加载、排序并校验唯一整数 ID 列。

    Args:
        path: CSV 文件路径。
        column: ID 列名。

    Returns:
        严格递增的 ``int64`` ID 数组。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
        ValueError: 列缺失、ID 重复或含负值时抛出。
    """

    if not path.is_file():
        raise FileNotFoundError(f"找不到 KuaiRec ID 文件：{path}")
    frame = pd.read_csv(path, usecols=[column])
    values = np.sort(frame[column].to_numpy(dtype=np.int64, copy=False))
    if values.size == 0 or values[0] < 0 or np.unique(values).size != values.size:
        raise ValueError(f"{path} 的 {column} 必须是非空、非负且唯一的整数 ID。")
    return values


def _build_position_map(ids: np.ndarray, universe_size: int) -> np.ndarray:
    """把原始 ID 映射为指定子矩阵的连续行列位置。

    Args:
        ids: 子集合中的原始 ID。
        universe_size: 完整 ID 空间大小。

    Returns:
        长度为 ``universe_size`` 的位置数组，不属于子集的位置为 ``-1``。
    """

    position = np.full(universe_size, -1, dtype=np.int64)
    position[ids] = np.arange(ids.size, dtype=np.int64)
    return position


def _accumulate_block(
    target: np.ndarray,
    row_position: np.ndarray,
    column_position: np.ndarray,
    users: np.ndarray,
    items: np.ndarray,
    values: np.ndarray,
) -> None:
    """把一个 CSV 分块中的有效交互累加到目标稠密块。

    Args:
        target: 待更新的用户-物品块。
        row_position: 原始用户 ID 到目标行的映射。
        column_position: 原始物品 ID 到目标列的映射。
        users: 当前分块的原始用户 ID。
        items: 当前分块的原始物品 ID。
        values: 当前分块的截断 watch ratio。
    """

    rows = row_position[users]
    columns = column_position[items]
    valid = (rows >= 0) & (columns >= 0)
    if np.any(valid):
        # scipy.csr_matrix 会对重复坐标求和，np.add.at 保持官方行为。
        np.add.at(target, (rows[valid], columns[valid]), values[valid])


def load_official_kuairec_blocks(
    data_root: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按官方 block1/block2/block3 定义读取 KuaiRec 大矩阵。

    ``block1`` 是训练用户在训练独有物品上的历史，``block3`` 是测试用户在
    相同物品上的历史，二者用于跨集合近邻搜索；``block2`` 是训练用户在测试
    物品上的反馈，用于汇聚测试用户的逐物品奖励。

    Args:
        data_root: 包含 ``big_matrix_processed.csv``、``user_id_small.csv`` 等
            文件的 KuaiRec ``data_raw`` 目录。
        chunk_size: 流式读取大矩阵的行数，避免一次加载一千余万行 DataFrame。

    Returns:
        ``(block1, block3, block2)``，均为 ``float32`` 稠密矩阵。

    Raises:
        ValueError: 分块大小非法、ID 不连续或交互 ID 越界时抛出。
    """

    data_root = Path(data_root)
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数。")
    big_path = data_root / "big_matrix_processed.csv"
    if not big_path.is_file():
        raise FileNotFoundError(f"找不到 KuaiRec 大矩阵：{big_path}")

    all_users = _load_id_column(data_root / "user_features.csv", "user_id")
    expected_users = np.arange(all_users.size, dtype=np.int64)
    if not np.array_equal(all_users, expected_users):
        raise ValueError("KuaiRec 用户 ID 必须连续，才能与环境编码保持一致。")
    with (data_root / "video_duration.csv").open(encoding="utf-8") as file_obj:
        item_count = sum(1 for _ in file_obj) - 1
    if item_count <= 0:
        raise ValueError("video_duration.csv 未提供有效物品数量。")

    test_users = _load_id_column(data_root / "user_id_small.csv", "user_id_small")
    test_items = _load_id_column(data_root / "item_id_small.csv", "item_id_small")
    if test_users[-1] >= all_users.size or test_items[-1] >= item_count:
        raise ValueError("KuaiRec small matrix ID 超出 big matrix ID 空间。")
    train_users = np.setdiff1d(expected_users, test_users, assume_unique=True)
    train_items = np.setdiff1d(
        np.arange(item_count, dtype=np.int64), test_items, assume_unique=True
    )

    train_user_position = _build_position_map(train_users, all_users.size)
    test_user_position = _build_position_map(test_users, all_users.size)
    train_item_position = _build_position_map(train_items, item_count)
    test_item_position = _build_position_map(test_items, item_count)

    block1 = np.zeros((train_users.size, train_items.size), dtype=np.float32)
    block3 = np.zeros((test_users.size, train_items.size), dtype=np.float32)
    block2 = np.zeros((train_users.size, test_items.size), dtype=np.float32)
    columns = ["user_id", "item_id", "watch_ratio"]
    for frame in pd.read_csv(big_path, usecols=columns, chunksize=chunk_size):
        users = frame["user_id"].to_numpy(dtype=np.int64, copy=False)
        items = frame["item_id"].to_numpy(dtype=np.int64, copy=False)
        if (
            np.any(users < 0)
            or np.any(users >= all_users.size)
            or np.any(items < 0)
            or np.any(items >= item_count)
        ):
            raise ValueError("big_matrix_processed.csv 包含越界用户或物品 ID。")
        values = frame["watch_ratio"].to_numpy(dtype=np.float32, copy=False)
        values = np.nan_to_num(
            values, nan=0.0, posinf=WATCH_RATIO_CAP, neginf=0.0
        )
        values = np.minimum(values, WATCH_RATIO_CAP)
        _accumulate_block(
            block1,
            train_user_position,
            train_item_position,
            users,
            items,
            values,
        )
        _accumulate_block(
            block3,
            test_user_position,
            train_item_position,
            users,
            items,
            values,
        )
        _accumulate_block(
            block2,
            train_user_position,
            test_item_position,
            users,
            items,
            values,
        )
    return block1, block3, block2


def compute_official_kuairec_reward(
    data_root: Path,
    k: int,
    uncertainty_type: str,
    search_batch_size: int = 256,
    aggregation_batch_size: int | None = None,
    csv_chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> KNNRewardResult:
    """用官方 KuaiRec 跨块定义计算测试用户的 ROLeR 奖励。

    Args:
        data_root: KuaiRec ``data_raw`` 目录。
        k: 奖励与不确定性共同使用的近邻数。
        uncertainty_type: ROLeR 不确定性/加权奖励模式。
        search_batch_size: 余弦搜索的查询批大小。
        aggregation_batch_size: 逐物品反馈聚合批大小。
        csv_chunk_size: 大矩阵 CSV 流式读取行数。

    Returns:
        测试用户 × 测试物品的 kNN 奖励结果。
    """

    block1, block3, block2 = load_official_kuairec_blocks(
        data_root=data_root,
        chunk_size=csv_chunk_size,
    )
    return compute_knn_reward(
        reference_indicator_matrix=block1,
        query_indicator_matrix=block3,
        reference_reward_matrix=block2,
        k=k,
        uncertainty_type=uncertainty_type,
        batch_size=search_batch_size,
        aggregation_batch_size=aggregation_batch_size,
        exclude_self=False,
    )

