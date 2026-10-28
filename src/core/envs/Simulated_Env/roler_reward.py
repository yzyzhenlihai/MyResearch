"""ROLeR 的近邻奖励塑形与不确定性估计。

本模块只依赖 NumPy，既可由训练入口调用，也便于用小型合成矩阵做单元测试。
与原始实现直接构造完整用户相似度矩阵不同，这里按查询批次执行 top-k，
从而将峰值相似度内存从 O(N^2) 降为 O(batch_size * N)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


UNCERTAINTY_ITEM_VARIANCE = "I-var"
UNCERTAINTY_COSINE_MIN = "I-cos_min"
UNCERTAINTY_COSINE_AVERAGE = "I-cos_avg"
UNCERTAINTY_COSINE_MAX = "I-cos_max"
UNCERTAINTY_WEIGHTED_SUM = "II-weight"
UNCERTAINTY_WEIGHTED_DIVIDE = "II-weight-div"
UNCERTAINTY_WEIGHTED_NORMALIZED = "II-weight-norm"

SUPPORTED_UNCERTAINTY_TYPES = (
    UNCERTAINTY_ITEM_VARIANCE,
    UNCERTAINTY_COSINE_MIN,
    UNCERTAINTY_COSINE_AVERAGE,
    UNCERTAINTY_COSINE_MAX,
    UNCERTAINTY_WEIGHTED_SUM,
    UNCERTAINTY_WEIGHTED_DIVIDE,
    UNCERTAINTY_WEIGHTED_NORMALIZED,
)

DEFAULT_MAX_AGGREGATION_ELEMENTS = 20_000_000


@dataclass(frozen=True)
class KNNRewardResult:
    """保存 ROLeR 近邻奖励计算结果。

    Attributes:
        reward: 形状为 ``[查询用户数, 物品数]`` 的近邻奖励矩阵。
        uncertainty: 与 ``reward`` 同形状的不确定性矩阵。
        neighbor_indices: 每个查询用户对应的近邻下标。
        neighbor_similarities: 与近邻下标对应的余弦相似度。
    """

    reward: np.ndarray
    uncertainty: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_similarities: np.ndarray


def _validate_matrix(name: str, matrix: np.ndarray) -> np.ndarray:
    """校验并转换二维有限数值矩阵。

    Args:
        name: 用于异常信息的矩阵名称。
        matrix: 待校验的数组。

    Returns:
        C 连续的 ``float32`` 二维数组。

    Raises:
        ValueError: 输入不是二维、含非有限值或存在空维度时抛出。
    """

    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"{name} 必须是二维矩阵，当前形状为 {array.shape}。")
    if min(array.shape) <= 0:
        raise ValueError(f"{name} 不允许包含空维度，当前形状为 {array.shape}。")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含 NaN 或无穷值。")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        array = array.astype(np.float32)
    return np.ascontiguousarray(array)


def _row_normalize(matrix: np.ndarray, epsilon: float) -> np.ndarray:
    """对矩阵逐行执行稳定的 L2 归一化。

    Args:
        matrix: 已通过校验的二维矩阵。
        epsilon: 防止除零的最小范数。

    Returns:
        与输入同形状的归一化矩阵；全零行仍保持全零。
    """

    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, epsilon)


def batched_cosine_topk(
    reference_matrix: np.ndarray,
    query_matrix: Optional[np.ndarray] = None,
    k: int = 30,
    batch_size: int = 256,
    exclude_self: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """分批搜索余弦相似度最高的 k 个参考用户。

    Args:
        reference_matrix: 参考用户的行为或预测奖励矩阵，形状为 ``[N, D]``。
        query_matrix: 查询用户矩阵，形状为 ``[M, D]``；省略时使用参考矩阵。
        k: 每个查询用户保留的近邻数量。
        batch_size: 单次参与矩阵乘法的查询用户数量。
        exclude_self: 查询矩阵就是参考矩阵时是否排除用户自身。官方 ROLeR
            代码保留自身，因此为复现论文默认使用 ``False``。

    Returns:
        ``(neighbor_indices, neighbor_similarities)``，形状均为 ``[M, k]``，
        且每行按相似度从高到低排列。

    Raises:
        ValueError: 特征维度、k、批大小或排除自身条件不合法时抛出。
    """

    reference = _validate_matrix("reference_matrix", reference_matrix)
    same_matrix = query_matrix is None or query_matrix is reference_matrix
    query = reference if query_matrix is None else _validate_matrix(
        "query_matrix", query_matrix
    )
    if reference.shape[1] != query.shape[1]:
        raise ValueError(
            "reference_matrix 与 query_matrix 的特征维度必须一致："
            f"{reference.shape[1]} != {query.shape[1]}。"
        )
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数。")
    available = reference.shape[0] - int(exclude_self and same_matrix)
    if not 1 <= k <= available:
        raise ValueError(f"k 必须位于 [1, {available}]，当前为 {k}。")

    epsilon = np.finfo(np.result_type(reference.dtype, query.dtype)).eps
    reference_normalized = _row_normalize(reference, epsilon)
    query_normalized = (
        reference_normalized
        if same_matrix
        else _row_normalize(query, epsilon)
    )
    indices = np.empty((query.shape[0], k), dtype=np.int64)
    similarity_dtype = np.result_type(
        reference_normalized.dtype, query_normalized.dtype
    )
    similarities = np.empty((query.shape[0], k), dtype=similarity_dtype)

    for start in range(0, query.shape[0], batch_size):
        stop = min(start + batch_size, query.shape[0])
        batch_similarity = query_normalized[start:stop] @ reference_normalized.T
        if exclude_self and same_matrix:
            local_rows = np.arange(stop - start)
            batch_similarity[local_rows, np.arange(start, stop)] = -np.inf

        partition = np.argpartition(batch_similarity, -k, axis=1)[:, -k:]
        partition_similarity = np.take_along_axis(
            batch_similarity, partition, axis=1
        )
        order = np.argsort(-partition_similarity, axis=1, kind="stable")
        indices[start:stop] = np.take_along_axis(partition, order, axis=1)
        similarities[start:stop] = np.take_along_axis(
            partition_similarity, order, axis=1
        )

    return indices, similarities


def _cosine_uncertainty(
    neighbor_similarities: np.ndarray,
    item_count: int,
    uncertainty_type: str,
) -> np.ndarray:
    """把近邻余弦距离汇聚为用户级不确定性并扩展到所有物品。

    Args:
        neighbor_similarities: 形状为 ``[M, k]`` 的近邻相似度。
        item_count: 需要扩展的物品数量。
        uncertainty_type: ``I-cos_min``、``I-cos_avg`` 或 ``I-cos_max``。

    Returns:
        形状为 ``[M, item_count]`` 的不确定性矩阵。
    """

    distances = 1.0 - neighbor_similarities
    reducers = {
        UNCERTAINTY_COSINE_MIN: np.min,
        UNCERTAINTY_COSINE_AVERAGE: np.mean,
        UNCERTAINTY_COSINE_MAX: np.max,
    }
    user_uncertainty = reducers[uncertainty_type](distances, axis=1)
    return np.repeat(user_uncertainty[:, None], item_count, axis=1).astype(
        np.float32, copy=False
    )


def compute_knn_reward(
    reference_indicator_matrix: np.ndarray,
    reference_reward_matrix: Optional[np.ndarray] = None,
    query_indicator_matrix: Optional[np.ndarray] = None,
    k: int = 30,
    uncertainty_type: str = UNCERTAINTY_ITEM_VARIANCE,
    batch_size: int = 256,
    aggregation_batch_size: Optional[int] = None,
    exclude_self: bool = False,
) -> KNNRewardResult:
    """根据 ROLeR 规则计算 kNN 奖励与不确定性。

    Args:
        reference_indicator_matrix: 用于寻找相似用户的参考矩阵。
        reference_reward_matrix: 近邻用户的逐物品反馈矩阵；省略时与参考指标
            矩阵相同。其行数必须等于参考用户数。
        query_indicator_matrix: 查询用户指标矩阵；省略时查询参考用户自身。
        k: 近邻数量，对应论文及官方代码中的 ``kr``/``ku``。当前实现为
            相同近邻集合，以避免两次完全相同的搜索。
        uncertainty_type: 官方实现支持的七种不确定性/加权奖励模式之一。
        batch_size: 相似度搜索的查询批大小。
        aggregation_batch_size: 聚合近邻逐物品反馈时的查询批大小。省略时根据
            ``k`` 与物品数自动限制临时数组不超过约两千万个元素。
        exclude_self: 是否在同集合搜索中排除自身；复现默认不排除。

    Returns:
        包含近邻奖励、不确定性、近邻下标和相似度的不可变结果对象。

    Raises:
        ValueError: 输入形状或不确定性类型不合法时抛出。

    Example:
        >>> matrix = np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
        >>> result = compute_knn_reward(matrix, k=1, uncertainty_type="I-var")
        >>> result.reward.shape
        (2, 2)
    """

    if uncertainty_type not in SUPPORTED_UNCERTAINTY_TYPES:
        raise ValueError(
            f"不支持的 uncertainty_type={uncertainty_type!r}，"
            f"可选值为 {SUPPORTED_UNCERTAINTY_TYPES}。"
        )
    indicator = _validate_matrix(
        "reference_indicator_matrix", reference_indicator_matrix
    )
    reward_source = (
        indicator
        if reference_reward_matrix is None
        else _validate_matrix("reference_reward_matrix", reference_reward_matrix)
    )
    if reward_source.shape[0] != indicator.shape[0]:
        raise ValueError(
            "reference_reward_matrix 与 reference_indicator_matrix 的用户数必须一致。"
        )

    neighbor_indices, neighbor_similarities = batched_cosine_topk(
        reference_matrix=indicator,
        query_matrix=query_indicator_matrix,
        k=k,
        batch_size=batch_size,
        exclude_self=exclude_self,
    )
    item_count = reward_source.shape[1]
    if aggregation_batch_size is None:
        aggregation_batch_size = max(
            1,
            min(
                batch_size,
                DEFAULT_MAX_AGGREGATION_ELEMENTS // max(1, k * item_count),
            ),
        )
    if aggregation_batch_size <= 0:
        raise ValueError("aggregation_batch_size 必须为正整数。")

    reward = np.empty((neighbor_indices.shape[0], item_count), dtype=np.float32)
    uncertainty = np.zeros_like(reward, dtype=np.float32)
    cosine_types = {
        UNCERTAINTY_COSINE_MIN,
        UNCERTAINTY_COSINE_AVERAGE,
        UNCERTAINTY_COSINE_MAX,
    }
    if uncertainty_type in cosine_types:
        uncertainty[:] = _cosine_uncertainty(
            neighbor_similarities, item_count, uncertainty_type
        )

    for start in range(0, neighbor_indices.shape[0], aggregation_batch_size):
        stop = min(start + aggregation_batch_size, neighbor_indices.shape[0])
        neighbor_rewards = reward_source[neighbor_indices[start:stop]]
        if uncertainty_type == UNCERTAINTY_ITEM_VARIANCE:
            accumulator_dtype = np.result_type(neighbor_rewards.dtype, np.float32)
            reward[start:stop] = neighbor_rewards.mean(
                axis=1, dtype=accumulator_dtype
            )
            uncertainty[start:stop] = neighbor_rewards.var(
                axis=1, dtype=accumulator_dtype
            )
        elif uncertainty_type in cosine_types:
            accumulator_dtype = np.result_type(neighbor_rewards.dtype, np.float32)
            reward[start:stop] = neighbor_rewards.mean(
                axis=1, dtype=accumulator_dtype
            )
        else:
            weights = neighbor_similarities[start:stop, :, None]
            accumulator_dtype = np.result_type(
                weights.dtype, neighbor_rewards.dtype, np.float32
            )
            weighted_sum = np.sum(
                weights * neighbor_rewards, axis=1, dtype=accumulator_dtype
            )
            if uncertainty_type == UNCERTAINTY_WEIGHTED_SUM:
                reward[start:stop] = weighted_sum
            elif uncertainty_type == UNCERTAINTY_WEIGHTED_DIVIDE:
                reward[start:stop] = weighted_sum / np.float32(k)
            else:
                denominator = np.sum(weights, axis=1, dtype=accumulator_dtype)
                epsilon = np.finfo(np.float32).eps
                safe_denominator = np.where(
                    np.abs(denominator) < epsilon,
                    np.copysign(epsilon, denominator + epsilon),
                    denominator,
                )
                reward[start:stop] = weighted_sum / safe_denominator

    return KNNRewardResult(
        reward=np.asarray(reward, dtype=np.float32),
        uncertainty=np.asarray(uncertainty, dtype=np.float32),
        neighbor_indices=neighbor_indices,
        neighbor_similarities=neighbor_similarities,
    )


def minmax_normalize(matrix: np.ndarray) -> np.ndarray:
    """把矩阵稳定地缩放到 ``[0, 1]``，常数矩阵映射为全零。

    Args:
        matrix: 待缩放的二维有限矩阵。

    Returns:
        ``float32`` 类型的归一化矩阵。
    """

    array = _validate_matrix("matrix", matrix)
    minimum = float(array.min())
    value_range = float(array.max()) - minimum
    if value_range <= np.finfo(np.float32).eps:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / value_range).astype(np.float32, copy=False)

