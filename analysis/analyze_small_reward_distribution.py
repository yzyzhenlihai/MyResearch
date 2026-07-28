"""统计 KuaiRec small reward matrix 的 reward 与 item 分布。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path(
    "data/KuaiRec/data_raw/small_matrix_processed.csv"
)
"""KuaiRec small matrix 的默认输入路径。"""

DEFAULT_OUTPUT_DIR = Path(
    "results_analysis/small-reward-item-distribution"
)
"""统计产物的默认输出目录。"""

DEFAULT_REWARD_COLUMN = "watch_ratio"
"""KuaiRec 环境 reward matrix 使用的 reward 列。"""

DEFAULT_ITEM_COLUMN = "item_id"
"""small matrix 中的 item 标识列。"""

DEFAULT_USER_COLUMN = "user_id"
"""small matrix 中的 user 标识列。"""

DEFAULT_CLIP_MIN = 0.0
"""reward 的默认下界。"""

DEFAULT_CLIP_MAX = 5.0
"""与 ``KuaiData.load_mat`` 一致的 reward 截断上界。"""

DEFAULT_BIN_EDGES = (
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
)
"""固定 reward 区间边界；重点保留完播阈值 1.0。"""

SUMMARY_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
"""摘要中报告的分位点。"""

ITEM_HISTOGRAM_BINS = 40
"""item 平均 reward 直方图使用的柱数。"""

MATRIX_HISTOGRAM_BINS = 80
"""reward matrix 单元格直方图使用的柱数。"""

PERCENT_MULTIPLIER = 100.0
"""将比例转换为百分比的倍率。"""


def parse_bin_edges(raw_edges: str) -> Tuple[float, ...]:
    """解析并验证逗号分隔的 reward 区间边界。

    Args:
        raw_edges (str): 逗号分隔的浮点数，例如
            ``"0,0.25,0.5,0.75,1,1.5,2,3,4,5"``。

    Returns:
        Tuple[float, ...]: 严格递增的区间边界。

    Raises:
        ValueError: 当边界不足两个、包含非有限值或不严格递增时抛出。
    """
    edges = tuple(float(value.strip()) for value in raw_edges.split(","))
    edge_array = np.asarray(edges, dtype=np.float64)
    if edge_array.size < 2:
        raise ValueError("bin edges 至少需要两个边界。")
    if not np.isfinite(edge_array).all():
        raise ValueError("bin edges 必须全部为有限数。")
    if np.any(np.diff(edge_array) <= 0):
        raise ValueError("bin edges 必须严格递增。")
    return edges


def load_reward_data(
    input_path: Path,
    user_column: str,
    item_column: str,
    reward_column: str,
) -> pd.DataFrame:
    """加载并验证 small matrix 的 user、item 与 reward 列。

    Args:
        input_path (Path): 输入 CSV 文件路径。
        user_column (str): user ID 列名。
        item_column (str): item ID 列名。
        reward_column (str): reward 列名。

    Returns:
        pd.DataFrame: 仅包含指定三列且通过类型检查的数据。

    Raises:
        FileNotFoundError: 当输入文件不存在时抛出。
        ValueError: 当数据为空、缺列、存在缺失/非有限 reward 或重复
            user-item 对时抛出。
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    required_columns = [user_column, item_column, reward_column]
    LOGGER.info("读取 small matrix：%s", input_path)
    try:
        frame = pd.read_csv(input_path, usecols=required_columns)
    except ValueError as error:
        raise ValueError(
            f"输入文件缺少必要列，要求列为：{required_columns}"
        ) from error

    if frame.empty:
        raise ValueError("输入 small matrix 为空。")
    if frame[required_columns].isna().any().any():
        raise ValueError("user、item 或 reward 列中存在缺失值。")

    numeric_rewards = pd.to_numeric(frame[reward_column], errors="raise")
    reward_values = numeric_rewards.to_numpy(dtype=np.float64)
    if not np.isfinite(reward_values).all():
        raise ValueError("reward 列中存在 NaN 或无穷值。")
    frame[reward_column] = numeric_rewards

    duplicate_count = int(frame.duplicated([user_column, item_column]).sum())
    if duplicate_count:
        raise ValueError(
            "small matrix 存在重复 user-item 对，无法直接还原矩阵："
            f"{duplicate_count} 条。"
        )
    LOGGER.info("完成读取：%s 条观测。", f"{len(frame):,}")
    return frame


def describe_values(values: np.ndarray) -> Dict[str, float]:
    """计算一维数值数组的描述性统计。

    Args:
        values (np.ndarray): 非空、有限的一维数值数组。

    Returns:
        Dict[str, float]: 样本数、均值、标准差、极值和指定分位点。

    Raises:
        ValueError: 当输入不是非空有限一维数组时抛出。
    """
    numeric_values = np.asarray(values, dtype=np.float64)
    if numeric_values.ndim != 1 or numeric_values.size == 0:
        raise ValueError("values 必须是非空一维数组。")
    if not np.isfinite(numeric_values).all():
        raise ValueError("values 必须全部为有限数。")

    quantile_values = np.quantile(numeric_values, SUMMARY_QUANTILES)
    summary = {
        "count": int(numeric_values.size),
        "min": float(np.min(numeric_values)),
        "max": float(np.max(numeric_values)),
        "mean": float(np.mean(numeric_values)),
        "std": float(np.std(numeric_values)),
    }
    for quantile, value in zip(SUMMARY_QUANTILES, quantile_values):
        summary[f"q{int(quantile * PERCENT_MULTIPLIER):02d}"] = float(value)
    return summary


def make_interval_labels(bin_edges: Sequence[float]) -> list[str]:
    """生成与 ``numpy.histogram`` 左闭右开规则一致的区间标签。

    Args:
        bin_edges (Sequence[float]): 严格递增的区间边界。

    Returns:
        list[str]: 区间标签；最后一个区间右端点闭合。
    """
    labels = []
    final_interval_index = len(bin_edges) - 2
    for interval_index, (left, right) in enumerate(
        zip(bin_edges[:-1], bin_edges[1:])
    ):
        closing_bracket = "]" if interval_index == final_interval_index else ")"
        labels.append(f"[{left:g}, {right:g}{closing_bracket}")
    return labels


def build_bin_table(
    values: np.ndarray,
    bin_edges: Sequence[float],
    count_name: str,
) -> pd.DataFrame:
    """按固定 reward 区间统计数值数量及占比。

    Args:
        values (np.ndarray): 待统计的一维数值数组。
        bin_edges (Sequence[float]): ``numpy.histogram`` 使用的区间边界。
        count_name (str): 输出数量列名。

    Returns:
        pd.DataFrame: 包含区间、左右边界、数量和百分比的表格。

    Raises:
        ValueError: 当数值落在给定区间范围之外时抛出。
    """
    numeric_values = np.asarray(values, dtype=np.float64)
    counts, numeric_edges = np.histogram(numeric_values, bins=bin_edges)
    if int(counts.sum()) != int(numeric_values.size):
        outside_count = int(numeric_values.size - counts.sum())
        raise ValueError(
            f"有 {outside_count} 个值落在 bin edges 覆盖范围之外。"
        )

    return pd.DataFrame(
        {
            "reward_interval": make_interval_labels(numeric_edges),
            "left_edge": numeric_edges[:-1],
            "right_edge": numeric_edges[1:],
            count_name: counts.astype(np.int64),
            "percentage": counts / numeric_values.size * PERCENT_MULTIPLIER,
        }
    )


def compute_item_statistics(
    frame: pd.DataFrame,
    item_column: str,
    reward_column: str,
    user_count: int,
) -> pd.DataFrame:
    """计算每个 item 在完整 reward matrix 上的 reward 统计。

    CSV 只存储已观测 user-item 对，而 ``KuaiData.load_mat`` 通过稀疏矩阵
    转稠密矩阵，使缺失 user-item 单元格取 0。本函数因此用完整 user 数作为
    每个 item 的分母，并将隐式缺失单元格纳入均值和标准差。

    Args:
        frame (pd.DataFrame): 包含 item 和已截断 reward 的观测表。
        item_column (str): item ID 列名。
        reward_column (str): 已截断 reward 列名。
        user_count (int): reward matrix 的 user 维度，必须大于 0。

    Returns:
        pd.DataFrame: 每个 item 的观测数、隐式零数、均值、标准差和极值。

    Raises:
        ValueError: 当 user 数无效或某 item 的观测数超过 user 数时抛出。
    """
    if user_count <= 0:
        raise ValueError("user_count 必须大于 0。")

    working_frame = frame[[item_column, reward_column]].copy()
    working_frame["_reward_squared"] = np.square(
        working_frame[reward_column].to_numpy(dtype=np.float64)
    )
    grouped = working_frame.groupby(item_column, sort=True).agg(
        observed_user_count=(reward_column, "size"),
        reward_sum=(reward_column, "sum"),
        reward_squared_sum=("_reward_squared", "sum"),
        observed_reward_min=(reward_column, "min"),
        reward_max=(reward_column, "max"),
    )
    if (grouped["observed_user_count"] > user_count).any():
        raise ValueError("至少一个 item 的观测 user 数超过矩阵 user 维度。")

    grouped["implicit_zero_count"] = (
        user_count - grouped["observed_user_count"]
    )
    grouped["mean_reward"] = grouped["reward_sum"] / user_count
    second_moment = grouped["reward_squared_sum"] / user_count
    variance = np.maximum(
        second_moment - np.square(grouped["mean_reward"]),
        0.0,
    )
    grouped["std_reward"] = np.sqrt(variance)
    grouped["min_reward"] = np.where(
        grouped["implicit_zero_count"] > 0,
        0.0,
        grouped["observed_reward_min"],
    )
    grouped["matrix_cell_count"] = user_count

    output_columns = [
        "matrix_cell_count",
        "observed_user_count",
        "implicit_zero_count",
        "mean_reward",
        "std_reward",
        "min_reward",
        "reward_max",
    ]
    return grouped[output_columns].reset_index()


def build_dense_reward_matrix(
    frame: pd.DataFrame,
    user_column: str,
    item_column: str,
    reward_column: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将无重复的观测表还原为完整 user-item reward matrix。

    未出现在 CSV 中的 user-item 单元格保留为 0，与项目环境中稀疏矩阵
    ``toarray()`` 的行为一致。

    Args:
        frame (pd.DataFrame): 包含 user、item 和已截断 reward 的观测表。
        user_column (str): user ID 列名。
        item_column (str): item ID 列名。
        reward_column (str): reward 列名。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 依次为二维 reward matrix、
        排序后的 user ID 数组和排序后的 item ID 数组。

    Raises:
        ValueError: 当 ID 无法映射到矩阵索引时抛出。
    """
    user_ids = np.sort(frame[user_column].unique())
    item_ids = np.sort(frame[item_column].unique())
    user_indices = pd.Index(user_ids).get_indexer(frame[user_column])
    item_indices = pd.Index(item_ids).get_indexer(frame[item_column])
    if np.any(user_indices < 0) or np.any(item_indices < 0):
        raise ValueError("至少一个 user 或 item ID 无法映射到矩阵索引。")

    reward_matrix = np.zeros(
        (len(user_ids), len(item_ids)),
        dtype=np.float64,
    )
    reward_matrix[user_indices, item_indices] = frame[
        reward_column
    ].to_numpy(dtype=np.float64)
    return reward_matrix, user_ids, item_ids


def compute_user_statistics(
    reward_matrix: np.ndarray,
    user_ids: np.ndarray,
    observed_item_counts: pd.Series,
    user_column: str,
) -> pd.DataFrame:
    """计算每个 user 在全部 item 上的 reward 分布统计。

    Args:
        reward_matrix (np.ndarray): 完整二维 user-item reward matrix。
        user_ids (np.ndarray): 与矩阵行一一对应的 user ID。
        observed_item_counts (pd.Series): 以 user ID 为索引的 CSV 观测 item 数。
        user_column (str): 输出使用的 user ID 列名。

    Returns:
        pd.DataFrame: 每个 user 的观测/隐式零数量、均值、标准差、极值和分位点。

    Raises:
        ValueError: 当矩阵形状、user ID 数量或观测数量不一致时抛出。
    """
    if reward_matrix.ndim != 2:
        raise ValueError("reward_matrix 必须是二维数组。")
    if reward_matrix.shape[0] != len(user_ids):
        raise ValueError("reward_matrix 行数必须与 user_ids 长度一致。")

    item_count = int(reward_matrix.shape[1])
    aligned_observed_counts = observed_item_counts.reindex(user_ids)
    if aligned_observed_counts.isna().any():
        raise ValueError("至少一个 user 缺少观测 item 数。")
    observed_counts = aligned_observed_counts.to_numpy(dtype=np.int64)
    if np.any(observed_counts > item_count):
        raise ValueError("至少一个 user 的观测 item 数超过矩阵 item 维度。")

    quantile_matrix = np.quantile(
        reward_matrix,
        SUMMARY_QUANTILES,
        axis=1,
    )
    statistics = pd.DataFrame(
        {
            user_column: user_ids,
            "matrix_cell_count": item_count,
            "observed_item_count": observed_counts,
            "implicit_zero_count": item_count - observed_counts,
            "mean_reward": np.mean(reward_matrix, axis=1),
            "std_reward": np.std(reward_matrix, axis=1),
            "min_reward": np.min(reward_matrix, axis=1),
            "max_reward": np.max(reward_matrix, axis=1),
        }
    )
    for quantile_index, quantile in enumerate(SUMMARY_QUANTILES):
        statistics[f"q{int(quantile * PERCENT_MULTIPLIER):02d}_reward"] = (
            quantile_matrix[quantile_index]
        )
    return statistics


def build_per_user_bin_table(
    reward_matrix: np.ndarray,
    user_ids: np.ndarray,
    user_column: str,
    bin_edges: Sequence[float],
) -> pd.DataFrame:
    """统计每个 user 在各 reward 区间中的 item 数量。

    Args:
        reward_matrix (np.ndarray): 完整二维 user-item reward matrix。
        user_ids (np.ndarray): 与矩阵行一一对应的 user ID。
        user_column (str): 输出使用的 user ID 列名。
        bin_edges (Sequence[float]): 固定 reward 区间边界。

    Returns:
        pd.DataFrame: 长表格式的 user ID、reward 区间、item 数量和占比。

    Raises:
        ValueError: 当矩阵维度、user 数或任一用户的分区计数不一致时抛出。
    """
    if reward_matrix.ndim != 2:
        raise ValueError("reward_matrix 必须是二维数组。")
    if reward_matrix.shape[0] != len(user_ids):
        raise ValueError("reward_matrix 行数必须与 user_ids 长度一致。")

    item_count = int(reward_matrix.shape[1])
    interval_labels = make_interval_labels(bin_edges)
    records = []
    for user_id, user_rewards in zip(user_ids, reward_matrix):
        counts, _ = np.histogram(user_rewards, bins=bin_edges)
        if int(counts.sum()) != item_count:
            raise ValueError(f"user_id={user_id} 的分区计数不等于 item 总数。")
        for interval_index, count in enumerate(counts):
            records.append(
                {
                    user_column: user_id,
                    "reward_interval": interval_labels[interval_index],
                    "left_edge": float(bin_edges[interval_index]),
                    "right_edge": float(bin_edges[interval_index + 1]),
                    "item_count": int(count),
                    "percentage": (
                        int(count) / item_count * PERCENT_MULTIPLIER
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def resolve_example_user_id(
    user_ids: np.ndarray,
    requested_user_id: Optional[int],
) -> object:
    """选择需要单独展示 reward 分布的示例 user ID。

    未指定时使用排序后的第一个 user ID，保证结果确定且可复现。

    Args:
        user_ids (np.ndarray): 非空的有效 user ID 数组。
        requested_user_id (Optional[int]): 命令行指定的 user ID；为空时自动选择。

    Returns:
        object: 数据中实际存在的示例 user ID。

    Raises:
        ValueError: 当 user ID 数组为空或指定 ID 不存在时抛出。
    """
    if len(user_ids) == 0:
        raise ValueError("user_ids 不能为空。")
    example_user_id = (
        user_ids[0] if requested_user_id is None else requested_user_id
    )
    matching_positions = np.flatnonzero(user_ids == example_user_id)
    if matching_positions.size == 0:
        raise ValueError(f"指定的 example user_id 不存在：{example_user_id}")
    resolved_user_id = user_ids[int(matching_positions[0])]
    return (
        resolved_user_id.item()
        if isinstance(resolved_user_id, np.generic)
        else resolved_user_id
    )


def plot_example_user_distribution(
    user_id: object,
    user_rewards: np.ndarray,
    user_bin_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制一个具体 user ID 的 reward 直方图与区间 item 数量。

    Args:
        user_id (object): 展示的 user ID。
        user_rewards (np.ndarray): 该用户在全部 item 上的 reward。
        user_bin_table (pd.DataFrame): 该用户的 reward 区间计数表。
        output_path (Path): PNG 输出路径。

    Returns:
        None
    """
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 5),
        constrained_layout=True,
    )
    axes[0].hist(
        user_rewards,
        bins=MATRIX_HISTOGRAM_BINS,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0].axvline(
        float(np.mean(user_rewards)),
        color="#E45756",
        linestyle="--",
        label=f"mean = {np.mean(user_rewards):.3f}",
    )
    axes[0].set_title(f"Reward distribution for user_id={user_id}")
    axes[0].set_xlabel("Clipped watch ratio reward")
    axes[0].set_ylabel("Item count")
    axes[0].legend()

    positions = np.arange(len(user_bin_table))
    counts = user_bin_table["item_count"].to_numpy(dtype=np.int64)
    axes[1].bar(
        positions,
        counts,
        color="#72B7B2",
        edgecolor="#356E6B",
        linewidth=0.5,
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        user_bin_table["reward_interval"],
        rotation=45,
        ha="right",
    )
    axes[1].set_title(f"Items by reward interval: user_id={user_id}")
    axes[1].set_xlabel("Reward interval")
    axes[1].set_ylabel("Item count")
    for position, count in zip(positions, counts):
        if count > 0:
            axes[1].text(
                position,
                count,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_distributions(
    matrix_rewards: np.ndarray,
    item_statistics: pd.DataFrame,
    item_bin_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制 matrix reward 与 item 平均 reward 分布图。

    Args:
        matrix_rewards (np.ndarray): 完整 reward matrix 展平后的 reward。
        item_statistics (pd.DataFrame): 每个 item 的 reward 统计表。
        item_bin_table (pd.DataFrame): item 平均 reward 固定区间计数表。
        output_path (Path): PNG 输出路径。

    Returns:
        None
    """
    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 5),
        constrained_layout=True,
    )

    axes[0].hist(
        matrix_rewards,
        bins=MATRIX_HISTOGRAM_BINS,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.3,
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Reward matrix cells")
    axes[0].set_xlabel("Clipped watch ratio reward")
    axes[0].set_ylabel("Cell count (log scale)")
    axes[0].axvline(1.0, color="#E45756", linestyle="--", label="reward = 1")
    axes[0].legend()

    item_means = item_statistics["mean_reward"].to_numpy(dtype=np.float64)
    axes[1].hist(
        item_means,
        bins=ITEM_HISTOGRAM_BINS,
        color="#72B7B2",
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].axvline(
        float(np.mean(item_means)),
        color="#E45756",
        linestyle="--",
        label="mean",
    )
    axes[1].set_title("Per-item mean reward")
    axes[1].set_xlabel("Mean reward across all users")
    axes[1].set_ylabel("Item count")
    axes[1].legend()

    interval_positions = np.arange(len(item_bin_table))
    item_counts = item_bin_table["item_count"].to_numpy(dtype=np.int64)
    axes[2].bar(
        interval_positions,
        item_counts,
        color="#F2CF5B",
        edgecolor="#8C6D1F",
        linewidth=0.5,
    )
    axes[2].set_xticks(interval_positions)
    axes[2].set_xticklabels(
        item_bin_table["reward_interval"],
        rotation=45,
        ha="right",
    )
    axes[2].set_title("Items by mean-reward interval")
    axes[2].set_xlabel("Mean reward interval")
    axes[2].set_ylabel("Item count")
    for position, count in zip(interval_positions, item_counts):
        if count > 0:
            axes[2].text(
                position,
                count,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def data_frame_to_markdown(frame: pd.DataFrame) -> str:
    """将 DataFrame 转换成不依赖 ``tabulate`` 的 Markdown 表格。

    Args:
        frame (pd.DataFrame): 待转换的二维表格。

    Returns:
        str: GitHub Flavored Markdown 表格文本。
    """

    def format_cell(value: object) -> str:
        """格式化单个 Markdown 表格单元格。

        Args:
            value (object): 单元格原始值。

        Returns:
            str: 转义竖线后的可展示文本。
        """
        if isinstance(value, (float, np.floating)):
            text = f"{float(value):.6f}"
        else:
            text = str(value)
        return text.replace("|", "\\|")

    headers = [format_cell(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(format_cell(value) for value in row) + " |"
        )
    return "\n".join(lines)


def write_markdown_summary(
    output_path: Path,
    input_path: Path,
    summary: Dict[str, object],
    matrix_bin_table: pd.DataFrame,
    item_bin_table: pd.DataFrame,
    example_user_bin_table: pd.DataFrame,
) -> None:
    """写出面向读者的中文 Markdown 统计摘要。

    Args:
        output_path (Path): Markdown 输出路径。
        input_path (Path): 分析使用的原始 CSV 路径。
        summary (Dict[str, object]): JSON 兼容的统计摘要。
        matrix_bin_table (pd.DataFrame): matrix 单元格区间计数。
        item_bin_table (pd.DataFrame): item 平均 reward 区间计数。
        example_user_bin_table (pd.DataFrame): 示例用户的 reward 区间计数。

    Returns:
        None
    """
    matrix_summary = summary["matrix_reward"]
    item_summary = summary["item_mean_reward"]
    user_summary = summary["user_mean_reward"]
    example_user_summary = summary["example_user_reward"]
    lines = [
        "# KuaiRec Small Reward Matrix：Item 与 User Reward 分布",
        "",
        "## 统计口径",
        "",
        f"- 输入：`{input_path}`",
        "- reward：`watch_ratio`，按环境实现截断到 `[0, 5]`。",
        "- 完整 reward matrix 中未出现在 CSV 的 user-item 单元格按 `0` 计。",
        "- item reward：一个 item 在全部 user 上的平均 matrix reward。",
        "- user reward：一个 user 在全部 item 上的 matrix reward 分布。",
        "",
        "## 总体摘要",
        "",
        f"- matrix 形状：`{summary['user_count']} × {summary['item_count']}`",
        f"- matrix 单元格数：`{summary['matrix_cell_count']:,}`",
        f"- CSV 已观测单元格数：`{summary['observed_cell_count']:,}`",
        f"- 隐式零单元格数：`{summary['implicit_zero_count']:,}`",
        (
            "- 单元格 reward："
            f"mean={matrix_summary['mean']:.6f}，"
            f"median={matrix_summary['q50']:.6f}，"
            f"min={matrix_summary['min']:.6f}，"
            f"max={matrix_summary['max']:.6f}"
        ),
        (
            "- item 平均 reward："
            f"mean={item_summary['mean']:.6f}，"
            f"median={item_summary['q50']:.6f}，"
            f"min={item_summary['min']:.6f}，"
            f"max={item_summary['max']:.6f}"
        ),
        (
            "- user 平均 reward："
            f"mean={user_summary['mean']:.6f}，"
            f"median={user_summary['q50']:.6f}，"
            f"min={user_summary['min']:.6f}，"
            f"max={user_summary['max']:.6f}"
        ),
        "",
        "## Reward Matrix 单元格区间分布",
        "",
        data_frame_to_markdown(matrix_bin_table),
        "",
        "## 不同平均 Reward 区间下的 Item 数量",
        "",
        data_frame_to_markdown(item_bin_table),
        "",
        f"## 示例 User Reward 分布：user_id={summary['example_user_id']}",
        "",
        (
            f"- mean={example_user_summary['mean']:.6f}，"
            f"median={example_user_summary['q50']:.6f}，"
            f"min={example_user_summary['min']:.6f}，"
            f"max={example_user_summary['max']:.6f}"
        ),
        "",
        data_frame_to_markdown(example_user_bin_table),
        "",
        "全部 user_id 的结果见 `per_user_reward_statistics.csv` 和 "
        "`per_user_reward_interval_counts.csv`。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def analyze_reward_distribution(
    input_path: Path,
    output_dir: Path,
    user_column: str,
    item_column: str,
    reward_column: str,
    clip_min: float,
    clip_max: float,
    bin_edges: Sequence[float],
    requested_example_user_id: Optional[int] = None,
) -> Dict[str, object]:
    """执行完整的 small reward matrix 分布统计并保存产物。

    Args:
        input_path (Path): small matrix CSV 路径。
        output_dir (Path): 结果输出目录。
        user_column (str): user ID 列名。
        item_column (str): item ID 列名。
        reward_column (str): reward 列名。
        clip_min (float): reward 截断下界。
        clip_max (float): reward 截断上界。
        bin_edges (Sequence[float]): 固定 reward 区间边界。
        requested_example_user_id (Optional[int]): 需要单独展示的 user ID；
            为空时使用排序后的第一个 user ID。

    Returns:
        Dict[str, object]: JSON 兼容的总体统计摘要。

    Raises:
        ValueError: 当截断边界无效、bin edges 未覆盖截断范围或矩阵
            不满足预期结构时抛出。
    """
    if not np.isfinite([clip_min, clip_max]).all() or clip_min >= clip_max:
        raise ValueError("clip_min 和 clip_max 必须有限且满足 min < max。")
    if bin_edges[0] > clip_min or bin_edges[-1] < clip_max:
        raise ValueError("bin edges 必须完整覆盖 reward 截断范围。")

    frame = load_reward_data(
        input_path,
        user_column,
        item_column,
        reward_column,
    )
    user_count = int(frame[user_column].nunique())
    item_count = int(frame[item_column].nunique())
    matrix_cell_count = user_count * item_count
    observed_cell_count = int(len(frame))
    if observed_cell_count > matrix_cell_count:
        raise ValueError("CSV 行数超过完整 user-item reward matrix 单元格数。")

    # 复现 KuaiData.load_mat 的 reward 截断，并显式补入稀疏转稠密产生的 0。
    clipped_rewards = np.clip(
        frame[reward_column].to_numpy(dtype=np.float64),
        clip_min,
        clip_max,
    )
    frame[reward_column] = clipped_rewards
    reward_matrix, user_ids, item_ids = build_dense_reward_matrix(
        frame,
        user_column,
        item_column,
        reward_column,
    )
    if reward_matrix.shape != (user_count, item_count):
        raise ValueError("还原后的 reward matrix 形状与 user/item 数不一致。")
    if len(item_ids) != item_count:
        raise ValueError("排序后的 item ID 数与 item 总数不一致。")
    implicit_zero_count = matrix_cell_count - observed_cell_count
    matrix_rewards = reward_matrix.reshape(-1)

    item_statistics = compute_item_statistics(
        frame,
        item_column,
        reward_column,
        user_count,
    )
    if len(item_statistics) != item_count:
        raise ValueError("item 统计表行数与矩阵 item 维度不一致。")

    observed_item_counts = frame.groupby(user_column, sort=False).size()
    user_statistics = compute_user_statistics(
        reward_matrix,
        user_ids,
        observed_item_counts,
        user_column,
    )
    if len(user_statistics) != user_count:
        raise ValueError("user 统计表行数与矩阵 user 维度不一致。")
    per_user_bin_table = build_per_user_bin_table(
        reward_matrix,
        user_ids,
        user_column,
        bin_edges,
    )
    per_user_count_sums = per_user_bin_table.groupby(user_column)[
        "item_count"
    ].sum()
    if not (per_user_count_sums == item_count).all():
        raise ValueError("至少一个 user 的区间 item 数之和不等于 item 总数。")

    example_user_id = resolve_example_user_id(
        user_ids,
        requested_example_user_id,
    )
    example_user_position = int(
        np.flatnonzero(user_ids == example_user_id)[0]
    )
    example_user_rewards = reward_matrix[example_user_position]
    example_user_bin_table = per_user_bin_table[
        per_user_bin_table[user_column] == example_user_id
    ].copy()

    matrix_bin_table = build_bin_table(
        matrix_rewards,
        bin_edges,
        "matrix_cell_count",
    )
    item_bin_table = build_bin_table(
        item_statistics["mean_reward"].to_numpy(dtype=np.float64),
        bin_edges,
        "item_count",
    )
    if int(item_bin_table["item_count"].sum()) != item_count:
        raise ValueError("item 分区计数之和与 item 总数不一致。")

    summary: Dict[str, object] = {
        "input_path": str(input_path),
        "reward_definition": (
            f"clip({reward_column}, {clip_min:g}, {clip_max:g}); "
            "missing user-item cells are zero"
        ),
        "user_count": user_count,
        "item_count": item_count,
        "matrix_cell_count": matrix_cell_count,
        "observed_cell_count": observed_cell_count,
        "implicit_zero_count": implicit_zero_count,
        "observed_coverage": observed_cell_count / matrix_cell_count,
        "matrix_reward": describe_values(matrix_rewards),
        "item_mean_reward": describe_values(
            item_statistics["mean_reward"].to_numpy(dtype=np.float64)
        ),
        "user_mean_reward": describe_values(
            user_statistics["mean_reward"].to_numpy(dtype=np.float64)
        ),
        "example_user_id": example_user_id,
        "example_user_reward": describe_values(example_user_rewards),
        "bin_edges": [float(value) for value in bin_edges],
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    matrix_bin_table.to_csv(
        output_dir / "matrix_reward_interval_counts.csv",
        index=False,
    )
    item_bin_table.to_csv(
        output_dir / "item_mean_reward_interval_counts.csv",
        index=False,
    )
    item_statistics.to_csv(
        output_dir / "per_item_reward_statistics.csv",
        index=False,
    )
    user_statistics.to_csv(
        output_dir / "per_user_reward_statistics.csv",
        index=False,
    )
    per_user_bin_table.to_csv(
        output_dir / "per_user_reward_interval_counts.csv",
        index=False,
    )
    example_user_bin_table.to_csv(
        output_dir / "example_user_reward_interval_counts.csv",
        index=False,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    plot_distributions(
        matrix_rewards,
        item_statistics,
        item_bin_table,
        output_dir / "reward_distribution.png",
    )
    plot_example_user_distribution(
        example_user_id,
        example_user_rewards,
        example_user_bin_table,
        output_dir / "example_user_reward_distribution.png",
    )
    write_markdown_summary(
        output_dir / "analysis_summary.md",
        input_path,
        summary,
        matrix_bin_table,
        item_bin_table,
        example_user_bin_table,
    )

    LOGGER.info("统计完成，结果目录：%s", output_dir)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    Returns:
        argparse.ArgumentParser: 已配置的命令行解析器。
    """
    parser = argparse.ArgumentParser(
        description="统计 KuaiRec small reward matrix 的 item reward 分布。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--user-column", default=DEFAULT_USER_COLUMN)
    parser.add_argument("--item-column", default=DEFAULT_ITEM_COLUMN)
    parser.add_argument("--reward-column", default=DEFAULT_REWARD_COLUMN)
    parser.add_argument("--clip-min", type=float, default=DEFAULT_CLIP_MIN)
    parser.add_argument("--clip-max", type=float, default=DEFAULT_CLIP_MAX)
    parser.add_argument(
        "--example-user-id",
        type=int,
        default=None,
        help="单独输出分布图的 user ID；不指定时使用排序后的第一个用户。",
    )
    parser.add_argument(
        "--bin-edges",
        default=",".join(str(value) for value in DEFAULT_BIN_EDGES),
        help="逗号分隔的 reward 区间边界。",
    )
    return parser


def main() -> None:
    """解析命令行参数并执行分布统计。

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = build_argument_parser()
    args = parser.parse_args()
    bin_edges = parse_bin_edges(args.bin_edges)
    summary = analyze_reward_distribution(
        input_path=args.input,
        output_dir=args.output_dir,
        user_column=args.user_column,
        item_column=args.item_column,
        reward_column=args.reward_column,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        bin_edges=bin_edges,
        requested_example_user_id=args.example_user_id,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
