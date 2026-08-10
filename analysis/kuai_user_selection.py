"""KuaiRec 测试矩阵中的高奖励用户统计与导出工具。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_TARGET_REWARD = 1.26
"""DARLR 论文在 KuaiRec 上报告的单步平均奖励。"""

DEFAULT_TOP_K = 100
"""默认导出的高奖励用户数量。"""

DEFAULT_ROUND_DECIMALS = 2
"""检查显示精度下 reward 等于目标值时使用的小数位数。"""

EXACT_ABSOLUTE_TOLERANCE = 1.0e-12
"""判断浮点 reward 与目标值严格相等时使用的绝对容差。"""


def compute_item_reward_statistics(
    reward_matrix: np.ndarray,
    raw_item_ids: Sequence[int],
    selected_user_indexes: Sequence[int],
    target_reward: float = DEFAULT_TARGET_REWARD,
) -> pd.DataFrame:
    """按指定用户子矩阵的真实 reward 均值统计并排序 item。

    该排序使用测试环境真实 reward，属于事后 oracle 筛选，只用于检验
    item 分布偏置能将指标抬高到何种程度，不代表可部署的选择方法。

    Args:
        reward_matrix (np.ndarray): 形状为 ``(user_count, item_count)`` 的
            KuaiRec 真实 reward 矩阵。
        raw_item_ids (Sequence[int]): 每列对应的原始 item ID。
        selected_user_indexes (Sequence[int]): 用于计算 item 统计的矩阵行号。
        target_reward (float): 高 reward 阈值，必须是有限数。

    Returns:
        pd.DataFrame: 以 mean_reward 降序为主键的逐 item 统计表。

    Raises:
        ValueError: 当矩阵、item ID、用户下标或目标值不合法时抛出。
    """

    matrix = np.asarray(reward_matrix)
    item_ids = np.asarray(raw_item_ids)
    user_indexes = np.asarray(selected_user_indexes)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("reward_matrix must be a non-empty two-dimensional array.")
    if not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise ValueError("reward_matrix must contain only finite numeric values.")
    if len(item_ids) != matrix.shape[1] or len(np.unique(item_ids)) != len(item_ids):
        raise ValueError("raw_item_ids must uniquely match reward_matrix columns.")
    if user_indexes.ndim != 1 or user_indexes.size == 0:
        raise ValueError("selected_user_indexes must be a non-empty 1D sequence.")
    if not np.issubdtype(user_indexes.dtype, np.integer):
        raise ValueError("selected_user_indexes must contain integers.")
    user_indexes = user_indexes.astype(np.int64, copy=False)
    if len(np.unique(user_indexes)) != len(user_indexes):
        raise ValueError("selected_user_indexes must not contain duplicates.")
    if user_indexes.min() < 0 or user_indexes.max() >= matrix.shape[0]:
        raise ValueError("selected_user_indexes contains an out-of-range row.")
    if not np.isfinite(target_reward):
        raise ValueError("target_reward must be finite.")

    selected_matrix = matrix[user_indexes]
    statistics = pd.DataFrame(
        {
            "item_id": item_ids.astype(np.int64),
            "matrix_item_index": np.arange(matrix.shape[1], dtype=np.int64),
            "selected_user_count": len(user_indexes),
            "target_reward": float(target_reward),
            "mean_reward": selected_matrix.mean(axis=0),
            "std_reward": selected_matrix.std(axis=0),
            "min_reward": selected_matrix.min(axis=0),
            "median_reward": np.median(selected_matrix, axis=0),
            "max_reward": selected_matrix.max(axis=0),
            "reward_ge_target_count": (selected_matrix >= target_reward).sum(axis=0),
            "reward_ge_target_rate": (selected_matrix >= target_reward).mean(axis=0),
        }
    )
    statistics = statistics.sort_values(
        by=["mean_reward", "reward_ge_target_rate", "item_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    statistics.insert(0, "rank", np.arange(1, len(statistics) + 1))
    return statistics


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """把小型 DataFrame 转成不依赖 ``tabulate`` 的 Markdown 表格。

    Args:
        dataframe (pd.DataFrame): 需要展示的二维表；列名和值会转换为字符串，
            其中竖线会转义以保持 Markdown 单元格边界。

    Returns:
        str: CommonMark 兼容的 Markdown 表格文本。

    Raises:
        ValueError: 当输入没有任何列时抛出。

    Example:
        >>> dataframe_to_markdown(pd.DataFrame({"a": [1]}))
        '| a |\\n| --- |\\n| 1 |'
    """

    if len(dataframe.columns) == 0:
        raise ValueError("dataframe must contain at least one column.")

    def escape_cell(value: Any) -> str:
        """转义单个 Markdown 表格单元格。

        Args:
            value (Any): 任意可字符串化的单元格值。

        Returns:
            str: 已转义换行和竖线的单行文本。
        """

        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(escape_cell(column) for column in dataframe.columns) + " |"
    separator = "| " + " | ".join("---" for _ in dataframe.columns) + " |"
    rows = [
        "| " + " | ".join(escape_cell(value) for value in row) + " |"
        for row in dataframe.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def compute_user_reward_statistics(
    reward_matrix: np.ndarray,
    raw_user_ids: Sequence[int],
    target_reward: float = DEFAULT_TARGET_REWARD,
    round_decimals: int = DEFAULT_ROUND_DECIMALS,
) -> pd.DataFrame:
    """计算每个用户相对于目标 reward 的完整统计量。

    排序主键为 ``reward >= target_reward`` 的比例，次键为用户平均
    reward。该口径用于检验“高 reward 用户抽样能否抬高单步平均奖励”；
    同时保留严格相等和按显示精度相等的统计，避免混淆论文中的均值指标。

    Args:
        reward_matrix (np.ndarray): KuaiRec 环境使用的二维真实 reward 矩阵，
            形状为 ``(user_count, item_count)``。
        raw_user_ids (Sequence[int]): 每一行对应的原始用户 ID，长度必须等于
            ``user_count``。
        target_reward (float): 高 reward 阈值，必须是有限数。
        round_decimals (int): 显示精度相等统计的小数位数，必须非负。

    Returns:
        pd.DataFrame: 按高 reward 比例降序排列的逐用户统计表，包含矩阵行号、
        原始用户 ID、均值、分位数和三种目标 reward 计数口径。

    Raises:
        ValueError: 当矩阵、用户 ID、目标 reward 或小数位数不合法时抛出。

    Example:
        >>> matrix = np.array([[0.0, 1.3], [1.26, 2.0]])
        >>> result = compute_user_reward_statistics(matrix, [10, 20])
        >>> int(result.iloc[0]["user_id"])
        20
    """

    matrix = np.asarray(reward_matrix)
    user_ids = np.asarray(raw_user_ids)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("reward_matrix must be a non-empty two-dimensional array.")
    if len(user_ids) != matrix.shape[0]:
        raise ValueError(
            "raw_user_ids length must match reward_matrix rows: "
            f"{len(user_ids)} != {matrix.shape[0]}."
        )
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("reward_matrix must contain numeric values.")
    if not np.isfinite(matrix).all():
        raise ValueError("reward_matrix must contain only finite values.")
    if not np.isfinite(target_reward):
        raise ValueError("target_reward must be finite.")
    if round_decimals < 0:
        raise ValueError("round_decimals must be non-negative.")
    if len(np.unique(user_ids)) != len(user_ids):
        raise ValueError("raw_user_ids must be unique.")

    item_count = matrix.shape[1]
    greater_equal_mask = matrix >= target_reward
    exact_mask = np.isclose(
        matrix,
        target_reward,
        rtol=0.0,
        atol=EXACT_ABSOLUTE_TOLERANCE,
    )
    rounded_mask = np.round(matrix, round_decimals) == round(
        target_reward,
        round_decimals,
    )

    statistics = pd.DataFrame(
        {
            "user_id": user_ids.astype(np.int64),
            "matrix_user_index": np.arange(matrix.shape[0], dtype=np.int64),
            "item_count": item_count,
            "target_reward": float(target_reward),
            "mean_reward": matrix.mean(axis=1),
            "std_reward": matrix.std(axis=1),
            "min_reward": matrix.min(axis=1),
            "q25_reward": np.quantile(matrix, 0.25, axis=1),
            "median_reward": np.median(matrix, axis=1),
            "q75_reward": np.quantile(matrix, 0.75, axis=1),
            "max_reward": matrix.max(axis=1),
            "reward_ge_target_count": greater_equal_mask.sum(axis=1),
            "reward_ge_target_rate": greater_equal_mask.mean(axis=1),
            "reward_exact_target_count": exact_mask.sum(axis=1),
            "reward_exact_target_rate": exact_mask.mean(axis=1),
            "reward_rounded_target_count": rounded_mask.sum(axis=1),
            "reward_rounded_target_rate": rounded_mask.mean(axis=1),
        }
    )
    statistics = statistics.sort_values(
        by=["reward_ge_target_rate", "mean_reward", "user_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    statistics.insert(0, "rank", np.arange(1, len(statistics) + 1))
    return statistics


def build_reward_selection_summary(
    reward_matrix: np.ndarray,
    statistics: pd.DataFrame,
    top_k: int,
) -> dict[str, Any]:
    """汇总全体用户与 Top-K 用户的 reward 分布。

    Args:
        reward_matrix (np.ndarray): 二维真实 reward 矩阵。
        statistics (pd.DataFrame):
            :func:`compute_user_reward_statistics` 生成的有序统计表。
        top_k (int): 汇总的头部用户数量，必须位于 ``[1, user_count]``。

    Returns:
        dict[str, Any]: 可直接序列化为 JSON 的总体、Top-K 和目标值摘要。

    Raises:
        ValueError: 当 ``top_k`` 越界或统计表缺少必要列时抛出。
    """

    required_columns = {
        "target_reward",
        "mean_reward",
        "reward_ge_target_count",
        "reward_ge_target_rate",
        "reward_exact_target_count",
        "reward_rounded_target_count",
    }
    missing_columns = sorted(required_columns.difference(statistics.columns))
    if missing_columns:
        raise ValueError(f"statistics is missing columns: {missing_columns}.")
    if top_k <= 0 or top_k > len(statistics):
        raise ValueError(
            f"top_k must be in [1, {len(statistics)}], got {top_k}."
        )

    matrix = np.asarray(reward_matrix)
    top_statistics = statistics.head(top_k)
    top_indexes = top_statistics["matrix_user_index"].to_numpy(dtype=np.int64)
    target_reward = float(statistics.iloc[0]["target_reward"])
    top_matrix = matrix[top_indexes]

    return {
        "ranking_definition": (
            "reward_ge_target_rate descending, then mean_reward descending, "
            "then user_id ascending"
        ),
        "target_reward": target_reward,
        "user_count": int(matrix.shape[0]),
        "item_count": int(matrix.shape[1]),
        "matrix_cell_count": int(matrix.size),
        "matrix_mean_reward": float(matrix.mean()),
        "matrix_reward_ge_target_count": int((matrix >= target_reward).sum()),
        "matrix_reward_ge_target_rate": float((matrix >= target_reward).mean()),
        "matrix_reward_exact_target_count": int(
            statistics["reward_exact_target_count"].sum()
        ),
        "matrix_reward_exact_target_rate": float(
            statistics["reward_exact_target_count"].sum() / matrix.size
        ),
        "matrix_reward_rounded_target_count": int(
            statistics["reward_rounded_target_count"].sum()
        ),
        "matrix_reward_rounded_target_rate": float(
            statistics["reward_rounded_target_count"].sum() / matrix.size
        ),
        "top_k": int(top_k),
        "top_k_matrix_mean_reward": float(top_matrix.mean()),
        "top_k_reward_ge_target_rate": float(
            (top_matrix >= target_reward).mean()
        ),
        "top_k_min_user_reward_ge_target_rate": float(
            top_statistics["reward_ge_target_rate"].min()
        ),
        "top_k_max_user_reward_ge_target_rate": float(
            top_statistics["reward_ge_target_rate"].max()
        ),
        "top_k_mean_of_user_mean_reward": float(
            top_statistics["mean_reward"].mean()
        ),
    }


def write_reward_selection_artifacts(
    output_dir: Path,
    statistics: pd.DataFrame,
    summary: dict[str, Any],
    top_k: int,
) -> dict[str, Path]:
    """写出全量统计、Top-K 表格、JSON 摘要和中文 Markdown 摘要。

    Args:
        output_dir (Path): 输出目录；不存在时自动创建。
        statistics (pd.DataFrame): 有序逐用户统计表。
        summary (dict[str, Any]):
            :func:`build_reward_selection_summary` 生成的摘要。
        top_k (int): Top-K 表格行数，必须与摘要一致。

    Returns:
        dict[str, Path]: 四类输出文件的路径映射。

    Raises:
        ValueError: 当 ``top_k`` 与数据或摘要不一致时抛出。
        OSError: 当目录或文件无法写入时由底层接口抛出。
    """

    if top_k <= 0 or top_k > len(statistics):
        raise ValueError("top_k is outside the available statistics rows.")
    if int(summary.get("top_k", -1)) != top_k:
        raise ValueError("summary top_k does not match the requested top_k.")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_users_path = output_dir / "all_user_reward_target_statistics.csv"
    top_users_path = output_dir / f"top_{top_k}_high_reward_users.csv"
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "analysis_summary.md"

    top_statistics = statistics.head(top_k).copy()
    statistics.to_csv(all_users_path, index=False)
    top_statistics.to_csv(top_users_path, index=False)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    display_columns = [
        "rank",
        "user_id",
        "matrix_user_index",
        "mean_reward",
        "median_reward",
        "reward_ge_target_count",
        "reward_ge_target_rate",
        "reward_exact_target_count",
        "reward_rounded_target_count",
    ]
    display_table = top_statistics[display_columns].copy()
    for column in (
        "mean_reward",
        "median_reward",
        "reward_ge_target_rate",
    ):
        display_table[column] = display_table[column].map(lambda value: f"{value:.6f}")

    target_reward = float(summary["target_reward"])
    markdown = "\n".join(
        [
            "# KuaiRec 高奖励用户筛选结果",
            "",
            "## 统计口径",
            "",
            f"- 目标值：`{target_reward}`。",
            "- 主排序：每个用户在全部 item 上 `reward >= 目标值` 的比例降序。",
            "- 次排序：用户平均 reward 降序、原始 user_id 升序。",
            "- `reward_exact_target_count` 使用绝对容差 `1e-12`；",
            "  `reward_rounded_target_count` 使用两位小数显示精度。",
            "- 论文中的 1.26 是测试交互的平均单步 reward，不是 reward 类别。",
            "",
            "## 总体摘要",
            "",
            f"- 矩阵形状：`{summary['user_count']} × {summary['item_count']}`。",
            f"- 全矩阵平均 reward：`{summary['matrix_mean_reward']:.6f}`。",
            f"- 全矩阵 reward >= {target_reward} 比例："
            f"`{summary['matrix_reward_ge_target_rate']:.6%}`。",
            f"- 全矩阵严格 reward = {target_reward} 比例："
            f"`{summary['matrix_reward_exact_target_rate']:.6%}`。",
            f"- Top-{top_k} 子矩阵平均 reward："
            f"`{summary['top_k_matrix_mean_reward']:.6f}`。",
            f"- Top-{top_k} 子矩阵 reward >= {target_reward} 比例："
            f"`{summary['top_k_reward_ge_target_rate']:.6%}`。",
            "",
            f"## Top-{top_k} 用户",
            "",
            dataframe_to_markdown(display_table),
            "",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    LOGGER.info("已写出 KuaiRec 用户筛选结果：%s", output_dir)
    return {
        "all_users_csv": all_users_path,
        "top_users_csv": top_users_path,
        "summary_json": summary_path,
        "summary_markdown": markdown_path,
    }
