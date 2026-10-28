#!/usr/bin/env python3
"""统计并绘制 EasyRL4Rec 原始数据集的用户轨迹长度分布。"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


LOGGER = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1_000_000
"""分块读取表格文件时每块的最大行数。"""

DEFAULT_DENSITY_GRID_SIZE = 512
"""平滑密度曲线在对数空间中的网格数量。"""

MIN_LOG_BANDWIDTH = 0.035
"""核平滑在 log10 空间中的最小带宽，用于稳定固定长度分布。"""

LOG_RANGE_PADDING = 0.08
"""密度网格在 log10 空间两端保留的绘图边距。"""

SILVERMAN_FACTOR = 1.06
"""Silverman 核带宽经验公式的系数。"""

FIGURE_WIDTH = 11.5
"""输出图像宽度，单位为英寸。"""

FIGURE_HEIGHT = 6.5
"""输出图像高度，单位为英寸。"""

OUTPUT_DPI = 180
"""PNG 图像输出分辨率。"""

DATASET_COLORS = {
    "Coat": "#0072B2",
    "KuaiRand_Pure": "#E69F00",
    "KuaiRec": "#009E73",
    "MovieLens": "#CC79A7",
    "YahooR3": "#D55E00",
}
"""各数据集在图中的固定配色。"""


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 包含数据目录、图片路径和汇总表路径的参数。

    Raises:
        SystemExit: 当命令行参数不合法或请求帮助信息时由 argparse 抛出。
    """
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="统计原始交互记录中每位用户的轨迹长度并绘制分布图。"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repository_root / "data",
        help="EasyRL4Rec 数据根目录。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository_root
            / "visual_results"
            / "figures"
            / "trajectory_length_distribution.png"
        ),
        help="分布图输出路径。",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=(
            repository_root
            / "visual_results"
            / "figures"
            / "trajectory_length_distribution_summary.csv"
        ),
        help="描述统计 CSV 输出路径。",
    )
    return parser.parse_args()


def get_dataset_specs(data_root: Path) -> dict[str, dict[str, Any]]:
    """构造各数据集原始交互文件的读取配置。

    Args:
        data_root (Path): 包含各数据集子目录的 ``data`` 根目录。

    Returns:
        dict[str, dict[str, Any]]: 数据集名称到读取方式、文件路径及用户列的映射。

    Raises:
        ValueError: 当 ``data_root`` 不是已存在目录时抛出。
    """
    if not data_root.is_dir():
        raise ValueError(f"数据根目录不存在或不是目录：{data_root}")

    return {
        "Coat": {
            "format": "matrix",
            "paths": [
                data_root / "Coat" / "data_raw" / "train.ascii",
                data_root / "Coat" / "data_raw" / "test.ascii",
            ],
        },
        "KuaiRand_Pure": {
            "format": "table",
            "paths": [
                data_root
                / "KuaiRand_Pure"
                / "data_raw"
                / "log_standard_4_08_to_4_21_pure.csv",
                data_root
                / "KuaiRand_Pure"
                / "data_raw"
                / "log_standard_4_22_to_5_08_pure.csv",
                data_root
                / "KuaiRand_Pure"
                / "data_raw"
                / "log_random_4_22_to_5_08_pure.csv",
            ],
            "user_column": "user_id",
            "header": "infer",
        },
        "KuaiRec": {
            "format": "table",
            "paths": [
                data_root / "KuaiRec" / "data_raw" / "big_matrix.csv",
                data_root / "KuaiRec" / "data_raw" / "small_matrix.csv",
            ],
            "user_column": "user_id",
            "header": "infer",
        },
        "MovieLens": {
            "format": "table",
            "paths": [
                data_root
                / "MovieLens"
                / "data_raw"
                / "movielens-1m-train.csv",
                data_root
                / "MovieLens"
                / "data_raw"
                / "movielens-1m-test.csv",
            ],
            "user_column": "UserID",
            "header": "infer",
        },
        "YahooR3": {
            "format": "table",
            "paths": [
                data_root
                / "YahooR3"
                / "data_raw"
                / "ydata-ymusic-rating-study-v1_0-train.txt",
                data_root
                / "YahooR3"
                / "data_raw"
                / "ydata-ymusic-rating-study-v1_0-test.txt",
            ],
            "user_column": 0,
            "header": None,
            "separator": "\t",
        },
    }


def validate_source_paths(dataset_specs: dict[str, dict[str, Any]]) -> None:
    """验证全部配置文件是否存在且为非空普通文件。

    Args:
        dataset_specs (dict[str, dict[str, Any]]): 数据集读取配置。

    Returns:
        None: 所有文件通过检查时正常返回。

    Raises:
        FileNotFoundError: 当任一数据文件不存在时抛出。
        ValueError: 当任一数据文件不是普通文件或内容为空时抛出。
    """
    for dataset_name, spec in dataset_specs.items():
        for source_path in spec["paths"]:
            if not source_path.exists():
                raise FileNotFoundError(
                    f"{dataset_name} 缺少原始数据文件：{source_path}"
                )
            if not source_path.is_file() or source_path.stat().st_size == 0:
                raise ValueError(
                    f"{dataset_name} 原始数据文件无效或为空：{source_path}"
                )


def count_table_interactions(
    paths: list[Path],
    user_column: str | int,
    header: str | None,
    separator: str = ",",
) -> Counter[int]:
    """按用户统计一组分隔符表格中的交互行数。

    Args:
        paths (list[Path]): 同一数据集的原始交互文件列表。
        user_column (str | int): 用户 ID 的列名或零基列下标。
        header (str | None): 传递给 pandas 的表头设置，通常为 ``"infer"``
            或 ``None``。
        separator (str): 字段分隔符，默认为逗号。

    Returns:
        Counter[int]: 用户 ID 到完整交互次数的映射。

    Raises:
        ValueError: 当文件不含有效交互行或用户 ID 为空时抛出。
        pandas.errors.ParserError: 当原始文件格式损坏、无法解析时抛出。
    """
    interaction_counts: Counter[int] = Counter()

    for source_path in paths:
        LOGGER.info("读取表格型原始数据：%s", source_path)
        reader = pd.read_csv(
            source_path,
            sep=separator,
            header=header,
            usecols=[user_column],
            chunksize=DEFAULT_CHUNK_SIZE,
        )
        file_rows = 0
        for chunk in reader:
            user_series = chunk.iloc[:, 0]
            if user_series.isna().any():
                raise ValueError(f"用户 ID 存在空值：{source_path}")
            value_counts = user_series.value_counts(sort=False)
            interaction_counts.update(
                {
                    int(user_id): int(count)
                    for user_id, count in value_counts.items()
                }
            )
            file_rows += len(chunk)
        if file_rows == 0:
            raise ValueError(f"原始交互文件不含数据行：{source_path}")
        LOGGER.info("完成 %s，共读取 %s 条交互", source_path.name, f"{file_rows:,}")

    return interaction_counts


def count_matrix_interactions(paths: list[Path]) -> Counter[int]:
    """统计评分矩阵中每位用户跨多个文件的非零交互数。

    Args:
        paths (list[Path]): 行表示用户、列表示物品的空格分隔评分矩阵列表；
            各文件必须具有相同形状和用户行顺序。

    Returns:
        Counter[int]: 零基用户行号到所有矩阵中非零评分总数的映射。

    Raises:
        ValueError: 当矩阵为空、不是二维结构、形状不一致，或存在完全无交互
            的用户时抛出。
    """
    combined_counts: np.ndarray | None = None
    expected_shape: tuple[int, int] | None = None

    for source_path in paths:
        LOGGER.info("读取矩阵型原始数据：%s", source_path)
        matrix = np.loadtxt(source_path, dtype=np.int16)
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError(f"评分矩阵必须为非空二维数组：{source_path}")
        if expected_shape is None:
            expected_shape = matrix.shape
        elif matrix.shape != expected_shape:
            raise ValueError(
                f"评分矩阵形状不一致：期望 {expected_shape}，实际 {matrix.shape}"
            )
        nonzero_counts = np.count_nonzero(matrix, axis=1).astype(np.int64)
        combined_counts = (
            nonzero_counts
            if combined_counts is None
            else combined_counts + nonzero_counts
        )
        LOGGER.info(
            "完成 %s，共识别 %s 条交互",
            source_path.name,
            f"{int(nonzero_counts.sum()):,}",
        )

    if combined_counts is None:
        raise ValueError("评分矩阵文件列表不能为空。")
    if np.any(combined_counts <= 0):
        empty_users = np.flatnonzero(combined_counts <= 0).tolist()
        raise ValueError(f"存在没有任何交互的用户行：{empty_users[:10]}")

    return Counter(
        {
            int(user_id): int(length)
            for user_id, length in enumerate(combined_counts)
        }
    )


def collect_trajectory_lengths(
    dataset_specs: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """按配置收集所有数据集的用户轨迹长度。

    Args:
        dataset_specs (dict[str, dict[str, Any]]): 数据集名称到读取配置的映射。

    Returns:
        dict[str, np.ndarray]: 数据集名称到一维正整数轨迹长度数组的映射。

    Raises:
        ValueError: 当读取格式未知或统计结果为空、包含非正长度时抛出。
    """
    lengths_by_dataset: dict[str, np.ndarray] = {}

    for dataset_name, spec in dataset_specs.items():
        LOGGER.info("开始统计数据集：%s", dataset_name)
        if spec["format"] == "table":
            counts = count_table_interactions(
                paths=spec["paths"],
                user_column=spec["user_column"],
                header=spec["header"],
                separator=spec.get("separator", ","),
            )
        elif spec["format"] == "matrix":
            counts = count_matrix_interactions(spec["paths"])
        else:
            raise ValueError(
                f"{dataset_name} 使用了未知读取格式：{spec['format']}"
            )

        trajectory_lengths = np.fromiter(
            counts.values(), dtype=np.int64, count=len(counts)
        )
        if trajectory_lengths.size == 0 or np.any(trajectory_lengths <= 0):
            raise ValueError(f"{dataset_name} 的轨迹长度统计结果为空或包含非正值。")
        lengths_by_dataset[dataset_name] = trajectory_lengths
        LOGGER.info(
            "%s：%s 位用户，%s 条交互",
            dataset_name,
            f"{trajectory_lengths.size:,}",
            f"{int(trajectory_lengths.sum()):,}",
        )

    return lengths_by_dataset


def build_summary(lengths_by_dataset: dict[str, np.ndarray]) -> pd.DataFrame:
    """计算各数据集轨迹长度的描述统计。

    Args:
        lengths_by_dataset (dict[str, np.ndarray]): 数据集到轨迹长度数组的映射。

    Returns:
        pandas.DataFrame: 每行一个数据集，包含用户数、交互数、均值及多个分位数。

    Raises:
        ValueError: 当输入映射为空或任一轨迹长度数组为空时抛出。
    """
    if not lengths_by_dataset:
        raise ValueError("轨迹长度统计结果不能为空。")

    summary_rows: list[dict[str, float | int | str]] = []
    for dataset_name, lengths in lengths_by_dataset.items():
        if lengths.size == 0:
            raise ValueError(f"{dataset_name} 不含可汇总的轨迹。")
        quantiles = np.percentile(lengths, [25, 50, 75, 90, 95, 99])
        summary_rows.append(
            {
                "dataset": dataset_name,
                "users": int(lengths.size),
                "interactions": int(lengths.sum()),
                "min": int(lengths.min()),
                "p25": float(quantiles[0]),
                "median": float(quantiles[1]),
                "mean": float(lengths.mean()),
                "p75": float(quantiles[2]),
                "p90": float(quantiles[3]),
                "p95": float(quantiles[4]),
                "p99": float(quantiles[5]),
                "max": int(lengths.max()),
            }
        )

    return pd.DataFrame(summary_rows)


def save_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """将描述统计保存为 CSV 文件。

    Args:
        summary (pandas.DataFrame): ``build_summary`` 生成的描述统计表。
        output_path (Path): CSV 输出路径，父目录会按需创建。

    Returns:
        None: 文件保存完成后正常返回。

    Raises:
        OSError: 当输出目录无法创建或文件无法写入时抛出。
        ValueError: 当汇总表为空时抛出。
    """
    if summary.empty:
        raise ValueError("不能保存空的描述统计表。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, float_format="%.3f")
    LOGGER.info("描述统计已保存至：%s", output_path)


def make_log_grid(lengths_by_dataset: dict[str, np.ndarray]) -> np.ndarray:
    """依据全部数据集的长度范围生成共享 log10 网格边界。

    Args:
        lengths_by_dataset (dict[str, np.ndarray]): 数据集到正整数轨迹长度的映射。

    Returns:
        numpy.ndarray: 严格递增且覆盖全部长度的 log10 空间网格边界。

    Raises:
        ValueError: 当输入为空或任一轨迹长度小于 1 时抛出。
    """
    if not lengths_by_dataset:
        raise ValueError("轨迹长度统计结果不能为空。")
    minimum_length = min(int(lengths.min()) for lengths in lengths_by_dataset.values())
    maximum_length = max(int(lengths.max()) for lengths in lengths_by_dataset.values())
    if minimum_length < 1:
        raise ValueError("所有轨迹长度都必须至少为 1。")

    log_minimum = np.log10(minimum_length) - LOG_RANGE_PADDING
    log_maximum = np.log10(maximum_length) + LOG_RANGE_PADDING
    return np.linspace(
        log_minimum,
        log_maximum,
        DEFAULT_DENSITY_GRID_SIZE + 1,
    )


def estimate_log_density(
    lengths: np.ndarray, log_grid_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """在 log10 轨迹长度空间估计平滑概率密度。

    该函数先在共享细网格上计算概率密度直方图，再使用 Silverman
    经验带宽进行高斯平滑。固定长度数据使用最小带宽形成窄峰，避免
    核密度估计因零方差失效。

    Args:
        lengths (numpy.ndarray): 一维正整数轨迹长度数组，至少包含一个元素。
        log_grid_edges (numpy.ndarray): 严格递增的 log10 空间网格边界。

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: 原始长度尺度的网格中心，以及对应的
        log10 空间平滑概率密度；密度积分归一化为 1。

    Raises:
        ValueError: 当长度数组为空、包含非正值，或网格边界不合法时抛出。
    """
    if lengths.ndim != 1 or lengths.size == 0:
        raise ValueError("轨迹长度必须为非空一维数组。")
    if np.any(lengths <= 0):
        raise ValueError("轨迹长度必须全部为正数。")
    if log_grid_edges.ndim != 1 or log_grid_edges.size < 3:
        raise ValueError("log10 网格至少需要三个一维边界点。")
    grid_widths = np.diff(log_grid_edges)
    if np.any(grid_widths <= 0) or not np.allclose(grid_widths, grid_widths[0]):
        raise ValueError("log10 网格边界必须严格递增且等距。")

    log_lengths = np.log10(lengths.astype(np.float64))
    histogram, _ = np.histogram(log_lengths, bins=log_grid_edges, density=False)
    grid_width = float(grid_widths[0])
    histogram_density = histogram.astype(np.float64) / (lengths.size * grid_width)

    log_standard_deviation = (
        float(np.std(log_lengths, ddof=1)) if lengths.size > 1 else 0.0
    )
    silverman_bandwidth = (
        SILVERMAN_FACTOR
        * log_standard_deviation
        * float(lengths.size) ** (-1.0 / 5.0)
    )
    bandwidth = max(silverman_bandwidth, MIN_LOG_BANDWIDTH)
    smoothed_density = gaussian_filter1d(
        histogram_density,
        sigma=bandwidth / grid_width,
        mode="constant",
    )

    log_grid_centers = (log_grid_edges[:-1] + log_grid_edges[1:]) / 2.0
    density_area = float(np.trapezoid(smoothed_density, log_grid_centers))
    if density_area <= 0.0 or not np.isfinite(density_area):
        raise ValueError("平滑密度无法归一化，请检查轨迹长度输入。")
    smoothed_density /= density_area
    return np.power(10.0, log_grid_centers), smoothed_density


def plot_distributions(
    lengths_by_dataset: dict[str, np.ndarray], output_path: Path
) -> None:
    """绘制共享对数尺度下的平滑轨迹长度密度曲线。

    Args:
        lengths_by_dataset (dict[str, np.ndarray]): 数据集到一维轨迹长度数组的映射。
        output_path (Path): PNG 图片输出路径，父目录会按需创建。

    Returns:
        None: 图片保存完成后正常返回。

    Raises:
        ValueError: 当输入映射为空时抛出。
        OSError: 当输出目录无法创建或图片无法写入时抛出。
    """
    if not lengths_by_dataset:
        raise ValueError("没有可绘制的轨迹长度统计结果。")

    log_grid_edges = make_log_grid(lengths_by_dataset)
    figure, density_axis = plt.subplots(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#FAFAFC")
    density_axis.set_facecolor("#FFFFFF")

    for dataset_name, lengths in lengths_by_dataset.items():
        color = DATASET_COLORS[dataset_name]
        grid_centers, smoothed_density = estimate_log_density(
            lengths,
            log_grid_edges,
        )
        legend_label = (
            f"{dataset_name}  ·  n={lengths.size:,}  ·  "
            f"median={np.median(lengths):,.1f}"
        )
        density_axis.plot(
            grid_centers,
            smoothed_density,
            color=color,
            linewidth=2.6,
            label=legend_label,
        )
        density_axis.fill_between(
            grid_centers,
            smoothed_density,
            0.0,
            color=color,
            alpha=0.10,
        )

    density_axis.set_xscale("log")
    density_axis.set_ylim(bottom=0.0)
    density_axis.set_title(
        "User trajectory-length distributions",
        fontsize=17,
        fontweight="semibold",
        loc="left",
        pad=18,
    )
    density_axis.text(
        0.0,
        1.01,
        "Smoothed density in log10 space · complete raw trajectories",
        transform=density_axis.transAxes,
        color="#667085",
        fontsize=10.5,
        va="bottom",
    )
    density_axis.set_xlabel(
        "Trajectory length (interactions per user, log scale)",
        fontsize=11.5,
        labelpad=10,
    )
    density_axis.set_ylabel("Probability density in log10 space", fontsize=11.5)
    density_axis.grid(axis="x", which="both", color="#D0D5DD", alpha=0.45)
    density_axis.grid(axis="y", which="major", color="#EAECF0", alpha=0.7)
    density_axis.set_axisbelow(True)
    density_axis.spines["top"].set_visible(False)
    density_axis.spines["right"].set_visible(False)
    density_axis.spines["left"].set_color("#98A2B3")
    density_axis.spines["bottom"].set_color("#98A2B3")
    density_axis.tick_params(colors="#475467")
    density_axis.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#EAECF0",
        framealpha=0.92,
        fontsize=9.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("轨迹长度分布图已保存至：%s", output_path)


def main() -> None:
    """执行原始轨迹长度统计、汇总保存和分布图绘制。

    Returns:
        None: 所有输出成功生成后正常返回。

    Raises:
        FileNotFoundError: 当所需原始数据文件缺失时抛出。
        ValueError: 当输入数据结构或统计结果不合法时抛出。
        OSError: 当输出文件无法写入时抛出。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    dataset_specs = get_dataset_specs(args.data_root.resolve())
    validate_source_paths(dataset_specs)
    lengths_by_dataset = collect_trajectory_lengths(dataset_specs)
    summary = build_summary(lengths_by_dataset)
    save_summary(summary, args.summary_output.resolve())
    plot_distributions(lengths_by_dataset, args.output.resolve())
    LOGGER.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
