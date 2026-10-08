"""绘制完整未来轨迹上的回报与 MCD 重规划优势双子图。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_SAMPLES = 2000
"""episode-level bootstrap 默认重采样次数。"""

DEFAULT_CONFIDENCE_LEVEL = 0.95
"""图中 bootstrap 置信区间的默认置信水平。"""

FIGURE_WIDTH = 11.0
"""回报与 MCD 双子图默认宽度，单位为英寸。"""

FIGURE_HEIGHT = 4.2
"""回报与 MCD 双子图默认高度，单位为英寸。"""

K_COLORS = ("#2F6BFF", "#F28E2B", "#2CA02C", "#D62728", "#9467BD")
"""不同 chunk size 曲线使用的色盲友好颜色。"""


def load_records(records_path: Path) -> list[dict[str, float | int]]:
    """读取配对反事实原始 CSV 并完成类型转换。

    Args:
        records_path (Path): runner 生成的 `paired_counterfactual_records.csv`。

    Returns:
        list[dict[str, float | int]]: 数值化后的记录列表。

    Raises:
        FileNotFoundError: 当 CSV 不存在时抛出。
        ValueError: 当 CSV 为空或缺少绘图字段时抛出。
    """

    if not records_path.is_file():
        raise FileNotFoundError(f"Records CSV does not exist: {records_path}")
    integer_fields = {
        "model_seed",
        "chunk_size",
        "episode_id",
        "user_id",
        "chunk_index",
        "chunk_position",
        "remaining_steps",
        "delay_steps",
        "immediate_local_executed_steps",
        "delayed_local_executed_steps",
        "immediate_future_executed_steps",
        "delayed_future_executed_steps",
        "immediate_local_terminated",
        "delayed_local_terminated",
        "immediate_terminated",
        "delayed_terminated",
        "return_noninferior_diversity_improved",
        "immediate_dominant_count",
        "immediate_exposure_count",
        "delayed_dominant_count",
        "delayed_exposure_count",
        "continue_dominant_count",
        "continue_exposure_count",
    }
    float_fields = {
        "immediate_local_reward",
        "delayed_local_reward",
        "immediate_future_return",
        "delayed_future_return",
        "long_term_replanning_advantage",
        "long_term_delay_cost",
        "short_term_replanning_gain",
        "short_term_delay_loss",
        "immediate_final_mcd",
        "delayed_final_mcd",
        "continue_final_mcd",
        "diversity_replanning_advantage",
        "diversity_delay_cost",
        "weighted_replanning_advantage",
    }
    records: list[dict[str, float | int]] = []
    with records_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = set(reader.fieldnames or [])
        required_fields = integer_fields | float_fields
        missing_fields = sorted(required_fields - fieldnames)
        if missing_fields:
            raise ValueError(
                f"Records CSV is missing required fields: {missing_fields}."
            )
        for row in reader:
            converted: dict[str, float | int] = {}
            for field_name in integer_fields:
                converted[field_name] = int(row[field_name])
            for field_name in float_fields:
                converted[field_name] = float(row[field_name])
            records.append(converted)
    if not records:
        raise ValueError(f"Records CSV is empty: {records_path}")
    return records


def bootstrap_mean_interval(
    values: Sequence[float],
    bootstrap_samples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """计算均值及 episode-level percentile bootstrap 置信区间。

    Args:
        values (Sequence[float]): 每个 episode 一个聚合值的样本。
        bootstrap_samples (int): bootstrap 重采样次数，必须大于 0。
        confidence_level (float): 置信水平，必须位于 `(0, 1)`。
        rng (np.random.Generator): 显式随机数生成器。

    Returns:
        tuple[float, float, float]: 样本均值、置信区间下界和上界。

    Raises:
        ValueError: 当输入为空或参数越界时抛出。

    Example:
        >>> rng = np.random.default_rng(0)
        >>> mean, lower, upper = bootstrap_mean_interval(
        ...     [1.0, 2.0, 3.0], 100, 0.95, rng
        ... )
        >>> lower <= mean <= upper
        True
    """

    values_array = np.asarray(values, dtype=np.float64)
    if values_array.size == 0:
        raise ValueError("values must not be empty.")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1).")
    mean = float(values_array.mean())
    if values_array.size == 1:
        return mean, mean, mean
    sampled_indices = rng.integers(
        low=0,
        high=values_array.size,
        size=(bootstrap_samples, values_array.size),
    )
    bootstrap_means = values_array[sampled_indices].mean(axis=1)
    tail_probability = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        bootstrap_means,
        [tail_probability, 1.0 - tail_probability],
    )
    return mean, float(lower), float(upper)


def _aggregate_episode_values(
    records: Sequence[dict[str, float | int]],
    value_field: str,
    group_fields: Sequence[str],
) -> dict[tuple[int, ...], list[float]]:
    """先按 episode 聚合，再返回绘图分组的 episode 样本。

    Args:
        records (Sequence[dict[str, float | int]]): 筛选后的原始记录。
        value_field (str): 待聚合指标字段。
        group_fields (Sequence[str]): 不含 episode id 的绘图分组字段。

    Returns:
        dict[tuple[int, ...], list[float]]: 分组到 episode 均值列表的映射。
    """

    episode_buckets: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for record in records:
        episode_key = tuple(
            [int(record[field_name]) for field_name in group_fields]
            + [int(record["model_seed"]), int(record["episode_id"])]
        )
        episode_buckets[episode_key].append(float(record[value_field]))

    grouped_values: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for episode_key, values in episode_buckets.items():
        grouped_values[episode_key[:-2]].append(float(np.mean(values)))
    return dict(grouped_values)


def _summarize_metric(
    records: Sequence[dict[str, float | int]],
    value_field: str,
    group_fields: Sequence[str],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int]]:
    """按训练 seed 与 episode 独立单元汇总一个指标。

    Args:
        records (Sequence[dict[str, float | int]]): 已筛选的状态或 delay 记录。
        value_field (str): 待汇总数值字段。
        group_fields (Sequence[str]): 曲线分组字段。
        bootstrap_samples (int): bootstrap 重采样次数。
        rng (np.random.Generator): 显式 bootstrap 随机数生成器。

    Returns:
        list[dict[str, float | int]]: 均值、双侧区间、单侧下界与样本量。
    """

    grouped_values = _aggregate_episode_values(records, value_field, group_fields)
    summaries: list[dict[str, float | int]] = []
    for group_key, episode_values in sorted(grouped_values.items()):
        mean, lower, upper = bootstrap_mean_interval(
            episode_values,
            bootstrap_samples=bootstrap_samples,
            confidence_level=DEFAULT_CONFIDENCE_LEVEL,
            rng=rng,
        )
        values_array = np.asarray(episode_values, dtype=np.float64)
        if len(values_array) == 1:
            lower_bound = mean
        else:
            indices = rng.integers(
                0,
                len(values_array),
                size=(bootstrap_samples, len(values_array)),
            )
            bootstrap_means = values_array[indices].mean(axis=1)
            lower_bound = float(
                np.quantile(bootstrap_means, 1.0 - DEFAULT_CONFIDENCE_LEVEL)
            )
        matching_records = [
            record
            for record in records
            if all(
                int(record[field_name]) == int(group_value)
                for field_name, group_value in zip(group_fields, group_key)
            )
        ]
        row: dict[str, float | int] = {
            field_name: int(group_value)
            for field_name, group_value in zip(group_fields, group_key)
        }
        row.update(
            {
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
                "one_sided_lcb": lower_bound,
                "episode_seed_unit_count": len(episode_values),
                "model_seed_count": len(
                    {int(record["model_seed"]) for record in matching_records}
                ),
                "state_count": len(matching_records),
                "positive_rate": float(
                    np.mean(
                        [float(record[value_field]) > 0.0 for record in matching_records]
                    )
                ),
            }
        )
        summaries.append(row)
    return summaries


def build_plot_summaries(
    records: Sequence[dict[str, float | int]],
    bootstrap_samples: int,
    seed: int,
    return_noninferiority_margin: float = 0.0,
    diversity_improvement_margin: float = 0.0,
) -> dict[str, Any]:
    """构造 LRA、DRA、delay 回报和最终 MCD 的统计摘要。

    每个状态只从完整 Continue 行读取一次 LRA/DRA。delay 曲线固定使用
    `chunk_position=1`，并以 `(model_seed, episode_id)` 为 bootstrap 单元。

    Args:
        records (Sequence[dict[str, float | int]]): schema v3 数值记录。
        bootstrap_samples (int): bootstrap 次数。
        seed (int): 汇总随机种子。
        return_noninferiority_margin (float): 回报非劣容忍量 epsilon_R。
        diversity_improvement_margin (float): 多样性改善阈值 epsilon_D。

    Returns:
        dict[str, Any]: 四条主曲线、补充代价、约束判定与最优 delay。

    Raises:
        ValueError: 当阈值为负或记录为空时抛出。
    """

    if not records:
        raise ValueError("records must not be empty.")
    if return_noninferiority_margin < 0.0:
        raise ValueError("return_noninferiority_margin must be non-negative.")
    if diversity_improvement_margin < 0.0:
        raise ValueError("diversity_improvement_margin must be non-negative.")
    rng = np.random.default_rng(seed)
    continue_records = [
        record
        for record in records
        if int(record["delay_steps"]) == int(record["remaining_steps"])
    ]
    first_position_records = [
        record for record in records if int(record["chunk_position"]) == 1
    ]
    lra = _summarize_metric(
        continue_records,
        "long_term_replanning_advantage",
        ("chunk_size", "chunk_position"),
        bootstrap_samples,
        rng,
    )
    dra = _summarize_metric(
        continue_records,
        "diversity_replanning_advantage",
        ("chunk_size", "chunk_position"),
        bootstrap_samples,
        rng,
    )
    delay_return = _summarize_metric(
        first_position_records,
        "delayed_future_return",
        ("chunk_size", "delay_steps"),
        bootstrap_samples,
        rng,
    )
    delay_mcd = _summarize_metric(
        first_position_records,
        "delayed_final_mcd",
        ("chunk_size", "delay_steps"),
        bootstrap_samples,
        rng,
    )
    return_cost = _summarize_metric(
        first_position_records,
        "long_term_delay_cost",
        ("chunk_size", "delay_steps"),
        bootstrap_samples,
        rng,
    )
    diversity_cost = _summarize_metric(
        first_position_records,
        "diversity_delay_cost",
        ("chunk_size", "delay_steps"),
        bootstrap_samples,
        rng,
    )

    decisions = []
    for lra_row in lra:
        matching_dra = next(
            row
            for row in dra
            if int(row["chunk_size"]) == int(lra_row["chunk_size"])
            and int(row["chunk_position"]) == int(lra_row["chunk_position"])
        )
        state_rows = [
            record
            for record in continue_records
            if int(record["chunk_size"]) == int(lra_row["chunk_size"])
            and int(record["chunk_position"]) == int(lra_row["chunk_position"])
        ]
        quadrants = {
            "lra_positive_dra_positive": 0,
            "lra_positive_dra_nonpositive": 0,
            "lra_nonpositive_dra_positive": 0,
            "lra_nonpositive_dra_nonpositive": 0,
        }
        for record in state_rows:
            lra_positive = float(record["long_term_replanning_advantage"]) > 0.0
            dra_positive = float(record["diversity_replanning_advantage"]) > 0.0
            key = (
                f"lra_{'positive' if lra_positive else 'nonpositive'}_"
                f"dra_{'positive' if dra_positive else 'nonpositive'}"
            )
            quadrants[key] += 1
        state_count = len(state_rows)
        decisions.append(
            {
                "chunk_size": int(lra_row["chunk_size"]),
                "chunk_position": int(lra_row["chunk_position"]),
                "return_noninferior_lcb": (
                    float(lra_row["one_sided_lcb"])
                    > -return_noninferiority_margin
                ),
                "diversity_improved_lcb": (
                    float(matching_dra["one_sided_lcb"])
                    > diversity_improvement_margin
                ),
                "constrained_replanning_supported": (
                    float(lra_row["one_sided_lcb"])
                    > -return_noninferiority_margin
                    and float(matching_dra["one_sided_lcb"])
                    > diversity_improvement_margin
                ),
                "state_constraint_rate": float(
                    np.mean(
                        [
                            int(record["return_noninferior_diversity_improved"])
                            for record in state_rows
                        ]
                    )
                ),
                "quadrant_rates": {
                    name: count / state_count
                    for name, count in quadrants.items()
                },
            }
        )

    delay_optima = []
    for chunk_size in sorted({int(row["chunk_size"]) for row in delay_return}):
        return_rows = [
            row for row in delay_return if int(row["chunk_size"]) == chunk_size
        ]
        mcd_rows = [row for row in delay_mcd if int(row["chunk_size"]) == chunk_size]
        best_return = max(float(row["mean"]) for row in return_rows)
        best_mcd = min(float(row["mean"]) for row in mcd_rows)
        return_delays = [
            int(row["delay_steps"])
            for row in return_rows
            if np.isclose(float(row["mean"]), best_return)
        ]
        diversity_delays = [
            int(row["delay_steps"])
            for row in mcd_rows
            if np.isclose(float(row["mean"]), best_mcd)
        ]
        delay_optima.append(
            {
                "chunk_size": chunk_size,
                "return_optimal_delays": return_delays,
                "diversity_optimal_delays": diversity_delays,
                "shared_optimum_exists": bool(
                    set(return_delays).intersection(diversity_delays)
                ),
            }
        )
    return {
        "long_term_replanning_advantage": lra,
        "diversity_replanning_advantage": dra,
        "delay_future_return": delay_return,
        "delay_final_mcd": delay_mcd,
        "long_term_delay_cost": return_cost,
        "diversity_delay_cost": diversity_cost,
        "constraint_decisions": decisions,
        "delay_optima": delay_optima,
    }


def _plot_curve(
    axis: Any,
    rows: Sequence[dict[str, float | int]],
    x_field: str,
    ylabel: str,
    title: str,
    zero_line: bool,
) -> None:
    """在指定坐标轴绘制按 K 分组的均值与 bootstrap 区间。

    Args:
        axis (Any): Matplotlib 坐标轴。
        rows (Sequence[dict[str, float | int]]): `_summarize_metric` 输出。
        x_field (str): 横坐标字段名。
        ylabel (str): 纵坐标标签。
        title (str): 子图标题。
        zero_line (bool): 是否绘制 y=0 参考线。

    Returns:
        None.
    """

    chunk_sizes = sorted({int(row["chunk_size"]) for row in rows})
    for color_index, chunk_size in enumerate(chunk_sizes):
        color = K_COLORS[color_index % len(K_COLORS)]
        chunk_rows = sorted(
            [row for row in rows if int(row["chunk_size"]) == chunk_size],
            key=lambda row: int(row[x_field]),
        )
        x_values = np.asarray([int(row[x_field]) for row in chunk_rows])
        means = np.asarray([float(row["mean"]) for row in chunk_rows])
        lower = np.asarray([float(row["ci_lower"]) for row in chunk_rows])
        upper = np.asarray([float(row["ci_upper"]) for row in chunk_rows])
        axis.plot(
            x_values,
            means,
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"K={chunk_size}",
        )
        axis.fill_between(x_values, lower, upper, color=color, alpha=0.16)
    if zero_line:
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel(
        "Executed position within chunk"
        if x_field == "chunk_position"
        else "Extra cached actions before replanning"
    )
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)


def _plot_summary_curves(
    summaries: dict[str, Any],
    output_base: Path,
) -> tuple[Path, Path]:
    """绘制并保存回报 LRA 与多样性 DRA 双子图。

    Args:
        summaries (dict[str, Any]): `build_plot_summaries` 生成的摘要。
        output_base (Path): 不含扩展名的输出路径。

    Returns:
        tuple[Path, Path]: PNG 与 PDF 路径。
    """

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        constrained_layout=True,
    )
    _plot_curve(
        axes[0],
        summaries["long_term_replanning_advantage"],
        "chunk_position",
        "Return replanning advantage (LRA)",
        "(a) Return: Replan vs Continue",
        True,
    )
    _plot_curve(
        axes[1],
        summaries["diversity_replanning_advantage"],
        "chunk_position",
        "MCD replanning advantage (DRA)",
        "(b) MCD: Replan vs Continue",
        True,
    )
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def plot_motivation_figure(
    records_path: Path,
    output_dir: Path,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 2023,
    return_noninferiority_margin: float = 0.0,
    diversity_improvement_margin: float = 0.0,
) -> tuple[tuple[Path, Path], Path]:
    """从 schema v3 原始记录生成统计摘要与 LRA/DRA 双子图。

    Args:
        records_path (Path): 配对反事实原始 CSV。
        output_dir (Path): 图与 JSON 摘要输出目录。
        bootstrap_samples (int): episode-level bootstrap 次数。
        seed (int): bootstrap 随机种子。
        return_noninferiority_margin (float): 回报非劣容忍量 epsilon_R。
        diversity_improvement_margin (float): 多样性改善阈值 epsilon_D。

    Returns:
        tuple[tuple[Path, Path], Path]: `(PNG/PDF 路径, summary JSON 路径)`。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(records_path)
    summaries = build_plot_summaries(
        records=records,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        return_noninferiority_margin=return_noninferiority_margin,
        diversity_improvement_margin=diversity_improvement_margin,
    )
    summary_path = output_dir / "motivation_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summaries, file_obj, indent=2, ensure_ascii=False)
    figure_paths = _plot_summary_curves(
        summaries=summaries,
        output_base=output_dir / "replanning_motivation",
    )
    LOGGER.info(
        "动机实验双目标图已生成：png=%s, pdf=%s, summary=%s",
        figure_paths[0],
        figure_paths[1],
        summary_path,
    )
    return figure_paths, summary_path


def build_parser() -> argparse.ArgumentParser:
    """构造独立重绘命令行解析器。

    Returns:
        argparse.ArgumentParser: 绘图参数解析器。
    """

    parser = argparse.ArgumentParser(
        description="Plot DORL-MAC replanning motivation results."
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument(
        "--return-noninferiority-margin",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--diversity-improvement-margin",
        type=float,
        default=0.0,
    )
    return parser


def main(argv: Optional[list[str]] = None) -> tuple[tuple[Path, Path], Path]:
    """执行独立重绘流程。

    Args:
        argv (Optional[list[str]]): 可选命令行参数。

    Returns:
        tuple[tuple[Path, Path], Path]: 图文件与摘要路径。
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_parser().parse_args(argv)
    return plot_motivation_figure(
        records_path=args.records,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        return_noninferiority_margin=args.return_noninferiority_margin,
        diversity_improvement_margin=args.diversity_improvement_margin,
    )


if __name__ == "__main__":
    main()
