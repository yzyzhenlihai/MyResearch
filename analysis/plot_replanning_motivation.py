"""绘制完整未来轨迹上的 chunk 失效与延迟重规划代价双子图。"""

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
"""双子图默认宽度，单位为英寸。"""

FIGURE_HEIGHT = 4.2
"""双子图默认高度，单位为英寸。"""

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
            + [int(record["episode_id"])]
        )
        episode_buckets[episode_key].append(float(record[value_field]))

    grouped_values: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for episode_key, values in episode_buckets.items():
        grouped_values[episode_key[:-1]].append(float(np.mean(values)))
    return dict(grouped_values)


def build_plot_summaries(
    records: Sequence[dict[str, float | int]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, list[dict[str, float | int]]]:
    """构造两个子图需要的均值、置信区间和样本量。

    左图只保留 `delay_steps == remaining_steps` 的完整 Continue 分支，
    避免同一状态的长期优势因多个 delay 行被重复计数。右图固定
    `chunk_position == 1`，保证同一 K 下不同 delay 使用同一批状态。

    Args:
        records (Sequence[dict[str, float | int]]): 数值化原始记录。
        bootstrap_samples (int): bootstrap 次数。
        seed (int): 汇总随机种子。

    Returns:
        dict[str, list[dict[str, float | int]]]: 长期重规划优势和长期
        延迟代价两组绘图摘要。
    """

    rng = np.random.default_rng(seed)
    continue_records = [
        record
        for record in records
        if int(record["delay_steps"]) == int(record["remaining_steps"])
    ]
    gain_episode_values = _aggregate_episode_values(
        records=continue_records,
        value_field="long_term_replanning_advantage",
        group_fields=("chunk_size", "chunk_position"),
    )
    gain_summaries: list[dict[str, float | int]] = []
    for (chunk_size, chunk_position), episode_values in sorted(
        gain_episode_values.items()
    ):
        mean, lower, upper = bootstrap_mean_interval(
            episode_values,
            bootstrap_samples=bootstrap_samples,
            confidence_level=DEFAULT_CONFIDENCE_LEVEL,
            rng=rng,
        )
        state_records = [
            record
            for record in continue_records
            if int(record["chunk_size"]) == chunk_size
            and int(record["chunk_position"]) == chunk_position
        ]
        gain_summaries.append(
            {
                "chunk_size": chunk_size,
                "chunk_position": chunk_position,
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
                "episode_count": len(episode_values),
                "state_count": len(state_records),
                "win_rate": float(
                    np.mean(
                        [
                            float(
                                record["long_term_replanning_advantage"]
                            )
                            > 0.0
                            for record in state_records
                        ]
                    )
                ),
            }
        )

    first_position_records = [
        record for record in records if int(record["chunk_position"]) == 1
    ]
    delay_episode_values = _aggregate_episode_values(
        records=first_position_records,
        value_field="long_term_delay_cost",
        group_fields=("chunk_size", "delay_steps"),
    )
    delay_summaries: list[dict[str, float | int]] = []
    for (chunk_size, delay_steps), episode_values in sorted(
        delay_episode_values.items()
    ):
        mean, lower, upper = bootstrap_mean_interval(
            episode_values,
            bootstrap_samples=bootstrap_samples,
            confidence_level=DEFAULT_CONFIDENCE_LEVEL,
            rng=rng,
        )
        state_count = sum(
            1
            for record in first_position_records
            if int(record["chunk_size"]) == chunk_size
            and int(record["delay_steps"]) == delay_steps
        )
        delay_summaries.append(
            {
                "chunk_size": chunk_size,
                "delay_steps": delay_steps,
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
                "episode_count": len(episode_values),
                "state_count": state_count,
            }
        )
    return {
        "long_term_replanning_advantage": gain_summaries,
        "long_term_delay_cost": delay_summaries,
    }


def _plot_summary_curves(
    summaries: dict[str, list[dict[str, float | int]]],
    output_base: Path,
) -> tuple[Path, Path]:
    """绘制并保存论文可用的双子图。

    Args:
        summaries (dict[str, list[dict[str, float | int]]]): 绘图摘要。
        output_base (Path): 不含扩展名的输出路径。

    Returns:
        tuple[Path, Path]: PNG 与 PDF 路径。
    """

    chunk_sizes = sorted(
        {
            int(record["chunk_size"])
            for record in summaries["long_term_replanning_advantage"]
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        constrained_layout=True,
    )
    gain_axis, delay_axis = axes

    for color_index, chunk_size in enumerate(chunk_sizes):
        color = K_COLORS[color_index % len(K_COLORS)]
        gain_rows = [
            row
            for row in summaries["long_term_replanning_advantage"]
            if int(row["chunk_size"]) == chunk_size
        ]
        positions = np.asarray(
            [int(row["chunk_position"]) for row in gain_rows],
            dtype=np.int64,
        )
        gain_means = np.asarray(
            [float(row["mean"]) for row in gain_rows],
            dtype=np.float64,
        )
        gain_lower = np.asarray(
            [float(row["ci_lower"]) for row in gain_rows],
            dtype=np.float64,
        )
        gain_upper = np.asarray(
            [float(row["ci_upper"]) for row in gain_rows],
            dtype=np.float64,
        )
        overall_rows = [
            row
            for row in gain_rows
            if int(row["state_count"]) > 0
        ]
        weighted_win_rate = sum(
            float(row["win_rate"]) * int(row["state_count"])
            for row in overall_rows
        ) / max(
            sum(int(row["state_count"]) for row in overall_rows),
            1,
        )
        gain_axis.plot(
            positions,
            gain_means,
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"K={chunk_size} (win {weighted_win_rate:.0%})",
        )
        gain_axis.fill_between(
            positions,
            gain_lower,
            gain_upper,
            color=color,
            alpha=0.16,
        )

        delay_rows = [
            row
            for row in summaries["long_term_delay_cost"]
            if int(row["chunk_size"]) == chunk_size
        ]
        delays = np.asarray(
            [int(row["delay_steps"]) for row in delay_rows],
            dtype=np.int64,
        )
        delay_means = np.asarray(
            [float(row["mean"]) for row in delay_rows],
            dtype=np.float64,
        )
        delay_lower = np.asarray(
            [float(row["ci_lower"]) for row in delay_rows],
            dtype=np.float64,
        )
        delay_upper = np.asarray(
            [float(row["ci_upper"]) for row in delay_rows],
            dtype=np.float64,
        )
        delay_axis.plot(
            delays,
            delay_means,
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"K={chunk_size}",
        )
        delay_axis.fill_between(
            delays,
            delay_lower,
            delay_upper,
            color=color,
            alpha=0.16,
        )

    gain_axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    gain_axis.set_title(
        "(a) Does continuing the cached chunk reduce long-term return?"
    )
    gain_axis.set_xlabel("Executed position within chunk")
    gain_axis.set_ylabel("Long-term replanning advantage")
    gain_axis.grid(alpha=0.22)
    gain_axis.legend(frameon=False)

    delay_axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    delay_axis.set_title(
        "(b) Is delayed replanning costly for long-term return?"
    )
    delay_axis.set_xlabel("Extra cached actions before replanning")
    delay_axis.set_ylabel("Long-term delay cost")
    delay_axis.grid(alpha=0.22)
    delay_axis.legend(frameon=False)

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
) -> tuple[tuple[Path, Path], Path]:
    """从原始记录生成统计摘要与双子图。

    Args:
        records_path (Path): 配对反事实原始 CSV。
        output_dir (Path): 图与 JSON 摘要输出目录。
        bootstrap_samples (int): episode-level bootstrap 次数。
        seed (int): bootstrap 随机种子。

    Returns:
        tuple[tuple[Path, Path], Path]: `(PNG/PDF 路径, summary JSON 路径)`。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(records_path)
    summaries = build_plot_summaries(
        records=records,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    summary_path = output_dir / "motivation_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summaries, file_obj, indent=2, ensure_ascii=False)
    figure_paths = _plot_summary_curves(
        summaries=summaries,
        output_base=output_dir / "replanning_motivation",
    )
    LOGGER.info(
        "动机实验双子图已生成：png=%s, pdf=%s, summary=%s",
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
    )


if __name__ == "__main__":
    main()
