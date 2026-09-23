"""汇总 RAVE-H 五随机种子消融，并计算相对主方法的配对差值。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_METRIC = "validation/mcd_macro"
"""默认汇总的验证集主指标。"""

DEFAULT_REFERENCE = "rave_multiobjective_adaptive"
"""配对比较的默认参考方法。"""

T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}
"""小样本双侧 95% t 区间临界值。"""


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--direction", choices=["maximize", "minimize"], default="maximize",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser


def read_manifest(path: Path) -> list[dict[str, str]]:
    """读取 launcher 生成的 TSV，并按条件与 seed 去重。"""

    if not path.is_file():
        raise FileNotFoundError(path)
    latest: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj, delimiter="\t"):
            latest[(row["condition"], row["seed"])] = row
    return list(latest.values())


def iter_metric_records(path: Path) -> Iterable[dict[str, float]]:
    """遍历 SwanLab JSONL 中的数值 metrics 字典。"""

    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            payload = json.loads(line)
            metrics = payload.get("metrics", {})
            if isinstance(metrics, dict):
                yield metrics


def last_metric(output_dir: Path, metric: str) -> float:
    """读取一次训练最后记录的指定指标。"""

    value: float | None = None
    metrics_path = output_dir / "swanlab_metrics.jsonl"
    for metrics in iter_metric_records(metrics_path):
        candidate = metrics.get(metric)
        if isinstance(candidate, (int, float)) and math.isfinite(float(candidate)):
            value = float(candidate)
    if value is None:
        raise KeyError(f"{metric!r} not found in {metrics_path}")
    return value


def confidence_interval(values: list[float]) -> tuple[float, float]:
    """返回均值的双侧 95% t 区间。"""

    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean
    std = statistics.stdev(values)
    critical = T_975.get(len(values) - 1, 1.96)
    radius = critical * std / math.sqrt(len(values))
    return mean - radius, mean + radius


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """写入统一列结构的 CSV。"""

    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """汇总每个条件，并对相同 seed 做配对比较。"""

    args = build_parser().parse_args()
    output_dir = args.output_dir or args.manifest.parent / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run: list[dict[str, object]] = []
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    failures: list[str] = []

    for row in read_manifest(args.manifest):
        condition = row["condition"]
        seed = int(row["seed"])
        try:
            value = last_metric(Path(row["output_dir"]), args.metric)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
            failures.append(f"{condition}/seed{seed}: {error}")
            continue
        grouped[condition][seed] = value
        per_run.append({"condition": condition, "seed": seed, "value": value})

    if failures:
        raise RuntimeError("Missing or invalid runs:\n" + "\n".join(failures))
    if args.reference not in grouped:
        raise KeyError(f"Reference condition not found: {args.reference}")

    summary_rows: list[dict[str, object]] = []
    for condition in sorted(grouped):
        values = list(grouped[condition].values())
        low, high = confidence_interval(values)
        summary_rows.append({
            "condition": condition,
            "n": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "ci95_low": low,
            "ci95_high": high,
        })

    reference = grouped[args.reference]
    paired_rows: list[dict[str, object]] = []
    sign = 1.0 if args.direction == "maximize" else -1.0
    for condition in sorted(grouped):
        if condition == args.reference:
            continue
        common_seeds = sorted(set(reference) & set(grouped[condition]))
        deltas = [
            sign * (reference[seed] - grouped[condition][seed])
            for seed in common_seeds
        ]
        low, high = confidence_interval(deltas)
        paired_rows.append({
            "reference": args.reference,
            "condition": condition,
            "n": len(deltas),
            "mean_reference_advantage": statistics.fmean(deltas),
            "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "ci95_low": low,
            "ci95_high": high,
            "reference_wins": sum(delta > 0 for delta in deltas),
        })

    write_csv(output_dir / "per_run.csv", sorted(
        per_run, key=lambda row: (str(row["condition"]), int(row["seed"])),
    ))
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "paired_vs_reference.csv", paired_rows)
    print(f"Wrote ablation summaries to {output_dir}")


if __name__ == "__main__":
    main()
