"""基于缓存结果重绘 OOD 粒度实验图表。"""

import argparse
import json
import os
import sys
from typing import Any, Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import seaborn as sns

from analysis.ood_granularity_experiment import (
    compute_lambda_metrics,
    generate_markdown_report,
    plot_distance_vs_true_reward,
    plot_distance_vs_uncertainty,
    plot_highvar_composition,
    plot_lambda_response,
    plot_lambda_suppressed_mix,
    plot_uncertainty_bucket_mix,
    plot_uncertainty_violin,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 包含输出目录参数的命名空间。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    parser = argparse.ArgumentParser(description="Redraw cached OOD granularity figures.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ood_granularity_experiment",
        help="包含 sample_metrics/summary_metrics/label_counts 的实验输出目录。",
    )
    return parser.parse_args()


def load_json(json_path: str) -> Dict[str, Any]:
    """读取 JSON 文件。

    Args:
        json_path (str): JSON 文件路径。

    Returns:
        Dict[str, Any]: 解析后的字典对象。

    Raises:
        FileNotFoundError: 当文件不存在时抛出。
    """

    with open(json_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def main() -> None:
    """重绘缓存实验图表并刷新结果报告。

    Returns:
        None: 图表与结果报告直接写回输出目录。

    Raises:
        FileNotFoundError: 当关键缓存文件缺失时抛出。
    """

    args = parse_args()
    output_dir = args.output_dir
    figures_dir = os.path.join(output_dir, "figures")

    summary_payload = load_json(os.path.join(output_dir, "summary_metrics.json"))
    label_counts = load_json(os.path.join(output_dir, "label_counts.json"))
    sample_path = os.path.join(output_dir, "sample_metrics.csv.gz")
    sample_df = pd.read_csv(sample_path)
    config = summary_payload["config"]
    data_summary = summary_payload["data_summary"]
    discrimination_metrics = summary_payload["discrimination_metrics"]
    lambda_df = compute_lambda_metrics(
        sample_df=sample_df,
        lambda_values=[float(value) for value in config["lambda_grid"]],
        topk_list=[int(value) for value in config["topk_list"]],
    )

    figure_paths = {
        "distance_vs_true_reward": os.path.join(figures_dir, "fig_distance_vs_true_reward.png"),
        "distance_vs_uncertainty": os.path.join(figures_dir, "fig_distance_vs_uncertainty.png"),
        "uncertainty_violin": os.path.join(figures_dir, "fig_uncertainty_violin_quadrants.png"),
        "uncertainty_bucket_mix": os.path.join(figures_dir, "fig_uncertainty_bucket_mix.png"),
        "highvar_composition": os.path.join(figures_dir, "fig_highvar_composition.png"),
        "lambda_response": os.path.join(figures_dir, "fig_lambda_response_curves.png"),
        "lambda_suppressed_mix": os.path.join(figures_dir, "fig_lambda_suppressed_mix.png"),
    }

    sns.set_theme(style="whitegrid")
    plot_distance_vs_true_reward(
        sample_df=sample_df,
        output_path=figure_paths["distance_vs_true_reward"],
        plot_sample_n=int(config.get("plot_sample_n", 120000)),
        seed=int(config["seed"]),
    )
    plot_distance_vs_uncertainty(
        sample_df=sample_df,
        output_path=figure_paths["distance_vs_uncertainty"],
        plot_sample_n=int(config.get("plot_sample_n", 120000)),
        seed=int(config["seed"]),
    )
    plot_uncertainty_violin(
        sample_df=sample_df,
        output_path=figure_paths["uncertainty_violin"],
        clip_quantile=float(config.get("uncertainty_violin_clip_quantile", 0.95)),
    )
    plot_uncertainty_bucket_mix(
        sample_df=sample_df,
        output_path=figure_paths["uncertainty_bucket_mix"],
        bucket_count=int(config.get("uncertainty_bucket_count", 8)),
    )
    plot_highvar_composition(
        sample_df=sample_df,
        output_path=figure_paths["highvar_composition"],
        alpha_value=float(config.get("highvar_plot_alpha", 0.1)),
    )
    plot_lambda_response(
        lambda_df=lambda_df,
        output_path=figure_paths["lambda_response"],
        lambda_plot_topk=int(config.get("lambda_plot_topk", 20)),
    )
    plot_lambda_suppressed_mix(
        lambda_df=lambda_df,
        output_path=figure_paths["lambda_suppressed_mix"],
        lambda_plot_topk=int(config.get("lambda_plot_topk", 20)),
    )

    report_text = generate_markdown_report(
        config=config,
        data_summary=data_summary,
        discrimination_metrics=discrimination_metrics,
        lambda_df=lambda_df,
        label_counts=label_counts,
        figure_paths=figure_paths,
        sample_save_info={"path": sample_path, "format": "csv_gzip"},
        output_dir=output_dir,
    )
    with open(os.path.join(output_dir, "ood_granularity_report.md"), "w", encoding="utf-8") as file_obj:
        file_obj.write(report_text)

    summary_payload["lambda_metrics"] = lambda_df.to_dict(orient="records")
    with open(os.path.join(output_dir, "summary_metrics.json"), "w", encoding="utf-8") as file_obj:
        json.dump(summary_payload, file_obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
