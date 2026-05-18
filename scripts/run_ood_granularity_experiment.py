"""OOD 识别粒度主实验的独立 CLI。"""

import argparse
import json
import os
import sys
from typing import Any, Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

sys.path.extend([".", "./src", "./examples/policy", "./src/DeepCTR-Torch", "./src/tianshou"])

from analysis.common import load_config
from analysis.ood_granularity_experiment import configure_logging, run_ood_granularity_experiment


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Args:
        无: 参数直接来自命令行。

    Returns:
        argparse.Namespace: 解析后的命令行对象。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    parser = argparse.ArgumentParser("Run OOD granularity experiment for MOPO uncertainty penalty")
    parser.add_argument("--config", type=str, default="configs/ood_granularity_experiment.yaml")
    parser.add_argument("--override", action="append", default=[], help="Override config entries with key=value")
    return parser.parse_args()


def apply_smoke_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """在 smoke 模式下覆写一组轻量默认值。

    Args:
        config (Dict[str, Any]): 原始配置字典。

    Returns:
        Dict[str, Any]: 覆写后的配置字典。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    if not bool(config.get("smoke_test", False)):
        return config

    smoke_config = dict(config)
    if smoke_config.get("output_dir") == "results/ood_granularity_experiment":
        smoke_config["output_dir"] = "results/ood_granularity_experiment_smoke"
    smoke_config["max_users"] = int(smoke_config.get("max_users", 0) or 8)
    smoke_config["anchor_states_per_user"] = min(int(smoke_config.get("anchor_states_per_user", 4)), 4)
    smoke_config["shortlist_top_rhat"] = min(int(smoke_config.get("shortlist_top_rhat", 64)), 24)
    smoke_config["shortlist_top_uncertainty"] = min(int(smoke_config.get("shortlist_top_uncertainty", 64)), 24)
    smoke_config["shortlist_top_true_high"] = min(int(smoke_config.get("shortlist_top_true_high", 64)), 24)
    smoke_config["shortlist_top_true_low"] = min(int(smoke_config.get("shortlist_top_true_low", 64)), 24)
    smoke_config["shortlist_top_leave"] = min(int(smoke_config.get("shortlist_top_leave", 32)), 12)
    smoke_config["shortlist_random"] = min(int(smoke_config.get("shortlist_random", 32)), 12)
    smoke_config["plot_sample_n"] = min(int(smoke_config.get("plot_sample_n", 120000)), 4000)
    return smoke_config


def main() -> None:
    """执行 CLI 主流程。

    Args:
        无: 参数直接来自命令行。

    Returns:
        None: 结果将写入配置指定的输出目录。

    Raises:
        FileNotFoundError: 当配置或输入数据文件缺失时抛出。
        ValueError: 当配置项非法或实验对齐失败时抛出。
    """

    configure_logging()
    cli_args = parse_args()
    config = load_config(cli_args.config, cli_args.override)
    config = apply_smoke_defaults(config)

    result = run_ood_granularity_experiment(config)
    print("========== OOD Granularity Experiment Finished ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
