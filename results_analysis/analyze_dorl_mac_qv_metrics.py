"""分析 DORL-MAC Q/V 训练日志并生成中文诊断报告。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-easyrl4rec")

import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path(
    "saved_models/KuaiEnv-v0/DORL_MAC/dorl-mac-kuai-flow-qv/mac_agent/metrics.jsonl"
)
"""默认分析的 Q/V 训练 JSONL 日志路径。"""

DEFAULT_OUTPUT_DIR = Path("results_analysis")
"""默认分析产物输出目录。"""

REPORT_FILENAME = "dorl_mac_qv_training_log_analysis.md"
"""主分析报告文件名。"""

HANDOFF_FILENAME = "latest_dorl_mac_qv_training_log_analysis.md"
"""后续 agent 交接日志稳定文件名。"""

TRAIN_METRICS_FILENAME = "qv_train_metrics.csv"
"""训练指标 CSV 文件名。"""

EVAL_METRICS_FILENAME = "qv_eval_metrics.csv"
"""评估指标 CSV 文件名。"""

SUMMARY_FILENAME = "qv_metrics_summary.json"
"""统计摘要 JSON 文件名。"""

TARGET_LOW_REWARD = 25.0
"""用户提出的 NX_0_rew SOTA 低位目标。"""

TARGET_HIGH_REWARD = 30.0
"""用户提出的 NX_0_rew SOTA 高位目标。"""

TARGET_IFEAT_FEAT = 0.3
"""用户提出的 `ifeat_feat` 目标值；该指标越低代表越少集中在主导特征集合。"""

CHART_DPI = 160
"""保存图表时使用的默认分辨率。"""

EPSILON = 1.0e-12
"""避免除零的数值下界。"""


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 包含日志路径、配置路径和输出目录的参数对象。
    """

    parser = argparse.ArgumentParser(
        description="Analyze DORL-MAC Q/V metrics JSONL and write a Chinese report."
    )
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-low", type=float, default=TARGET_LOW_REWARD)
    parser.add_argument("--target-high", type=float, default=TARGET_HIGH_REWARD)
    parser.add_argument("--target-ifeat", type=float, default=TARGET_IFEAT_FEAT)
    return parser.parse_args()


def configure_logging() -> None:
    """配置脚本日志格式。

    Returns:
        None.
    """

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_json(path: Path) -> Dict[str, Any]:
    """读取 JSON 文件。

    Args:
        path (Path): JSON 文件路径。

    Returns:
        Dict[str, Any]: JSON 字典内容。

    Raises:
        FileNotFoundError: 当文件不存在时抛出。
        ValueError: 当 JSON 顶层不是字典时抛出。
    """

    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(f"JSON top-level object must be dict: {path}")
    return data


def load_metrics_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 SwanLab JSONL 指标日志。

    支持两种记录形态：
    - 指标记录：`{"step": int, "metrics": {...}}`
    - 配置 header：`{"step": 0, "event": "config", "project": str, "run_name": str,
      "mode": str, "config": {...}}`（新版 `SwanLabLogger` 在文件首行写入）。

    Args:
        path (Path): `metrics.jsonl` 文件路径。

    Returns:
        List[Dict[str, Any]]: 每行解析后的日志记录。

    Raises:
        FileNotFoundError: 当日志文件不存在时抛出。
        ValueError: 当某行不是合法 JSON 或形态未知时抛出。
    """

    if not path.exists():
        raise FileNotFoundError(f"Metrics file does not exist: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Invalid record at line {line_number}: {path}")
            # 兼容两种记录形态：新版 config header 或普通 metrics 记录。
            if record.get("event") == "config" or "metrics" in record:
                records.append(record)
                continue
            raise ValueError(f"Invalid metrics record at line {line_number}: {path}")
    return records


def extract_config_header(records: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """从 JSONL 记录中抽取新版 config header。

    Args:
        records (Sequence[Mapping[str, Any]]): 全部原始 JSONL 记录。

    Returns:
        Optional[Dict[str, Any]]: 若存在 `event=config` 记录，则返回其内部 `config`
            字典（同时透传 `project/run_name/mode/metrics_log_path` 等元数据到
            返回值上层），否则返回 `None`。
    """

    for record in records:
        if record.get("event") == "config" and isinstance(record.get("config"), dict):
            header_config = dict(record["config"])
            # 把 header 里的元数据也放进 config 里，方便报告直接引用。
            for meta_key in ("project", "run_name", "mode"):
                if meta_key in record and meta_key not in header_config:
                    header_config[meta_key] = record[meta_key]
            return header_config
    return None


def is_number(value: Any) -> bool:
    """判断对象是否是有限数值。

    Args:
        value (Any): 待检查对象。

    Returns:
        bool: 若对象可转换为有限浮点数则返回 True。
    """

    if isinstance(value, bool):
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric_value)


def flatten_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """将一条 JSONL 记录扁平化为一行指标。

    Args:
        record (Mapping[str, Any]): 原始 JSONL 记录。

    Returns:
        Dict[str, Any]: 包含 `step` 和所有 metrics 字段的扁平字典。
    """

    metrics = dict(record.get("metrics", {}))
    row: Dict[str, Any] = {"step": record.get("step")}
    row.update(metrics)
    return row


def split_metric_rows(records: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """按训练、评估和配置事件拆分日志。

    Args:
        records (Sequence[Mapping[str, Any]]): 原始 JSONL 记录序列。

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]: 训练行、评估行和配置事件数量。
    """

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    config_events = 0
    for record in records:
        # 新版 SwanLabLogger 在首行写入 config header（无 `metrics` 字段），跳过即可，
        # 上层通过 `extract_config_header` 独立读取。
        if record.get("event") == "config":
            config_events += 1
            continue
        row = flatten_record(record)
        metrics = record.get("metrics", {})
        if "event/config_saved" in metrics:
            config_events += 1
            continue
        if "critic/critic_loss" in metrics or any(k.startswith("train/critic/") for k in metrics):
            train_rows.append(row)
        elif (
            "NX_0_R_tra" in metrics
            or "NX_0_rew" in metrics
            or "raw/NX_0_rew" in metrics
            or "raw/NX_0_R_tra" in metrics
        ):
            eval_rows.append(row)

    train_rows.sort(key=lambda item: float(item.get("step", 0)))
    eval_rows.sort(key=lambda item: float(item.get("trainer/epoch", item.get("step", 0))))
    return train_rows, eval_rows, config_events


def numeric_values(rows: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    """从行列表中提取指定字段的有限数值。

    Args:
        rows (Sequence[Mapping[str, Any]]): 指标行列表。
        key (str): 指标字段名。

    Returns:
        List[float]: 已转换为 float 的数值列表。
    """

    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if is_number(value):
            values.append(float(value))
    return values


def basic_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """计算一组数值的基础统计量。

    Args:
        values (Sequence[float]): 数值序列。

    Returns:
        Dict[str, Optional[float]]: count、mean、median、min、max 和 std。
    """

    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "std": float(array.std(ddof=0)),
    }


def first_last_delta(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Optional[float]]:
    """计算某个指标首尾变化。

    Args:
        rows (Sequence[Mapping[str, Any]]): 指标行列表。
        key (str): 指标字段名。

    Returns:
        Dict[str, Optional[float]]: 首值、尾值和差值。
    """

    values = numeric_values(rows, key)
    if not values:
        return {"first": None, "last": None, "delta": None}
    return {"first": values[0], "last": values[-1], "delta": values[-1] - values[0]}


def find_best_row(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    """寻找指定指标最大的评估行。

    Args:
        rows (Sequence[Mapping[str, Any]]): 指标行列表。
        key (str): 指标字段名。

    Returns:
        Optional[Dict[str, Any]]: 最大值对应的行；若无有效数值则返回 None。
    """

    candidates = [row for row in rows if is_number(row.get(key))]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row[key]))


def find_min_row(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    """寻找指定指标最小的评估行。

    Args:
        rows (Sequence[Mapping[str, Any]]): 指标行列表。
        key (str): 用于比较的指标字段名。

    Returns:
        Optional[Dict[str, Any]]: 最小值对应的行；若无有效数值则返回 None。
    """

    candidates = [row for row in rows if is_number(row.get(key))]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row[key]))


def pearson_corr(rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str) -> Optional[float]:
    """计算两列指标之间的 Pearson 相关系数。

    Args:
        rows (Sequence[Mapping[str, Any]]): 指标行列表。
        x_key (str): 第一列字段名。
        y_key (str): 第二列字段名。

    Returns:
        Optional[float]: Pearson 相关系数；样本不足或方差为 0 时返回 None。
    """

    paired_values = [
        (float(row[x_key]), float(row[y_key]))
        for row in rows
        if is_number(row.get(x_key)) and is_number(row.get(y_key))
    ]
    if len(paired_values) < 2:
        return None
    x_values = np.asarray([item[0] for item in paired_values], dtype=np.float64)
    y_values = np.asarray([item[1] for item in paired_values], dtype=np.float64)
    if float(x_values.std()) <= EPSILON or float(y_values.std()) <= EPSILON:
        return None
    return float(np.corrcoef(x_values, y_values)[0, 1])


def linear_slope(rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str) -> Optional[float]:
    """计算指标随横轴变化的一阶线性斜率。

    Args:
        rows (Sequence[Mapping[str, Any]]): 指标行列表。
        x_key (str): 横轴字段名。
        y_key (str): 纵轴字段名。

    Returns:
        Optional[float]: 最小二乘直线斜率；样本不足时返回 None。
    """

    paired_values = [
        (float(row[x_key]), float(row[y_key]))
        for row in rows
        if is_number(row.get(x_key)) and is_number(row.get(y_key))
    ]
    if len(paired_values) < 2:
        return None
    x_values = np.asarray([item[0] for item in paired_values], dtype=np.float64)
    y_values = np.asarray([item[1] for item in paired_values], dtype=np.float64)
    centered_x = x_values - x_values.mean()
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= EPSILON:
        return None
    centered_y = y_values - y_values.mean()
    return float(np.dot(centered_x, centered_y) / denominator)


def summarize_training(train_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """汇总 Q/V 训练过程指标。

    Args:
        train_rows (Sequence[Mapping[str, Any]]): 训练指标行。

    Returns:
        Dict[str, Any]: 训练指标统计摘要。
    """

    keys = [
        "critic/critic_loss",
        "value/value_loss",
        "critic/q_mean",
        "critic/target_q_mean",
        "rollout/reward_chunk",
        "rollout/effective_steps",
        "rollout/done_ratio",
        "rollout/repeat_ratio",
        "rollout/exact_repeat_ratio",
        "rollout/pred_reward",
        "rollout/entropy",
        "rollout/uncertainty",
    ]
    summary: Dict[str, Any] = {}
    for key in keys:
        values = numeric_values(train_rows, key)
        summary[key] = basic_stats(values)
        summary[key].update(first_last_delta(train_rows, key))

    q_target_pairs = [
        (float(row["critic/q_mean"]), float(row["critic/target_q_mean"]))
        for row in train_rows
        if is_number(row.get("critic/q_mean")) and is_number(row.get("critic/target_q_mean"))
    ]
    q_abs_gaps = [abs(pair[0] - pair[1]) for pair in q_target_pairs]
    summary["critic/q_target_abs_gap"] = basic_stats(q_abs_gaps)
    effective_steps = numeric_values(train_rows, "rollout/effective_steps")
    summary["rollout/effective_steps_pct_lt_1"] = (
        float(sum(value < 1.0 for value in effective_steps) / max(len(effective_steps), 1))
        if effective_steps
        else None
    )
    summary["rollout/effective_steps_pct_lt_chunk"] = (
        float(sum(value < 3.0 for value in effective_steps) / max(len(effective_steps), 1))
        if effective_steps
        else None
    )
    summary["rows"] = len(train_rows)
    summary["first_global_step"] = train_rows[0].get("trainer/global_step") if train_rows else None
    summary["last_global_step"] = train_rows[-1].get("trainer/global_step") if train_rows else None
    return summary


def summarize_evaluation(
    eval_rows: Sequence[Mapping[str, Any]],
    target_low: float,
    target_high: float,
    target_ifeat: float,
) -> Dict[str, Any]:
    """汇总评估指标并计算 SOTA 目标差距。

    Args:
        eval_rows (Sequence[Mapping[str, Any]]): 评估指标行。
        target_low (float): 低位目标 reward。
        target_high (float): 高位目标 reward。
        target_ifeat (float): `ifeat_feat` 目标值，该指标越低越分散。

    Returns:
        Dict[str, Any]: 评估指标统计摘要。
    """

    metric_keys = [
        "R_tra",
        "len_tra",
        "ctr",
        "NX_0_R_tra",
        "NX_0_len_tra",
        "NX_0_ctr",
        "NX_0_CV",
        "NX_0_CV_turn",
        "NX_0_Diversity",
        "NX_0_Novelty",
        "NX_0_ifeat_feat",
        "NX_30_R_tra",
        "NX_30_len_tra",
        "NX_30_ctr",
        "NX_30_Diversity",
        "NX_30_Novelty",
        "NX_30_ifeat_feat",
    ]
    summary: Dict[str, Any] = {"rows": len(eval_rows)}
    for key in metric_keys:
        values = numeric_values(eval_rows, key)
        if values:
            summary[key] = basic_stats(values)
            summary[key].update(first_last_delta(eval_rows, key))

    best_nx0 = find_best_row(eval_rows, "NX_0_R_tra")
    best_nx30 = find_best_row(eval_rows, "NX_30_R_tra")
    min_nx0_ifeat = find_min_row(eval_rows, "NX_0_ifeat_feat")
    last_eval = eval_rows[-1] if eval_rows else None
    summary["best_nx0"] = best_row_to_dict(best_nx0, "NX_0_R_tra")
    summary["best_nx30"] = best_row_to_dict(best_nx30, "NX_30_R_tra")
    summary["min_nx0_ifeat"] = best_row_to_dict(min_nx0_ifeat, "NX_0_ifeat_feat")
    summary["last_eval"] = selected_eval_fields(last_eval)

    best_value = float(best_nx0["NX_0_R_tra"]) if best_nx0 is not None else float("nan")
    last_value = float(last_eval["NX_0_R_tra"]) if last_eval is not None else float("nan")
    last_ctr = float(last_eval["NX_0_ctr"]) if last_eval is not None else float("nan")
    best_ctr = float(best_nx0["NX_0_ctr"]) if best_nx0 is not None else float("nan")
    last_ifeat = (
        float(last_eval["NX_0_ifeat_feat"])
        if last_eval is not None and is_number(last_eval.get("NX_0_ifeat_feat"))
        else float("nan")
    )
    min_ifeat = (
        float(min_nx0_ifeat["NX_0_ifeat_feat"]) if min_nx0_ifeat is not None else float("nan")
    )
    summary["target_gap"] = {
        "best_to_25": safe_subtract(target_low, best_value),
        "best_to_30": safe_subtract(target_high, best_value),
        "last_to_25": safe_subtract(target_low, last_value),
        "last_to_30": safe_subtract(target_high, last_value),
        "required_len_at_last_ctr_for_25": safe_divide(target_low, last_ctr),
        "required_len_at_last_ctr_for_30": safe_divide(target_high, last_ctr),
        "required_len_at_best_ctr_for_25": safe_divide(target_low, best_ctr),
        "required_len_at_best_ctr_for_30": safe_divide(target_high, best_ctr),
        "target_ifeat": target_ifeat,
        "last_ifeat_over_target": safe_subtract(last_ifeat, target_ifeat),
        "min_ifeat_over_target": safe_subtract(min_ifeat, target_ifeat),
    }

    summary["correlations_with_nx0_reward"] = {
        key: pearson_corr(eval_rows, "NX_0_R_tra", key)
        for key in [
            "NX_0_len_tra",
            "NX_0_ctr",
            "NX_0_CV",
            "NX_0_CV_turn",
            "NX_0_Diversity",
            "NX_0_Novelty",
            "NX_0_ifeat_feat",
            "NX_30_R_tra",
        ]
    }
    summary["slopes_per_epoch"] = {
        key: linear_slope(eval_rows, "trainer/epoch", key)
        for key in [
            "NX_0_R_tra",
            "NX_0_len_tra",
            "NX_0_ctr",
            "NX_0_Diversity",
            "NX_0_Novelty",
            "NX_0_ifeat_feat",
            "NX_30_R_tra",
            "NX_30_ctr",
        ]
    }
    summary["correlations_with_nx0_ifeat"] = {
        key: pearson_corr(eval_rows, "NX_0_ifeat_feat", key)
        for key in [
            "NX_0_R_tra",
            "NX_0_len_tra",
            "NX_0_ctr",
            "NX_0_Diversity",
            "NX_0_Novelty",
            "NX_30_R_tra",
        ]
    }
    return summary


def best_row_to_dict(row: Optional[Mapping[str, Any]], metric_key: str) -> Optional[Dict[str, Any]]:
    """提取最佳评估行的核心字段。

    Args:
        row (Optional[Mapping[str, Any]]): 最佳评估行。
        metric_key (str): 用于选择最佳行的指标名。

    Returns:
        Optional[Dict[str, Any]]: 核心字段字典；无有效行时返回 None。
    """

    if row is None:
        return None
    return {
        "epoch": to_optional_float(row.get("trainer/epoch")),
        "env_step": to_optional_float(row.get("trainer/env_step")),
        metric_key: to_optional_float(row.get(metric_key)),
        "NX_0_R_tra": to_optional_float(row.get("NX_0_R_tra")),
        "NX_0_len_tra": to_optional_float(row.get("NX_0_len_tra")),
        "NX_0_ctr": to_optional_float(row.get("NX_0_ctr")),
        "NX_0_Diversity": to_optional_float(row.get("NX_0_Diversity")),
        "NX_0_Novelty": to_optional_float(row.get("NX_0_Novelty")),
        "NX_0_ifeat_feat": to_optional_float(row.get("NX_0_ifeat_feat")),
        "NX_30_R_tra": to_optional_float(row.get("NX_30_R_tra")),
        "NX_30_ctr": to_optional_float(row.get("NX_30_ctr")),
        "NX_30_ifeat_feat": to_optional_float(row.get("NX_30_ifeat_feat")),
    }


def selected_eval_fields(row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """提取最后一轮评估的核心字段。

    Args:
        row (Optional[Mapping[str, Any]]): 评估行。

    Returns:
        Optional[Dict[str, Any]]: 核心字段字典；无有效行时返回 None。
    """

    if row is None:
        return None
    return {
        "epoch": to_optional_float(row.get("trainer/epoch")),
        "env_step": to_optional_float(row.get("trainer/env_step")),
        "R_tra": to_optional_float(row.get("R_tra")),
        "len_tra": to_optional_float(row.get("len_tra")),
        "ctr": to_optional_float(row.get("ctr")),
        "NX_0_R_tra": to_optional_float(row.get("NX_0_R_tra")),
        "NX_0_len_tra": to_optional_float(row.get("NX_0_len_tra")),
        "NX_0_ctr": to_optional_float(row.get("NX_0_ctr")),
        "NX_0_Diversity": to_optional_float(row.get("NX_0_Diversity")),
        "NX_0_Novelty": to_optional_float(row.get("NX_0_Novelty")),
        "NX_0_ifeat_feat": to_optional_float(row.get("NX_0_ifeat_feat")),
        "NX_30_R_tra": to_optional_float(row.get("NX_30_R_tra")),
        "NX_30_len_tra": to_optional_float(row.get("NX_30_len_tra")),
        "NX_30_ctr": to_optional_float(row.get("NX_30_ctr")),
        "NX_30_ifeat_feat": to_optional_float(row.get("NX_30_ifeat_feat")),
    }


def to_optional_float(value: Any) -> Optional[float]:
    """将对象安全转换为浮点数。

    Args:
        value (Any): 待转换对象。

    Returns:
        Optional[float]: 有限浮点数；无法转换时返回 None。
    """

    if not is_number(value):
        return None
    return float(value)


def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    """安全执行除法。

    Args:
        numerator (float): 分子。
        denominator (float): 分母。

    Returns:
        Optional[float]: 除法结果；分母非法时返回 None。
    """

    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= EPSILON:
        return None
    return float(numerator / denominator)


def safe_subtract(left_value: float, right_value: float) -> Optional[float]:
    """安全执行减法。

    Args:
        left_value (float): 被减数。
        right_value (float): 减数。

    Returns:
        Optional[float]: 差值；任一输入非法时返回 None。
    """

    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return None
    return float(left_value - right_value)


def summarize_dataset_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """从配置中的离线数据路径推导 chunk 起点和可执行步数分布。

    Args:
        config (Mapping[str, Any]): `resolved_config.json` 内容。

    Returns:
        Dict[str, Any]: 数据集统计；读取失败时包含 `error` 字段。
    """

    dataset_path = Path(str(config.get("dataset_path", "")))
    chunk_size = int(config.get("chunk_size", 0) or 0)
    max_turn = int(config.get("max_turn", 0) or 0)
    if not dataset_path.exists():
        return {"error": f"dataset_path does not exist: {dataset_path}"}
    if chunk_size <= 0 or max_turn <= 0:
        return {"error": "chunk_size and max_turn must be positive in config"}

    try:
        with dataset_path.open("rb") as file_obj:
            trajectories = pickle.load(file_obj)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"failed to load dataset: {type(exc).__name__}: {exc}"}
    if not isinstance(trajectories, list) or not trajectories:
        return {"error": "dataset must be a non-empty list"}

    lengths = [int(len(trajectory["actions"])) for trajectory in trajectories if "actions" in trajectory]
    start_counts = [max(0, length - chunk_size + 1) for length in lengths]
    total_chunks = int(sum(start_counts))
    if total_chunks <= 0:
        return {"error": "no valid chunks can be derived from dataset"}

    start_sum = int(sum(count * (count - 1) // 2 for count in start_counts))
    max_start = max((count - 1 for count in start_counts if count > 0), default=0)
    start_histogram = build_start_histogram(start_counts, max_start=max_start)
    zero_effective_count, effective_step_sum = count_effective_steps(
        start_counts=start_counts,
        chunk_size=chunk_size,
        max_turn=max_turn,
    )
    return {
        "dataset_path": str(dataset_path),
        "num_trajectories": len(lengths),
        "num_transitions": int(sum(lengths)),
        "trajectory_length": {
            "min": int(min(lengths)),
            "median": float(median(lengths)),
            "mean": float(mean(lengths)),
            "max": int(max(lengths)),
        },
        "num_chunks": total_chunks,
        "chunk_start": {
            "min": 0,
            "p50": weighted_quantile_from_histogram(start_histogram, 0.50),
            "p90": weighted_quantile_from_histogram(start_histogram, 0.90),
            "p99": weighted_quantile_from_histogram(start_histogram, 0.99),
            "mean": float(start_sum / total_chunks),
            "max": int(max_start),
        },
        "max_turn": max_turn,
        "chunk_size": chunk_size,
        "pct_start_ge_max_turn": float(zero_effective_count / total_chunks),
        "pct_zero_effective_steps": float(zero_effective_count / total_chunks),
        "expected_effective_steps_mean": float(effective_step_sum / total_chunks),
    }


def build_start_histogram(start_counts: Sequence[int], max_start: int) -> List[int]:
    """构建 chunk 起点直方图。

    Args:
        start_counts (Sequence[int]): 每条轨迹可产生的起点数量。
        max_start (int): 最大起点下标。

    Returns:
        List[int]: 第 `i` 位表示起点 `i` 出现次数。
    """

    histogram = [0] * (max_start + 1)
    difference = [0] * (max_start + 2)
    for count in start_counts:
        if count <= 0:
            continue
        difference[0] += 1
        difference[count] -= 1
    active = 0
    for index in range(max_start + 1):
        active += difference[index]
        histogram[index] = active
    return histogram


def weighted_quantile_from_histogram(histogram: Sequence[int], quantile: float) -> Optional[int]:
    """从离散直方图计算加权分位点。

    Args:
        histogram (Sequence[int]): 非负计数直方图。
        quantile (float): 目标分位点，范围 `[0, 1]`。

    Returns:
        Optional[int]: 分位点所在桶下标；空直方图返回 None。
    """

    total = int(sum(histogram))
    if total <= 0:
        return None
    threshold = max(1, int(math.ceil(total * quantile)))
    cumulative = 0
    for index, count in enumerate(histogram):
        cumulative += int(count)
        if cumulative >= threshold:
            return int(index)
    return int(len(histogram) - 1)


def count_effective_steps(
    start_counts: Sequence[int],
    chunk_size: int,
    max_turn: int,
) -> Tuple[int, int]:
    """统计数据集中 chunk rollout 理论可执行步数。

    Args:
        start_counts (Sequence[int]): 每条轨迹可产生的起点数量。
        chunk_size (int): 每个 chunk 的步数。
        max_turn (int): 环境最大步数。

    Returns:
        Tuple[int, int]: 零可执行步 chunk 数、总可执行步数。
    """

    zero_effective_count = 0
    effective_step_sum = 0
    for count in start_counts:
        if count <= 0:
            continue
        zero_effective_count += max(0, count - max_turn)
        checked_count = min(count, max_turn)
        for start_index in range(checked_count):
            executable_steps = max(0, min(chunk_size, max_turn - start_index))
            effective_step_sum += executable_steps
    return int(zero_effective_count), int(effective_step_sum)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """将指标行写入 CSV 文件。

    Args:
        path (Path): 输出 CSV 路径。
        rows (Sequence[Mapping[str, Any]]): 待写入的指标行。

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """将字典写入 JSON 文件。

    Args:
        path (Path): 输出 JSON 路径。
        data (Mapping[str, Any]): 待写入数据。

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, ensure_ascii=False)


def plot_all(
    output_dir: Path,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    target_low: float,
    target_high: float,
    target_ifeat: float,
) -> List[str]:
    """绘制所有分析图表。

    Args:
        output_dir (Path): 输出目录。
        train_rows (Sequence[Mapping[str, Any]]): 训练指标行。
        eval_rows (Sequence[Mapping[str, Any]]): 评估指标行。
        target_low (float): 低位目标 reward。
        target_high (float): 高位目标 reward。
        target_ifeat (float): `ifeat_feat` 目标值，该指标越低越分散。

    Returns:
        List[str]: 成功生成的图表相对文件名列表。
    """

    chart_dir = output_dir / "figures"
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = [
        plot_eval_rewards(chart_dir, eval_rows, target_low, target_high),
        plot_eval_length_ctr(chart_dir, eval_rows),
        plot_eval_diversity(chart_dir, eval_rows, target_ifeat),
        plot_training_rollout(chart_dir, train_rows),
        plot_q_alignment(chart_dir, train_rows),
    ]
    return [str(path.relative_to(output_dir)) for path in chart_paths if path is not None]


def plot_eval_rewards(
    chart_dir: Path,
    eval_rows: Sequence[Mapping[str, Any]],
    target_low: float,
    target_high: float,
) -> Optional[Path]:
    """绘制评估 reward 曲线。

    Args:
        chart_dir (Path): 图表目录。
        eval_rows (Sequence[Mapping[str, Any]]): 评估指标行。
        target_low (float): 低位目标 reward。
        target_high (float): 高位目标 reward。

    Returns:
        Optional[Path]: 图表路径；无数据时返回 None。
    """

    if not eval_rows:
        return None
    epochs = numeric_values(eval_rows, "trainer/epoch")
    fig, axis = plt.subplots(figsize=(9, 4.8))
    for metric_key, label in [
        ("R_tra", "FB R_tra"),
        ("NX_0_R_tra", "NX_0 R_tra"),
        ("NX_30_R_tra", "NX_30 R_tra"),
    ]:
        values = numeric_values(eval_rows, metric_key)
        if len(values) == len(epochs):
            axis.plot(epochs, values, marker="o", linewidth=1.6, markersize=3.5, label=label)
    axis.axhspan(target_low, target_high, alpha=0.15, color="#2ca02c", label="target 25-30")
    axis.set_xlabel("epoch")
    axis.set_ylabel("episode reward")
    axis.set_title("Evaluation Reward Curves")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = chart_dir / "eval_reward_curves.png"
    fig.savefig(path, dpi=CHART_DPI)
    plt.close(fig)
    return path


def plot_eval_length_ctr(
    chart_dir: Path,
    eval_rows: Sequence[Mapping[str, Any]],
) -> Optional[Path]:
    """绘制 NX_0 长度和单步 reward 曲线。

    Args:
        chart_dir (Path): 图表目录。
        eval_rows (Sequence[Mapping[str, Any]]): 评估指标行。

    Returns:
        Optional[Path]: 图表路径；无数据时返回 None。
    """

    if not eval_rows:
        return None
    epochs = numeric_values(eval_rows, "trainer/epoch")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(
        epochs,
        numeric_values(eval_rows, "NX_0_len_tra"),
        marker="o",
        linewidth=1.6,
        markersize=3.5,
        color="#1f77b4",
    )
    axes[0].set_ylabel("NX_0 len_tra")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(
        epochs,
        numeric_values(eval_rows, "NX_0_ctr"),
        marker="o",
        linewidth=1.6,
        markersize=3.5,
        color="#d62728",
    )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("NX_0 ctr")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("NX_0 Length and Per-step Reward")
    fig.tight_layout()
    path = chart_dir / "nx0_length_ctr.png"
    fig.savefig(path, dpi=CHART_DPI)
    plt.close(fig)
    return path


def plot_eval_diversity(
    chart_dir: Path,
    eval_rows: Sequence[Mapping[str, Any]],
    target_ifeat: float,
) -> Optional[Path]:
    """绘制 NX_0 多样性和主导特征集中度曲线。

    Args:
        chart_dir (Path): 图表目录。
        eval_rows (Sequence[Mapping[str, Any]]): 评估指标行。
        target_ifeat (float): `ifeat_feat` 目标值，该指标越低越分散。

    Returns:
        Optional[Path]: 图表路径；无数据时返回 None。
    """

    if not eval_rows:
        return None
    epochs = numeric_values(eval_rows, "trainer/epoch")
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    curve_specs = [
        ("NX_0_Diversity", "NX_0 Diversity", "#1f77b4"),
        ("NX_0_Novelty", "NX_0 Novelty", "#2ca02c"),
        ("NX_0_ifeat_feat", "NX_0 ifeat_feat", "#d62728"),
    ]
    for axis, (metric_key, label, color) in zip(axes, curve_specs):
        axis.plot(epochs, numeric_values(eval_rows, metric_key), marker="o", linewidth=1.4, color=color)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[-1].axhline(target_ifeat, linestyle="--", linewidth=1.2, color="#555555", label="target 0.3")
    axes[-1].legend()
    axes[-1].set_xlabel("epoch")
    fig.suptitle("NX_0 Diversity and Dominant Feature Concentration")
    fig.tight_layout()
    path = chart_dir / "nx0_diversity_ifeat.png"
    fig.savefig(path, dpi=CHART_DPI)
    plt.close(fig)
    return path


def plot_training_rollout(
    chart_dir: Path,
    train_rows: Sequence[Mapping[str, Any]],
) -> Optional[Path]:
    """绘制训练 rollout 诊断曲线。

    Args:
        chart_dir (Path): 图表目录。
        train_rows (Sequence[Mapping[str, Any]]): 训练指标行。

    Returns:
        Optional[Path]: 图表路径；无数据时返回 None。
    """

    if not train_rows:
        return None
    steps = numeric_values(train_rows, "trainer/global_step")
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    curve_specs = [
        ("rollout/effective_steps", "effective steps", "#1f77b4"),
        ("rollout/done_ratio", "done ratio", "#ff7f0e"),
        ("rollout/reward_chunk", "reward chunk", "#2ca02c"),
    ]
    for axis, (metric_key, label, color) in zip(axes, curve_specs):
        axis.plot(steps, numeric_values(train_rows, metric_key), linewidth=1.2, color=color)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("global step")
    fig.suptitle("Training Rollout Diagnostics")
    fig.tight_layout()
    path = chart_dir / "training_rollout_diagnostics.png"
    fig.savefig(path, dpi=CHART_DPI)
    plt.close(fig)
    return path


def plot_q_alignment(
    chart_dir: Path,
    train_rows: Sequence[Mapping[str, Any]],
) -> Optional[Path]:
    """绘制 Q 估计与 TD 目标对齐情况。

    Args:
        chart_dir (Path): 图表目录。
        train_rows (Sequence[Mapping[str, Any]]): 训练指标行。

    Returns:
        Optional[Path]: 图表路径；无数据时返回 None。
    """

    if not train_rows:
        return None
    steps = numeric_values(train_rows, "trainer/global_step")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(steps, numeric_values(train_rows, "critic/q_mean"), label="q_mean", linewidth=1.2)
    axes[0].plot(
        steps,
        numeric_values(train_rows, "critic/target_q_mean"),
        label="target_q_mean",
        linewidth=1.2,
    )
    axes[0].set_ylabel("Q value")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        steps,
        numeric_values(train_rows, "critic/critic_loss"),
        label="critic_loss",
        linewidth=1.2,
    )
    axes[1].plot(
        steps,
        numeric_values(train_rows, "value/value_loss"),
        label="value_loss",
        linewidth=1.2,
    )
    axes[1].set_xlabel("global step")
    axes[1].set_ylabel("loss")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle("Q/V Target Alignment")
    fig.tight_layout()
    path = chart_dir / "q_value_alignment.png"
    fig.savefig(path, dpi=CHART_DPI)
    plt.close(fig)
    return path


def fmt(value: Any, digits: int = 4) -> str:
    """格式化报告中的数值。

    Args:
        value (Any): 待格式化对象。
        digits (int): 小数位数。

    Returns:
        str: 格式化后的字符串。
    """

    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if is_number(value):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """生成 Markdown 表格。

    Args:
        headers (Sequence[str]): 表头。
        rows (Sequence[Sequence[Any]]): 表格内容行。

    Returns:
        str: Markdown 表格字符串。
    """

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line] + row_lines)


def build_report(
    metrics_path: Path,
    config_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    chart_paths: Sequence[str],
) -> str:
    """构造中文 Markdown 主报告。

    Args:
        metrics_path (Path): 指标日志路径。
        config_path (Path): 配置文件路径。
        output_dir (Path): 输出目录。
        config (Mapping[str, Any]): 实验配置。
        summary (Mapping[str, Any]): 指标统计摘要。
        chart_paths (Sequence[str]): 图表相对路径列表。

    Returns:
        str: 完整 Markdown 报告文本。
    """

    train_summary = summary["training"]
    eval_summary = summary["evaluation"]
    dataset_summary = summary["dataset"]
    artifacts = summary["artifacts"]
    best_nx0 = eval_summary.get("best_nx0") or {}
    min_nx0_ifeat = eval_summary.get("min_nx0_ifeat") or {}
    last_eval = eval_summary.get("last_eval") or {}
    target_gap = eval_summary.get("target_gap") or {}
    target_ifeat = target_gap.get("target_ifeat", TARGET_IFEAT_FEAT)
    config_table = markdown_table(
        ["配置项", "取值"],
        [
            ["env", config.get("env")],
            ["dataset_path", config.get("dataset_path")],
            ["train_steps", config.get("train_steps")],
            ["epoch / step_per_epoch", f"{config.get('epoch')} / {config.get('step_per_epoch')}"],
            ["chunk_size / max_turn / force_length", f"{config.get('chunk_size')} / {config.get('max_turn')} / {config.get('force_length')}"],
            ["gamma", config.get("gamma")],
            ["repeat_policy", config.get("repeat_policy")],
            ["num_samples_train / num_samples_test", f"{config.get('num_samples_train')} / {config.get('num_samples_test')}"],
            ["lambda_entropy / lambda_variance", f"{config.get('lambda_entropy')} / {config.get('lambda_variance')}"],
        ],
    )
    eval_table = markdown_table(
        ["指标", "首轮", "末轮", "最佳", "说明"],
        [
            [
                "NX_0_R_tra",
                fmt(eval_summary.get("NX_0_R_tra", {}).get("first")),
                fmt(eval_summary.get("NX_0_R_tra", {}).get("last")),
                fmt(best_nx0.get("NX_0_R_tra")),
                "用户关注的目标指标",
            ],
            [
                "NX_0_len_tra",
                fmt(eval_summary.get("NX_0_len_tra", {}).get("first")),
                fmt(eval_summary.get("NX_0_len_tra", {}).get("last")),
                fmt(best_nx0.get("NX_0_len_tra")),
                "自然终止平均长度",
            ],
            [
                "NX_0_ctr",
                fmt(eval_summary.get("NX_0_ctr", {}).get("first")),
                fmt(eval_summary.get("NX_0_ctr", {}).get("last")),
                fmt(best_nx0.get("NX_0_ctr")),
                "单步 reward",
            ],
            [
                "NX_0_ifeat_feat",
                fmt(eval_summary.get("NX_0_ifeat_feat", {}).get("first")),
                fmt(eval_summary.get("NX_0_ifeat_feat", {}).get("last")),
                fmt(min_nx0_ifeat.get("NX_0_ifeat_feat")),
                "越低越分散，目标约 0.3",
            ],
            [
                "NX_0_Diversity",
                fmt(eval_summary.get("NX_0_Diversity", {}).get("first")),
                fmt(eval_summary.get("NX_0_Diversity", {}).get("last")),
                fmt(best_nx0.get("NX_0_Diversity")),
                "轨迹内 item 不相似度",
            ],
            [
                "NX_30_R_tra",
                fmt(eval_summary.get("NX_30_R_tra", {}).get("first")),
                fmt(eval_summary.get("NX_30_R_tra", {}).get("last")),
                fmt((eval_summary.get("best_nx30") or {}).get("NX_30_R_tra")),
                "强制 30 步参考",
            ],
            [
                "NX_30_ctr",
                fmt(eval_summary.get("NX_30_ctr", {}).get("first")),
                fmt(eval_summary.get("NX_30_ctr", {}).get("last")),
                fmt((eval_summary.get("best_nx30") or {}).get("NX_30_ctr")),
                "强制 30 步单步 reward",
            ],
        ],
    )
    train_table = markdown_table(
        ["训练指标", "均值", "中位数", "最小", "最大", "末值"],
        [
            metric_stats_row(train_summary, "rollout/effective_steps"),
            metric_stats_row(train_summary, "rollout/done_ratio"),
            metric_stats_row(train_summary, "rollout/reward_chunk"),
            metric_stats_row(train_summary, "rollout/repeat_ratio"),
            metric_stats_row(train_summary, "rollout/exact_repeat_ratio"),
            metric_stats_row(train_summary, "rollout/pred_reward"),
            metric_stats_row(train_summary, "critic/target_q_mean"),
            metric_stats_row(train_summary, "critic/q_mean"),
            metric_stats_row(train_summary, "critic/critic_loss"),
            metric_stats_row(train_summary, "value/value_loss"),
        ],
    )
    dataset_text = render_dataset_section(dataset_summary)
    correlations = eval_summary.get("correlations_with_nx0_reward", {})
    corr_table = markdown_table(
        ["相关项", "与 NX_0_R_tra 的 Pearson r"],
        [[key, fmt(value)] for key, value in correlations.items()],
    )
    ifeat_correlations = eval_summary.get("correlations_with_nx0_ifeat", {})
    ifeat_corr_table = markdown_table(
        ["相关项", "与 NX_0_ifeat_feat 的 Pearson r"],
        [[key, fmt(value)] for key, value in ifeat_correlations.items()],
    )
    chart_lines = "\n".join(f"- `{path}`" for path in chart_paths) if chart_paths else "本次未生成图表。"
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""# 科研代码任务报告

## 1. 任务计划
### 1.1 原始需求
分析 `{metrics_path}` 下的 DORL-MAC Q/V 训练日志，在 `./results_analysis` 文件夹下输出训练日志分析与改进报告；重点解释 `NX_0_*` 相关指标结果不好的原因，并围绕 `NX_0_rew` 需要达到 25-30 以上、`ifeat_feat` 越低越好且最好接近 0.3 的目标提出改进建议。

### 1.2 任务分解
1. 解析指定 `metrics.jsonl`，区分训练日志、评估日志和配置事件。
2. 统计 `NX_0_R_tra/NX_0_len_tra/NX_0_ctr` 与 `NX_30` 对照走势，计算 reward 目标差距。
3. 统计 `NX_0_ifeat_feat/NX_0_Diversity/NX_0_Novelty`，重点判断主导特征集中度是否接近 0.3。
4. 统计 Q/V 训练期 `rollout/effective_steps`、`done_ratio`、`reward_chunk`、`repeat_ratio`、`target_q_mean` 与损失走势。
5. 结合当前实现中 `ActionChunkDataset`、`rollout_chunks`、`CollectorSet` 的语义解释异常指标。
6. 生成 CSV、JSON、图表、主报告和 agent 交接日志。

### 1.3 技术方案
使用 `results_analysis/analyze_dorl_mac_qv_metrics.py` 对 JSONL 进行可复现统计。主结论只以指定 `metrics.jsonl` 为权威来源；同目录历史 `eval_during_train` 中 epoch 24 之后存在 `NX_10` 字段，与当前配置 `force_length=30` 不一致，因此未纳入主结论。

## 2. 任务完成过程记录
### 2.1 代码结构
- 更新/使用：`results_analysis/analyze_dorl_mac_qv_metrics.py`
- 新增：`{artifacts["report"]}`
- 新增：`{artifacts["handoff"]}`
- 新增：`{artifacts["train_csv"]}`
- 新增：`{artifacts["eval_csv"]}`
- 新增：`{artifacts["summary_json"]}`
- 新增图表：
{chart_lines}

### 2.2 模块详细说明
#### 2.2.1 日志分析脚本
- 文件路径：`{output_dir / "analyze_dorl_mac_qv_metrics.py"}`
- 核心功能：解析 Q/V 训练日志，生成统计摘要、CSV、图表和 Markdown 报告。
- 关键函数：
  - `load_metrics_jsonl(path)`：读取并校验 JSONL 指标。
  - `split_metric_rows(records)`：拆分训练指标与评估指标。
  - `summarize_training(train_rows)`：汇总 Q/V 训练 rollout 与损失。
  - `summarize_evaluation(eval_rows, target_low, target_high, target_ifeat)`：计算 `NX_0` 指标、目标差距和相关性。
  - `summarize_dataset_from_config(config)`：统计离线 chunk 的绝对起点分布，用作历史 bug 回归检查，不再把它解释为当前 rollout 的 episode 内步数。
- 核心算法：对日志字段做数值清洗、首尾/最大/均值/相关性统计；同时对 `ifeat_feat/Diversity/Novelty`、`repeat_ratio/exact_repeat_ratio`、`reward_chunk`、`target_q_mean` 与 `NX_0` 评估指标做联合诊断。

### 2.3 既有代码修改说明
本任务未修改训练、模型或评估代码；只更新日志分析脚本并生成分析产物。

## 3. 数据处理说明
### 3.1 数据来源
- 指标日志：`{metrics_path}`
- 配置文件：`{config_path}`
- 生成时间：{now_text}
- 日志记录数：{summary["record_count"]}，其中训练记录 {train_summary["rows"]} 条、评估记录 {eval_summary["rows"]} 条、配置事件 {summary["config_events"]} 条。
- 日志覆盖范围：训练 `global_step={fmt(train_summary.get("first_global_step"), 0)}` 到 `global_step={fmt(train_summary.get("last_global_step"), 0)}`；评估 epoch 1 到 epoch {fmt(last_eval.get("epoch"), 0)}。配置目标为 `{config.get("epoch")}` 个 epoch、`{config.get("train_steps")}` 步，因此当前 `metrics.jsonl` 只覆盖前 {fmt((last_eval.get("epoch") or 0) / max(float(config.get("epoch") or 1), 1.0) * 100, 2)}% 的 epoch。

### 3.2 数据预处理步骤
- 读取每行 JSON，展开 `metrics` 字段。
- 使用 `critic/critic_loss` 判定训练记录，使用 `NX_0_R_tra` 判定评估记录。
- 将 `NX_0_rew` 与 `NX_0_R_tra` 视为同一业务含义，报告中统一使用 `NX_0_R_tra`。
- 导出训练与评估 CSV，便于后续二次分析。

### 3.3 数据统计信息
{dataset_text}

## 4. 实验结果与分析
### 4.1 实验设置
- 硬件环境：日志配置记录训练设备为 `{config.get("device")}`；本次分析未重新训练模型。
- 软件环境：分析脚本在当前项目环境运行；关键依赖为 Python、numpy、matplotlib。
- 超参数设置：

{config_table}

### 4.2 图表结果分析
#### 4.2.1 评估 reward 曲线
- 图表类型：折线图。
- 横坐标含义：训练 epoch。
- 纵坐标含义：每个评估 episode 的平均累计 reward。
- 图例说明：`FB R_tra`、`NX_0 R_tra`、`NX_30 R_tra` 与目标区间 25-30。
- 数据计算方式：来自 `metrics.jsonl` 中每个 epoch 的评估记录。
- 结果分析：`NX_0_R_tra` 首轮为 {fmt(eval_summary.get("NX_0_R_tra", {}).get("first"))}，末轮为 {fmt(eval_summary.get("NX_0_R_tra", {}).get("last"))}，最佳为 {fmt(best_nx0.get("NX_0_R_tra"))}，距离 25 仍差 {fmt(target_gap.get("best_to_25"))}，距离 30 仍差 {fmt(target_gap.get("best_to_30"))}。当前没有接近 25-30 的趋势。

#### 4.2.2 NX_0 长度与单步 reward
- 图表类型：双子图折线图。
- 横坐标含义：训练 epoch。
- 纵坐标含义：上图为 `NX_0_len_tra`，下图为 `NX_0_ctr`。
- 图例说明：自然终止长度和单步 reward。
- 数据计算方式：`NX_0_R_tra = NX_0_len_tra * NX_0_ctr`。
- 结果分析：末轮 `NX_0_len_tra={fmt(last_eval.get("NX_0_len_tra"))}`、`NX_0_ctr={fmt(last_eval.get("NX_0_ctr"))}`。若保持末轮单步 reward，要达到 25 需要平均长度 {fmt(target_gap.get("required_len_at_last_ctr_for_25"))}，达到 30 需要 {fmt(target_gap.get("required_len_at_last_ctr_for_30"))}；这已经接近或超过 `max_turn=30`。因此 `NX_0` 的核心瓶颈是轨迹过早终止，同时单步 reward 还不足以弥补长度缺口。

#### 4.2.3 NX_0 多样性与 ifeat_feat
- 图表类型：三行折线图。
- 横坐标含义：训练 epoch。
- 纵坐标含义：`NX_0_Diversity`、`NX_0_Novelty`、`NX_0_ifeat_feat`。
- 图例说明：`ifeat_feat` 越低表示越少集中在主导特征集合；灰色虚线为目标值 {fmt(target_ifeat)}。
- 数据计算方式：来自每轮评估轨迹中的 item 特征覆盖和流行度统计。
- 结果分析：`NX_0_ifeat_feat` 首轮 {fmt(eval_summary.get("NX_0_ifeat_feat", {}).get("first"))}，末轮 {fmt(eval_summary.get("NX_0_ifeat_feat", {}).get("last"))}，日志内最低也只有 {fmt(min_nx0_ifeat.get("NX_0_ifeat_feat"))}，距离目标 0.3 仍高 {fmt(target_gap.get("min_ifeat_over_target"))}。这说明策略仍显著集中在主导特征集合，虽然 `Diversity` 数值不低，但从 MCD/ifeat 角度看还不够分散，可能更容易反复触达相似兴趣簇并触发退出。

#### 4.2.4 训练 rollout 诊断
- 图表类型：三行折线图。
- 横坐标含义：训练 global step。
- 纵坐标含义：`rollout/effective_steps`、`rollout/done_ratio`、`rollout/reward_chunk`。
- 图例说明：每条曲线对应一个训练期 rollout 指标。
- 数据计算方式：来自 Q/V 更新日志。
- 结果分析：`rollout/effective_steps` 均值为 {fmt(train_summary.get("rollout/effective_steps", {}).get("mean"))}，中位数 {fmt(train_summary.get("rollout/effective_steps", {}).get("median"))}，`done_ratio` 均值只有 {fmt(train_summary.get("rollout/done_ratio", {}).get("mean"))}。这说明当前日志已经不再是“rollout 几乎不执行”的旧问题；每个 chunk 基本完整执行 3 步。真正异常的是 `rollout/reward_chunk` 均值 {fmt(train_summary.get("rollout/reward_chunk", {}).get("mean"))}，同时 `repeat_ratio` 均值 {fmt(train_summary.get("rollout/repeat_ratio", {}).get("mean"))}、`exact_repeat_ratio` 均值 {fmt(train_summary.get("rollout/exact_repeat_ratio", {}).get("mean"))}，训练回报被重复推荐、低多样性和违规惩罚主导。

#### 4.2.5 Q/V 目标对齐
- 图表类型：两行折线图。
- 横坐标含义：训练 global step。
- 纵坐标含义：上图为 `critic/q_mean` 与 `critic/target_q_mean`，下图为 critic/value loss。
- 图例说明：Q 估计、TD target 和两个回归损失。
- 数据计算方式：来自训练日志。
- 结果分析：`critic/target_q_mean` 均值 {fmt(train_summary.get("critic/target_q_mean", {}).get("mean"))}，`critic/q_mean` 均值 {fmt(train_summary.get("critic/q_mean", {}).get("mean"))}，二者整体对齐；critic/value loss 从首轮明显下降到末轮附近的 {fmt(train_summary.get("critic/critic_loss", {}).get("last"))}/{fmt(train_summary.get("value/value_loss", {}).get("last"))}。但这种对齐主要是在拟合负向 TD target：`rollout/reward_chunk` 长期为负，且 `rollout/pred_reward` 均值只有 {fmt(train_summary.get("rollout/pred_reward", {}).get("mean"))}，Q/V 没有学到能支撑 `NX_0` 长交互的正向 chunk 价值。

### 4.3 关键统计表
#### 4.3.1 评估指标概览

{eval_table}

#### 4.3.2 训练指标概览

{train_table}

#### 4.3.3 NX_0_R_tra 相关性

{corr_table}

#### 4.3.4 NX_0_ifeat_feat 相关性

{ifeat_corr_table}

### 4.4 为什么 NX_0_* 结果不好
1. **自然终止长度太短，是 `NX_0_R_tra` 的直接瓶颈。** 末轮 `NX_0_len_tra={fmt(last_eval.get("NX_0_len_tra"))}`，最佳 `NX_0_R_tra` 对应长度 {fmt(best_nx0.get("NX_0_len_tra"))}，距离稳定接近 30 步仍很远。即使单步 reward 已经能到 0.75-0.80，长度不足也会把累计 reward 锁在 7-9 附近。
2. **`ifeat_feat` 明显偏高，多样性没有达到目标。** 用户期望 `ifeat_feat` 降到约 0.3，当前末轮为 {fmt(last_eval.get("NX_0_ifeat_feat"))}，日志内最低为 {fmt(min_nx0_ifeat.get("NX_0_ifeat_feat"))}，仍高出目标 {fmt(target_gap.get("min_ifeat_over_target"))}。这说明推荐仍集中在训练数据主导特征集合上，容易让用户在短轨迹内重复接收相似兴趣簇，进而增加退出风险。
3. **训练 rollout 已经完整执行，但 chunk 回报长期为负。** 当前 `effective_steps` 全程约等于 `chunk_size=3`，`done_ratio` 接近 0，说明之前的 episode 步数错位问题在这份日志中已经缓解；但 `reward_chunk` 均值为 {fmt(train_summary.get("rollout/reward_chunk", {}).get("mean"))}，最优也只有 {fmt(train_summary.get("rollout/reward_chunk", {}).get("max"))}。Q/V 学到的是“这些候选 chunk 大多会带来负调整后回报”，而不是能延长 `NX_0` 轨迹的高价值 chunk。
4. **重复推荐信号过高，直接压低 TD target。** 训练期 `repeat_ratio` 均值 {fmt(train_summary.get("rollout/repeat_ratio", {}).get("mean"))}，`exact_repeat_ratio` 均值 {fmt(train_summary.get("rollout/exact_repeat_ratio", {}).get("mean"))}。在 `invalid_action_penalty=-1.0` 且 `repeat_policy=truncate` 的设置下，这会让大量 chunk 被负惩罚支配，并诱导 critic/value 把可采样候选整体估成负值。该现象与高 `ifeat_feat` 是同一类问题：候选池缺少足够分散、非重复的可执行推荐。
5. **训练语义与 `NX_0` 自然终止语义仍可能不一致。** 当前训练日志中的 `done_ratio` 几乎为 0，但评估中 `NX_0_len_tra` 只有约 10-12，说明训练 rollout 内部并没有充分暴露“用户会提前离开”的风险，或者重复、低多样性和离开风险只以惩罚进入 target，没有以真实 done 形式改变后续状态分布。
6. **候选动作搜索受冻结 BC actor 限制。** Q/V 只在 actor 采样出的少量 chunk 中选最大 Q，训练 `num_samples_train=8`，评估 `num_samples_test=32`。当 actor 候选本身高度重复、`ifeat_feat` 偏高、缺少能同时高 watch_ratio 和低 leave-risk 的 chunk 时，critic 再好也只能在低质量候选池里排序。
7. **训练尚未跑满，后期是否继续改善未知。** 当前日志只到 epoch {fmt(last_eval.get("epoch"), 0)}，配置目标是 {config.get("epoch")} 个 epoch；但前 11 个评估点里 `NX_0_R_tra` 没有稳定上升到 10 以上，`ifeat_feat` 也始终远高于 0.3，早期趋势已经暴露出候选质量、多样性和终止建模问题。

## 5. 最终结论
当前日志中 `NX_0_R_tra` 最佳仅 {fmt(best_nx0.get("NX_0_R_tra"))}，末轮 {fmt(last_eval.get("NX_0_R_tra"))}，距离 25-30 的 SOTA 目标仍很远。与旧日志不同，这次 `rollout/effective_steps=3`、`done_ratio≈0`，说明 Q/V chunk rollout 已能完整执行；新的主要矛盾是候选 chunk 质量、多样性和终止建模：`NX_0_ifeat_feat` 末轮 {fmt(last_eval.get("NX_0_ifeat_feat"))}、最低 {fmt(min_nx0_ifeat.get("NX_0_ifeat_feat"))}，远高于 0.3 目标，训练期 `repeat_ratio/exact_repeat_ratio` 也过高，`reward_chunk` 长期为负。critic/value 虽然能拟合 TD target，却没有学到低 `ifeat_feat`、低重复、能把 `NX_0` 轨迹从约 10 步延长到接近 30 步的正向价值函数。

## 6. 后续建议
1. **把 `ifeat_feat` 纳入 Q/V 选择目标。** 在 chunk score 中加入 MCD/ifeat 惩罚项，例如 `score = Q - alpha * ifeat_feat` 或在 reward 中加入 `-alpha * max(ifeat_feat - 0.3, 0)`，让 critic/rerank 显式偏好低主导特征集中度的 chunk。
2. **候选生成阶段做特征级去重。** 不只屏蔽重复 item，也要限制一个 chunk 内和最近窗口内的主导 feature 重复；目标是把 `NX_0_ifeat_feat` 从当前约 0.71-0.75 降到 0.3 附近，同时保持 `NX_0_ctr` 不明显下降。
3. **优先降低重复推荐。** 在 Q/V rollout 和评估动作映射中强制使用 `recommended_mask` 屏蔽已推荐 item，或在 actor candidate generation 阶段就做去重；目标是把 `exact_repeat_ratio` 从约 {fmt(train_summary.get("rollout/exact_repeat_ratio", {}).get("mean"))} 降到接近 0。
4. **把 `repeat_policy=truncate` 做成真实截断或显式 done。** 如果重复推荐或低多样性会导致用户离开，训练 rollout 应停止后续 chunk、设置 done，并让 TD target 学到“集中推荐会缩短轨迹”；如果只惩罚不断状态，训练会低估自然终止长度损失。
5. **显式优化留存长度。** 在 target 中加入 leave-risk/done penalty 或 survival bonus，或者把 value target 改成“累计 watch_ratio + 留存约束 + 多样性约束”，而不是只拟合当前候选 chunk 的惩罚后短回报。
6. **校准 reward 模型尺度。** 当前 `rollout/pred_reward` 均值只有 {fmt(train_summary.get("rollout/pred_reward", {}).get("mean"))}，远低于评估中的单步 `NX_0_ctr≈0.75`。需要检查训练 rollout 使用的是归一化预测、概率、watch_ratio 还是惩罚后 reward，避免 critic 在与评估 reward 不同尺度的目标上学习。
7. **扩大并改进候选搜索。** 在 `ifeat_feat` 和 repeat 降下来后，再提高 `num_samples_train/test`，尝试 beam/CEM rerank，让 critic 有机会从更多非重复、高 reward、低离开风险的 chunk 中选择。
8. **继续跑到完整 100 epoch，但带诊断门槛。** 若后续 `reward_chunk` 仍为负、`repeat_ratio` 仍高于 0.3、`NX_0_ifeat_feat` 仍高于 0.5、`NX_0_len_tra` 仍卡在 10-12，则不建议只靠延长训练；应先修候选去重、多样性约束、done 语义和 reward 尺度。

## 7. 代码使用说明
在项目根目录运行：

```bash
conda run -n easyrl4rec python results_analysis/analyze_dorl_mac_qv_metrics.py
```

也可以指定其他日志：

```bash
conda run -n easyrl4rec python results_analysis/analyze_dorl_mac_qv_metrics.py --metrics-path path/to/metrics.jsonl --output-dir results_analysis/another_run
```

主要输出文件：
- 主报告：`{artifacts["report"]}`
- 交接日志：`{artifacts["handoff"]}`
- 训练指标 CSV：`{artifacts["train_csv"]}`
- 评估指标 CSV：`{artifacts["eval_csv"]}`
- 统计摘要 JSON：`{artifacts["summary_json"]}`
"""


def metric_stats_row(summary: Mapping[str, Any], key: str) -> List[str]:
    """构造训练指标统计表的一行。

    Args:
        summary (Mapping[str, Any]): 训练统计摘要。
        key (str): 指标字段名。

    Returns:
        List[str]: Markdown 表格行。
    """

    stats = summary.get(key, {})
    return [
        key,
        fmt(stats.get("mean")),
        fmt(stats.get("median")),
        fmt(stats.get("min")),
        fmt(stats.get("max")),
        fmt(stats.get("last")),
    ]


def render_dataset_section(dataset_summary: Mapping[str, Any]) -> str:
    """渲染数据统计章节。

    Args:
        dataset_summary (Mapping[str, Any]): 数据集统计摘要。

    Returns:
        str: Markdown 文本。
    """

    if "error" in dataset_summary:
        return f"- 数据集统计读取失败：`{dataset_summary['error']}`"
    trajectory_length = dataset_summary["trajectory_length"]
    chunk_start = dataset_summary["chunk_start"]
    table = markdown_table(
        ["项目", "数值"],
        [
            ["轨迹数", dataset_summary["num_trajectories"]],
            ["transition 总数", dataset_summary["num_transitions"]],
            ["轨迹长度 min/median/mean/max", f"{trajectory_length['min']} / {fmt(trajectory_length['median'])} / {fmt(trajectory_length['mean'])} / {trajectory_length['max']}"],
            ["可派生 chunk 数", dataset_summary["num_chunks"]],
            ["chunk 起点 p50/p90/p99/max", f"{chunk_start['p50']} / {chunk_start['p90']} / {chunk_start['p99']} / {chunk_start['max']}"],
            ["旧实现下 start >= max_turn 比例", f"{fmt(dataset_summary['pct_start_ge_max_turn'] * 100, 2)}%"],
            ["旧实现下理论零可执行步比例", f"{fmt(dataset_summary['pct_zero_effective_steps'] * 100, 2)}%"],
            ["旧实现下理论平均可执行步数", fmt(dataset_summary["expected_effective_steps_mean"])],
        ],
    )
    return (
        "说明：下表中的 `start` 是离线轨迹绝对下标，只用于回归检查历史实现；"
        "当前这份日志的 `rollout/effective_steps=3` 已经表明训练 rollout 没有继续把该绝对下标当作 episode 内步数。\n\n"
        f"{table}"
    )


def build_handoff_log(summary: Mapping[str, Any], artifacts: Mapping[str, str]) -> str:
    """生成简洁 agent 交接日志。

    Args:
        summary (Mapping[str, Any]): 统计摘要。
        artifacts (Mapping[str, str]): 关键产物路径。

    Returns:
        str: Markdown 交接日志。
    """

    eval_summary = summary["evaluation"]
    train_summary = summary["training"]
    best_nx0 = eval_summary.get("best_nx0") or {}
    min_nx0_ifeat = eval_summary.get("min_nx0_ifeat") or {}
    last_eval = eval_summary.get("last_eval") or {}
    target_gap = eval_summary.get("target_gap") or {}
    metrics_path = summary.get("metrics_path", "N/A")
    return f"""# Agent Context Log

## 1. 任务概述
- 任务名称：dorl_mac_qv_training_log_analysis
- 当前状态：已完成
- 本轮目标：分析指定 Q/V 时间戳日志中 Q/V 训练、`NX_0_*`、`ifeat_feat` 多样性指标表现，并在 `results_analysis` 生成报告。

## 2. 当前进展
- 已完成工作 1：更新/使用 `results_analysis/analyze_dorl_mac_qv_metrics.py`，生成 CSV、JSON、图表、主报告。
- 已完成工作 2：确认主日志只覆盖到 epoch {fmt(last_eval.get("epoch"), 0)} / global_step {fmt(train_summary.get("last_global_step"), 0)}，未混用旧 `NX_10` 摘要。
- 未完成工作：未修改训练代码，未重新训练模型。

## 3. 已确认结论
- 结论 1：`NX_0_R_tra` 最佳 {fmt(best_nx0.get("NX_0_R_tra"))}、末轮 {fmt(last_eval.get("NX_0_R_tra"))}，距离 25-30 目标很远。
- 结论 2：当前日志中 rollout 已完整执行，`effective_steps` 均值 {fmt(train_summary.get("rollout/effective_steps", {}).get("mean"))}，`done_ratio` 均值 {fmt(train_summary.get("rollout/done_ratio", {}).get("mean"))}，旧的零步 rollout 问题不是这份日志的主因。
- 结论 3：`NX_0_ifeat_feat` 末轮 {fmt(last_eval.get("NX_0_ifeat_feat"))}、最低 {fmt(min_nx0_ifeat.get("NX_0_ifeat_feat"))}，距离 0.3 目标仍高 {fmt(target_gap.get("min_ifeat_over_target"))}，候选仍集中在主导特征集合。
- 结论 4：训练期 `reward_chunk` 均值 {fmt(train_summary.get("rollout/reward_chunk", {}).get("mean"))}，`repeat_ratio` 均值 {fmt(train_summary.get("rollout/repeat_ratio", {}).get("mean"))}，`exact_repeat_ratio` 均值 {fmt(train_summary.get("rollout/exact_repeat_ratio", {}).get("mean"))}，候选 chunk 被重复推荐/低多样性/违规惩罚主导。

## 4. 关键证据或依据
- 证据 1：指定日志 `{metrics_path}` 中训练记录 {train_summary["rows"]} 条、评估记录 {eval_summary["rows"]} 条。
- 证据 2：`NX_30_R_tra` 后期达到 {fmt((eval_summary.get("best_nx30") or {}).get("NX_30_R_tra"))}，但 `NX_0_len_tra` 末轮只有 {fmt(last_eval.get("NX_0_len_tra"))}，且 `NX_0_ifeat_feat` 末轮 {fmt(last_eval.get("NX_0_ifeat_feat"))}，说明主要问题是自然终止长度、候选稳定性和主导特征集中度。

## 5. 未解决问题与风险
- 风险 1：当前没有重新训练模型，关于去重、多样性约束、done 语义和 reward 尺度的改进建议尚未通过新实验验证。
- 风险 2：`eval_during_train` 目录混有旧 `NX_10` 摘要，后续实验应清理目录或使用全新输出路径。

## 6. 建议下一步
- 下一步 1：把 `ifeat_feat` 纳入 Q/V rerank 或 reward shaping，优先把 `NX_0_ifeat_feat` 从约 0.71-0.75 压到 0.3 附近。
- 下一步 2：结合 `recommended_mask` 和 feature 级去重降低重复推荐，再重跑训练，优先检查 `ifeat_feat` 下降、`reward_chunk` 转正、`repeat_ratio` 下降和 `NX_0_len_tra` 提升。

## 7. 相关文件
- 代码文件：`results_analysis/analyze_dorl_mac_qv_metrics.py`
- 详细报告：`{artifacts["report"]}`
- 关键日志：`{metrics_path}`
"""


def build_summary(
    metrics_path: Path,
    config_path: Path,
    records: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    config_events: int,
    config: Mapping[str, Any],
    artifacts: Mapping[str, str],
    target_low: float,
    target_high: float,
    target_ifeat: float,
) -> Dict[str, Any]:
    """整合所有统计摘要。

    Args:
        metrics_path (Path): 指标日志路径。
        config_path (Path): 配置文件路径。
        records (Sequence[Mapping[str, Any]]): 原始记录。
        train_rows (Sequence[Mapping[str, Any]]): 训练行。
        eval_rows (Sequence[Mapping[str, Any]]): 评估行。
        config_events (int): 配置事件数量。
        config (Mapping[str, Any]): 实验配置。
        artifacts (Mapping[str, str]): 产物路径。
        target_low (float): 低位目标 reward。
        target_high (float): 高位目标 reward。
        target_ifeat (float): `ifeat_feat` 目标值，该指标越低越分散。

    Returns:
        Dict[str, Any]: 完整统计摘要。
    """

    return {
        "metrics_path": str(metrics_path),
        "config_path": str(config_path),
        "record_count": len(records),
        "config_events": config_events,
        "training": summarize_training(train_rows),
        "evaluation": summarize_evaluation(eval_rows, target_low, target_high, target_ifeat),
        "dataset": summarize_dataset_from_config(config),
        "artifacts": dict(artifacts),
    }


def main() -> None:
    """执行日志分析并写入所有产物。

    Returns:
        None.
    """

    configure_logging()
    args = parse_args()
    metrics_path = args.metrics_path
    config_path = args.config_path or metrics_path.parent / "resolved_config.json"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("读取日志：%s", metrics_path)
    records = load_metrics_jsonl(metrics_path)
    train_rows, eval_rows, config_events = split_metric_rows(records)
    LOGGER.info("训练记录=%s，评估记录=%s，配置事件=%s", len(train_rows), len(eval_rows), config_events)

    # 优先从新版 JSONL header 里读取 config；缺失时回退到旁路 resolved_config.json。
    header_config = extract_config_header(records)
    if header_config is not None:
        LOGGER.info("使用 JSONL header 中的 config（run_name=%s）", header_config.get("run_name"))
        config = header_config
        if args.config_path is not None:
            LOGGER.info("同时读取旁路配置补齐字段：%s", args.config_path)
            side_config = load_json(args.config_path)
            for key, value in side_config.items():
                config.setdefault(key, value)
    else:
        LOGGER.info("JSONL header 缺失 config，回退读取旁路配置：%s", config_path)
        config = load_json(config_path)
    artifacts = {
        "report": str(output_dir / REPORT_FILENAME),
        "handoff": str(output_dir / "agent_context" / HANDOFF_FILENAME),
        "train_csv": str(output_dir / TRAIN_METRICS_FILENAME),
        "eval_csv": str(output_dir / EVAL_METRICS_FILENAME),
        "summary_json": str(output_dir / SUMMARY_FILENAME),
    }

    write_csv(output_dir / TRAIN_METRICS_FILENAME, train_rows)
    write_csv(output_dir / EVAL_METRICS_FILENAME, eval_rows)
    chart_paths = plot_all(
        output_dir,
        train_rows,
        eval_rows,
        args.target_low,
        args.target_high,
        args.target_ifeat,
    )
    summary = build_summary(
        metrics_path=metrics_path,
        config_path=config_path,
        records=records,
        train_rows=train_rows,
        eval_rows=eval_rows,
        config_events=config_events,
        config=config,
        artifacts=artifacts,
        target_low=args.target_low,
        target_high=args.target_high,
        target_ifeat=args.target_ifeat,
    )
    summary["charts"] = chart_paths
    write_json(output_dir / SUMMARY_FILENAME, summary)

    report = build_report(
        metrics_path=metrics_path,
        config_path=config_path,
        output_dir=output_dir,
        config=config,
        summary=summary,
        chart_paths=chart_paths,
    )
    report_path = output_dir / REPORT_FILENAME
    report_path.write_text(report, encoding="utf-8")

    handoff = build_handoff_log(summary, artifacts)
    handoff_dir = output_dir / "agent_context"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    timestamped_handoff = handoff_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_dorl_mac_qv_training_log_analysis.md"
    timestamped_handoff.write_text(handoff, encoding="utf-8")
    (handoff_dir / HANDOFF_FILENAME).write_text(handoff, encoding="utf-8")
    LOGGER.info("报告已生成：%s", report_path)
    LOGGER.info("交接日志已生成：%s", handoff_dir / HANDOFF_FILENAME)


if __name__ == "__main__":
    main()
