from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import re
import sys
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.envs.KuaiRec.KuaiData import KuaiData
from src.core.util.entropy_penalty import (
    accumulate_entropy_counts_for_row,
    compute_step_entropy,
    finalize_entropy_map,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_VARIANCE_REFERENCE_LAMBDA = 0.05
"""当前项目中 MOPO 常用的默认不确定性惩罚系数。"""


def parse_args() -> argparse.Namespace:
    """解析离线标定脚本参数。

    Returns:
        argparse.Namespace: 命令行参数对象。
    """
    parser = argparse.ArgumentParser(
        description="Calibrate MOPO lambda_variance against DORL entropy penalty scale."
    )
    parser.add_argument(
        "--big-csv",
        type=Path,
        default=Path("data/KuaiRec/data_raw/big_matrix_processed.csv"),
    )
    parser.add_argument(
        "--prediction-mat-path",
        type=Path,
        default=Path("saved_models/KuaiEnv-v0/DeepFM/matsPre/[pointneg]_matPre.pickle"),
    )
    parser.add_argument(
        "--small-csv",
        type=Path,
        default=Path("data/KuaiRec/data_raw/small_matrix_processed.csv"),
    )
    parser.add_argument(
        "--var-mat-path",
        type=Path,
        default=Path("saved_models/KuaiEnv-v0/DeepFM/matsVar/[pointneg]_matVar.pickle"),
    )
    parser.add_argument(
        "--mopo-log-dir",
        type=Path,
        default=Path("saved_models/KuaiEnv-v0/MOPO/logs"),
    )
    parser.add_argument(
        "--lambda-entropy",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--entropy-window",
        type=int,
        nargs="*",
        default=[1, 2],
    )
    parser.add_argument(
        "--feature-level",
        dest="feature_level",
        action="store_true",
    )
    parser.add_argument(
        "--no-feature-level",
        dest="feature_level",
        action="store_false",
    )
    parser.set_defaults(feature_level=True)
    parser.add_argument(
        "--is-sorted",
        dest="is_sorted",
        action="store_true",
    )
    parser.add_argument(
        "--no-is-sorted",
        dest="is_sorted",
        action="store_false",
    )
    parser.set_defaults(is_sorted=True)
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/mopo_lambda_calibration.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/mopo_lambda_calibration.json"),
    )
    parser.add_argument(
        "--entropy-map-cache",
        type=Path,
        default=Path("results/mopo_lambda_entropy_map.pkl"),
    )
    return parser.parse_args()


def round_sig(value: float, sig: int = 3) -> float:
    """按有效数字进行四舍五入。

    Args:
        value (float): 待处理数值。
        sig (int): 保留的有效数字位数。

    Returns:
        float: 保留有效数字后的结果。
    """
    if value == 0 or not math.isfinite(value):
        return float(value)
    return round(value, sig - int(math.floor(math.log10(abs(value)))) - 1)


def summarize(values: np.ndarray) -> Dict[str, float]:
    """汇总数组的基础统计量。

    Args:
        values (np.ndarray): 一维数值数组。

    Returns:
        Dict[str, float]: 包含 `count/mean/median/p95/p99/min/max` 的字典。

    Raises:
        ValueError: 当输入数组为空时抛出。
    """
    if values.size == 0:
        raise ValueError("Cannot summarize an empty array")
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def format_float(value: float) -> str:
    """将浮点数格式化为便于报告展示的短字符串。"""
    if not math.isfinite(value):
        return str(value)
    return f"{value:.6g}"


def load_small_space_metadata() -> Dict[str, Any]:
    """加载 KuaiEnv small-space 的编码器与物品特征映射。

    Returns:
        Dict[str, Any]: 包含 small-space 形状、user/item 编码器及物品特征映射。
    """
    mat, lbe_user, lbe_item = KuaiData.load_mat()
    _, df_feat = KuaiData.load_category()
    map_item_feat = dict(zip(df_feat.index.astype(int), df_feat["tags"]))
    user_to_small = {int(user): idx for idx, user in enumerate(lbe_user.classes_)}
    item_to_small = {int(item): idx for idx, item in enumerate(lbe_item.classes_)}
    return {
        "mat_shape": tuple(mat.shape),
        "lbe_user": lbe_user,
        "lbe_item": lbe_item,
        "user_to_small": user_to_small,
        "item_to_small": item_to_small,
        "map_item_feat": map_item_feat,
    }


def read_chunks(csv_path: Path, chunksize: int):
    """按块读取交互日志所需字段。

    Args:
        csv_path (Path): 目标 CSV 路径。
        chunksize (int): 单块读取行数。

    Returns:
        TextFileReader: pandas 分块读取迭代器。
    """
    return pd.read_csv(
        csv_path,
        usecols=["user_id", "item_id", "timestamp"],
        chunksize=chunksize,
    )


def build_entropy_map(
    csv_path: Path,
    chunksize: int,
    entropy_window: Sequence[int],
    feature_level: bool,
    map_item_feat: Mapping[int, Sequence[int]],
    is_sorted: bool,
) -> Dict[Any, float]:
    """按 DORL 熵项定义构造“历史窗口 -> 熵值”的查表。

    Args:
        csv_path (Path): 用于统计熵分布的交互日志。
        chunksize (int): 分块读取大小。
        entropy_window (Sequence[int]): 熵窗口长度集合。
        feature_level (bool): 是否使用特征级熵。
        map_item_feat (Mapping[int, Sequence[int]]): 物品特征映射。
        is_sorted (bool): 是否对窗口内容排序。

    Returns:
        Dict[Any, float]: 历史窗口到熵值的映射。

    Raises:
        ValueError: 当 CSV 未按 `user_id, timestamp` 排序时抛出。
    """
    map_hist_count: MutableMapping[Any, MutableMapping[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    last_user = None
    last_timestamp = None
    history: List[int] = []
    row_idx = 0

    for chunk in tqdm(read_chunks(csv_path, chunksize), desc="Building entropy map", unit="chunk"):
        values = chunk[["user_id", "item_id", "timestamp"]].to_numpy()
        for user_raw, item_raw, timestamp_raw in values:
            user = int(user_raw)
            item = int(item_raw)
            timestamp = float(timestamp_raw)
            row_idx += 1

            if last_user is not None:
                if user < last_user:
                    raise ValueError(
                        f"CSV order check failed at row {row_idx}: user_id decreased from {last_user} to {user}"
                    )
                if user == last_user and last_timestamp is not None and timestamp < last_timestamp:
                    raise ValueError(
                        f"CSV order check failed at row {row_idx}: timestamp decreased within user {user}"
                    )

            if user != last_user:
                history = []
                last_user = user

            accumulate_entropy_counts_for_row(
                map_hist_count=map_hist_count,
                hist_tra=history,
                item=item,
                entropy_window=entropy_window,
                feature_level=feature_level,
                map_item_feat=map_item_feat,
                is_sorted=is_sorted,
            )
            history.append(item)
            last_timestamp = timestamp

    return finalize_entropy_map(map_hist_count)


def collect_filtered_step_values(
    csv_path: Path,
    chunksize: int,
    entropy_window: Sequence[int],
    feature_level: bool,
    map_item_feat: Mapping[int, Sequence[int]],
    is_sorted: bool,
    entropy_map: Mapping[Any, float],
    user_to_small: Mapping[int, int],
    item_to_small: Mapping[int, int],
    predicted_mat: np.ndarray,
    maxvar_mat: np.ndarray,
) -> Dict[str, Any]:
    """收集可对齐交互上的熵项、方差项与预测 reward。

    Args:
        csv_path (Path): 回放交互日志路径。
        chunksize (int): 分块读取大小。
        entropy_window (Sequence[int]): 熵窗口长度集合。
        feature_level (bool): 是否使用特征级熵。
        map_item_feat (Mapping[int, Sequence[int]]): 物品特征映射。
        is_sorted (bool): 是否对窗口内容排序。
        entropy_map (Mapping[Any, float]): 预先构造的熵查表。
        user_to_small (Mapping[int, int]): 原始 user_id 到 small-space 编码的映射。
        item_to_small (Mapping[int, int]): 原始 item_id 到 small-space 编码的映射。
        predicted_mat (np.ndarray): user model 预测 reward 矩阵。
        maxvar_mat (np.ndarray): ensemble 最大方差矩阵。

    Returns:
        Dict[str, Any]: 包含逐步熵值、方差值、预测 reward 及保留样本统计。

    Raises:
        ValueError: 当 CSV 未按 `user_id, timestamp` 排序时抛出。
    """
    entropy_values = array("f")
    maxvar_values = array("f")
    pred_values = array("f")
    filtered_users = set()

    last_user = None
    last_timestamp = None
    history: List[int] = []
    row_idx = 0
    kept_rows = 0

    for chunk in tqdm(read_chunks(csv_path, chunksize), desc="Collecting filtered penalties", unit="chunk"):
        values = chunk[["user_id", "item_id", "timestamp"]].to_numpy()
        for user_raw, item_raw, timestamp_raw in values:
            user = int(user_raw)
            item = int(item_raw)
            timestamp = float(timestamp_raw)
            row_idx += 1

            if last_user is not None:
                if user < last_user:
                    raise ValueError(
                        f"CSV order check failed at row {row_idx}: user_id decreased from {last_user} to {user}"
                    )
                if user == last_user and last_timestamp is not None and timestamp < last_timestamp:
                    raise ValueError(
                        f"CSV order check failed at row {row_idx}: timestamp decreased within user {user}"
                    )

            if user != last_user:
                history = []
                last_user = user

            if user in user_to_small and item in item_to_small:
                history_with_current = history + [item]
                entropy_t = compute_step_entropy(
                    history_with_current=history_with_current,
                    entropy_dict=entropy_map,
                    entropy_window=entropy_window,
                    feature_level=feature_level,
                    map_item_feat=map_item_feat,
                    is_sorted=is_sorted,
                )
                u_idx = user_to_small[user]
                i_idx = item_to_small[item]
                entropy_values.append(float(entropy_t))
                maxvar_values.append(float(maxvar_mat[u_idx, i_idx]))
                pred_values.append(float(predicted_mat[u_idx, i_idx]))
                history.append(item)
                kept_rows += 1
                filtered_users.add(user)

            last_timestamp = timestamp

    return {
        "entropy": np.asarray(entropy_values, dtype=np.float64),
        "maxvar": np.asarray(maxvar_values, dtype=np.float64),
        "pred_reward": np.asarray(pred_values, dtype=np.float64),
        "kept_rows": kept_rows,
        "kept_users": len(filtered_users),
    }


def collect_existing_lambda_history(log_dir: Path) -> Dict[str, Any]:
    """从历史 MOPO 日志中提取已有的 `lambda_variance`。

    Args:
        log_dir (Path): MOPO 日志目录。

    Returns:
        Dict[str, Any]: 包含去重后的 λ 列表及其日志路径映射。
    """
    if not log_dir.exists():
        return {"unique_lambdas": [], "logs_by_lambda": {}}

    logs_by_lambda: Dict[str, List[str]] = defaultdict(list)
    pattern = re.compile(r'"lambda_variance":\s*([0-9.eE+-]+)')
    for log_path in sorted(log_dir.glob("*.log")):
        text = log_path.read_text(errors="ignore")
        match = pattern.search(text)
        if not match:
            continue
        lambda_value = format_float(float(match.group(1)))
        logs_by_lambda[lambda_value].append(str(log_path))
    unique_lambdas = sorted(float(v) for v in logs_by_lambda.keys())
    return {
        "unique_lambdas": unique_lambdas,
        "logs_by_lambda": dict(logs_by_lambda),
    }


def build_manual_commands(lambdas: Sequence[float]) -> List[str]:
    """为推荐 λ 生成可直接手动执行的 MOPO 命令。"""
    commands = []
    for lambda_value in lambdas:
        lambda_str = format_float(lambda_value)
        message = f"MOPO_lambda_{lambda_str.replace('.', 'p').replace('-', 'm')}"
        commands.append(
            "CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=.:./src:./src/DeepCTR-Torch "
            "SWANLAB_MODE=offline conda run -n easyrl4rec "
            "python examples/advance/run_MOPO.py "
            "--env KuaiEnv-v0 --user_model_name DeepFM --read_message pointneg --seed 2023 "
            "--cuda <cuda_id> --which_tracker avg --window_size 3 --epoch 100 "
            "--batch-size 1024 --hidden-sizes 64 64 --leave_threshold 0 --num_leave_compute 1 "
            "--max_turn 30 --force_length 10 --lambda_entropy 0.0 "
            f"--lambda_variance {lambda_str} --message {message}"
        )
    return commands


def build_markdown(result: Dict[str, Any]) -> str:
    """将标定结果渲染为中文 Markdown 报告。"""
    penalty_rows = [
        ("entropy_t", result["stats"]["entropy"]),
        (
            f"{format_float(result['config']['lambda_entropy'])} * entropy_t",
            result["stats"]["lambda_entropy_entropy"],
        ),
        ("maxvar_t", result["stats"]["maxvar"]),
        ("0.05 * maxvar_t", result["stats"]["default_variance_penalty"]),
    ]
    reward_rows = result["reward_sanity"]

    lines = [
        "# MOPO `lambda_variance` 标定报告",
        "",
        "## 摘要",
        f"- 参考 DORL 熵项口径：`lambda_entropy={format_float(result['config']['lambda_entropy'])}`，"
        f"`entropy_window={result['config']['entropy_window']}`，"
        f"`feature_level={result['config']['feature_level']}`，"
        f"`is_sorted={result['config']['is_sorted']}`。",
        f"- 训练分布：`big_matrix_processed.csv` 中可映射到 KuaiEnv small-space 的交互，"
        f"实际回放源为 `{result['data_summary']['replay_csv_used']}`，"
        f"共 `{result['data_summary']['kept_rows']}` 条，涉及 `{result['data_summary']['kept_users']}` 个用户。",
        f"- 标定中心值：`lambda_star = {format_float(result['lambda_candidates']['lambda_star'])}`。",
        f"- 官方 5 点单种子方案：`{result['lambda_candidates']['official_mopo_sweep']}`。",
        "",
        "## 数据与对齐",
        f"- `big_csv`: `{result['config']['big_csv']}`",
        f"- `prediction_mat_path`: `{result['config']['prediction_mat_path']}`",
        f"- `var_mat_path`: `{result['config']['var_mat_path']}`",
        f"- `small_csv`: `{result['config']['small_csv']}`",
        f"- `predicted_mat.shape`: `{result['data_summary']['predicted_mat_shape']}`",
        f"- `maxvar_mat.shape`: `{result['data_summary']['maxvar_mat_shape']}`",
        f"- `KuaiEnv small-space shape`: `{result['data_summary']['small_space_shape']}`",
        f"- `replay_csv_used`: `{result['data_summary']['replay_csv_used']}`",
        "",
        "## 惩罚量级统计",
        "",
        "| 项 | count | mean | median | p95 | p99 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, stats in penalty_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(stats["count"]),
                    format_float(stats["mean"]),
                    format_float(stats["median"]),
                    format_float(stats["p95"]),
                    format_float(stats["p99"]),
                    format_float(stats["max"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 标定结果",
            f"- `lambda_mean = {format_float(result['lambda_candidates']['lambda_mean'])}`",
            f"- `lambda_median = {format_float(result['lambda_candidates']['lambda_median'])}`",
            f"- `lambda_star = {format_float(result['lambda_candidates']['lambda_star'])}`",
            f"- `official_mopo_sweep = {result['lambda_candidates']['official_mopo_sweep']}`",
            "",
            "## MOPO reward sanity check",
            "",
            "| lambda_variance | reward mean | reward p95 | reward max |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in reward_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    format_float(row["lambda_variance"]),
                    format_float(row["reward_mean"]),
                    format_float(row["reward_p95"]),
                    format_float(row["reward_max"]),
                ]
            )
            + " |"
        )

    history = result["historical_mopo_logs"]["unique_lambdas"]
    if history:
        lines.extend(
            [
                "",
                "## 历史 MOPO λ",
                f"- 现有日志里已发现的 `lambda_variance`：`{history}`",
            ]
        )
        if any(abs(v - 0.01) < 1e-12 for v in history):
            lines.append("- 其中 `0.01` 已经存在，可作为附录对比点，不占这次 5 个正式点位。")

    lines.extend(
        [
            "",
            "## 手动训练命令",
        ]
    )
    for command in result["manual_commands"]:
        lines.append(f"- `{command}`")

    if result["warnings"]:
        lines.extend(["", "## 警告"])
        for warning in result["warnings"]:
            lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """执行 MOPO `lambda_variance` 的离线标定流程。

    Returns:
        None

    Raises:
        ValueError: 当输入数据无法对齐到 KuaiEnv small-space 时抛出。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading KuaiEnv metadata and user-model matrices.")
    metadata = load_small_space_metadata()
    predicted_mat = pd.read_pickle(args.prediction_mat_path).astype(np.float64)
    maxvar_mat = pd.read_pickle(args.var_mat_path).astype(np.float64)

    warnings: List[str] = []
    if args.entropy_map_cache.exists():
        LOGGER.info("Loading cached entropy map from %s", args.entropy_map_cache)
        with args.entropy_map_cache.open("rb") as file:
            entropy_map = pickle.load(file)
    else:
        LOGGER.info("Building entropy map from %s", args.big_csv)
        entropy_map = build_entropy_map(
            csv_path=args.big_csv,
            chunksize=args.chunksize,
            entropy_window=args.entropy_window,
            feature_level=args.feature_level,
            map_item_feat=metadata["map_item_feat"],
            is_sorted=args.is_sorted,
        )
        args.entropy_map_cache.parent.mkdir(parents=True, exist_ok=True)
        with args.entropy_map_cache.open("wb") as file:
            pickle.dump(entropy_map, file)
        LOGGER.info("Saved entropy map cache to %s", args.entropy_map_cache)

    LOGGER.info("Collecting aligned replay samples from %s", args.big_csv)
    filtered = collect_filtered_step_values(
        csv_path=args.big_csv,
        chunksize=args.chunksize,
        entropy_window=args.entropy_window,
        feature_level=args.feature_level,
        map_item_feat=metadata["map_item_feat"],
        is_sorted=args.is_sorted,
        entropy_map=entropy_map,
        user_to_small=metadata["user_to_small"],
        item_to_small=metadata["item_to_small"],
        predicted_mat=predicted_mat,
        maxvar_mat=maxvar_mat,
    )
    replay_csv_used = args.big_csv
    if filtered["kept_rows"] == 0:
        LOGGER.warning("No aligned rows found in %s, falling back to %s", args.big_csv, args.small_csv)
        warnings.append(
            "big_matrix_processed.csv 中不存在同时落在 KuaiEnv small-space user/item 的交互；"
            "已自动回退到 small_matrix_processed.csv 作为回放分布。"
        )
        LOGGER.info("Collecting aligned replay samples from %s", args.small_csv)
        filtered = collect_filtered_step_values(
            csv_path=args.small_csv,
            chunksize=args.chunksize,
            entropy_window=args.entropy_window,
            feature_level=args.feature_level,
            map_item_feat=metadata["map_item_feat"],
            is_sorted=args.is_sorted,
            entropy_map=entropy_map,
            user_to_small=metadata["user_to_small"],
            item_to_small=metadata["item_to_small"],
            predicted_mat=predicted_mat,
            maxvar_mat=maxvar_mat,
        )
        replay_csv_used = args.small_csv
    if filtered["kept_rows"] == 0:
        raise ValueError(
            "small_matrix_processed.csv 也无法对齐到 KuaiEnv small-space，无法继续标定。"
        )

    entropy_values = filtered["entropy"]
    maxvar_values = filtered["maxvar"]
    pred_values = filtered["pred_reward"]
    lambda_entropy_values = args.lambda_entropy * entropy_values
    default_variance_penalty = DEFAULT_VARIANCE_REFERENCE_LAMBDA * maxvar_values

    entropy_stats = summarize(entropy_values)
    lambda_entropy_stats = summarize(lambda_entropy_values)
    maxvar_stats = summarize(maxvar_values)
    default_variance_stats = summarize(default_variance_penalty)

    lambda_mean = float(lambda_entropy_values.mean() / maxvar_values.mean())
    lambda_median = float(np.median(lambda_entropy_values) / np.median(maxvar_values))
    lambda_star = float(np.percentile(lambda_entropy_values, 95) / np.percentile(maxvar_values, 95))

    raw_lambdas = [
        0.0,
        DEFAULT_VARIANCE_REFERENCE_LAMBDA,
        0.1 * lambda_star,
        lambda_star / 3.0,
        lambda_star,
    ]
    official_lambdas = sorted({round_sig(value, sig=3) for value in raw_lambdas})

    predicted_min = float(np.min(predicted_mat))
    maxvar_max = float(np.max(maxvar_mat))
    reward_sanity = []
    for lambda_value in official_lambdas:
        rewards = (
            pred_values
            - lambda_value * maxvar_values
            - (predicted_min - lambda_value * maxvar_max)
        )
        reward_sanity.append(
            {
                "lambda_variance": float(lambda_value),
                "reward_mean": float(np.mean(rewards)),
                "reward_p95": float(np.percentile(rewards, 95)),
                "reward_max": float(np.max(rewards)),
            }
        )

    if lambda_star <= 0.05:
        warnings.append(
            "lambda_star <= 0.05，和预期不符；优先检查 CSV 顺序、ID 对齐或统计口径。"
        )

    result = {
        "config": {
            "big_csv": str(args.big_csv),
            "small_csv": str(args.small_csv),
            "prediction_mat_path": str(args.prediction_mat_path),
            "var_mat_path": str(args.var_mat_path),
            "default_variance_reference_lambda": DEFAULT_VARIANCE_REFERENCE_LAMBDA,
            "lambda_entropy": float(args.lambda_entropy),
            "entropy_window": list(args.entropy_window),
            "feature_level": bool(args.feature_level),
            "is_sorted": bool(args.is_sorted),
            "chunksize": int(args.chunksize),
            "entropy_map_cache": str(args.entropy_map_cache),
        },
        "data_summary": {
            "kept_rows": int(filtered["kept_rows"]),
            "kept_users": int(filtered["kept_users"]),
            "predicted_mat_shape": list(predicted_mat.shape),
            "maxvar_mat_shape": list(maxvar_mat.shape),
            "small_space_shape": list(metadata["mat_shape"]),
            "entropy_map_size": int(len(entropy_map)),
            "replay_csv_used": str(replay_csv_used),
        },
        "stats": {
            "entropy": entropy_stats,
            "lambda_entropy_entropy": lambda_entropy_stats,
            "maxvar": maxvar_stats,
            "default_variance_penalty": default_variance_stats,
        },
        "lambda_candidates": {
            "lambda_mean": float(lambda_mean),
            "lambda_median": float(lambda_median),
            "lambda_star": float(lambda_star),
            "official_mopo_sweep": [float(v) for v in official_lambdas],
        },
        "reward_sanity": reward_sanity,
        "historical_mopo_logs": collect_existing_lambda_history(args.mopo_log_dir),
        "manual_commands": build_manual_commands(official_lambdas),
        "warnings": warnings,
    }

    LOGGER.info("Writing calibration outputs to %s and %s", args.output_md, args.output_json)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.output_md.write_text(build_markdown(result), encoding="utf-8")

    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")
    print(f"lambda_star={lambda_star:.6g}")
    print(f"official_mopo_sweep={official_lambdas}")


if __name__ == "__main__":
    main()
