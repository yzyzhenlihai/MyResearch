#!/usr/bin/env python3
"""在固定 Top-100 高奖励用户上评测 DARLR 的 Top item oracle 子集。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""仓库根目录。"""

POLICY_EXAMPLES_DIR = REPOSITORY_ROOT / "examples" / "policy"
"""包含历史 ``policy_utils`` 模块的目录。"""

for import_path in (REPOSITORY_ROOT, POLICY_EXAMPLES_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

os.environ.setdefault("SWANLAB_MODE", "disabled")

from analysis.kuai_user_selection import (  # noqa: E402
    compute_item_reward_statistics,
    dataframe_to_markdown,
)
from examples.advance.run_DARLR import (  # noqa: E402
    setup_policy_model,
    validate_darlr_args,
)
from policy_utils import prepare_user_model, setup_state_tracker  # noqa: E402
from scripts.evaluate_darlr_selected_users import (  # noqa: E402
    PAPER_SINGLE_STEP_REWARD,
    build_darlr_arguments,
    build_fixed_user_envs,
    compute_file_sha256,
    evaluate_group,
    load_checkpoint,
    load_mcd_inputs,
    load_prediction_matrices,
    load_selected_users,
    seed_everything,
)
from src.core.collector.collector_set import CollectorSet  # noqa: E402
from src.core.util.data import get_true_env  # noqa: E402


LOGGER = logging.getLogger(__name__)

DEFAULT_TOP_ITEM_COUNTS = (50, 10)
"""用户要求评测的默认 Top item 数量。"""

ANNOTATED_PREVIOUS_NX0_REWARD = 0.985068
"""用户标注的上一轮 Top-100 用户完整候选 NX0 单步 reward。"""


def build_parser() -> argparse.ArgumentParser:
    """构造 Top item oracle 交互评测参数解析器。

    Returns:
        argparse.ArgumentParser: 仅包含本实验专用参数的解析器。
    """

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--user-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--top-item-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOP_ITEM_COUNTS),
    )
    parser.add_argument(
        "--paper-single-step-reward",
        type=float,
        default=PAPER_SINGLE_STEP_REWARD,
    )
    parser.add_argument(
        "--limit-users",
        type=int,
        default=None,
        help="仅用于冒烟验证；默认使用用户表中的全部用户。",
    )
    return parser


def validate_top_item_counts(
    top_item_counts: Sequence[int],
    item_count: int,
    force_length: int,
) -> list[int]:
    """校验并保留用户指定的 Top item 数量顺序。

    Args:
        top_item_counts (Sequence[int]): 待评测的候选集合规模。
        item_count (int): 环境完整 item 数量。
        force_length (int): NX 强制评测步数。

    Returns:
        list[int]: 不重复且合法的整数规模。

    Raises:
        ValueError: 当规模重复、越界或小于强制评测步数时抛出。
    """

    normalized = [int(count) for count in top_item_counts]
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("top_item_counts must be a non-empty unique sequence.")
    for count in normalized:
        if count < force_length or count > item_count:
            raise ValueError(
                f"Each top item count must be in [{force_length}, {item_count}]."
            )
    return normalized


def build_item_subset_definitions(
    item_statistics: pd.DataFrame,
    top_item_counts: Sequence[int],
) -> list[tuple[str, np.ndarray | None]]:
    """生成完整候选基线与各 Top item oracle 候选集合。

    Args:
        item_statistics (pd.DataFrame): 已按真实 reward 均值排序的 item 表。
        top_item_counts (Sequence[int]): 需要截取的 Top-K 规模。

    Returns:
        list[tuple[str, np.ndarray | None]]: 组名及矩阵 item 下标；基线用
        ``None`` 表示完整候选空间。
    """

    definitions: list[tuple[str, np.ndarray | None]] = [
        ("all_items_baseline", None)
    ]
    for count in top_item_counts:
        indexes = item_statistics.head(count)["matrix_item_index"].to_numpy(
            dtype=np.int64
        )
        definitions.append((f"top_{count}_items_oracle", indexes))
    return definitions


def add_comparison_fields(metrics: list[dict[str, Any]]) -> None:
    """原位加入相对完整候选基线的逐协议 R_each 差值。

    Args:
        metrics (list[dict[str, Any]]): 所有候选组和协议的聚合指标。

    Returns:
        None: 每行新增 ``R_each_delta_vs_all_items``。

    Raises:
        ValueError: 当任一协议缺少完整候选基线时抛出。
    """

    baseline = {
        row["protocol"]: float(row["R_each"])
        for row in metrics
        if row["group"] == "all_items_baseline"
    }
    for row in metrics:
        protocol = row["protocol"]
        if protocol not in baseline:
            raise ValueError(f"Missing all-items baseline for protocol {protocol}.")
        row["R_each_delta_vs_all_items"] = float(row["R_each"]) - baseline[protocol]


def write_artifacts(
    output_dir: Path,
    metrics: list[dict[str, Any]],
    per_user_metrics: pd.DataFrame,
    selected_user_table: pd.DataFrame,
    item_statistics: pd.DataFrame,
    top_item_counts: Sequence[int],
    manifest: dict[str, Any],
) -> dict[str, Path]:
    """写出 item 排序、交互指标、运行清单和中文实验摘要。

    Args:
        output_dir (Path): 输出目录。
        metrics (list[dict[str, Any]]): 各候选组的协议聚合指标。
        per_user_metrics (pd.DataFrame): 逐用户 episode 指标。
        selected_user_table (pd.DataFrame): 固定测试用户表。
        item_statistics (pd.DataFrame): 全 item oracle 排序表。
        top_item_counts (Sequence[int]): 导出的 Top-K 规模。
        manifest (dict[str, Any]): checkpoint、环境和筛选口径清单。

    Returns:
        dict[str, Path]: 核心输出文件路径。

    Raises:
        OSError: 当目录或文件不可写时由底层接口抛出。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation_metrics.csv"
    per_user_path = output_dir / "per_user_episode_metrics.csv"
    selected_users_path = output_dir / "selected_users.csv"
    all_items_path = output_dir / "all_item_reward_statistics.csv"
    summary_json_path = output_dir / "evaluation_summary.json"
    manifest_path = output_dir / "run_manifest.json"
    markdown_path = output_dir / "interaction_evaluation_summary.md"

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(metrics_path, index=False)
    per_user_metrics.to_csv(per_user_path, index=False)
    selected_user_table.to_csv(selected_users_path, index=False)
    item_statistics.to_csv(all_items_path, index=False)
    top_item_paths: dict[int, Path] = {}
    for count in top_item_counts:
        top_item_path = output_dir / f"top_{count}_items_by_mean_reward.csv"
        item_statistics.head(count).to_csv(top_item_path, index=False)
        top_item_paths[count] = top_item_path
    with summary_json_path.open("w", encoding="utf-8") as summary_file:
        json.dump(metrics, summary_file, ensure_ascii=False, indent=2)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

    display_columns = [
        "group",
        "protocol",
        "allowed_item_count",
        "R_tra",
        "R_each",
        "R_each_delta_vs_all_items",
        "Length",
        "MCD",
        "reaches_paper_R_each",
    ]
    display_frame = metrics_frame[display_columns].copy()
    for column in (
        "R_tra",
        "R_each",
        "R_each_delta_vs_all_items",
        "Length",
        "MCD",
    ):
        display_frame[column] = display_frame[column].map(lambda value: f"{value:.6f}")

    oracle_rows = metrics_frame.loc[
        metrics_frame["group"] != "all_items_baseline"
    ]
    baseline_nx0 = metrics_frame.loc[
        (metrics_frame["group"] == "all_items_baseline")
        & (metrics_frame["protocol"] == "NX_0")
    ].iloc[0]
    best_row = oracle_rows.loc[oracle_rows["R_each"].idxmax()]
    reaches_target = bool(oracle_rows["reaches_paper_R_each"].any())
    target_conclusion = "至少一组达到" if reaches_target else "所有组均未达到"
    markdown = "\n".join(
        [
            "# DARLR Top item oracle 子集交互评测",
            "",
            "## 口径",
            "",
            "- 用户固定为上一轮按 `reward >= 1.26` 占比筛出的 Top-100。",
            "- item 按这 100 名测试用户的真实 reward 均值降序筛选 Top50/Top10。",
            "- 该 item 排序使用测试标签，是有意构造的事后 oracle，不是无泄漏评估。",
            f"- 完整 item 独立复跑的 NX0 `R_each={baseline_nx0['R_each']:.6f}`；"
            f"上一轮标注值为 `{ANNOTATED_PREVIOUS_NX0_REWARD:.6f}`，"
            f"本次相差 `{baseline_nx0['R_each'] - ANNOTATED_PREVIOUS_NX0_REWARD:+.6f}`。",
            "- NX0 禁止重复推荐；Top10 因而最多交互 10 步。FB 仍使用原始最大 30 步。",
            "- 受限组不计 reward 的随机初始上下文从 Top item 集合之外采样，避免占用 NX 候选。",
            "- 每个用户、每个协议恰好评测一个 episode，checkpoint 和随机种子不变。",
            "",
            "## 结果",
            "",
            dataframe_to_markdown(display_frame),
            "",
            "## 结论",
            "",
            f"Top item oracle 组中最高 `R_each={best_row['R_each']:.6f}`，来自 "
            f"`{best_row['group']}/{best_row['protocol']}`；{target_conclusion}论文目标 `1.26`。",
            "这些结果只能量化测试 item 事后筛选能够造成的指标增益，不能证明论文作者采用过该筛选。",
            "",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    paths = {
        "metrics_csv": metrics_path,
        "per_user_csv": per_user_path,
        "selected_users_csv": selected_users_path,
        "all_items_csv": all_items_path,
        "summary_json": summary_json_path,
        "manifest_json": manifest_path,
        "summary_markdown": markdown_path,
    }
    paths.update(
        {f"top_{count}_items_csv": path for count, path in top_item_paths.items()}
    )
    return paths


def main() -> None:
    """加载既有 checkpoint 并完成全 item、Top50 和 Top10 受控评测。

    Returns:
        None: 结果写入 ``--output-dir`` 并将聚合表打印到标准输出。

    Raises:
        ValueError: 当配置、用户映射或候选规模不合法时抛出。
        OSError: 当必要输入不可读或输出不可写时由底层接口抛出。
        RuntimeError: 当 checkpoint 不兼容或交互候选耗尽时抛出。
    """

    experiment_args, _ = build_parser().parse_known_args()
    darlr_args = build_darlr_arguments()
    validate_darlr_args(darlr_args)
    if darlr_args.env != "KuaiEnv-v0":
        raise ValueError("This evaluator currently supports only KuaiEnv-v0.")
    if experiment_args.limit_users is not None and experiment_args.limit_users <= 0:
        raise ValueError("--limit-users must be positive when provided.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    selected_user_table = load_selected_users(experiment_args.user_table)
    if experiment_args.limit_users is not None:
        selected_user_table = selected_user_table.head(
            experiment_args.limit_users
        ).copy()
    selected_user_indexes = selected_user_table[
        "matrix_user_index"
    ].to_numpy(dtype=np.int64)
    darlr_args.test_num = len(selected_user_indexes)
    seed_everything(darlr_args.seed)

    LOGGER.info("加载 KuaiRec、DeepFM ensemble 与 DARLR checkpoint 架构。")
    ensemble_models = prepare_user_model(darlr_args)
    true_environment, dataset, environment_kwargs = get_true_env(darlr_args)
    expected_raw_user_ids = true_environment.lbe_user.inverse_transform(
        selected_user_indexes
    ).astype(np.int64)
    if not np.array_equal(
        expected_raw_user_ids,
        selected_user_table["user_id"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Selected user IDs do not match matrix_user_index values.")

    top_item_counts = validate_top_item_counts(
        experiment_args.top_item_counts,
        true_environment.mat.shape[1],
        darlr_args.force_length,
    )
    raw_item_ids = true_environment.lbe_item.inverse_transform(
        np.arange(true_environment.mat.shape[1], dtype=np.int64)
    )
    item_statistics = compute_item_reward_statistics(
        true_environment.mat,
        raw_item_ids,
        selected_user_indexes,
        experiment_args.paper_single_step_reward,
    )
    subset_definitions = build_item_subset_definitions(
        item_statistics,
        top_item_counts,
    )

    baseline_envs = build_fixed_user_envs(
        selected_user_indexes,
        environment_kwargs,
        darlr_args.force_length,
        darlr_args.seed,
    )
    predicted_matrix, variance_matrix = load_prediction_matrices(ensemble_models)
    state_tracker = setup_state_tracker(
        darlr_args,
        ensemble_models,
        true_environment,
        train_envs=None,
        test_envs_dict=baseline_envs,
    )
    rec_policy, _, baseline_collectors, _ = setup_policy_model(
        darlr_args,
        state_tracker,
        train_envs=None,
        test_envs_dict=baseline_envs,
        predicted_mat=predicted_matrix,
        maxvar_mat=variance_matrix,
        build_train_collector=False,
    )
    item_features, feature_domination = load_mcd_inputs(dataset)

    all_metrics: list[dict[str, Any]] = []
    per_user_frames: list[pd.DataFrame] = []
    for group_name, allowed_item_indexes in subset_definitions:
        LOGGER.info("开始评测候选组：%s。", group_name)
        if allowed_item_indexes is None:
            # 与上一轮固定用户评测复用完全相同的构造路径，保证标注基线可复现。
            collectors = baseline_collectors
        else:
            group_envs = build_fixed_user_envs(
                selected_user_indexes,
                environment_kwargs,
                darlr_args.force_length,
                darlr_args.seed,
                allowed_item_indexes=allowed_item_indexes,
            )
            collectors = CollectorSet(
                rec_policy,
                group_envs,
                darlr_args.buffer_size,
                len(selected_user_indexes),
                exploration_noise=darlr_args.exploration_noise,
                force_length=darlr_args.force_length,
            )
        load_checkpoint(
            experiment_args.checkpoint,
            rec_policy,
            state_tracker,
            darlr_args.device,
        )
        rec_policy.set_allowed_item_indexes(allowed_item_indexes)
        seed_everything(darlr_args.seed)
        group_metrics, group_per_user = evaluate_group(
            group_name,
            selected_user_indexes,
            collectors,
            rec_policy,
            item_features,
            feature_domination,
            true_environment.lbe_item,
            true_environment.lbe_user,
            darlr_args.top_rate,
            experiment_args.paper_single_step_reward,
        )
        allowed_item_count = (
            true_environment.mat.shape[1]
            if allowed_item_indexes is None
            else len(allowed_item_indexes)
        )
        eligible_matrix = true_environment.mat[selected_user_indexes]
        if allowed_item_indexes is not None:
            eligible_matrix = eligible_matrix[:, allowed_item_indexes]
        for metric in group_metrics:
            metric["allowed_item_count"] = int(allowed_item_count)
            metric["eligible_matrix_mean_reward"] = float(eligible_matrix.mean())
            metric["eligible_matrix_reward_ge_target_rate"] = float(
                (eligible_matrix >= experiment_args.paper_single_step_reward).mean()
            )
        group_per_user["allowed_item_count"] = int(allowed_item_count)
        all_metrics.extend(group_metrics)
        per_user_frames.append(group_per_user)

    add_comparison_fields(all_metrics)
    device_name = "CPU"
    if torch.cuda.is_available() and str(darlr_args.device).startswith("cuda"):
        device_name = torch.cuda.get_device_name(darlr_args.device)
    manifest = {
        "checkpoint": str(experiment_args.checkpoint),
        "checkpoint_sha256": compute_file_sha256(experiment_args.checkpoint),
        "user_table": str(experiment_args.user_table),
        "selected_user_count": int(len(selected_user_indexes)),
        "user_selection_definition": "reward >= 1.26 rate descending",
        "item_ranking_definition": (
            "selected-user true mean_reward descending, then "
            "reward_ge_target_rate descending, then item_id ascending"
        ),
        "item_ranking_is_test_label_oracle": True,
        "restricted_group_initial_context": (
            "random item from outside the allowed Top-K set; it receives no reward"
        ),
        "top_item_counts": [int(count) for count in top_item_counts],
        "paper_single_step_reward": float(experiment_args.paper_single_step_reward),
        "seed": int(darlr_args.seed),
        "device": str(darlr_args.device),
        "device_name": device_name,
        "environment": {
            "name": darlr_args.env,
            "max_turn": int(darlr_args.max_turn),
            "num_leave_compute": int(darlr_args.num_leave_compute),
            "leave_threshold": float(darlr_args.leave_threshold),
            "force_length": int(darlr_args.force_length),
            "random_init": bool(darlr_args.random_init),
            "top_10_nx0_effective_max_turn": min(
                int(darlr_args.max_turn),
                min(top_item_counts),
            ),
        },
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    paths = write_artifacts(
        experiment_args.output_dir,
        all_metrics,
        pd.concat(per_user_frames, ignore_index=True),
        selected_user_table,
        item_statistics,
        top_item_counts,
        manifest,
    )
    LOGGER.info("聚合指标：%s", paths["metrics_csv"])
    print(pd.DataFrame(all_metrics).to_string(index=False))


if __name__ == "__main__":
    main()
