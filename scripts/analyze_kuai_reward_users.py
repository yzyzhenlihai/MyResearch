#!/usr/bin/env python3
"""分析 KuaiRec 真实 reward 矩阵并导出高奖励 Top-K 用户。"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""仓库根目录，用于支持从任意工作目录直接运行脚本。"""

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.kuai_user_selection import (
    DEFAULT_ROUND_DECIMALS,
    DEFAULT_TARGET_REWARD,
    DEFAULT_TOP_K,
    build_reward_selection_summary,
    compute_user_reward_statistics,
    write_reward_selection_artifacts,
)
from src.core.envs.KuaiRec.KuaiData import KuaiData


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_ROOT = Path("results_analysis")
"""默认实验产物根目录。"""


def build_argument_parser() -> argparse.ArgumentParser:
    """构造用户 reward 统计命令行解析器。

    Returns:
        argparse.ArgumentParser: 已注册目标值、Top-K 和输出目录参数的解析器。
    """

    parser = argparse.ArgumentParser(
        description="统计 KuaiRec 中 reward 高于论文单步均值目标的用户。"
    )
    parser.add_argument(
        "--target-reward",
        type=float,
        default=DEFAULT_TARGET_REWARD,
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=DEFAULT_ROUND_DECIMALS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="默认在 results_analysis 下创建带 UTC 时间戳的目录。",
    )
    return parser


def resolve_output_dir(requested_dir: Path | None) -> Path:
    """解析本次分析的输出目录。

    Args:
        requested_dir (Path | None): 用户显式传入的目录；为空时使用 UTC 时间戳。

    Returns:
        Path: 本次分析应写入的目录。
    """

    if requested_dir is not None:
        return requested_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"{timestamp}_kuai-high-reward-users"


def main() -> None:
    """加载环境真实矩阵，计算统计并写出分析产物。

    Returns:
        None: 输出路径通过日志和标准输出展示。

    Raises:
        ValueError: 当命令行统计参数不合法时抛出。
        OSError: 当数据或输出文件无法访问时由底层接口抛出。
    """

    parser = build_argument_parser()
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    LOGGER.info("加载 KuaiRec 环境真实 reward 矩阵。")
    reward_matrix, user_encoder, _ = KuaiData.load_mat()
    statistics = compute_user_reward_statistics(
        reward_matrix=reward_matrix,
        raw_user_ids=user_encoder.classes_,
        target_reward=args.target_reward,
        round_decimals=args.round_decimals,
    )
    summary = build_reward_selection_summary(
        reward_matrix=reward_matrix,
        statistics=statistics,
        top_k=args.top_k,
    )
    output_dir = resolve_output_dir(args.output_dir)
    paths = write_reward_selection_artifacts(
        output_dir=output_dir,
        statistics=statistics,
        summary=summary,
        top_k=args.top_k,
    )
    LOGGER.info("Top-%d 用户表：%s", args.top_k, paths["top_users_csv"])
    print(output_dir)


if __name__ == "__main__":
    main()
