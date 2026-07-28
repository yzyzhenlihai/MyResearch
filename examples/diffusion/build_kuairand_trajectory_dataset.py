"""构造与 KuaiRec 轨迹 pickle 格式兼容的 KuaiRand 离线 RL 数据集。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.our_model.data.kuairand_trajectory_builder import (
    DEFAULT_PADDING_STD,
    DEFAULT_WINDOW_SIZE,
    build_kuairand_trajectories,
    compute_sha256,
    load_item_embeddings,
    load_kuairand_interactions,
    save_summary,
    save_trajectory_dataset,
    validate_trajectories,
)


DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data/KuaiRand_Pure/data_raw/test_processed.csv"
)
DEFAULT_EMBEDDING_PATH = (
    PROJECT_ROOT
    / "saved_models/KuaiRand-v0/DeepFM/embeddings/"
    "[pointneg]_emb_item_val_M0.pt"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/KuaiRand_Pure/data_processed/"
    "DM_KuaiRand-v0_test_data.pkl"
)
DEFAULT_SEED = 2023
"""与仓库 KuaiRand 示例一致的默认随机种子。"""


def build_parser() -> argparse.ArgumentParser:
    """创建 KuaiRand 轨迹数据集构造参数解析器。

    Returns:
        argparse.ArgumentParser: 配置完成的命令行解析器。
    """

    parser = argparse.ArgumentParser(
        description=(
            "将 KuaiRand test_processed.csv 转换为按 user_id 组织、"
            "与 DM_KuaiEnv-v0_small_data.pkl 兼容的离线 RL 轨迹。"
        )
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="KuaiRand test_processed.csv 路径。",
    )
    parser.add_argument(
        "--embedding_path",
        type=Path,
        default=DEFAULT_EMBEDDING_PATH,
        help="DeepFM 验证集 item embedding tensor 路径。",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="输出轨迹 pickle 路径。",
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=None,
        help="可选摘要 JSON 路径；默认与输出 pickle 同名并追加 .summary.json。",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="StateTrackerAvg 历史窗口长度。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="padding item embedding 随机种子。",
    )
    parser.add_argument(
        "--padding_std",
        type=float,
        default=DEFAULT_PADDING_STD,
        help="padding item embedding 正态分布标准差。",
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=None,
        help="仅构造前 N 个用户，供 smoke test 使用；默认构造全量数据。",
    )
    return parser


def resolve_summary_path(output_path: Path, summary_path: Optional[Path]) -> Path:
    """解析摘要文件路径。

    Args:
        output_path (Path): 轨迹 pickle 输出路径。
        summary_path (Optional[Path]): 用户显式提供的摘要路径。

    Returns:
        Path: 最终摘要 JSON 路径。
    """

    if summary_path is not None:
        return summary_path
    return output_path.with_suffix(output_path.suffix + ".summary.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """执行 KuaiRand 离线 RL 轨迹构造流程。

    Args:
        argv (Optional[Sequence[str]]): 可选命令行参数序列；为空时读取
            `sys.argv`。

    Returns:
        int: 成功时返回 0。

    Raises:
        FileNotFoundError: 输入 CSV 或 embedding 不存在时抛出。
        ValueError: 数据、embedding、参数或输出轨迹校验失败时抛出。
        OSError: 输出写入失败时抛出。
    """

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    input_path = args.input_path.resolve()
    embedding_path = args.embedding_path.resolve()
    output_path = args.output_path.resolve()
    summary_path = resolve_summary_path(
        output_path=output_path,
        summary_path=args.summary_path,
    ).resolve()

    logger.info("读取 KuaiRand 交互：%s", input_path)
    interactions = load_kuairand_interactions(
        input_path=input_path,
        max_users=args.max_users,
    )
    logger.info(
        "交互加载完成：users=%d, transitions=%d",
        interactions["user_id"].nunique(),
        len(interactions),
    )

    logger.info("读取 user model item embedding：%s", embedding_path)
    item_embeddings = load_item_embeddings(embedding_path)
    logger.info("Embedding shape=%s", item_embeddings.shape)

    logger.info(
        "按 StateTrackerAvg 语义构造轨迹：window_size=%d, seed=%d",
        args.window_size,
        args.seed,
    )
    trajectories = build_kuairand_trajectories(
        interactions=interactions,
        item_embeddings=item_embeddings,
        window_size=args.window_size,
        seed=args.seed,
        padding_std=args.padding_std,
    )
    statistics = validate_trajectories(
        trajectories,
        expected_action_dim=int(item_embeddings.shape[1]),
    )
    logger.info("轨迹校验通过：%s", statistics)

    save_trajectory_dataset(trajectories, output_path)
    summary = {
        **statistics,
        "input_path": str(input_path),
        "input_sha256": compute_sha256(input_path),
        "embedding_path": str(embedding_path),
        "embedding_sha256": compute_sha256(embedding_path),
        "output_path": str(output_path),
        "output_sha256": compute_sha256(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "reward_column": "is_click",
        "window_size": int(args.window_size),
        "seed": int(args.seed),
        "padding_std": float(args.padding_std),
        "max_users": args.max_users,
        "terminal_rule": "only_last_transition_true",
    }
    save_summary(summary, summary_path)
    logger.info("数据集已保存：%s", output_path)
    logger.info("构造摘要已保存：%s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
