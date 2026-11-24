"""构造与 DORL-MAC 兼容的 KuaiRec big/small 离线 RL 轨迹。"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.our_model.data.kuairand_trajectory_builder import (  # noqa: E402
    DEFAULT_PADDING_STD,
    DEFAULT_WINDOW_SIZE,
    build_and_save_trajectory_dataset,
)


DEFAULT_SPLIT = "train"
"""用户当前所需 big matrix 对应的默认数据划分。"""

DEFAULT_SEED = 2023
"""padding embedding 的默认可复现随机种子。"""

SPLIT_NAMES = ("train", "test")
"""KuaiRec 支持的逻辑数据划分；分别对应 big 与 small matrix。"""


@dataclass(frozen=True)
class DatasetConfig:
    """保存一个 KuaiRec 数据划分的默认输入、embedding 与输出路径。"""

    input_path: Path
    embedding_path: Path
    output_path: Path


SPLIT_CONFIGS = {
    "train": DatasetConfig(
        input_path=(
            PROJECT_ROOT / "data/KuaiRec/data_raw/big_matrix_processed.csv"
        ),
        embedding_path=(
            PROJECT_ROOT
            / "saved_models/KuaiEnv-v0/DeepFM/embeddings/"
            "[pointneg]_emb_item_M0.pt"
        ),
        output_path=(
            PROJECT_ROOT
            / "data/KuaiRec/data_processed/DM_KuaiEnv-v0_data.pkl"
        ),
    ),
    "test": DatasetConfig(
        input_path=(
            PROJECT_ROOT / "data/KuaiRec/data_raw/small_matrix_processed.csv"
        ),
        embedding_path=(
            PROJECT_ROOT
            / "saved_models/KuaiEnv-v0/DeepFM/embeddings/"
            "[pointneg]_emb_item_val_M0.pt"
        ),
        output_path=(
            PROJECT_ROOT
            / "data/KuaiRec/data_processed/DM_KuaiEnv-v0_small_data.pkl"
        ),
    ),
}
"""KuaiRec train(big)/test(small) 的默认构建配置。"""


def build_parser() -> argparse.ArgumentParser:
    """创建 KuaiRec 轨迹数据集构造参数解析器。

    Returns:
        argparse.ArgumentParser: 配置完成的命令行解析器。
    """

    parser = argparse.ArgumentParser(
        description=(
            "将 KuaiRec big/small_matrix_processed.csv 转换为按 user_id "
            "组织、与 DORL-MAC action chunk loader 兼容的离线 RL 轨迹。"
        )
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_NAMES,
        default=DEFAULT_SPLIT,
        help="train 对应 big matrix，test 对应 small matrix；默认 train。",
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=None,
        help="可选输入 CSV；未指定时由 --split 自动选择。",
    )
    parser.add_argument(
        "--embedding_path",
        type=Path,
        default=None,
        help="可选 item embedding；未指定时按 big/small item 空间选择。",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="可选输出 pickle；未指定时由 --split 自动选择。",
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=None,
        help="可选摘要 JSON；默认在 pickle 文件名后追加 .summary.json。",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有 pickle 和摘要；默认拒绝覆盖。",
    )
    return parser


def resolve_dataset_config(
    split: str,
    input_path: Optional[Path] = None,
    embedding_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> DatasetConfig:
    """根据数据划分和可选覆盖参数解析 KuaiRec 构建配置。

    Args:
        split (str): `train`（big matrix）或 `test`（small matrix）。
        input_path (Optional[Path]): 用户覆盖的输入 CSV 路径。
        embedding_path (Optional[Path]): 用户覆盖的 embedding 路径。
        output_path (Optional[Path]): 用户覆盖的输出 pickle 路径。

    Returns:
        DatasetConfig: 解析后的构建配置。

    Raises:
        ValueError: `split` 不受支持时抛出。
    """

    if split not in SPLIT_CONFIGS:
        supported = ", ".join(SPLIT_NAMES)
        raise ValueError(f"不支持 KuaiRec split={split!r}；可选值：{supported}。")
    default_config = SPLIT_CONFIGS[split]
    return DatasetConfig(
        input_path=input_path or default_config.input_path,
        embedding_path=embedding_path or default_config.embedding_path,
        output_path=output_path or default_config.output_path,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """执行 KuaiRec 离线 RL 轨迹构造流程。

    Args:
        argv (Optional[Sequence[str]]): 可选命令行参数序列；为空时读取
            `sys.argv`。

    Returns:
        int: 成功时返回 0。

    Raises:
        FileExistsError: 输出存在且未传 `--overwrite` 时抛出。
        FileNotFoundError: 输入 CSV 或 embedding 不存在时抛出。
        ValueError: 数据、embedding、参数或输出轨迹校验失败时抛出。
        OSError: 输出写入失败时抛出。
    """

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = resolve_dataset_config(
        split=args.split,
        input_path=args.input_path,
        embedding_path=args.embedding_path,
        output_path=args.output_path,
    )
    build_and_save_trajectory_dataset(
        dataset_name="KuaiRec",
        split=args.split,
        input_path=config.input_path,
        embedding_path=config.embedding_path,
        output_path=config.output_path,
        summary_path=args.summary_path,
        time_column="timestamp",
        reward_column="watch_ratio_normed",
        window_size=args.window_size,
        seed=args.seed,
        padding_std=args.padding_std,
        max_users=args.max_users,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
