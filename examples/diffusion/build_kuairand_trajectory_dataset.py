"""构造与 DORL-MAC 兼容的 KuaiRand train/test 离线 RL 轨迹。"""

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


DEFAULT_SPLIT = "test"
"""保持旧入口行为的默认数据划分。"""

DEFAULT_SEED = 2023
"""与仓库 KuaiRand 示例一致的默认随机种子。"""

DEFAULT_EMBEDDING_PATH = (
    PROJECT_ROOT
    / "saved_models/KuaiRand-v0/DeepFM/embeddings/"
    "[pointneg]_emb_item_val_M0.pt"
)
"""覆盖 KuaiRand train/test 统一 item ID 空间的 embedding。"""

SPLIT_INPUT_PATHS = {
    "train": PROJECT_ROOT / "data/KuaiRand_Pure/data_raw/train_processed.csv",
    "test": PROJECT_ROOT / "data/KuaiRand_Pure/data_raw/test_processed.csv",
}
"""KuaiRand 各数据划分的默认输入路径。"""

SPLIT_OUTPUT_PATHS = {
    "train": (
        PROJECT_ROOT
        / "data/KuaiRand_Pure/data_processed/DM_KuaiRand-v0_train_data.pkl"
    ),
    "test": (
        PROJECT_ROOT
        / "data/KuaiRand_Pure/data_processed/DM_KuaiRand-v0_test_data.pkl"
    ),
}
"""KuaiRand 各数据划分的默认输出路径。"""

DEFAULT_INPUT_PATH = SPLIT_INPUT_PATHS[DEFAULT_SPLIT]
"""兼容旧调用方的默认 test 输入路径别名。"""

DEFAULT_OUTPUT_PATH = SPLIT_OUTPUT_PATHS[DEFAULT_SPLIT]
"""兼容旧调用方的默认 test 输出路径别名。"""


@dataclass(frozen=True)
class DatasetPaths:
    """保存一次 KuaiRand 构建任务解析后的输入、embedding 与输出路径。"""

    input_path: Path
    embedding_path: Path
    output_path: Path


def build_parser() -> argparse.ArgumentParser:
    """创建 KuaiRand 轨迹数据集构造参数解析器。

    Returns:
        argparse.ArgumentParser: 配置完成的命令行解析器。
    """

    parser = argparse.ArgumentParser(
        description=(
            "将 KuaiRand train/test_processed.csv 转换为按 user_id 组织、"
            "与 DORL-MAC action chunk loader 兼容的离线 RL 轨迹。"
        )
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_INPUT_PATHS),
        default=DEFAULT_SPLIT,
        help="数据划分；默认 test 以保持旧脚本行为。",
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
        help="可选 item embedding；默认使用统一的验证 item embedding。",
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


def resolve_dataset_paths(
    split: str,
    input_path: Optional[Path] = None,
    embedding_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> DatasetPaths:
    """根据数据划分和可选覆盖参数解析 KuaiRand 构建路径。

    Args:
        split (str): `train` 或 `test`。
        input_path (Optional[Path]): 用户覆盖的输入 CSV 路径。
        embedding_path (Optional[Path]): 用户覆盖的 embedding 路径。
        output_path (Optional[Path]): 用户覆盖的输出 pickle 路径。

    Returns:
        DatasetPaths: 解析后的三类路径。

    Raises:
        ValueError: `split` 不受支持时抛出。
    """

    if split not in SPLIT_INPUT_PATHS:
        supported = ", ".join(sorted(SPLIT_INPUT_PATHS))
        raise ValueError(f"不支持 KuaiRand split={split!r}；可选值：{supported}。")
    return DatasetPaths(
        input_path=input_path or SPLIT_INPUT_PATHS[split],
        embedding_path=embedding_path or DEFAULT_EMBEDDING_PATH,
        output_path=output_path or SPLIT_OUTPUT_PATHS[split],
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """执行 KuaiRand 离线 RL 轨迹构造流程。

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
    paths = resolve_dataset_paths(
        split=args.split,
        input_path=args.input_path,
        embedding_path=args.embedding_path,
        output_path=args.output_path,
    )
    build_and_save_trajectory_dataset(
        dataset_name="KuaiRand",
        split=args.split,
        input_path=paths.input_path,
        embedding_path=paths.embedding_path,
        output_path=paths.output_path,
        summary_path=args.summary_path,
        time_column="time_ms",
        reward_column="is_click",
        window_size=args.window_size,
        seed=args.seed,
        padding_std=args.padding_std,
        max_users=args.max_users,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
