"""生成可供 DORL/ROLeR 训练使用的近邻塑形奖励矩阵。"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.core.envs.Simulated_Env.roler_reward import (
    SUPPORTED_UNCERTAINTY_TYPES,
    compute_knn_reward,
    minmax_normalize,
)


DEFAULT_BATCH_SIZE = 256
DEFAULT_NEIGHBOR_COUNT = 30


def get_args() -> argparse.Namespace:
    """解析 ROLeR 奖励矩阵预处理参数。

    Returns:
        包含输入路径、输出目录和近邻配置的命令行参数。
    """

    parser = argparse.ArgumentParser(
        description="按 ROLeR 规则生成 kNN 奖励、不确定性和塑形奖励矩阵。"
    )
    parser.add_argument(
        "--input-matrix",
        type=Path,
        required=True,
        help="用户-物品预测奖励矩阵，支持 .npy、.npz、pickle。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出 .npy 矩阵和 metadata.json 的目录。",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_NEIGHBOR_COUNT)
    parser.add_argument(
        "--uncertainty-type",
        choices=SUPPORTED_UNCERTAINTY_TYPES,
        default="I-var",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help="同集合搜索时排除用户自身；官方复现默认保留自身。",
    )
    parser.add_argument(
        "--formula",
        choices=("paper", "official-minus", "reward-only"),
        default="paper",
        help="paper: r_knn*(1-U_norm)；official-minus: r_knn-U；reward-only: r_knn。",
    )
    parser.add_argument(
        "--uncertainty-coefficient",
        type=float,
        default=1.0,
        help="official-minus 公式中的不确定性系数。",
    )
    return parser.parse_args()


def _extract_array(value: Any, source: Path) -> np.ndarray:
    """从常见序列化对象中提取二维 NumPy 数组。

    Args:
        value: NumPy 数组、Pandas DataFrame 或包含单个数组的容器。
        source: 源文件路径，用于产生可定位的异常信息。

    Returns:
        二维 ``float32`` 或 ``float64`` 数组；已有浮点精度会被保留。

    Raises:
        ValueError: 无法唯一识别矩阵或矩阵不是二维时抛出。
    """

    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    if isinstance(value, dict):
        candidates = [item for item in value.values() if hasattr(item, "shape")]
        if len(candidates) != 1:
            raise ValueError(
                f"{source} 中包含 {len(candidates)} 个候选数组，无法唯一确定输入矩阵。"
            )
        value = candidates[0]
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{source} 中的输入必须为二维矩阵，当前形状为 {array.shape}。")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{source} 中的输入必须为数值矩阵。")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        array = array.astype(np.float32)
    return array


def load_matrix(path: Path) -> np.ndarray:
    """从磁盘加载用户-物品矩阵。

    Args:
        path: ``.npy``、``.npz`` 或 pickle 文件路径。

    Returns:
        保留原浮点精度的二维矩阵。

    Raises:
        FileNotFoundError: 输入文件不存在时抛出。
        ValueError: 文件格式不受支持或内容无法解析时抛出。
    """

    if not path.is_file():
        raise FileNotFoundError(f"找不到输入矩阵：{path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if len(archive.files) != 1:
            raise ValueError(
                f"{path} 包含 {len(archive.files)} 个数组，请改用只含一个数组的文件。"
            )
        value = archive[archive.files[0]]
    elif suffix in {".pkl", ".pickle"}:
        with path.open("rb") as file_obj:
            value = pickle.load(file_obj)
    else:
        raise ValueError(
            f"不支持的输入格式 {suffix!r}；请使用 .npy、.npz、.pkl 或 .pickle。"
        )
    return _extract_array(value, path)


def shape_reward(
    reward: np.ndarray,
    uncertainty: np.ndarray,
    formula: str,
    uncertainty_coefficient: float,
) -> np.ndarray:
    """按论文公式或官方代码公式合成最终奖励矩阵。

    Args:
        reward: kNN 奖励矩阵。
        uncertainty: kNN 不确定性矩阵。
        formula: ``paper``、``official-minus`` 或 ``reward-only``。
        uncertainty_coefficient: 减法公式的不确定性系数。

    Returns:
        与输入同形状的 ``float32`` 塑形奖励矩阵。

    Raises:
        ValueError: 公式名称不合法或两矩阵形状不一致时抛出。

    Example:
        >>> reward = np.ones((1, 2), dtype=np.float32)
        >>> uncertainty = np.asarray([[0.0, 1.0]], dtype=np.float32)
        >>> shape_reward(reward, uncertainty, "paper", 1.0).tolist()
        [[1.0, 0.0]]
    """

    if reward.shape != uncertainty.shape:
        raise ValueError("reward 与 uncertainty 的形状必须一致。")
    if formula == "paper":
        result = reward * (1.0 - minmax_normalize(uncertainty))
    elif formula == "official-minus":
        result = reward - uncertainty_coefficient * uncertainty
    elif formula == "reward-only":
        result = reward
    else:
        raise ValueError(f"不支持的 formula={formula!r}。")
    return np.asarray(result, dtype=np.float32)


def save_artifacts(
    output_dir: Path,
    reward: np.ndarray,
    uncertainty: np.ndarray,
    shaped_reward: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    """保存 ROLeR 矩阵和可审计元数据。

    Args:
        output_dir: 输出目录。
        reward: kNN 奖励矩阵。
        uncertainty: 不确定性矩阵。
        shaped_reward: 最终塑形奖励矩阵。
        metadata: 输入、参数和统计信息。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "knn_reward.npy", reward, allow_pickle=False)
    np.save(output_dir / "knn_uncertainty.npy", uncertainty, allow_pickle=False)
    np.save(output_dir / "shaped_reward.npy", shaped_reward, allow_pickle=False)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2, sort_keys=True)


def _statistics(matrix: np.ndarray) -> Dict[str, float]:
    """计算用于复现审计的紧凑矩阵统计量。

    Args:
        matrix: 二维数值矩阵。

    Returns:
        包含最小值、最大值、均值和标准差的字典。
    """

    return {
        "min": float(np.min(matrix)),
        "max": float(np.max(matrix)),
        "mean": float(np.mean(matrix)),
        "std": float(np.std(matrix)),
    }


def main() -> None:
    """执行 ROLeR 矩阵生成并输出结果摘要。"""

    args = get_args()
    input_matrix = load_matrix(args.input_matrix)
    result = compute_knn_reward(
        reference_indicator_matrix=input_matrix,
        k=args.k,
        uncertainty_type=args.uncertainty_type,
        batch_size=args.batch_size,
        exclude_self=args.exclude_self,
    )
    shaped_reward = shape_reward(
        reward=result.reward,
        uncertainty=result.uncertainty,
        formula=args.formula,
        uncertainty_coefficient=args.uncertainty_coefficient,
    )
    metadata: Dict[str, Any] = {
        "input_matrix": str(args.input_matrix.resolve()),
        "shape": list(input_matrix.shape),
        "k": args.k,
        "batch_size": args.batch_size,
        "exclude_self": args.exclude_self,
        "uncertainty_type": args.uncertainty_type,
        "formula": args.formula,
        "uncertainty_coefficient": args.uncertainty_coefficient,
        "reward_statistics": _statistics(result.reward),
        "uncertainty_statistics": _statistics(result.uncertainty),
        "shaped_reward_statistics": _statistics(shaped_reward),
    }
    save_artifacts(
        output_dir=args.output_dir,
        reward=result.reward,
        uncertainty=result.uncertainty,
        shaped_reward=shaped_reward,
        metadata=metadata,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

