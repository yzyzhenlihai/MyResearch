"""在 EasyRL4Rec 的 DORL 管线上复现 ROLeR。"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import logzero
import numpy as np
import torch

sys.path.extend(
    [".", "./examples/policy", "./src", "./src/DeepCTR-Torch", "./src/tianshou"]
)

from policy_utils import (  # noqa: E402
    get_args_all,
    learn_policy,
    prepare_dir_log,
    prepare_test_envs,
    setup_state_tracker,
)
from run_DORL import (  # noqa: E402
    get_args_DORL,
    prepare_train_envs,
    setup_policy_model,
)
from prepare_ROLeR import (  # noqa: E402
    load_matrix,
    save_artifacts,
    shape_reward,
)
from src.core.envs.Simulated_Env.roler_reward import (  # noqa: E402
    SUPPORTED_UNCERTAINTY_TYPES,
    compute_knn_reward,
)
from src.core.envs.Simulated_Env.roler_kuairec import (  # noqa: E402
    compute_official_kuairec_reward,
)
from src.core.userModel.precomputed_artifacts import (  # noqa: E402
    PrecomputedModelArtifacts,
)
from src.core.util.data import get_env_args, get_true_env  # noqa: E402


DEFAULT_DATASET_CONFIG: Dict[str, Tuple[int, str, str, float, float]] = {
    "KuaiEnv-v0": (30, "I-var", "att", 0.05, 5.0),
    "KuaiRand-v0": (150, "II-weight-norm", "avg", 0.01, 0.05),
    "CoatEnv-v0": (15, "II-weight-norm", "att", 0.005, 1.0),
    "YahooEnv-v0": (20, "II-weight-norm", "att", 0.005, 1.0),
}


def get_args_roler() -> argparse.Namespace:
    """解析 ROLeR 独有参数，并保留 DORL 的其余参数。

    Returns:
        ROLeR 邻居、奖励组合、缓存和 embedding 配置。
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_name", type=str, default="ROLeR")
    parser.add_argument("--message", type=str, default="ROLeR")
    parser.add_argument("--kr", type=int, default=None)
    parser.add_argument("--ku", type=int, default=None)
    parser.add_argument(
        "--uncertain_type",
        choices=SUPPORTED_UNCERTAINTY_TYPES,
        default=None,
    )
    parser.add_argument(
        "--reward_formula",
        choices=("paper", "official-minus", "reward-only"),
        default="official-minus",
        help=(
            "official-minus 对齐公开代码；paper 使用论文式 r_knn*(1-U_norm)；"
            "reward-only 用于消融。"
        ),
    )
    parser.add_argument("--lambda_variance", type=float, default=None)
    parser.add_argument("--knn_batch_size", type=int, default=256)
    parser.add_argument("--aggregation_batch_size", type=int, default=None)
    parser.add_argument("--kuairec_csv_chunk_size", type=int, default=500_000)
    parser.add_argument("--num_att_layers", type=int, default=2)
    parser.add_argument("--att_dropout", type=float, default=0.1)
    parser.add_argument("--roler_reward_path", type=str, default=None)
    parser.add_argument("--roler_cache_dir", type=str, default=None)
    parser.add_argument("--rebuild_roler_cache", action="store_true", default=False)
    parser.add_argument("--prepare_only", action="store_true", default=False)
    parser.add_argument("--exclude_self_neighbor", action="store_true", default=False)
    parser.add_argument("--is_scratch", dest="scratch", action="store_true")
    parser.add_argument("--no_scratch", dest="scratch", action="store_false")
    parser.set_defaults(scratch=True)
    return parser.parse_known_args()[0]


def _resolve_roler_config(args: argparse.Namespace) -> None:
    """补齐数据集默认参数并验证官方复现约束。

    Args:
        args: 合并后的全部命令行参数；本函数会原地写入 ``kr``、``ku``、
            ``uncertain_type`` 和必要时的状态追踪器。

    Raises:
        ValueError: 数据集未知、邻居数不一致或邻居数非法时抛出。
    """

    if args.env not in DEFAULT_DATASET_CONFIG:
        raise ValueError(
            f"ROLeR 当前支持 {tuple(DEFAULT_DATASET_CONFIG)}，收到 {args.env!r}。"
        )
    (
        default_k,
        default_uncertainty,
        default_tracker,
        default_lambda_variance,
        default_lambda_entropy,
    ) = DEFAULT_DATASET_CONFIG[args.env]
    args.kr = default_k if args.kr is None else args.kr
    args.ku = args.kr if args.ku is None else args.ku
    if args.kr != args.ku:
        raise ValueError(
            "当前内存优化实现要求 kr == ku；官方四组主实验也均使用相同值。"
        )
    if args.kr <= 0:
        raise ValueError("kr/ku 必须为正整数。")
    args.uncertain_type = (
        default_uncertainty
        if args.uncertain_type is None
        else args.uncertain_type
    )
    args.lambda_variance = (
        default_lambda_variance
        if args.lambda_variance is None
        else args.lambda_variance
    )
    if "--lambda_entropy" not in sys.argv:
        args.lambda_entropy = default_lambda_entropy
    if "--window_size" not in sys.argv:
        args.window_size = 5 if args.env == "YahooEnv-v0" else 3
    if "--num_leave_compute" not in sys.argv:
        args.num_leave_compute = 1
    if "--leave_threshold" not in sys.argv:
        args.leave_threshold = 0
    feature_level_was_explicit = (
        "--is_feature_level" in sys.argv or "--no_feature_level" in sys.argv
    )
    if args.env in {"CoatEnv-v0", "YahooEnv-v0"} and not feature_level_was_explicit:
        # 两个显式评分数据集的物品表没有 DORL 所需的 tags 列，默认退化为物品级熵。
        args.feature_level = False
    tracker_was_explicit = "--which_tracker" in sys.argv
    if not tracker_was_explicit:
        args.which_tracker = default_tracker


def _matrix_statistics(matrix: np.ndarray) -> Dict[str, float]:
    """生成缓存审计所需的矩阵统计量。

    Args:
        matrix: 二维奖励或不确定性矩阵。

    Returns:
        最小值、最大值、均值和标准差。
    """

    return {
        "min": float(np.min(matrix)),
        "max": float(np.max(matrix)),
        "mean": float(np.mean(matrix)),
        "std": float(np.std(matrix)),
    }


def _default_cache_dir(args: argparse.Namespace) -> Path:
    """根据配置构造互不冲突的奖励缓存目录。

    Args:
        args: 已补齐的 ROLeR 参数。

    Returns:
        位于 ``saved_models/<env>/ROLeR/reward_cache`` 下的配置目录。
    """

    uncertainty_slug = args.uncertain_type.replace("-", "_")
    formula_slug = args.reward_formula.replace("-", "_")
    self_slug = "exclude_self" if args.exclude_self_neighbor else "include_self"
    return Path("saved_models") / args.env / "ROLeR" / "reward_cache" / (
        f"k{args.kr}_{uncertainty_slug}_{formula_slug}_lv{args.lambda_variance:g}_"
        f"{self_slug}"
    )


def prepare_roler_reward(
    args: argparse.Namespace,
    artifacts: PrecomputedModelArtifacts,
) -> Path:
    """生成或复用 ROLeR 塑形奖励矩阵。

    Args:
        args: ROLeR 配置。
        artifacts: 原始 DeepFM 预测矩阵和 embedding 的轻量适配器。

    Returns:
        可由 DORL 模拟环境以 mmap 方式加载的 ``shaped_reward.npy`` 路径。

    Raises:
        FileNotFoundError: 用户显式指定的奖励矩阵不存在时抛出。
    """

    if args.roler_reward_path is not None:
        reward_path = Path(args.roler_reward_path).resolve()
        if not reward_path.is_file():
            raise FileNotFoundError(f"找不到 --roler_reward_path：{reward_path}")
        return reward_path

    cache_dir = (
        Path(args.roler_cache_dir).resolve()
        if args.roler_cache_dir is not None
        else _default_cache_dir(args).resolve()
    )
    reward_path = cache_dir / "shaped_reward.npy"
    metadata_path = cache_dir / "metadata.json"
    if reward_path.is_file() and metadata_path.is_file() and not args.rebuild_roler_cache:
        logzero.logger.info("复用 ROLeR 奖励缓存：%s", reward_path)
        return reward_path

    if args.env == "KuaiEnv-v0":
        source_path = Path("data/KuaiRec/data_raw/big_matrix_processed.csv")
        logzero.logger.info("按官方跨块定义读取 KuaiRec 大矩阵：%s", source_path)
        result = compute_official_kuairec_reward(
            data_root=source_path.parent,
            k=args.kr,
            uncertainty_type=args.uncertain_type,
            search_batch_size=args.knn_batch_size,
            aggregation_batch_size=args.aggregation_batch_size,
            csv_chunk_size=args.kuairec_csv_chunk_size,
        )
    else:
        source_path = Path(artifacts.PREDICTION_MAT_PATH)
        logzero.logger.info("读取预测矩阵并计算 ROLeR 近邻：%s", source_path)
        source_matrix = load_matrix(source_path)
        result = compute_knn_reward(
            reference_indicator_matrix=source_matrix,
            k=args.kr,
            uncertainty_type=args.uncertain_type,
            batch_size=args.knn_batch_size,
            aggregation_batch_size=args.aggregation_batch_size,
            exclude_self=args.exclude_self_neighbor,
        )
    shaped_reward = shape_reward(
        reward=result.reward,
        uncertainty=result.uncertainty,
        formula=args.reward_formula,
        uncertainty_coefficient=args.lambda_variance,
    )
    metadata: Dict[str, Any] = {
        "env": args.env,
        "source_matrix": str(source_path.resolve()),
        "shape": list(result.reward.shape),
        "kr": args.kr,
        "ku": args.ku,
        "uncertain_type": args.uncertain_type,
        "reward_formula": args.reward_formula,
        "lambda_variance": args.lambda_variance,
        "knn_batch_size": args.knn_batch_size,
        "aggregation_batch_size": args.aggregation_batch_size,
        "kuairec_csv_chunk_size": args.kuairec_csv_chunk_size,
        "exclude_self_neighbor": args.exclude_self_neighbor,
        "reward_statistics": _matrix_statistics(result.reward),
        "uncertainty_statistics": _matrix_statistics(result.uncertainty),
        "shaped_reward_statistics": _matrix_statistics(shaped_reward),
    }
    save_artifacts(
        output_dir=cache_dir,
        reward=result.reward,
        uncertainty=result.uncertainty,
        shaped_reward=shaped_reward,
        metadata=metadata,
    )
    logzero.logger.info("ROLeR 奖励缓存已生成：%s", reward_path)
    return reward_path


def _artifact_fingerprint(path: Path) -> Dict[str, Any]:
    """生成足以检测本地产物替换的轻量文件指纹。

    Args:
        path: 预计算矩阵文件。

    Returns:
        包含绝对路径、字节数和纳秒修改时间的字典。
    """

    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _pickle_matrix_bounds(path: Path) -> Tuple[float, float]:
    """读取 pickle 数值矩阵并返回有限最小/最大值。

    Args:
        path: DeepFM 预测或方差矩阵路径。

    Returns:
        ``(minimum, maximum)``。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
        ValueError: 内容不是二维有限矩阵时抛出。
    """

    if not path.is_file():
        raise FileNotFoundError(f"找不到奖励边界所需矩阵：{path}")
    with path.open("rb") as file_obj:
        value = pickle.load(file_obj)
    matrix = np.asarray(value)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{path} 必须是二维有限数值矩阵。")
    return float(matrix.min()), float(matrix.max())


def prepare_official_reward_bounds(
    args: argparse.Namespace,
    artifacts: PrecomputedModelArtifacts,
) -> None:
    """复现官方环境用原始模型矩阵计算奖励平移边界的行为。

    官方代码在切换为 kNN 奖励与 kNN 不确定性后，仍以原始 DeepFM 预测矩阵
    和 ensemble 方差矩阵确定 ``MIN_R/MAX_R``。这里缓存两个大矩阵的极值，
    避免 KuaiRand 每次启动都重复读取数 GB 文件。

    Args:
        args: ROLeR 参数；本函数会原地写入两个 base override。
        artifacts: 提供预测矩阵与方差矩阵路径的适配器。
    """

    if args.reward_formula != "official-minus":
        args.reward_min_base_override = None
        args.reward_max_base_override = None
        return

    prediction_path = Path(artifacts.PREDICTION_MAT_PATH)
    variance_path = Path(artifacts.VAR_MAT_PATH)
    fingerprints = {
        "prediction": _artifact_fingerprint(prediction_path),
        "variance": _artifact_fingerprint(variance_path),
    }
    cache_path = (
        Path("saved_models") / args.env / "ROLeR" / "source_bounds.json"
    )
    cached: Dict[str, Any] = {}
    if cache_path.is_file():
        with cache_path.open(encoding="utf-8") as file_obj:
            cached = json.load(file_obj)
    if cached.get("fingerprints") == fingerprints:
        prediction_min = float(cached["prediction_min"])
        prediction_max = float(cached["prediction_max"])
        variance_min = float(cached["variance_min"])
        variance_max = float(cached["variance_max"])
    else:
        prediction_min, prediction_max = _pickle_matrix_bounds(prediction_path)
        variance_min, variance_max = _pickle_matrix_bounds(variance_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprints": fingerprints,
            "prediction_min": prediction_min,
            "prediction_max": prediction_max,
            "variance_min": variance_min,
            "variance_max": variance_max,
        }
        with cache_path.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)

    args.reward_min_base_override = (
        prediction_min - args.lambda_variance * variance_max
    )
    args.reward_max_base_override = (
        prediction_max - args.lambda_variance * variance_min
    )
    logzero.logger.info(
        "官方奖励边界 base=[%.6f, %.6f]",
        args.reward_min_base_override,
        args.reward_max_base_override,
    )


def main(args: argparse.Namespace) -> None:
    """准备 ROLeR 奖励、复用 DORL A2C 并开始训练。

    Args:
        args: 通用策略、环境、DORL 与 ROLeR 参数的合并结果。
    """

    _resolve_roler_config(args)
    model_save_path, logger_path = prepare_dir_log(args)
    args.device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_root = Path("saved_models") / args.env / args.user_model_name
    artifacts = PrecomputedModelArtifacts(
        model_root=model_root,
        message=args.read_message,
        scratch_action_embedding=args.scratch,
        seed=args.seed,
    )
    args.predicted_mat_path = str(prepare_roler_reward(args, artifacts))
    prepare_official_reward_bounds(args, artifacts)
    if args.prepare_only:
        logzero.logger.info("--prepare_only 已完成，跳过策略训练。")
        return

    env, dataset, kwargs_um = get_true_env(args)
    train_envs = prepare_train_envs(args, artifacts, env, dataset, kwargs_um)
    test_envs_dict = prepare_test_envs(args, env, kwargs_um)
    state_tracker = setup_state_tracker(
        args, artifacts, env, train_envs, test_envs_dict
    )
    if state_tracker is None:
        raise ValueError(f"不支持的 which_tracker={args.which_tracker!r}。")
    policy, train_collector, test_collector_set, optim = setup_policy_model(
        args, state_tracker, train_envs, test_envs_dict
    )
    learn_policy(
        args,
        env,
        dataset,
        policy,
        train_collector,
        test_collector_set,
        state_tracker,
        optim,
        model_save_path,
        logger_path,
        trainer="onpolicy",
    )


def build_args() -> argparse.Namespace:
    """合并通用、环境、DORL 和 ROLeR 参数。

    Returns:
        可直接传给 :func:`main` 的完整命名空间。
    """

    args_all = get_args_all("onpolicy")
    args_env = get_env_args(args_all)
    args_dorl = get_args_DORL()
    args_roler = get_args_roler()
    args_all.__dict__.update(args_env.__dict__)
    args_all.__dict__.update(args_dorl.__dict__)
    args_all.__dict__.update(args_roler.__dict__)
    return args_all


if __name__ == "__main__":
    try:
        main(build_args())
    except Exception:
        error = traceback.format_exc()
        print(error)
        logzero.logger.error(error)
        raise

