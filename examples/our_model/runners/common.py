"""DORL-MAC runner 共用工具。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for relative_path in ["./src", "./src/DeepCTR-Torch", "./src/tianshou", "./examples/policy"]:
    resolved = str((PROJECT_ROOT / relative_path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

from src.core.util.data import get_env_args, get_true_env  # noqa: E402
from src.core.util.entropy_penalty import (  # noqa: E402
    accumulate_entropy_counts_for_row,
    finalize_entropy_map,
)

from examples.our_model.data import ActionChunkDataset, TrajectoryLoader  # noqa: E402
import examples.our_model.models.mac_agent as mac_agent_module  # noqa: E402
from examples.our_model.models.dorl_reward import DORLRewardModel  # noqa: E402
from examples.our_model.models.leave_model import RuleBasedLeaveModel  # noqa: E402
from examples.our_model.models.state_tracker_dynamics import StateTrackerDynamics  # noqa: E402
from examples.our_model.policy import ActionMapper  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = "data/KuaiRec/data_processed/DM_KuaiEnv-v0_small_data.pkl"
"""默认 KuaiRec small 离线轨迹路径。"""

DEFAULT_SAVE_ROOT = "saved_models"
"""默认模型保存根目录。"""

DEFAULT_ACTOR_HIDDEN_DIMS = (256, 256)
"""MVP actor 默认隐藏层维度。"""

DEFAULT_VALUE_HIDDEN_DIMS = (256, 256)
"""MVP critic/value 默认隐藏层维度。"""


def configure_logging() -> None:
    """配置命令行日志格式。

    Returns:
        None.
    """

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


def set_seed(seed: int) -> None:
    """设置随机种子。

    Args:
        seed (int): 随机种子。

    Returns:
        None.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    """解析用户传入的设备字符串。

    Args:
        device_arg (str): `cpu`、`cuda:0` 或数字 GPU id。

    Returns:
        torch.device: 解析后的设备。
    """

    requested = str(device_arg).strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{requested}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    LOGGER.warning("CUDA 不可用或设备参数无效，回退到 CPU：%s", device_arg)
    return torch.device("cpu")


def namespace_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """将 argparse Namespace 转为 JSON 友好字典。

    Args:
        args (argparse.Namespace): 命令行参数。

    Returns:
        Dict[str, Any]: 可序列化配置。
    """

    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, torch.device):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def parse_hidden_dims(raw_dims: Iterable[int]) -> Tuple[int, ...]:
    """解析隐藏层维度。

    Args:
        raw_dims (Iterable[int]): 命令行传入的维度序列。

    Returns:
        Tuple[int, ...]: 正整数维度元组。

    Raises:
        ValueError: 当任一维度非正时抛出。
    """

    dims = tuple(int(dim) for dim in raw_dims)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(f"hidden dims must be positive, got {dims}.")
    return dims


def default_item_embedding_path(env: str, user_model_name: str, read_message: str) -> str:
    """构造默认 item embedding 路径。

    Args:
        env (str): 环境名。
        user_model_name (str): user model 名称。
        read_message (str): user model 训练标识。

    Returns:
        str: item embedding 路径。
    """

    return str(
        PROJECT_ROOT
        / "saved_models"
        / env
        / user_model_name
        / "embeddings"
        / f"[{read_message}]_emb_item_val_M0.pt"
    )


def default_predicted_mat_path(env: str, user_model_name: str, read_message: str) -> str:
    """构造默认 predicted reward matrix 路径。

    Args:
        env (str): 环境名。
        user_model_name (str): user model 名称。
        read_message (str): user model 训练标识。

    Returns:
        str: predicted matrix 路径。
    """

    return str(
        PROJECT_ROOT
        / "saved_models"
        / env
        / user_model_name
        / "matsPre"
        / f"[{read_message}]_matPre.pickle"
    )


def default_maxvar_mat_path(env: str, user_model_name: str, read_message: str) -> str:
    """构造默认 user model variance matrix 路径。

    Args:
        env (str): 环境名。
        user_model_name (str): user model 名称。
        read_message (str): user model 训练标识。

    Returns:
        str: max variance matrix 路径。
    """

    return str(
        PROJECT_ROOT
        / "saved_models"
        / env
        / user_model_name
        / "matsVar"
        / f"[{read_message}]_matVar.pickle"
    )


def load_item_embeddings(item_embedding_path: str) -> torch.Tensor:
    """加载 item embedding 表。

    Args:
        item_embedding_path (str): `.pt` 文件路径。

    Returns:
        torch.Tensor: item embedding 表，形状为 `(num_items, action_dim)`。

    Raises:
        FileNotFoundError: 当文件不存在时抛出。
        ValueError: 当文件内容不是二维 Tensor 时抛出。
    """

    path = Path(item_embedding_path)
    if not path.exists():
        raise FileNotFoundError(f"Item embedding file does not exist: {path}")
    item_embeddings = torch.load(path, map_location="cpu")
    if not isinstance(item_embeddings, torch.Tensor) or item_embeddings.ndim != 2:
        raise ValueError(f"Item embedding file must contain a 2D Tensor: {path}")
    return item_embeddings.float()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """向 parser 注册 DORL-MAC 通用参数。

    Args:
        parser (argparse.ArgumentParser): 参数解析器。

    Returns:
        None.
    """

    parser.add_argument("--env", type=str, default="KuaiEnv-v0")
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument("--read_message", type=str, default="pointneg")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--which_tracker", type=str, default="avg")
    parser.add_argument("--reward_handle", type=str, default="cat")
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--random_init", action="store_true", default=False)
    parser.add_argument("--item_embedding_path", type=str, default="")
    parser.add_argument("--predicted_mat_path", type=str, default="")
    parser.add_argument("--maxvar_mat_path", type=str, default="")
    parser.add_argument("--save_root", type=str, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--actor_hidden_dims", type=int, nargs="*", default=list(DEFAULT_ACTOR_HIDDEN_DIMS))
    parser.add_argument("--value_hidden_dims", type=int, nargs="*", default=list(DEFAULT_VALUE_HIDDEN_DIMS))
    parser.add_argument("--swanlab_project", type=str, default="DORL-MAC")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--num_leave_compute", type=int, default=9)
    parser.add_argument("--leave_threshold", type=float, default=1.0)
    parser.add_argument("--max_turn", type=int, default=30)
    parser.add_argument("--force_length", type=int, default=30)
    parser.add_argument("--invalid_action_penalty", type=float, default=-1.0)
    parser.add_argument("--use_exposure_intervention", action="store_true", default=False)
    parser.add_argument("--use_entropy_reward", dest="use_entropy_reward", action="store_true")
    parser.add_argument("--no_entropy_reward", dest="use_entropy_reward", action="store_false")
    parser.set_defaults(use_entropy_reward=True)
    parser.add_argument("--use_uncertainty_penalty", dest="use_uncertainty_penalty", action="store_true")
    parser.add_argument("--no_uncertainty_penalty", dest="use_uncertainty_penalty", action="store_false")
    parser.set_defaults(use_uncertainty_penalty=True)
    parser.add_argument("--lambda_entropy", type=float, default=5.0)
    parser.add_argument("--lambda_variance", type=float, default=0.05)
    parser.add_argument("--entropy_window", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--feature_level", dest="feature_level", action="store_true")
    parser.add_argument("--no_feature_level", dest="feature_level", action="store_false")
    parser.set_defaults(feature_level=True)
    parser.add_argument("--is_sorted", dest="is_sorted", action="store_true")
    parser.add_argument("--no_sorted", dest="is_sorted", action="store_false")
    parser.set_defaults(is_sorted=True)
    parser.add_argument("--dynamics_loss_weight", type=float, default=1.0)


def resolve_common_paths(args: argparse.Namespace) -> None:
    """补齐默认 item embedding 和 predicted matrix 路径。

    Args:
        args (argparse.Namespace): 命令行参数，会被原地更新。

    Returns:
        None.
    """

    if not args.item_embedding_path:
        args.item_embedding_path = default_item_embedding_path(
            args.env,
            args.user_model_name,
            args.read_message,
        )
    if not args.predicted_mat_path:
        args.predicted_mat_path = default_predicted_mat_path(
            args.env,
            args.user_model_name,
            args.read_message,
        )
    if not args.maxvar_mat_path:
        args.maxvar_mat_path = default_maxvar_mat_path(
            args.env,
            args.user_model_name,
            args.read_message,
        )


def build_dataset_and_mapper(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[ActionChunkDataset, ActionMapper, torch.Tensor]:
    """构造 action chunk 数据集和 action mapper。

    Args:
        args (argparse.Namespace): 命令行参数。
        device (torch.device): 计算设备。

    Returns:
        Tuple[ActionChunkDataset, ActionMapper, torch.Tensor]: 数据集、映射器和 item embedding。
    """

    item_embeddings = load_item_embeddings(args.item_embedding_path)
    bundle = TrajectoryLoader(args.dataset_path).load(max_trajectories=args.max_trajectories)
    entropy_history_size = max([args.num_leave_compute, args.max_turn] + list(args.entropy_window or [0]))
    dataset = ActionChunkDataset(
        bundle=bundle,
        item_embeddings=item_embeddings,
        chunk_size=args.chunk_size,
        gamma=args.gamma,
        window_size=args.window_size,
        leave_history_size=entropy_history_size,
        max_chunks=args.max_chunks,
    )
    dataset.validate_action_lookup(max_trajectories=args.max_trajectories)
    action_mapper = ActionMapper(item_embeddings=item_embeddings, device=device)
    return dataset, action_mapper, item_embeddings


def build_env_assets(args: argparse.Namespace) -> Tuple[Any, Any, Dict[str, Any]]:
    """构造 KuaiEnv 与环境参数。

    Args:
        args (argparse.Namespace): 命令行参数。

    Returns:
        Tuple[Any, Any, Dict[str, Any]]: env、dataset 和 env kwargs。
    """

    preserved_names = (
        "num_leave_compute",
        "leave_threshold",
        "max_turn",
        "force_length",
        "entropy_window",
    )
    preserved_values = {
        name: getattr(args, name)
        for name in preserved_names
        if hasattr(args, name)
    }
    env_args = get_env_args(args)
    for name, value in preserved_values.items():
        setattr(env_args, name, value)
    LOGGER.info("开始构造 DORL true env：env=%s", env_args.env)
    env, dataset, kwargs_um = get_true_env(env_args)
    LOGGER.info("DORL true env 构造完成：env=%s", env_args.env)
    return env, dataset, kwargs_um


def build_reward_and_leave(
    args: argparse.Namespace,
    env: Any,
    dataset: Any,
    device: torch.device,
) -> Tuple[DORLRewardModel, RuleBasedLeaveModel]:
    """构造 reward model 和 leave model。

    Args:
        args (argparse.Namespace): 命令行参数。
        env (Any): KuaiEnv 实例。
        dataset (Any): 原项目数据集对象，用于构造 entropy 统计。
        device (torch.device): 计算设备。

    Returns:
        Tuple[DORLRewardModel, RuleBasedLeaveModel]: reward 与 leave 模型。
    """

    raw_user_to_index = {
        int(raw_user_id): int(index)
        for index, raw_user_id in enumerate(env.lbe_user.classes_)
    }
    entropy_map, map_item_feat, entropy_min = build_entropy_reward_assets(args, dataset)
    if hasattr(env, "lbe_item") and env.lbe_item is not None:
        internal_to_raw_item_ids = [int(item_id) for item_id in env.lbe_item.classes_]
    else:
        internal_to_raw_item_ids = list(range(env.mat.shape[1]))
    reward_model = DORLRewardModel(
        predicted_mat_path=args.predicted_mat_path,
        maxvar_mat_path=args.maxvar_mat_path,
        raw_user_to_index=raw_user_to_index,
        internal_to_raw_item_ids=internal_to_raw_item_ids,
        device=device,
        reward_shift=True,
        use_exposure_intervention=args.use_exposure_intervention,
        use_entropy_reward=args.use_entropy_reward,
        entropy_map=entropy_map,
        entropy_window=args.entropy_window,
        lambda_entropy=args.lambda_entropy,
        entropy_min=entropy_min,
        feature_level=args.feature_level,
        map_item_feat=map_item_feat,
        is_sorted=args.is_sorted,
        use_uncertainty_penalty=args.use_uncertainty_penalty,
        lambda_variance=args.lambda_variance,
    )
    leave_model = RuleBasedLeaveModel(
        list_feat_small=env.list_feat_small,
        num_leave_compute=args.num_leave_compute,
        leave_threshold=args.leave_threshold,
        max_turn=args.max_turn,
    )
    return reward_model, leave_model


def build_entropy_reward_assets(
    args: argparse.Namespace,
    dataset: Any,
) -> Tuple[Dict[Tuple[int, ...], float], Dict[int, Any], float]:
    """构造 DORL entropy reward 所需查表和下界。

    Args:
        args (argparse.Namespace): 命令行参数。
        dataset (Any): 原项目数据集对象。

    Returns:
        Tuple[Dict[Tuple[int, ...], float], Dict[int, Any], float]: entropy 查表、
        raw item 到特征列表的映射和 entropy 下界。
    """

    if not args.use_entropy_reward:
        return {}, {}, 0.0
    LOGGER.info("开始构造 DORL entropy reward assets")
    df_train, _, df_item, _ = dataset.get_train_data()
    if "timestamp" in df_train.columns:
        df_uit = df_train[["user_id", "item_id", "timestamp"]].sort_values(["user_id", "timestamp"])
    elif "time_ms" in df_train.columns:
        df_uit = df_train[["user_id", "item_id", "time_ms"]].sort_values(["user_id", "time_ms"])
    else:
        df_uit = df_train[["user_id", "item_id"]].sort_values(["user_id"])

    map_item_feat = dict(zip(df_item.index, df_item["tags"])) if args.feature_level else {}
    map_hist_count = defaultdict(lambda: defaultdict(int))
    last_user = None
    history_items = []
    for row in df_uit.to_numpy():
        user_id = int(row[0])
        item_id = int(row[1])
        if user_id != last_user:
            last_user = user_id
            history_items = []
        accumulate_entropy_counts_for_row(
            map_hist_count=map_hist_count,
            hist_tra=history_items,
            item=item_id,
            entropy_window=args.entropy_window,
            feature_level=args.feature_level,
            map_item_feat=map_item_feat if args.feature_level else None,
            is_sorted=args.is_sorted,
        )
        history_items.append(item_id)

    entropy_map = finalize_entropy_map(map_hist_count)
    entropy_min = 0.0
    for entropy_term in set(args.entropy_window) - {0}:
        values = [value for key, value in entropy_map.items() if len(key) == entropy_term]
        entropy_min += min(values + [1.0])
    LOGGER.info(
        "DORL entropy reward assets 构造完成：entries=%s, entropy_min=%s",
        len(entropy_map),
        entropy_min,
    )
    return entropy_map, map_item_feat, float(entropy_min)


def build_agent(
    args: argparse.Namespace,
    device: torch.device,
    action_mapper: ActionMapper,
    reward_model: DORLRewardModel | None = None,
    leave_model: RuleBasedLeaveModel | None = None,
) -> mac_agent_module.MACAgent:
    """构造 MACAgent。

    Args:
        args (argparse.Namespace): 命令行参数。
        device (torch.device): 计算设备。
        action_mapper (ActionMapper): action mapper。
        reward_model (DORLRewardModel | None): reward 模型，可为空。
        leave_model (RuleBasedLeaveModel | None): leave 模型，可为空。

    Returns:
        mac_agent_module.MACAgent: 初始化后的 agent。
    """

    state_dim = action_mapper.action_dim + 1
    dynamics = StateTrackerDynamics(
        window_size=args.window_size,
        action_dim=action_mapper.action_dim,
        state_dim=state_dim,
    )
    return mac_agent_module.MACAgent(
        state_dim=state_dim,
        action_dim=action_mapper.action_dim,
        chunk_size=args.chunk_size,
        actor_hidden_dims=parse_hidden_dims(args.actor_hidden_dims),
        value_hidden_dims=parse_hidden_dims(args.value_hidden_dims),
        gamma=args.gamma,
        device=device,
        action_mapper=action_mapper,
        dynamics=dynamics,
        reward_model=reward_model,
        leave_model=leave_model,
        target_tau=getattr(args, "target_tau", 0.005),
        invalid_action_penalty=getattr(args, "invalid_action_penalty", -1.0),
        dynamics_loss_weight=getattr(args, "dynamics_loss_weight", 1.0),
    )


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在。

    Args:
        path (str | Path): 目录路径。

    Returns:
        Path: 目录对象。
    """

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_resolved_config(save_dir: str | Path, config: Dict[str, Any]) -> Path:
    """保存 resolved config JSON。

    Args:
        save_dir (str | Path): 保存目录。
        config (Dict[str, Any]): 配置字典。

    Returns:
        Path: 配置文件路径。
    """

    save_path = ensure_dir(save_dir) / "resolved_config.json"
    with save_path.open("w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2, ensure_ascii=False)
    return save_path


def default_run_name(prefix: str, args: argparse.Namespace) -> str:
    """生成默认 run 名称。

    Args:
        prefix (str): run 名称前缀。
        args (argparse.Namespace): 命令行参数。

    Returns:
        str: run 名称。
    """

    if args.run_name:
        return args.run_name
    return f"{prefix}-{args.env}-K{args.chunk_size}-seed{args.seed}"


def set_mpl_cache_to_tmp() -> None:
    """避免 matplotlib 尝试写入不可写的用户配置目录。

    Returns:
        None.
    """

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-easyrl4rec")
