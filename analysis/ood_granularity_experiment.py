"""OOD 识别粒度主实验的独立实现模块。"""

import logging
import os
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
from tqdm import tqdm

from analysis.common import (
    CoreAssets,
    build_manifest,
    dataframe_to_markdown,
    load_core_assets,
    matrix_slice,
    normalize_rows,
    ranks_from_order,
    safe_average_precision,
    safe_roc_auc,
    save_json,
    stable_argsort_desc,
)
from src.core.envs.KuaiRec.KuaiData import KuaiData


LOGGER = logging.getLogger(__name__)

EMPTY_TEXT = "No valid data available"
FIGURE_DPI = 220
HIGHVAR_LABELS = ["NH", "DO", "Near-Low", "Far-High"]
MAIN_LABEL_ORDER = ["NH", "DO", "Near-Low", "Far-High", "Other"]
QUADRANT_LABEL_ORDER = ["Near-High", "Near-Low", "Far-High", "Far-Low", "Other"]
PLOTTED_QUADRANT_LABELS = ["Near-High", "Near-Low", "Far-High", "Far-Low"]
VIOLIN_MAIN_LABELS = ["NH", "DO"]


@dataclass
class UserTrajectoryBundle:
    """保存单个用户实验所需的轨迹与原始历史信息。

    该对象只保留本实验真正会用到的字段，用于避免在实现层面反复操作
    原始 `pickle` 字典。`observations/actions` 来自离线 RL 轨迹，
    `history_item_ids/history_reward_normed` 来自按时间排序后的原始 CSV。

    Attributes:
        raw_user_id (int): 原始用户 ID。
        user_local_index (int): 当前实验切片内的用户局部行号。
        observations (np.ndarray): 轨迹状态向量，形状为 `(T, state_dim)`。
        actions (np.ndarray): 轨迹动作嵌入，形状为 `(T, action_dim)`。
        history_item_ids (np.ndarray): 按时间排序的原始物品 ID 序列，形状为 `(T,)`。
        history_reward_normed (np.ndarray): 与原始历史对齐的 `watch_ratio_normed` 序列。
    """

    raw_user_id: int
    user_local_index: int
    observations: np.ndarray
    actions: np.ndarray
    history_item_ids: np.ndarray
    history_reward_normed: np.ndarray


@dataclass
class AnchorSample:
    """表示单个锚点状态的轻量记录。

    Attributes:
        raw_user_id (int): 原始用户 ID。
        user_local_index (int): 当前实验切片内的用户局部行号。
        step_index (int): 锚点在该用户轨迹中的时间步索引。
        trajectory_length (int): 当前用户轨迹总长度。
        observation (np.ndarray): 锚点状态向量，形状为 `(state_dim,)`。
        logged_item_id (int): 该时间步真实执行的原始物品 ID。
        logged_action (np.ndarray): 该时间步真实动作嵌入，形状为 `(action_dim,)`。
    """

    raw_user_id: int
    user_local_index: int
    step_index: int
    trajectory_length: int
    observation: np.ndarray
    logged_item_id: int
    logged_action: np.ndarray


@dataclass
class PreparedExperimentAssets:
    """保存 OOD 粒度实验的静态资源。

    该对象聚合了 user model 资产、候选物品空间、离线轨迹对齐结果与
    预切片后的矩阵，供后续实验主循环直接消费。

    Attributes:
        core_assets (CoreAssets): 通用分析资产。
        user_bundles (Dict[int, UserTrajectoryBundle]): 选中用户的轨迹与历史信息。
        candidate_raw_item_ids (np.ndarray): 当前 small-space 候选物品原始 ID。
        candidate_small_indices (np.ndarray): 与候选物品对齐的 small-space 编码。
        candidate_action_embeddings (np.ndarray): 候选动作嵌入，形状为 `(n_item, action_dim)`。
        candidate_action_norm (np.ndarray): 行归一化后的动作嵌入。
        candidate_feature_lists (List[Tuple[int, ...]]): 与候选物品顺序对齐的 feature 集。
        item_feature_lookup (Dict[int, Tuple[int, ...]]): 原始物品 ID 到 feature 集的映射。
        item_position_lookup (Dict[int, int]): 原始物品 ID 到候选列位置的映射。
        pred_slice (np.ndarray): 选中用户与候选物品对应的 `r_hat` 矩阵。
        var_slice (np.ndarray): 选中用户与候选物品对应的 `U_current` 矩阵。
        raw_reward_slice (np.ndarray): 选中用户与候选物品对应的真实 `watch_ratio` 矩阵。
        norm_reward_slice (np.ndarray): 选中用户与候选物品对应的 `watch_ratio_normed` 矩阵。
        action_alignment_summary (Dict[str, Any]): 动作嵌入对齐的 sanity check 结果。
    """

    core_assets: CoreAssets
    user_bundles: Dict[int, UserTrajectoryBundle]
    candidate_raw_item_ids: np.ndarray
    candidate_small_indices: np.ndarray
    candidate_action_embeddings: np.ndarray
    candidate_action_norm: np.ndarray
    candidate_feature_lists: List[Tuple[int, ...]]
    item_feature_lookup: Dict[int, Tuple[int, ...]]
    item_position_lookup: Dict[int, int]
    pred_slice: np.ndarray
    var_slice: np.ndarray
    raw_reward_slice: np.ndarray
    norm_reward_slice: np.ndarray
    action_alignment_summary: Dict[str, Any]


def configure_logging() -> None:
    """配置实验日志格式。

    Returns:
        None: 该函数原地修改根日志器配置。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )


def load_trajectory_list(trajectory_path: str) -> List[Dict[str, Any]]:
    """从磁盘读取离线 RL 轨迹列表。

    Args:
        trajectory_path (str): 轨迹 `pickle` 文件路径。

    Returns:
        List[Dict[str, Any]]: 顶层为 `list[dict]` 的轨迹对象。

    Raises:
        FileNotFoundError: 当 `trajectory_path` 不存在时抛出。
        ValueError: 当顶层对象不是 `list` 时抛出。
    """

    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(f"Trajectory pickle does not exist: {trajectory_path}")
    with open(trajectory_path, "rb") as file_obj:
        trajectories = pickle.load(file_obj)
    if not isinstance(trajectories, list):
        raise ValueError(f"Trajectory pickle must be list[dict], got {type(trajectories)}")
    return trajectories


def load_selected_user_histories(
    small_csv_path: str,
    target_user_ids: np.ndarray,
) -> Dict[int, pd.DataFrame]:
    """读取并按时间排序选中用户的原始历史序列。

    Args:
        small_csv_path (str): `small_matrix_processed.csv` 的路径。
        target_user_ids (np.ndarray): 需要保留的原始用户 ID。

    Returns:
        Dict[int, pd.DataFrame]: 以 `user_id` 为键的历史 DataFrame 映射。

    Raises:
        FileNotFoundError: 当 CSV 文件不存在时抛出。
        ValueError: 当目标用户集合为空时抛出。
    """

    if len(target_user_ids) == 0:
        raise ValueError("target_user_ids must be non-empty.")
    if not os.path.exists(small_csv_path):
        raise FileNotFoundError(f"small_matrix_processed.csv does not exist: {small_csv_path}")

    df_small = pd.read_csv(
        small_csv_path,
        usecols=["user_id", "item_id", "timestamp", "watch_ratio_normed"],
    )
    df_small = df_small[df_small["user_id"].isin(set(target_user_ids.tolist()))].copy()
    df_small = df_small.sort_values(["user_id", "timestamp"], ascending=[True, True])

    history_map: Dict[int, pd.DataFrame] = {}
    for raw_user_id, group in df_small.groupby("user_id", observed=False):
        history_map[int(raw_user_id)] = group.reset_index(drop=True)
    return history_map


def build_user_bundles(
    trajectories: List[Dict[str, Any]],
    history_map: Dict[int, pd.DataFrame],
    selected_user_ids: np.ndarray,
) -> Dict[int, UserTrajectoryBundle]:
    """构建选中用户的轨迹与历史对齐结果。

    Args:
        trajectories (List[Dict[str, Any]]): 从 `pickle` 读取出的完整轨迹列表。
        history_map (Dict[int, pd.DataFrame]): 按用户分组后的原始历史信息。
        selected_user_ids (np.ndarray): 当前实验选中的原始用户 ID。

    Returns:
        Dict[int, UserTrajectoryBundle]: 对齐完成的用户轨迹字典。

    Raises:
        KeyError: 当某个目标用户在轨迹或 CSV 中缺失时抛出。
        ValueError: 当轨迹长度与原始历史长度不一致时抛出。
    """

    local_index_lookup = {int(raw_user_id): idx for idx, raw_user_id in enumerate(selected_user_ids.tolist())}
    trajectory_lookup: Dict[int, Dict[str, Any]] = {}
    for trajectory in trajectories:
        raw_user_id = int(trajectory["user_id"])
        if raw_user_id in local_index_lookup:
            trajectory_lookup[raw_user_id] = trajectory

    bundles: Dict[int, UserTrajectoryBundle] = {}
    for raw_user_id in selected_user_ids.tolist():
        if raw_user_id not in trajectory_lookup:
            raise KeyError(f"User {raw_user_id} is missing from trajectory pickle.")
        if raw_user_id not in history_map:
            raise KeyError(f"User {raw_user_id} is missing from small_matrix_processed.csv.")

        trajectory = trajectory_lookup[raw_user_id]
        history_df = history_map[raw_user_id]

        trajectory_length = int(len(trajectory["observations"]))
        history_length = int(len(history_df))
        if trajectory_length != history_length:
            raise ValueError(
                "Trajectory length mismatch for user "
                f"{raw_user_id}: pkl={trajectory_length}, csv={history_length}"
            )

        bundles[raw_user_id] = UserTrajectoryBundle(
            raw_user_id=raw_user_id,
            user_local_index=local_index_lookup[raw_user_id],
            observations=np.asarray(trajectory["observations"], dtype=np.float32),
            actions=np.asarray(trajectory["actions"], dtype=np.float32),
            history_item_ids=history_df["item_id"].to_numpy(dtype=np.int64),
            history_reward_normed=history_df["watch_ratio_normed"].to_numpy(dtype=np.float32),
        )

    return bundles


def load_candidate_action_embeddings(
    core_assets: CoreAssets,
) -> Tuple[np.ndarray, np.ndarray]:
    """加载与离线轨迹 `actions` 对齐的训练集动作嵌入。

    Args:
        core_assets (CoreAssets): 通用资产对象。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            第一个返回值是候选动作嵌入矩阵，
            第二个返回值是行归一化后的动作嵌入矩阵。

    Raises:
        ValueError: 当候选物品无法映射到训练嵌入行时抛出。
    """

    saved_embedding = core_assets.ensemble.load_user_item_embedding(model_i=0, freeze_emb=True)
    item_embedding = saved_embedding["feat_item"].weight.detach().cpu().numpy().astype(np.float32)
    candidate_raw_item_ids = np.asarray(core_assets.item_raw_ids, dtype=np.int64)

    if np.max(candidate_raw_item_ids) >= item_embedding.shape[0]:
        raise ValueError(
            "Candidate raw item id exceeds training embedding rows: "
            f"max_item_id={int(np.max(candidate_raw_item_ids))}, emb_rows={item_embedding.shape[0]}"
        )

    candidate_action_embeddings = item_embedding[candidate_raw_item_ids]
    candidate_action_norm = normalize_rows(candidate_action_embeddings)
    return candidate_action_embeddings.astype(np.float32), candidate_action_norm.astype(np.float32)


def load_item_feature_lookup(
    candidate_raw_item_ids: np.ndarray,
) -> Tuple[List[Tuple[int, ...]], Dict[int, Tuple[int, ...]]]:
    """读取候选物品的 feature 集合。

    Args:
        candidate_raw_item_ids (np.ndarray): 当前实验候选物品的原始 ID。

    Returns:
        Tuple[List[Tuple[int, ...]], Dict[int, Tuple[int, ...]]]:
            第一个返回值与候选物品顺序对齐；
            第二个返回值提供按原始物品 ID 查询的映射。

    Raises:
        ValueError: 当候选物品 ID 越界时抛出。
    """

    list_feat, _ = KuaiData.load_category()
    candidate_feature_lists: List[Tuple[int, ...]] = []
    feature_lookup: Dict[int, Tuple[int, ...]] = {}
    for raw_item_id in candidate_raw_item_ids.tolist():
        if raw_item_id < 0 or raw_item_id >= len(list_feat):
            raise ValueError(f"Item id {raw_item_id} is out of feature lookup range.")
        feature_tuple = tuple(int(feature_id) for feature_id in list_feat[raw_item_id])
        candidate_feature_lists.append(feature_tuple)
        feature_lookup[int(raw_item_id)] = feature_tuple
    return candidate_feature_lists, feature_lookup


def compute_action_alignment_summary(
    user_bundles: Dict[int, UserTrajectoryBundle],
    item_position_lookup: Dict[int, int],
    candidate_action_embeddings: np.ndarray,
    atol: float = 1.0e-5,
    max_samples: int = 4096,
) -> Dict[str, Any]:
    """检查离线轨迹动作向量与候选动作嵌入是否对齐。

    Args:
        user_bundles (Dict[int, UserTrajectoryBundle]): 选中用户轨迹映射。
        item_position_lookup (Dict[int, int]): 原始物品 ID 到候选列位置的映射。
        candidate_action_embeddings (np.ndarray): 候选动作嵌入矩阵。
        atol (float): `np.allclose` 使用的容差。
        max_samples (int): 最多参与对齐检查的 transition 数。

    Returns:
        Dict[str, Any]: 对齐检查摘要。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    observed_actions: List[np.ndarray] = []
    lookup_actions: List[np.ndarray] = []

    for bundle in user_bundles.values():
        if len(observed_actions) >= max_samples:
            break
        sample_count = min(len(bundle.actions), max_samples - len(observed_actions))
        logged_items = bundle.history_item_ids[:sample_count]
        for index in range(sample_count):
            raw_item_id = int(logged_items[index])
            candidate_position = item_position_lookup.get(raw_item_id, None)
            if candidate_position is None:
                continue
            observed_actions.append(bundle.actions[index])
            lookup_actions.append(candidate_action_embeddings[candidate_position])

    if len(observed_actions) == 0:
        return {
            "checked_samples": 0,
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "allclose": None,
            "atol": float(atol),
        }

    observed_matrix = np.stack(observed_actions, axis=0).astype(np.float32)
    lookup_matrix = np.stack(lookup_actions, axis=0).astype(np.float32)
    diff = np.abs(observed_matrix - lookup_matrix)
    return {
        "checked_samples": int(len(observed_matrix)),
        "mean_abs_diff": float(np.mean(diff)),
        "max_abs_diff": float(np.max(diff)),
        "allclose": bool(np.allclose(observed_matrix, lookup_matrix, atol=atol, rtol=atol)),
        "atol": float(atol),
    }


def prepare_experiment_assets(config: Mapping[str, Any]) -> PreparedExperimentAssets:
    """加载 OOD 粒度实验所需的全部静态资产。

    Args:
        config (Mapping[str, Any]): 实验配置字典。

    Returns:
        PreparedExperimentAssets: 供主实验循环直接消费的资产对象。

    Raises:
        FileNotFoundError: 当输入轨迹或原始 CSV 缺失时抛出。
        KeyError: 当轨迹与原始历史无法按用户对齐时抛出。
        ValueError: 当长度或嵌入对齐不一致时抛出。
    """

    core_assets = load_core_assets(dict(config))
    trajectories = load_trajectory_list(str(config["trajectory_pkl_path"]))
    selected_user_ids = np.asarray(core_assets.user_raw_ids, dtype=np.int64)
    history_map = load_selected_user_histories(str(config["small_csv_path"]), selected_user_ids)
    user_bundles = build_user_bundles(trajectories, history_map, selected_user_ids)
    del trajectories

    candidate_raw_item_ids = np.asarray(core_assets.item_raw_ids, dtype=np.int64)
    candidate_small_indices = core_assets.env.lbe_item.transform(candidate_raw_item_ids)
    candidate_action_embeddings, candidate_action_norm = load_candidate_action_embeddings(core_assets)
    candidate_feature_lists, item_feature_lookup = load_item_feature_lookup(candidate_raw_item_ids)
    item_position_lookup = {
        int(raw_item_id): int(position)
        for position, raw_item_id in enumerate(candidate_raw_item_ids.tolist())
    }

    pred_slice = matrix_slice(core_assets.pred_mat, core_assets)
    var_slice = matrix_slice(core_assets.var_mat, core_assets)
    raw_reward_slice = matrix_slice(core_assets.oracle_raw_mat, core_assets)
    norm_reward_slice = matrix_slice(core_assets.oracle_norm_mat, core_assets)

    action_alignment_summary = compute_action_alignment_summary(
        user_bundles=user_bundles,
        item_position_lookup=item_position_lookup,
        candidate_action_embeddings=candidate_action_embeddings,
        atol=float(config.get("action_alignment_atol", 1.0e-5)),
        max_samples=int(config.get("action_alignment_max_samples", 4096)),
    )

    return PreparedExperimentAssets(
        core_assets=core_assets,
        user_bundles=user_bundles,
        candidate_raw_item_ids=candidate_raw_item_ids,
        candidate_small_indices=np.asarray(candidate_small_indices, dtype=np.int32),
        candidate_action_embeddings=candidate_action_embeddings,
        candidate_action_norm=candidate_action_norm,
        candidate_feature_lists=candidate_feature_lists,
        item_feature_lookup=item_feature_lookup,
        item_position_lookup=item_position_lookup,
        pred_slice=pred_slice,
        var_slice=var_slice,
        raw_reward_slice=raw_reward_slice,
        norm_reward_slice=norm_reward_slice,
        action_alignment_summary=action_alignment_summary,
    )


def sample_anchor_indices(trajectory_length: int, anchor_states_per_user: int) -> np.ndarray:
    """对单个用户轨迹做确定性均匀子采样。

    Args:
        trajectory_length (int): 当前用户轨迹长度。
        anchor_states_per_user (int): 该用户最多抽取的锚点数。

    Returns:
        np.ndarray: 升序排列的锚点索引数组。

    Raises:
        ValueError: 当 `anchor_states_per_user` 小于等于 0 时抛出。
    """

    if anchor_states_per_user <= 0:
        raise ValueError("anchor_states_per_user must be positive.")
    if trajectory_length <= anchor_states_per_user:
        return np.arange(trajectory_length, dtype=np.int32)
    sampled = np.linspace(0, trajectory_length - 1, num=anchor_states_per_user, dtype=np.int32)
    return np.unique(sampled.astype(np.int32))


def build_anchor_samples(
    user_bundles: Dict[int, UserTrajectoryBundle],
    anchor_states_per_user: int,
) -> Dict[int, List[AnchorSample]]:
    """按用户构建锚点状态列表。

    Args:
        user_bundles (Dict[int, UserTrajectoryBundle]): 已对齐的用户轨迹字典。
        anchor_states_per_user (int): 每个用户最多保留的锚点数。

    Returns:
        Dict[int, List[AnchorSample]]: 以原始用户 ID 为键的锚点列表映射。

    Raises:
        ValueError: 当轨迹长度非法时抛出。
    """

    anchor_map: Dict[int, List[AnchorSample]] = {}
    for raw_user_id, bundle in user_bundles.items():
        trajectory_length = int(len(bundle.observations))
        if trajectory_length <= 0:
            raise ValueError(f"User {raw_user_id} has empty trajectory.")
        anchor_indices = sample_anchor_indices(trajectory_length, anchor_states_per_user)
        anchor_map[raw_user_id] = [
            AnchorSample(
                raw_user_id=raw_user_id,
                user_local_index=bundle.user_local_index,
                step_index=int(step_index),
                trajectory_length=trajectory_length,
                observation=bundle.observations[int(step_index)],
                logged_item_id=int(bundle.history_item_ids[int(step_index)]),
                logged_action=bundle.actions[int(step_index)],
            )
            for step_index in anchor_indices.tolist()
        ]
    return anchor_map


def build_support_joint_vectors(bundle: UserTrajectoryBundle) -> np.ndarray:
    """为单个用户构造归一化后的 support state-action 向量。

    Args:
        bundle (UserTrajectoryBundle): 当前用户的轨迹对象。

    Returns:
        np.ndarray: 形状为 `(T, state_dim + action_dim)` 的 support 向量矩阵。

    Raises:
        ValueError: 当状态和动作长度不一致时抛出。
    """

    if len(bundle.observations) != len(bundle.actions):
        raise ValueError(
            f"Support length mismatch for user {bundle.raw_user_id}: "
            f"obs={len(bundle.observations)}, actions={len(bundle.actions)}"
        )
    obs_norm = normalize_rows(bundle.observations.astype(np.float32))
    action_norm = normalize_rows(bundle.actions.astype(np.float32))
    joint_raw = np.concatenate([obs_norm, action_norm], axis=1)
    return normalize_rows(joint_raw).astype(np.float32)


def compute_leave_one_step_flags(
    history_prefix: np.ndarray,
    candidate_feature_lists: Sequence[Tuple[int, ...]],
    item_feature_lookup: Mapping[int, Tuple[int, ...]],
    leave_window_size: int,
    leave_threshold: float,
) -> np.ndarray:
    """按 KuaiEnv 真实规则计算所有候选动作的 `leave_1`。

    Args:
        history_prefix (np.ndarray): 产生当前锚点状态的真实物品历史前缀。
        candidate_feature_lists (Sequence[Tuple[int, ...]]): 与候选物品顺序对齐的 feature 集。
        item_feature_lookup (Mapping[int, Tuple[int, ...]]): 原始物品 ID 到 feature 集的映射。
        leave_window_size (int): `num_leave_compute`。
        leave_threshold (float): `leave_threshold`。

    Returns:
        np.ndarray: 形状为 `(n_item,)` 的布尔数组，`True` 表示一步触发离开。

    Raises:
        ValueError: 当 `leave_window_size` 非正时抛出。
    """

    if leave_window_size <= 0:
        raise ValueError("leave_window_size must be positive.")

    window_items = history_prefix[max(0, len(history_prefix) - leave_window_size):]
    feature_counter: Counter = Counter()
    for raw_item_id in window_items.tolist():
        for feature_id in item_feature_lookup.get(int(raw_item_id), ()):
            feature_counter[int(feature_id)] += 1

    leave_flags = np.zeros(len(candidate_feature_lists), dtype=bool)
    for item_index, feature_tuple in enumerate(candidate_feature_lists):
        leave_flags[item_index] = any(
            feature_counter[int(feature_id)] > leave_threshold
            for feature_id in feature_tuple
        )
    return leave_flags


def build_valid_candidate_mask(
    history_prefix: np.ndarray,
    candidate_raw_item_ids: np.ndarray,
    use_nx0_mask: bool,
    logged_item_id: int,
    item_position_lookup: Mapping[int, int],
) -> np.ndarray:
    """根据 `NX_0` 规则构造当前锚点的有效候选掩码。

    Args:
        history_prefix (np.ndarray): 当前锚点前的真实物品历史前缀。
        candidate_raw_item_ids (np.ndarray): 候选物品原始 ID 数组。
        use_nx0_mask (bool): 是否启用 `NX_0` 过滤。
        logged_item_id (int): 当前时间步真实执行的原始物品 ID。
        item_position_lookup (Mapping[int, int]): 原始物品 ID 到候选列位置的映射。

    Returns:
        np.ndarray: 形状为 `(n_item,)` 的布尔掩码。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    valid_mask = np.ones(len(candidate_raw_item_ids), dtype=bool)
    if use_nx0_mask and len(history_prefix) > 0:
        seen_items = set(int(raw_item_id) for raw_item_id in history_prefix.tolist())
        valid_mask = ~np.isin(candidate_raw_item_ids, np.asarray(list(seen_items), dtype=np.int64))

    logged_position = item_position_lookup.get(int(logged_item_id), None)
    if logged_position is not None:
        valid_mask[int(logged_position)] = True
    return valid_mask


def select_top_indices(
    scores: np.ndarray,
    mask: np.ndarray,
    top_k: int,
    item_ids: np.ndarray,
    largest: bool = True,
) -> np.ndarray:
    """从候选全集中按分数选择前若干个全局索引。

    Args:
        scores (np.ndarray): 与候选顺序对齐的一维分数数组。
        mask (np.ndarray): 指示可选候选的布尔掩码。
        top_k (int): 需要保留的数量。
        item_ids (np.ndarray): 与候选顺序对齐的原始物品 ID，用于稳定排序。
        largest (bool): 为 `True` 时取大值，反之取小值。

    Returns:
        np.ndarray: 全局候选索引数组。

    Raises:
        ValueError: 当输入维度不一致时抛出。
    """

    if len(scores) != len(mask) or len(scores) != len(item_ids):
        raise ValueError("scores/mask/item_ids must have the same length.")
    if top_k <= 0:
        return np.zeros(0, dtype=np.int32)

    candidate_indices = np.where(mask)[0]
    if len(candidate_indices) == 0:
        return np.zeros(0, dtype=np.int32)

    masked_scores = scores[candidate_indices]
    masked_item_ids = item_ids[candidate_indices]
    sort_scores = masked_scores if largest else -masked_scores
    order = stable_argsort_desc(sort_scores, masked_item_ids)
    keep_count = min(top_k, len(candidate_indices))
    return candidate_indices[order[:keep_count]].astype(np.int32)


def build_shortlist_indices(
    valid_mask: np.ndarray,
    rhat_full: np.ndarray,
    uncertainty_full: np.ndarray,
    rtrue_full: np.ndarray,
    leave_flags_full: np.ndarray,
    candidate_raw_item_ids: np.ndarray,
    logged_item_id: int,
    item_position_lookup: Mapping[int, int],
    rng: np.random.Generator,
    config: Mapping[str, Any],
) -> np.ndarray:
    """按两阶段策略构造当前锚点的候选短名单。

    Args:
        valid_mask (np.ndarray): 当前锚点的有效候选掩码。
        rhat_full (np.ndarray): 全候选 `r_hat` 分数。
        uncertainty_full (np.ndarray): 全候选 `U_current` 分数。
        rtrue_full (np.ndarray): 全候选真实 reward。
        leave_flags_full (np.ndarray): 全候选一步离开标签。
        candidate_raw_item_ids (np.ndarray): 当前候选原始物品 ID。
        logged_item_id (int): 当前时间步真实执行的原始物品 ID。
        item_position_lookup (Mapping[int, int]): 原始物品 ID 到候选列位置的映射。
        rng (np.random.Generator): 随机数生成器，用于确定性随机采样。
        config (Mapping[str, Any]): 实验配置字典。

    Returns:
        np.ndarray: 升序排列的短名单全局索引。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    selected_indices: set = set()

    for field_name, scores, largest in [
        ("shortlist_top_rhat", rhat_full, True),
        ("shortlist_top_uncertainty", uncertainty_full, True),
        ("shortlist_top_true_high", rtrue_full, True),
        ("shortlist_top_true_low", rtrue_full, False),
    ]:
        top_k = int(config.get(field_name, 0))
        chosen = select_top_indices(
            scores=scores,
            mask=valid_mask,
            top_k=top_k,
            item_ids=candidate_raw_item_ids,
            largest=largest,
        )
        selected_indices.update(int(index) for index in chosen.tolist())

    leave_mask = valid_mask & leave_flags_full
    leave_top_k = int(config.get("shortlist_top_leave", 0))
    leave_indices = select_top_indices(
        scores=uncertainty_full,
        mask=leave_mask,
        top_k=leave_top_k,
        item_ids=candidate_raw_item_ids,
        largest=True,
    )
    selected_indices.update(int(index) for index in leave_indices.tolist())

    random_count = int(config.get("shortlist_random", 0))
    valid_indices = np.where(valid_mask)[0]
    if random_count > 0 and len(valid_indices) > 0:
        chosen_random = rng.choice(valid_indices, size=min(random_count, len(valid_indices)), replace=False)
        selected_indices.update(int(index) for index in chosen_random.tolist())

    logged_position = item_position_lookup.get(int(logged_item_id), None)
    if logged_position is not None:
        selected_indices.add(int(logged_position))

    if len(selected_indices) == 0:
        return np.zeros(0, dtype=np.int32)
    return np.asarray(sorted(selected_indices), dtype=np.int32)


def compute_state_action_distance(
    observation: np.ndarray,
    shortlist_indices: np.ndarray,
    candidate_action_norm: np.ndarray,
    support_joint: np.ndarray,
) -> np.ndarray:
    """计算当前锚点到同用户 support set 的最小 state-action 余弦距离。

    Args:
        observation (np.ndarray): 当前锚点状态向量。
        shortlist_indices (np.ndarray): 需要计算距离的候选全局索引。
        candidate_action_norm (np.ndarray): 归一化后的候选动作嵌入。
        support_joint (np.ndarray): 当前用户的 support state-action joint 向量。

    Returns:
        np.ndarray: 与 `shortlist_indices` 对齐的距离数组。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    if len(shortlist_indices) == 0 or len(support_joint) == 0:
        return np.full(len(shortlist_indices), np.nan, dtype=np.float32)

    obs_norm = normalize_rows(np.expand_dims(observation.astype(np.float32), axis=0))[0]
    action_norm = candidate_action_norm[shortlist_indices]
    obs_tile = np.repeat(np.expand_dims(obs_norm, axis=0), len(shortlist_indices), axis=0)
    candidate_joint = normalize_rows(np.concatenate([obs_tile, action_norm], axis=1))
    cosine_sim = np.matmul(candidate_joint, support_joint.T)
    return (1.0 - np.max(cosine_sim, axis=1)).astype(np.float32)


def lambda_to_token(lambda_value: float) -> str:
    """把浮点 λ 值转为适合列名的稳定 token。

    Args:
        lambda_value (float): 惩罚系数。

    Returns:
        str: 仅包含字母数字和下划线的 token。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    token = f"{lambda_value:.6g}"
    token = token.replace("-", "m").replace(".", "p")
    return token


def build_rank_lookup(
    scores: np.ndarray,
    item_ids: np.ndarray,
) -> np.ndarray:
    """根据分数构造从局部候选位置到排名的映射。

    Args:
        scores (np.ndarray): 有效候选上的打分数组。
        item_ids (np.ndarray): 与 `scores` 对齐的原始物品 ID。

    Returns:
        np.ndarray: 与 `scores` 对齐的 1-based 排名数组。

    Raises:
        ValueError: 当输入长度不一致时抛出。
    """

    if len(scores) != len(item_ids):
        raise ValueError("scores and item_ids must have the same length.")
    order = stable_argsort_desc(scores, item_ids)
    return ranks_from_order(order).astype(np.int32)


def evaluate_single_anchor(
    anchor: AnchorSample,
    bundle: UserTrajectoryBundle,
    prepared_assets: PreparedExperimentAssets,
    support_joint: np.ndarray,
    lambda_values: Sequence[float],
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, str]:
    """评估单个锚点状态上的候选、标签与排名信息。

    Args:
        anchor (AnchorSample): 当前锚点状态。
        bundle (UserTrajectoryBundle): 该用户的轨迹与历史对象。
        prepared_assets (PreparedExperimentAssets): 实验静态资源。
        support_joint (np.ndarray): 当前用户的 support state-action 向量。
        lambda_values (Sequence[float]): 需要扫描的 λ 列表。
        config (Mapping[str, Any]): 实验配置字典。

    Returns:
        Tuple[pd.DataFrame, str]:
            第一个返回值是当前锚点的候选样本表；
            第二个返回值是锚点状态码，例如 `ok`、`no_valid_candidate`。

    Raises:
        KeyError: 当真实 logged item 无法映射到候选物品空间时抛出。
    """

    history_prefix = bundle.history_item_ids[: anchor.step_index]
    rng_seed = int(config["seed"]) + int(anchor.raw_user_id) * 1_000_003 + int(anchor.step_index)
    rng = np.random.default_rng(rng_seed)

    valid_mask = build_valid_candidate_mask(
        history_prefix=history_prefix,
        candidate_raw_item_ids=prepared_assets.candidate_raw_item_ids,
        use_nx0_mask=bool(config.get("use_nx0_mask", True)),
        logged_item_id=anchor.logged_item_id,
        item_position_lookup=prepared_assets.item_position_lookup,
    )
    valid_global_indices = np.where(valid_mask)[0]
    if len(valid_global_indices) == 0:
        return pd.DataFrame(), "no_valid_candidate"

    user_index = anchor.user_local_index
    rhat_full = prepared_assets.pred_slice[user_index]
    uncertainty_full = prepared_assets.var_slice[user_index]
    rtrue_full = prepared_assets.raw_reward_slice[user_index]
    rtrue_norm_full = prepared_assets.norm_reward_slice[user_index]
    leave_flags_full = compute_leave_one_step_flags(
        history_prefix=history_prefix,
        candidate_feature_lists=prepared_assets.candidate_feature_lists,
        item_feature_lookup=prepared_assets.item_feature_lookup,
        leave_window_size=int(config["num_leave_compute"]),
        leave_threshold=float(config["leave_threshold"]),
    )

    shortlist_indices = build_shortlist_indices(
        valid_mask=valid_mask,
        rhat_full=rhat_full,
        uncertainty_full=uncertainty_full,
        rtrue_full=rtrue_full,
        leave_flags_full=leave_flags_full,
        candidate_raw_item_ids=prepared_assets.candidate_raw_item_ids,
        logged_item_id=anchor.logged_item_id,
        item_position_lookup=prepared_assets.item_position_lookup,
        rng=rng,
        config=config,
    )
    if len(shortlist_indices) == 0:
        return pd.DataFrame(), "empty_shortlist"

    valid_item_ids = prepared_assets.candidate_raw_item_ids[valid_global_indices]
    valid_rhat = rhat_full[valid_global_indices]
    valid_uncertainty = uncertainty_full[valid_global_indices]
    rank_raw = build_rank_lookup(scores=valid_rhat, item_ids=valid_item_ids)
    rank_pen_map: Dict[str, np.ndarray] = {}
    for lambda_value in lambda_values:
        rank_pen_map[lambda_to_token(lambda_value)] = build_rank_lookup(
            scores=valid_rhat - float(lambda_value) * valid_uncertainty,
            item_ids=valid_item_ids,
        )

    global_to_valid_position = {int(global_index): position for position, global_index in enumerate(valid_global_indices.tolist())}
    shortlist_valid_positions = np.asarray(
        [global_to_valid_position[int(global_index)] for global_index in shortlist_indices.tolist()],
        dtype=np.int32,
    )

    distances = compute_state_action_distance(
        observation=anchor.observation,
        shortlist_indices=shortlist_indices,
        candidate_action_norm=prepared_assets.candidate_action_norm,
        support_joint=support_joint,
    )
    if not np.isfinite(distances).any():
        return pd.DataFrame(), "nan_distance"

    q20 = float(np.quantile(distances, float(config["near_q"])))
    q50 = float(np.quantile(distances, 0.5))
    q80 = float(np.quantile(distances, float(config["far_q"])))

    is_id = distances <= q20
    is_near = (distances > q20) & (distances <= q50)
    is_far = distances >= q80
    ood_mask = is_near | is_far
    can_label = bool(np.sum(ood_mask) >= 2)

    reward_q_high = np.nan
    reward_q_low = np.nan
    is_high = np.zeros(len(shortlist_indices), dtype=bool)
    is_low = np.zeros(len(shortlist_indices), dtype=bool)
    if can_label:
        ood_rewards = rtrue_full[shortlist_indices][ood_mask]
        reward_q_high = float(np.quantile(ood_rewards, float(config["reward_high_q"])))
        reward_q_low = float(np.quantile(ood_rewards, float(config["reward_low_q"])))
        is_high = rtrue_full[shortlist_indices] >= reward_q_high
        is_low = rtrue_full[shortlist_indices] <= reward_q_low

    leave_short = leave_flags_full[shortlist_indices]
    quadrant_label = np.full(len(shortlist_indices), "Other", dtype=object)
    quadrant_label[is_near & is_high] = "Near-High"
    quadrant_label[is_near & is_low] = "Near-Low"
    quadrant_label[is_far & is_high] = "Far-High"
    quadrant_label[is_far & is_low] = "Far-Low"

    main_label = np.full(len(shortlist_indices), "Other", dtype=object)
    is_nh = is_near & is_high & (~leave_short)
    is_do = is_far & (is_low | leave_short)
    is_near_low = is_near & is_low
    is_far_high = is_far & is_high & (~leave_short)

    main_label[is_nh] = "NH"
    main_label[is_do] = "DO"
    main_label[~is_nh & ~is_do & is_near_low] = "Near-Low"
    main_label[~is_nh & ~is_do & is_far_high] = "Far-High"

    ood_region = np.full(len(shortlist_indices), "Mid-OOD", dtype=object)
    ood_region[is_id] = "ID"
    ood_region[is_near] = "Near-OOD"
    ood_region[is_far] = "Far-OOD"

    value_region = np.full(len(shortlist_indices), "Unavailable", dtype=object)
    if can_label:
        value_region[:] = "Mid"
        value_region[is_high] = "High"
        value_region[is_low] = "Low"

    frame_dict: Dict[str, Any] = {
        "user_id": np.full(len(shortlist_indices), anchor.raw_user_id, dtype=np.int64),
        "anchor_t": np.full(len(shortlist_indices), anchor.step_index, dtype=np.int32),
        "trajectory_length": np.full(len(shortlist_indices), anchor.trajectory_length, dtype=np.int32),
        "history_length": np.full(len(shortlist_indices), len(history_prefix), dtype=np.int32),
        "valid_candidate_count": np.full(len(shortlist_indices), len(valid_global_indices), dtype=np.int32),
        "shortlist_size": np.full(len(shortlist_indices), len(shortlist_indices), dtype=np.int32),
        "item_id": prepared_assets.candidate_raw_item_ids[shortlist_indices].astype(np.int64),
        "item_small_index": prepared_assets.candidate_small_indices[shortlist_indices].astype(np.int32),
        "is_logged_action": (
            prepared_assets.candidate_raw_item_ids[shortlist_indices].astype(np.int64) == int(anchor.logged_item_id)
        ).astype(np.uint8),
        "r_hat": rhat_full[shortlist_indices].astype(np.float32),
        "u_current": uncertainty_full[shortlist_indices].astype(np.float32),
        "r_true": rtrue_full[shortlist_indices].astype(np.float32),
        "r_true_normed": rtrue_norm_full[shortlist_indices].astype(np.float32),
        "leave_1": leave_short.astype(np.uint8),
        "distance_sa": distances.astype(np.float32),
        "distance_q20": np.full(len(shortlist_indices), q20, dtype=np.float32),
        "distance_q50": np.full(len(shortlist_indices), q50, dtype=np.float32),
        "distance_q80": np.full(len(shortlist_indices), q80, dtype=np.float32),
        "reward_q_high": np.full(len(shortlist_indices), reward_q_high, dtype=np.float32),
        "reward_q_low": np.full(len(shortlist_indices), reward_q_low, dtype=np.float32),
        "ood_region": ood_region,
        "value_region": value_region,
        "quadrant_label": quadrant_label,
        "main_label": main_label,
        "is_nh": is_nh.astype(np.uint8),
        "is_do": is_do.astype(np.uint8),
        "can_label": np.full(len(shortlist_indices), int(can_label), dtype=np.uint8),
        "rank_raw": rank_raw[shortlist_valid_positions].astype(np.int32),
    }
    for lambda_value in lambda_values:
        lambda_token = lambda_to_token(lambda_value)
        frame_dict[f"rank_pen_{lambda_token}"] = rank_pen_map[lambda_token][shortlist_valid_positions].astype(np.int32)

    return pd.DataFrame(frame_dict), "ok"


def save_dataframe_with_fallback(
    dataframe: pd.DataFrame,
    output_dir: str,
    base_filename: str,
) -> Dict[str, Optional[str]]:
    """优先保存 Parquet，失败时回退为压缩 CSV。

    Args:
        dataframe (pd.DataFrame): 需要保存的数据表。
        output_dir (str): 输出目录。
        base_filename (str): 不带扩展名的文件名。

    Returns:
        Dict[str, Optional[str]]: 保存路径与回退信息。

    Raises:
        OSError: 当两种保存方式都失败时抛出。
    """

    parquet_path = os.path.join(output_dir, f"{base_filename}.parquet")
    csv_path = os.path.join(output_dir, f"{base_filename}.csv.gz")
    try:
        dataframe.to_parquet(parquet_path, index=False)
        return {"path": parquet_path, "format": "parquet", "fallback_error": None}
    except Exception as error_msg:
        dataframe.to_csv(csv_path, index=False, compression="gzip")
        return {"path": csv_path, "format": "csv_gzip", "fallback_error": str(error_msg)}


def safe_ks_statistic(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """安全计算两组样本的 KS 统计量。

    Args:
        values_a (np.ndarray): 第一组样本。
        values_b (np.ndarray): 第二组样本。

    Returns:
        float: KS 统计量；样本不足时返回 `nan`。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    if len(values_a) == 0 or len(values_b) == 0:
        return float("nan")
    return float(ks_2samp(values_a, values_b).statistic)


def safe_wasserstein(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """安全计算两组样本的一维 Wasserstein 距离。

    Args:
        values_a (np.ndarray): 第一组样本。
        values_b (np.ndarray): 第二组样本。

    Returns:
        float: Wasserstein 距离；样本不足时返回 `nan`。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    if len(values_a) == 0 or len(values_b) == 0:
        return float("nan")
    return float(wasserstein_distance(values_a, values_b))


def compute_discrimination_metrics(
    sample_df: pd.DataFrame,
    alpha_list: Sequence[float],
) -> Dict[str, Any]:
    """计算区分 NH 与 DO 的核心统计量。

    Args:
        sample_df (pd.DataFrame): 候选样本表。
        alpha_list (Sequence[float]): 需要评估的高不确定比例列表。

    Returns:
        Dict[str, Any]: 区分能力相关的指标字典。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    if sample_df.empty or "main_label" not in sample_df.columns or "u_current" not in sample_df.columns:
        return {
            "nh_do_pair_count": 0,
            "strict_pool_count": 0,
            "auc_do_vs_nh": float("nan"),
            "pr_auc_do_vs_nh": float("nan"),
            "ks_u_current_nh_vs_do": float("nan"),
            "wasserstein_u_current_nh_vs_do": float("nan"),
            "highvar_stats": {
                str(alpha_value): {
                    "threshold": float("nan"),
                    "highvar_count": 0,
                    "contamination_nh": float("nan"),
                    "danger_capture": float("nan"),
                }
                for alpha_value in alpha_list
            },
        }

    nh_do_df = sample_df[sample_df["main_label"].isin(["NH", "DO"])].copy()
    strict_df = sample_df[sample_df["main_label"].isin(HIGHVAR_LABELS)].copy()

    metrics: Dict[str, Any] = {
        "nh_do_pair_count": int(len(nh_do_df)),
        "strict_pool_count": int(len(strict_df)),
    }
    if len(nh_do_df) == 0:
        metrics.update(
            {
                "auc_do_vs_nh": float("nan"),
                "pr_auc_do_vs_nh": float("nan"),
                "ks_u_current_nh_vs_do": float("nan"),
                "wasserstein_u_current_nh_vs_do": float("nan"),
            }
        )
    else:
        y_true = (nh_do_df["main_label"] == "DO").astype(np.uint8).to_numpy()
        y_score = nh_do_df["u_current"].to_numpy(dtype=np.float32)
        nh_scores = nh_do_df.loc[nh_do_df["main_label"] == "NH", "u_current"].to_numpy(dtype=np.float32)
        do_scores = nh_do_df.loc[nh_do_df["main_label"] == "DO", "u_current"].to_numpy(dtype=np.float32)
        metrics.update(
            {
                "auc_do_vs_nh": safe_roc_auc(y_true, y_score),
                "pr_auc_do_vs_nh": safe_average_precision(y_true, y_score),
                "ks_u_current_nh_vs_do": safe_ks_statistic(nh_scores, do_scores),
                "wasserstein_u_current_nh_vs_do": safe_wasserstein(nh_scores, do_scores),
                "nh_u_current_mean": float(np.mean(nh_scores)) if len(nh_scores) else float("nan"),
                "do_u_current_mean": float(np.mean(do_scores)) if len(do_scores) else float("nan"),
                "nh_u_current_median": float(np.median(nh_scores)) if len(nh_scores) else float("nan"),
                "do_u_current_median": float(np.median(do_scores)) if len(do_scores) else float("nan"),
            }
        )

    highvar_stats: Dict[str, Any] = {}
    if len(strict_df) == 0:
        for alpha_value in alpha_list:
            highvar_stats[str(alpha_value)] = {
                "threshold": float("nan"),
                "highvar_count": 0,
                "contamination_nh": float("nan"),
                "danger_capture": float("nan"),
            }
    else:
        strict_scores = strict_df["u_current"].to_numpy(dtype=np.float32)
        strict_labels = strict_df["main_label"].to_numpy(dtype=object)
        do_total = int(np.sum(strict_labels == "DO"))
        for alpha_value in alpha_list:
            threshold = float(np.quantile(strict_scores, 1.0 - float(alpha_value)))
            highvar_mask = strict_scores >= threshold
            highvar_labels = strict_labels[highvar_mask]
            highvar_stats[str(alpha_value)] = {
                "threshold": threshold,
                "highvar_count": int(np.sum(highvar_mask)),
                "contamination_nh": float(np.mean(highvar_labels == "NH")) if len(highvar_labels) else float("nan"),
                "danger_capture": (
                    float(np.sum((strict_labels == "DO") & highvar_mask) / do_total)
                    if do_total > 0 else float("nan")
                ),
            }
    metrics["highvar_stats"] = highvar_stats
    return metrics


def compute_lambda_metrics(
    sample_df: pd.DataFrame,
    lambda_values: Sequence[float],
    topk_list: Sequence[int],
) -> pd.DataFrame:
    """计算不同 λ 下的惩罚影响指标。

    Args:
        sample_df (pd.DataFrame): 候选样本表。
        lambda_values (Sequence[float]): 需要扫描的 λ 列表。
        topk_list (Sequence[int]): 需要计算的 `k` 列表。

    Returns:
        pd.DataFrame: 以 `lambda_value` 与 `topk` 为索引维度的指标表。

    Raises:
        KeyError: 当所需的排名列缺失时抛出。
    """

    if sample_df.empty or "main_label" not in sample_df.columns or "rank_raw" not in sample_df.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    nh_mask = sample_df["main_label"] == "NH"
    do_mask = sample_df["main_label"] == "DO"
    uncertainty = sample_df["u_current"].to_numpy(dtype=np.float32)
    rank_raw = sample_df["rank_raw"].to_numpy(dtype=np.int32)

    for lambda_value in lambda_values:
        lambda_token = lambda_to_token(lambda_value)
        rank_column = f"rank_pen_{lambda_token}"
        if rank_column not in sample_df.columns:
            raise KeyError(f"Missing ranking column: {rank_column}")
        rank_pen = sample_df[rank_column].to_numpy(dtype=np.int32)

        nh_loss = float(np.mean(float(lambda_value) * uncertainty[nh_mask.to_numpy()])) if np.any(nh_mask) else float("nan")
        do_gain = float(np.mean(float(lambda_value) * uncertainty[do_mask.to_numpy()])) if np.any(do_mask) else float("nan")

        for top_k in topk_list:
            nh_denom_mask = nh_mask.to_numpy() & (rank_raw <= int(top_k))
            do_denom_mask = do_mask.to_numpy() & (rank_raw <= int(top_k))
            nh_suppressed_count = int(np.sum(nh_denom_mask & (rank_pen > int(top_k))))
            do_suppressed_count = int(np.sum(do_denom_mask & (rank_pen > int(top_k))))
            suppressed_total = int(nh_suppressed_count + do_suppressed_count)

            fsr = (
                float(nh_suppressed_count / np.sum(nh_denom_mask))
                if np.sum(nh_denom_mask) > 0 else float("nan")
            )
            drr = (
                float(do_suppressed_count / np.sum(do_denom_mask))
                if np.sum(do_denom_mask) > 0 else float("nan")
            )
            gg = float(drr - fsr) if np.isfinite(fsr) and np.isfinite(drr) else float("nan")

            rows.append(
                {
                    "lambda_value": float(lambda_value),
                    "lambda_token": lambda_token,
                    "topk": int(top_k),
                    "fsr": fsr,
                    "drr": drr,
                    "gg": gg,
                    "nh_loss": nh_loss,
                    "do_gain": do_gain,
                    "nh_suppressed_count": nh_suppressed_count,
                    "do_suppressed_count": do_suppressed_count,
                    "suppressed_total": suppressed_total,
                    "nh_suppressed_share": (
                        float(nh_suppressed_count / suppressed_total)
                        if suppressed_total > 0 else float("nan")
                    ),
                    "do_suppressed_share": (
                        float(do_suppressed_count / suppressed_total)
                        if suppressed_total > 0 else float("nan")
                    ),
                    "nh_rank_raw_count": int(np.sum(nh_denom_mask)),
                    "do_rank_raw_count": int(np.sum(do_denom_mask)),
                }
            )

    return pd.DataFrame(rows)


def summarize_anchor_statuses(status_list: Sequence[str]) -> Dict[str, int]:
    """汇总锚点评估状态计数。

    Args:
        status_list (Sequence[str]): 逐锚点状态码列表。

    Returns:
        Dict[str, int]: 每种状态码的出现次数。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    return {str(status): int(count) for status, count in Counter(status_list).items()}


def save_empty_figure(output_path: str, text: str) -> None:
    """生成占位空图。

    Args:
        output_path (str): 图片输出路径。
        text (str): 图中央展示的文本。

    Returns:
        None: 图片直接写入磁盘。

    Raises:
        OSError: 当输出路径不可写时抛出。
    """

    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.5, text, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_distance_vs_true_reward(
    sample_df: pd.DataFrame,
    output_path: str,
    plot_sample_n: int,
    seed: int,
) -> None:
    """绘制四象限样本上的 `d_sa` 与 `r_true` 散点图。

    Args:
        sample_df (pd.DataFrame): 候选样本表。
        output_path (str): 图片输出路径。
        plot_sample_n (int): 最多采样的点数。
        seed (int): 随机种子。

    Returns:
        None: 图片直接写入磁盘。

    Raises:
        OSError: 当输出路径不可写时抛出。
    """

    if (
        sample_df.empty
        or "distance_sa" not in sample_df.columns
        or "quadrant_label" not in sample_df.columns
        or "r_true" not in sample_df.columns
    ):
        save_empty_figure(output_path, EMPTY_TEXT)
        return

    plot_df = sample_df[
        np.isfinite(sample_df["distance_sa"])
        & sample_df["quadrant_label"].isin(PLOTTED_QUADRANT_LABELS)
    ].copy()
    if len(plot_df) == 0:
        save_empty_figure(output_path, EMPTY_TEXT)
        return

    plot_df = plot_df.sample(min(len(plot_df), plot_sample_n), random_state=seed)

    plt.figure(figsize=(7.2, 5.2))
    sns.scatterplot(
        data=plot_df,
        x="distance_sa",
        y="r_true",
        hue="quadrant_label",
        hue_order=PLOTTED_QUADRANT_LABELS,
        alpha=0.38,
        linewidth=0.0,
    )
    plt.xlabel("State-Action Distance $d_{sa}(s_t, a)$")
    plt.ylabel("True Reward $r_{true}(s_t, a)$")
    plt.title("Distance vs True Reward on Quadrant-Labeled Samples")
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_distance_vs_uncertainty(
    sample_df: pd.DataFrame,
    output_path: str,
    plot_sample_n: int,
    seed: int,
) -> None:
    """绘制四象限样本上的距离与不确定性关系图。

    该图将原先散点图里通过点大小编码的不确定性单独拆出，用更直观的
    二维散点图展示 `distance_sa` 与 `u_current` 的关系。为了减小长尾
    对纵轴可读性的影响，纵轴使用对数尺度，但不裁剪原始数值。

    Args:
        sample_df (pd.DataFrame): 候选样本表。
        output_path (str): 图片输出路径。
        plot_sample_n (int): 最多采样的点数。
        seed (int): 随机种子。

    Returns:
        None: 图片直接写入磁盘。

    Raises:
        OSError: 当输出路径不可写时抛出。
    """

    required_columns = {"distance_sa", "u_current", "quadrant_label"}
    if sample_df.empty or not required_columns.issubset(sample_df.columns):
        save_empty_figure(output_path, EMPTY_TEXT)
        return

    plot_df = sample_df[
        np.isfinite(sample_df["distance_sa"])
        & np.isfinite(sample_df["u_current"])
        & (sample_df["u_current"] > 0.0)
        & sample_df["quadrant_label"].isin(PLOTTED_QUADRANT_LABELS)
    ].copy()
    if len(plot_df) == 0:
        save_empty_figure(output_path, EMPTY_TEXT)
        return

    plot_df = plot_df.sample(min(len(plot_df), plot_sample_n), random_state=seed)

    plt.figure(figsize=(7.2, 5.2))
    sns.scatterplot(
        data=plot_df,
        x="distance_sa",
        y="u_current",
        hue="quadrant_label",
        hue_order=PLOTTED_QUADRANT_LABELS,
        alpha=0.38,
        linewidth=0.0,
    )
    plt.yscale("log")
    plt.xlabel("State-Action Distance $d_{sa}(s_t, a)$")
    plt.ylabel("Uncertainty $U_{current}(u, a)$ (log scale)")
    plt.title("Distance vs Uncertainty on Quadrant-Labeled Samples")
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_uncertainty_violin(
    sample_df: pd.DataFrame,
    output_path: str,
    clip_quantile: float,
) -> None:
    """绘制 `NH/DO` 主标签上的不确定性分布图。

    为了避免极少数长尾样本把主体分布压缩到底部，该图会按给定分位数
    裁掉最上侧极端值，仅用于可视化展示，不影响原始统计指标。

    Args:
        sample_df (pd.DataFrame): 候选样本表。
        output_path (str): 图片输出路径。
        clip_quantile (float): 仅保留该分位数以下的不确定性样本。

    Returns:
        None: 图片直接写入磁盘。

    Raises:
        OSError: 当输出路径不可写时抛出。
    """

    if sample_df.empty or "main_label" not in sample_df.columns or "u_current" not in sample_df.columns:
        save_empty_figure(output_path, EMPTY_TEXT)
        return

    if not (0.0 < clip_quantile <= 1.0):
        raise ValueError("clip_quantile must be in (0, 1].")

    plot_df = sample_df[sample_df["main_label"].isin(VIOLIN_MAIN_LABELS)].copy()
    if len(plot_df) == 0:
        save_empty_figure(output_path, EMPTY_TEXT)
        return

    upper_bound = float(np.quantile(plot_df["u_current"], clip_quantile))
    plot_df = plot_df[plot_df["u_current"] <= upper_bound].copy()
    if len(plot_df) == 0:
        save_empty_figure(output_path, EMPTY_TEXT)
        return

    plt.figure(figsize=(7.4, 5.2))
    sns.violinplot(
        data=plot_df,
        x="main_label",
        y="u_current",
        order=VIOLIN_MAIN_LABELS,
        inner="quartile",
        cut=0,
    )
    plt.xlabel("Main Label")
    plt.ylabel("Uncertainty $U_{current}(u, a)$")
    plt.title(f"NH vs DO Uncertainty (<= p{clip_quantile * 100.0:.0f} for readability)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_highvar_composition(
    sample_df: pd.DataFrame,
    output_path: str,
    alpha_value: float,
) -> None:
    """绘制高不确定集合的标签组成柱状图。

    Args:
        sample_df (pd.DataFrame): 候选样本表。
        output_path (str): 图片输出路径。
        alpha_value (float): 取前多少比例的不确定样本。

    Returns:
        None: 图片直接写入磁盘。

    Raises:
        OSError: 当输出路径不可写时抛出。
    """

    if sample_df.empty or "main_label" not in sample_df.columns or "u_current" not in sample_df.columns:
        save_empty_figure(output_path, EMPTY_TEXT)
        return

    strict_df = sample_df[sample_df["main_label"].isin(HIGHVAR_LABELS)].copy()
    if len(strict_df) == 0:
        save_empty_figure(output_path, EMPTY_TEXT)
        return

    threshold = float(np.quantile(strict_df["u_current"], 1.0 - alpha_value))
    highvar_df = strict_df[strict_df["u_current"] >= threshold].copy()

    all_ratio = (
        strict_df["main_label"].value_counts(normalize=True)
        .reindex(HIGHVAR_LABELS, fill_value=0.0)
        .rename_axis("main_label")
        .reset_index(name="ratio")
    )
    all_ratio["subset"] = "All"

    highvar_ratio = (
        highvar_df["main_label"].value_counts(normalize=True)
        .reindex(HIGHVAR_LABELS, fill_value=0.0)
        .rename_axis("main_label")
        .reset_index(name="ratio")
    )
    highvar_ratio["subset"] = f"HighVar@{alpha_value:.0%}"

    plot_df = pd.concat([all_ratio, highvar_ratio], axis=0, ignore_index=True)
    plt.figure(figsize=(7.4, 5.2))
    sns.barplot(data=plot_df, x="main_label", y="ratio", hue="subset", order=HIGHVAR_LABELS)
    plt.xlabel("Main Label")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_lambda_response(
    lambda_df: pd.DataFrame,
    output_path: str,
    lambda_plot_topk: int,
) -> None:
    """绘制 λ 扫描下的 `FSR/DRR/GG` 响应曲线。

    Args:
        lambda_df (pd.DataFrame): λ 指标汇总表。
        output_path (str): 图片输出路径。
        lambda_plot_topk (int): 用于作图的 `k`。

    Returns:
        None: 图片直接写入磁盘。

    Raises:
        OSError: 当输出路径不可写时抛出。
    """

    plot_df = lambda_df[lambda_df["topk"] == int(lambda_plot_topk)].copy()
    if len(plot_df) == 0:
        save_empty_figure(output_path, EMPTY_TEXT)
        return

    plot_df = plot_df.melt(
        id_vars=["lambda_value"],
        value_vars=["fsr", "drr", "gg"],
        var_name="metric",
        value_name="value",
    )
    metric_mapping = {"fsr": "FSR", "drr": "DRR", "gg": "GG"}
    plot_df["metric"] = plot_df["metric"].map(metric_mapping)

    plt.figure(figsize=(7.4, 5.2))
    sns.lineplot(
        data=plot_df,
        x="lambda_value",
        y="value",
        hue="metric",
        marker="o",
        linewidth=2.0,
    )
    plt.xlabel("Lambda Variance")
    plt.ylabel("Metric Value")
    plt.title(f"Penalty Response @ top-{int(lambda_plot_topk)}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_lambda_suppressed_mix(
    lambda_df: pd.DataFrame,
    output_path: str,
    lambda_plot_topk: int,
) -> None:
    """绘制不同 λ 下被惩罚抑制的 `NH/DO` 数量与组成图。

    左侧子图展示在给定 `top-k` 下，被惩罚从前 `k` 名中挤出的 `NH/DO`
    绝对数量；右侧子图展示同一被抑制集合内部的 `NH/DO` 组成比例。
    该图用于直接回答“同一惩罚强度下，被压制样本中是否同时混有 NH
    与 DO”这一问题。

    Args:
        lambda_df (pd.DataFrame): λ 指标汇总表。
        output_path (str): 图片输出路径。
        lambda_plot_topk (int): 用于作图的 `k`。

    Returns:
        None: 图片直接写入磁盘。

    Raises:
        OSError: 当输出路径不可写时抛出。
    """

    plot_df = lambda_df[lambda_df["topk"] == int(lambda_plot_topk)].copy()
    required_columns = {
        "lambda_value",
        "nh_suppressed_count",
        "do_suppressed_count",
        "nh_suppressed_share",
        "do_suppressed_share",
    }
    if len(plot_df) == 0 or not required_columns.issubset(plot_df.columns):
        save_empty_figure(output_path, EMPTY_TEXT)
        return

    plot_df["lambda_label"] = plot_df["lambda_value"].map(lambda value: f"{float(value):g}")
    x_labels = plot_df["lambda_label"].tolist()
    x_positions = np.arange(len(plot_df), dtype=np.float32)
    bar_width = 0.36

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]})

    axes[0].bar(
        x_positions - bar_width / 2.0,
        plot_df["nh_suppressed_count"].to_numpy(dtype=np.float32),
        width=bar_width,
        label="NH Suppressed",
        color="#4C72B0",
    )
    axes[0].bar(
        x_positions + bar_width / 2.0,
        plot_df["do_suppressed_count"].to_numpy(dtype=np.float32),
        width=bar_width,
        label="DO Suppressed",
        color="#DD8452",
    )
    axes[0].set_xticks(x_positions, x_labels)
    axes[0].set_xlabel("Lambda Variance")
    axes[0].set_ylabel("Suppressed Count")
    axes[0].set_title(f"Suppressed NH/DO Counts @ top-{int(lambda_plot_topk)}")
    axes[0].legend()

    axes[1].bar(
        x_positions,
        plot_df["do_suppressed_share"].to_numpy(dtype=np.float32),
        width=0.58,
        label="DO Share",
        color="#DD8452",
    )
    axes[1].bar(
        x_positions,
        plot_df["nh_suppressed_share"].to_numpy(dtype=np.float32),
        width=0.58,
        bottom=plot_df["do_suppressed_share"].to_numpy(dtype=np.float32),
        label="NH Share",
        color="#4C72B0",
    )
    axes[1].set_xticks(x_positions, x_labels)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Lambda Variance")
    axes[1].set_ylabel("Share Within Suppressed Set")
    axes[1].set_title(f"Suppressed Mix @ top-{int(lambda_plot_topk)}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def build_uncertainty_bucket_frame(
    sample_df: pd.DataFrame,
    bucket_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按不确定性从低到高构造等频分桶结果。

    这里仅在 `NH/DO` 样本上分桶，以便直接回答“相同不确定性区间内
    同时混有多少 NH 和 DO”。分桶采用等频切分，保证每个桶样本量大致
    接近，减少长尾分布导致的极端稀疏问题。

    Args:
        sample_df (pd.DataFrame): 候选样本表。
        bucket_count (int): 需要切分的桶数。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - 带有 `bucket_index/bucket_label` 的样本表
            - 每个桶的 `min/max/count` 摘要表

    Raises:
        ValueError: 当 `bucket_count` 非正时抛出。
    """

    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")

    required_columns = {"main_label", "u_current"}
    if sample_df.empty or not required_columns.issubset(sample_df.columns):
        return pd.DataFrame(), pd.DataFrame()

    bucket_df = sample_df[sample_df["main_label"].isin(VIOLIN_MAIN_LABELS)].copy()
    if len(bucket_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    sort_columns = ["u_current"]
    for optional_column in ["user_id", "anchor_t", "item_id"]:
        if optional_column in bucket_df.columns:
            sort_columns.append(optional_column)
    bucket_df = bucket_df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    effective_bucket_count = int(min(bucket_count, len(bucket_df)))
    raw_bucket_index = np.floor(np.arange(len(bucket_df), dtype=np.float64) * effective_bucket_count / len(bucket_df))
    bucket_df["bucket_index"] = raw_bucket_index.astype(np.int32) + 1

    bucket_summary = (
        bucket_df.groupby("bucket_index", as_index=False)["u_current"]
        .agg(["min", "max", "count"])
        .reset_index()
        .rename(columns={"min": "u_min", "max": "u_max", "count": "bucket_count"})
    )
    bucket_summary["bucket_label"] = bucket_summary["bucket_index"].map(lambda value: f"B{int(value)}")

    label_lookup = {
        int(row.bucket_index): str(row.bucket_label)
        for row in bucket_summary.itertuples(index=False)
    }
    bucket_df["bucket_label"] = bucket_df["bucket_index"].map(label_lookup)
    return bucket_df, bucket_summary


def plot_uncertainty_bucket_mix(
    sample_df: pd.DataFrame,
    output_path: str,
    bucket_count: int,
) -> None:
    """绘制按不确定性从低到高分桶后的 `NH/DO` 数量与组成图。

    左图展示不同 uncertainty bucket 中 `NH/DO` 的绝对数量，右图展示
    各桶内部的 `NH/DO` 组成比例，用于观察“同一不确定性惩罚区间内”
    是否同时混有 `NH` 与 `DO`。

    Args:
        sample_df (pd.DataFrame): 候选样本表。
        output_path (str): 图片输出路径。
        bucket_count (int): 需要绘制的等频桶数。

    Returns:
        None: 图片直接写入磁盘。

    Raises:
        OSError: 当输出路径不可写时抛出。
    """

    bucket_df, bucket_summary = build_uncertainty_bucket_frame(sample_df=sample_df, bucket_count=bucket_count)
    if len(bucket_df) == 0 or len(bucket_summary) == 0:
        save_empty_figure(output_path, EMPTY_TEXT)
        return

    count_df = (
        bucket_df.groupby(["bucket_label", "main_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    count_pivot = (
        count_df.pivot(index="bucket_label", columns="main_label", values="count")
        .reindex(bucket_summary["bucket_label"].tolist(), fill_value=0)
        .reindex(columns=VIOLIN_MAIN_LABELS, fill_value=0)
    )

    ratio_pivot = count_pivot.div(count_pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    x_positions = np.arange(len(count_pivot), dtype=np.float32)
    bar_width = 0.36

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), gridspec_kw={"width_ratios": [1.1, 1.0]})

    axes[0].bar(
        x_positions - bar_width / 2.0,
        count_pivot["NH"].to_numpy(dtype=np.float32),
        width=bar_width,
        label="NH Count",
        color="#4C72B0",
    )
    axes[0].bar(
        x_positions + bar_width / 2.0,
        count_pivot["DO"].to_numpy(dtype=np.float32),
        width=bar_width,
        label="DO Count",
        color="#DD8452",
    )
    axes[0].set_xticks(x_positions, count_pivot.index.tolist())
    axes[0].set_xlabel("Uncertainty Bucket (Low -> High)")
    axes[0].set_ylabel("Sample Count")
    axes[0].set_title("NH/DO Counts by Uncertainty Bucket")
    axes[0].legend()

    axes[1].bar(
        x_positions,
        ratio_pivot["DO"].to_numpy(dtype=np.float32),
        width=0.58,
        label="DO Share",
        color="#DD8452",
    )
    axes[1].bar(
        x_positions,
        ratio_pivot["NH"].to_numpy(dtype=np.float32),
        width=0.58,
        bottom=ratio_pivot["DO"].to_numpy(dtype=np.float32),
        label="NH Share",
        color="#4C72B0",
    )
    axes[1].set_xticks(x_positions, count_pivot.index.tolist())
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Uncertainty Bucket (Low -> High)")
    axes[1].set_ylabel("Share Within Bucket")
    axes[1].set_title("NH/DO Mix by Uncertainty Bucket")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def generate_markdown_report(
    config: Mapping[str, Any],
    data_summary: Mapping[str, Any],
    discrimination_metrics: Mapping[str, Any],
    lambda_df: pd.DataFrame,
    label_counts: Mapping[str, Any],
    figure_paths: Mapping[str, str],
    sample_save_info: Mapping[str, Any],
    output_dir: str,
) -> str:
    """生成实验结果 Markdown 报告文本。

    Args:
        config (Mapping[str, Any]): 实验配置。
        data_summary (Mapping[str, Any]): 数据规模与对齐摘要。
        discrimination_metrics (Mapping[str, Any]): 区分能力指标。
        lambda_df (pd.DataFrame): λ 指标表。
        label_counts (Mapping[str, Any]): 标签计数摘要。
        figure_paths (Mapping[str, str]): 图像路径字典。
        sample_save_info (Mapping[str, Any]): 样本表保存信息。
        output_dir (str): 当前实验输出根目录。

    Returns:
        str: 完整 Markdown 文本。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    lambda_table = lambda_df.copy()
    if len(lambda_table) > 0:
        lambda_table = lambda_table[["lambda_value", "topk", "fsr", "drr", "gg", "nh_loss", "do_gain"]]

    rel_figure_paths = {
        name: os.path.relpath(path, output_dir)
        for name, path in figure_paths.items()
    }
    report_lines = [
        "# OOD 识别粒度实验报告",
        "",
        "## 1. 实验设置",
        "",
        f"- 环境：`{config['env']}`",
        f"- user model：`{config['user_model_name']}`",
        f"- read_message：`{config['read_message']}`",
        f"- 轨迹文件：`{config['trajectory_pkl_path']}`",
        f"- 原始 CSV：`{config['small_csv_path']}`",
        f"- `anchor_states_per_user`：`{config['anchor_states_per_user']}`",
        f"- `use_nx0_mask`：`{config['use_nx0_mask']}`",
        f"- `lambda_grid`：`{config['lambda_grid']}`",
        f"- 样本表：`{sample_save_info['path']}` ({sample_save_info['format']})",
        "",
        "## 2. 数据与对齐摘要",
        "",
        f"- 用户数：`{data_summary['selected_user_count']}`",
        f"- 候选物品数：`{data_summary['candidate_item_count']}`",
        f"- 锚点总数：`{data_summary['anchor_total']}`",
        f"- 有效锚点数：`{data_summary['anchor_status'].get('ok', 0)}`",
        f"- 动作嵌入对齐检查：`{data_summary['action_alignment_summary']}`",
        "",
        "## 3. 标签计数",
        "",
        f"- 主标签计数：`{label_counts['main_label_counts']}`",
        f"- 四象限计数：`{label_counts['quadrant_label_counts']}`",
        "",
        "## 4. 区分能力指标",
        "",
        f"- `AUC_DO_vs_NH`：`{discrimination_metrics['auc_do_vs_nh']}`",
        f"- `PR-AUC_DO_vs_NH`：`{discrimination_metrics['pr_auc_do_vs_nh']}`",
        f"- `KS(U_current | NH, DO)`：`{discrimination_metrics['ks_u_current_nh_vs_do']}`",
        f"- `Wasserstein(U_current | NH, DO)`：`{discrimination_metrics['wasserstein_u_current_nh_vs_do']}`",
        f"- `HighVar` 统计：`{discrimination_metrics['highvar_stats']}`",
        "",
        "## 5. λ 扫描指标",
        "",
        dataframe_to_markdown(lambda_table, float_digits=4) if len(lambda_table) > 0 else "本次未得到有效 λ 扫描结果。",
        "",
        "## 6. 图表",
        "",
        f"- `d_sa` vs `r_true`：`{rel_figure_paths['distance_vs_true_reward']}`",
        f"- `d_sa` vs `U_current`：`{rel_figure_paths['distance_vs_uncertainty']}`",
        f"- uncertainty violin (`NH/DO` only, clipped)：`{rel_figure_paths['uncertainty_violin']}`",
        f"- uncertainty bucket mix：`{rel_figure_paths['uncertainty_bucket_mix']}`",
        f"- high-var composition：`{rel_figure_paths['highvar_composition']}`",
        f"- lambda response：`{rel_figure_paths['lambda_response']}`",
        f"- lambda suppressed mix：`{rel_figure_paths['lambda_suppressed_mix']}`",
        "",
        f"![distance_vs_true_reward]({rel_figure_paths['distance_vs_true_reward']})",
        "",
        f"![distance_vs_uncertainty]({rel_figure_paths['distance_vs_uncertainty']})",
        "",
        f"![uncertainty_violin]({rel_figure_paths['uncertainty_violin']})",
        "",
        f"![uncertainty_bucket_mix]({rel_figure_paths['uncertainty_bucket_mix']})",
        "",
        f"![highvar_composition]({rel_figure_paths['highvar_composition']})",
        "",
        f"![lambda_response]({rel_figure_paths['lambda_response']})",
        "",
        f"![lambda_suppressed_mix]({rel_figure_paths['lambda_suppressed_mix']})",
        "",
        "## 7. 结论草案",
        "",
        "若 `AUC_DO_vs_NH` 接近随机、`HighVar` 集合仍混入较多 `NH`，同时随 λ 增大 `FSR` 与 `DRR` 一起上升而 `GG` 不明显为正，则支持“当前不确定性惩罚粒度过粗”的判断。",
        "",
    ]
    return "\n".join(report_lines)


def run_ood_granularity_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    """执行 OOD 识别粒度主实验。

    Args:
        config (Mapping[str, Any]): 完整实验配置。

    Returns:
        Dict[str, Any]: 包含输出路径与摘要指标的结果字典。

    Raises:
        FileNotFoundError: 当输入数据文件缺失时抛出。
        ValueError: 当数据对齐或配置校验失败时抛出。
    """

    start_time = time.time()
    prepared_assets = prepare_experiment_assets(config)
    anchor_map = build_anchor_samples(
        user_bundles=prepared_assets.user_bundles,
        anchor_states_per_user=int(config["anchor_states_per_user"]),
    )

    lambda_values = [float(value) for value in config["lambda_grid"]]
    topk_list = [int(value) for value in config["topk_list"]]
    sample_frames: List[pd.DataFrame] = []
    anchor_status_list: List[str] = []

    progress_total = int(sum(len(anchor_list) for anchor_list in anchor_map.values()))
    progress_bar = tqdm(total=progress_total, desc="Evaluating OOD anchors")
    for raw_user_id, anchor_list in anchor_map.items():
        bundle = prepared_assets.user_bundles[raw_user_id]
        support_joint = build_support_joint_vectors(bundle)
        for anchor in anchor_list:
            frame, status = evaluate_single_anchor(
                anchor=anchor,
                bundle=bundle,
                prepared_assets=prepared_assets,
                support_joint=support_joint,
                lambda_values=lambda_values,
                config=config,
            )
            if len(frame) > 0:
                sample_frames.append(frame)
            anchor_status_list.append(status)
            progress_bar.update(1)
    progress_bar.close()

    sample_df = pd.concat(sample_frames, axis=0, ignore_index=True) if sample_frames else pd.DataFrame()
    output_dir = prepared_assets.core_assets.output_paths.root
    figures_dir = prepared_assets.core_assets.output_paths.figures

    sample_save_info = save_dataframe_with_fallback(
        dataframe=sample_df,
        output_dir=output_dir,
        base_filename="sample_metrics",
    )

    discrimination_metrics = compute_discrimination_metrics(
        sample_df=sample_df,
        alpha_list=[float(alpha) for alpha in config.get("highvar_alpha_list", [0.1, 0.2])],
    )
    lambda_df = compute_lambda_metrics(
        sample_df=sample_df,
        lambda_values=lambda_values,
        topk_list=topk_list,
    )

    label_counts = {
        "main_label_counts": sample_df["main_label"].value_counts().to_dict() if len(sample_df) > 0 else {},
        "quadrant_label_counts": sample_df["quadrant_label"].value_counts().to_dict() if len(sample_df) > 0 else {},
    }
    save_json(label_counts, os.path.join(output_dir, "label_counts.json"))

    data_summary = {
        "selected_user_count": int(len(prepared_assets.user_bundles)),
        "candidate_item_count": int(len(prepared_assets.candidate_raw_item_ids)),
        "anchor_total": int(progress_total),
        "anchor_status": summarize_anchor_statuses(anchor_status_list),
        "action_alignment_summary": prepared_assets.action_alignment_summary,
        "runtime_seconds": float(time.time() - start_time),
    }

    summary_payload = {
        "config": dict(config),
        "data_summary": data_summary,
        "discrimination_metrics": discrimination_metrics,
        "lambda_metrics": lambda_df.to_dict(orient="records"),
    }
    save_json(summary_payload, os.path.join(output_dir, "summary_metrics.json"))

    figure_paths = {
        "distance_vs_true_reward": os.path.join(figures_dir, "fig_distance_vs_true_reward.png"),
        "distance_vs_uncertainty": os.path.join(figures_dir, "fig_distance_vs_uncertainty.png"),
        "uncertainty_violin": os.path.join(figures_dir, "fig_uncertainty_violin_quadrants.png"),
        "uncertainty_bucket_mix": os.path.join(figures_dir, "fig_uncertainty_bucket_mix.png"),
        "highvar_composition": os.path.join(figures_dir, "fig_highvar_composition.png"),
        "lambda_response": os.path.join(figures_dir, "fig_lambda_response_curves.png"),
        "lambda_suppressed_mix": os.path.join(figures_dir, "fig_lambda_suppressed_mix.png"),
    }
    sns.set_theme(style="whitegrid")
    plot_distance_vs_true_reward(
        sample_df=sample_df,
        output_path=figure_paths["distance_vs_true_reward"],
        plot_sample_n=int(config.get("plot_sample_n", 120000)),
        seed=int(config["seed"]),
    )
    plot_distance_vs_uncertainty(
        sample_df=sample_df,
        output_path=figure_paths["distance_vs_uncertainty"],
        plot_sample_n=int(config.get("plot_sample_n", 120000)),
        seed=int(config["seed"]),
    )
    plot_uncertainty_violin(
        sample_df=sample_df,
        output_path=figure_paths["uncertainty_violin"],
        clip_quantile=float(config.get("uncertainty_violin_clip_quantile", 0.95)),
    )
    plot_uncertainty_bucket_mix(
        sample_df=sample_df,
        output_path=figure_paths["uncertainty_bucket_mix"],
        bucket_count=int(config.get("uncertainty_bucket_count", 8)),
    )
    plot_highvar_composition(
        sample_df=sample_df,
        output_path=figure_paths["highvar_composition"],
        alpha_value=float(config.get("highvar_plot_alpha", 0.1)),
    )
    plot_lambda_response(
        lambda_df=lambda_df,
        output_path=figure_paths["lambda_response"],
        lambda_plot_topk=int(config.get("lambda_plot_topk", 20)),
    )
    plot_lambda_suppressed_mix(
        lambda_df=lambda_df,
        output_path=figure_paths["lambda_suppressed_mix"],
        lambda_plot_topk=int(config.get("lambda_plot_topk", 20)),
    )

    report_text = generate_markdown_report(
        config=config,
        data_summary=data_summary,
        discrimination_metrics=discrimination_metrics,
        lambda_df=lambda_df,
        label_counts=label_counts,
        figure_paths=figure_paths,
        sample_save_info=sample_save_info,
        output_dir=output_dir,
    )
    report_path = os.path.join(output_dir, "ood_granularity_report.md")
    with open(report_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(report_text)

    manifest = build_manifest(
        config=config,
        extra={
            "sample_metrics_path": sample_save_info["path"],
            "summary_metrics_path": os.path.join(output_dir, "summary_metrics.json"),
            "label_counts_path": os.path.join(output_dir, "label_counts.json"),
            "report_path": report_path,
            "figure_paths": figure_paths,
        },
    )
    save_json(manifest, os.path.join(output_dir, "run_manifest.json"))

    LOGGER.info("OOD granularity experiment finished. Output dir: %s", output_dir)
    return {
        "output_dir": output_dir,
        "sample_metrics_path": sample_save_info["path"],
        "sample_metrics_format": sample_save_info["format"],
        "summary_metrics_path": os.path.join(output_dir, "summary_metrics.json"),
        "label_counts_path": os.path.join(output_dir, "label_counts.json"),
        "report_path": report_path,
        "figure_paths": figure_paths,
        "data_summary": data_summary,
    }
