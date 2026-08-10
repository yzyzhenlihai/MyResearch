#!/usr/bin/env python3
"""使用既有 DARLR checkpoint 评测固定高奖励用户与随机对照组。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import random
import sys
from functools import partial
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

# 纯评测不需要远程实验记录；必须在导入 DARLR 入口前关闭登录。
os.environ.setdefault("SWANLAB_MODE", "disabled")

from examples.advance.run_DARLR import (  # noqa: E402
    get_args_DARLR,
    setup_policy_model,
    validate_darlr_args,
)
from analysis.kuai_user_selection import dataframe_to_markdown  # noqa: E402
from policy_utils import (  # noqa: E402
    get_args_all,
    prepare_user_model,
    setup_state_tracker,
)
from src.core.collector.collector_set import CollectorSet  # noqa: E402
from src.core.envs.KuaiRec.KuaiEnv import KuaiEnv  # noqa: E402
from src.core.evaluation.evaluator import Evaluator_Feat  # noqa: E402
from src.core.util.data import get_env_args, get_true_env  # noqa: E402
from src.tianshou.tianshou.env import DummyVectorEnv  # noqa: E402


LOGGER = logging.getLogger(__name__)

DEFAULT_CONTROL_SIZE = 100
"""默认随机对照用户数，与 Top-100 选择组保持一致。"""

PAPER_SINGLE_STEP_REWARD = 1.26
"""DARLR 论文表 2 在 KuaiRec 上报告的 R_each。"""

FEATURE_DOMINATION_PATH = (
    REPOSITORY_ROOT
    / "data/KuaiRec/data_processed/feature_domination.pickle"
)
"""仓库既有的 KuaiRec 头部类别统计缓存。"""

FILE_HASH_CHUNK_BYTES = 1024 * 1024
"""流式计算 checkpoint 摘要时每次读取的字节数。"""


class FixedUserKuaiEnv(KuaiEnv):
    """每次 reset 都固定到指定矩阵行的 KuaiRec 环境。

    该类只改变测试用户采样，不改变 reward 矩阵、动作空间、离开条件或
    交互动态。它用于保证 Top-100 中每个用户恰好贡献一个测试 episode。

    Attributes:
        fixed_user_index (int): KuaiRec ``lbe_user`` 编码后的矩阵行号。
    """

    def __init__(
        self,
        fixed_user_index: int,
        allowed_item_indexes: Sequence[int] | None = None,
        **environment_kwargs: Any,
    ) -> None:
        """初始化固定用户环境。

        Args:
            fixed_user_index (int): 目标用户的矩阵行号，必须位于 reward
                矩阵行范围内。
            allowed_item_indexes (Sequence[int] | None): 可选静态 item 候选
                集合；启用随机初始 item 时，初始上下文从该集合之外采样，
                避免尚未产生 reward 的上下文占用一个 NX 去重候选。
            **environment_kwargs (Any): 原始 :class:`KuaiEnv` 初始化参数。

        Raises:
            ValueError: 当 ``fixed_user_index`` 超出矩阵范围时抛出。
        """

        self.fixed_user_index = int(fixed_user_index)
        self.allowed_item_indexes = self._normalize_allowed_item_indexes(
            allowed_item_indexes
        )
        super().__init__(**environment_kwargs)
        if not 0 <= self.fixed_user_index < self.mat.shape[0]:
            raise ValueError(
                "fixed_user_index is outside the reward matrix: "
                f"{self.fixed_user_index}."
            )

    @staticmethod
    def _normalize_allowed_item_indexes(
        allowed_item_indexes: Sequence[int] | None,
    ) -> tuple[int, ...] | None:
        """规范化可选 item 候选序列并拒绝歧义输入。

        Args:
            allowed_item_indexes (Sequence[int] | None): 原始候选序列。

        Returns:
            tuple[int, ...] | None: 保序整数元组，或表示不限制的 ``None``。

        Raises:
            ValueError: 当序列为空、重复或包含非整数值时抛出。
        """

        if allowed_item_indexes is None:
            return None
        indexes = np.asarray(allowed_item_indexes)
        if indexes.ndim != 1 or indexes.size == 0:
            raise ValueError("allowed_item_indexes must be a non-empty 1D sequence.")
        if not np.issubdtype(indexes.dtype, np.integer):
            raise ValueError("allowed_item_indexes must contain integers.")
        normalized = tuple(int(index) for index in indexes)
        if len(set(normalized)) != len(normalized):
            raise ValueError("allowed_item_indexes must not contain duplicates.")
        return normalized

    def reset(self):
        """重置轨迹并把当前用户覆盖为固定矩阵行。

        Returns:
            tuple[np.ndarray, dict[str, float]]: 固定用户的初始状态和累计
            reward 为零的环境信息。
        """

        _, info = super().reset()
        self.cur_user = self.fixed_user_index
        if self.allowed_item_indexes is not None:
            if min(self.allowed_item_indexes) < 0 or max(
                self.allowed_item_indexes
            ) >= self.mat.shape[1]:
                raise ValueError(
                    "allowed_item_indexes contains an index outside the reward matrix."
                )
            if self.random_init:
                allowed_item_set = set(self.allowed_item_indexes)
                context_candidates = [
                    item_index
                    for item_index in range(self.mat.shape[1])
                    if item_index not in allowed_item_set
                ]
                if context_candidates:
                    self.action = random.choice(context_candidates)
        return self.state, info


def build_evaluation_parser() -> argparse.ArgumentParser:
    """构造 checkpoint 用户子集评测的专用参数解析器。

    DARLR 网络结构参数继续由原训练入口的解析器读取，本解析器只处理文件、
    对照组和论文目标值等评测专用参数。

    Returns:
        argparse.ArgumentParser: 评测专用命令行解析器。
    """

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--user-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--paper-single-step-reward",
        type=float,
        default=PAPER_SINGLE_STEP_REWARD,
    )
    parser.add_argument(
        "--control-size",
        type=int,
        default=DEFAULT_CONTROL_SIZE,
    )
    parser.add_argument(
        "--limit-users",
        type=int,
        default=None,
        help="仅用于冒烟验证；默认评测用户表中的全部用户。",
    )
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="仅评测选择组，不构造固定随机对照组。",
    )
    return parser


def build_darlr_arguments() -> argparse.Namespace:
    """复用训练入口解析并合并 DARLR、环境和通用参数。

    Returns:
        argparse.Namespace: 可传给模型构造函数的完整参数。
    """

    arguments = get_args_all("onpolicy")
    environment_arguments = get_env_args(arguments)
    darlr_arguments = get_args_DARLR()
    arguments.__dict__.update(environment_arguments.__dict__)
    arguments.__dict__.update(darlr_arguments.__dict__)
    return arguments


def load_selected_users(user_table_path: Path) -> pd.DataFrame:
    """读取并校验第一阶段导出的 Top-K 用户表。

    Args:
        user_table_path (Path): 包含 ``user_id``、``matrix_user_index`` 和
            ``rank`` 列的 CSV 文件。

    Returns:
        pd.DataFrame: 按 rank 升序排列且用户不重复的选择表。

    Raises:
        FileNotFoundError: 当用户表不存在时抛出。
        ValueError: 当列缺失、表为空或用户重复时抛出。
    """

    if not user_table_path.is_file():
        raise FileNotFoundError(f"User table not found: {user_table_path}")
    user_table = pd.read_csv(user_table_path)
    required_columns = {"rank", "user_id", "matrix_user_index"}
    missing_columns = sorted(required_columns.difference(user_table.columns))
    if missing_columns:
        raise ValueError(f"User table is missing columns: {missing_columns}.")
    if user_table.empty:
        raise ValueError("User table must contain at least one row.")
    if user_table["user_id"].duplicated().any():
        raise ValueError("User table contains duplicate user_id values.")
    if user_table["matrix_user_index"].duplicated().any():
        raise ValueError("User table contains duplicate matrix_user_index values.")
    return user_table.sort_values("rank", kind="mergesort").reset_index(drop=True)


def seed_everything(seed: int) -> None:
    """统一设置 Python、NumPy 和 PyTorch 随机种子。

    Args:
        seed (int): 非负随机种子。

    Raises:
        ValueError: 当 ``seed`` 为负数时抛出。
    """

    if seed < 0:
        raise ValueError("seed must be non-negative.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_fixed_user_envs(
    user_indexes: Sequence[int],
    environment_kwargs: dict[str, Any],
    force_length: int,
    seed: int,
    allowed_item_indexes: Sequence[int] | None = None,
) -> dict[str, DummyVectorEnv]:
    """为一组固定用户构造三套论文兼容测试环境。

    Args:
        user_indexes (Sequence[int]): 互不重复的 KuaiRec 矩阵行号。
        environment_kwargs (dict[str, Any]): 原始 KuaiEnv 初始化参数。
        force_length (int): NX 强制长度，必须大于零。
        seed (int): 环境构造前设置的随机种子。
        allowed_item_indexes (Sequence[int] | None): 可选静态 item 候选集合。
            NX 去重协议的最大步数会自动收紧到候选数，避免候选耗尽。

    Returns:
        dict[str, DummyVectorEnv]: ``FB``、``NX_0`` 和 ``NX_force`` 三套
        向量环境；每套环境中每个用户恰好对应一个子环境。

    Raises:
        ValueError: 当用户列表为空、重复或强制长度不合法时抛出。
    """

    normalized_indexes = [int(index) for index in user_indexes]
    if not normalized_indexes:
        raise ValueError("user_indexes must not be empty.")
    if len(set(normalized_indexes)) != len(normalized_indexes):
        raise ValueError("user_indexes must be unique.")
    if force_length <= 0:
        raise ValueError("force_length must be positive.")
    normalized_allowed_items = FixedUserKuaiEnv._normalize_allowed_item_indexes(
        allowed_item_indexes
    )
    if (
        normalized_allowed_items is not None
        and len(normalized_allowed_items) < force_length
    ):
        raise ValueError(
            "allowed_item_indexes must contain at least force_length items."
        )

    def build_vector_environment(remove_recommended_items: bool) -> DummyVectorEnv:
        """构造一套固定用户向量环境。

        Args:
            remove_recommended_items (bool): 是否按 NX 协议禁止重复 item。

        Returns:
            DummyVectorEnv: 按输入顺序包含全部固定用户的环境。
        """

        protocol_kwargs = environment_kwargs
        if normalized_allowed_items is not None:
            protocol_kwargs = dict(environment_kwargs)
        if remove_recommended_items and normalized_allowed_items is not None:
            protocol_kwargs["max_turn"] = min(
                int(protocol_kwargs["max_turn"]),
                len(normalized_allowed_items),
            )

        if normalized_allowed_items is None:
            factories = [
                partial(
                    FixedUserKuaiEnv,
                    fixed_user_index=user_index,
                    **protocol_kwargs,
                )
                for user_index in normalized_indexes
            ]
        else:
            factories = [
                partial(
                    FixedUserKuaiEnv,
                    fixed_user_index=user_index,
                    allowed_item_indexes=normalized_allowed_items,
                    **protocol_kwargs,
                )
                for user_index in normalized_indexes
            ]
        vector_environment = DummyVectorEnv(factories)
        vector_environment.seed(seed)
        return vector_environment

    seed_everything(seed)
    return {
        "FB": build_vector_environment(remove_recommended_items=False),
        "NX_0": build_vector_environment(remove_recommended_items=True),
        f"NX_{force_length}": build_vector_environment(
            remove_recommended_items=True
        ),
    }


def load_prediction_matrices(ensemble_models: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """从现有 DeepFM ensemble 读取预测均值和静态方差矩阵。

    Args:
        ensemble_models (Any): ``prepare_user_model`` 返回的 ensemble，必须
            暴露预测矩阵与方差矩阵路径。

    Returns:
        tuple[np.ndarray, np.ndarray | None]: 预测 reward 矩阵和可选方差矩阵。

    Raises:
        ValueError: 当均值与方差矩阵形状不一致时抛出。
        OSError: 当必要预测矩阵无法读取时由底层接口抛出。
    """

    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as prediction_file:
        predicted_matrix = np.asarray(pickle.load(prediction_file))
    variance_matrix = None
    try:
        with open(ensemble_models.VAR_MAT_PATH, "rb") as variance_file:
            variance_matrix = np.asarray(pickle.load(variance_file))
    except FileNotFoundError:
        LOGGER.warning("未找到静态方差矩阵；仅当 checkpoint 结构允许时继续。")
    if variance_matrix is not None and variance_matrix.shape != predicted_matrix.shape:
        raise ValueError(
            "Variance matrix shape differs from prediction matrix: "
            f"{variance_matrix.shape} != {predicted_matrix.shape}."
        )
    return predicted_matrix, variance_matrix


def load_checkpoint(
    checkpoint_path: Path,
    rec_policy: Any,
    state_tracker: torch.nn.Module,
    device: torch.device | str,
) -> None:
    """严格恢复 DARLR policy、状态追踪器和动态 reward store。

    Args:
        checkpoint_path (Path): ``save_model_fn`` 生成的 DARLR checkpoint。
        rec_policy (Any): :class:`RecPolicy` 包装器。
        state_tracker (torch.nn.Module): 与 policy 共享的状态追踪器。
        device (torch.device | str): checkpoint 张量映射设备。

    Raises:
        FileNotFoundError: 当 checkpoint 不存在时抛出。
        KeyError: 当 checkpoint 缺少必要状态时抛出。
        RuntimeError: 当 checkpoint 与当前模型结构不一致时抛出。
    """

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "policy" not in checkpoint or "state_tracker" not in checkpoint:
        raise KeyError("Checkpoint must contain policy and state_tracker states.")
    rec_policy.policy.load_state_dict(checkpoint["policy"], strict=True)
    state_tracker.load_state_dict(checkpoint["state_tracker"], strict=True)

    extra_state = checkpoint.get("darlr_extra", {})
    dynamic_store_state = extra_state.get("dynamic_reward_store")
    if dynamic_store_state is not None:
        rec_policy.policy.darlr_load_extra_state(
            {"dynamic_reward_store": dynamic_store_state}
        )
    rec_policy.eval()
    LOGGER.info("已严格加载 DARLR checkpoint：%s", checkpoint_path)


def load_mcd_inputs(dataset: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取 MCD 计算所需的 item 特征和既有头部类别缓存。

    Args:
        dataset (Any): KuaiData 实例。

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]: 小矩阵 item 特征与类别占优字典。

    Raises:
        FileNotFoundError: 当类别占优缓存不存在时抛出。
        OSError: 当缓存或 item 特征无法读取时由底层接口抛出。
    """

    item_features = dataset.load_item_feat(only_small=True)
    if not FEATURE_DOMINATION_PATH.is_file():
        raise FileNotFoundError(
            f"Feature domination cache not found: {FEATURE_DOMINATION_PATH}"
        )
    with FEATURE_DOMINATION_PATH.open("rb") as domination_file:
        feature_domination = pickle.load(domination_file)
    return item_features, feature_domination


def get_result_key(protocol_name: str, metric_name: str) -> str:
    """返回 CollectorSet 中指定协议和指标的结果键。

    Args:
        protocol_name (str): ``FB``、``NX_0`` 或 ``NX_force``。
        metric_name (str): collector 返回的原始指标名。

    Returns:
        str: ``FB`` 无前缀、其他协议带协议前缀的结果键。
    """

    return metric_name if protocol_name == "FB" else f"{protocol_name}_{metric_name}"


def summarize_protocol(
    protocol_name: str,
    results: dict[str, Any],
    paper_single_step_reward: float,
) -> dict[str, Any]:
    """按论文定义汇总单个交互协议的 R_tra、R_each、Length 和 MCD。

    Args:
        protocol_name (str): CollectorSet 中的协议名。
        results (dict[str, Any]): 完成 collect 和 MCD callback 后的结果。
        paper_single_step_reward (float): 论文目标 R_each。

    Returns:
        dict[str, Any]: 可 JSON 序列化的协议指标。

    Raises:
        ValueError: 当 episode 数、步数或数组为空时抛出。
    """

    rewards = np.asarray(results[get_result_key(protocol_name, "rews")], dtype=float)
    lengths = np.asarray(results[get_result_key(protocol_name, "lens")], dtype=float)
    if rewards.size == 0 or lengths.size == 0 or lengths.sum() <= 0:
        raise ValueError(f"Protocol {protocol_name} returned no valid episodes.")
    episode_count = int(results[get_result_key(protocol_name, "n/ep")])
    step_count = int(results[get_result_key(protocol_name, "n/st")])
    if episode_count != len(rewards) or step_count != int(lengths.sum()):
        raise ValueError(
            f"Protocol {protocol_name} result counts are inconsistent."
        )

    single_step_reward = float(rewards.sum() / lengths.sum())
    mcd_key = get_result_key(protocol_name, "ifeat_feat")
    return {
        "protocol": protocol_name,
        "episode_count": episode_count,
        "step_count": step_count,
        "R_tra": float(rewards.mean()),
        "R_tra_episode_std": float(rewards.std()),
        "R_each": single_step_reward,
        "Length": float(lengths.mean()),
        "Length_episode_std": float(lengths.std()),
        "MCD": float(results[mcd_key]),
        "paper_R_each": float(paper_single_step_reward),
        "R_each_gap_to_paper": single_step_reward - paper_single_step_reward,
        "reaches_paper_R_each": bool(single_step_reward >= paper_single_step_reward),
    }


def extract_per_user_metrics(
    group_name: str,
    protocol_name: str,
    collector_set: CollectorSet,
    results: dict[str, Any],
    user_encoder: Any,
    expected_user_indexes: Sequence[int],
) -> pd.DataFrame:
    """从 replay buffer 恢复每个固定用户的 episode 指标并校验覆盖。

    Args:
        group_name (str): ``selected_top_reward_rate`` 或 ``random_control``。
        protocol_name (str): 当前交互协议名。
        collector_set (CollectorSet): 已完成评测的 collector 集合。
        results (dict[str, Any]): collect 结果。
        user_encoder (Any): KuaiRec 原始 ID 与矩阵行号之间的 LabelEncoder。
        expected_user_indexes (Sequence[int]): 预期恰好出现一次的矩阵行号。

    Returns:
        pd.DataFrame: 每个用户的 episode reward、长度和 episode 内单步均值。

    Raises:
        ValueError: 当实际用户覆盖不是预期集合的一一对应时抛出。
    """

    collector = collector_set.collector_dict[protocol_name]
    start_indexes = np.asarray(
        results[get_result_key(protocol_name, "idxs")],
        dtype=np.int64,
    )
    encoded_user_ids = np.asarray(
        collector.buffer.obs[start_indexes][:, 0],
        dtype=np.int64,
    )
    expected_indexes = np.asarray(expected_user_indexes, dtype=np.int64)
    if len(encoded_user_ids) != len(expected_indexes) or not np.array_equal(
        np.sort(encoded_user_ids),
        np.sort(expected_indexes),
    ):
        raise ValueError(
            f"Protocol {protocol_name} did not evaluate every fixed user exactly once."
        )
    episode_rewards = np.asarray(
        results[get_result_key(protocol_name, "rews")],
        dtype=float,
    )
    episode_lengths = np.asarray(
        results[get_result_key(protocol_name, "lens")],
        dtype=np.int64,
    )
    raw_user_ids = user_encoder.inverse_transform(encoded_user_ids)
    return pd.DataFrame(
        {
            "group": group_name,
            "protocol": protocol_name,
            "user_id": raw_user_ids.astype(np.int64),
            "matrix_user_index": encoded_user_ids,
            "episode_reward": episode_rewards,
            "episode_length": episode_lengths,
            "episode_single_step_reward": episode_rewards / episode_lengths,
        }
    )


def evaluate_group(
    group_name: str,
    user_indexes: Sequence[int],
    collector_set: CollectorSet,
    rec_policy: Any,
    item_features: pd.DataFrame,
    feature_domination: dict[str, Any],
    item_encoder: Any,
    user_encoder: Any,
    top_rate: float,
    paper_single_step_reward: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """运行一个固定用户组的完整交互评测并返回论文指标。

    Args:
        group_name (str): 结果中的用户组标签。
        user_indexes (Sequence[int]): 每个用户对应的矩阵行号。
        collector_set (CollectorSet): 已绑定三套固定用户环境的 collector。
        rec_policy (Any): 已恢复 checkpoint 的推荐 policy。
        item_features (pd.DataFrame): MCD 计算需要的 item 特征。
        feature_domination (dict[str, Any]): MCD 头部类别定义。
        item_encoder (Any): item 原始 ID 与矩阵列号的 LabelEncoder。
        user_encoder (Any): user 原始 ID 与矩阵行号的 LabelEncoder。
        top_rate (float): MCD 头部类别累计比例。
        paper_single_step_reward (float): 论文报告的目标 R_each。

    Returns:
        tuple[list[dict[str, Any]], pd.DataFrame]: 三种协议聚合指标和逐用户指标。

    Raises:
        ValueError: 当 ``top_rate`` 越界或固定用户覆盖异常时抛出。
    """

    if not 0 < top_rate <= 1:
        raise ValueError("top_rate must be in (0, 1].")
    rec_policy.eval()
    results = collector_set.collect(n_episode=len(user_indexes))
    mcd_evaluator = Evaluator_Feat(
        collector_set,
        item_features,
        need_transform=True,
        item_feat_domination=feature_domination,
        lbe_item=item_encoder,
        top_rate=top_rate,
        draw_bar=False,
    )
    mcd_evaluator.on_epoch_end(epoch=0, results=results)

    protocol_metrics = []
    user_metric_frames = []
    for protocol_name in collector_set.collector_dict:
        metrics = summarize_protocol(
            protocol_name,
            results,
            paper_single_step_reward,
        )
        metrics["group"] = group_name
        protocol_metrics.append(metrics)
        user_metric_frames.append(
            extract_per_user_metrics(
                group_name,
                protocol_name,
                collector_set,
                results,
                user_encoder,
                user_indexes,
            )
        )
    LOGGER.info("完成用户组 %s 的 %d 个固定用户评测。", group_name, len(user_indexes))
    return protocol_metrics, pd.concat(user_metric_frames, ignore_index=True)


def choose_control_users(
    user_count: int,
    selected_user_indexes: Sequence[int],
    control_size: int,
    seed: int,
) -> np.ndarray:
    """从选择组之外均匀无放回抽取固定随机对照用户。

    Args:
        user_count (int): KuaiRec 测试矩阵总用户数。
        selected_user_indexes (Sequence[int]): 需要排除的选择组矩阵行号。
        control_size (int): 对照用户数。
        seed (int): NumPy 随机种子。

    Returns:
        np.ndarray: 长度为 ``control_size`` 的矩阵行号。

    Raises:
        ValueError: 当用户总数、对照规模或候选余量不合法时抛出。
    """

    if user_count <= 0 or control_size <= 0:
        raise ValueError("user_count and control_size must be positive.")
    selected_indexes = np.asarray(selected_user_indexes, dtype=np.int64)
    candidate_indexes = np.setdiff1d(
        np.arange(user_count, dtype=np.int64),
        selected_indexes,
    )
    if control_size > len(candidate_indexes):
        raise ValueError(
            f"control_size={control_size} exceeds {len(candidate_indexes)} candidates."
        )
    random_generator = np.random.default_rng(seed)
    return np.sort(
        random_generator.choice(candidate_indexes, control_size, replace=False)
    )


def compute_file_sha256(file_path: Path) -> str:
    """流式计算文件 SHA-256，记录 checkpoint 身份。

    Args:
        file_path (Path): 需要计算摘要的文件。

    Returns:
        str: 64 位十六进制 SHA-256 字符串。

    Raises:
        OSError: 当文件无法读取时由底层接口抛出。
    """

    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        while True:
            chunk = input_file.read(FILE_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_evaluation_artifacts(
    output_dir: Path,
    metrics: list[dict[str, Any]],
    per_user_metrics: pd.DataFrame,
    selected_user_table: pd.DataFrame,
    control_user_table: pd.DataFrame | None,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    """写出交互评测表、逐用户结果、清单和中文摘要。

    Args:
        output_dir (Path): 评测输出目录。
        metrics (list[dict[str, Any]]): 用户组与协议聚合指标。
        per_user_metrics (pd.DataFrame): 逐用户 episode 指标。
        selected_user_table (pd.DataFrame): 第一阶段 Top-K 用户表。
        control_user_table (pd.DataFrame | None): 可选随机对照用户表。
        manifest (dict[str, Any]): checkpoint、设备和参数清单。

    Returns:
        dict[str, Path]: 所有核心输出文件路径。

    Raises:
        OSError: 当输出目录或文件无法写入时由底层接口抛出。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation_metrics.csv"
    per_user_path = output_dir / "per_user_episode_metrics.csv"
    selected_path = output_dir / "selected_users.csv"
    control_path = output_dir / "control_users.csv"
    summary_json_path = output_dir / "evaluation_summary.json"
    manifest_path = output_dir / "run_manifest.json"
    markdown_path = output_dir / "interaction_evaluation_summary.md"

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(metrics_path, index=False)
    per_user_metrics.to_csv(per_user_path, index=False)
    selected_user_table.to_csv(selected_path, index=False)
    if control_user_table is not None:
        control_user_table.to_csv(control_path, index=False)
    with summary_json_path.open("w", encoding="utf-8") as summary_file:
        json.dump(metrics, summary_file, ensure_ascii=False, indent=2)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

    display_frame = metrics_frame[
        [
            "group",
            "protocol",
            "R_tra",
            "R_each",
            "Length",
            "MCD",
            "R_each_gap_to_paper",
            "reaches_paper_R_each",
        ]
    ].copy()
    for column in ("R_tra", "R_each", "Length", "MCD", "R_each_gap_to_paper"):
        display_frame[column] = display_frame[column].map(lambda value: f"{value:.6f}")

    selected_fb = metrics_frame.loc[
        (metrics_frame["group"] == "selected_top_reward_rate")
        & (metrics_frame["protocol"] == "FB")
    ].iloc[0]
    conclusion = (
        "达到"
        if bool(selected_fb["reaches_paper_R_each"])
        else "未达到"
    )
    markdown = "\n".join(
        [
            "# DARLR 固定高奖励用户交互评测",
            "",
            "## 口径",
            "",
            "- 选择组：第一阶段按 `reward >= 1.26` 占比排序的 Top 用户。",
            "- 每个用户固定对应一个环境，每种协议恰好评测一个 episode。",
            "- `R_tra`、`R_each`、`Length` 和 `MCD` 沿用仓库及论文定义。",
            "- `R_each = 全部 episode reward 总和 / 全部交互步数`。",
            "- 这是单个既有 checkpoint 的受控评测，不等同于论文多随机种子结果。",
            "",
            "## 结果",
            "",
            dataframe_to_markdown(display_frame),
            "",
            "## 针对假设的结论",
            "",
            f"选择组在 FB 口径下的 `R_each={selected_fb['R_each']:.6f}`，"
            f"{conclusion}论文目标 `1.26`。",
            "该结果只能说明高 reward 用户筛选是否足以解释当前 checkpoint 的差距；",
            "不能据此证明原作者实际筛选过测试用户。",
            "",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    paths = {
        "metrics_csv": metrics_path,
        "per_user_csv": per_user_path,
        "selected_users_csv": selected_path,
        "summary_json": summary_json_path,
        "manifest_json": manifest_path,
        "summary_markdown": markdown_path,
    }
    if control_user_table is not None:
        paths["control_users_csv"] = control_path
    return paths


def main() -> None:
    """构造模型、恢复 checkpoint 并完成选择组与对照组交互评测。

    Returns:
        None: 聚合结果与逐用户结果写入 ``--output-dir``。

    Raises:
        ValueError: 当数据、用户、配置或评测覆盖不合法时抛出。
        OSError: 当必要数据、checkpoint 或输出不可访问时由底层接口抛出。
        RuntimeError: 当 checkpoint 与指定 DARLR 结构不匹配时抛出。
    """

    evaluation_parser = build_evaluation_parser()
    evaluation_args, _ = evaluation_parser.parse_known_args()
    darlr_args = build_darlr_arguments()
    validate_darlr_args(darlr_args)
    if darlr_args.env != "KuaiEnv-v0":
        raise ValueError("This evaluator currently supports only KuaiEnv-v0.")
    if evaluation_args.control_size <= 0:
        raise ValueError("--control-size must be positive.")
    if evaluation_args.limit_users is not None and evaluation_args.limit_users <= 0:
        raise ValueError("--limit-users must be positive when provided.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    selected_user_table = load_selected_users(evaluation_args.user_table)
    if evaluation_args.limit_users is not None:
        selected_user_table = selected_user_table.head(
            evaluation_args.limit_users
        ).copy()
    selected_user_indexes = selected_user_table[
        "matrix_user_index"
    ].to_numpy(dtype=np.int64)
    darlr_args.test_num = len(selected_user_indexes)
    seed_everything(darlr_args.seed)

    LOGGER.info("加载 KuaiRec 环境、DeepFM ensemble 与 checkpoint 架构。")
    ensemble_models = prepare_user_model(darlr_args)
    true_environment, dataset, environment_kwargs = get_true_env(darlr_args)
    expected_raw_ids = true_environment.lbe_user.inverse_transform(
        selected_user_indexes
    )
    if not np.array_equal(
        expected_raw_ids.astype(np.int64),
        selected_user_table["user_id"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Selected raw user IDs do not match matrix_user_index values.")

    selected_envs = build_fixed_user_envs(
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
        test_envs_dict=selected_envs,
    )
    rec_policy, _, selected_collectors, _ = setup_policy_model(
        darlr_args,
        state_tracker,
        train_envs=None,
        test_envs_dict=selected_envs,
        predicted_mat=predicted_matrix,
        maxvar_mat=variance_matrix,
        build_train_collector=False,
    )
    load_checkpoint(
        evaluation_args.checkpoint,
        rec_policy,
        state_tracker,
        darlr_args.device,
    )
    item_features, feature_domination = load_mcd_inputs(dataset)

    all_metrics: list[dict[str, Any]] = []
    per_user_frames: list[pd.DataFrame] = []
    # 模型构造会消耗 PyTorch 随机数；评测前复位以固定 selector 采样序列。
    seed_everything(darlr_args.seed)
    selected_metrics, selected_per_user = evaluate_group(
        "selected_top_reward_rate",
        selected_user_indexes,
        selected_collectors,
        rec_policy,
        item_features,
        feature_domination,
        true_environment.lbe_item,
        true_environment.lbe_user,
        darlr_args.top_rate,
        evaluation_args.paper_single_step_reward,
    )
    all_metrics.extend(selected_metrics)
    per_user_frames.append(selected_per_user)

    control_user_table = None
    if not evaluation_args.skip_control:
        control_user_indexes = choose_control_users(
            true_environment.mat.shape[0],
            selected_user_indexes,
            evaluation_args.control_size,
            darlr_args.seed,
        )
        control_raw_ids = true_environment.lbe_user.inverse_transform(
            control_user_indexes
        ).astype(np.int64)
        control_user_table = pd.DataFrame(
            {
                "control_rank": np.arange(1, len(control_user_indexes) + 1),
                "user_id": control_raw_ids,
                "matrix_user_index": control_user_indexes,
                "sampling_seed": darlr_args.seed,
            }
        )
        control_envs = build_fixed_user_envs(
            control_user_indexes,
            environment_kwargs,
            darlr_args.force_length,
            darlr_args.seed,
        )
        control_collectors = CollectorSet(
            rec_policy,
            control_envs,
            darlr_args.buffer_size,
            len(control_user_indexes),
            exploration_noise=darlr_args.exploration_noise,
            force_length=darlr_args.force_length,
        )
        # 对照组复用同一随机序列起点，减少 selector 采样噪声带来的组间偏差。
        seed_everything(darlr_args.seed)
        control_metrics, control_per_user = evaluate_group(
            "random_control",
            control_user_indexes,
            control_collectors,
            rec_policy,
            item_features,
            feature_domination,
            true_environment.lbe_item,
            true_environment.lbe_user,
            darlr_args.top_rate,
            evaluation_args.paper_single_step_reward,
        )
        all_metrics.extend(control_metrics)
        per_user_frames.append(control_per_user)

    device_name = "CPU"
    if torch.cuda.is_available() and str(darlr_args.device).startswith("cuda"):
        device_name = torch.cuda.get_device_name(darlr_args.device)
    manifest = {
        "checkpoint": str(evaluation_args.checkpoint),
        "checkpoint_sha256": compute_file_sha256(evaluation_args.checkpoint),
        "user_table": str(evaluation_args.user_table),
        "selection_definition": "reward >= 1.26 rate descending",
        "selected_user_count": int(len(selected_user_indexes)),
        "control_user_count": (
            0 if control_user_table is None else int(len(control_user_table))
        ),
        "paper_single_step_reward": float(
            evaluation_args.paper_single_step_reward
        ),
        "seed": int(darlr_args.seed),
        "device": str(darlr_args.device),
        "device_name": device_name,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "environment": {
            "name": darlr_args.env,
            "max_turn": int(darlr_args.max_turn),
            "num_leave_compute": int(darlr_args.num_leave_compute),
            "leave_threshold": float(darlr_args.leave_threshold),
            "force_length": int(darlr_args.force_length),
            "random_init": bool(darlr_args.random_init),
        },
        "model": {
            "which_tracker": darlr_args.which_tracker,
            "window_size": int(darlr_args.window_size),
            "hidden_sizes": [int(size) for size in darlr_args.hidden_sizes],
            "read_message": darlr_args.read_message,
            "selector_k": int(darlr_args.selector_k),
            "selector_candidate_size": int(darlr_args.selector_candidate_size),
            "selector_candidate_mode": darlr_args.selector_candidate_mode,
            "selector_policy_mode": darlr_args.selector_policy_mode,
            "selector_pref_dim": int(darlr_args.selector_pref_dim),
            "selector_num_heads": int(darlr_args.selector_num_heads),
            "selector_num_layers": int(darlr_args.selector_num_layers),
            "selector_dropout_rate": float(
                darlr_args.selector_dropout_rate
            ),
            "selector_lambda_s": float(darlr_args.selector_lambda_s),
            "selector_lambda_d": float(darlr_args.selector_lambda_d),
            "selector_reward_mode": darlr_args.selector_reward_mode,
            "dynamic_reward_mode": darlr_args.dynamic_reward_mode,
            "dynamic_uncertainty_mode": darlr_args.dynamic_uncertainty_mode,
            "lambda_uncertainty": float(darlr_args.lambda_uncertainty),
            "lambda_entropy": float(darlr_args.lambda_entropy),
        },
    }
    paths = write_evaluation_artifacts(
        evaluation_args.output_dir,
        all_metrics,
        pd.concat(per_user_frames, ignore_index=True),
        selected_user_table,
        control_user_table,
        manifest,
    )
    LOGGER.info("聚合指标：%s", paths["metrics_csv"])
    print(pd.DataFrame(all_metrics).to_string(index=False))


if __name__ == "__main__":
    main()
