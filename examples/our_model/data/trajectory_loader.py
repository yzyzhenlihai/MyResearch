"""离线推荐轨迹加载器。"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

REQUIRED_TRAJECTORY_KEYS = (
    "actions",
    "next_observations",
    "observations",
    "rewards",
    "terminals",
    "user_id",
)
"""轨迹 pkl 中必须存在的字段集合。"""


@dataclass
class TrajectoryBundle:
    """封装离线推荐轨迹及其基础统计信息。

    Attributes:
        trajectories (List[Dict[str, Any]]): 从 pkl 文件读取并校验后的用户轨迹列表。
        state_dim (int): 状态向量维度。
        action_dim (int): action embedding 维度。
        num_users (int): 轨迹用户数量。
        num_transitions (int): transition 总数。
        dataset_path (str): 原始 pkl 文件路径。
    """

    trajectories: List[Dict[str, Any]]
    state_dim: int
    action_dim: int
    num_users: int
    num_transitions: int
    dataset_path: str
    initial_observations_by_user: Dict[int, np.ndarray] = field(default_factory=dict)


class TrajectoryLoader:
    """读取并校验 DORL-MAC 使用的离线推荐轨迹。

    该类只负责读取已有 pkl，不会重写数据文件。它会检查字段完整性、
    shape 一致性和维度一致性，为后续 action chunk 派生提供稳定输入。
    """

    def __init__(self, dataset_path: str) -> None:
        """初始化轨迹加载器。

        Args:
            dataset_path (str): `DM_KuaiEnv-v0_small_data.pkl` 或同格式数据路径。

        Raises:
            FileNotFoundError: 当数据文件不存在时抛出。
        """

        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Trajectory dataset does not exist: {self.dataset_path}")

    def load(
        self, max_trajectories: Optional[int] = None,
        require_observations: bool = True,
    ) -> TrajectoryBundle:
        """读取轨迹 pkl 并返回校验后的数据包。

        Args:
            max_trajectories (Optional[int]): 仅保留前若干条用户轨迹，主要用于
                smoke test；为 `None` 时读取全部轨迹。
            require_observations (bool): 是否要求并校验离线数据中的
                `observations` 与 `next_observations`。MAC_origin 训练必须为 True。

        Returns:
            TrajectoryBundle: 包含轨迹列表和基础统计信息的数据包。

        Raises:
            ValueError: 当顶层对象或轨迹字段不符合预期时抛出。
        """

        LOGGER.info("加载离线轨迹数据：%s", self.dataset_path)
        with self.dataset_path.open("rb") as file_obj:
            trajectories = pickle.load(file_obj)

        if not isinstance(trajectories, list):
            raise ValueError(
                f"Trajectory dataset must be list[dict], got {type(trajectories).__name__}."
            )
        all_initial_observations = {
            int(trajectory["user_id"]): np.asarray(
                trajectory["observations"], dtype=np.float32,
            )[0].copy()
            for trajectory in trajectories
            if "user_id" in trajectory
            and "observations" in trajectory
            and len(trajectory["observations"]) > 0
        } if require_observations else {}

        if max_trajectories is not None:
            if max_trajectories <= 0:
                raise ValueError("max_trajectories must be positive when provided.")
            trajectories = trajectories[:max_trajectories]
        if not trajectories:
            raise ValueError("Trajectory dataset is empty.")

        state_dim: Optional[int] = None
        action_dim: Optional[int] = None
        num_transitions = 0

        for trajectory_index, trajectory in enumerate(trajectories):
            self._validate_trajectory_keys(trajectory_index, trajectory, require_observations)
            actions = np.asarray(trajectory["actions"])
            if require_observations:
                observations = np.asarray(trajectory["observations"])
                next_observations = np.asarray(trajectory["next_observations"])
            else:
                # 零列矩阵仅用于共享长度校验；不从旧 tracker 状态读取任何值。
                observations = np.empty((len(actions), 0), dtype=np.float32)
                next_observations = observations
            rewards = np.asarray(trajectory["rewards"])
            terminals = np.asarray(trajectory["terminals"])

            self._validate_shapes(
                trajectory_index,
                observations,
                actions,
                next_observations,
                rewards,
                terminals,
            )

            if state_dim is None:
                state_dim = int(observations.shape[1])
                action_dim = int(actions.shape[1])
            elif state_dim != int(observations.shape[1]) or action_dim != int(actions.shape[1]):
                raise ValueError(
                    "Trajectory dimensions are inconsistent: "
                    f"expected state/action=({state_dim}, {action_dim}), got "
                    f"({observations.shape[1]}, {actions.shape[1]}) at #{trajectory_index}."
                )
            num_transitions += int(observations.shape[0])

        assert state_dim is not None
        assert action_dim is not None
        LOGGER.info(
            "轨迹加载完成：users=%s, transitions=%s, state_dim=%s, action_dim=%s",
            len(trajectories),
            num_transitions,
            state_dim,
            action_dim,
        )
        return TrajectoryBundle(
            trajectories=trajectories,
            state_dim=state_dim,
            action_dim=action_dim,
            num_users=len(trajectories),
            num_transitions=num_transitions,
            dataset_path=str(self.dataset_path),
            initial_observations_by_user=all_initial_observations,
        )

    @staticmethod
    def _validate_trajectory_keys(
        trajectory_index: int, trajectory: Dict[str, Any],
        require_observations: bool = True,
    ) -> None:
        """检查单条轨迹字段完整性。

        Args:
            trajectory_index (int): 当前轨迹在列表中的下标。
            trajectory (Dict[str, Any]): 单条用户轨迹。
            require_observations (bool): 是否要求旧 observations 字段。

        Returns:
            None.

        Raises:
            KeyError: 当轨迹缺少必需字段时抛出。
        """

        required = set(REQUIRED_TRAJECTORY_KEYS)
        if not require_observations:
            required -= {"observations", "next_observations"}
        missing_keys = sorted(required.difference(trajectory.keys()))
        if missing_keys:
            raise KeyError(f"Trajectory #{trajectory_index} missing keys: {missing_keys}.")

    @staticmethod
    def _validate_shapes(
        trajectory_index: int,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
    ) -> None:
        """检查单条轨迹各字段 shape 是否一致。

        Args:
            trajectory_index (int): 当前轨迹下标。
            observations (np.ndarray): 状态矩阵，形状为 `(T, state_dim)`。
            actions (np.ndarray): 动作矩阵，形状为 `(T, action_dim)`。
            next_observations (np.ndarray): 下一状态矩阵，形状为 `(T, state_dim)`。
            rewards (np.ndarray): 奖励数组，长度为 `T`。
            terminals (np.ndarray): 终止标记数组，长度为 `T`。

        Returns:
            None.

        Raises:
            ValueError: 当维度或长度不合法时抛出。
        """

        if observations.ndim != 2:
            raise ValueError(f"Trajectory #{trajectory_index} observations must be 2D.")
        if actions.ndim != 2:
            raise ValueError(f"Trajectory #{trajectory_index} actions must be 2D.")
        if next_observations.ndim != 2:
            raise ValueError(f"Trajectory #{trajectory_index} next_observations must be 2D.")
        trajectory_length = int(observations.shape[0])
        if trajectory_length <= 0:
            raise ValueError(f"Trajectory #{trajectory_index} is empty.")
        if actions.shape[0] != trajectory_length:
            raise ValueError(f"Trajectory #{trajectory_index} action length mismatch.")
        if next_observations.shape != observations.shape:
            raise ValueError(f"Trajectory #{trajectory_index} next_observations shape mismatch.")
        if rewards.shape[0] != trajectory_length:
            raise ValueError(f"Trajectory #{trajectory_index} reward length mismatch.")
        if terminals.shape[0] != trajectory_length:
            raise ValueError(f"Trajectory #{trajectory_index} terminal length mismatch.")
