"""Action chunk 数据集派生逻辑。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from examples.our_model.data.trajectory_loader import TrajectoryBundle

LOGGER = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZE = 3
"""StateTrackerAvg 默认历史窗口长度。"""

DEFAULT_LEAVE_HISTORY_SIZE = 9
"""KuaiEnv-v0 默认用于退出判断的最近动作数量。"""

INVALID_ITEM_ID = -1
"""padding 或 dummy 历史位置使用的 item id。"""

ACTION_MATCH_ATOL = 1e-6
"""action embedding 精确反查失败时允许的数值误差上限。"""


class ActionChunkDataset(Dataset):
    """从离线用户轨迹在线派生 action chunk 样本。

    该数据集不会把所有 chunk 预先展开成巨型数组，而是仅保存 `(trajectory,
    start_index)` 索引。每次取样时构造 flatten chunk、折扣累计 reward、
    K 步 next state、StateTrackerAvg 历史向量、推荐 mask 和 KuaiEnv leave
    历史。
    """

    def __init__(
        self,
        bundle: TrajectoryBundle,
        item_embeddings: torch.Tensor,
        chunk_size: int,
        gamma: float,
        window_size: int = DEFAULT_WINDOW_SIZE,
        leave_history_size: int = DEFAULT_LEAVE_HISTORY_SIZE,
        max_chunks: Optional[int] = None,
    ) -> None:
        """初始化 action chunk 数据集。

        Args:
            bundle (TrajectoryBundle): 已校验的离线轨迹数据包。
            item_embeddings (torch.Tensor): item embedding 表，形状为 `(num_items, action_dim)`。
            chunk_size (int): 每个 chunk 覆盖的环境步数，必须大于 0。
            gamma (float): 折扣因子，取值范围为 `[0, 1]`。
            window_size (int): StateTrackerAvg 历史窗口长度。
            leave_history_size (int): KuaiEnv leave 判断保留的最近 item 数量。
            max_chunks (Optional[int]): 最多暴露的 chunk 样本数，主要用于 smoke test。

        Raises:
            ValueError: 当参数或维度非法时抛出。
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1].")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if leave_history_size <= 0:
            raise ValueError("leave_history_size must be positive.")
        if item_embeddings.ndim != 2:
            raise ValueError("item_embeddings must be a 2D tensor.")
        if int(item_embeddings.shape[1]) != bundle.action_dim:
            raise ValueError(
                f"item embedding dim mismatch: expected {bundle.action_dim}, "
                f"got {item_embeddings.shape[1]}."
            )

        self.bundle = bundle
        self.item_embeddings = item_embeddings.detach().cpu().float().contiguous()
        self.item_embeddings_np = self.item_embeddings.numpy().astype(np.float32, copy=False)
        self.num_items = int(self.item_embeddings.shape[0])
        self.chunk_size = int(chunk_size)
        self.gamma = float(gamma)
        self.window_size = int(window_size)
        self.leave_history_size = int(leave_history_size)
        self.discount = np.power(self.gamma, np.arange(self.chunk_size, dtype=np.float32))
        self._embedding_lookup = self._build_embedding_lookup(self.item_embeddings_np)
        self._trajectory_item_cache: Dict[int, np.ndarray] = {}
        self.sample_index = self._build_sample_index(max_chunks=max_chunks)

        if not self.sample_index:
            raise ValueError("No valid action chunk samples were found.")
        LOGGER.info(
            "ActionChunkDataset 构造完成：chunks=%s, chunk_size=%s, gamma=%s",
            len(self.sample_index),
            self.chunk_size,
            self.gamma,
        )

    def __len__(self) -> int:
        """返回可采样 chunk 数量。

        Returns:
            int: 样本数量。
        """

        return len(self.sample_index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """按索引构造单条 action chunk 样本。

        Args:
            index (int): 数据集样本下标。

        Returns:
            Dict[str, torch.Tensor]: 包含 state、chunk action、K 步 next state、
            reward、history 和 mask 的张量字典。

        Raises:
            IndexError: 当索引越界时抛出。
        """

        if index < 0 or index >= len(self.sample_index):
            raise IndexError(f"Action chunk index out of range: {index}.")

        trajectory_index, start_index = self.sample_index[index]
        trajectory = self.bundle.trajectories[trajectory_index]
        observations = np.asarray(trajectory["observations"], dtype=np.float32)
        next_observations = np.asarray(trajectory["next_observations"], dtype=np.float32)
        actions = np.asarray(trajectory["actions"], dtype=np.float32)
        rewards = np.asarray(trajectory["rewards"], dtype=np.float32)
        terminals = np.asarray(trajectory["terminals"], dtype=np.bool_)
        item_ids = self._get_item_ids(trajectory_index, actions)

        end_index = start_index + self.chunk_size
        action_chunk = actions[start_index:end_index]
        reward_steps = rewards[start_index:end_index]
        item_chunk = item_ids[start_index:end_index]
        done_chunk = bool(end_index >= len(actions) or terminals[start_index:end_index].any())
        if end_index < len(actions):
            target_next_observation = observations[end_index]
        else:
            target_next_observation = next_observations[end_index - 1]

        history_vectors, history_valid = self._build_history_vectors(
            observations=observations,
            actions=actions,
            rewards=rewards,
            start_index=start_index,
        )
        leave_history = self._build_padded_item_history(
            item_ids=item_ids,
            start_index=start_index,
            history_size=self.leave_history_size,
        )
        history_item_ids = self._build_padded_item_history(
            item_ids=item_ids,
            start_index=start_index,
            history_size=self.window_size,
        )
        recommended_mask = np.zeros(self.num_items, dtype=np.bool_)
        if start_index > 0:
            recommended_mask[item_ids[:start_index]] = True

        reward_chunk = float(np.sum(reward_steps * self.discount))
        sample = {
            "observations": observations[start_index],
            "actions": action_chunk.reshape(-1),
            "next_observations": target_next_observation,
            "rewards": np.asarray([reward_chunk], dtype=np.float32),
            "terminals": np.asarray([done_chunk], dtype=np.float32),
            "history_vectors": history_vectors,
            "history_valid": history_valid,
            "history_item_ids": history_item_ids,
            "leave_history_item_ids": leave_history,
            "recommended_mask": recommended_mask,
            "chunk_step_rewards": reward_steps,
            "chunk_item_ids": item_chunk.astype(np.int64),
            "user_id": np.asarray([int(trajectory["user_id"])], dtype=np.int64),
            "start_index": np.asarray([start_index], dtype=np.int64),
            "env_step": np.asarray([start_index], dtype=np.int64),
        }
        return {key: torch.as_tensor(value) for key, value in sample.items()}

    def validate_action_lookup(self, max_trajectories: Optional[int] = None) -> None:
        """验证离线 action embedding 是否能精确反查到 item id。

        Args:
            max_trajectories (Optional[int]): 最多验证的轨迹数量；`None` 表示全部。

        Returns:
            None.

        Raises:
            ValueError: 当任意 action embedding 无法精确匹配 item 表时抛出。
        """

        total_checked = 0
        trajectories = self.bundle.trajectories
        if max_trajectories is not None:
            trajectories = trajectories[:max_trajectories]
        for trajectory_index, trajectory in enumerate(trajectories):
            actions = np.asarray(trajectory["actions"], dtype=np.float32)
            self._get_item_ids(trajectory_index, actions)
            total_checked += int(actions.shape[0])
        LOGGER.info("action embedding 精确反查验证通过：transitions=%s", total_checked)

    def _build_sample_index(self, max_chunks: Optional[int]) -> List[Tuple[int, int]]:
        """预计算 `(trajectory_index, start_index)` 样本索引。

        Args:
            max_chunks (Optional[int]): 最多保留的样本数。

        Returns:
            List[Tuple[int, int]]: 可采样 chunk 起点列表。
        """

        sample_index: List[Tuple[int, int]] = []
        for trajectory_index, trajectory in enumerate(self.bundle.trajectories):
            trajectory_length = int(np.asarray(trajectory["actions"]).shape[0])
            if trajectory_length < self.chunk_size:
                continue
            for start_index in range(0, trajectory_length - self.chunk_size + 1):
                sample_index.append((trajectory_index, start_index))
                if max_chunks is not None and len(sample_index) >= max_chunks:
                    return sample_index
        return sample_index

    @staticmethod
    def _build_embedding_lookup(item_embeddings: np.ndarray) -> Dict[bytes, int]:
        """构建从 item embedding 原始字节到 item id 的精确查表。

        Args:
            item_embeddings (np.ndarray): item embedding 表，形状为 `(num_items, action_dim)`。

        Returns:
            Dict[bytes, int]: embedding bytes 到 item id 的映射。
        """

        lookup: Dict[bytes, int] = {}
        contiguous_embeddings = np.ascontiguousarray(item_embeddings.astype(np.float32, copy=False))
        for item_id, item_embedding in enumerate(contiguous_embeddings):
            lookup[item_embedding.tobytes()] = int(item_id)
        return lookup

    def _get_item_ids(self, trajectory_index: int, actions: np.ndarray) -> np.ndarray:
        """获取指定轨迹每一步 action embedding 对应的 item id。

        Args:
            trajectory_index (int): 轨迹下标。
            actions (np.ndarray): action embedding 矩阵。

        Returns:
            np.ndarray: 每一步 action 对应的 item id，形状为 `(T,)`。

        Raises:
            ValueError: 当 action embedding 无法精确匹配 item 表时抛出。
        """

        if trajectory_index in self._trajectory_item_cache:
            return self._trajectory_item_cache[trajectory_index]

        actions = np.ascontiguousarray(actions.astype(np.float32, copy=False))
        item_ids = np.empty(actions.shape[0], dtype=np.int64)
        missing_rows: List[int] = []
        for row_index, action_embedding in enumerate(actions):
            item_id = self._embedding_lookup.get(action_embedding.tobytes())
            if item_id is None:
                missing_rows.append(row_index)
                continue
            item_ids[row_index] = item_id

        if missing_rows:
            self._raise_lookup_error(trajectory_index, actions, missing_rows)
        self._trajectory_item_cache[trajectory_index] = item_ids
        return item_ids

    def _raise_lookup_error(
        self,
        trajectory_index: int,
        actions: np.ndarray,
        missing_rows: Sequence[int],
    ) -> None:
        """生成 action embedding 反查失败的诊断信息并抛错。

        Args:
            trajectory_index (int): 轨迹下标。
            actions (np.ndarray): 当前轨迹 action embedding。
            missing_rows (Sequence[int]): 无法精确匹配的行下标。

        Returns:
            None.

        Raises:
            ValueError: 总是抛出，包含最近邻误差信息。
        """

        probe_index = int(missing_rows[0])
        action = torch.as_tensor(actions[probe_index : probe_index + 1], dtype=torch.float32)
        item_embeddings = self.item_embeddings
        distances = torch.linalg.norm(item_embeddings - action, dim=1)
        min_distance, nearest_item = torch.min(distances, dim=0)
        if float(min_distance.item()) <= ACTION_MATCH_ATOL:
            LOGGER.warning(
                "发现接近精确匹配但 bytes 不一致：trajectory=%s, row=%s, item=%s, l2=%s",
                trajectory_index,
                probe_index,
                int(nearest_item.item()),
                float(min_distance.item()),
            )
        raise ValueError(
            "Offline action embedding cannot be exactly mapped to item embedding: "
            f"trajectory={trajectory_index}, missing_count={len(missing_rows)}, "
            f"first_missing_row={probe_index}, nearest_item={int(nearest_item.item())}, "
            f"l2={float(min_distance.item()):.8f}."
        )

    def _build_history_vectors(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        start_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构造 StateTrackerAvg 所需的固定窗口历史向量。

        Args:
            observations (np.ndarray): 轨迹状态矩阵。
            actions (np.ndarray): 轨迹 action embedding 矩阵。
            rewards (np.ndarray): 轨迹逐步 reward。
            start_index (int): 当前 chunk 起点。

        Returns:
            Tuple[np.ndarray, np.ndarray]: `history_vectors` 形状为
            `(window_size, state_dim)`，`history_valid` 形状为 `(window_size,)`。
        """

        state_dim = int(observations.shape[1])
        history_rows: List[np.ndarray] = []
        if start_index < self.window_size:
            # observations[0] 对应 reset dummy item，可精确复原早期 timestep 的 avg state。
            history_rows.append(observations[0].astype(np.float32, copy=False))
            real_start = 0
        else:
            real_start = start_index - self.window_size

        for step_index in range(real_start, start_index):
            history_rows.append(
                np.concatenate(
                    [actions[step_index], np.asarray([rewards[step_index]], dtype=np.float32)]
                ).astype(np.float32, copy=False)
            )

        history_rows = history_rows[-self.window_size :]
        history_vectors = np.zeros((self.window_size, state_dim), dtype=np.float32)
        history_valid = np.zeros(self.window_size, dtype=np.float32)
        if history_rows:
            offset = self.window_size - len(history_rows)
            history_vectors[offset:] = np.stack(history_rows, axis=0)
            history_valid[offset:] = 1.0
        return history_vectors, history_valid

    @staticmethod
    def _build_padded_item_history(
        item_ids: np.ndarray,
        start_index: int,
        history_size: int,
    ) -> np.ndarray:
        """构造固定长度 item 历史，左侧使用 `-1` padding。

        Args:
            item_ids (np.ndarray): 当前轨迹全部 item id。
            start_index (int): 当前 chunk 起点。
            history_size (int): 历史窗口长度。

        Returns:
            np.ndarray: 固定长度 item id 历史，形状为 `(history_size,)`。
        """

        history = np.full(history_size, INVALID_ITEM_ID, dtype=np.int64)
        if start_index <= 0:
            return history
        prefix = item_ids[max(0, start_index - history_size) : start_index]
        history[-len(prefix) :] = prefix
        return history
