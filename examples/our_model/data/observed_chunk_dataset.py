"""直接使用离线文件既有状态构造动作块监督数据。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from examples.our_model.data.action_chunk_dataset import ActionChunkDataset


class ObservedChunkDataset(ActionChunkDataset):
    """从已编码 transition 组合 MAC 动作块，不再调用 StateTracker。

    pkl 中的 `observations` 和 `next_observations` 是数据生成阶段保存的固定
    状态标签。本类直接使用这些数值；每个有效动作前缀的目标是对应
    transition 的 `next_observations`，因此训练期无需重新编码交互历史。
    """

    def __init__(self, *args, **kwargs) -> None:
        """初始化预编码状态动作块数据集。

        Args:
            *args: `ActionChunkDataset` 位置参数。
            **kwargs: `ActionChunkDataset` 关键字参数。
        """
        super().__init__(*args, **kwargs)
        self.state_dim = self.bundle.state_dim

    def _build_sample_index(self, max_chunks: Optional[int]) -> List[Tuple[int, int]]:
        """保留所有真实起点，包括短会话和尾部不足 K 的片段。

        Args:
            max_chunks (Optional[int]): 正数样本上限或 None。

        Returns:
            List[Tuple[int, int]]: 轨迹编号与起点列表。

        Raises:
            ValueError: 上限非正时抛出。
        """
        if max_chunks is not None and max_chunks <= 0:
            raise ValueError("max_chunks must be positive.")
        sample_index = []
        for trajectory_index, trajectory in enumerate(self.bundle.trajectories):
            for start in range(len(trajectory["actions"])):
                sample_index.append((trajectory_index, start))
                if max_chunks is not None and len(sample_index) >= max_chunks:
                    return sample_index
        return sample_index

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """返回动作块以及 pkl 中对应的真实状态与前缀目标。

        Args:
            index (int): `[0, len(dataset))` 样本编号。

        Returns:
            Dict[str, torch.Tensor]: dynamics、BC 与 Q/V 共用的 batch 行。

        Raises:
            IndexError: 下标越界时抛出。
            ValueError: 状态、奖励或动作包含非有限值时抛出。
        """
        if not 0 <= index < len(self):
            raise IndexError(index)
        trajectory_index, position = self.sample_index[index]
        trajectory = self.bundle.trajectories[trajectory_index]
        observations = np.asarray(trajectory["observations"], dtype=np.float32)
        next_observations = np.asarray(trajectory["next_observations"], dtype=np.float32)
        actions = np.asarray(trajectory["actions"], dtype=np.float32)
        rewards = np.asarray(trajectory["rewards"], dtype=np.float32).reshape(-1)
        terminals = np.asarray(trajectory["terminals"], dtype=bool).reshape(-1)
        if not (
            np.isfinite(observations).all()
            and np.isfinite(next_observations).all()
            and np.isfinite(actions).all()
            and np.isfinite(rewards).all()
        ):
            raise ValueError("Stored transitions must contain finite values.")

        item_ids = self._get_item_ids(trajectory_index, actions)
        previous_terminals = np.flatnonzero(terminals[:position])
        episode_start = int(previous_terminals[-1] + 1) if len(previous_terminals) else 0
        end = min(position + self.chunk_size, len(actions))
        future_terminals = np.flatnonzero(terminals[position:end])
        if len(future_terminals):
            end = position + int(future_terminals[0]) + 1
        length = end - position

        action_chunk = torch.zeros((self.chunk_size, self.bundle.action_dim))
        action_chunk[:length] = torch.from_numpy(actions[position:end])
        item_chunk = torch.zeros(self.chunk_size, dtype=torch.long)
        item_chunk[:length] = torch.from_numpy(item_ids[position:end])
        reward_steps = torch.zeros(self.chunk_size)
        reward_steps[:length] = torch.from_numpy(rewards[position:end])
        valid = torch.arange(self.chunk_size) < length
        prefix_targets = torch.from_numpy(next_observations[end - 1]).repeat(
            self.chunk_size, 1,
        )
        prefix_targets[:length] = torch.from_numpy(next_observations[position:end])
        leave_history = self._build_padded_item_history(
            item_ids[episode_start:], position - episode_start, self.leave_history_size,
        )
        recommended_mask = torch.zeros(self.num_items, dtype=torch.bool)
        recommended_mask[torch.as_tensor(item_ids[episode_start:position])] = True
        return {
            "observations": torch.from_numpy(observations[position]),
            "next_observations": torch.from_numpy(next_observations[end - 1]),
            "prefix_next_observations": prefix_targets,
            "actions": action_chunk.flatten(),
            "chunk_item_ids": item_chunk,
            "chunk_valid": valid,
            "chunk_step_rewards": reward_steps,
            "rewards": (reward_steps * torch.from_numpy(self.discount)).sum().reshape(1),
            "terminals": torch.tensor([bool(terminals[end - 1] or end == len(actions))]),
            "initial_terminals": torch.tensor([False]),
            "leave_history_item_ids": torch.from_numpy(leave_history),
            "recommended_mask": recommended_mask,
            "user_id": torch.tensor([int(trajectory["user_id"])]),
            "start_index": torch.tensor([position]),
            "env_step": torch.tensor([0]),
        }
