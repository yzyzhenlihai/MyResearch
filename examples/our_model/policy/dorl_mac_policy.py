"""DORL-MAC 到现有 CollectorSet 的策略适配器。"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from src.tianshou.tianshou.data import Batch, ReplayBuffer

import examples.our_model.models.mac_agent as mac_agent_module
from examples.our_model.policy.action_mapper import ActionMapper


class DORLMACPolicyAdapter:
    """把 MACAgent 适配成现有推荐 Collector 可调用的 policy。

    Collector 需要两个阶段：先调用 `__call__` 生成连续 action embedding，
    再调用 `map_action` 映射为环境可执行 item id。本适配器复用已有
    StateTracker 构造 state embedding，不引入 Tianshou A2C 训练链路。
    """

    def __init__(
        self,
        agent: mac_agent_module.MACAgent,
        state_tracker: torch.nn.Module,
        action_mapper: ActionMapper,
        num_samples_test: int,
        device: torch.device,
    ) -> None:
        """初始化策略适配器。

        Args:
            agent (MACAgent): 已训练或已加载 checkpoint 的 DORL-MAC agent。
            state_tracker (torch.nn.Module): 与离线数据一致的 StateTrackerAvg。
            action_mapper (ActionMapper): action embedding 到 item id 的映射器。
            num_samples_test (int): 评估时 rejection sampling 候选数。
            device (torch.device): 计算设备。

        Raises:
            ValueError: 当候选数非法时抛出。
        """

        if num_samples_test <= 0:
            raise ValueError("num_samples_test must be positive.")
        self.agent = agent
        self.state_tracker = state_tracker
        self.action_mapper = action_mapper
        self.num_samples_test = int(num_samples_test)
        self.device = device
        self.n_items = action_mapper.num_items
        self.action_dim = action_mapper.action_dim
        self.chunk_size = agent.chunk_size
        self._cached_chunks: Optional[torch.Tensor] = None
        self._cached_positions: Optional[torch.Tensor] = None

    def __call__(
        self,
        batch: Batch,
        buffer: Optional[ReplayBuffer],
        indices: Optional[np.ndarray] = None,
        is_obs: bool = True,
        remove_recommended_ids: bool = False,
        is_train: bool = False,
        state: Optional[Any] = None,
        use_batch_in_statetracker: bool = True,
        **kwargs: Any,
    ) -> Batch:
        """生成当前环境步的第一个 chunk action embedding。

        Args:
            batch (Batch): Collector 当前 batch。
            buffer (Optional[ReplayBuffer]): Collector replay buffer。
            indices (Optional[np.ndarray]): 当前环境对应的 buffer last indices。
            is_obs (bool): 是否构造 obs state。
            remove_recommended_ids (bool): 是否启用已推荐 item mask。
            is_train (bool): Collector 当前是否训练模式。
            state (Optional[Any]): 兼容 Collector 的 hidden state 参数，当前未使用。
            use_batch_in_statetracker (bool): 是否允许 StateTracker 使用当前 batch。
            **kwargs (Any): 兼容旧 policy 接口的额外参数。

        Returns:
            Batch: 包含连续 action embedding 和推荐 mask 的结果。
        """

        del state, kwargs
        states = self.state_tracker(
            buffer=buffer,
            indices=indices,
            is_obs=is_obs,
            batch=batch,
            is_train=is_train,
            use_batch_in_statetracker=use_batch_in_statetracker,
        ).to(self.device)
        recommended_mask = self._get_recommend_mask(
            remove_recommended_ids=remove_recommended_ids,
            batch_size=states.shape[0],
            buffer=buffer,
            indices=indices,
        )
        chunk_actions = self._next_chunk_actions(
            states=states.float(),
            reset_mask=self._get_reset_mask(batch=batch, batch_size=states.shape[0]),
        )
        return Batch(
            act=chunk_actions.detach().cpu().numpy(),
            policy=Batch(
                mac_recommended_mask=recommended_mask.detach().cpu().numpy(),
                mac_chunk_position=self._cached_positions.detach().cpu().numpy()
                if self._cached_positions is not None
                else None,
            ),
        )

    def map_action(self, batch: Batch) -> np.ndarray:
        """把连续 action embedding 映射为 item id。

        Args:
            batch (Batch): Collector 当前数据，其中 `act` 是连续 action embedding。

        Returns:
            np.ndarray: 环境可执行 item id，形状为 `(B,)`。
        """

        action_embeddings = torch.as_tensor(batch.act, dtype=torch.float32, device=self.device)
        recommended_mask = None
        if hasattr(batch, "policy") and hasattr(batch.policy, "mac_recommended_mask"):
            recommended_mask = torch.as_tensor(
                batch.policy.mac_recommended_mask,
                dtype=torch.bool,
                device=self.device,
            )
        item_ids, _ = self.action_mapper.map_embeddings(
            action_embeddings,
            recommended_mask=recommended_mask,
            topk=1,
        )
        return item_ids.squeeze(1).detach().cpu().numpy()

    def map_action_inverse(self, actions: Any) -> Any:
        """兼容 Collector 随机动作接口。

        Args:
            actions (Any): 原始动作。

        Returns:
            Any: 未修改的动作。
        """

        return actions

    def exploration_noise(self, actions: Any, batch: Batch) -> Any:
        """兼容 Collector exploration noise 接口。

        Args:
            actions (Any): 原始动作。
            batch (Batch): Collector batch，当前未使用。

        Returns:
            Any: 未修改的动作。
        """

        del batch
        return actions

    def state_dict(self) -> dict:
        """返回 agent 参数字典。

        Returns:
            dict: agent 的 checkpoint state。
        """

        return self.agent.state_dict()

    def train(self, mode: bool = True) -> None:
        """切换训练模式。

        Args:
            mode (bool): 是否训练模式。

        Returns:
            None.
        """

        self.agent.train(mode)
        self.state_tracker.train(mode)
        self.reset_chunk_cache()

    def eval(self, mode: bool = True) -> None:
        """切换评估模式。

        Args:
            mode (bool): 兼容旧接口的参数。

        Returns:
            None.
        """

        del mode
        self.agent.eval()
        self.state_tracker.eval()
        self.reset_chunk_cache()

    def reset_chunk_cache(self) -> None:
        """清空评估期 action chunk 缓存。

        Returns:
            None.
        """

        self._cached_chunks = None
        self._cached_positions = None

    def _next_chunk_actions(self, states: torch.Tensor, reset_mask: torch.Tensor) -> torch.Tensor:
        """按 chunk 缓存顺序返回当前步 action embedding。

        Args:
            states (torch.Tensor): 当前状态，形状为 `(B, state_dim)`。
            reset_mask (torch.Tensor): 哪些行是新 episode，需要丢弃旧缓存。

        Returns:
            torch.Tensor: 当前步 action embedding，形状为 `(B, action_dim)`。
        """

        batch_size = int(states.shape[0])
        if (
            self._cached_chunks is None
            or self._cached_positions is None
            or self._cached_chunks.shape[0] != batch_size
        ):
            self._cached_chunks = torch.zeros(
                batch_size,
                self.chunk_size,
                self.action_dim,
                dtype=torch.float32,
                device=self.device,
            )
            self._cached_positions = torch.full(
                (batch_size,),
                self.chunk_size,
                dtype=torch.long,
                device=self.device,
            )

        reset_mask = reset_mask.to(device=self.device, dtype=torch.bool)
        self._cached_positions[reset_mask] = self.chunk_size
        refill_mask = self._cached_positions >= self.chunk_size
        if refill_mask.any():
            refill_states = states[refill_mask].to(device=self.device, dtype=torch.float32)
            new_chunks = self.agent.select_chunks(refill_states, num_samples=self.num_samples_test)
            self._cached_chunks[refill_mask] = new_chunks.view(-1, self.chunk_size, self.action_dim)
            self._cached_positions[refill_mask] = 0

        row_indices = torch.arange(batch_size, device=self.device)
        positions = self._cached_positions.clamp(min=0, max=self.chunk_size - 1)
        actions = self._cached_chunks[row_indices, positions]
        self._cached_positions = self._cached_positions + 1
        return actions

    def _get_reset_mask(self, batch: Batch, batch_size: int) -> torch.Tensor:
        """从 Collector batch 中读取 episode 起始标记。

        Args:
            batch (Batch): Collector 当前 batch。
            batch_size (int): 当前 batch 大小。

        Returns:
            torch.Tensor: bool reset mask，形状为 `(B,)`。
        """

        if hasattr(batch, "is_start"):
            is_start = np.asarray(batch.is_start, dtype=bool).reshape(-1)
            if is_start.shape[0] == batch_size:
                return torch.as_tensor(is_start, dtype=torch.bool, device=self.device)
        return torch.zeros(batch_size, dtype=torch.bool, device=self.device)

    def _get_recommend_mask(
        self,
        remove_recommended_ids: bool,
        batch_size: int,
        buffer: Optional[ReplayBuffer],
        indices: Optional[np.ndarray],
    ) -> torch.Tensor:
        """从 Collector buffer 构造已推荐 item mask。

        Args:
            remove_recommended_ids (bool): 是否移除已推荐 item。
            batch_size (int): 当前 batch 大小。
            buffer (Optional[ReplayBuffer]): Collector replay buffer。
            indices (Optional[np.ndarray]): 当前环境对应的 buffer last indices。

        Returns:
            torch.Tensor: bool mask，形状为 `(B, num_items)`。
        """

        mask = torch.zeros((batch_size, self.n_items), dtype=torch.bool, device=self.device)
        if not remove_recommended_ids or buffer is None or indices is None or len(buffer) == 0:
            return mask
        recommended_ids = self._collect_recommended_ids(buffer, np.asarray(indices))
        for batch_index, item_list in enumerate(recommended_ids):
            valid_items = item_list[(item_list >= 0) & (item_list < self.n_items)]
            if len(valid_items) > 0:
                mask[batch_index, torch.as_tensor(valid_items, dtype=torch.long, device=self.device)] = True
        return mask

    @staticmethod
    def _collect_recommended_ids(buffer: ReplayBuffer, indices: np.ndarray) -> np.ndarray:
        """沿 replay buffer 回溯当前 episode 已推荐 item。

        Args:
            buffer (ReplayBuffer): Collector replay buffer。
            indices (np.ndarray): 当前环境 last indices。

        Returns:
            np.ndarray: 每个样本的历史 item id 矩阵，缺省位置为 `-1`。
        """

        histories = []
        for start_index in indices:
            current_index = int(start_index)
            item_history = []
            while True:
                obs_next = buffer[current_index].obs_next
                if np.asarray(obs_next).ndim == 1:
                    item_history.append(int(obs_next[1]))
                if bool(buffer.is_start[current_index]):
                    break
                previous_index = int(buffer.prev(current_index))
                if previous_index == current_index:
                    break
                current_index = previous_index
            histories.append(item_history)
        max_length = max((len(history) for history in histories), default=0)
        padded = np.full((len(histories), max_length), -1, dtype=np.int64)
        for row_index, history in enumerate(histories):
            if history:
                padded[row_index, : len(history)] = np.asarray(history, dtype=np.int64)
        return padded
