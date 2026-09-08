"""DORL-MAC 到现有 CollectorSet 的策略适配器（离散 Categorical + rejection sampling）。"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch

from src.tianshou.tianshou.data import Batch, ReplayBuffer

import examples.our_model.models.mac_agent as mac_agent_module
from examples.our_model.policy.action_mapper import ActionMapper


ADR_COMPARISON_CATEGORY_OVERLAP = "category_overlap"
"""ADR 使用“类别集合是否有交集”的比较口径。"""


class DORLMACPolicyAdapter:
    """把离散版 MACAgent 适配成现有推荐 Collector 可调用的 policy。

    Collector 需要两个阶段：先调用 `__call__` 生成 action，再调用 `map_action` 得到
    环境可执行的 item id。在离散 Categorical MAC 下这两步都直接以 item id 为输出
    格式：`__call__` 使用 `MACAgent.select_chunks` 采样多组长度为 K 的
    item id 并用 critic 打分选出最优 chunk，然后最多连续返回其
    前 H 步；H 步后基于最新状态重新规划。`map_action` 直接返回
    identity。因此评估路径不再存在"连续向量 → 相似度点积 →
    top-1 item"的映射失真环节。
    """

    def __init__(
        self,
        agent: mac_agent_module.MACAgent,
        initial_states: torch.Tensor,
        action_mapper: ActionMapper,
        num_samples_test: int,
        device: torch.device,
        execution_horizon: Optional[int] = None,
        enable_open_loop_diagnostics: bool = False,
        item_categories: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        """初始化策略适配器。

        Args:
            agent (MACAgent): 已训练或已加载 checkpoint 的 DORL-MAC agent。
            initial_states (torch.Tensor): 每个环境内部用户的离线轨迹初始
                observation，形状为 `(num_users, state_dim)`。
            action_mapper (ActionMapper): 只提供 `num_items` 与 item embedding 表；
                本适配器不使用它做相似度映射。
            num_samples_test (int): 评估时 rejection sampling 候选数。
            device (torch.device): 计算设备。
            execution_horizon (Optional[int]): 每次规划后最多连续执行的
                chunk 前缀长度 `H`，必须位于 `[1, chunk_size]`。为
                `None` 时使用完整 chunk，即 `H = K`。
            enable_open_loop_diagnostics (bool): 是否在执行缓存后缀前执行
                shadow replan，用于统计动作重规划分歧率 ADR。
            item_categories (Optional[Sequence[Sequence[int]]]): 内部 item id
                到多标签类别列表的映射。启用 ADR 时必须提供；缓存动作与
                shadow replan 动作至少共享一个类别即视为一致。

        Raises:
            ValueError: 当初始状态、候选数、execution horizon 或类别映射非法时抛出。
        """

        if num_samples_test <= 0:
            raise ValueError("num_samples_test must be positive.")
        initial_states = torch.as_tensor(
            initial_states, dtype=torch.float32, device=device,
        )
        if initial_states.ndim != 2 or initial_states.shape[1] != agent.state_dim:
            raise ValueError(
                "initial_states must have shape (num_users, agent.state_dim), "
                f"got {tuple(initial_states.shape)} and state_dim={agent.state_dim}."
            )
        if initial_states.shape[0] <= 0 or not torch.isfinite(initial_states).all():
            raise ValueError("initial_states must be non-empty and finite.")
        self.agent = agent
        self.initial_states = initial_states
        self.action_mapper = action_mapper
        self.num_samples_test = int(num_samples_test)
        self.device = device
        self.n_items = action_mapper.num_items
        self.action_dim = action_mapper.action_dim
        self.chunk_size = agent.chunk_size
        resolved_execution_horizon = (
            self.chunk_size if execution_horizon is None else int(execution_horizon)
        )
        if not 1 <= resolved_execution_horizon <= self.chunk_size:
            raise ValueError(
                "execution_horizon must satisfy "
                f"1 <= H <= chunk_size ({self.chunk_size}), got {resolved_execution_horizon}."
            )
        self.execution_horizon = resolved_execution_horizon
        self.enable_open_loop_diagnostics = bool(enable_open_loop_diagnostics)
        self.adr_comparison = ADR_COMPARISON_CATEGORY_OVERLAP
        self._item_category_matrix = self._build_item_category_matrix(
            item_categories=item_categories,
        )
        self.adr_num_categories = (
            int(self._item_category_matrix.shape[1])
            if self._item_category_matrix is not None
            else 0
        )
        # chunk 内每步的 item id 缓存，`(B, K)`；`positions` 指向下一步应该出的 chunk step。
        self._cached_item_ids: Optional[torch.Tensor] = None
        self._cached_positions: Optional[torch.Tensor] = None
        self._model_states: Optional[torch.Tensor] = None
        self._pending_actions: Optional[torch.Tensor] = None
        self._pending_lengths: Optional[torch.Tensor] = None
        self._last_item_ids: Optional[torch.Tensor] = None
        self.reset_open_loop_diagnostics()

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
        """生成当前环境步要执行的离散 item id。

        Args:
            batch (Batch): Collector 当前 batch。
            buffer (Optional[ReplayBuffer]): Collector replay buffer。
            indices (Optional[np.ndarray]): 当前环境对应的 buffer last indices。
            is_obs (bool): 是否构造 obs state。
            remove_recommended_ids (bool): 是否在 chunk 边界的候选采样阶段屏蔽已推荐 item。
            is_train (bool): Collector 当前是否训练模式。
            state (Optional[Any]): 兼容 Collector 的 hidden state 参数，当前未使用。
            use_batch_in_statetracker (bool): 是否允许 StateTracker 使用当前 batch。
            **kwargs (Any): 兼容旧 policy 接口的额外参数。

        Returns:
            Batch: `act` 字段为整数 item id，形状 `(B,)`；`policy` 记录当前 chunk 内位置。
        """

        del state, kwargs, is_obs, is_train, use_batch_in_statetracker
        batch_size = self._infer_batch_size(batch)
        reset_mask = self._get_reset_mask(batch=batch, batch_size=batch_size)
        states = self._advance_model_states(batch=batch, reset_mask=reset_mask)
        recommended_mask = self._get_recommend_mask(
            remove_recommended_ids=remove_recommended_ids,
            batch_size=states.shape[0],
            buffer=buffer,
            indices=indices,
        )
        item_ids = self._next_chunk_item_ids(
            states=states.float(),
            reset_mask=reset_mask,
            recommended_mask=recommended_mask if remove_recommended_ids else None,
        )
        self._last_item_ids = item_ids.detach().clone()
        return Batch(
            act=item_ids.detach().cpu().numpy().astype(np.int64),
            policy=Batch(
                mac_chunk_position=self._cached_positions.detach().cpu().numpy()
                if self._cached_positions is not None
                else None,
            ),
        )

    def map_action(self, batch: Batch) -> np.ndarray:
        """直接返回 `batch.act` 中的 item id。

        离散 Categorical MAC 下 `__call__` 已经输出离散 item id，因此这里只做
        dtype 与形状规范化，不再执行"连续向量 → 相似度点积 → top-1"的映射。

        Args:
            batch (Batch): Collector 当前数据，其中 `act` 是离散 item id。

        Returns:
            np.ndarray: 环境可执行 item id，形状为 `(B,)`。
        """

        return np.asarray(batch.act, dtype=np.int64).reshape(-1)

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
        self.reset_chunk_cache()
        self.reset_open_loop_diagnostics()

    def eval(self, mode: bool = True) -> None:
        """切换评估模式。

        Args:
            mode (bool): 兼容旧接口的参数。

        Returns:
            None.
        """

        del mode
        self.agent.eval()
        self.reset_chunk_cache()
        self.reset_open_loop_diagnostics()

    def reset_chunk_cache(self) -> None:
        """清空评估期 chunk item id 缓存。

        Returns:
            None.
        """

        self._cached_item_ids = None
        self._cached_positions = None
        self._model_states = None
        self._pending_actions = None
        self._pending_lengths = None
        self._last_item_ids = None

    @staticmethod
    def _infer_batch_size(batch: Batch) -> int:
        """从 Collector batch 推断并校验并行环境数量。

        Args:
            batch (Batch): 当前 Collector batch，必须包含二维 `obs`。

        Returns:
            int: batch 行数。

        Raises:
            ValueError: 当 `obs` 缺失或不是二维数组时抛出。
        """

        observations = np.asarray(batch.obs)
        if observations.ndim != 2 or observations.shape[0] <= 0:
            raise ValueError(
                "Collector batch.obs must be a non-empty 2D user-item array."
            )
        return int(observations.shape[0])

    def _user_ids_from_batch(self, batch: Batch, batch_size: int) -> torch.Tensor:
        """读取环境 observation 第一列中的内部用户 ID。

        Args:
            batch (Batch): 当前 Collector batch。
            batch_size (int): 已校验的 batch 行数。

        Returns:
            torch.Tensor: 形状为 `(B,)` 的内部用户 ID。

        Raises:
            ValueError: 当用户 ID 数量或取值越界时抛出。
        """

        user_ids = torch.as_tensor(
            np.asarray(batch.obs)[:, 0], dtype=torch.long, device=self.device,
        ).reshape(-1)
        if user_ids.shape[0] != batch_size:
            raise ValueError("Collector user id count does not match batch size.")
        if torch.any(user_ids < 0) or torch.any(user_ids >= self.initial_states.shape[0]):
            raise ValueError("Collector user id is outside the initial state table.")
        return user_ids

    def _predict_pending_states(self) -> torch.Tensor:
        """用 dynamics 预测当前未提交动作前缀后的状态。

        Returns:
            torch.Tensor: 每个并行环境在当前决策时刻的模型状态。

        Raises:
            RuntimeError: 当内部状态缓存尚未初始化时抛出。
        """

        if (
            self._model_states is None
            or self._pending_actions is None
            or self._pending_lengths is None
        ):
            raise RuntimeError("Policy model-state cache is not initialized.")
        valid = torch.arange(
            self.chunk_size, device=self.device,
        ).unsqueeze(0) < self._pending_lengths.unsqueeze(1)
        predicted = self.agent.dynamics(
            self._model_states,
            self._pending_actions,
            valid,
        )
        has_pending = self._pending_lengths > 0
        return torch.where(has_pending.unsqueeze(1), predicted, self._model_states)

    def _advance_model_states(
        self, batch: Batch, reset_mask: torch.Tensor,
    ) -> torch.Tensor:
        """在每次环境交互后按已执行动作递推策略状态。

        新 episode 直接读取该用户在离线 pkl 中的初始 observation。连续
        episode 将上一步实际执行的 item embedding 写入待提交前缀；前缀达到
        execution horizon 时通过 chunk dynamics 提交一次边界状态。未达到
        边界时也用相同 dynamics 生成只供当前 shadow replan 使用的状态预览。

        Args:
            batch (Batch): 当前 Collector batch，第一列为内部用户 ID。
            reset_mask (torch.Tensor): 新 episode 行标记，形状为 `(B,)`。

        Returns:
            torch.Tensor: 当前决策时刻状态，形状为 `(B, state_dim)`。
        """

        batch_size = self._infer_batch_size(batch)
        user_ids = self._user_ids_from_batch(batch, batch_size)
        cache_missing = (
            self._model_states is None
            or self._model_states.shape[0] != batch_size
        )
        if cache_missing:
            self._model_states = self.initial_states[user_ids].clone()
            self._pending_actions = torch.zeros(
                (batch_size, self.chunk_size, self.action_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self._pending_lengths = torch.zeros(
                batch_size, dtype=torch.long, device=self.device,
            )
            self._last_item_ids = None
            reset_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        assert self._model_states is not None
        assert self._pending_actions is not None
        assert self._pending_lengths is not None
        reset_mask = reset_mask.to(device=self.device, dtype=torch.bool)
        continuation_mask = ~reset_mask
        if self._last_item_ids is not None and continuation_mask.any():
            rows = torch.nonzero(continuation_mask, as_tuple=False).squeeze(1)
            positions = self._pending_lengths[rows]
            if torch.any(positions >= self.chunk_size):
                raise RuntimeError("Pending dynamics prefix exceeded chunk_size.")
            action_embeddings = self.action_mapper.item_embeddings[
                self._last_item_ids[rows]
            ].to(device=self.device, dtype=torch.float32)
            self._pending_actions[rows, positions] = action_embeddings
            self._pending_lengths[rows] = positions + 1

            commit_mask = self._pending_lengths >= self.execution_horizon
            if commit_mask.any():
                committed_predictions = self._predict_pending_states()
                self._model_states[commit_mask] = committed_predictions[commit_mask]
                self._pending_actions[commit_mask] = 0.0
                self._pending_lengths[commit_mask] = 0

        if reset_mask.any():
            self._model_states[reset_mask] = self.initial_states[user_ids[reset_mask]]
            self._pending_actions[reset_mask] = 0.0
            self._pending_lengths[reset_mask] = 0
        return self._predict_pending_states()

    def reset_open_loop_diagnostics(self) -> None:
        """清空当前评估分支的 ADR 累计量。

        Returns:
            None.
        """

        self._adr_total_steps = 0
        self._adr_cached_tail_steps = 0
        self._adr_disagreements = 0
        self._adr_position_totals: dict[int, int] = {}
        self._adr_position_disagreements: dict[int, int] = {}

    def get_open_loop_diagnostics(self) -> dict[str, Optional[float]]:
        """返回当前评估分支的 ADR 汇总指标。

        ADR 使用类别集合交集而不是 item id 判断分歧：缓存动作与
        shadow replan 动作没有任何共享类别时才计为一次分歧。
        `ADR` 以实际执行的缓存后缀动作为分母；`ADR_exposed`
        以全部执行动作为分母。`H=1` 时不存在缓存后缀，
        因此条件 ADR 返回 `None`，暴露 ADR 返回 0。

        Returns:
            dict[str, Optional[float]]: ADR、暴露 ADR、计数及分位置 ADR。
        """

        conditional_adr: Optional[float] = None
        if self._adr_cached_tail_steps > 0:
            conditional_adr = (
                float(self._adr_disagreements) / float(self._adr_cached_tail_steps)
            )
        exposed_adr = 0.0
        if self._adr_total_steps > 0:
            exposed_adr = float(self._adr_disagreements) / float(self._adr_total_steps)
        metrics: dict[str, Optional[float]] = {
            "ADR": conditional_adr,
            "ADR_exposed": exposed_adr,
            "ADR_disagreements": float(self._adr_disagreements),
            "ADR_cached_tail_steps": float(self._adr_cached_tail_steps),
            "ADR_total_steps": float(self._adr_total_steps),
        }
        for position, total_count in sorted(self._adr_position_totals.items()):
            disagreement_count = self._adr_position_disagreements.get(position, 0)
            metrics[f"ADR@position_{position + 1}"] = (
                float(disagreement_count) / float(total_count)
            )
        return metrics

    def _build_item_category_matrix(
        self,
        item_categories: Optional[Sequence[Sequence[int]]],
    ) -> Optional[torch.Tensor]:
        """构造 ADR 使用的 item×category 布尔矩阵。

        Args:
            item_categories (Optional[Sequence[Sequence[int]]]): 内部 item id
                到类别列表的映射，长度必须等于 `self.n_items`；类别 id
                必须为非负整数。未启用 ADR 时允许为 `None`。

        Returns:
            Optional[torch.Tensor]: 形状为 `(num_items, num_categories)` 的
            bool 矩阵；未提供映射且 ADR 关闭时返回 `None`。

        Raises:
            ValueError: 当启用 ADR 却未提供映射、映射长度不匹配、
                类别为空或类别 id 非法时抛出。

        Example:
            若 `item_categories=[[1, 2], [2, 3]]`，返回矩阵的两行
            在类别 2 上都为 True，因此两个 item 被视为类别一致。
        """

        if item_categories is None:
            if self.enable_open_loop_diagnostics:
                raise ValueError(
                    "item_categories is required when open-loop ADR "
                    "diagnostics are enabled."
                )
            return None
        if len(item_categories) != self.n_items:
            raise ValueError(
                "item_categories length must match action_mapper.num_items, "
                f"got {len(item_categories)} and {self.n_items}."
            )

        normalized_categories: list[list[int]] = []
        all_categories: set[int] = set()
        for item_id, categories in enumerate(item_categories):
            normalized_item_categories: list[int] = []
            for raw_category in categories:
                category = int(raw_category)
                if category < 0:
                    raise ValueError(
                        "Category ids must be non-negative, "
                        f"got category={category} for item={item_id}."
                    )
                normalized_item_categories.append(category)
                all_categories.add(category)
            normalized_categories.append(normalized_item_categories)
        if not all_categories:
            raise ValueError("item_categories must contain at least one category.")

        num_categories = max(all_categories) + 1
        category_matrix = torch.zeros(
            (self.n_items, num_categories),
            dtype=torch.bool,
            device=self.device,
        )
        for item_id, categories in enumerate(normalized_categories):
            if categories:
                category_matrix[item_id, categories] = True
        return category_matrix

    def _category_disagreements(
        self,
        cached_item_ids: torch.Tensor,
        replanned_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """判断缓存动作与重规划动作是否类别完全不同。

        两个多标签 item 至少共享一个类别即视为一致；仅当类别集合
        交集为空时判为分歧。类别列表为空的 item 与任何 item 都没有
        可确认的共享类别，因此按分歧处理。

        Args:
            cached_item_ids (torch.Tensor): 实际将执行的缓存 item id，
                形状为 `(B_tail,)`。
            replanned_item_ids (torch.Tensor): 最新状态下 shadow replan
                得到的首 item id，形状为 `(B_tail,)`。

        Returns:
            torch.Tensor: bool 分歧标记，形状为 `(B_tail,)`。

        Raises:
            RuntimeError: 当类别矩阵未初始化时抛出。
            ValueError: 当两个输入形状不同或包含越界 item id 时抛出。
        """

        if self._item_category_matrix is None:
            raise RuntimeError("ADR category matrix is not initialized.")
        if cached_item_ids.shape != replanned_item_ids.shape:
            raise ValueError(
                "cached_item_ids and replanned_item_ids must have the same shape, "
                f"got {tuple(cached_item_ids.shape)} and "
                f"{tuple(replanned_item_ids.shape)}."
            )
        if (
            torch.any(cached_item_ids < 0)
            or torch.any(cached_item_ids >= self.n_items)
            or torch.any(replanned_item_ids < 0)
            or torch.any(replanned_item_ids >= self.n_items)
        ):
            raise ValueError("ADR item ids must be within the configured item range.")

        category_matrix = self._item_category_matrix
        cached_categories = category_matrix[cached_item_ids]
        replanned_categories = category_matrix[replanned_item_ids]
        has_shared_category = (cached_categories & replanned_categories).any(dim=1)
        return ~has_shared_category

    def _shadow_replan_first_item_ids(
        self,
        states: torch.Tensor,
        recommended_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """保持全局 RNG 不变地计算 shadow replan 的首动作。

        Args:
            states (torch.Tensor): 需要重新规划的当前状态，形状为
                `(B_shadow, state_dim)`。
            recommended_mask (Optional[torch.Tensor]): 可选的已推荐 item 屏蔽矩阵。

        Returns:
            torch.Tensor: shadow replan 选中 chunk 的第一个 item id，
            形状为 `(B_shadow,)`。
        """

        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state: Optional[torch.Tensor] = None
        device = torch.device(self.device)
        if device.type == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state(device)
        try:
            shadow_item_ids, _ = self.agent.select_chunks(
                states,
                num_samples=self.num_samples_test,
                recommended_mask=recommended_mask,
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device)
        return shadow_item_ids[:, 0]

    def _next_chunk_item_ids(
        self,
        states: torch.Tensor,
        reset_mask: torch.Tensor,
        recommended_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """按 chunk 缓存顺序返回当前步 item id。

        当已执行的缓存前缀达到 `execution_horizon` 时，未执行的
        chunk 后缀会被丢弃，并基于当前最新状态重新规划长度为
        `chunk_size` 的候选块。

        Args:
            states (torch.Tensor): 当前状态，形状为 `(B, state_dim)`。
            reset_mask (torch.Tensor): 哪些行是新 episode，需要丢弃旧缓存。
            recommended_mask (Optional[torch.Tensor]): 若非空，则在 chunk 边界重新
                做 rejection sampling 时屏蔽已推荐 item。

        Returns:
            torch.Tensor: 当前步 item id，形状为 `(B,)`。
        """

        batch_size = int(states.shape[0])
        if (
            self._cached_item_ids is None
            or self._cached_positions is None
            or self._cached_item_ids.shape[0] != batch_size
        ):
            self._cached_item_ids = torch.full(
                (batch_size, self.chunk_size),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            self._cached_positions = torch.full(
                (batch_size,),
                self.execution_horizon,
                dtype=torch.long,
                device=self.device,
            )

        reset_mask = reset_mask.to(device=self.device, dtype=torch.bool)
        self._cached_positions[reset_mask] = self.execution_horizon
        # H < K 时主动丢弃旧 chunk 的未执行后缀，形成滚动时域重规划。
        refill_mask = self._cached_positions >= self.execution_horizon
        if refill_mask.any():
            refill_states = states[refill_mask].to(device=self.device, dtype=torch.float32)
            refill_recommended_mask = None
            if recommended_mask is not None:
                refill_recommended_mask = recommended_mask.to(
                    device=self.device, dtype=torch.bool
                )[refill_mask]
            new_item_ids, _ = self.agent.select_chunks(
                refill_states,
                num_samples=self.num_samples_test,
                recommended_mask=refill_recommended_mask,
            )
            self._cached_item_ids[refill_mask] = new_item_ids
            self._cached_positions[refill_mask] = 0

        row_indices = torch.arange(batch_size, device=self.device)
        positions = self._cached_positions.clamp(min=0, max=self.chunk_size - 1)
        item_ids = self._cached_item_ids[row_indices, positions]
        if self.enable_open_loop_diagnostics:
            self._adr_total_steps += batch_size
            cached_tail_mask = (~refill_mask) & (positions > 0)
            if cached_tail_mask.any():
                shadow_recommended_mask = None
                if recommended_mask is not None:
                    shadow_recommended_mask = recommended_mask.to(
                        device=self.device,
                        dtype=torch.bool,
                    )[cached_tail_mask]
                shadow_first_item_ids = self._shadow_replan_first_item_ids(
                    states[cached_tail_mask].to(
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    shadow_recommended_mask,
                )
                cached_item_ids = item_ids[cached_tail_mask]
                # ADR 只关心推荐类别方向是否改变：两个 item 共享任一类别
                # 即视为一致，避免大离散 item 空间导致精确 id 指标饱和。
                disagreements = self._category_disagreements(
                    cached_item_ids=cached_item_ids,
                    replanned_item_ids=shadow_first_item_ids,
                )
                tail_positions = positions[cached_tail_mask]
                self._adr_cached_tail_steps += int(cached_item_ids.numel())
                self._adr_disagreements += int(disagreements.sum().item())
                for position in torch.unique(tail_positions).tolist():
                    position_int = int(position)
                    position_mask = tail_positions == position_int
                    self._adr_position_totals[position_int] = (
                        self._adr_position_totals.get(position_int, 0)
                        + int(position_mask.sum().item())
                    )
                    self._adr_position_disagreements[position_int] = (
                        self._adr_position_disagreements.get(position_int, 0)
                        + int(disagreements[position_mask].sum().item())
                    )
        self._cached_positions = self._cached_positions + 1
        return item_ids

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
