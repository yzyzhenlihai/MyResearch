"""将 RAVE Transformer Q 策略接入原 DORL CollectorSet。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import numpy as np
import torch

from src.tianshou.tianshou.data import Batch, ReplayBuffer

from examples.our_model.rave.state import StateManager
from examples.our_model.rave.training import CandidatePolicy


ADAPTIVE_EXECUTION_HORIZON = 0
"""评估元数据中表示由 Transformer Q 动态选择前缀长度的哨兵值。"""


def summarize_horizon_counts(
    counts: Counter[int],
    chunk_size: int,
) -> dict[str, float | int]:
    """把执行长度频数转换为可比较的分布统计。

    Args:
        counts (Counter[int]): 每个规划事件选择的前缀长度及其频数；长度必须
            位于 ``[1, chunk_size]``，频数必须为非负整数。
        chunk_size (int): 最大前缀长度，必须为正数。

    Returns:
        dict[str, float | int]: 包含总规划次数、平均规划长度、归一化熵、每个
            长度的原始频数与选择比例。空计数器的均值、熵和比例均为零。

    Raises:
        ValueError: 当 ``chunk_size`` 非正，或计数器包含非法长度和频数时抛出。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if any(
        not isinstance(horizon, int)
        or not 1 <= horizon <= chunk_size
        or not isinstance(count, int)
        or count < 0
        for horizon, count in counts.items()
    ):
        raise ValueError("Horizon counts must be non-negative integers in [1,K]")

    total = int(sum(counts.values()))
    summary: dict[str, float | int] = {
        "selected_horizon_total": total,
    }
    probabilities = []
    weighted_horizon = 0
    for horizon in range(1, chunk_size + 1):
        count = int(counts.get(horizon, 0))
        probability = count / total if total else 0.0
        summary[f"selected_horizon/h{horizon}"] = count
        summary[f"selected_horizon_fraction/h{horizon}"] = probability
        weighted_horizon += horizon * count
        if probability > 0:
            probabilities.append(probability)

    summary["selected_horizon_mean"] = weighted_horizon / total if total else 0.0
    if chunk_size == 1 or not probabilities:
        normalized_entropy = 0.0
    else:
        normalized_entropy = -sum(
            probability * float(np.log(probability))
            for probability in probabilities
        ) / float(np.log(chunk_size))
    summary["selected_horizon_normalized_entropy"] = normalized_entropy
    return summary


class RAVEDORLPolicyAdapter:
    """使用 Transformer Q 选块，并实现原 DORL Collector 策略接口。

    该适配器只替换 ordinary 策略中的 Q 打分器。候选 chunk 仍由同一个冻结
    MAC actor 生成，实际 item 仍交给 DORL 的 FB、NX_0 和 NX_force 环境执行。
    每个并行环境使用 Collector hidden state 中的稳定会话编号维护 Transformer
    所需的完整 item/reward 历史，避免活跃环境缩减时发生状态串行。

    Attributes:
        candidate_policy (CandidatePolicy): 冻结 actor 与 Transformer Q 的组合。
        manager (StateManager): 与训练一致的 chunk dynamics 状态维护器。
        initial_states (torch.Tensor): 按环境内部用户 ID 对齐的初始状态表。
        num_samples_test (int): 每次规划使用的候选 chunk 数。
        execution_horizon (int): 固定 H；动态 H 时为
            `ADAPTIVE_EXECUTION_HORIZON`。
    """

    def __init__(
        self,
        candidate_policy: CandidatePolicy,
        manager: StateManager,
        initial_states: torch.Tensor,
        device: torch.device,
    ) -> None:
        """初始化 DORL Collector 适配器。

        Args:
            candidate_policy (CandidatePolicy): 已加载 Transformer Q 的候选策略。
            manager (StateManager): 使用冻结 dynamics 的会话状态维护器。
            initial_states (torch.Tensor): 形状为 `(num_users, state_dim)` 的有限
                浮点初始状态表。
            device (torch.device): actor、Q 和状态张量所在设备。

        Returns:
            None.

        Raises:
            ValueError: 当初始状态表为空、维度错误、包含非有限值，或策略与
                manager 的 chunk 大小不一致时抛出。
        """

        initial_states = torch.as_tensor(
            initial_states,
            dtype=torch.float32,
            device=device,
        )
        if initial_states.ndim != 2 or initial_states.shape[0] <= 0:
            raise ValueError("initial_states must be a non-empty 2D tensor")
        if not torch.isfinite(initial_states).all():
            raise ValueError("initial_states must contain only finite values")
        if candidate_policy.network.chunk_size != manager.chunk_size:
            raise ValueError("Candidate policy and state manager chunk sizes differ")

        self.candidate_policy = candidate_policy
        self.manager = manager
        self.initial_states = initial_states
        self.device = torch.device(device)
        self.chunk_size = int(manager.chunk_size)
        self.num_samples_test = int(candidate_policy.num_candidates)
        self.execution_horizon = (
            ADAPTIVE_EXECUTION_HORIZON
            if candidate_policy.fixed_horizon is None
            else int(candidate_policy.fixed_horizon)
        )
        self.execution_horizon_mode = (
            "adaptive"
            if candidate_policy.fixed_horizon is None
            else "fixed"
        )
        self.enable_open_loop_diagnostics = False
        self.adr_comparison = "category_overlap"
        self.adr_num_categories = 0
        self.callbacks: list[Any] = []

        self._sessions: dict[int, Any] = {}
        self._action_cache: dict[int, list[int]] = {}
        self._pending_actions: dict[int, tuple[int, bool]] = {}
        self._next_session_id = 0
        self._selected_horizons: Counter[int] = Counter()
        self._branch_selected_horizons: Counter[int] = Counter()

    @torch.no_grad()
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
        """为 Collector 当前活跃环境生成下一批离散 item。

        Args:
            batch (Batch): Collector 当前 batch；`obs[:, 0]` 为内部用户 ID，
                `rew_prev` 为上一步真实环境反馈，`is_start` 标识 episode 起点。
            buffer (ReplayBuffer | None): DORL Collector buffer；Transformer 状态
                由内部 Session 维护，因此这里只保留接口兼容。
            indices (np.ndarray | None): 当前 buffer 索引，仅用于接口兼容。
            is_obs (bool): 原 DORL policy 接口参数，仅用于兼容。
            remove_recommended_ids (bool): FB 为 False，NX 分支为 True；直接控制
                actor 候选采样是否屏蔽完整会话历史。
            is_train (bool): Collector 是否处于训练收集模式，仅用于兼容。
            state (Any | None): 上一轮返回的稳定会话编号数组。
            use_batch_in_statetracker (bool): 原 StateTracker 参数，仅用于兼容。
            **kwargs (Any): 其他 Collector 兼容参数。

        Returns:
            Batch: `act` 是形状 `(B,)` 的 item ID；`state` 保存稳定会话编号。

        Raises:
            ValueError: 当 batch 形状、用户 ID、reward 或 hidden state 非法时抛出。
            RuntimeError: 当连续会话缺少上一步待提交动作或内部 Session 时抛出。
        """

        del buffer, indices, is_obs, is_train, use_batch_in_statetracker, kwargs
        observations = np.asarray(batch.obs)
        if observations.ndim != 2 or observations.shape[0] <= 0:
            raise ValueError("Collector batch.obs must be a non-empty 2D array")
        batch_size = int(observations.shape[0])
        user_ids = torch.as_tensor(
            observations[:, 0],
            dtype=torch.long,
            device=self.device,
        )
        if torch.any(user_ids < 0) or torch.any(user_ids >= len(self.initial_states)):
            raise ValueError("Collector user id is outside the initial state table")

        reset_mask = self._reset_mask(batch, batch_size)
        session_ids = self._resolve_session_ids(
            state=state,
            reset_mask=reset_mask,
            user_ids=user_ids,
        )
        self._commit_previous_feedback(
            batch=batch,
            session_ids=session_ids,
            reset_mask=reset_mask,
        )
        item_ids, new_plan_flags = self._select_actions(
            session_ids=session_ids,
            mask_history=remove_recommended_ids,
        )
        for row_index, session_id in enumerate(session_ids.tolist()):
            self._pending_actions[int(session_id)] = (
                int(item_ids[row_index]),
                bool(new_plan_flags[row_index]),
            )
        return Batch(
            act=item_ids.astype(np.int64),
            policy=Batch(rave_new_plan=new_plan_flags.astype(bool)),
            state=session_ids.astype(np.int64),
        )

    def map_action(self, batch: Batch) -> np.ndarray:
        """把已经离散化的动作原样映射到 DORL 环境。

        Args:
            batch (Batch): 包含整数 `act` 字段的 Collector batch。

        Returns:
            np.ndarray: 形状为 `(B,)` 的环境内部 item ID。
        """

        return np.asarray(batch.act, dtype=np.int64).reshape(-1)

    def map_action_inverse(self, actions: Any) -> Any:
        """兼容 Collector 随机动作接口并原样返回动作。

        Args:
            actions (Any): 环境采样得到的原始动作。

        Returns:
            Any: 未修改的动作。
        """

        return actions

    def exploration_noise(self, actions: Any, batch: Batch) -> Any:
        """保持评估动作不变以兼容 Collector 接口。

        Args:
            actions (Any): 当前策略动作。
            batch (Batch): 当前 Collector batch，仅用于接口兼容。

        Returns:
            Any: 未修改的动作。
        """

        del batch
        return actions

    def state_dict(self) -> dict[str, torch.Tensor]:
        """返回 Transformer Q 网络参数。

        Returns:
            dict[str, torch.Tensor]: 当前在线 Transformer Q 的 state dict。
        """

        return self.candidate_policy.network.state_dict()

    def train(self, mode: bool = True) -> None:
        """切换网络模式并清空仅属于评估会话的缓存。

        Args:
            mode (bool): True 为训练模式，False 为评估模式。

        Returns:
            None.
        """

        self.candidate_policy.network.train(mode)
        self.candidate_policy.mac.train(mode)
        self.reset_chunk_cache()

    def eval(self, mode: bool = True) -> None:
        """切换到评估模式并清空上一次 DORL 评估状态。

        Args:
            mode (bool): 兼容原 policy 接口；传入值不改变评估语义。

        Returns:
            None.
        """

        del mode
        self.candidate_policy.network.eval()
        self.candidate_policy.mac.eval()
        self.reset_chunk_cache()

    def reset_chunk_cache(self) -> None:
        """清空所有并行会话、动作缓存和评估统计。

        Returns:
            None.
        """

        self._sessions.clear()
        self._action_cache.clear()
        self._pending_actions.clear()
        self._next_session_id = 0
        self._selected_horizons.clear()
        self._branch_selected_horizons.clear()

    def reset_open_loop_diagnostics(self) -> None:
        """清空当前 DORL 分支的执行长度统计。

        Returns:
            None. 汇总计数继续保留，供一次完整三分支评估结束后使用。
        """

        self._branch_selected_horizons.clear()

    def get_open_loop_diagnostics(self) -> dict[str, float | int]:
        """返回当前 DORL 分支的执行长度频数和归一化分布。

        Returns:
            dict[str, float | int]: 当前分支的规划次数、平均 H、归一化熵，
            以及每个 H 的频数和选择比例。
        """

        return summarize_horizon_counts(
            self._branch_selected_horizons,
            self.chunk_size,
        )

    def get_evaluation_metadata(self) -> dict[str, Any]:
        """返回 Transformer Q 的执行长度元数据。

        Returns:
            dict[str, Any]: 固定/动态 H 模式，以及三分支合计的 H 频数、
            选择比例、均值和归一化熵。
        """

        if self.execution_horizon_mode == "fixed":
            return {}
        metadata: dict[str, Any] = {
            "evaluation/execution_horizon_mode": self.execution_horizon_mode,
        }
        metadata.update({
            f"evaluation/{name}": value
            for name, value in summarize_horizon_counts(
                self._selected_horizons,
                self.chunk_size,
            ).items()
        })
        return metadata

    def _resolve_session_ids(
        self,
        state: Optional[Any],
        reset_mask: np.ndarray,
        user_ids: torch.Tensor,
    ) -> np.ndarray:
        """恢复连续行会话编号，并为 episode 起点分配新编号。

        Args:
            state (Any | None): Collector 保存的上一轮 hidden state。
            reset_mask (np.ndarray): 形状 `(B,)` 的 episode 起点标记。
            user_ids (torch.Tensor): 形状 `(B,)` 的内部用户 ID。

        Returns:
            np.ndarray: 形状 `(B,)` 的稳定会话编号。

        Raises:
            ValueError: 当 hidden state 数量与当前 batch 不一致时抛出。
            RuntimeError: 当连续行引用未知会话时抛出。
        """

        batch_size = len(reset_mask)
        if state is None:
            session_ids = np.full(batch_size, -1, dtype=np.int64)
        else:
            session_ids = np.asarray(state, dtype=np.int64).reshape(-1).copy()
            if len(session_ids) != batch_size:
                raise ValueError("Collector hidden state size does not match batch size")

        for row_index in range(batch_size):
            if reset_mask[row_index]:
                session_id = self._next_session_id
                self._next_session_id += 1
                session_ids[row_index] = session_id
                self._sessions[session_id] = self.manager.reset(
                    int(user_ids[row_index]),
                    self.initial_states[user_ids[row_index]],
                )
                self._action_cache[session_id] = []
                self._pending_actions.pop(session_id, None)
            elif int(session_ids[row_index]) not in self._sessions:
                raise RuntimeError("Continuing Collector row references an unknown session")
        return session_ids

    def _commit_previous_feedback(
        self,
        batch: Batch,
        session_ids: np.ndarray,
        reset_mask: np.ndarray,
    ) -> None:
        """把上一步真实环境反馈写入所有连续会话。

        Args:
            batch (Batch): 必须包含与当前 batch 对齐的 `rew_prev`。
            session_ids (np.ndarray): 当前稳定会话编号。
            reset_mask (np.ndarray): episode 起点标记；起点没有待提交动作。

        Returns:
            None.

        Raises:
            ValueError: 当 `rew_prev` 数量不匹配或包含非有限值时抛出。
            RuntimeError: 当连续会话缺少上一轮发出的动作时抛出。
        """

        rewards = np.asarray(batch.rew_prev, dtype=np.float64).reshape(-1)
        if len(rewards) != len(session_ids) or not np.isfinite(rewards).all():
            raise ValueError("Collector rew_prev must be finite and batch-aligned")
        for row_index, session_id_value in enumerate(session_ids.tolist()):
            if reset_mask[row_index]:
                continue
            session_id = int(session_id_value)
            if session_id not in self._pending_actions:
                raise RuntimeError("Continuing session is missing its previous action")
            item_id, new_plan = self._pending_actions.pop(session_id)
            self.manager.advance(
                self._sessions[session_id],
                item_id,
                float(rewards[row_index]),
                new_plan,
                natural_done=False,
            )

    def _select_actions(
        self,
        session_ids: np.ndarray,
        mask_history: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """补充空缓存并返回每个会话下一步动作。

        Args:
            session_ids (np.ndarray): 当前 batch 的稳定会话编号。
            mask_history (bool): 是否按 DORL NX 分支屏蔽历史 item。

        Returns:
            tuple[np.ndarray, np.ndarray]: item ID 和是否为新规划首步，形状均为
            `(B,)`。

        Raises:
            RuntimeError: 当 Transformer 返回空前缀或缓存状态不完整时抛出。
        """

        refill_ids = [
            int(session_id)
            for session_id in session_ids.tolist()
            if not self._action_cache.get(int(session_id), [])
        ]
        if refill_ids:
            sessions = [self._sessions[session_id] for session_id in refill_ids]
            candidates, horizons, _ = self.candidate_policy.choose(
                sessions,
                epsilon=0.0,
                mask_history=mask_history,
            )
            for row_index, session_id in enumerate(refill_ids):
                horizon = int(horizons[row_index])
                selected_prefix = candidates[row_index, :horizon].tolist()
                if not selected_prefix:
                    raise RuntimeError("Transformer Q selected an empty action prefix")
                self._action_cache[session_id] = [int(item) for item in selected_prefix]
                self._selected_horizons[horizon] += 1
                self._branch_selected_horizons[horizon] += 1

        refill_set = set(refill_ids)
        item_ids = np.empty(len(session_ids), dtype=np.int64)
        new_plan_flags = np.zeros(len(session_ids), dtype=bool)
        for row_index, session_id_value in enumerate(session_ids.tolist()):
            session_id = int(session_id_value)
            action_cache = self._action_cache.get(session_id)
            if not action_cache:
                raise RuntimeError("RAVE action cache is unexpectedly empty")
            item_ids[row_index] = int(action_cache.pop(0))
            new_plan_flags[row_index] = session_id in refill_set
        return item_ids, new_plan_flags

    @staticmethod
    def _reset_mask(batch: Batch, batch_size: int) -> np.ndarray:
        """读取并校验 Collector episode 起点标记。

        Args:
            batch (Batch): 当前 Collector batch。
            batch_size (int): 当前活跃环境数。

        Returns:
            np.ndarray: 形状 `(B,)` 的 bool 起点标记。

        Raises:
            ValueError: 当 `is_start` 缺失或数量不匹配时抛出。
        """

        if not hasattr(batch, "is_start"):
            raise ValueError("Collector batch must provide is_start")
        reset_mask = np.asarray(batch.is_start, dtype=bool).reshape(-1)
        if len(reset_mask) != batch_size:
            raise ValueError("Collector is_start size does not match batch size")
        return reset_mask
