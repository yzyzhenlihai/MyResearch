"""在 ACSAC fresh imagined rollout 上训练 R/T/D/C 四头前缀 Q。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from examples.our_model.models.acsac_agent import (
    ACSACTransformerAgent,
    DEFAULT_ALL_PREFIX_AUX_WEIGHT,
)
from examples.our_model.models.chunk_value import (
    CRITIC_BACKBONE_MULTI_PREFIX_TRANSFORMER,
    DEFAULT_CRITIC_TRANSFORMER_DROPOUT,
    DEFAULT_CRITIC_TRANSFORMER_HEADS,
    DEFAULT_CRITIC_TRANSFORMER_HIDDEN_DIM,
    DEFAULT_CRITIC_TRANSFORMER_LAYERS,
    MultiHeadCausalTransformerPrefixCritic,
)
from examples.our_model.models.mac_agent import (
    LEAVE_POLICY_PENALTY,
    REPEAT_POLICY_MASK,
    REPEAT_POLICY_TRUNCATE,
    RolloutHistoryState,
    RolloutResult,
)
from examples.our_model.rave.model import NUM_COMPONENTS, Utility
from examples.our_model.rave.state import COMPONENTS

MULTI_PREFIX_CHECKPOINT_FORMAT = "model_based_multi_prefix_transformer_v1"
"""模型化 R/T/D/C 四头 Transformer 分支的 checkpoint 格式。"""

DEFAULT_MULTI_WEIGHTS = (1.0, 0.2, 0.2, -0.05)
"""R/T/D/C 默认业务权重；成本头只在综合评分时取负号。"""

DEFAULT_MULTI_SCALES = (1.0, 1.0, 1.0, 1.0)
"""加载 checkpoint 前使用的安全占位尺度；正式训练会传入冻结尺度。"""


@dataclass
class MultiAdaptiveTrajectoryStep:
    """四头自适应 imagined rollout 的一个宏动作决策。

    Attributes:
        states (torch.Tensor): chunk 起点状态，形状 ``(B,state_dim)``。
        selected_chunks (torch.Tensor): 选中候选 embedding，形状 ``(B,K*A)``。
        selected_item_ids (torch.Tensor): 选中候选 item，形状 ``(B,K)``。
        selected_lengths (torch.Tensor): 每行选中的执行长度，形状 ``(B,)``。
        rollout (RolloutResult): 当前宏动作的逐步执行结果。
        active (torch.Tensor): 当前 chunk 起点是否仍可决策，形状 ``(B,)``。
        chunk_discount (torch.Tensor): ``gamma^effective_steps``，形状 ``(B,)``。
        component_rewards (torch.Tensor): R/T/D/C chunk 回报，形状 ``(B,4)``。
        bootstrap_values (torch.Tensor): 同一续接动作的 target-Q 四分量，
            形状 ``(B,4)``。
    """

    states: torch.Tensor
    selected_chunks: torch.Tensor
    selected_item_ids: torch.Tensor
    selected_lengths: torch.Tensor
    rollout: RolloutResult
    active: torch.Tensor
    chunk_discount: torch.Tensor
    component_rewards: torch.Tensor
    bootstrap_values: torch.Tensor


@dataclass
class MultiAdaptiveTrajectoryRollout:
    """包含可变长度宏动作与四分量回报的 imagined trajectory。

    Attributes:
        steps (list[MultiAdaptiveTrajectoryStep]): 逐 chunk 决策记录。
        trajectory_utility (torch.Tensor): 按冻结尺度和权重合成的折扣效用。
        executed_chunks (torch.Tensor): 每行实际进入 rollout 的 chunk 数。
    """

    steps: list[MultiAdaptiveTrajectoryStep]
    trajectory_utility: torch.Tensor
    executed_chunks: torch.Tensor


@dataclass
class MultiAllPrefixOneStepTarget:
    """同一候选所有合法前缀的 R/T/D/C 一步 target-Q 标签。

    Attributes:
        values (torch.Tensor): 一步 target-Q，形状 ``(B,K,4)``。
        valid (torch.Tensor): 合法且起点活跃的前缀 mask，形状 ``(B,K)``。
        rewards (torch.Tensor): 各前缀自身的四分量 chunk 回报，
            形状 ``(B,K,4)``。
        effective_steps (torch.Tensor): 各前缀实际执行步数，形状 ``(B,K)``。
        done (torch.Tensor): 各前缀末端终止标记，形状 ``(B,K)``。
    """

    values: torch.Tensor
    valid: torch.Tensor
    rewards: torch.Tensor
    effective_steps: torch.Tensor
    done: torch.Tensor


class MultiPrefixTransformerAgent(ACSACTransformerAgent):
    """完全复用 ACSAC 训练骨架的 R/T/D/C 四头前缀 Q agent。

    Actor、Dynamics、DORL shaped R、退出机制、离线任意起点和 fresh imagined
    rollout 均与单头 :class:`ACSACTransformerAgent` 一致。不同之处仅为：critic
    输出 R/T/D/C 四个长期价值；选择和 bootstrap 使用冻结综合效用的同一个
    ``(candidate,H)``；训练标签额外从实际展示与完整会话 MCD 记账构造 T/D/C。
    """

    def __init__(
        self,
        *args: object,
        critic_transformer_hidden_dim: int = DEFAULT_CRITIC_TRANSFORMER_HIDDEN_DIM,
        critic_transformer_layers: int = DEFAULT_CRITIC_TRANSFORMER_LAYERS,
        critic_transformer_heads: int = DEFAULT_CRITIC_TRANSFORMER_HEADS,
        critic_transformer_dropout: float = DEFAULT_CRITIC_TRANSFORMER_DROPOUT,
        all_prefix_aux_weight: float = DEFAULT_ALL_PREFIX_AUX_WEIGHT,
        utility: Optional[Utility] = None,
        dominant_hits: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> None:
        """初始化共享 Actor/Dynamics 和 online/target 四头 Q。

        Args:
            *args (object): 传给单头 ACSAC agent 的位置参数。
            critic_transformer_hidden_dim (int): Transformer token 维度。
            critic_transformer_layers (int): Transformer 编码层数。
            critic_transformer_heads (int): Transformer attention head 数。
            critic_transformer_dropout (float): dropout 概率。
            all_prefix_aux_weight (float): 全合法前缀一步辅助 MSE 权重。
            utility (Optional[Utility]): 冻结 R/T/D/C 尺度与业务权重；为空时
                使用只供 checkpoint 加载前占位的默认值。
            dominant_hits (Optional[torch.Tensor]): 每个内部 item 是否命中冻结
                主导类别，形状 ``(num_items,)``；为空时初始化为全假。
            **kwargs (object): 传给单头 ACSAC agent 的其余参数。

        Raises:
            ValueError: 当主导类别命中表形状或数值不合法时抛出。
        """

        super().__init__(
            *args,
            critic_transformer_hidden_dim=critic_transformer_hidden_dim,
            critic_transformer_layers=critic_transformer_layers,
            critic_transformer_heads=critic_transformer_heads,
            critic_transformer_dropout=critic_transformer_dropout,
            all_prefix_aux_weight=all_prefix_aux_weight,
            **kwargs,
        )
        self.critic = MultiHeadCausalTransformerPrefixCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            hidden_dim=critic_transformer_hidden_dim,
            num_layers=critic_transformer_layers,
            num_heads=critic_transformer_heads,
            dropout=critic_transformer_dropout,
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic.requires_grad_(False)
        self.critic_backbone = CRITIC_BACKBONE_MULTI_PREFIX_TRANSFORMER
        self.utility = utility or Utility(
            scales=DEFAULT_MULTI_SCALES,
            weights=DEFAULT_MULTI_WEIGHTS,
        )
        resolved_hits = (
            torch.zeros(self.num_items, dtype=torch.bool)
            if dominant_hits is None
            else torch.as_tensor(dominant_hits, dtype=torch.bool)
        )
        if resolved_hits.shape != (self.num_items,):
            raise ValueError(
                "dominant_hits must have shape "
                f"({self.num_items},), got {tuple(resolved_hits.shape)}."
            )
        self.register_buffer(
            "dominant_hits",
            resolved_hits.to(self.device),
        )

    def _score_components(self, components: torch.Tensor) -> torch.Tensor:
        """按冻结尺度和业务权重合成 R/T/D/C 长期价值。

        Args:
            components (torch.Tensor): 末维必须为四的价值张量。

        Returns:
            torch.Tensor: 移除末维后的综合效用分数。

        Raises:
            ValueError: 当输入末维不是 R/T/D/C 四分量时抛出。
        """

        if components.shape[-1] != NUM_COMPONENTS:
            raise ValueError("components must end with four R/T/D/C values.")
        return self.utility.scalarize(components)

    @torch.no_grad()
    def select_adaptive_chunks(
        self,
        states: torch.Tensor,
        num_samples: int,
        recommended_mask: Optional[torch.Tensor] = None,
        execution_budgets: Optional[torch.Tensor] = None,
        critic: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """用四头综合分数联合选择唯一候选和合法执行长度。

        Args:
            states (torch.Tensor): 状态，形状 ``(B,state_dim)``。
            num_samples (int): 每个状态的候选 chunk 数。
            recommended_mask (Optional[torch.Tensor]): 历史 item 屏蔽矩阵。
            execution_budgets (Optional[torch.Tensor]): 每行最大合法 H。
            critic (Optional[nn.Module]): 可选打分 critic；为空使用 online Q。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 完整候选 item、候选
            embedding 和执行长度，形状为 ``(B,K)``、``(B,K*A)``、``(B,)``。

        Raises:
            ValueError: 当候选数或执行预算不合法时抛出。
        """

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        states = states.to(device=self.device, dtype=torch.float32)
        candidate_ids, candidate_chunks = self.sample_candidate_chunks(
            states,
            num_samples=num_samples,
            recommended_mask=recommended_mask,
        )
        num_candidates, batch_size, _ = candidate_chunks.shape
        repeated_states = states.unsqueeze(0).expand(num_candidates, -1, -1)
        scoring_critic = self.critic if critic is None else critic
        q_components = scoring_critic(
            repeated_states.reshape(-1, self.state_dim),
            candidate_chunks.reshape(-1, self.chunk_action_dim),
        ).reshape(
            num_candidates,
            batch_size,
            self.chunk_size,
            NUM_COMPONENTS,
        )
        scores = self._score_components(q_components)
        if execution_budgets is None:
            budgets = torch.full(
                (batch_size,),
                self.chunk_size,
                dtype=torch.long,
                device=self.device,
            )
        else:
            budgets = execution_budgets.to(
                device=self.device,
                dtype=torch.long,
            ).reshape(-1)
            if budgets.shape[0] != batch_size:
                raise ValueError("execution_budgets must have shape (B,).")
            if torch.any(budgets < 1) or torch.any(budgets > self.chunk_size):
                raise ValueError("execution_budgets must lie in [1, chunk_size].")
        prefix_lengths = torch.arange(1, self.chunk_size + 1, device=self.device)
        scores = scores.masked_fill(
            prefix_lengths.view(1, 1, -1) > budgets.view(1, -1, 1),
            -torch.inf,
        )
        flat_indices = scores.permute(1, 0, 2).reshape(batch_size, -1).argmax(1)
        candidate_indices = torch.div(
            flat_indices,
            self.chunk_size,
            rounding_mode="floor",
        )
        selected_lengths = flat_indices.remainder(self.chunk_size) + 1
        row_indices = torch.arange(batch_size, device=self.device)
        return (
            candidate_ids[candidate_indices, row_indices],
            candidate_chunks[candidate_indices, row_indices],
            selected_lengths,
        )

    @torch.no_grad()
    def _target_state_values(
        self,
        history_state: RolloutHistoryState,
        num_samples: int,
        repeat_policy: str,
    ) -> torch.Tensor:
        """在 target Q 上用同一个综合 argmax 返回四分量续接价值。

        Args:
            history_state (RolloutHistoryState): 前缀末端历史状态。
            num_samples (int): 每个末端状态重新采样的候选数。
            repeat_policy (str): 重复推荐处理策略。

        Returns:
            torch.Tensor: 同一被选续接动作的 R/T/D/C target Q，形状 ``(B,4)``。
        """

        terminal = history_state.terminated.clone()
        candidate_mask: Optional[torch.Tensor] = None
        if repeat_policy == REPEAT_POLICY_MASK:
            exhausted = history_state.recommended_mask.all(dim=1)
            terminal = terminal | exhausted
            candidate_mask = history_state.recommended_mask.clone()
            candidate_mask[terminal] = False
        candidate_ids, candidate_chunks = self.sample_candidate_chunks(
            history_state.states,
            num_samples=num_samples,
            recommended_mask=candidate_mask,
        )
        del candidate_ids
        num_candidates, batch_size, _ = candidate_chunks.shape
        repeated_states = history_state.states.unsqueeze(0).expand(
            num_candidates,
            -1,
            -1,
        )
        q_components = self.target_critic(
            repeated_states.reshape(-1, self.state_dim),
            candidate_chunks.reshape(-1, self.chunk_action_dim),
        ).reshape(
            num_candidates,
            batch_size,
            self.chunk_size,
            NUM_COMPONENTS,
        )
        scores = self._score_components(q_components)
        budgets = self._execution_budgets(history_state)
        prefix_lengths = torch.arange(1, self.chunk_size + 1, device=self.device)
        scores = scores.masked_fill(
            prefix_lengths.view(1, 1, -1) > budgets.view(1, -1, 1),
            -torch.inf,
        )
        flat_scores = scores.permute(1, 0, 2).reshape(batch_size, -1)
        flat_indices = flat_scores.argmax(dim=1)
        flattened_components = q_components.permute(1, 0, 2, 3).reshape(
            batch_size,
            -1,
            NUM_COMPONENTS,
        )
        row_indices = torch.arange(batch_size, device=self.device)
        selected_components = flattened_components[row_indices, flat_indices]
        return torch.where(
            terminal.unsqueeze(1),
            torch.zeros_like(selected_components),
            selected_components,
        )

    def _initial_component_counts(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从离线任意起点的完整会话历史恢复 MCD 的 n/k 计数。

        Args:
            batch (Dict[str, torch.Tensor]): 必须包含 padding 为 -1 的
                ``leave_history_item_ids``，其宽度由 runner 设为 ``max_turn``。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 已展示 item 数 n 和主导类别命中数 k，
            均为形状 ``(B,)`` 的 long tensor。

        Raises:
            ValueError: 当历史 item ID 超出内部目录时抛出。
        """

        history_ids = self._batch_tensor(batch, "leave_history_item_ids").long()
        valid = history_ids >= 0
        if torch.any(history_ids[valid] >= self.num_items):
            raise ValueError("leave history contains an item outside the catalog.")
        safe_ids = history_ids.clamp_min(0)
        item_counts = valid.long().sum(dim=1)
        dominant_counts = (
            self.dominant_hits[safe_ids].long() * valid.long()
        ).sum(dim=1)
        return item_counts, dominant_counts

    def _component_chunk_rewards(
        self,
        rollout: RolloutResult,
        start_item_counts: torch.Tensor,
        start_dominant_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """按实际执行序列构造 R/T/D/C chunk 回报并更新 MCD 计数。

        R 直接复用当前 ACSAC/DORL 的逐步 reward；T 每个实际展示为 1；D 为
        ``F_{t+1}-F_t``；C 仅在当前新 chunk 的首次实际展示为 1。四分量均按
        环境步 gamma 折扣后在 chunk 内求和。

        Args:
            rollout (RolloutResult): 当前 chunk 的逐步执行结果。
            start_item_counts (torch.Tensor): chunk 起点完整会话展示数 n。
            start_dominant_counts (torch.Tensor): chunk 起点主导类别命中数 k。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(B,4)`` chunk 回报、
            chunk 末 n 和 chunk 末 k。

        Raises:
            ValueError: 当计数形状与 rollout batch 不一致时抛出。
        """

        batch_size = rollout.step_rewards.shape[0]
        expected_shape = (batch_size,)
        if (
            tuple(start_item_counts.shape) != expected_shape
            or tuple(start_dominant_counts.shape) != expected_shape
        ):
            raise ValueError("MCD count tensors must have shape (B,).")
        item_counts = start_item_counts.clone()
        dominant_counts = start_dominant_counts.clone()
        diversity_rewards = rollout.step_rewards.new_zeros(
            batch_size,
            self.chunk_size,
        )
        safe_ids = rollout.selected_item_ids.clamp_min(0)
        for position in range(self.chunk_size):
            valid = rollout.chunk_valid[:, position]
            previous_fraction = torch.where(
                item_counts > 0,
                1.0 - dominant_counts.float() / item_counts.clamp_min(1).float(),
                torch.zeros_like(item_counts, dtype=torch.float32),
            )
            next_item_counts = item_counts + valid.long()
            next_dominant_counts = dominant_counts + (
                self.dominant_hits[safe_ids[:, position]].long() * valid.long()
            )
            next_fraction = torch.where(
                next_item_counts > 0,
                1.0
                - next_dominant_counts.float()
                / next_item_counts.clamp_min(1).float(),
                torch.zeros_like(next_item_counts, dtype=torch.float32),
            )
            diversity_rewards[:, position] = torch.where(
                valid,
                next_fraction - previous_fraction,
                torch.zeros_like(next_fraction),
            )
            item_counts = next_item_counts
            dominant_counts = next_dominant_counts

        valid_float = rollout.chunk_valid.to(dtype=rollout.step_rewards.dtype)
        length_rewards = valid_float
        cost_rewards = torch.zeros_like(valid_float)
        cost_rewards[:, 0] = valid_float[:, 0]
        step_components = torch.stack(
            (
                rollout.step_rewards,
                length_rewards,
                diversity_rewards,
                cost_rewards,
            ),
            dim=-1,
        )
        step_discounts = torch.as_tensor(
            self.gamma,
            dtype=step_components.dtype,
            device=self.device,
        ).pow(
            torch.arange(
                self.chunk_size,
                dtype=step_components.dtype,
                device=self.device,
            )
        )
        component_returns = (
            step_components * step_discounts.view(1, -1, 1)
        ).sum(dim=1)
        return component_returns, item_counts, dominant_counts

    @torch.no_grad()
    def rollout_adaptive_trajectory(
        self,
        batch: Dict[str, torch.Tensor],
        num_samples: int,
        rollout_depth: int,
        repeat_policy: str,
        leave_policy: str,
    ) -> MultiAdaptiveTrajectoryRollout:
        """生成与单头 ACSAC 相同路径、但携带四分量回报的 imagined rollout。

        Args:
            batch (Dict[str, torch.Tensor]): 离线任意起点训练 batch。
            num_samples (int): 每个 chunk 边界的候选数。
            rollout_depth (int): imagined rollout 的 chunk 深度。
            repeat_policy (str): 重复推荐处理策略。
            leave_policy (str): 退出机制处理策略。

        Returns:
            MultiAdaptiveTrajectoryRollout: 四分量逐 chunk 轨迹及统计。

        Raises:
            ValueError: 当 rollout 深度或重复策略非法时抛出。
        """

        if rollout_depth <= 0:
            raise ValueError("rollout_depth must be positive.")
        if repeat_policy not in {REPEAT_POLICY_TRUNCATE, REPEAT_POLICY_MASK}:
            raise ValueError(f"Unsupported repeat_policy: {repeat_policy}")
        self._batch_user_ids = self._batch_tensor(batch, "user_id").long().view(-1)
        history_state = self._initial_history_state(batch)
        if repeat_policy == REPEAT_POLICY_MASK:
            self._reset_local_rollout_recommended_mask(history_state)
        item_counts, dominant_counts = self._initial_component_counts(batch)
        batch_size = int(history_state.states.shape[0])
        steps: list[MultiAdaptiveTrajectoryStep] = []
        trajectory_utility = torch.zeros(batch_size, device=self.device)
        executed_chunks = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=self.device,
        )
        cumulative_discount = torch.ones(batch_size, device=self.device)
        for _ in range(rollout_depth):
            if repeat_policy == REPEAT_POLICY_MASK:
                active, candidate_mask = self._prepare_rollout_sampling_mask(
                    history_state,
                )
            else:
                active = ~history_state.terminated
                candidate_mask = None
            states = history_state.states
            selected_ids, selected_chunks, selected_lengths = (
                self.select_adaptive_chunks(
                    states,
                    num_samples=num_samples,
                    recommended_mask=candidate_mask,
                    execution_budgets=self._execution_budgets(history_state),
                )
            )
            result, next_history_state = self._rollout_one_chunk(
                selected_ids,
                history_state,
                repeat_policy=repeat_policy,
                leave_policy=leave_policy,
                execution_lengths=selected_lengths,
            )
            component_rewards, item_counts, dominant_counts = (
                self._component_chunk_rewards(
                    result,
                    item_counts,
                    dominant_counts,
                )
            )
            chunk_discount = self.gamma ** result.effective_steps.float()
            bootstrap_values = self._target_state_values(
                next_history_state,
                num_samples=num_samples,
                repeat_policy=repeat_policy,
            )
            steps.append(MultiAdaptiveTrajectoryStep(
                states=states,
                selected_chunks=selected_chunks.detach(),
                selected_item_ids=selected_ids.detach(),
                selected_lengths=selected_lengths.detach(),
                rollout=result,
                active=active,
                chunk_discount=chunk_discount,
                component_rewards=component_rewards,
                bootstrap_values=bootstrap_values,
            ))
            trajectory_utility += (
                cumulative_discount
                * self._score_components(component_rewards)
                * active.float()
            )
            executed_chunks += active.long()
            cumulative_discount *= chunk_discount
            history_state = next_history_state
        return MultiAdaptiveTrajectoryRollout(
            steps=steps,
            trajectory_utility=trajectory_utility,
            executed_chunks=executed_chunks,
        )

    def _td_lambda_target(
        self,
        trajectory: MultiAdaptiveTrajectoryRollout,
        lambda_chunk: float,
    ) -> torch.Tensor:
        """为 selected-H 的四个头构造共同策略下完整 chunk TD(lambda)。

        Args:
            trajectory (MultiAdaptiveTrajectoryRollout): fresh imagined trajectory。
            lambda_chunk (float): chunk TD(lambda) 系数，范围 ``[0,1]``。

        Returns:
            torch.Tensor: 首个 selected-H 的 R/T/D/C 标签，形状 ``(B,4)``。

        Raises:
            ValueError: 当 lambda 不在 ``[0,1]`` 时抛出。
        """

        if not 0.0 <= lambda_chunk <= 1.0:
            raise ValueError("lambda_chunk must be in [0, 1].")
        continuation = trajectory.steps[-1].bootstrap_values
        for step in reversed(trajectory.steps):
            not_done = 1.0 - step.rollout.done_chunks.float()
            mixed_bootstrap = (
                (1.0 - lambda_chunk) * step.bootstrap_values
                + lambda_chunk * continuation
            )
            continuation = step.component_rewards + (
                not_done.unsqueeze(1)
                * step.chunk_discount.unsqueeze(1)
                * mixed_bootstrap
            )
        return continuation.detach()

    @torch.no_grad()
    def _all_prefix_one_step_target(
        self,
        batch: Dict[str, torch.Tensor],
        first_step: MultiAdaptiveTrajectoryStep,
        num_samples: int,
        repeat_policy: str,
        leave_policy: str,
    ) -> MultiAllPrefixOneStepTarget:
        """向量化构造所有合法 H 的独立 R/T/D/C 一步 target-Q。

        Args:
            batch (Dict[str, torch.Tensor]): 离线任意起点 batch。
            first_step (MultiAdaptiveTrajectoryStep): 主 rollout 的首个已选候选。
            num_samples (int): 每个前缀末端的 target-Q 候选数。
            repeat_policy (str): 重复推荐处理策略。
            leave_policy (str): 退出机制处理策略。

        Returns:
            MultiAllPrefixOneStepTarget: ``(B,K,4)`` 标签和合法 mask。

        Raises:
            ValueError: 当候选数、策略或首步张量形状非法时抛出。
        """

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if repeat_policy not in {REPEAT_POLICY_TRUNCATE, REPEAT_POLICY_MASK}:
            raise ValueError(f"Unsupported repeat_policy: {repeat_policy}")
        history_state = self._initial_history_state(batch)
        if repeat_policy == REPEAT_POLICY_MASK:
            self._reset_local_rollout_recommended_mask(history_state)
            active, _ = self._prepare_rollout_sampling_mask(history_state)
        else:
            active = ~history_state.terminated
        batch_size = int(history_state.states.shape[0])
        expected_shape = (batch_size, self.chunk_size)
        if tuple(first_step.selected_item_ids.shape) != expected_shape:
            raise ValueError(
                "selected_item_ids shape mismatch: expected "
                f"{expected_shape}, got {tuple(first_step.selected_item_ids.shape)}."
            )
        prefix_lengths = torch.arange(
            1,
            self.chunk_size + 1,
            dtype=torch.long,
            device=self.device,
        )
        budgets = self._execution_budgets(history_state)
        valid = active.unsqueeze(1) & (
            prefix_lengths.unsqueeze(0) <= budgets.unsqueeze(1)
        )
        expanded_history = self._repeat_history_state(
            history_state,
            repeats=self.chunk_size,
        )
        expanded_item_ids = first_step.selected_item_ids.repeat_interleave(
            self.chunk_size,
            dim=0,
        )
        expanded_lengths = prefix_lengths.repeat(batch_size)
        item_counts, dominant_counts = self._initial_component_counts(batch)
        item_counts = item_counts.repeat_interleave(self.chunk_size)
        dominant_counts = dominant_counts.repeat_interleave(self.chunk_size)

        original_user_ids = self._batch_user_ids
        batch_user_ids = self._batch_tensor(batch, "user_id").long().view(-1)
        self._batch_user_ids = batch_user_ids.repeat_interleave(self.chunk_size)
        try:
            prefix_rollout, next_history = self._rollout_one_chunk(
                expanded_item_ids,
                expanded_history,
                repeat_policy=repeat_policy,
                leave_policy=leave_policy,
                execution_lengths=expanded_lengths,
            )
            component_rewards, _, _ = self._component_chunk_rewards(
                prefix_rollout,
                item_counts,
                dominant_counts,
            )
            next_values = self._target_state_values(
                next_history,
                num_samples=num_samples,
                repeat_policy=repeat_policy,
            )
        finally:
            self._batch_user_ids = original_user_ids
        chunk_discount = self.gamma ** prefix_rollout.effective_steps.float()
        flat_targets = component_rewards + (
            (1.0 - prefix_rollout.done_chunks.float()).unsqueeze(1)
            * chunk_discount.unsqueeze(1)
            * next_values
        )
        return MultiAllPrefixOneStepTarget(
            values=flat_targets.reshape(
                batch_size,
                self.chunk_size,
                NUM_COMPONENTS,
            ).detach(),
            valid=valid,
            rewards=component_rewards.reshape(
                batch_size,
                self.chunk_size,
                NUM_COMPONENTS,
            ),
            effective_steps=prefix_rollout.effective_steps.reshape(
                batch_size,
                self.chunk_size,
            ),
            done=prefix_rollout.done_chunks.reshape(batch_size, self.chunk_size),
        )

    def _scaled_mse(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """对四头残差按冻结尺度归一化后计算等权 MSE 和。

        Args:
            predictions (torch.Tensor): 末维为四的 online Q。
            targets (torch.Tensor): 与预测同形状的停止梯度标签。

        Returns:
            torch.Tensor: 先对四头求和、再对样本求平均的标量损失。

        Raises:
            ValueError: 当预测和目标形状不同或末维不是四时抛出。
        """

        if predictions.shape != targets.shape or predictions.shape[-1] != NUM_COMPONENTS:
            raise ValueError("predictions and targets must share shape (...,4).")
        scales = predictions.new_tensor(self.utility.scales)
        return ((predictions - targets) / scales).square().sum(dim=-1).mean()

    def qv_update(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        num_samples_train: int,
        repeat_policy: str = REPEAT_POLICY_TRUNCATE,
        rollout_depth: int = 1,
        lambda_chunk: float = 1.0,
        leave_policy: str = LEAVE_POLICY_PENALTY,
    ) -> Dict[str, float]:
        """执行四头 selected-H TD(lambda) 与全前缀一步 TD 联合更新。

        Args:
            batch (Dict[str, torch.Tensor]): 离线任意起点训练 batch。
            optimizer (torch.optim.Optimizer): online Q 与可选 Dynamics 优化器。
            num_samples_train (int): 每个状态的候选 chunk 数。
            repeat_policy (str): 重复推荐处理策略。
            rollout_depth (int): selected-H imagined rollout chunk 深度。
            lambda_chunk (float): selected-H chunk TD(lambda) 系数。
            leave_policy (str): 退出机制处理策略。

        Returns:
            Dict[str, float]: 四头 loss、Q、目标和 rollout 统计。

        Raises:
            RuntimeError: 当 rollout 依赖未配置时抛出。
            ValueError: 当候选数、深度或 lambda 非法时抛出。
        """

        if self.dynamics is None or self.reward_model is None or self.leave_model is None:
            raise RuntimeError("qv_update requires dynamics, reward_model and leave_model.")
        if rollout_depth <= 0:
            raise ValueError("rollout_depth must be positive.")
        if num_samples_train <= 0:
            raise ValueError("num_samples_train must be positive.")
        if not 0.0 <= lambda_chunk <= 1.0:
            raise ValueError("lambda_chunk must be in [0, 1].")
        self.train()
        self.target_critic.eval()
        if not any(parameter.requires_grad for parameter in self.dynamics.parameters()):
            self.dynamics.eval()
        with torch.no_grad():
            trajectory = self.rollout_adaptive_trajectory(
                batch=batch,
                num_samples=num_samples_train,
                rollout_depth=rollout_depth,
                repeat_policy=repeat_policy,
                leave_policy=leave_policy,
            )
            target_components = self._td_lambda_target(
                trajectory,
                lambda_chunk,
            )
            first_step = trajectory.steps[0]
            all_prefix_target = None
            if self.all_prefix_aux_weight > 0.0:
                all_prefix_target = self._all_prefix_one_step_target(
                    batch=batch,
                    first_step=first_step,
                    num_samples=num_samples_train,
                    repeat_policy=repeat_policy,
                    leave_policy=leave_policy,
                )
        optimizer.zero_grad(set_to_none=True)
        prefix_q_components = self.critic(
            first_step.states,
            first_step.selected_chunks,
        )
        gather_indices = (
            first_step.selected_lengths.view(-1, 1, 1)
            .sub(1)
            .expand(-1, 1, NUM_COMPONENTS)
        )
        selected_q_components = prefix_q_components.gather(
            dim=1,
            index=gather_indices,
        ).squeeze(1)
        selected_td_lambda_loss = self._scaled_mse(
            selected_q_components,
            target_components,
        )
        if all_prefix_target is None:
            all_prefix_aux_loss = prefix_q_components.sum() * 0.0
            valid_prefix_q = prefix_q_components.new_empty((0, NUM_COMPONENTS))
            valid_prefix_targets = prefix_q_components.new_empty(
                (0, NUM_COMPONENTS),
            )
            legal_prefixes_mean = 0.0
        else:
            valid_prefix_q = prefix_q_components[all_prefix_target.valid]
            valid_prefix_targets = all_prefix_target.values[all_prefix_target.valid]
            if valid_prefix_q.numel() == 0:
                all_prefix_aux_loss = prefix_q_components.sum() * 0.0
            else:
                all_prefix_aux_loss = self._scaled_mse(
                    valid_prefix_q,
                    valid_prefix_targets,
                )
            legal_prefixes_mean = float(
                all_prefix_target.valid.sum(dim=1).float().mean().cpu()
            )
        critic_loss = selected_td_lambda_loss + (
            self.all_prefix_aux_weight * all_prefix_aux_loss
        )
        dynamics_loss = self.supervised_dynamics_loss(batch)
        loss = critic_loss + self.dynamics_loss_weight * dynamics_loss
        loss.backward()
        optimizer.step()
        self.soft_update_target_critic()

        selected_scores = self._score_components(selected_q_components)
        target_scores = self._score_components(target_components)
        first_rollout = first_step.rollout
        metrics = {
            "critic/critic_loss": float(critic_loss.detach().cpu()),
            "critic/selected_td_lambda_loss": float(
                selected_td_lambda_loss.detach().cpu()
            ),
            "critic/all_prefix_aux_loss": float(all_prefix_aux_loss.detach().cpu()),
            "critic/all_prefix_aux_weight": self.all_prefix_aux_weight,
            "critic/q_mean": float(selected_scores.detach().mean().cpu()),
            "critic/target_q_mean": float(target_scores.detach().mean().cpu()),
            "dynamics/loss": float(dynamics_loss.detach().cpu()),
            "rollout/depth": float(rollout_depth),
            "rollout/lambda_chunk": float(lambda_chunk),
            "rollout/trajectory_reward": float(
                trajectory.trajectory_utility.detach().mean().cpu()
            ),
            "rollout/trajectory_chunks": float(
                trajectory.executed_chunks.float().mean().cpu()
            ),
            "rollout/reward_chunk": float(
                first_step.component_rewards[:, 0].detach().mean().cpu()
            ),
            "rollout/pred_reward": self._masked_mean(
                first_rollout.pred_rewards,
                first_rollout.chunk_valid,
            ),
            "rollout/entropy": self._masked_mean(
                first_rollout.entropy_rewards,
                first_rollout.chunk_valid,
            ),
            "rollout/uncertainty": self._masked_mean(
                first_rollout.uncertainty_penalties,
                first_rollout.chunk_valid,
            ),
            "rollout/effective_steps": float(
                first_rollout.effective_steps.float().mean().cpu()
            ),
            "rollout/selected_h_mean": float(
                first_step.selected_lengths.float().mean().cpu()
            ),
            "rollout/legal_prefixes_mean": legal_prefixes_mean,
            "rollout/done_ratio": float(
                first_rollout.done_chunks.float().mean().cpu()
            ),
            "rollout/repeat_ratio": float(first_rollout.repeat_ratio.cpu()),
            "rollout/exact_repeat_ratio": float(
                first_rollout.exact_repeat_ratio.cpu()
            ),
        }
        scales = selected_q_components.new_tensor(self.utility.scales)
        selected_residual = (selected_q_components - target_components) / scales
        for component_index, component_name in enumerate(COMPONENTS):
            metrics[f"critic/q_mean/{component_name}"] = float(
                selected_q_components[:, component_index].detach().mean().cpu()
            )
            metrics[f"critic/target_q_mean/{component_name}"] = float(
                target_components[:, component_index].detach().mean().cpu()
            )
            metrics[f"critic/selected_td_lambda_loss/{component_name}"] = float(
                selected_residual[:, component_index].detach().square().mean().cpu()
            )
            if valid_prefix_q.numel() > 0:
                aux_residual = (
                    valid_prefix_q[:, component_index]
                    - valid_prefix_targets[:, component_index]
                ) / scales[component_index]
                metrics[f"critic/all_prefix_aux_loss/{component_name}"] = float(
                    aux_residual.detach().square().mean().cpu()
                )
            else:
                metrics[f"critic/all_prefix_aux_loss/{component_name}"] = 0.0
        for length in range(1, self.chunk_size + 1):
            metrics[f"rollout/selected_h_{length}_ratio"] = float(
                (first_step.selected_lengths == length).float().mean().cpu()
            )
        return metrics

    def checkpoint_state(self, config: Dict[str, object]) -> Dict[str, object]:
        """构造含四头 Q、冻结效用和主导类别命中表的 checkpoint。

        Args:
            config (Dict[str, object]): 完整训练配置快照。

        Returns:
            Dict[str, object]: 可供训练恢复和独立评估使用的 checkpoint。
        """

        return {
            "format": MULTI_PREFIX_CHECKPOINT_FORMAT,
            "actor_type": self.actor_type,
            "critic_backbone": self.critic_backbone,
            "config": dict(config),
            "observation_spec": getattr(self, "observation_spec", None),
            "actor": self.actor.state_dict(),
            "actor_onestep": (
                self.actor_onestep.state_dict()
                if self.actor_onestep is not None else None
            ),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "dynamics": self.dynamics.state_dict() if self.dynamics is not None else None,
            "utility": {
                "scales": tuple(self.utility.scales),
                "weights": tuple(self.utility.weights),
            },
            "dominant_hits": self.dominant_hits.detach().cpu(),
        }

    def load_checkpoint_state(
        self,
        checkpoint: Dict[str, object],
        strict: bool = True,
        load_qv: bool = True,
    ) -> None:
        """加载共享 BC checkpoint 或四头完整 checkpoint。

        Args:
            checkpoint (Dict[str, object]): BC 或四头 checkpoint。
            strict (bool): 是否严格匹配参数名称和形状。
            load_qv (bool): 是否加载 online/target 四头 Q。

        Returns:
            None.

        Raises:
            ValueError: 当要求加载 Q/V 时 checkpoint 格式或多头资产不兼容。
        """

        if checkpoint.get("format") != MULTI_PREFIX_CHECKPOINT_FORMAT:
            if load_qv:
                raise ValueError(
                    "Multi-prefix Q loading requires a multi-prefix checkpoint."
                )
            super().load_checkpoint_state(
                checkpoint,
                strict=strict,
                load_qv=False,
            )
            return
        base_checkpoint = dict(checkpoint)
        base_checkpoint["format"] = "mac_origin_precomputed_state_v2"
        base_checkpoint["critic_backbone"] = "mlp"
        super().load_checkpoint_state(
            base_checkpoint,
            strict=strict,
            load_qv=False,
        )
        if load_qv:
            utility_state = checkpoint.get("utility")
            dominant_hits = checkpoint.get("dominant_hits")
            if not isinstance(utility_state, dict) or dominant_hits is None:
                raise ValueError("Multi-prefix checkpoint is missing utility assets.")
            restored_hits = torch.as_tensor(
                dominant_hits,
                dtype=torch.bool,
                device=self.device,
            )
            if restored_hits.shape != self.dominant_hits.shape:
                raise ValueError("Checkpoint dominant_hits shape is incompatible.")
            self.utility = Utility(**utility_state)
            self.dominant_hits.copy_(restored_hits)
            self.critic.load_state_dict(checkpoint["critic"], strict=strict)
            target_state = checkpoint.get("target_critic", checkpoint["critic"])
            self.target_critic.load_state_dict(target_state, strict=strict)
