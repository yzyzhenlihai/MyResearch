"""在 ordinary imagined-rollout 流程中使用 ACSAC 因果前缀 Q。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from examples.our_model.models.chunk_value import (
    CRITIC_BACKBONE_ACSAC_TRANSFORMER,
    CRITIC_BACKBONE_MLP,
    DEFAULT_CRITIC_TRANSFORMER_DROPOUT,
    DEFAULT_CRITIC_TRANSFORMER_HEADS,
    DEFAULT_CRITIC_TRANSFORMER_HIDDEN_DIM,
    DEFAULT_CRITIC_TRANSFORMER_LAYERS,
    CausalTransformerPrefixCritic,
)
from examples.our_model.models.mac_agent import (
    LEAVE_POLICY_PENALTY,
    REPEAT_POLICY_MASK,
    REPEAT_POLICY_TRUNCATE,
    MACAgent,
    RolloutHistoryState,
    RolloutResult,
)

ACSAC_CHECKPOINT_FORMAT = "model_based_acsac_transformer_v1"
"""模型化 ACSAC Transformer 分支的 checkpoint 格式。"""

DEFAULT_ALL_PREFIX_AUX_WEIGHT = 1.0
"""所有合法前缀一步 target-Q 辅助损失的默认权重。"""


@dataclass
class AdaptiveTrajectoryStep:
    """自适应 imagined rollout 的一个宏动作决策。"""

    states: torch.Tensor
    selected_chunks: torch.Tensor
    selected_item_ids: torch.Tensor
    selected_lengths: torch.Tensor
    rollout: RolloutResult
    active: torch.Tensor
    chunk_discount: torch.Tensor
    bootstrap_values: torch.Tensor


@dataclass
class AdaptiveTrajectoryRollout:
    """包含可变长度宏动作的 imagined trajectory。"""

    steps: list[AdaptiveTrajectoryStep]
    trajectory_returns: torch.Tensor
    executed_chunks: torch.Tensor


@dataclass
class AllPrefixOneStepTarget:
    """同一候选 chunk 的全部合法前缀一步 TD 目标。

    Attributes:
        values (torch.Tensor): 每个前缀的一步 target-Q 目标，形状 ``(B,K)``。
        valid (torch.Tensor): 决策时合法且样本未终止的前缀标记，形状 ``(B,K)``。
        rewards (torch.Tensor): 每个前缀的折扣累计 chunk reward，形状 ``(B,K)``。
        effective_steps (torch.Tensor): 各前缀实际执行的底层步数，形状 ``(B,K)``。
        done (torch.Tensor): 各前缀执行后的终止标记，形状 ``(B,K)``。
    """

    values: torch.Tensor
    valid: torch.Tensor
    rewards: torch.Tensor
    effective_steps: torch.Tensor
    done: torch.Tensor


class ACSACTransformerAgent(MACAgent):
    """保留 ordinary 流程、仅替换 Q/V 与 chunk 选择的独立 agent。

    Actor、Dynamics、reward、退出规则和任意离线起点 batch 均复用
    :class:`MACAgent`。本类删除独立 V，以在线因果 Transformer Q 联合选择
    候选 chunk 与执行长度，并用 target Q 的候选最大值 bootstrap。
    """

    adaptive_chunk_selection = True
    """供评估适配器识别学习得到的逐样本执行长度。"""

    def __init__(
        self,
        *args: object,
        critic_transformer_hidden_dim: int = DEFAULT_CRITIC_TRANSFORMER_HIDDEN_DIM,
        critic_transformer_layers: int = DEFAULT_CRITIC_TRANSFORMER_LAYERS,
        critic_transformer_heads: int = DEFAULT_CRITIC_TRANSFORMER_HEADS,
        critic_transformer_dropout: float = DEFAULT_CRITIC_TRANSFORMER_DROPOUT,
        all_prefix_aux_weight: float = DEFAULT_ALL_PREFIX_AUX_WEIGHT,
        **kwargs: object,
    ) -> None:
        """初始化共享 Actor/Dynamics 与 ACSAC online/target Q。

        Args:
            *args (object): 传给 ``MACAgent`` 的位置参数。
            critic_transformer_hidden_dim (int): Transformer token 维度。
            critic_transformer_layers (int): Transformer 层数。
            critic_transformer_heads (int): 注意力头数。
            critic_transformer_dropout (float): dropout 概率。
            all_prefix_aux_weight (float): 所有合法前缀一步 target-Q 辅助损失
                的非负权重；设为 0 时关闭辅助目标并跳过额外 rollout。
            **kwargs (object): 传给 ``MACAgent`` 的其余参数。

        Raises:
            ValueError: 当 ``all_prefix_aux_weight`` 为负数时抛出。
        """

        if all_prefix_aux_weight < 0.0:
            raise ValueError("all_prefix_aux_weight must be non-negative.")

        base_kwargs = dict(kwargs)
        base_kwargs["critic_backbone"] = CRITIC_BACKBONE_MLP
        super().__init__(
            *args,
            critic_transformer_hidden_dim=critic_transformer_hidden_dim,
            critic_transformer_layers=critic_transformer_layers,
            critic_transformer_heads=critic_transformer_heads,
            critic_transformer_dropout=critic_transformer_dropout,
            **base_kwargs,
        )
        self.critic = CausalTransformerPrefixCritic(
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
        # 从模块树中彻底移除 ordinary 的 V/target-V，防止误入优化器或 checkpoint。
        del self.value
        del self.target_value
        self.critic_backbone = CRITIC_BACKBONE_ACSAC_TRANSFORMER
        self.all_prefix_aux_weight = float(all_prefix_aux_weight)

    def qv_parameters(self, include_dynamics: bool = True) -> Iterable[nn.Parameter]:
        """返回 online Q 与可选 Dynamics 参数，不包含独立 V。

        Args:
            include_dynamics (bool): 是否联合优化 Dynamics。

        Returns:
            Iterable[nn.Parameter]: 优化器参数。
        """

        parameters = list(self.critic.parameters())
        if include_dynamics and self.dynamics is not None:
            parameters += list(self.dynamics.parameters())
        return parameters

    @torch.no_grad()
    def select_adaptive_chunks(
        self,
        states: torch.Tensor,
        num_samples: int,
        recommended_mask: Optional[torch.Tensor] = None,
        execution_budgets: Optional[torch.Tensor] = None,
        critic: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在所有候选和合法前缀长度上联合执行 argmax。

        Args:
            states (torch.Tensor): 状态，形状 ``(B, state_dim)``。
            num_samples (int): 每个状态的候选 chunk 数 N。
            recommended_mask (Optional[torch.Tensor]): 历史 item 屏蔽。
            execution_budgets (Optional[torch.Tensor]): 每行允许的最大 H；为空为 K。
            critic (Optional[nn.Module]): 打分网络；为空使用 online Q。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 选中的完整候选 ID、
            embedding 以及执行长度，形状分别为 ``(B,K)``、``(B,K*A)``、``(B,)``。
        """

        states = states.to(device=self.device, dtype=torch.float32)
        candidate_ids, candidate_chunks = self.sample_candidate_chunks(
            states,
            num_samples=num_samples,
            recommended_mask=recommended_mask,
        )
        num_candidates, batch_size, _ = candidate_chunks.shape
        repeated_states = states.unsqueeze(0).expand(num_candidates, -1, -1)
        scoring_critic = self.critic if critic is None else critic
        q_values = scoring_critic(
            repeated_states.reshape(-1, self.state_dim),
            candidate_chunks.reshape(-1, self.chunk_action_dim),
        ).reshape(num_candidates, batch_size, self.chunk_size)
        if execution_budgets is None:
            budgets = torch.full(
                (batch_size,), self.chunk_size, dtype=torch.long, device=self.device,
            )
        else:
            budgets = execution_budgets.to(
                device=self.device, dtype=torch.long,
            ).reshape(-1)
            if budgets.shape[0] != batch_size:
                raise ValueError("execution_budgets must have shape (B,).")
            if torch.any(budgets < 1) or torch.any(budgets > self.chunk_size):
                raise ValueError("execution_budgets must lie in [1, chunk_size].")
        prefix_lengths = torch.arange(1, self.chunk_size + 1, device=self.device)
        q_values = q_values.masked_fill(
            prefix_lengths.view(1, 1, -1) > budgets.view(1, -1, 1),
            -torch.inf,
        )
        flat_indices = q_values.permute(1, 0, 2).reshape(batch_size, -1).argmax(dim=1)
        candidate_indices = torch.div(
            flat_indices, self.chunk_size, rounding_mode="floor",
        )
        selected_lengths = flat_indices.remainder(self.chunk_size) + 1
        row_indices = torch.arange(batch_size, device=self.device)
        return (
            candidate_ids[candidate_indices, row_indices],
            candidate_chunks[candidate_indices, row_indices],
            selected_lengths,
        )

    @torch.no_grad()
    def select_chunks(
        self,
        states: torch.Tensor,
        num_samples: int,
        recommended_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """提供与旧调用方兼容的二元返回值接口。

        自适应执行方应调用 :meth:`select_adaptive_chunks` 获取 H。
        """

        item_ids, chunks, _ = self.select_adaptive_chunks(
            states,
            num_samples=num_samples,
            recommended_mask=recommended_mask,
        )
        return item_ids, chunks

    def _execution_budgets(self, history_state: RolloutHistoryState) -> torch.Tensor:
        """按 max-turn 计算每行当前可执行的最大前缀长度。"""

        assert self.leave_model is not None
        remaining = self.leave_model.max_turn - history_state.env_step
        return remaining.clamp(min=1, max=self.chunk_size)

    @torch.no_grad()
    def _target_state_values(
        self,
        history_state: RolloutHistoryState,
        num_samples: int,
        repeat_policy: str,
    ) -> torch.Tensor:
        """用 target Q 对下一状态所有候选前缀取最大值。"""

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
            num_candidates, -1, -1,
        )
        q_values = self.target_critic(
            repeated_states.reshape(-1, self.state_dim),
            candidate_chunks.reshape(-1, self.chunk_action_dim),
        ).reshape(num_candidates, batch_size, self.chunk_size)
        budgets = self._execution_budgets(history_state)
        prefix_lengths = torch.arange(1, self.chunk_size + 1, device=self.device)
        q_values = q_values.masked_fill(
            prefix_lengths.view(1, 1, -1) > budgets.view(1, -1, 1),
            -torch.inf,
        )
        values = q_values.amax(dim=(0, 2))
        return torch.where(terminal, torch.zeros_like(values), values)

    @torch.no_grad()
    def rollout_adaptive_trajectory(
        self,
        batch: Dict[str, torch.Tensor],
        num_samples: int,
        rollout_depth: int,
        repeat_policy: str,
        leave_policy: str,
    ) -> AdaptiveTrajectoryRollout:
        """重新生成一条逐 chunk 联合选择候选与 H 的 imagined trajectory。"""

        self._batch_user_ids = self._batch_tensor(batch, "user_id").long().view(-1)
        history_state = self._initial_history_state(batch)
        if repeat_policy == REPEAT_POLICY_MASK:
            self._reset_local_rollout_recommended_mask(history_state)
        batch_size = int(history_state.states.shape[0])
        steps: list[AdaptiveTrajectoryStep] = []
        trajectory_returns = torch.zeros(batch_size, device=self.device)
        executed_chunks = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        cumulative_discount = torch.ones(batch_size, device=self.device)
        for _ in range(rollout_depth):
            if repeat_policy == REPEAT_POLICY_MASK:
                active, candidate_mask = self._prepare_rollout_sampling_mask(history_state)
            else:
                active = ~history_state.terminated
                candidate_mask = None
            states = history_state.states
            selected_ids, selected_chunks, selected_lengths = self.select_adaptive_chunks(
                states,
                num_samples=num_samples,
                recommended_mask=candidate_mask,
                execution_budgets=self._execution_budgets(history_state),
            )
            result, next_history_state = self._rollout_one_chunk(
                selected_ids,
                history_state,
                repeat_policy=repeat_policy,
                leave_policy=leave_policy,
                execution_lengths=selected_lengths,
            )
            chunk_discount = self.gamma ** result.effective_steps.float()
            bootstrap_values = self._target_state_values(
                next_history_state,
                num_samples=num_samples,
                repeat_policy=repeat_policy,
            )
            steps.append(AdaptiveTrajectoryStep(
                states=states,
                selected_chunks=selected_chunks.detach(),
                selected_item_ids=selected_ids.detach(),
                selected_lengths=selected_lengths.detach(),
                rollout=result,
                active=active,
                chunk_discount=chunk_discount,
                bootstrap_values=bootstrap_values,
            ))
            trajectory_returns += (
                cumulative_discount * result.reward_chunks * active.float()
            )
            executed_chunks += active.long()
            cumulative_discount *= chunk_discount
            history_state = next_history_state
        return AdaptiveTrajectoryRollout(
            steps=steps,
            trajectory_returns=trajectory_returns,
            executed_chunks=executed_chunks,
        )

    def _td_lambda_target(
        self,
        trajectory: AdaptiveTrajectoryRollout,
        lambda_chunk: float,
    ) -> torch.Tensor:
        """以 target-Q bootstrap 构造第一个宏动作的 chunk TD(lambda) 目标。"""

        continuation = trajectory.steps[-1].bootstrap_values
        for step in reversed(trajectory.steps):
            not_done = 1.0 - step.rollout.done_chunks.float()
            mixed_bootstrap = (
                (1.0 - lambda_chunk) * step.bootstrap_values
                + lambda_chunk * continuation
            )
            continuation = (
                step.rollout.reward_chunks
                + not_done * step.chunk_discount * mixed_bootstrap
            )
        return continuation.detach()

    @staticmethod
    def _repeat_history_state(
        history_state: RolloutHistoryState,
        repeats: int,
    ) -> RolloutHistoryState:
        """为每个样本复制多份相互独立的 rollout 历史。

        复制顺序采用逐样本 ``repeat_interleave``，从而与扁平化后的
        ``[h=1, ..., K]`` 前缀长度顺序严格对齐。

        Args:
            history_state (RolloutHistoryState): 原始 batch 的 chunk 起点历史。
            repeats (int): 每个样本的复制次数，必须为正数。

        Returns:
            RolloutHistoryState: batch 维扩展为原来 ``repeats`` 倍的新历史。

        Raises:
            ValueError: 当 ``repeats`` 不是正数时抛出。
        """

        if repeats <= 0:
            raise ValueError("repeats must be positive.")
        return RolloutHistoryState(
            states=history_state.states.repeat_interleave(repeats, dim=0),
            leave_history=history_state.leave_history.repeat_interleave(
                repeats, dim=0,
            ),
            recommended_mask=history_state.recommended_mask.repeat_interleave(
                repeats, dim=0,
            ),
            env_step=history_state.env_step.repeat_interleave(repeats, dim=0),
            terminated=history_state.terminated.repeat_interleave(repeats, dim=0),
        )

    @torch.no_grad()
    def _all_prefix_one_step_target(
        self,
        batch: Dict[str, torch.Tensor],
        first_step: AdaptiveTrajectoryStep,
        num_samples: int,
        repeat_policy: str,
        leave_policy: str,
    ) -> AllPrefixOneStepTarget:
        """向量化构造首个候选 chunk 的全部合法前缀一步 TD 目标。

        对每个起点复制 ``K`` 份历史，并分别计划执行 ``h=1...K``。每个
        前缀使用自己的累计 reward、Dynamics 末状态、done 与
        ``gamma ** effective_steps``，随后由 target Q 在下一状态的候选与合法
        长度上取最大值。该辅助目标只展开一个 chunk，不改变 selected-H 的
        完整 TD(lambda) imagined trajectory。

        Args:
            batch (Dict[str, torch.Tensor]): ActionChunkDataset 训练 batch。
            first_step (AdaptiveTrajectoryStep): 主 rollout 的第一个宏动作，提供
                已经联合选中的完整候选 chunk。
            num_samples (int): 下一状态 target-Q 的候选 chunk 数量。
            repeat_policy (str): 重复推荐处理策略。
            leave_policy (str): 退出规则处理策略。

        Returns:
            AllPrefixOneStepTarget: 形状均为 ``(B,K)`` 的目标及辅助统计。

        Raises:
            ValueError: 当候选数非正、repeat 策略非法或首步张量形状不匹配时抛出。
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
        expected_item_shape = (batch_size, self.chunk_size)
        if tuple(first_step.selected_item_ids.shape) != expected_item_shape:
            raise ValueError(
                "selected_item_ids shape mismatch: expected "
                f"{expected_item_shape}, got {tuple(first_step.selected_item_ids.shape)}."
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

        # Reward 模型通过 agent 暂存的 user id 对齐 batch；前缀展开时必须同步复制。
        original_user_ids = self._batch_user_ids
        batch_user_ids = self._batch_tensor(batch, "user_id").long().view(-1)
        if batch_user_ids.shape[0] != batch_size:
            raise ValueError("user_id batch size does not match observations.")
        self._batch_user_ids = batch_user_ids.repeat_interleave(self.chunk_size)
        try:
            prefix_rollout, next_history = self._rollout_one_chunk(
                expanded_item_ids,
                expanded_history,
                repeat_policy=repeat_policy,
                leave_policy=leave_policy,
                execution_lengths=expanded_lengths,
            )
            next_values = self._target_state_values(
                next_history,
                num_samples=num_samples,
                repeat_policy=repeat_policy,
            )
        finally:
            self._batch_user_ids = original_user_ids

        chunk_discount = self.gamma ** prefix_rollout.effective_steps.float()
        flat_targets = prefix_rollout.reward_chunks + (
            1.0 - prefix_rollout.done_chunks.float()
        ) * chunk_discount * next_values
        return AllPrefixOneStepTarget(
            values=flat_targets.reshape(batch_size, self.chunk_size).detach(),
            valid=valid,
            rewards=prefix_rollout.reward_chunks.reshape(
                batch_size, self.chunk_size,
            ),
            effective_steps=prefix_rollout.effective_steps.reshape(
                batch_size, self.chunk_size,
            ),
            done=prefix_rollout.done_chunks.reshape(batch_size, self.chunk_size),
        )

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
        """执行 selected-H TD(lambda) 与全前缀一步 TD 的联合 Q 更新。

        主损失仅在联合 argmax 选中的 H 上回归完整 imagined rollout 的
        TD(lambda) 目标；辅助损失则让同一候选的所有合法前缀分别回归各自的
        一步 target-Q 目标。两项均为 MSE，且不更新 Actor 与 target Q。

        Args:
            batch (Dict[str, torch.Tensor]): ActionChunkDataset 训练 batch。
            optimizer (torch.optim.Optimizer): online Q 与可选 Dynamics 优化器。
            num_samples_train (int): 每个状态的候选 chunk 数量。
            repeat_policy (str): 重复推荐处理策略。
            rollout_depth (int): selected-H imagined rollout 的 chunk 深度。
            lambda_chunk (float): selected-H chunk TD(lambda) 系数。
            leave_policy (str): 退出规则处理策略。

        Returns:
            Dict[str, float]: loss、Q 值、rollout 与合法前缀统计。

        Raises:
            RuntimeError: 当 rollout 所需依赖未配置时抛出。
            ValueError: 当候选数、深度或 lambda 参数非法时抛出。
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
            target_q = self._td_lambda_target(trajectory, lambda_chunk)
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
        prefix_q_values = self.critic(
            first_step.states,
            first_step.selected_chunks,
        )
        selected_q_values = prefix_q_values.gather(
            1, first_step.selected_lengths.unsqueeze(1) - 1,
        ).squeeze(1)
        selected_td_lambda_loss = F.mse_loss(selected_q_values, target_q)
        if all_prefix_target is None:
            all_prefix_aux_loss = prefix_q_values.sum() * 0.0
            valid_prefix_q = prefix_q_values.new_empty(0)
            valid_prefix_targets = prefix_q_values.new_empty(0)
            legal_prefixes_mean = 0.0
        else:
            valid_prefix_q = prefix_q_values[all_prefix_target.valid]
            valid_prefix_targets = all_prefix_target.values[all_prefix_target.valid]
            if valid_prefix_q.numel() == 0:
                all_prefix_aux_loss = prefix_q_values.sum() * 0.0
            else:
                all_prefix_aux_loss = F.mse_loss(
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
        first_rollout = first_step.rollout
        metrics = {
            "critic/critic_loss": float(critic_loss.detach().cpu()),
            "critic/selected_td_lambda_loss": float(
                selected_td_lambda_loss.detach().cpu()
            ),
            "critic/all_prefix_aux_loss": float(
                all_prefix_aux_loss.detach().cpu()
            ),
            "critic/all_prefix_aux_weight": self.all_prefix_aux_weight,
            "dynamics/loss": float(dynamics_loss.detach().cpu()),
            "critic/q_mean": float(selected_q_values.detach().mean().cpu()),
            "critic/target_q_mean": float(target_q.detach().mean().cpu()),
            "critic/all_prefix_q_mean": (
                float(valid_prefix_q.detach().mean().cpu())
                if valid_prefix_q.numel() > 0 else 0.0
            ),
            "critic/all_prefix_target_q_mean": (
                float(valid_prefix_targets.detach().mean().cpu())
                if valid_prefix_targets.numel() > 0 else 0.0
            ),
            "rollout/depth": float(rollout_depth),
            "rollout/lambda_chunk": float(lambda_chunk),
            "rollout/trajectory_reward": float(
                trajectory.trajectory_returns.detach().mean().cpu()
            ),
            "rollout/trajectory_chunks": float(
                trajectory.executed_chunks.float().detach().mean().cpu()
            ),
            "rollout/reward_chunk": float(
                first_rollout.reward_chunks.detach().mean().cpu()
            ),
            "rollout/pred_reward": self._masked_mean(
                first_rollout.pred_rewards, first_rollout.chunk_valid,
            ),
            "rollout/entropy": self._masked_mean(
                first_rollout.entropy_rewards, first_rollout.chunk_valid,
            ),
            "rollout/uncertainty": self._masked_mean(
                first_rollout.uncertainty_penalties, first_rollout.chunk_valid,
            ),
            "rollout/effective_steps": float(
                first_rollout.effective_steps.float().detach().mean().cpu()
            ),
            "rollout/selected_h_mean": float(
                first_step.selected_lengths.float().detach().mean().cpu()
            ),
            "rollout/legal_prefixes_mean": legal_prefixes_mean,
            "rollout/done_ratio": float(
                first_rollout.done_chunks.float().detach().mean().cpu()
            ),
            "rollout/repeat_ratio": float(first_rollout.repeat_ratio.detach().cpu()),
            "rollout/exact_repeat_ratio": float(
                first_rollout.exact_repeat_ratio.detach().cpu()
            ),
        }
        for length in range(1, self.chunk_size + 1):
            metrics[f"rollout/selected_h_{length}_ratio"] = float(
                (first_step.selected_lengths == length).float().mean().cpu()
            )
        return metrics

    @torch.no_grad()
    def soft_update_target_critic(self) -> None:
        """按 Polyak 系数软更新 target Q。"""

        for target_param, source_param in zip(
            self.target_critic.parameters(), self.critic.parameters(),
        ):
            target_param.data.mul_(1.0 - self.target_tau).add_(
                source_param.data, alpha=self.target_tau,
            )

    def checkpoint_state(self, config: Dict[str, object]) -> Dict[str, object]:
        """构造不含 V、但含 online/target Q 的独立 checkpoint。"""

        return {
            "format": ACSAC_CHECKPOINT_FORMAT,
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
        }

    def load_checkpoint_state(
        self,
        checkpoint: Dict[str, object],
        strict: bool = True,
        load_qv: bool = True,
    ) -> None:
        """加载共享 ordinary BC 或本分支完整 checkpoint。"""

        if checkpoint.get("format") != ACSAC_CHECKPOINT_FORMAT:
            if load_qv:
                raise ValueError("ACSAC Q loading requires an ACSAC checkpoint.")
            super().load_checkpoint_state(checkpoint, strict=strict, load_qv=False)
            return
        base_checkpoint = dict(checkpoint)
        base_checkpoint["format"] = "mac_origin_precomputed_state_v2"
        base_checkpoint["critic_backbone"] = CRITIC_BACKBONE_MLP
        super().load_checkpoint_state(
            base_checkpoint,
            strict=strict,
            load_qv=False,
        )
        if load_qv:
            self.critic.load_state_dict(checkpoint["critic"], strict=strict)
            target_state = checkpoint.get("target_critic", checkpoint["critic"])
            self.target_critic.load_state_dict(target_state, strict=strict)
