"""DORL-MAC 核心学习器。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from examples.our_model.models.chunk_actor import ChunkFlowActor, ChunkOneStepActor
from examples.our_model.models.chunk_value import ChunkCritic, ChunkValue
from examples.our_model.models.dorl_reward import DORLRewardModel
from examples.our_model.models.leave_model import RuleBasedLeaveModel
from examples.our_model.models.state_tracker_dynamics import StateTrackerDynamics
from examples.our_model.policy.action_mapper import ActionMapper

ACTOR_BACKEND_FLOW = "flow"
"""正式 flow actor 后端名称。"""

ACTOR_BACKEND_MLP_BC = "mlp_bc"
"""smoke test 使用的普通 MLP BC actor 后端名称。"""

REPEAT_POLICY_TRUNCATE = "truncate"
"""训练 rollout 默认重复推荐处理：立即截断 chunk。"""

REPEAT_POLICY_MASK = "mask"
"""评估 rollout 默认重复推荐处理：映射时屏蔽已推荐 item。"""

DEFAULT_INVALID_ACTION_PENALTY = -1.0
"""重复推荐被截断时的固定惩罚。"""


@dataclass
class RolloutResult:
    """chunk rollout 的结果容器。

    Attributes:
        next_states (torch.Tensor): rollout 后状态，形状为 `(B, state_dim)`。
        reward_chunks (torch.Tensor): chunk 折扣累计 reward，形状为 `(B,)`。
        done_chunks (torch.Tensor): chunk 是否终止，形状为 `(B,)`。
        effective_steps (torch.Tensor): 实际进入 reward/state 的步数，形状为 `(B,)`。
        selected_item_ids (torch.Tensor): 每个 chunk step 映射出的 item id，形状为 `(B, K)`。
        step_rewards (torch.Tensor): 每个有效 step 的 reward，形状为 `(B, K)`。
        chunk_valid (torch.Tensor): 每个 step 是否实际执行，形状为 `(B, K)`。
        repeat_ratio (torch.Tensor): 当前 batch 中重复截断比例。
    """

    next_states: torch.Tensor
    reward_chunks: torch.Tensor
    done_chunks: torch.Tensor
    effective_steps: torch.Tensor
    selected_item_ids: torch.Tensor
    step_rewards: torch.Tensor
    chunk_valid: torch.Tensor
    repeat_ratio: torch.Tensor


class MACAgent(nn.Module):
    """PyTorch 版 DORL-MAC 最小闭环 agent。

    该 agent 管理 chunk actor、critic/value、显式 StateTracker dynamics、
    DORL reward model、leave model 和 action mapper。训练阶段用 offline chunk
    做 actor 预训练，用完整 chunk rollout 做 Q/V 更新。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        actor_hidden_dims: Iterable[int],
        value_hidden_dims: Iterable[int],
        gamma: float,
        device: torch.device,
        action_mapper: ActionMapper,
        dynamics: Optional[StateTrackerDynamics] = None,
        reward_model: Optional[DORLRewardModel] = None,
        leave_model: Optional[RuleBasedLeaveModel] = None,
        target_tau: float = 0.005,
        invalid_action_penalty: float = DEFAULT_INVALID_ACTION_PENALTY,
    ) -> None:
        """初始化 DORL-MAC agent。

        Args:
            state_dim (int): 状态维度。
            action_dim (int): 单步 action embedding 维度。
            chunk_size (int): action chunk 长度。
            actor_hidden_dims (Iterable[int]): actor 隐藏层维度。
            value_hidden_dims (Iterable[int]): critic/value 隐藏层维度。
            gamma (float): 折扣因子。
            device (torch.device): 计算设备。
            action_mapper (ActionMapper): action embedding 到 item id 的映射器。
            dynamics (Optional[StateTrackerDynamics]): 显式 StateTracker dynamics。
            reward_model (Optional[DORLRewardModel]): DORL reward 模型。
            leave_model (Optional[RuleBasedLeaveModel]): 规则退出模型。
            target_tau (float): target value 软更新系数。
            invalid_action_penalty (float): 重复推荐截断惩罚。

        Raises:
            ValueError: 当关键超参数非法时抛出。
        """

        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1].")
        if not 0 < target_tau <= 1:
            raise ValueError("target_tau must be in (0, 1].")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.chunk_action_dim = self.action_dim * self.chunk_size
        self.gamma = float(gamma)
        self.device = device
        self.action_mapper = action_mapper
        self.dynamics = dynamics
        self.reward_model = reward_model
        self.leave_model = leave_model
        self.target_tau = float(target_tau)
        self.invalid_action_penalty = float(invalid_action_penalty)

        self.flow_actor = ChunkFlowActor(
            self.state_dim,
            self.chunk_action_dim,
            actor_hidden_dims,
        ).to(device)
        self.one_step_actor = ChunkOneStepActor(
            self.state_dim,
            self.chunk_action_dim,
            actor_hidden_dims,
        ).to(device)
        self.critic = ChunkCritic(
            self.state_dim,
            self.chunk_action_dim,
            value_hidden_dims,
        ).to(device)
        self.value = ChunkValue(self.state_dim, value_hidden_dims).to(device)
        self.target_value = copy.deepcopy(self.value).to(device)
        self.target_value.requires_grad_(False)

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        """返回 actor 预训练需要优化的参数。

        Returns:
            Iterable[nn.Parameter]: flow actor 与 one-step actor 参数。
        """

        return list(self.flow_actor.parameters()) + list(self.one_step_actor.parameters())

    def qv_parameters(self) -> Iterable[nn.Parameter]:
        """返回 Q/V 训练需要优化的参数。

        Returns:
            Iterable[nn.Parameter]: critic 与 value 参数。
        """

        return list(self.critic.parameters()) + list(self.value.parameters())

    def pretrain_actor_update(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        actor_backend: str,
        flow_steps: int,
        bc_weight: float = 1.0,
    ) -> Dict[str, float]:
        """执行一次 actor 预训练更新。

        Args:
            batch (Dict[str, torch.Tensor]): ActionChunkDataset 输出的 batch。
            optimizer (torch.optim.Optimizer): actor 优化器。
            actor_backend (str): `flow` 或 `mlp_bc`。
            flow_steps (int): flow actor 蒸馏采样步数。
            bc_weight (float): one-step BC loss 权重。

        Returns:
            Dict[str, float]: 训练指标。

        Raises:
            ValueError: 当 actor backend 不支持时抛出。
        """

        self.train()
        states = self._batch_tensor(batch, "observations").float()
        target_chunks = self._batch_tensor(batch, "actions").float()
        batch_size = states.shape[0]
        noises = torch.randn(batch_size, self.chunk_action_dim, device=self.device)
        optimizer.zero_grad(set_to_none=True)

        one_step_chunks = self.one_step_actor(states, noises if actor_backend == ACTOR_BACKEND_FLOW else None)
        bc_loss = F.mse_loss(one_step_chunks, target_chunks)
        flow_loss = torch.zeros((), device=self.device)
        distill_loss = torch.zeros((), device=self.device)

        if actor_backend == ACTOR_BACKEND_FLOW:
            x_0 = torch.randn_like(target_chunks)
            times = torch.rand(batch_size, 1, device=self.device)
            x_t = (1.0 - times) * x_0 + times * target_chunks
            target_velocity = target_chunks - x_0
            predicted_velocity = self.flow_actor(states, x_t, times)
            flow_loss = F.mse_loss(predicted_velocity, target_velocity)
            with torch.no_grad():
                target_flow_chunks = self.flow_actor.sample_flow(states, noises, flow_steps=flow_steps)
            distill_loss = F.mse_loss(one_step_chunks, target_flow_chunks)
            loss = flow_loss + distill_loss + bc_weight * bc_loss
        elif actor_backend == ACTOR_BACKEND_MLP_BC:
            loss = bc_weight * bc_loss
        else:
            raise ValueError(f"Unsupported actor_backend: {actor_backend}")

        loss.backward()
        optimizer.step()
        dynamics_mse = self.compute_dynamics_mse(batch)
        return {
            "actor/loss": float(loss.detach().cpu()),
            "actor/bc_loss": float(bc_loss.detach().cpu()),
            "actor/flow_loss": float(flow_loss.detach().cpu()),
            "actor/distill_loss": float(distill_loss.detach().cpu()),
            "state_tracker/next_state_mse": dynamics_mse,
        }

    @torch.no_grad()
    def compute_dynamics_mse(self, batch: Dict[str, torch.Tensor]) -> float:
        """用真实 chunk 校验显式 StateTracker dynamics。

        Args:
            batch (Dict[str, torch.Tensor]): ActionChunkDataset 输出 batch。

        Returns:
            float: 重算 next state 与数据集中 target next state 的 MSE；
            若未配置 dynamics，则返回 `0.0`。
        """

        if self.dynamics is None:
            return 0.0
        target_next_states = self._batch_tensor(batch, "next_observations").float()
        chunk_actions = self._batch_tensor(batch, "actions").float().view(
            -1, self.chunk_size, self.action_dim
        )
        chunk_rewards = self._batch_tensor(batch, "chunk_step_rewards").float()
        chunk_valid = torch.ones_like(chunk_rewards, dtype=torch.bool)
        next_states = self.dynamics.next_state(
            self._batch_tensor(batch, "history_vectors").float(),
            self._batch_tensor(batch, "history_valid").float(),
            chunk_actions,
            chunk_rewards,
            chunk_valid,
        )
        return float(F.mse_loss(next_states, target_next_states).detach().cpu())

    @torch.no_grad()
    def sample_candidate_chunks(self, states: torch.Tensor, num_samples: int) -> torch.Tensor:
        """从 one-step actor 采样多个候选 action chunk。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。
            num_samples (int): 每个状态采样的候选数量。

        Returns:
            torch.Tensor: 候选 chunk，形状为 `(N, B, chunk_action_dim)`。

        Raises:
            ValueError: 当 `num_samples` 非正时抛出。
        """

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        states = states.to(device=self.device, dtype=torch.float32)
        repeated_states = states.unsqueeze(0).expand(num_samples, -1, -1)
        noises = torch.randn(
            num_samples,
            states.shape[0],
            self.chunk_action_dim,
            device=self.device,
        )
        chunks = self.one_step_actor(
            repeated_states.reshape(-1, self.state_dim),
            noises.reshape(-1, self.chunk_action_dim),
        )
        return chunks.view(num_samples, states.shape[0], self.chunk_action_dim)

    @torch.no_grad()
    def select_chunks(self, states: torch.Tensor, num_samples: int) -> torch.Tensor:
        """用 critic 从候选 chunks 中选择 Q 值最高的 chunk。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。
            num_samples (int): 候选 chunk 数量。

        Returns:
            torch.Tensor: 选中的 chunk，形状为 `(B, chunk_action_dim)`。
        """

        candidates = self.sample_candidate_chunks(states, num_samples=num_samples)
        num_samples_actual, batch_size, _ = candidates.shape
        repeated_states = states.unsqueeze(0).expand(num_samples_actual, -1, -1)
        q_values = self.critic(
            repeated_states.reshape(-1, self.state_dim),
            candidates.reshape(-1, self.chunk_action_dim),
        ).view(num_samples_actual, batch_size)
        best_indices = torch.argmax(q_values, dim=0)
        return candidates[best_indices, torch.arange(batch_size, device=self.device)]

    def qv_update(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        num_samples_train: int,
        repeat_policy: str = REPEAT_POLICY_TRUNCATE,
    ) -> Dict[str, float]:
        """执行一次 chunk-level Q/V 更新。

        Args:
            batch (Dict[str, torch.Tensor]): ActionChunkDataset 输出 batch。
            optimizer (torch.optim.Optimizer): critic/value 优化器。
            num_samples_train (int): rejection sampling 候选 chunk 数量。
            repeat_policy (str): 重复推荐处理策略，支持 `truncate` 和 `mask`。

        Returns:
            Dict[str, float]: 训练与 rollout 指标。
        """

        if self.dynamics is None or self.reward_model is None or self.leave_model is None:
            raise RuntimeError("qv_update requires dynamics, reward_model and leave_model.")
        self.train()
        states = self._batch_tensor(batch, "observations").float()
        with torch.no_grad():
            selected_chunks = self.select_chunks(states, num_samples=num_samples_train).detach()
            rollout = self.rollout_chunks(batch, selected_chunks, repeat_policy=repeat_policy)
            next_values = self.target_value(rollout.next_states)
            bootstrap_discount = torch.pow(
                torch.full_like(rollout.effective_steps.float(), self.gamma),
                rollout.effective_steps.float(),
            )
            target_q = rollout.reward_chunks + (1.0 - rollout.done_chunks.float()) * bootstrap_discount * next_values

        optimizer.zero_grad(set_to_none=True)
        q_values = self.critic(states, selected_chunks)
        value_predictions = self.value(states)
        critic_loss = F.mse_loss(q_values, target_q)
        value_loss = F.mse_loss(value_predictions, target_q)
        loss = critic_loss + value_loss
        loss.backward()
        optimizer.step()
        self.soft_update_target_value()
        return {
            "critic/critic_loss": float(critic_loss.detach().cpu()),
            "value/value_loss": float(value_loss.detach().cpu()),
            "critic/q_mean": float(q_values.detach().mean().cpu()),
            "critic/target_q_mean": float(target_q.detach().mean().cpu()),
            "rollout/reward_chunk": float(rollout.reward_chunks.detach().mean().cpu()),
            "rollout/effective_steps": float(rollout.effective_steps.float().detach().mean().cpu()),
            "rollout/done_ratio": float(rollout.done_chunks.float().detach().mean().cpu()),
            "rollout/repeat_ratio": float(rollout.repeat_ratio.detach().cpu()),
        }

    @torch.no_grad()
    def rollout_chunks(
        self,
        batch: Dict[str, torch.Tensor],
        chunks: torch.Tensor,
        repeat_policy: str,
    ) -> RolloutResult:
        """对完整 action chunk 逐步执行推荐系统 rollout。

        Args:
            batch (Dict[str, torch.Tensor]): 起始状态与历史信息。
            chunks (torch.Tensor): flatten action chunk，形状为 `(B, K * action_dim)`。
            repeat_policy (str): 重复推荐处理策略。

        Returns:
            RolloutResult: rollout 后的状态、reward、done 和统计信息。

        Raises:
            ValueError: 当重复策略不支持时抛出。
        """

        if repeat_policy not in {REPEAT_POLICY_TRUNCATE, REPEAT_POLICY_MASK}:
            raise ValueError(f"Unsupported repeat_policy: {repeat_policy}")
        assert self.dynamics is not None
        assert self.reward_model is not None
        assert self.leave_model is not None

        chunks = chunks.to(device=self.device, dtype=torch.float32)
        batch_size = int(chunks.shape[0])
        chunk_actions_raw = chunks.view(batch_size, self.chunk_size, self.action_dim)
        executed_action_embeddings = torch.zeros_like(chunk_actions_raw)
        step_rewards = torch.zeros(batch_size, self.chunk_size, device=self.device)
        chunk_valid = torch.zeros(batch_size, self.chunk_size, dtype=torch.bool, device=self.device)
        selected_item_ids = torch.full(
            (batch_size, self.chunk_size),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        reward_chunks = torch.zeros(batch_size, device=self.device)
        effective_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        done_chunks = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        repeat_events = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        recommended_mask = self._batch_tensor(batch, "recommended_mask").bool().clone()
        leave_history = self._batch_tensor(batch, "leave_history_item_ids").long().clone()
        raw_user_ids = self._batch_tensor(batch, "user_id").long().view(-1)
        env_steps = self._batch_tensor(batch, "env_step").long().view(-1)

        for step_index in range(self.chunk_size):
            if not bool(active.any().item()):
                break
            mapper_mask = recommended_mask if repeat_policy == REPEAT_POLICY_MASK else None
            item_ids, _ = self.action_mapper.map_embeddings(
                chunk_actions_raw[:, step_index, :],
                recommended_mask=mapper_mask,
                topk=1,
            )
            item_ids = item_ids.squeeze(1)
            selected_item_ids[:, step_index] = item_ids
            row_indices = torch.arange(batch_size, device=self.device)
            repeated = recommended_mask[row_indices, item_ids]
            invalid = active & repeated & (repeat_policy == REPEAT_POLICY_TRUNCATE)
            if invalid.any():
                discount = torch.pow(
                    torch.full_like(effective_steps[invalid].float(), self.gamma),
                    effective_steps[invalid].float(),
                )
                reward_chunks[invalid] += discount * self.invalid_action_penalty
                repeat_events[invalid] = True
                done_chunks[invalid] = True

            executable = active & ~invalid
            if executable.any():
                rewards = self.reward_model.reward(raw_user_ids[executable], item_ids[executable])
                discount = torch.pow(
                    torch.full_like(effective_steps[executable].float(), self.gamma),
                    effective_steps[executable].float(),
                )
                reward_chunks[executable] += discount * rewards
                step_rewards[executable, step_index] = rewards
                chunk_valid[executable, step_index] = True
                executed_action_embeddings[executable, step_index] = self.action_mapper.item_embeddings[
                    item_ids[executable]
                ]
                leave_done = self.leave_model.should_leave_batch(
                    leave_history[executable],
                    item_ids[executable],
                    env_steps[executable],
                )
                effective_steps[executable] += 1
                recommended_mask[executable, item_ids[executable]] = True
                leave_history[executable] = self.leave_model.append_items(
                    leave_history[executable],
                    item_ids[executable],
                    torch.ones_like(item_ids[executable], dtype=torch.bool),
                )
                env_steps[executable] += 1
                executable_indices = torch.nonzero(executable, as_tuple=False).view(-1)
                done_chunks[executable_indices[leave_done]] = True

            active = active & ~done_chunks

        next_states = self.dynamics.next_state(
            self._batch_tensor(batch, "history_vectors").float(),
            self._batch_tensor(batch, "history_valid").float(),
            executed_action_embeddings,
            step_rewards,
            chunk_valid,
        )
        repeat_ratio = repeat_events.float().mean()
        return RolloutResult(
            next_states=next_states,
            reward_chunks=reward_chunks,
            done_chunks=done_chunks,
            effective_steps=effective_steps,
            selected_item_ids=selected_item_ids,
            step_rewards=step_rewards,
            chunk_valid=chunk_valid,
            repeat_ratio=repeat_ratio,
        )

    @torch.no_grad()
    def soft_update_target_value(self) -> None:
        """按 `target_tau` 软更新 target value 网络。

        Returns:
            None.
        """

        for target_param, source_param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.mul_(1.0 - self.target_tau).add_(source_param.data, alpha=self.target_tau)

    def checkpoint_state(self, config: Dict[str, object]) -> Dict[str, object]:
        """构造可保存的 checkpoint 字典。

        Args:
            config (Dict[str, object]): 本次实验配置快照。

        Returns:
            Dict[str, object]: 可传给 `torch.save` 的 checkpoint。
        """

        return {
            "config": dict(config),
            "flow_actor": self.flow_actor.state_dict(),
            "one_step_actor": self.one_step_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "value": self.value.state_dict(),
            "target_value": self.target_value.state_dict(),
        }

    def load_checkpoint_state(self, checkpoint: Dict[str, object], strict: bool = True) -> None:
        """从 checkpoint 恢复模型参数。

        Args:
            checkpoint (Dict[str, object]): `torch.load` 读取出的 checkpoint。
            strict (bool): 是否使用严格 state_dict 加载。

        Returns:
            None.

        Raises:
            KeyError: 当 checkpoint 缺少必要字段时抛出。
        """

        for key in ["flow_actor", "one_step_actor"]:
            if key not in checkpoint:
                raise KeyError(f"checkpoint missing key: {key}")
        self.flow_actor.load_state_dict(checkpoint["flow_actor"], strict=strict)
        self.one_step_actor.load_state_dict(checkpoint["one_step_actor"], strict=strict)
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"], strict=strict)
        if "value" in checkpoint:
            self.value.load_state_dict(checkpoint["value"], strict=strict)
        if "target_value" in checkpoint:
            self.target_value.load_state_dict(checkpoint["target_value"], strict=strict)
        else:
            self.target_value.load_state_dict(self.value.state_dict(), strict=strict)

    def _batch_tensor(self, batch: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        """从 batch 中取张量并移动到 agent 设备。

        Args:
            batch (Dict[str, torch.Tensor]): DataLoader batch。
            key (str): 字段名。

        Returns:
            torch.Tensor: 已移动到目标设备的张量。

        Raises:
            KeyError: 当 batch 缺少字段时抛出。
        """

        if key not in batch:
            raise KeyError(f"batch missing key: {key}")
        return batch[key].to(self.device)
