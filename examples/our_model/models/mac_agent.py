"""DORL-MAC 核心学习器（离散 Categorical BC + Q rejection sampling 版）。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from examples.our_model.models.chunk_actor import CategoricalChunkActor
from examples.our_model.models.chunk_value import ChunkCritic, ChunkValue
from examples.our_model.models.dorl_reward import DORLRewardModel
from examples.our_model.models.leave_model import RuleBasedLeaveModel
from examples.our_model.models.state_tracker_dynamics import StateTrackerDynamics
from examples.our_model.policy.action_mapper import ActionMapper

REPEAT_POLICY_TRUNCATE = "truncate"
"""兼容旧 CLI 的训练策略名；当前语义为重复/退出违规只惩罚不截断。"""

REPEAT_POLICY_MASK = "mask"
"""严格去重策略：采样时屏蔽历史 item，并保证候选 chunk 内 item 唯一。"""

LEAVE_POLICY_PENALTY = "penalty"
"""退出规则处理（历史默认）：违规只加 reward penalty，不截断 chunk。"""

LEAVE_POLICY_TERMINATE = "terminate"
"""退出规则处理（对齐 DORL）：触发退出的动作执行完后立即终止 episode。"""

DEFAULT_INVALID_ACTION_PENALTY = -1.0
"""重复 item 或类别退出规则违规时的固定 reward 惩罚。"""

_MASK_LOGIT_VALUE = -1.0e9
"""被 `recommended_mask` 屏蔽的候选 item 用于 softmax 的极小 logit。"""


@dataclass
class RolloutResult:
    """chunk rollout 的结果容器。

    Attributes:
        next_states (torch.Tensor): rollout 后状态，形状为 `(B, state_dim)`。
        reward_chunks (torch.Tensor): chunk 折扣累计 reward，形状为 `(B,)`。
        done_chunks (torch.Tensor): chunk 是否终止，形状为 `(B,)`。
        effective_steps (torch.Tensor): 实际进入 reward/state 的步数，形状为 `(B,)`。
        selected_item_ids (torch.Tensor): 每个 chunk step 使用的 item id，形状为 `(B, K)`。
        step_rewards (torch.Tensor): 每个有效 step 的 reward，形状为 `(B, K)`。
        chunk_valid (torch.Tensor): 每个 step 是否实际执行，形状为 `(B, K)`。
        pred_rewards (torch.Tensor): user model 原始预测 reward，形状为 `(B, K)`。
        entropy_rewards (torch.Tensor): DORL entropy 项，形状为 `(B, K)`。
        uncertainty_penalties (torch.Tensor): uncertainty 数值，形状为 `(B, K)`。
        repeat_ratio (torch.Tensor): 当前 batch 中类别退出规则违规比例。
        exact_repeat_ratio (torch.Tensor): 当前 batch 中 exact item 重复比例。
    """

    next_states: torch.Tensor
    reward_chunks: torch.Tensor
    done_chunks: torch.Tensor
    effective_steps: torch.Tensor
    selected_item_ids: torch.Tensor
    step_rewards: torch.Tensor
    chunk_valid: torch.Tensor
    pred_rewards: torch.Tensor
    entropy_rewards: torch.Tensor
    uncertainty_penalties: torch.Tensor
    repeat_ratio: torch.Tensor
    exact_repeat_ratio: torch.Tensor


@dataclass
class RolloutHistoryState:
    """多步 imagined rollout 在 chunk 之间传递的显式历史缓存。

    该结构只携带"已观测/已执行的 item 序列与 reward"等历史信息，不携带
    dynamics 自身预测的 state 向量。每执行一个 chunk 后，把映射出的真实 item
    追加进历史窗口并重新交给 StateTrackerDynamics 重算状态，从而保持"重算范式"
    而非"自回归预测范式"，避免 dynamics 误差沿 rollout 深度累积。

    Attributes:
        history_vectors (torch.Tensor): StateTrackerAvg 历史向量，形状 `(B, W, state_dim)`。
        history_valid (torch.Tensor): 历史有效标记，形状 `(B, W)`。
        leave_history (torch.Tensor): KuaiEnv leave 判断的 item 历史，形状 `(B, H_leave)`。
        recommended_mask (torch.Tensor): 已推荐 item mask，形状 `(B, num_items)`。
        env_step (torch.Tensor): 当前 episode 内已实际执行的底层 action 数，形状 `(B,)`。
        terminated (torch.Tensor): 该 episode 是否已终止，形状 `(B,)`。
    """

    history_vectors: torch.Tensor
    history_valid: torch.Tensor
    leave_history: torch.Tensor
    recommended_mask: torch.Tensor
    env_step: torch.Tensor
    terminated: torch.Tensor


@dataclass
class TrajectoryStep:
    """imagined rollout 中单个 chunk 决策步的结果。

    Attributes:
        states (torch.Tensor): 该 chunk 起点状态，形状 `(B, state_dim)`。
        selected_chunks (torch.Tensor): 选出的 chunk embedding 向量，形状 `(B, K*action_dim)`。
        selected_item_ids (torch.Tensor): 选出的 chunk 内 item id，形状 `(B, K)`。
        rollout (RolloutResult): 执行该 chunk 得到的 rollout 结果。
        active (torch.Tensor): 该步是否为有效决策步（episode 尚未终止），形状 `(B,)`。
        chunk_discount (torch.Tensor): 该 chunk 的折扣因子 `gamma^effective_steps`，形状 `(B,)`。
        q_target (torch.Tensor): 该 chunk 的 chunk-level TD 目标 `R_chunk + discount*(1-done)*V(s_next)`。
    """

    states: torch.Tensor
    selected_chunks: torch.Tensor
    selected_item_ids: torch.Tensor
    rollout: RolloutResult
    active: torch.Tensor
    chunk_discount: torch.Tensor
    q_target: torch.Tensor


@dataclass
class TrajectoryRollout:
    """完整 H 步 imagined rollout 结果容器。

    Attributes:
        steps (list): 每个 chunk 决策步的 `TrajectoryStep`，长度不超过 `rollout_depth`。
        trajectory_returns (torch.Tensor): 整条轨迹折扣累计 reward（诊断用），形状 `(B,)`。
        executed_chunks (torch.Tensor): 每个样本实际执行的 chunk 数，形状 `(B,)`。
    """

    steps: list
    trajectory_returns: torch.Tensor
    executed_chunks: torch.Tensor


class MACAgent(nn.Module):
    """PyTorch 版 DORL-MAC 最小闭环 agent（离散 Categorical + rejection sampling）。

    该 agent 管理 chunk actor（离散 Categorical BC）、critic/value、显式
    StateTracker dynamics、DORL reward model、leave model。整个流水线中动作始终
    以「离散 item id」为一等公民：actor 直接在 N 个 item 上输出 categorical
    分布，rejection sampling 阶段从每步分布采 N_samples 组 item id 序列，通过
    item embedding 表查表拼出 chunk 向量后交给 critic 打分，选出 Q 最高的一组
    item id 作为该 state 的 chunk 决策。因此不再存在"连续 action 向量 → 归一化
    点积映射到最近 item"的失真环节。
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
        dynamics_loss_weight: float = 1.0,
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
            action_mapper (ActionMapper): 持有 item embedding 表，供离散 item id
                查表拼 chunk 向量使用（不再执行"连续→离散"的相似度映射）。
            dynamics (Optional[StateTrackerDynamics]): 显式 StateTracker dynamics。
            reward_model (Optional[DORLRewardModel]): DORL reward 模型。
            leave_model (Optional[RuleBasedLeaveModel]): 规则退出模型。
            target_tau (float): target value 软更新系数。
            invalid_action_penalty (float): 重复 item 或类别退出规则违规惩罚。
            dynamics_loss_weight (float): dynamics 监督 next-state loss 权重。

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
        self.num_items = int(action_mapper.num_items)
        self.dynamics = dynamics.to(device) if dynamics is not None else None
        self.reward_model = reward_model
        self.leave_model = leave_model
        self.target_tau = float(target_tau)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.dynamics_loss_weight = float(dynamics_loss_weight)

        self.actor = CategoricalChunkActor(
            state_dim=self.state_dim,
            chunk_size=self.chunk_size,
            num_items=self.num_items,
            hidden_dims=actor_hidden_dims,
        ).to(device)
        self.critic = ChunkCritic(
            self.state_dim,
            self.chunk_action_dim,
            value_hidden_dims,
        ).to(device)
        self.value = ChunkValue(self.state_dim, value_hidden_dims).to(device)
        self.target_value = copy.deepcopy(self.value).to(device)
        self.target_value.requires_grad_(False)
        # rollout 期间缓存的原始 user id；由 rollout 入口设置后供 _rollout_one_chunk 复用。
        self._batch_user_ids: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Parameter grouping
    # ------------------------------------------------------------------
    def actor_parameters(self) -> Iterable[nn.Parameter]:
        """返回 actor 预训练需要优化的参数。

        Returns:
            Iterable[nn.Parameter]: categorical actor 参数。
        """

        return list(self.actor.parameters())

    def qv_parameters(self) -> Iterable[nn.Parameter]:
        """返回 Q/V 训练需要优化的参数。

        Returns:
            Iterable[nn.Parameter]: critic 与 value 参数。
        """

        parameters = list(self.critic.parameters()) + list(self.value.parameters())
        if self.dynamics is not None:
            parameters += list(self.dynamics.parameters())
        return parameters

    # ------------------------------------------------------------------
    # Pretraining: categorical BC
    # ------------------------------------------------------------------
    def pretrain_actor_update(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """执行一次离散 categorical BC actor 预训练更新。

        对 chunk 内每一步 item id 做交叉熵监督，等价于把 chunk 看作 K 个条件独立
        的离散 action，用离线数据集的真实 item id 作为标签。

        Args:
            batch (Dict[str, torch.Tensor]): ActionChunkDataset 输出的 batch。
            optimizer (torch.optim.Optimizer): actor 优化器。

        Returns:
            Dict[str, float]: 训练指标，含 CE loss、每步平均 log-prob、entropy 等。
        """

        self.train()
        states = self._batch_tensor(batch, "observations").float()
        chunk_item_ids = self._batch_tensor(batch, "chunk_item_ids").long()
        optimizer.zero_grad(set_to_none=True)

        logits = self.actor(states)  # (B, K, N)
        log_probs_all = torch.log_softmax(logits, dim=-1)
        gathered = log_probs_all.gather(-1, chunk_item_ids.unsqueeze(-1)).squeeze(-1)  # (B, K)
        bc_loss = -gathered.mean()
        # 诊断：每步分布熵，用来观察 BC 是否过早坍缩到少量 item。
        probs = torch.softmax(logits, dim=-1)
        step_entropy = -(probs * log_probs_all).sum(dim=-1).mean()

        bc_loss.backward()
        optimizer.step()
        dynamics_mse = self.compute_dynamics_mse(batch)
        return {
            "actor/loss": float(bc_loss.detach().cpu()),
            "actor/bc_loss": float(bc_loss.detach().cpu()),
            "actor/log_prob": float(gathered.detach().mean().cpu()),
            "actor/entropy": float(step_entropy.detach().cpu()),
            "state_tracker/next_state_mse": dynamics_mse,
        }

    # ------------------------------------------------------------------
    # Dynamics diagnostics / loss
    # ------------------------------------------------------------------
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

    def supervised_dynamics_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """用离线真实 chunk 监督训练 StateTrackerDynamics。

        Args:
            batch (Dict[str, torch.Tensor]): ActionChunkDataset 输出 batch。

        Returns:
            torch.Tensor: next-state MSE loss；未配置 dynamics 时为 0。
        """

        if self.dynamics is None or self.dynamics_loss_weight <= 0:
            return torch.zeros((), device=self.device)
        target_next_states = self._batch_tensor(batch, "next_observations").float()
        chunk_actions = self._batch_tensor(batch, "actions").float().view(
            -1,
            self.chunk_size,
            self.action_dim,
        )
        chunk_rewards = self._batch_tensor(batch, "chunk_step_rewards").float()
        chunk_valid = torch.ones_like(chunk_rewards, dtype=torch.bool)
        predicted_next_states = self.dynamics.next_state(
            self._batch_tensor(batch, "history_vectors").float(),
            self._batch_tensor(batch, "history_valid").float(),
            chunk_actions,
            chunk_rewards,
            chunk_valid,
        )
        return F.mse_loss(predicted_next_states, target_next_states)

    # ------------------------------------------------------------------
    # Discrete rejection sampling helpers
    # ------------------------------------------------------------------
    def _item_ids_to_chunk_vectors(self, chunk_item_ids: torch.Tensor) -> torch.Tensor:
        """把 chunk 内的 item id 用 item embedding 表拼成 chunk 向量。

        Args:
            chunk_item_ids (torch.Tensor): chunk 内 item id，形状 `(..., K)`；
                非负 id 被查表，`-1` 视为占位（对应 embedding 置零）。

        Returns:
            torch.Tensor: chunk 向量，形状 `(..., K*action_dim)`。
        """

        chunk_item_ids_long = chunk_item_ids.to(dtype=torch.long, device=self.device)
        valid_mask = chunk_item_ids_long >= 0
        safe_ids = torch.where(
            valid_mask,
            chunk_item_ids_long,
            torch.zeros_like(chunk_item_ids_long),
        )
        embeddings = self.action_mapper.item_embeddings[safe_ids]
        embeddings = embeddings * valid_mask.unsqueeze(-1).to(embeddings.dtype)
        return embeddings.reshape(*chunk_item_ids_long.shape[:-1], self.chunk_size * self.action_dim)

    @torch.no_grad()
    def sample_candidate_chunks(
        self,
        states: torch.Tensor,
        num_samples: int,
        recommended_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从离散 categorical actor 采样多个候选 item chunk。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。
            num_samples (int): 每个状态采样的候选数量。
            recommended_mask (Optional[torch.Tensor]): 已推荐 item mask，形状
                `(B, num_items)`，True 表示不允许再采样。传入 mask 时，
                除屏蔽历史 item 外，还会逐位置更新候选 mask，保证每个
                候选 chunk 内 item 严格唯一；`None` 保留允许重复的旧语义。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `(item_ids, chunk_embeddings)`。
                - `item_ids`：形状 `(N, B, K)` 的 item id 张量。
                - `chunk_embeddings`：形状 `(N, B, K*action_dim)` 的 chunk 向量。

        Raises:
            ValueError: 当 `num_samples`、mask 形状非法，或剩余可用 item
                数量不足以构造无重复 chunk 时抛出。
        """

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        states = states.to(device=self.device, dtype=torch.float32)
        logits = self.actor(states)  # (B, K, N)
        batch_size = int(logits.shape[0])
        if recommended_mask is None:
            probs = torch.softmax(logits, dim=-1)
            flat_probs = probs.reshape(batch_size * self.chunk_size, self.num_items)
            sampled_flat = torch.multinomial(
                flat_probs,
                num_samples=num_samples,
                replacement=True,
            )  # (B*K, N_samples)
            sampled_ids = sampled_flat.view(
                batch_size,
                self.chunk_size,
                num_samples,
            )
            sampled_ids = sampled_ids.permute(2, 0, 1).contiguous()
        else:
            sampled_ids = self._sample_unique_candidate_chunks(
                logits=logits,
                num_samples=num_samples,
                recommended_mask=recommended_mask,
            )
        chunk_vectors = self._item_ids_to_chunk_vectors(sampled_ids)
        return sampled_ids, chunk_vectors

    @torch.no_grad()
    def _sample_unique_candidate_chunks(
        self,
        logits: torch.Tensor,
        num_samples: int,
        recommended_mask: torch.Tensor,
    ) -> torch.Tensor:
        """逐位置采样严格无重复的候选 chunks。

        每个候选 chunk 都维护独立 mask：初始 mask 来自当前 episode
        已执行历史；每采样一个位置后，立即把该位置的 item 加入该候选
        自己的 mask，再采样下一个位置。因此不同候选之间允许相同 item，
        但单个候选内部不允许重复。

        Args:
            logits (torch.Tensor): actor 输出，形状为
                `(batch_size, chunk_size, num_items)`。
            num_samples (int): 每个 batch 样本需要生成的候选 chunk 数，
                必须大于 0。
            recommended_mask (torch.Tensor): 历史已执行 item mask，形状为
                `(batch_size, num_items)`。

        Returns:
            torch.Tensor: 严格无重复的候选 item id，形状为
            `(num_samples, batch_size, chunk_size)`。

        Raises:
            ValueError: 当输入形状、候选数非法，或任一样本剩余可用 item
                少于 `chunk_size` 时抛出。

        Example:
            当历史 mask 屏蔽 item 0，且 `chunk_size=3` 时，每个返回候选
            都不会包含 0，并且三个位置的 item id 两两不同。
        """

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        expected_logits_shape = (
            int(recommended_mask.shape[0]),
            self.chunk_size,
            self.num_items,
        )
        if tuple(logits.shape) != expected_logits_shape:
            raise ValueError(
                "logits shape mismatch: expected "
                f"{expected_logits_shape}, got {tuple(logits.shape)}."
            )
        history_mask = recommended_mask.to(
            device=self.device,
            dtype=torch.bool,
        )
        expected_mask_shape = (int(logits.shape[0]), self.num_items)
        if tuple(history_mask.shape) != expected_mask_shape:
            raise ValueError(
                "recommended_mask shape mismatch: expected "
                f"{expected_mask_shape}, got {tuple(history_mask.shape)}."
            )

        available_counts = (~history_mask).sum(dim=1)
        insufficient_rows = torch.nonzero(
            available_counts < self.chunk_size,
            as_tuple=False,
        ).flatten()
        if insufficient_rows.numel() > 0:
            row_indices = insufficient_rows.detach().cpu().tolist()
            row_counts = available_counts[insufficient_rows].detach().cpu().tolist()
            raise ValueError(
                "Not enough unmasked items to sample unique chunks: "
                f"chunk_size={self.chunk_size}, rows={row_indices}, "
                f"available_counts={row_counts}."
            )

        batch_size = int(logits.shape[0])
        candidate_masks = history_mask.unsqueeze(0).expand(
            num_samples,
            -1,
            -1,
        ).clone()
        sampled_steps = []
        for step_index in range(self.chunk_size):
            # 每个候选使用同一位置的 actor logits，但拥有独立的动态 mask。
            step_logits = logits[:, step_index, :].unsqueeze(0).expand(
                num_samples,
                -1,
                -1,
            )
            masked_logits = step_logits.masked_fill(
                candidate_masks,
                _MASK_LOGIT_VALUE,
            )
            step_probs = torch.softmax(masked_logits, dim=-1).reshape(
                num_samples * batch_size,
                self.num_items,
            )
            sampled_step = torch.multinomial(
                step_probs,
                num_samples=1,
                replacement=False,
            ).view(num_samples, batch_size)
            sampled_steps.append(sampled_step)
            candidate_masks.scatter_(
                dim=2,
                index=sampled_step.unsqueeze(-1),
                value=True,
            )

        return torch.stack(sampled_steps, dim=-1).contiguous()

    @torch.no_grad()
    def select_chunks(
        self,
        states: torch.Tensor,
        num_samples: int,
        recommended_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """用 critic 从候选 chunks 中选择 Q 值最高的 chunk。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。
            num_samples (int): 候选 chunk 数量。
            recommended_mask (Optional[torch.Tensor]): 已推荐 item mask，形状
                `(B, num_items)`；`None` 表示不屏蔽。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: `(item_ids, chunk_embeddings)`；
                - `item_ids`：形状 `(B, K)`。
                - `chunk_embeddings`：形状 `(B, K*action_dim)`。
        """

        candidate_ids, candidate_chunks = self.sample_candidate_chunks(
            states,
            num_samples=num_samples,
            recommended_mask=recommended_mask,
        )
        num_samples_actual, batch_size, _ = candidate_chunks.shape
        repeated_states = states.unsqueeze(0).expand(num_samples_actual, -1, -1)
        q_values = self.critic(
            repeated_states.reshape(-1, self.state_dim),
            candidate_chunks.reshape(-1, self.chunk_action_dim),
        ).view(num_samples_actual, batch_size)
        best_indices = torch.argmax(q_values, dim=0)
        row_indices = torch.arange(batch_size, device=self.device)
        best_ids = candidate_ids[best_indices, row_indices]
        best_chunks = candidate_chunks[best_indices, row_indices]
        return best_ids, best_chunks

    # ------------------------------------------------------------------
    # Q/V update with chunk-level value expansion
    # ------------------------------------------------------------------
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
        """执行一次 chunk-level Q/V 更新。

        当 `rollout_depth == 1` 时退化为单 chunk 一步 TD，与历史实现完全一致；
        当 `rollout_depth > 1` 时使用 MAC 完整版的多步 imagined rollout + chunk-level
        value expansion 目标（沿 H 个 chunk 折扣累计 reward，末尾接 target V bootstrap）。

        Args:
            batch (Dict[str, torch.Tensor]): ActionChunkDataset 输出 batch。
            optimizer (torch.optim.Optimizer): critic/value 优化器。
            num_samples_train (int): rejection sampling 候选 chunk 数量。
            repeat_policy (str): 重复推荐处理策略，支持 `truncate` 和 `mask`。
            rollout_depth (int): imagined rollout 的 chunk 数 `H`，必须为正。
            lambda_chunk (float): chunk-level GAE 系数，取值范围 `[0, 1]`；
                `0` 近似单 chunk 一步 TD，`1` 为 chunk-level 蒙特卡洛。
            leave_policy (str): 退出规则处理策略，支持 `penalty`（违规只惩罚不截断）
                和 `terminate`（与 DORL 一致，触发退出即终止）。

        Returns:
            Dict[str, float]: 训练与 rollout 指标。

        Raises:
            RuntimeError: 当缺少 dynamics/reward/leave 模型时抛出。
            ValueError: 当 `rollout_depth` 或 `lambda_chunk` 非法时抛出。
        """

        if self.dynamics is None or self.reward_model is None or self.leave_model is None:
            raise RuntimeError("qv_update requires dynamics, reward_model and leave_model.")
        if rollout_depth <= 0:
            raise ValueError("rollout_depth must be positive.")
        if not 0.0 <= lambda_chunk <= 1.0:
            raise ValueError("lambda_chunk must be in [0, 1].")
        self.train()
        states = self._batch_tensor(batch, "observations").float()
        with torch.no_grad():
            trajectory = self.rollout_trajectory(
                batch,
                num_samples=num_samples_train,
                rollout_depth=rollout_depth,
                repeat_policy=repeat_policy,
                leave_policy=leave_policy,
            )
            target_v = self._chunk_value_expansion_target(trajectory, lambda_chunk=lambda_chunk)
            first_step = trajectory.steps[0]
            target_q = first_step.q_target

        optimizer.zero_grad(set_to_none=True)
        q_values = self.critic(states, first_step.selected_chunks)
        value_predictions = self.value(states)
        critic_loss = F.mse_loss(q_values, target_q)
        value_loss = F.mse_loss(value_predictions, target_v)
        dynamics_loss = self.supervised_dynamics_loss(batch)
        loss = critic_loss + value_loss + self.dynamics_loss_weight * dynamics_loss
        loss.backward()
        optimizer.step()
        self.soft_update_target_value()
        first_rollout = first_step.rollout
        return {
            "critic/critic_loss": float(critic_loss.detach().cpu()),
            "value/value_loss": float(value_loss.detach().cpu()),
            "state_tracker/dynamics_loss": float(dynamics_loss.detach().cpu()),
            "critic/q_mean": float(q_values.detach().mean().cpu()),
            "critic/target_q_mean": float(target_q.detach().mean().cpu()),
            "value/target_v_mean": float(target_v.detach().mean().cpu()),
            "rollout/depth": float(rollout_depth),
            "rollout/lambda_chunk": float(lambda_chunk),
            "rollout/trajectory_reward": float(trajectory.trajectory_returns.detach().mean().cpu()),
            "rollout/trajectory_chunks": float(trajectory.executed_chunks.float().detach().mean().cpu()),
            "rollout/reward_chunk": float(first_rollout.reward_chunks.detach().mean().cpu()),
            "rollout/pred_reward": self._masked_mean(first_rollout.pred_rewards, first_rollout.chunk_valid),
            "rollout/entropy": self._masked_mean(first_rollout.entropy_rewards, first_rollout.chunk_valid),
            "rollout/uncertainty": self._masked_mean(first_rollout.uncertainty_penalties, first_rollout.chunk_valid),
            "rollout/effective_steps": float(first_rollout.effective_steps.float().detach().mean().cpu()),
            "rollout/done_ratio": float(first_rollout.done_chunks.float().detach().mean().cpu()),
            "rollout/repeat_ratio": float(first_rollout.repeat_ratio.detach().cpu()),
            "rollout/exact_repeat_ratio": float(first_rollout.exact_repeat_ratio.detach().cpu()),
        }

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout_chunks(
        self,
        batch: Dict[str, torch.Tensor],
        chunk_item_ids: torch.Tensor,
        repeat_policy: str,
        leave_policy: str = LEAVE_POLICY_PENALTY,
    ) -> RolloutResult:
        """对完整 action chunk 逐步执行推荐系统 rollout（单 chunk 兼容接口）。

        Args:
            batch (Dict[str, torch.Tensor]): 起始状态与历史信息。
            chunk_item_ids (torch.Tensor): 每步选定的离散 item id，形状为 `(B, K)`。
            repeat_policy (str): 重复推荐处理策略。
            leave_policy (str): 退出规则处理策略，支持 `penalty` 和 `terminate`。

        Returns:
            RolloutResult: rollout 后的状态、reward、done 和统计信息。

        Raises:
            ValueError: 当重复策略不支持时抛出。
        """

        if repeat_policy not in {REPEAT_POLICY_TRUNCATE, REPEAT_POLICY_MASK}:
            raise ValueError(f"Unsupported repeat_policy: {repeat_policy}")
        self._batch_user_ids = self._batch_tensor(batch, "user_id").long().view(-1)
        history_state = self._initial_history_state(batch)
        result, _ = self._rollout_one_chunk(
            chunk_item_ids,
            history_state,
            repeat_policy=repeat_policy,
            leave_policy=leave_policy,
        )
        return result

    def _initial_history_state(self, batch: Dict[str, torch.Tensor]) -> RolloutHistoryState:
        """从 ActionChunkDataset batch 构造初始 rollout 历史缓存。

        Args:
            batch (Dict[str, torch.Tensor]): ActionChunkDataset 输出 batch。

        Returns:
            RolloutHistoryState: 初始历史缓存。
        """

        return RolloutHistoryState(
            history_vectors=self._batch_tensor(batch, "history_vectors").float().clone(),
            history_valid=self._batch_tensor(batch, "history_valid").float().clone(),
            leave_history=self._batch_tensor(batch, "leave_history_item_ids").long().clone(),
            recommended_mask=self._batch_tensor(batch, "recommended_mask").bool().clone(),
            env_step=self._batch_tensor(batch, "env_step").long().view(-1).clone(),
            terminated=self._batch_tensor(batch, "terminals").float().view(-1) > 0,
        )

    def _prepare_rollout_sampling_mask(
        self,
        history_state: RolloutHistoryState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """构造严格去重 rollout 的采样 mask，并处理候选耗尽样本。

        已经没有任何未推荐 item 的样本不能再产生有效动作。该函数先把这些
        样本标记为终止；随后仅为所有已终止行解除采样 mask，使批量候选采样
        保持张量形状一致。占位候选会在 `_rollout_one_chunk()` 中被
        `terminated` 状态屏蔽，因此不会执行、计 reward 或写回推荐历史。

        Args:
            history_state (RolloutHistoryState): 当前 imagined rollout 的历史状态。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 更新后的活跃行标记和供候选采样
            使用的 mask，二者形状分别为 `(B,)` 与 `(B, num_items)`。

        Raises:
            ValueError: 当推荐历史 mask 的形状与 agent item 数不一致时抛出。
        """

        recommended_mask = history_state.recommended_mask
        expected_mask_shape = (int(recommended_mask.shape[0]), self.num_items)
        if tuple(recommended_mask.shape) != expected_mask_shape:
            raise ValueError(
                "recommended_mask shape mismatch: expected "
                f"{expected_mask_shape}, got {tuple(recommended_mask.shape)}."
            )

        exhausted_rows = recommended_mask.all(dim=1)
        # 无合法候选的样本在本次及后续 rollout chunk 中均不应再执行动作。
        history_state.terminated = history_state.terminated | exhausted_rows
        active_rows = ~history_state.terminated
        sampling_mask = recommended_mask.clone()
        # 非活跃行的候选仅用于维持批量张量形状，后续执行路径会完全屏蔽它们。
        sampling_mask[~active_rows] = False
        return active_rows, sampling_mask

    def _reset_local_rollout_recommended_mask(
        self,
        history_state: RolloutHistoryState,
    ) -> None:
        """清空局部 imagined rollout 的 item 重复记录。

        Q/V 训练从任意离线状态切入，离线轨迹在 `start_index` 之前的 item
        仅用于构造状态和退出历史，不属于本次新生成的 imagined trajectory。
        因此严格去重只从当前 rollout 的第一步开始累计。

        Args:
            history_state (RolloutHistoryState): 将被原地清空推荐 item mask 的
                局部 rollout 历史状态。

        Returns:
            None.
        """

        history_state.recommended_mask = torch.zeros_like(
            history_state.recommended_mask,
            dtype=torch.bool,
        )

    @torch.no_grad()
    def _rollout_one_chunk(
        self,
        chunk_item_ids: torch.Tensor,
        history_state: RolloutHistoryState,
        repeat_policy: str,
        leave_policy: str = LEAVE_POLICY_PENALTY,
    ) -> tuple[RolloutResult, RolloutHistoryState]:
        """执行单个 chunk，直接消费离散 item id，不再走连续→离散映射。

        `repeat_policy` 语义在离散版下的作用略有差异：`mask` 时候选采样阶段
        同时屏蔽历史 item，并通过逐位置动态 mask 保证采样 chunk 内 item
        严格唯一；`truncate` 时不预先屏蔽。该函数仍检测 exact repeat，
        以兼容外部直接传入的任意 chunk 和离线行为数据。

        `leave_policy` 控制触发退出规则时的行为：
        - `penalty`：违规只在当前 step reward 上加 `invalid_action_penalty`，
          不截断 chunk，`done` 只由 `max_turn` 或离线 terminal 决定。
        - `terminate`：与原版 DORL `KuaiEnv.step` 一致——触发退出的这一步动作仍
          正常映射、计 reward、进 history（可叠加 penalty），但执行完该步后立即
          将该样本标记为终止，chunk 内后续 step 不再执行、多 chunk rollout 停止。

        Args:
            chunk_item_ids (torch.Tensor): 每步 item id，形状为 `(B, K)`。
            history_state (RolloutHistoryState): 当前 chunk 起点的历史缓存。
            repeat_policy (str): 重复推荐处理策略。
            leave_policy (str): 退出规则处理策略，支持 `penalty` 和 `terminate`。

        Returns:
            tuple[RolloutResult, RolloutHistoryState]: rollout 结果与更新后的历史缓存。

        Raises:
            ValueError: 当 `leave_policy` 不支持时抛出。
        """

        if leave_policy not in {LEAVE_POLICY_PENALTY, LEAVE_POLICY_TERMINATE}:
            raise ValueError(f"Unsupported leave_policy: {leave_policy}")
        assert self.dynamics is not None
        assert self.reward_model is not None
        assert self.leave_model is not None

        chunk_item_ids = chunk_item_ids.to(device=self.device, dtype=torch.long)
        batch_size = int(chunk_item_ids.shape[0])
        executed_action_embeddings = torch.zeros(
            batch_size,
            self.chunk_size,
            self.action_dim,
            device=self.device,
        )
        step_rewards = torch.zeros(batch_size, self.chunk_size, device=self.device)
        chunk_valid = torch.zeros(batch_size, self.chunk_size, dtype=torch.bool, device=self.device)
        selected_item_ids = torch.full(
            (batch_size, self.chunk_size),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        pred_rewards = torch.zeros(batch_size, self.chunk_size, device=self.device)
        entropy_rewards = torch.zeros(batch_size, self.chunk_size, device=self.device)
        uncertainty_penalties = torch.zeros(batch_size, self.chunk_size, device=self.device)
        reward_chunks = torch.zeros(batch_size, device=self.device)
        effective_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        exact_repeat_events = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        leave_violation_events = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        recommended_mask = history_state.recommended_mask.clone()
        leave_history = history_state.leave_history.clone()
        raw_user_ids = self._batch_user_ids
        rollout_start_steps = history_state.env_step.clone()
        already_terminated = history_state.terminated.clone()
        terminal_flags = already_terminated.clone()
        leave_terminated = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        row_indices = torch.arange(batch_size, device=self.device)

        for step_index in range(self.chunk_size):
            # `rollout_start_steps + step_index` 是当前 episode 内已实际执行的底层
            # action 数，不能使用离线长轨迹的绝对 start_index，否则会误判超过 max_turn。
            executable = (
                (rollout_start_steps + step_index < self.leave_model.max_turn)
                & (~already_terminated)
                & (~leave_terminated)
            )
            # 关键性能点：不再 `.item()` 提前判空跳过 —— 该 sync 会强制 CPU↔GPU 屏障，
            # 严重拖慢 chunk 循环。直接进入后续计算，让 batch 内无效样本走 mask 分支即可。
            item_ids = chunk_item_ids[:, step_index]
            # 「safe_item_ids」在 non-executable 样本上仍是合法 id（用 0 代替），
            # 便于对整 batch 做 vectorized 计算；最后用 executable mask 屏蔽结果。
            safe_item_ids = torch.where(
                executable,
                item_ids,
                torch.zeros_like(item_ids),
            )
            selected_item_ids[:, step_index] = torch.where(
                executable,
                item_ids,
                selected_item_ids[:, step_index],
            )

            repeated = recommended_mask[row_indices, safe_item_ids] & executable
            exact_repeat_events |= repeated
            # 对整 batch 一次调 leave_model；避免每 step masked select 造成 shape 抖动。
            leave_violation_full = (
                self.leave_model.first_violation_steps(
                    leave_history,
                    safe_item_ids.unsqueeze(1),
                ) == 0
            ) & executable
            leave_violation_events |= leave_violation_full

            history_with_current = self.leave_model.append_items(
                leave_history,
                safe_item_ids,
                executable,
            )
            # reward 计算需要正整数 item id；对 non-executable 样本仍传 safe id，
            # 之后用 executable mask 屏蔽其贡献。
            reward_components = self.reward_model.reward_components(
                raw_user_ids,
                safe_item_ids,
                history_item_ids=history_with_current,
            )
            rewards_full = reward_components["reward"]
            penalty_mask_full = repeated | leave_violation_full
            rewards_full = rewards_full + penalty_mask_full.float() * self.invalid_action_penalty
            executable_f = executable.to(dtype=rewards_full.dtype)
            discount_full = self.gamma ** effective_steps.float()
            reward_chunks = reward_chunks + executable_f * discount_full * rewards_full
            step_rewards[:, step_index] = torch.where(
                executable,
                rewards_full,
                step_rewards[:, step_index],
            )
            pred_rewards[:, step_index] = torch.where(
                executable,
                reward_components["pred_reward"],
                pred_rewards[:, step_index],
            )
            entropy_rewards[:, step_index] = torch.where(
                executable,
                reward_components["entropy"],
                entropy_rewards[:, step_index],
            )
            uncertainty_penalties[:, step_index] = torch.where(
                executable,
                reward_components["uncertainty"],
                uncertainty_penalties[:, step_index],
            )
            chunk_valid[:, step_index] = chunk_valid[:, step_index] | executable
            step_embeddings = self.action_mapper.item_embeddings[safe_item_ids]
            executed_action_embeddings[:, step_index] = torch.where(
                executable.unsqueeze(-1),
                step_embeddings,
                executed_action_embeddings[:, step_index],
            )
            effective_steps = effective_steps + executable.long()
            # 用 scatter 而非 fancy index，避免 non-executable 位置误写。
            update_mask = torch.zeros_like(recommended_mask)
            update_mask[row_indices, safe_item_ids] = executable
            recommended_mask = recommended_mask | update_mask
            leave_history = history_with_current

            if leave_policy == LEAVE_POLICY_TERMINATE:
                # 与 DORL 一致：触发退出的这一步动作照常执行、计 reward、进 history，
                # 但执行完后该样本立即终止，后续 step 不再执行。
                leave_terminated = leave_terminated | leave_violation_full
        # `repeat_policy` 只影响候选采样阶段是否屏蔽已推荐 item；本函数在采样之后消费
        # 已确定的 item id，因此不再需要在 step 循环内判定 repeat_policy 分支。
        _ = repeat_policy

        done_chunks = terminal_flags | leave_terminated | (
            rollout_start_steps + effective_steps >= self.leave_model.max_turn
        )
        next_states = self.dynamics.next_state(
            history_state.history_vectors,
            history_state.history_valid,
            executed_action_embeddings,
            step_rewards,
            chunk_valid,
        )
        repeat_ratio = leave_violation_events.float().mean()
        exact_repeat_ratio = exact_repeat_events.float().mean()
        result = RolloutResult(
            next_states=next_states,
            reward_chunks=reward_chunks,
            done_chunks=done_chunks,
            effective_steps=effective_steps,
            selected_item_ids=selected_item_ids,
            step_rewards=step_rewards,
            chunk_valid=chunk_valid,
            pred_rewards=pred_rewards,
            entropy_rewards=entropy_rewards,
            uncertainty_penalties=uncertainty_penalties,
            repeat_ratio=repeat_ratio,
            exact_repeat_ratio=exact_repeat_ratio,
        )
        next_history_vectors, next_history_valid = self._update_history_vectors(
            history_state.history_vectors,
            history_state.history_valid,
            executed_action_embeddings,
            step_rewards,
            chunk_valid,
        )
        next_history_state = RolloutHistoryState(
            history_vectors=next_history_vectors,
            history_valid=next_history_valid,
            leave_history=leave_history,
            recommended_mask=recommended_mask,
            env_step=rollout_start_steps + effective_steps,
            terminated=done_chunks,
        )
        return result, next_history_state

    def _update_history_vectors(
        self,
        history_vectors: torch.Tensor,
        history_valid: torch.Tensor,
        executed_action_embeddings: torch.Tensor,
        step_rewards: torch.Tensor,
        chunk_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """把本 chunk 实际执行的 `(action_emb, reward)` 追加进 StateTracker 历史窗口。

        该更新遵循"重算范式"：把真实映射出的 item 表征追加到历史序列尾部并保留最近
        `window_size` 项，供下一个 chunk 的 `StateTrackerDynamics` 重新求平均，而不是
        把 dynamics 自身预测的 state 向量喂回输入，因此多 chunk rollout 不累积 dynamics 误差。

        Args:
            history_vectors (torch.Tensor): 当前历史向量，形状 `(B, W, state_dim)`。
            history_valid (torch.Tensor): 当前历史有效标记，形状 `(B, W)`。
            executed_action_embeddings (torch.Tensor): 本 chunk 执行的 action embedding，
                形状 `(B, K, action_dim)`；未执行位置为 0。
            step_rewards (torch.Tensor): 本 chunk 每步 reward，形状 `(B, K)`。
            chunk_valid (torch.Tensor): 每步是否实际执行，形状 `(B, K)`。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 更新后的 `(history_vectors, history_valid)`。
        """

        window_size = int(history_vectors.shape[1])
        batch_size = int(history_vectors.shape[0])
        chunk_vectors = torch.cat([executed_action_embeddings, step_rewards.unsqueeze(-1)], dim=-1)
        history_valid_bool = history_valid.bool()
        chunk_valid_bool = chunk_valid.bool()

        combined_vectors = torch.cat([history_vectors, chunk_vectors], dim=1)  # (B, W+K, D)
        combined_valid_bool = torch.cat([history_valid_bool, chunk_valid_bool], dim=1)  # (B, W+K)
        combined_valid_f = combined_valid_bool.to(dtype=history_valid.dtype)
        masked_vectors = combined_vectors * combined_valid_f.unsqueeze(-1)

        # 用 reverse cumsum 标记"最近 window_size 个有效位置"。
        valid_int = combined_valid_bool.to(torch.int64)
        reverse_cum = torch.flip(torch.cumsum(torch.flip(valid_int, dims=[1]), dim=1), dims=[1])
        recent_mask = (reverse_cum <= window_size) & combined_valid_bool  # (B, W+K)

        # 每个 batch 的最近有效条数 <= window_size；把它们按顺序 pack 到长度 window_size
        # 的输出末尾。做法：对 recent_mask 内每个 True 分配一个 rank（从左到右计数），
        # 再算出每个 True 应该落到目标索引 window_size - total_count + rank。
        total_count = recent_mask.sum(dim=1)  # (B,)
        left_cum = torch.cumsum(recent_mask.to(torch.int64), dim=1)  # (B, W+K)
        offsets = (window_size - total_count).unsqueeze(1)  # (B, 1)
        target_pos = offsets + left_cum - 1  # (B, W+K)  仅 True 位置有意义
        target_pos = torch.where(
            recent_mask,
            target_pos,
            torch.full_like(target_pos, window_size),  # 无效位置指向哨兵槽
        )

        new_vectors_ext = torch.zeros(
            batch_size,
            window_size + 1,
            combined_vectors.shape[-1],
            dtype=combined_vectors.dtype,
            device=combined_vectors.device,
        )
        new_valid_ext = torch.zeros(
            batch_size,
            window_size + 1,
            dtype=history_valid.dtype,
            device=history_valid.device,
        )
        idx_vec = target_pos.unsqueeze(-1).expand(-1, -1, combined_vectors.shape[-1])
        new_vectors_ext.scatter_(1, idx_vec, masked_vectors)
        new_valid_ext.scatter_(1, target_pos, recent_mask.to(dtype=history_valid.dtype))

        updated_vectors = new_vectors_ext[:, :window_size, :].contiguous()
        updated_valid = new_valid_ext[:, :window_size].contiguous()
        return updated_vectors, updated_valid

    @torch.no_grad()
    def rollout_trajectory(
        self,
        batch: Dict[str, torch.Tensor],
        num_samples: int,
        rollout_depth: int,
        repeat_policy: str = REPEAT_POLICY_TRUNCATE,
        leave_policy: str = LEAVE_POLICY_PENALTY,
    ) -> TrajectoryRollout:
        """在模型内想象出一条 H 个 chunk 的 rollout（MAC value expansion 用）。

        每个 chunk 边界都用 `select_chunks`（rejection sampling on discrete
        item id）在行为分布内重新选择价值最高的 chunk，再用 `StateTrackerDynamics
        + DORLRewardModel + RuleBasedLeaveModel` 执行该 chunk 并推进历史缓存。
        全程不与真实环境交互，是 on-policy 的模型内 imagined rollout。
        `leave_policy=terminate` 时，某个 chunk 内触发退出会让该样本终止，
        后续 chunk 不再 roll（与 DORL 一致）。

        Args:
            batch (Dict[str, torch.Tensor]): ActionChunkDataset 输出 batch。
            num_samples (int): 每个 chunk 边界 rejection sampling 候选数。
            rollout_depth (int): imagined rollout 的 chunk 数 `H`，必须为正。
            repeat_policy (str): 重复推荐处理策略；`mask` 只屏蔽本次 imagined
                rollout 已执行 item 和当前候选 chunk 内已选择 item。
            leave_policy (str): 退出规则处理策略，支持 `penalty` 和 `terminate`。

        Returns:
            TrajectoryRollout: 逐 chunk 决策步与整条轨迹统计。

        Raises:
            ValueError: 当 `rollout_depth` 或 repeat 策略非法时抛出。
        """

        if rollout_depth <= 0:
            raise ValueError("rollout_depth must be positive.")
        if repeat_policy not in {REPEAT_POLICY_TRUNCATE, REPEAT_POLICY_MASK}:
            raise ValueError(f"Unsupported repeat_policy: {repeat_policy}")

        self._batch_user_ids = self._batch_tensor(batch, "user_id").long().view(-1)
        history_state = self._initial_history_state(batch)
        if repeat_policy == REPEAT_POLICY_MASK:
            self._reset_local_rollout_recommended_mask(history_state)
        batch_size = int(history_state.env_step.shape[0])
        steps: list = []
        trajectory_returns = torch.zeros(batch_size, device=self.device)
        executed_chunks = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        cumulative_discount = torch.ones(batch_size, device=self.device)

        for _ in range(rollout_depth):
            if repeat_policy == REPEAT_POLICY_MASK:
                active, candidate_mask = self._prepare_rollout_sampling_mask(history_state)
            else:
                # truncate 策略允许重复推荐，不能因历史 mask 已满而提前终止。
                active = ~history_state.terminated
                candidate_mask = None
            # rollout_depth 通常很小（默认 1）；跳过 `.item()` 同步以让 GPU 流水连续。
            states = self.dynamics.next_state(
                history_state.history_vectors,
                history_state.history_valid,
                torch.zeros(batch_size, self.chunk_size, self.action_dim, device=self.device),
                torch.zeros(batch_size, self.chunk_size, device=self.device),
                torch.zeros(batch_size, self.chunk_size, dtype=torch.bool, device=self.device),
            )
            selected_item_ids, selected_chunks = self.select_chunks(
                states,
                num_samples=num_samples,
                recommended_mask=candidate_mask,
            )
            selected_item_ids = selected_item_ids.detach()
            selected_chunks = selected_chunks.detach()
            result, next_history_state = self._rollout_one_chunk(
                selected_item_ids,
                history_state,
                repeat_policy=repeat_policy,
                leave_policy=leave_policy,
            )
            chunk_discount = self.gamma ** result.effective_steps.float()
            next_values = self.target_value(result.next_states)
            q_target = result.reward_chunks + (1.0 - result.done_chunks.float()) * chunk_discount * next_values
            steps.append(
                TrajectoryStep(
                    states=states,
                    selected_chunks=selected_chunks,
                    selected_item_ids=selected_item_ids,
                    rollout=result,
                    active=active,
                    chunk_discount=chunk_discount,
                    q_target=q_target,
                )
            )
            trajectory_returns = trajectory_returns + cumulative_discount * result.reward_chunks * active.float()
            executed_chunks = executed_chunks + active.long()
            cumulative_discount = cumulative_discount * chunk_discount
            history_state = next_history_state

        return TrajectoryRollout(
            steps=steps,
            trajectory_returns=trajectory_returns,
            executed_chunks=executed_chunks,
        )

    def _chunk_value_expansion_target(
        self,
        trajectory: TrajectoryRollout,
        lambda_chunk: float,
    ) -> torch.Tensor:
        """用 imagined 轨迹构造第一个 chunk 起点状态的 chunk-level value expansion 目标。

        采用 chunk-level TD(λ)：把折扣从 `gamma` 提升为每个 chunk 的
        `gamma^effective_steps`，沿 rollout 反向累计，`lambda_chunk=0` 近似单 chunk
        一步 TD，`lambda_chunk=1` 为 chunk-level 蒙特卡洛。该目标仅用于回归 V/Q，
        不参与任何策略梯度（MAC 是 value-based）。

        Args:
            trajectory (TrajectoryRollout): `rollout_trajectory` 的结果。
            lambda_chunk (float): chunk-level GAE 系数，取值范围 `[0, 1]`。

        Returns:
            torch.Tensor: 第一个 chunk 起点状态的 value 目标，形状 `(B,)`。
        """

        steps = trajectory.steps
        first_step = steps[0]
        batch_size = int(first_step.states.shape[0])
        gae = torch.zeros(batch_size, device=self.device)
        for step in reversed(steps):
            next_value = self.target_value(step.rollout.next_states)
            delta = (
                step.rollout.reward_chunks
                + (1.0 - step.rollout.done_chunks.float()) * step.chunk_discount * next_value
                - self.target_value(step.states)
            )
            not_done = (1.0 - step.rollout.done_chunks.float())
            gae = delta + lambda_chunk * step.chunk_discount * not_done * gae
        target_v = gae + self.target_value(first_step.states)
        return target_v.detach()

    @torch.no_grad()
    def soft_update_target_value(self) -> None:
        """按 `target_tau` 软更新 target value 网络。

        Returns:
            None.
        """

        for target_param, source_param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.mul_(1.0 - self.target_tau).add_(source_param.data, alpha=self.target_tau)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def checkpoint_state(self, config: Dict[str, object]) -> Dict[str, object]:
        """构造可保存的 checkpoint 字典。

        Args:
            config (Dict[str, object]): 本次实验配置快照。

        Returns:
            Dict[str, object]: 可传给 `torch.save` 的 checkpoint。
        """

        return {
            "config": dict(config),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "value": self.value.state_dict(),
            "target_value": self.target_value.state_dict(),
            "dynamics": self.dynamics.state_dict() if self.dynamics is not None else None,
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

        if "actor" not in checkpoint:
            raise KeyError("checkpoint missing key: actor")
        self.actor.load_state_dict(checkpoint["actor"], strict=strict)
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"], strict=strict)
        if "value" in checkpoint:
            self.value.load_state_dict(checkpoint["value"], strict=strict)
        if "target_value" in checkpoint:
            self.target_value.load_state_dict(checkpoint["target_value"], strict=strict)
        else:
            self.target_value.load_state_dict(self.value.state_dict(), strict=strict)
        if self.dynamics is not None and checkpoint.get("dynamics") is not None:
            self.dynamics.load_state_dict(checkpoint["dynamics"], strict=strict)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        """计算有效 step 上的均值。

        Args:
            values (torch.Tensor): 待统计数值。
            mask (torch.Tensor): bool 有效标记。

        Returns:
            float: 有效位置均值；没有有效位置时返回 0。
        """

        valid_values = values.detach()[mask.detach().bool()]
        if valid_values.numel() == 0:
            return 0.0
        return float(valid_values.mean().cpu())

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
