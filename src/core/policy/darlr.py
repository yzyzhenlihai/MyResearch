"""DARLR 双智能体推荐策略实现。"""

import math
import sys
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

sys.path.extend(["./src/tianshou"])

from tianshou.data import Batch, ReplayBuffer, to_torch_as  # noqa: E402
from tianshou.policy import A2CPolicy  # noqa: E402
from tianshou.utils.net.common import ActorCritic  # noqa: E402


NEGATIVE_MASK_VALUE = -1.0e9
"""候选用户已被选择后，在 softmax 前使用的屏蔽分数。"""

MIN_DENOMINATOR = 1.0e-8
"""动态不确定性分母的最小稳定值。"""


class PreferenceEncoder(nn.Module):
    """将用户预测偏好向量投影到 selector 使用的低维空间。

    该模块负责把 `predicted_mat[user, :]` 从物品维度压缩到
    `selector_pref_dim`，供 selector actor、critic 和状态编码器复用。

    Args:
        num_items (int): 物品数量，即预测偏好向量长度，必须大于 0。
        pref_dim (int): 投影后的偏好维度，必须大于 0。

    Raises:
        ValueError: 当 `num_items` 或 `pref_dim` 非正时抛出。
    """

    def __init__(self, num_items: int, pref_dim: int) -> None:
        super().__init__()
        if num_items <= 0:
            raise ValueError("num_items must be positive.")
        if pref_dim <= 0:
            raise ValueError("pref_dim must be positive.")
        self.net = nn.Sequential(
            nn.Linear(num_items, pref_dim),
            nn.ReLU(),
            nn.Linear(pref_dim, pref_dim),
        )

    def forward(self, preferences: torch.Tensor) -> torch.Tensor:
        """编码用户预测偏好向量。

        Args:
            preferences (torch.Tensor): 用户偏好矩阵，形状为
                `(batch_size, num_items)` 或 `(batch_size, candidate_size, num_items)`。

        Returns:
            torch.Tensor: 编码后的偏好张量，最后一维为 `pref_dim`。

        Raises:
            RuntimeError: 当输入最后一维与线性层不匹配时由 PyTorch 抛出。
        """

        return self.net(preferences)


class SelectorStateEncoder(nn.Module):
    """用 Transformer 编码已选参考用户序列。

    该模块模拟 DARLR 论文中 selector 的状态转移。每次 selector 新选择一个
    reference user 后，将其偏好表示加入序列，再通过 Transformer 得到集合状态。

    Args:
        pref_dim (int): 参考用户偏好表示维度。
        max_len (int): selector 最多选择的参考用户数量。
        num_heads (int): Transformer attention head 数量。
        num_layers (int): Transformer encoder 层数。
        dropout_rate (float): Dropout 概率。

    Raises:
        ValueError: 当关键维度或层数非法时抛出。
    """

    def __init__(
        self,
        pref_dim: int,
        max_len: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        if pref_dim <= 0:
            raise ValueError("pref_dim must be positive.")
        if max_len <= 0:
            raise ValueError("max_len must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if pref_dim % num_heads != 0:
            raise ValueError("pref_dim must be divisible by num_heads.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        self.pref_dim = pref_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, pref_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pref_dim,
            nhead=num_heads,
            dim_feedforward=pref_dim * 2,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        selected_embeddings: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """编码已选 reference users。

        Args:
            selected_embeddings (torch.Tensor): 已选用户偏好表示，形状为
                `(batch_size, max_len, pref_dim)`。
            selected_mask (torch.Tensor): 有效位置掩码，形状为
                `(batch_size, max_len)`，True 表示该位置已有参考用户。

        Returns:
            torch.Tensor: selector 集合状态，形状为 `(batch_size, pref_dim)`。

        Raises:
            ValueError: 当输入张量维度不符合预期时抛出。
        """

        if selected_embeddings.ndim != 3:
            raise ValueError("selected_embeddings must be a 3-D tensor.")
        if selected_mask.ndim != 2:
            raise ValueError("selected_mask must be a 2-D tensor.")
        batch_size, seq_len, _ = selected_embeddings.shape
        if seq_len != self.max_len:
            raise ValueError("selected_embeddings seq_len must equal max_len.")

        positions = torch.arange(seq_len, device=selected_embeddings.device)
        encoded_input = selected_embeddings + self.position_embedding(positions)
        key_padding_mask = ~selected_mask.bool()

        # Transformer 在全空序列上会产生无意义输出，因此空序列直接返回零向量。
        has_selected = selected_mask.any(dim=1)
        encoded = torch.zeros(batch_size, self.pref_dim, device=selected_embeddings.device)
        if has_selected.any():
            active_input = encoded_input[has_selected]
            active_padding = key_padding_mask[has_selected]
            transformer_output = self.encoder(
                active_input,
                src_key_padding_mask=active_padding,
            )
            active_mask = selected_mask[has_selected].float().unsqueeze(-1)
            masked_output = transformer_output * active_mask
            denominator = active_mask.sum(dim=1).clamp_min(MIN_DENOMINATOR)
            encoded[has_selected] = masked_output.sum(dim=1) / denominator
        return encoded


class SelectorActor(nn.Module):
    """selector actor，在候选用户子集内输出选择分布。

    Args:
        recommender_state_dim (int): recommender 状态表示维度。
        pref_dim (int): 用户偏好表示维度。
        hidden_sizes (Sequence[int]): 上下文 MLP 隐层维度。
    """

    def __init__(
        self,
        recommender_state_dim: int,
        pref_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        context_dim = recommender_state_dim + pref_dim * 2
        layers: List[nn.Module] = []
        last_dim = context_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, pref_dim))
        self.query_net = nn.Sequential(*layers)

    def forward(
        self,
        recommender_state: torch.Tensor,
        current_preference: torch.Tensor,
        selected_state: torch.Tensor,
        candidate_preferences: torch.Tensor,
    ) -> torch.Tensor:
        """计算候选 reference users 的选择 logits。

        Args:
            recommender_state (torch.Tensor): recommender 当前状态，形状为
                `(batch_size, state_dim)`。
            current_preference (torch.Tensor): 当前用户偏好表示，形状为
                `(batch_size, pref_dim)`。
            selected_state (torch.Tensor): 已选用户集合状态，形状为
                `(batch_size, pref_dim)`。
            candidate_preferences (torch.Tensor): 候选用户偏好表示，形状为
                `(batch_size, candidate_size, pref_dim)`。

        Returns:
            torch.Tensor: 候选用户 logits，形状为 `(batch_size, candidate_size)`。

        Raises:
            RuntimeError: 当张量形状不兼容时由 PyTorch 抛出。
        """

        context = torch.cat(
            [recommender_state, current_preference, selected_state],
            dim=-1,
        )
        query = self.query_net(context)
        logits = torch.sum(candidate_preferences * query.unsqueeze(1), dim=-1)
        return logits / math.sqrt(candidate_preferences.shape[-1])


class SelectorCritic(nn.Module):
    """selector critic，估计每个选择步骤的状态价值。

    Args:
        recommender_state_dim (int): recommender 状态表示维度。
        pref_dim (int): 用户偏好表示维度。
        hidden_sizes (Sequence[int]): critic MLP 隐层维度。
    """

    def __init__(
        self,
        recommender_state_dim: int,
        pref_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        context_dim = recommender_state_dim + pref_dim * 2
        layers: List[nn.Module] = []
        last_dim = context_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.value_net = nn.Sequential(*layers)

    def forward(
        self,
        recommender_state: torch.Tensor,
        current_preference: torch.Tensor,
        selected_state: torch.Tensor,
    ) -> torch.Tensor:
        """估计 selector 当前状态价值。

        Args:
            recommender_state (torch.Tensor): recommender 当前状态，形状为
                `(batch_size, state_dim)`。
            current_preference (torch.Tensor): 当前用户偏好表示，形状为
                `(batch_size, pref_dim)`。
            selected_state (torch.Tensor): 已选用户集合状态，形状为
                `(batch_size, pref_dim)`。

        Returns:
            torch.Tensor: 状态价值，形状为 `(batch_size,)`。
        """

        context = torch.cat(
            [recommender_state, current_preference, selected_state],
            dim=-1,
        )
        return self.value_net(context).squeeze(-1)


class DARLRPolicy(A2CPolicy):
    """DARLR 双智能体策略。

    该策略复用 DORL 的 recommender A2C，并在训练采样阶段额外运行 selector。
    selector 在候选用户子集内选择 reference users，将选择结果写入
    `Batch.policy.darlr_context`，由动态奖励环境用于 reward shaping。
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        optim: Sequence[torch.optim.Optimizer],
        dist_fn: Any,
        state_tracker: nn.Module,
        predicted_mat: np.ndarray,
        selector_actor: SelectorActor,
        selector_critic: SelectorCritic,
        preference_encoder: PreferenceEncoder,
        selector_state_encoder: SelectorStateEncoder,
        selector_k: int,
        selector_candidate_size: int,
        selector_candidate_mode: str,
        selector_lambda_s: float,
        selector_lambda_d: float,
        selector_reward_mode: str,
        darlr_eps: float,
        selector_discount_factor: float,
        **kwargs: Any,
    ) -> None:
        """初始化 DARLRPolicy。

        Args:
            actor (nn.Module): recommender actor。
            critic (nn.Module): recommender critic。
            optim (Sequence[torch.optim.Optimizer]): 优化器列表，`optim[0]`
                更新 recommender 与 selector，`optim[1]` 更新 state tracker。
            dist_fn (Any): recommender 动作分布类。
            state_tracker (nn.Module): 推荐状态追踪器。
            predicted_mat (np.ndarray): world model 预测矩阵。
            selector_actor (SelectorActor): selector actor。
            selector_critic (SelectorCritic): selector critic。
            preference_encoder (PreferenceEncoder): 偏好编码器。
            selector_state_encoder (SelectorStateEncoder): 参考用户集合编码器。
            selector_k (int): 每个推荐步选择的参考用户数量。
            selector_candidate_size (int): 候选用户子集大小。
            selector_candidate_mode (str): 候选池生成模式，支持 `embedding_topk`
                和 `random`。
            selector_lambda_s (float): 相似性增益权重。
            selector_lambda_d (float): 多样性增益权重。
            selector_reward_mode (str): selector reward 消融模式。
            darlr_eps (float): 数值稳定项。
            selector_discount_factor (float): selector 内部回报折扣因子。
            **kwargs (Any): 透传给 A2CPolicy 的参数。

        Raises:
            ValueError: 当 selector 相关超参数非法时抛出。
        """

        if selector_k <= 0:
            raise ValueError("selector_k must be positive.")
        if selector_candidate_size <= 0:
            raise ValueError("selector_candidate_size must be positive.")
        if selector_candidate_mode not in {"embedding_topk", "random"}:
            raise ValueError("Unsupported selector_candidate_mode.")
        if selector_reward_mode not in {"full", "base", "sim", "div"}:
            raise ValueError("Unsupported selector_reward_mode.")
        super().__init__(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            state_tracker=state_tracker,
            **kwargs,
        )
        try:
            selector_device = next(preference_encoder.parameters()).device
        except StopIteration:
            selector_device = torch.device("cpu")
        # predicted_mat 会频繁参与 selector 前向，需与 selector 网络保持同一设备。
        predicted_tensor = torch.as_tensor(
            predicted_mat,
            dtype=torch.float32,
            device=selector_device,
        )
        self.register_buffer("predicted_mat", predicted_tensor)
        self.selector_actor = selector_actor
        self.selector_critic = selector_critic
        self.preference_encoder = preference_encoder
        self.selector_state_encoder = selector_state_encoder
        self.selector_k = selector_k
        self.selector_candidate_size = selector_candidate_size
        self.selector_candidate_mode = selector_candidate_mode
        self.selector_lambda_s = selector_lambda_s
        self.selector_lambda_d = selector_lambda_d
        self.selector_reward_mode = selector_reward_mode
        self.darlr_eps = max(float(darlr_eps), MIN_DENOMINATOR)
        self.selector_discount_factor = selector_discount_factor
        self._actor_critic = ActorCritic(self.actor, self.critic)

    def forward(
        self,
        batch: Batch,
        buffer: Optional[ReplayBuffer],
        indices: np.ndarray = None,
        is_obs: Optional[bool] = None,
        is_train: bool = True,
        state: Optional[Any] = None,
        use_batch_in_statetracker: bool = False,
        collect_selector: Optional[bool] = None,
        **kwargs: Any,
    ) -> Batch:
        """执行 recommender 前向，并在训练采样时执行 selector。

        Args:
            batch (Batch): 当前采样 batch，需包含 `obs`、`mask`。
            buffer (Optional[ReplayBuffer]): 轨迹 buffer。
            indices (np.ndarray): buffer 中的当前位置。
            is_obs (Optional[bool]): 是否构造当前状态。
            is_train (bool): 当前是否处于训练 collector。
            state (Optional[Any]): recurrent hidden state，当前实现透传给 actor。
            use_batch_in_statetracker (bool): 是否使用 collector 当前 batch 构造状态。
            collect_selector (Optional[bool]): 是否显式运行 selector；为 None 时
                仅在训练采样阶段运行。
            **kwargs (Any): 兼容外部调用的额外参数。

        Returns:
            Batch: 包含 recommender `act`、`dist`、`logits` 以及可选
            `policy.darlr_context` 的结果。
        """

        obs_emb = self.state_tracker(
            buffer=buffer,
            indices=indices,
            is_obs=is_obs,
            batch=batch,
            is_train=is_train,
            use_batch_in_statetracker=use_batch_in_statetracker,
        )
        batch_info = getattr(batch, "info", {})
        logits, hidden = self.actor(obs_emb, state=state, info=batch_info)
        if self.action_type == "discrete":
            mask_name = "mask" if is_obs else "next_mask"
            action_mask = getattr(batch, mask_name, None)
            if action_mask is not None:
                logits = logits * action_mask
        dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            act = logits.argmax(-1)
        else:
            act = dist.sample()

        should_collect_selector = (
            is_train and self.training if collect_selector is None else collect_selector
        )
        policy_batch = Batch()
        if should_collect_selector:
            policy_batch.darlr_context = self._select_reference_users(
                recommender_state=obs_emb.detach(),
                batch=batch,
                recommender_action=act.detach(),
            )
        return Batch(
            logits=logits,
            act=act,
            state=hidden,
            dist=dist,
            policy=policy_batch,
        )

    def learn(
        self,
        batch: Batch,
        batch_size: int,
        repeat: int,
        **kwargs: Any,
    ) -> dict:
        """更新 recommender A2C 与 selector A2C。

        Args:
            batch (Batch): 已经由 `process_fn()` 处理过的训练 batch。
            batch_size (int): minibatch 大小。
            repeat (int): 重复更新次数。
            **kwargs (Any): 兼容 trainer 传入的额外参数。

        Returns:
            dict: 训练损失日志，包含 recommender 与 selector loss。
        """

        losses: List[float] = []
        actor_losses: List[float] = []
        value_losses: List[float] = []
        entropy_losses: List[float] = []
        selector_losses: List[float] = []
        selector_actor_losses: List[float] = []
        selector_value_losses: List[float] = []
        selector_entropy_losses: List[float] = []

        optim_rl, optim_state = self.optim
        trainable_modules = nn.ModuleList(
            [
                self.actor,
                self.critic,
                self.selector_actor,
                self.selector_critic,
                self.preference_encoder,
                self.selector_state_encoder,
            ]
        )

        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                result = self(
                    minibatch,
                    self._buffer,
                    indices=minibatch.indices,
                    is_obs=True,
                    collect_selector=False,
                )
                log_prob = result.dist.log_prob(minibatch.act)
                log_prob = log_prob.reshape(len(minibatch.adv), -1).transpose(0, 1)
                actor_loss = -(log_prob * minibatch.adv).mean()

                obs_emb = self.state_tracker(
                    self._buffer,
                    minibatch.indices,
                    is_obs=True,
                )
                value = self.critic(obs_emb).flatten()
                recommender_value_loss = F.mse_loss(minibatch.returns, value)
                entropy_loss = result.dist.entropy().mean()
                recommender_loss = (
                    actor_loss
                    + self._weight_vf * recommender_value_loss
                    - self._weight_ent * entropy_loss
                )

                selector_loss_tuple = self._compute_selector_loss(
                    minibatch=minibatch,
                    recommender_state=obs_emb.detach(),
                )
                selector_loss = selector_loss_tuple[0]
                total_loss = recommender_loss + selector_loss

                optim_rl.zero_grad()
                optim_state.zero_grad()
                total_loss.backward()
                if self._grad_norm:
                    nn.utils.clip_grad_norm_(
                        trainable_modules.parameters(),
                        max_norm=self._grad_norm,
                    )
                optim_rl.step()
                optim_state.step()

                losses.append(total_loss.item())
                actor_losses.append(actor_loss.item())
                value_losses.append(recommender_value_loss.item())
                entropy_losses.append(entropy_loss.item())
                selector_losses.append(selector_loss.item())
                selector_actor_losses.append(selector_loss_tuple[1].item())
                selector_value_losses.append(selector_loss_tuple[2].item())
                selector_entropy_losses.append(selector_loss_tuple[3].item())

        return {
            "loss": losses,
            "loss/actor": actor_losses,
            "loss/vf": value_losses,
            "loss/ent": entropy_losses,
            "loss/selector": selector_losses,
            "loss/selector_actor": selector_actor_losses,
            "loss/selector_vf": selector_value_losses,
            "loss/selector_ent": selector_entropy_losses,
        }

    def _select_reference_users(
        self,
        recommender_state: torch.Tensor,
        batch: Batch,
        recommender_action: torch.Tensor,
    ) -> Batch:
        """为当前推荐动作选择 reference users 并构造环境上下文。

        Args:
            recommender_state (torch.Tensor): recommender 状态，形状为
                `(batch_size, state_dim)`。
            batch (Batch): collector 当前 batch，`obs[:, 0]` 为当前用户。
            recommender_action (torch.Tensor): recommender 采样的物品 id。

        Returns:
            Batch: 写入 replay buffer 和环境的 DARLR 上下文。
        """

        current_users = self._extract_current_users(batch).to(self.predicted_mat.device)
        recommender_action = recommender_action.to(self.predicted_mat.device).long()
        candidate_users = self._build_candidate_users(current_users)
        current_preferences = self._encode_user_preferences(current_users)
        candidate_preferences = self._encode_user_preferences(candidate_users)

        selected_embeddings = torch.zeros(
            current_users.shape[0],
            self.selector_k,
            current_preferences.shape[-1],
            device=self.predicted_mat.device,
        )
        selected_mask = torch.zeros(
            current_users.shape[0],
            self.selector_k,
            dtype=torch.bool,
            device=self.predicted_mat.device,
        )

        selected_users: List[torch.Tensor] = []
        selector_actions: List[torch.Tensor] = []
        selector_rewards: List[torch.Tensor] = []
        similarity_values: List[torch.Tensor] = []
        diversity_values: List[torch.Tensor] = []
        selected_candidate_indices: List[torch.Tensor] = []

        for step_index in range(self.selector_k):
            selected_state = self.selector_state_encoder(
                selected_embeddings,
                selected_mask,
            )
            logits = self.selector_actor(
                recommender_state.to(self.predicted_mat.device),
                current_preferences,
                selected_state,
                candidate_preferences,
            )
            logits = self._mask_selected_candidates(logits, selected_candidate_indices)
            distribution = Categorical(logits=logits)
            action_index = distribution.sample()
            reference_users = candidate_users[
                torch.arange(candidate_users.shape[0], device=candidate_users.device),
                action_index,
            ]
            reference_embeddings = candidate_preferences[
                torch.arange(candidate_users.shape[0], device=candidate_users.device),
                action_index,
            ]

            similarity_gain = self._compute_similarity_gain(current_users, reference_users)
            diversity_gain = self._compute_diversity_gain(reference_users, selected_users)
            base_reward = self.predicted_mat[current_users, recommender_action]
            selector_reward = self._compute_selector_reward(
                base_reward,
                similarity_gain,
                diversity_gain,
            )

            selected_embeddings[:, step_index, :] = reference_embeddings
            selected_mask[:, step_index] = True
            selected_users.append(reference_users)
            selector_actions.append(action_index)
            selector_rewards.append(selector_reward)
            similarity_values.append(similarity_gain)
            diversity_values.append(diversity_gain)
            selected_candidate_indices.append(action_index)

        selected_users_tensor = torch.stack(selected_users, dim=1)
        selector_actions_tensor = torch.stack(selector_actions, dim=1)
        selector_rewards_tensor = torch.stack(selector_rewards, dim=1)
        similarity_tensor = torch.stack(similarity_values, dim=1)
        diversity_tensor = torch.stack(diversity_values, dim=1)

        return Batch(
            current_users=self._to_numpy(current_users),
            recommender_actions=self._to_numpy(recommender_action),
            candidate_users=self._to_numpy(candidate_users),
            selected_users=self._to_numpy(selected_users_tensor),
            selector_actions=self._to_numpy(selector_actions_tensor),
            selector_rewards=self._to_numpy(selector_rewards_tensor),
            similarity_gain=self._to_numpy(similarity_tensor.mean(dim=1)),
            diversity_gain=self._to_numpy(diversity_tensor.mean(dim=1)),
        )

    def _compute_selector_loss(
        self,
        minibatch: Batch,
        recommender_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据 replay buffer 中保存的 selector 轨迹计算 A2C 损失。

        Args:
            minibatch (Batch): 训练 minibatch，需包含 `policy.darlr_context`。
            recommender_state (torch.Tensor): 当前 recommender 状态表示。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            总 selector loss、actor loss、value loss、entropy loss。
        """

        if not hasattr(minibatch.policy, "darlr_context"):
            zero_loss = recommender_state.sum() * 0.0
            return zero_loss, zero_loss, zero_loss, zero_loss

        context = minibatch.policy.darlr_context
        current_users = self._as_long_tensor(context.current_users)
        candidate_users = self._as_long_tensor(context.candidate_users)
        selected_users = self._as_long_tensor(context.selected_users)
        selector_actions = self._as_long_tensor(context.selector_actions)
        selector_rewards = self._as_float_tensor(context.selector_rewards)

        if selector_rewards.ndim != 2 or selector_rewards.shape[1] != self.selector_k:
            zero_loss = recommender_state.sum() * 0.0
            return zero_loss, zero_loss, zero_loss, zero_loss

        current_preferences = self._encode_user_preferences(current_users)
        candidate_preferences = self._encode_user_preferences(candidate_users)
        selector_returns = self._discounted_returns(selector_rewards)

        selected_embeddings = torch.zeros(
            current_users.shape[0],
            self.selector_k,
            current_preferences.shape[-1],
            device=self.predicted_mat.device,
        )
        selected_mask = torch.zeros(
            current_users.shape[0],
            self.selector_k,
            dtype=torch.bool,
            device=self.predicted_mat.device,
        )
        selected_candidate_indices: List[torch.Tensor] = []

        actor_losses: List[torch.Tensor] = []
        value_losses: List[torch.Tensor] = []
        entropy_losses: List[torch.Tensor] = []

        for step_index in range(self.selector_k):
            selected_state = self.selector_state_encoder(
                selected_embeddings,
                selected_mask,
            )
            logits = self.selector_actor(
                recommender_state.to(self.predicted_mat.device),
                current_preferences,
                selected_state,
                candidate_preferences,
            )
            logits = self._mask_selected_candidates(logits, selected_candidate_indices)
            distribution = Categorical(logits=logits)
            step_actions = selector_actions[:, step_index]
            values = self.selector_critic(
                recommender_state.to(self.predicted_mat.device),
                current_preferences,
                selected_state,
            )
            step_returns = selector_returns[:, step_index]
            advantages = step_returns - values.detach()
            actor_losses.append(-(distribution.log_prob(step_actions) * advantages).mean())
            value_losses.append(F.mse_loss(values, step_returns))
            entropy_losses.append(distribution.entropy().mean())

            reference_users = selected_users[:, step_index]
            selected_embeddings[:, step_index, :] = self._encode_user_preferences(reference_users)
            selected_mask[:, step_index] = True
            selected_candidate_indices.append(step_actions)

        actor_loss = torch.stack(actor_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy_loss = torch.stack(entropy_losses).mean()
        total_loss = actor_loss + self._weight_vf * value_loss - self._weight_ent * entropy_loss
        return total_loss, actor_loss, value_loss, entropy_loss

    def _extract_current_users(self, batch: Batch) -> torch.Tensor:
        """从环境观测中提取当前用户 id。

        Args:
            batch (Batch): collector 或 replay buffer 中的 batch。

        Returns:
            torch.Tensor: 当前用户 id，形状为 `(batch_size,)`。
        """

        obs = torch.as_tensor(batch.obs, device=self.predicted_mat.device)
        if obs.ndim == 1:
            return obs.long()
        return obs[:, 0].long()

    def _build_candidate_users(self, current_users: torch.Tensor) -> torch.Tensor:
        """为每个样本构造 selector 候选用户池。

        Args:
            current_users (torch.Tensor): 当前用户 id，形状为 `(batch_size,)`。

        Returns:
            torch.Tensor: 候选用户 id，形状为 `(batch_size, candidate_size)`。
        """

        num_users = self.predicted_mat.shape[0]
        candidate_size = min(self.selector_candidate_size, max(1, num_users - 1))
        if self.selector_candidate_mode == "embedding_topk":
            user_embeddings = self._get_user_embeddings(num_users)
            if user_embeddings is not None:
                norm_embeddings = F.normalize(user_embeddings, dim=-1)
                current_embeddings = norm_embeddings[current_users]
                scores = current_embeddings @ norm_embeddings.t()
                scores.scatter_(1, current_users.view(-1, 1), NEGATIVE_MASK_VALUE)
                return torch.topk(scores, k=candidate_size, dim=1).indices
        return self._random_candidate_users(current_users, num_users, candidate_size)

    def _get_user_embeddings(self, num_users: int) -> Optional[torch.Tensor]:
        """读取 state tracker 中的用户 embedding。

        Args:
            num_users (int): predicted matrix 中的用户数量。

        Returns:
            Optional[torch.Tensor]: 可用时返回用户 embedding，否则返回 None。
        """

        embedding_dict = getattr(self.state_tracker, "embedding_dict", None)
        if embedding_dict is None or not hasattr(embedding_dict, "feat_user"):
            return None
        user_embeddings = embedding_dict.feat_user.weight
        if user_embeddings.shape[0] < num_users:
            return None
        return user_embeddings[:num_users].detach().to(self.predicted_mat.device)

    def _random_candidate_users(
        self,
        current_users: torch.Tensor,
        num_users: int,
        candidate_size: int,
    ) -> torch.Tensor:
        """随机生成候选用户池。

        Args:
            current_users (torch.Tensor): 当前用户 id。
            num_users (int): 用户总数。
            candidate_size (int): 候选池大小。

        Returns:
            torch.Tensor: 随机候选用户 id。
        """

        candidates = torch.randint(
            low=0,
            high=num_users,
            size=(current_users.shape[0], candidate_size),
            device=self.predicted_mat.device,
        )
        current_expanded = current_users.view(-1, 1)
        candidates = torch.where(
            candidates == current_expanded,
            (candidates + 1) % num_users,
            candidates,
        )
        return candidates

    def _encode_user_preferences(self, user_ids: torch.Tensor) -> torch.Tensor:
        """编码一个或一组用户的预测偏好。

        Args:
            user_ids (torch.Tensor): 用户 id，形状可以为 `(batch_size,)` 或
                `(batch_size, candidate_size)`。

        Returns:
            torch.Tensor: 编码后的用户偏好。
        """

        user_ids = user_ids.to(self.predicted_mat.device).long()
        preferences = self.predicted_mat[user_ids]
        return self.preference_encoder(preferences)

    def _mask_selected_candidates(
        self,
        logits: torch.Tensor,
        selected_candidate_indices: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """屏蔽已经选择过的候选用户，避免 selector 重复选择。

        Args:
            logits (torch.Tensor): 当前候选用户 logits。
            selected_candidate_indices (Sequence[torch.Tensor]): 历史选择的候选索引。

        Returns:
            torch.Tensor: 屏蔽后的 logits。
        """

        if not selected_candidate_indices:
            return logits
        masked_logits = logits.clone()
        for selected_index in selected_candidate_indices:
            masked_logits.scatter_(1, selected_index.view(-1, 1), NEGATIVE_MASK_VALUE)
        return masked_logits

    def _compute_similarity_gain(
        self,
        current_users: torch.Tensor,
        reference_users: torch.Tensor,
    ) -> torch.Tensor:
        """计算当前用户与参考用户的相似性增益。

        Args:
            current_users (torch.Tensor): 当前用户 id。
            reference_users (torch.Tensor): 参考用户 id。

        Returns:
            torch.Tensor: 映射到 `[0, 1]` 的 cosine similarity。
        """

        current_pref = self.predicted_mat[current_users]
        reference_pref = self.predicted_mat[reference_users]
        cosine = F.cosine_similarity(current_pref, reference_pref, dim=-1)
        return (cosine + 1.0) * 0.5

    def _compute_diversity_gain(
        self,
        reference_users: torch.Tensor,
        selected_users: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """计算新参考用户相对已选集合的多样性增益。

        Args:
            reference_users (torch.Tensor): 新选择的参考用户 id。
            selected_users (Sequence[torch.Tensor]): 已选择的参考用户 id 列表。

        Returns:
            torch.Tensor: 多样性增益，越大表示越不同。
        """

        if not selected_users:
            return torch.ones(reference_users.shape[0], device=self.predicted_mat.device)
        reference_pref = self.predicted_mat[reference_users]
        diversity_terms = []
        for previous_users in selected_users:
            previous_pref = self.predicted_mat[previous_users]
            cosine = F.cosine_similarity(reference_pref, previous_pref, dim=-1)
            similarity = (cosine + 1.0) * 0.5
            diversity_terms.append(1.0 - similarity)
        return torch.stack(diversity_terms, dim=1).mean(dim=1)

    def _compute_selector_reward(
        self,
        base_reward: torch.Tensor,
        similarity_gain: torch.Tensor,
        diversity_gain: torch.Tensor,
    ) -> torch.Tensor:
        """根据消融模式计算 selector intrinsic reward。

        Args:
            base_reward (torch.Tensor): world model 基础预测奖励。
            similarity_gain (torch.Tensor): 相似性增益。
            diversity_gain (torch.Tensor): 多样性增益。

        Returns:
            torch.Tensor: selector 当前 step 的 intrinsic reward。
        """

        if self.selector_reward_mode == "base":
            return base_reward
        if self.selector_reward_mode == "sim":
            return base_reward + self.selector_lambda_s * similarity_gain
        if self.selector_reward_mode == "div":
            return base_reward + self.selector_lambda_d * diversity_gain
        return (
            base_reward
            + self.selector_lambda_s * similarity_gain
            + self.selector_lambda_d * diversity_gain
        )

    def _discounted_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """计算 selector 选择序列内部的折扣回报。

        Args:
            rewards (torch.Tensor): selector rewards，形状为 `(batch_size, K)`。

        Returns:
            torch.Tensor: 折扣回报，形状与输入一致。
        """

        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(rewards.shape[0], device=rewards.device)
        for step_index in reversed(range(rewards.shape[1])):
            running_return = rewards[:, step_index] + self.selector_discount_factor * running_return
            returns[:, step_index] = running_return
        return returns

    def _as_long_tensor(self, value: Any) -> torch.Tensor:
        """将 replay buffer 中的值转成 long tensor。

        Args:
            value (Any): numpy、torch 或可转换对象。

        Returns:
            torch.Tensor: 位于策略设备上的 long tensor。
        """

        return torch.as_tensor(value, device=self.predicted_mat.device).long()

    def _as_float_tensor(self, value: Any) -> torch.Tensor:
        """将 replay buffer 中的值转成 float tensor。

        Args:
            value (Any): numpy、torch 或可转换对象。

        Returns:
            torch.Tensor: 位于策略设备上的 float tensor。
        """

        return torch.as_tensor(value, device=self.predicted_mat.device).float()

    @staticmethod
    def _to_numpy(value: torch.Tensor) -> np.ndarray:
        """将 tensor 转为 numpy 数组。

        Args:
            value (torch.Tensor): 待转换张量。

        Returns:
            np.ndarray: CPU numpy 数组。
        """

        return value.detach().cpu().numpy()
