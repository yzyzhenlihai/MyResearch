"""DARLR 双智能体推荐策略实现。

本模块严格按照 `docs/DARLR_DORL融合复现方案.md` 中列出的 P0 修正
项落地：

* 首个 diversity 按论文定义为 0（原实现返回 1）。
* `paper_core` 模式下 cosine 相似性/多样性使用原始 cosine，不做
  `[0,1]` 映射；`stabilized` 模式保留旧行为便于消融。
* 动态奖励 `r̂_t` 与动态不确定性 `P'_U` 的计算统一放在策略侧，
  环境只做 exposure/entropy/clip 后处理，从 `darlr_context` 读取
  已计算好的量。
* previous reward 由整个 run 共享的 `DynamicRewardStore` 维护，
  支持 checkpoint / resume 并可诊断 batch 内重复更新。
* 静态不确定性消融使用世界模型 ensemble 的 `V0[u,i]`（`maxvar_mat`），
  而不是 `abs(dynamic-previous)`。
* 候选用户池无放回采样，排除自身，越界立即断言。
* selector 与 recommender 拆成两个独立 optimizer，可分别设置 lr 与
  梯度裁剪；总 loss 中 selector 项使用 `selector_loss_coef` 系数。
* selector context 支持在 Collector 获得最终动作后重新聚合 reward，
  修复原有 `exploration_noise` 前生成 context 的动作对齐隐患。
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

sys.path.extend(["./src/tianshou"])

from tianshou.data import Batch, ReplayBuffer  # noqa: E402
from tianshou.policy import A2CPolicy  # noqa: E402
from tianshou.utils.net.common import ActorCritic  # noqa: E402

from src.core.darlr.dynamic_reward_store import DynamicRewardStore  # noqa: E402
from src.core.darlr.normalization import standardize_tensor  # noqa: E402
from src.core.darlr.selector_metrics import WeightedScalarAccumulator  # noqa: E402


NEGATIVE_MASK_VALUE = -1.0e9
"""屏蔽已选候选用户时使用的 logits 值。"""

MIN_DENOMINATOR = 1.0e-8
"""动态不确定性分母的下界。"""

TOP_CANDIDATE_RANK_CUTOFF = 10
"""统计 selector 选择候选池前若干名用户比例时使用的排名阈值。"""


# ---------------------------------------------------------------------------
# 网络模块 (与上一版相同，仅为完整性保留)
# ---------------------------------------------------------------------------


class PreferenceEncoder(nn.Module):
    """将用户预测偏好 (R0 中一整行) 投影到 selector 使用的低维空间."""

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
        return self.net(preferences)


class SelectorStateEncoder(nn.Module):
    """用 Transformer 编码已选参考用户序列 (集合状态)."""

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
        # 空集合状态使用可学习 CLS embedding，避免全零输入导致 Transformer
        # 在全 padding 时输出不稳定。
        self.empty_state = nn.Parameter(torch.zeros(1, pref_dim))
        nn.init.normal_(self.empty_state, mean=0.0, std=0.02)
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

        has_selected = selected_mask.any(dim=1)
        # 全空序列直接返回可学习 empty embedding。
        encoded = self.empty_state.expand(batch_size, -1).to(selected_embeddings.device)
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
            encoded = encoded.clone()
            encoded[has_selected] = masked_output.sum(dim=1) / denominator
        return encoded


class SelectorActor(nn.Module):
    """selector actor: 在候选用户子集内输出选择 logits."""

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
        context = torch.cat(
            [recommender_state, current_preference, selected_state],
            dim=-1,
        )
        query = self.query_net(context)
        logits = torch.sum(candidate_preferences * query.unsqueeze(1), dim=-1)
        return logits / math.sqrt(candidate_preferences.shape[-1])


class SelectorCritic(nn.Module):
    """selector critic: 估计 selector 每一步的状态价值."""

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
        context = torch.cat(
            [recommender_state, current_preference, selected_state],
            dim=-1,
        )
        return self.value_net(context).squeeze(-1)


# ---------------------------------------------------------------------------
# 主策略
# ---------------------------------------------------------------------------


class DARLRPolicy(A2CPolicy):
    """DARLR 双智能体策略 (recommender A2C + selector A2C)."""

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
        selector_gain_mode: str = "paper_core",
        selector_loss_coef: float = 1.0,
        selector_policy_mode: str = "learned",
        selector_ent_coef: Optional[float] = None,
        selector_reward_normalization: bool = False,
        selector_advantage_normalization: bool = False,
        selector_normalization_eps: float = MIN_DENOMINATOR,
        dynamic_reward_store: Optional[DynamicRewardStore] = None,
        maxvar_mat: Optional[np.ndarray] = None,
        optim_selector: Optional[torch.optim.Optimizer] = None,
        **kwargs: Any,
    ) -> None:
        if selector_k <= 0:
            raise ValueError("selector_k must be positive.")
        if selector_candidate_size <= 0:
            raise ValueError("selector_candidate_size must be positive.")
        if selector_candidate_size < selector_k:
            raise ValueError(
                "selector_candidate_size must be >= selector_k, "
                f"got {selector_candidate_size} < {selector_k}."
            )
        if selector_candidate_mode not in {"embedding_topk", "random"}:
            raise ValueError("Unsupported selector_candidate_mode.")
        if selector_reward_mode not in {"full", "base", "sim", "div"}:
            raise ValueError("Unsupported selector_reward_mode.")
        if selector_gain_mode not in {"paper_core", "stabilized"}:
            raise ValueError("Unsupported selector_gain_mode.")
        if selector_policy_mode not in {"learned", "random", "fixed"}:
            raise ValueError(
                "selector_policy_mode must be one of: learned, random, fixed."
            )
        if selector_loss_coef < 0:
            raise ValueError("selector_loss_coef must be non-negative.")
        if selector_ent_coef is not None and selector_ent_coef < 0:
            raise ValueError("selector_ent_coef must be non-negative.")
        if selector_normalization_eps <= 0:
            raise ValueError("selector_normalization_eps must be positive.")

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
        predicted_tensor = torch.as_tensor(
            predicted_mat,
            dtype=torch.float32,
            device=selector_device,
        )
        self.register_buffer("predicted_mat", predicted_tensor)
        # 为动态奖励聚合准备 CPU numpy 视图，避免每步都从 GPU 搬运。
        self._predicted_mat_np = np.asarray(predicted_mat, dtype=np.float32)
        if maxvar_mat is not None:
            if maxvar_mat.shape != predicted_mat.shape:
                raise ValueError(
                    "maxvar_mat.shape must match predicted_mat.shape, "
                    f"got {maxvar_mat.shape} vs {predicted_mat.shape}."
                )
            maxvar_tensor = torch.as_tensor(
                maxvar_mat,
                dtype=torch.float32,
                device=selector_device,
            )
            self.register_buffer("maxvar_mat", maxvar_tensor)
            self._maxvar_mat_np = np.asarray(maxvar_mat, dtype=np.float32)
        else:
            self.maxvar_mat = None  # type: ignore[assignment]
            self._maxvar_mat_np = None

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
        self.selector_gain_mode = selector_gain_mode
        self.selector_loss_coef = float(selector_loss_coef)
        self.selector_policy_mode = selector_policy_mode
        self.selector_ent_coef = (
            float(self._weight_ent)
            if selector_ent_coef is None
            else float(selector_ent_coef)
        )
        self.selector_reward_normalization = bool(
            selector_reward_normalization
        )
        self.selector_advantage_normalization = bool(
            selector_advantage_normalization
        )
        self.selector_normalization_eps = float(selector_normalization_eps)
        self.darlr_eps = max(float(darlr_eps), MIN_DENOMINATOR)
        self.selector_discount_factor = selector_discount_factor
        self.dynamic_reward_store = dynamic_reward_store
        self._optim_selector = optim_selector
        self._actor_critic = ActorCritic(self.actor, self.critic)
        self._selector_training_metrics = WeightedScalarAccumulator()

    def reset_selector_training_metrics(self) -> None:
        """清空当前 epoch 的 selector 训练诊断指标。

        Returns:
            None: 聚合器被原地重置。
        """

        self._selector_training_metrics.reset()

    def get_selector_training_metrics(self) -> Dict[str, float]:
        """导出当前 epoch 可上传 SwanLab 的 selector 训练指标。

        Returns:
            Dict[str, float]: 使用 ``selector/train`` 前缀的标量指标。
        """

        return self._selector_training_metrics.summary()

    # ---------------------------------------------------------------------
    # forward / selector context 生命周期
    # ---------------------------------------------------------------------

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

    def recompute_context_with_final_actions(
        self,
        context: Batch,
        final_actions: np.ndarray,
    ) -> Batch:
        """在 Collector 得到最终动作后，用最终动作重算 reward 相关量。

        Args:
            context (Batch): `forward()` 中生成的 darlr context。
            final_actions (np.ndarray): 经过 `map_action` (+ 可选
                exploration_noise) 之后真正送入环境的物品 id。

        Returns:
            Batch: 更新过 `recommender_actions/base_reward/selector_rewards/
            dynamic_reward/previous_reward/dynamic_uncertainty` 等字段的 context。
        """
        if context is None or len(context.keys()) == 0:
            return context

        final_actions_np = np.asarray(final_actions, dtype=np.int64).reshape(-1)
        cur_actions = np.asarray(context.recommender_actions, dtype=np.int64).reshape(-1)
        if final_actions_np.shape != cur_actions.shape:
            raise ValueError(
                "final_actions length mismatch with darlr_context, "
                f"got {final_actions_np.shape} vs {cur_actions.shape}."
            )
        # 无论 action 是否与采样一致, 都在此处提交 store: `_select_reference_users`
        # 阶段没有提交, Collector 拿到最终动作后统一 commit 一次。
        return self._finalize_context(context, final_actions_np, commit_store=True)

    # ---------------------------------------------------------------------
    # 训练更新
    # ---------------------------------------------------------------------

    def learn(
        self,
        batch: Batch,
        batch_size: int,
        repeat: int,
        **kwargs: Any,
    ) -> Dict[str, List[float]]:
        losses: List[float] = []
        actor_losses: List[float] = []
        value_losses: List[float] = []
        entropy_losses: List[float] = []
        selector_losses: List[float] = []
        selector_actor_losses: List[float] = []
        selector_value_losses: List[float] = []
        selector_entropy_losses: List[float] = []

        optim_rec, optim_state = self.optim  # 由 run_DARLR 保证长度为 2
        rec_modules = nn.ModuleList([self.actor, self.critic, self.state_tracker])
        selector_modules = nn.ModuleList(
            [
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
                selector_diagnostics = dict(selector_loss_tuple[4])

                # ---- recommender + state tracker backward ----
                optim_rec.zero_grad()
                optim_state.zero_grad()
                recommender_loss.backward(retain_graph=True)
                if self._grad_norm:
                    nn.utils.clip_grad_norm_(rec_modules.parameters(), max_norm=self._grad_norm)
                optim_rec.step()
                optim_state.step()

                # ---- selector backward ----
                selector_optimizer = (
                    self._optim_selector
                    if self._optim_selector is not None
                    else optim_rec
                )
                selector_optimizer.zero_grad()
                selector_is_trainable = (
                    self.selector_policy_mode == "learned"
                    and self.selector_loss_coef > 0
                )
                effective_selector_loss_coef = (
                    self.selector_loss_coef if selector_is_trainable else 0.0
                )
                if selector_is_trainable:
                    (effective_selector_loss_coef * selector_loss).backward()
                    if self._grad_norm:
                        gradient_norm_before_clip = float(
                            nn.utils.clip_grad_norm_(
                                selector_modules.parameters(),
                                max_norm=self._grad_norm,
                            ).item()
                        )
                        gradient_norm_after_clip = min(
                            gradient_norm_before_clip,
                            float(self._grad_norm),
                        )
                    else:
                        gradient_norm_before_clip = (
                            self._compute_module_gradient_norm(selector_modules)
                        )
                        gradient_norm_after_clip = gradient_norm_before_clip
                    selector_optimizer.step()
                else:
                    gradient_norm_before_clip = 0.0
                    gradient_norm_after_clip = 0.0

                total_loss = (
                    recommender_loss.detach()
                    + effective_selector_loss_coef * selector_loss.detach()
                )
                selector_diagnostics.update(
                    {
                        "loss_total": float(selector_loss.detach().item()),
                        "loss_actor": float(selector_loss_tuple[1].detach().item()),
                        "loss_value": float(selector_loss_tuple[2].detach().item()),
                        "entropy": float(selector_loss_tuple[3].detach().item()),
                        "gradient_norm_before_clip": gradient_norm_before_clip,
                        "gradient_norm_after_clip": gradient_norm_after_clip,
                        "gradient_clip_applied_rate": float(
                            self._grad_norm is not None
                            and gradient_norm_before_clip > self._grad_norm
                        ),
                        "policy_trainable_rate": float(selector_is_trainable),
                        "effective_loss_coef": float(
                            effective_selector_loss_coef
                        ),
                        "learning_rate": float(
                            selector_optimizer.param_groups[0]["lr"]
                        ),
                    }
                )
                self._selector_training_metrics.update(
                    selector_diagnostics,
                    weight=len(minibatch),
                )

                losses.append(float(total_loss.item()))
                actor_losses.append(float(actor_loss.item()))
                value_losses.append(float(recommender_value_loss.item()))
                entropy_losses.append(float(entropy_loss.item()))
                selector_losses.append(float(selector_loss.item()))
                selector_actor_losses.append(float(selector_loss_tuple[1].item()))
                selector_value_losses.append(float(selector_loss_tuple[2].item()))
                selector_entropy_losses.append(float(selector_loss_tuple[3].item()))

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

    # ---------------------------------------------------------------------
    # selector 采样
    # ---------------------------------------------------------------------

    def _select_reference_users(
        self,
        recommender_state: torch.Tensor,
        batch: Batch,
        recommender_action: torch.Tensor,
    ) -> Batch:
        """采样 K 个参考用户，返回 selector context (含 reward 相关量)."""
        current_users = self._extract_current_users(batch).to(self.predicted_mat.device)
        recommender_action = recommender_action.to(self.predicted_mat.device).long()
        self._assert_user_ids_valid(current_users)
        self._assert_action_ids_valid(recommender_action)

        candidate_users = self._build_candidate_users(current_users)
        self._assert_user_ids_valid(candidate_users)
        # 候选池必须排除自身，且不含重复。
        assert (candidate_users != current_users.view(-1, 1)).all(), (
            "Candidate pool contains current user; check candidate builder."
        )
        _assert_no_duplicate_along_last_dim(candidate_users, "candidate_users")

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
        similarity_values: List[torch.Tensor] = []
        diversity_values: List[torch.Tensor] = []
        selected_candidate_indices: List[torch.Tensor] = []

        for step_index in range(self.selector_k):
            selected_state = self.selector_state_encoder(selected_embeddings, selected_mask)
            if self.selector_policy_mode == "random":
                logits = torch.zeros(
                    current_users.shape[0],
                    candidate_users.shape[1],
                    device=self.predicted_mat.device,
                )
            else:
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

            selected_embeddings[:, step_index, :] = reference_embeddings
            selected_mask[:, step_index] = True
            selected_users.append(reference_users)
            selector_actions.append(action_index)
            similarity_values.append(similarity_gain)
            diversity_values.append(diversity_gain)
            selected_candidate_indices.append(action_index)

        selected_users_tensor = torch.stack(selected_users, dim=1)
        selector_actions_tensor = torch.stack(selector_actions, dim=1)
        similarity_tensor = torch.stack(similarity_values, dim=1)
        diversity_tensor = torch.stack(diversity_values, dim=1)

        # 用初始 recommender_action 生成 base/dynamic/uncertainty；
        # Collector 之后会用最终动作调 `recompute_context_with_final_actions`。
        pre_context = Batch(
            current_users=self._to_numpy(current_users),
            recommender_actions=self._to_numpy(recommender_action),
            candidate_users=self._to_numpy(candidate_users),
            selected_users=self._to_numpy(selected_users_tensor),
            selector_actions=self._to_numpy(selector_actions_tensor),
            similarity_per_step=self._to_numpy(similarity_tensor),
            diversity_per_step=self._to_numpy(diversity_tensor),
        )
        # 用采样动作先算一版 base/dynamic/uncertainty, 但不提交 store。
        # Collector 拿到最终动作后, 会调用 recompute_context_with_final_actions,
        # 那时才提交 store, 确保 previous_reward 语义与最终训练动作一致。
        return self._finalize_context(
            pre_context,
            self._to_numpy(recommender_action),
            commit_store=False,
        )

    # ---------------------------------------------------------------------
    # 动态奖励 / 不确定性 (集中在策略侧)
    # ---------------------------------------------------------------------

    def _finalize_context(
        self,
        context: Batch,
        final_actions: np.ndarray,
        commit_store: bool = False,
    ) -> Batch:
        """基于最终动作补全 base/dynamic/uncertainty 与 selector_rewards.

        Args:
            context (Batch): 已包含 current_users / selected_users /
                selector_actions / similarity_per_step / diversity_per_step
                的 pre-context.
            final_actions (np.ndarray): 每个样本对应的最终物品 id。
            commit_store (bool): 是否将本次动态奖励提交到共享 store。
                只应在 Collector 拿到最终真实动作后调用一次, 避免采样阶段
                和最终动作阶段的双重提交污染 previous reward。
        """
        current_users = np.asarray(context.current_users, dtype=np.int64).reshape(-1)
        selected_users = np.asarray(context.selected_users, dtype=np.int64)
        similarity_per_step = np.asarray(context.similarity_per_step, dtype=np.float32)
        diversity_per_step = np.asarray(context.diversity_per_step, dtype=np.float32)

        # 边界校验，避免 np.clip 静默修正。
        num_users, num_items = self.predicted_mat.shape
        if final_actions.min() < 0 or final_actions.max() >= num_items:
            raise IndexError(
                "final_actions out of item id range: "
                f"min={int(final_actions.min())}, max={int(final_actions.max())}, "
                f"num_items={num_items}."
            )
        if selected_users.min() < 0 or selected_users.max() >= num_users:
            raise IndexError(
                "selected_users out of user id range: "
                f"min={int(selected_users.min())}, max={int(selected_users.max())}, "
                f"num_users={num_users}."
            )
        if current_users.min() < 0 or current_users.max() >= num_users:
            raise IndexError(
                "current_users out of user id range: "
                f"min={int(current_users.min())}, max={int(current_users.max())}, "
                f"num_users={num_users}."
            )

        # base_reward = R0[u, a]
        base_reward = self._predicted_mat_np[current_users, final_actions].astype(np.float32)

        # dynamic_reward = mean(R0[U_S, a])
        expanded_actions = np.broadcast_to(
            final_actions.reshape(-1, 1),
            selected_users.shape,
        )
        dynamic_matrix = self._predicted_mat_np[selected_users, expanded_actions]
        dynamic_reward = dynamic_matrix.mean(axis=1).astype(np.float32)

        # previous_reward: shared store; 第一次访问回退到 R0[u,a]
        if self.dynamic_reward_store is not None:
            previous_reward = self.dynamic_reward_store.read(current_users, final_actions)
            if commit_store:
                self.dynamic_reward_store.stage_and_commit(
                    current_users, final_actions, dynamic_reward
                )
        else:
            previous_reward = base_reward.copy()

        # uncertainty numerator / denominator (paper_core)
        similarity_set = similarity_per_step.mean(axis=1)
        diversity_set = diversity_per_step.mean(axis=1)
        numerator = np.abs(dynamic_reward - previous_reward).astype(np.float32)
        denominator = np.maximum(similarity_set + diversity_set, self.darlr_eps).astype(np.float32)
        uncertainty_dynamic = (numerator / denominator).astype(np.float32)

        # 静态 uncertainty (消融用): 直接读 V0[u,a]
        if self._maxvar_mat_np is not None:
            uncertainty_static = self._maxvar_mat_np[current_users, final_actions].astype(np.float32)
        else:
            uncertainty_static = np.zeros_like(uncertainty_dynamic)

        # per-step selector rewards (使用 base_reward 与 sim/div 增益)
        base_broadcast = base_reward.reshape(-1, 1).astype(np.float32)
        if self.selector_reward_mode == "base":
            selector_rewards = np.broadcast_to(base_broadcast, similarity_per_step.shape).copy()
        elif self.selector_reward_mode == "sim":
            selector_rewards = base_broadcast + self.selector_lambda_s * similarity_per_step
        elif self.selector_reward_mode == "div":
            selector_rewards = base_broadcast + self.selector_lambda_d * diversity_per_step
        else:
            selector_rewards = (
                base_broadcast
                + self.selector_lambda_s * similarity_per_step
                + self.selector_lambda_d * diversity_per_step
            )

        context.recommender_actions = final_actions.astype(np.int64)
        context.base_reward = base_reward
        context.dynamic_reward = dynamic_reward
        context.previous_reward = previous_reward
        context.dynamic_uncertainty = uncertainty_dynamic
        context.static_uncertainty = uncertainty_static
        context.uncertainty_numerator = numerator
        context.uncertainty_denominator = denominator
        context.similarity_gain = similarity_set.astype(np.float32)
        context.diversity_gain = diversity_set.astype(np.float32)
        context.selector_rewards = selector_rewards.astype(np.float32)
        return context

    # ---------------------------------------------------------------------
    # selector loss
    # ---------------------------------------------------------------------

    def _compute_selector_loss(
        self,
        minibatch: Batch,
        recommender_state: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, float],
    ]:
        """计算 selector A2C 损失并提取训练诊断量。

        Args:
            minibatch (Batch): 包含 selector 采样上下文的训练 minibatch。
            recommender_state (torch.Tensor): 推荐器状态表示，形状为
                ``(batch_size, state_dim)``。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            Dict[str, float]]: 依次为总损失、actor 损失、value 损失、
            策略熵以及当前 minibatch 的 selector 诊断指标。
        """

        if not hasattr(minibatch.policy, "darlr_context"):
            zero_loss = next(self.selector_actor.parameters()).sum() * 0.0
            return zero_loss, zero_loss, zero_loss, zero_loss, {
                "context_present_rate": 0.0,
                "context_valid_rate": 0.0,
            }

        context = minibatch.policy.darlr_context
        current_users = self._as_long_tensor(context.current_users)
        candidate_users = self._as_long_tensor(context.candidate_users)
        selected_users = self._as_long_tensor(context.selected_users)
        selector_actions = self._as_long_tensor(context.selector_actions)
        raw_selector_rewards = self._as_float_tensor(context.selector_rewards)

        if (
            raw_selector_rewards.ndim != 2
            or raw_selector_rewards.shape[1] != self.selector_k
        ):
            zero_loss = next(self.selector_actor.parameters()).sum() * 0.0
            return zero_loss, zero_loss, zero_loss, zero_loss, {
                "context_present_rate": 1.0,
                "context_valid_rate": 0.0,
            }

        current_preferences = self._encode_user_preferences(current_users)
        candidate_preferences = self._encode_user_preferences(candidate_users)
        optimized_selector_rewards = raw_selector_rewards
        if self.selector_reward_normalization:
            optimized_selector_rewards = standardize_tensor(
                raw_selector_rewards,
                eps=self.selector_normalization_eps,
            )
        selector_returns = self._discounted_returns(
            optimized_selector_rewards
        )

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

        value_losses: List[torch.Tensor] = []
        entropy_losses: List[torch.Tensor] = []
        advantage_values: List[torch.Tensor] = []
        critic_values: List[torch.Tensor] = []
        chosen_log_probabilities: List[torch.Tensor] = []
        chosen_probabilities: List[torch.Tensor] = []
        maximum_probabilities: List[torch.Tensor] = []

        for step_index in range(self.selector_k):
            selected_state = self.selector_state_encoder(selected_embeddings, selected_mask)
            if self.selector_policy_mode == "random":
                logits = torch.zeros(
                    current_users.shape[0],
                    candidate_users.shape[1],
                    device=self.predicted_mat.device,
                )
            else:
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
            chosen_log_probability = distribution.log_prob(step_actions)
            value_losses.append(F.mse_loss(values, step_returns))
            entropy_losses.append(distribution.entropy().mean())
            advantage_values.append(advantages)
            critic_values.append(values)
            chosen_log_probabilities.append(chosen_log_probability)
            chosen_probabilities.append(chosen_log_probability.exp())
            maximum_probabilities.append(distribution.probs.max(dim=-1).values)

            reference_users = selected_users[:, step_index]
            selected_embeddings[:, step_index, :] = self._encode_user_preferences(reference_users)
            selected_mask[:, step_index] = True
            selected_candidate_indices.append(step_actions)

        raw_advantages = torch.stack(advantage_values, dim=1)
        optimized_advantages = raw_advantages
        if self.selector_advantage_normalization:
            optimized_advantages = standardize_tensor(
                raw_advantages,
                eps=self.selector_normalization_eps,
            )
        stacked_log_probabilities = torch.stack(
            chosen_log_probabilities,
            dim=1,
        )
        actor_loss = -(
            stacked_log_probabilities * optimized_advantages
        ).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy_loss = torch.stack(entropy_losses).mean()
        total_loss = (
            actor_loss
            + self._weight_vf * value_loss
            - self.selector_ent_coef * entropy_loss
        )
        diagnostics = self._build_selector_training_diagnostics(
            context=context,
            selector_rewards=raw_selector_rewards,
            optimized_selector_rewards=optimized_selector_rewards,
            selector_returns=selector_returns,
            advantages=raw_advantages,
            optimized_advantages=optimized_advantages,
            critic_values=torch.stack(critic_values, dim=1),
            chosen_log_probabilities=stacked_log_probabilities,
            chosen_probabilities=torch.stack(chosen_probabilities, dim=1),
            maximum_probabilities=torch.stack(maximum_probabilities, dim=1),
            selector_actions=selector_actions,
            selected_users=selected_users,
        )
        return total_loss, actor_loss, value_loss, entropy_loss, diagnostics

    def _build_selector_training_diagnostics(
        self,
        context: Batch,
        selector_rewards: torch.Tensor,
        optimized_selector_rewards: torch.Tensor,
        selector_returns: torch.Tensor,
        advantages: torch.Tensor,
        optimized_advantages: torch.Tensor,
        critic_values: torch.Tensor,
        chosen_log_probabilities: torch.Tensor,
        chosen_probabilities: torch.Tensor,
        maximum_probabilities: torch.Tensor,
        selector_actions: torch.Tensor,
        selected_users: torch.Tensor,
    ) -> Dict[str, float]:
        """汇总一个 minibatch 的 selector 奖励、策略与价值诊断量。

        Args:
            context (Batch): 收集阶段保存的 DARLR selector 上下文。
            selector_rewards (torch.Tensor): 每个选择步的内在奖励。
            optimized_selector_rewards (torch.Tensor): 可选标准化后实际用于
                计算 return 的 selector 奖励。
            selector_returns (torch.Tensor): 折扣累计回报。
            advantages (torch.Tensor): 未标准化的优势函数值。
            optimized_advantages (torch.Tensor): actor 实际使用的优势函数值。
            critic_values (torch.Tensor): selector critic 的价值估计。
            chosen_log_probabilities (torch.Tensor): 已采样动作的对数概率。
            chosen_probabilities (torch.Tensor): 已采样动作的概率。
            maximum_probabilities (torch.Tensor): 每一步动作分布的最大概率。
            selector_actions (torch.Tensor): 候选池内的动作索引。
            selected_users (torch.Tensor): 实际选择的参考用户 id。

        Returns:
            Dict[str, float]: 可由 epoch 聚合器接收的标量指标。
        """

        metrics: Dict[str, float] = {
            "context_present_rate": 1.0,
            "context_valid_rate": 1.0,
        }
        self._add_tensor_statistics(metrics, "reward", selector_rewards, True)
        self._add_tensor_statistics(
            metrics,
            "optimized_reward",
            optimized_selector_rewards,
            True,
        )
        self._add_tensor_statistics(metrics, "return", selector_returns, True)
        self._add_tensor_statistics(metrics, "advantage", advantages, True)
        self._add_tensor_statistics(
            metrics,
            "optimized_advantage",
            optimized_advantages,
            True,
        )
        self._add_tensor_statistics(metrics, "critic_value", critic_values, True)
        self._add_tensor_statistics(
            metrics,
            "chosen_log_probability",
            chosen_log_probabilities,
            True,
        )
        self._add_tensor_statistics(
            metrics,
            "chosen_probability",
            chosen_probabilities,
            True,
        )
        self._add_tensor_statistics(
            metrics,
            "maximum_probability",
            maximum_probabilities,
            True,
        )

        base_reward = self._as_float_tensor(context.base_reward)
        similarity = self._as_float_tensor(context.similarity_per_step)
        diversity = self._as_float_tensor(context.diversity_per_step)
        dynamic_reward = self._as_float_tensor(context.dynamic_reward)
        dynamic_uncertainty = self._as_float_tensor(context.dynamic_uncertainty)
        self._add_tensor_statistics(metrics, "base_reward", base_reward, True)
        self._add_tensor_statistics(metrics, "similarity", similarity, True)
        self._add_tensor_statistics(metrics, "diversity", diversity, True)
        self._add_tensor_statistics(metrics, "dynamic_reward", dynamic_reward, True)
        self._add_tensor_statistics(
            metrics,
            "dynamic_uncertainty",
            dynamic_uncertainty,
            True,
        )

        metrics["base_reward_abs_mean"] = float(base_reward.abs().mean().item())
        metrics["similarity_term_abs_mean"] = float(
            (self.selector_lambda_s * similarity).abs().mean().item()
        )
        metrics["diversity_term_abs_mean"] = float(
            (self.selector_lambda_d * diversity).abs().mean().item()
        )
        metrics["selected_candidate_index_mean"] = float(
            selector_actions.float().mean().item()
        )
        metrics["top_candidate_rate"] = float(
            (selector_actions == 0).float().mean().item()
        )
        rank_cutoff = min(TOP_CANDIDATE_RANK_CUTOFF, self.selector_candidate_size)
        metrics["top_ranked_candidate_rate"] = float(
            (selector_actions < rank_cutoff).float().mean().item()
        )
        metrics["selected_user_unique_ratio"] = float(
            torch.unique(selected_users).numel() / max(1, selected_users.numel())
        )
        return metrics

    @staticmethod
    def _add_tensor_statistics(
        metrics: Dict[str, float],
        prefix: str,
        values: torch.Tensor,
        include_extrema: bool,
    ) -> None:
        """把张量的均值、标准差及可选极值写入指标字典。

        Args:
            metrics (Dict[str, float]): 接收统计值的可变指标字典。
            prefix (str): 生成指标名时使用的语义前缀。
            values (torch.Tensor): 非空数值张量。
            include_extrema (bool): 是否额外记录最小值和最大值。

        Returns:
            None: 统计结果直接写入 ``metrics``。

        Raises:
            ValueError: 当 ``values`` 为空时抛出。
        """

        if values.numel() == 0:
            raise ValueError(f"Cannot summarize empty tensor for metric '{prefix}'.")
        detached_values = values.detach().float()
        metrics[f"{prefix}_mean"] = float(detached_values.mean().item())
        metrics[f"{prefix}_std"] = float(
            detached_values.std(unbiased=False).item()
        )
        if include_extrema:
            metrics[f"{prefix}_min"] = float(detached_values.min().item())
            metrics[f"{prefix}_max"] = float(detached_values.max().item())

    @staticmethod
    def _compute_module_gradient_norm(module: nn.Module) -> float:
        """计算模块全部有效梯度的二范数。

        Args:
            module (nn.Module): 已完成反向传播的网络模块。

        Returns:
            float: 所有参数梯度拼接后的全局二范数；没有梯度时返回 0。
        """

        squared_norm_sum = 0.0
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            parameter_norm = float(parameter.grad.detach().norm(2).item())
            squared_norm_sum += parameter_norm * parameter_norm
        return math.sqrt(squared_norm_sum)

    # ---------------------------------------------------------------------
    # 辅助工具
    # ---------------------------------------------------------------------

    def _extract_current_users(self, batch: Batch) -> torch.Tensor:
        obs = torch.as_tensor(batch.obs, device=self.predicted_mat.device)
        if obs.ndim == 1:
            return obs.long()
        return obs[:, 0].long()

    def _build_candidate_users(self, current_users: torch.Tensor) -> torch.Tensor:
        num_users = self.predicted_mat.shape[0]
        candidate_size = min(self.selector_candidate_size, max(1, num_users - 1))
        if candidate_size < self.selector_k:
            raise ValueError(
                "Not enough unique candidate users to sample selector_k, "
                f"got candidate_size={candidate_size}, selector_k={self.selector_k}, "
                f"num_users={num_users}."
            )
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
        """无放回随机采样候选用户。

        每个 batch 样本独立在 `[0, num_users)` 中排除自身后无放回采样
        `candidate_size` 个用户。
        """
        batch_size = current_users.shape[0]
        device = current_users.device
        # 生成 [B, num_users] 的随机噪声，把当前用户位置屏蔽到最小值。
        noise = torch.rand(batch_size, num_users, device=device)
        noise.scatter_(1, current_users.view(-1, 1), -1.0)
        # 取每行 top-candidate_size 的位置索引即无放回样本。
        return torch.topk(noise, k=candidate_size, dim=1).indices

    def _encode_user_preferences(self, user_ids: torch.Tensor) -> torch.Tensor:
        user_ids = user_ids.to(self.predicted_mat.device).long()
        preferences = self.predicted_mat[user_ids]
        return self.preference_encoder(preferences)

    def _mask_selected_candidates(
        self,
        logits: torch.Tensor,
        selected_candidate_indices: Sequence[torch.Tensor],
    ) -> torch.Tensor:
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
        """论文定义: r_s^{sel} = cos(p_u, p_{u_t}).

        `paper_core` 保留原始 cosine (范围 [-1,1])，`stabilized` 映射到
        `[0,1]` 便于工程稳定性对比。
        """
        current_pref = self.predicted_mat[current_users]
        reference_pref = self.predicted_mat[reference_users]
        cosine = F.cosine_similarity(current_pref, reference_pref, dim=-1)
        if self.selector_gain_mode == "stabilized":
            return (cosine + 1.0) * 0.5
        return cosine

    def _compute_diversity_gain(
        self,
        reference_users: torch.Tensor,
        selected_users: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """论文定义:
        r_d^{sel} = 1/|U_S| * sum_{u_i in U_S} [1 - cos(p_{u_i}, p_{u_t})],
        且 `U_S == 空集` 时 `r_d^{sel} = 0`。
        """
        if not selected_users:
            # P0#2: 首个 selection 明确为 0，不是原实现里的 1。
            return torch.zeros(reference_users.shape[0], device=self.predicted_mat.device)
        reference_pref = self.predicted_mat[reference_users]
        diversity_terms = []
        for previous_users in selected_users:
            previous_pref = self.predicted_mat[previous_users]
            cosine = F.cosine_similarity(reference_pref, previous_pref, dim=-1)
            if self.selector_gain_mode == "stabilized":
                similarity = (cosine + 1.0) * 0.5
            else:
                similarity = cosine
            diversity_terms.append(1.0 - similarity)
        return torch.stack(diversity_terms, dim=1).mean(dim=1)

    def _discounted_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(rewards.shape[0], device=rewards.device)
        for step_index in reversed(range(rewards.shape[1])):
            running_return = rewards[:, step_index] + self.selector_discount_factor * running_return
            returns[:, step_index] = running_return
        return returns

    def _as_long_tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, device=self.predicted_mat.device).long()

    def _as_float_tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, device=self.predicted_mat.device).float()

    def _assert_user_ids_valid(self, user_ids: torch.Tensor) -> None:
        num_users = self.predicted_mat.shape[0]
        if user_ids.numel() == 0:
            return
        min_id = int(user_ids.min().item())
        max_id = int(user_ids.max().item())
        if min_id < 0 or max_id >= num_users:
            raise IndexError(
                "user id out of range for predicted_mat: "
                f"min={min_id}, max={max_id}, num_users={num_users}."
            )

    def _assert_action_ids_valid(self, action_ids: torch.Tensor) -> None:
        num_items = self.predicted_mat.shape[1]
        if action_ids.numel() == 0:
            return
        min_id = int(action_ids.min().item())
        max_id = int(action_ids.max().item())
        if min_id < 0 or max_id >= num_items:
            raise IndexError(
                "action id out of range for predicted_mat: "
                f"min={min_id}, max={max_id}, num_items={num_items}."
            )

    @staticmethod
    def _to_numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()

    # ---------------------------------------------------------------------
    # checkpoint hooks
    # ---------------------------------------------------------------------

    def darlr_extra_state(self) -> Dict[str, Any]:
        """返回 policy 之外需要额外 checkpoint 的状态。"""
        state: Dict[str, Any] = {
            "selector_loss_coef": self.selector_loss_coef,
            "selector_gain_mode": self.selector_gain_mode,
            "selector_reward_mode": self.selector_reward_mode,
            "selector_policy_mode": self.selector_policy_mode,
            "selector_ent_coef": self.selector_ent_coef,
            "selector_reward_normalization": self.selector_reward_normalization,
            "selector_advantage_normalization": (
                self.selector_advantage_normalization
            ),
            "selector_normalization_eps": self.selector_normalization_eps,
        }
        if self.dynamic_reward_store is not None:
            state["dynamic_reward_store"] = self.dynamic_reward_store.state_dict()
        if self._optim_selector is not None:
            state["optim_selector"] = self._optim_selector.state_dict()
        return state

    def darlr_load_extra_state(self, state: Dict[str, Any]) -> None:
        """恢复由 `darlr_extra_state()` 保存的状态。"""
        if "dynamic_reward_store" in state and self.dynamic_reward_store is not None:
            self.dynamic_reward_store.load_state_dict(state["dynamic_reward_store"])
        if "optim_selector" in state and self._optim_selector is not None:
            self._optim_selector.load_state_dict(state["optim_selector"])


def _assert_no_duplicate_along_last_dim(tensor: torch.Tensor, name: str) -> None:
    """向量化断言: `tensor[i]` 中的元素互不相同 (int)。

    对候选用户矩阵这样的 `[B, C]` 张量，我们希望每一行都是 C 个不同的
    用户 id。通过 sort+相邻比较可以完全避免 Python 层的 row-loop。
    """
    if tensor.ndim < 2 or tensor.shape[-1] < 2:
        return
    sorted_values, _ = torch.sort(tensor, dim=-1)
    if bool((sorted_values[..., 1:] == sorted_values[..., :-1]).any().item()):
        raise ValueError(f"{name} contains duplicate ids in the same row.")
