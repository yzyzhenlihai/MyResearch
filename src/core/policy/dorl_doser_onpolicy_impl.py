"""On-policy 版本的 DORL-DOSER 策略实现。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Discrete
from torch import nn

from src.core.policy.dorl_doser_impl import (
    DEFAULT_LOG_INTERVAL,
    CounterfactualRewardModel,
    DORLDOSEROODHelper,
    DiffusionArtifact,
    _safe_wandb_log,
    build_mlp,
)
from src.tianshou.tianshou.data import Batch, ReplayBuffer, to_torch_as
from src.tianshou.tianshou.policy.modelfree.a2c import A2CPolicy


class A2CDOSERAugmentedCritic(nn.Module):
    """同时提供 A2C 标量值函数和 DOSER 辅助 Q/V 头的 critic。

    该 critic 保留原始 DORL A2C 所需的 `forward(state) -> V_base(s)` 接口，
    同时额外挂载四个辅助 Q 头和两个辅助 V 头，供 critic 侧 OOD penalty /
    compensation 使用。
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_dim: int,
        hidden_sizes: Sequence[int] = (),
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        """初始化增强 critic。

        Args:
            preprocess_net (nn.Module): 状态特征抽取 backbone。
            action_dim (int): 动作 embedding 维度。
            hidden_sizes (Sequence[int]): 辅助 Q/V 头使用的隐藏层配置。
            preprocess_net_output_dim (Optional[int]): backbone 输出维度。
            device (Union[str, int, torch.device]): 运行设备。
        """

        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        self.action_dim = action_dim
        feature_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        if feature_dim is None:
            raise ValueError("preprocess_net must expose output_dim for critic heads.")

        self.base_value_head = nn.Linear(feature_dim, 1)
        q_input_dim = int(feature_dim) + int(action_dim)
        self.aux_q1 = build_mlp(q_input_dim, hidden_sizes, 1)
        self.aux_q2 = build_mlp(q_input_dim, hidden_sizes, 1)
        self.aux_q3 = build_mlp(q_input_dim, hidden_sizes, 1)
        self.aux_q4 = build_mlp(q_input_dim, hidden_sizes, 1)
        self.aux_v1 = build_mlp(int(feature_dim), hidden_sizes, 1)
        self.aux_v2 = build_mlp(int(feature_dim), hidden_sizes, 1)

    def encode_state(self, obs: Union[np.ndarray, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        """把状态编码到 backbone 特征空间。"""

        feature, _ = self.preprocess(obs, state=kwargs.get("state", None))
        return feature

    def forward(self, obs: Union[np.ndarray, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        """返回 A2C 使用的基础值函数 `V_base(s)`。"""

        feature = self.encode_state(obs, **kwargs)
        return self.base_value_head(feature)

    def aux_q(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算 DOSER 辅助 Q 头。"""

        feature = self.encode_state(state, **kwargs)
        critic_input = torch.cat([feature, action], dim=-1)
        return (
            self.aux_q1(critic_input),
            self.aux_q2(critic_input),
            self.aux_q3(critic_input),
            self.aux_q4(critic_input),
        )

    def aux_q_from_feature(
        self,
        feature: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """基于已编码特征计算 DOSER 辅助 Q 头。

        Args:
            feature (torch.Tensor): 由 critic preprocess 网络得到的状态特征，
                形状为 `(batch_size, feature_dim)`。
            action (torch.Tensor): 动作 embedding，形状为
                `(batch_size, action_dim)`。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            四个辅助 Q 头的输出，每个张量形状为 `(batch_size, 1)`。

        Raises:
            ValueError: 当状态特征与动作 embedding 的 batch size 不一致时抛出。
        """

        if feature.shape[0] != action.shape[0]:
            raise ValueError(
                "feature and action must have the same batch size: "
                f"{feature.shape[0]} != {action.shape[0]}"
            )
        critic_input = torch.cat([feature, action], dim=-1)
        return (
            self.aux_q1(critic_input),
            self.aux_q2(critic_input),
            self.aux_q3(critic_input),
            self.aux_q4(critic_input),
        )

    def q_min(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """计算辅助 Q 头的最小值。"""

        q1, q2, q3, q4 = self.aux_q(state, action, **kwargs)
        return torch.min(torch.min(q1, q2), torch.min(q3, q4))

    def aux_v(
        self,
        state: Union[np.ndarray, torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 DOSER 辅助双 V 头。"""

        feature = self.encode_state(state, **kwargs)
        return self.aux_v1(feature), self.aux_v2(feature)

    def aux_v_from_feature(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于已编码特征计算 DOSER 辅助双 V 头。

        Args:
            feature (torch.Tensor): 由 critic preprocess 网络得到的状态特征，
                形状为 `(batch_size, feature_dim)`。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 两个辅助 V 头的输出，每个张量
            形状为 `(batch_size, 1)`。

        Raises:
            RuntimeError: 当前实现不会主动抛出该异常。
        """

        return self.aux_v1(feature), self.aux_v2(feature)

    def aux_v_min(
        self,
        state: Union[np.ndarray, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """计算辅助双 V 头中的最小值。"""

        v1, v2 = self.aux_v(state, **kwargs)
        return torch.min(v1, v2)


class OnPolicyDORLDOSERPolicy(A2CPolicy):
    """保留 A2C 更新范式、仅在 critic 侧加入 DOSER OOD 正则的 on-policy 策略。"""

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: A2CDOSERAugmentedCritic,
        optim: Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
        dist_fn: Any,
        state_tracker: nn.Module,
        diffusion_artifact: DiffusionArtifact,
        reward_model: CounterfactualRewardModel,
        doser_beta: float = 0.001,
        doser_lam: float = 0.001,
        doser_eta: float = 0.9,
        doser_expectile: float = 0.9,
        doser_q_min: float = 0.0,
        doser_aux_critic_coef: float = 1.0,
        doser_detach_aux_state: bool = False,
        doser_action_samples: int = 10,
        diffusion_sample_steps: int = 20,
        log_interval: int = DEFAULT_LOG_INTERVAL,
        **kwargs: Any,
    ) -> None:
        """初始化 on-policy DORL-DOSER。

        Args:
            actor (torch.nn.Module): 原始 DORL A2C actor。
            critic (A2CDOSERAugmentedCritic): 增强 critic。
            optim (Tuple[torch.optim.Optimizer, torch.optim.Optimizer]): RL 与
                state tracker 的优化器。
            dist_fn (Any): A2C 使用的离散动作分布类型。
            state_tracker (nn.Module): 推荐系统状态编码器。
            diffusion_artifact (DiffusionArtifact): DOSER 扩散预训练产物。
            reward_model (CounterfactualRewardModel): counterfactual reward 重建器。
            doser_beta (float): negative OOD penalty 系数。
            doser_lam (float): positive OOD compensation 系数。
            doser_eta (float): compensation target 系数。
            doser_expectile (float): expectile value loss 系数。
            doser_q_min (float): Q 最小先验值。
            doser_aux_critic_coef (float): 辅助 Q/V 主损失在总 loss 中的权重。
            doser_detach_aux_state (bool): 是否阻断辅助 Q/V 与 OOD loss 对
                共享 state feature backbone 的梯度。
            doser_action_samples (int): 行为扩散每个状态采样的动作数。
            diffusion_sample_steps (int): 行为扩散采样步数。
            log_interval (int): 日志汇报间隔。
            **kwargs (Any): 传递给 `A2CPolicy` 的其余参数。
        """

        policy_kwargs = dict(kwargs)
        policy_kwargs.setdefault("action_space", Discrete(state_tracker.num_item))
        policy_kwargs.setdefault("action_scaling", False)
        policy_kwargs.setdefault("action_bound_method", "")
        super().__init__(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            state_tracker=state_tracker,
            **policy_kwargs,
        )
        self.critic = critic
        self.state_tracker = state_tracker
        self.reward_model = reward_model

        self.diffusion_model = diffusion_artifact.diffusion_model
        self.behavior_model = diffusion_artifact.behavior_model
        self.state_distribution = diffusion_artifact.state_distribution
        self.state_threshold = diffusion_artifact.state_threshold
        self.action_threshold = diffusion_artifact.action_threshold
        self.state_dim = diffusion_artifact.state_dim
        self.action_dim = diffusion_artifact.action_dim

        self.device = next(actor.parameters()).device
        self.beta = doser_beta
        self.lam = doser_lam
        self.eta = doser_eta
        self.expectile = doser_expectile
        self.q_min = doser_q_min
        self.aux_critic_coef = doser_aux_critic_coef
        self.detach_aux_state = doser_detach_aux_state
        self.action_samples = doser_action_samples
        self.diffusion_sample_steps = diffusion_sample_steps
        self.log_interval = log_interval
        self.num_items = state_tracker.num_item
        self.total_it = 0
        self.ood_helper = DORLDOSEROODHelper(self)
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        """校验训练状态空间与扩散产物元信息是否一致。"""

        if int(self.state_tracker.final_dim) != self.state_dim:
            raise ValueError(
                "State tracker final_dim does not match diffusion state_dim: "
                f"{self.state_tracker.final_dim} != {self.state_dim}"
            )
        if int(self.state_tracker.emb_dim) != self.action_dim:
            raise ValueError(
                "State tracker emb_dim does not match diffusion action_dim: "
                f"{self.state_tracker.emb_dim} != {self.action_dim}"
            )
        if self.aux_critic_coef < 0.0:
            raise ValueError(
                "doser_aux_critic_coef must be non-negative: "
                f"{self.aux_critic_coef}"
            )

    def _encode_auxiliary_feature(self, state_emb: torch.Tensor) -> torch.Tensor:
        """为辅助 Q/V 损失准备 critic feature。

        Args:
            state_emb (torch.Tensor): state tracker 输出的推荐状态表示，形状为
                `(batch_size, state_dim)`。

        Returns:
            torch.Tensor: critic preprocess 网络输出的状态特征。若
            `doser_detach_aux_state=True`，该特征会从计算图中分离，辅助 Q/V 与
            OOD loss 不再更新共享 actor/critic backbone。

        Raises:
            RuntimeError: 当 critic preprocess 前向失败时由底层模块抛出。
        """

        if self.detach_aux_state:
            # detach 模式用于隔离辅助 OOD 目标，避免其扰动 A2C 主干表示。
            with torch.no_grad():
                return self.critic.encode_state(state_emb).detach()
        return self.critic.encode_state(state_emb)

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """计算 expectile loss。"""

        weight = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        return weight * (diff ** 2)

    def _compute_auxiliary_critic_loss(
        self,
        state_feature: torch.Tensor,
        action_embeddings: torch.Tensor,
        target_returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 DOSER 辅助 Q/V 主损失。

        Args:
            state_feature (torch.Tensor): 已编码的 critic feature，形状为
                `(batch_size, feature_dim)`。
            action_embeddings (torch.Tensor): 动作 embedding，形状为
                `(batch_size, action_dim)`。
            target_returns (torch.Tensor): A2C process_fn 计算得到的 return 目标。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 第一个张量为辅助 Q/V 总损失，
            第二个张量为四个辅助 Q 头拼接后的当前动作 Q 值。

        Raises:
            ValueError: 当 feature 与 action embedding 的 batch size 不一致时抛出。
        """

        q1, q2, q3, q4 = self.critic.aux_q_from_feature(
            state_feature,
            action_embeddings,
        )
        target_q = target_returns.view(-1, 1)
        v1, v2 = self.critic.aux_v_from_feature(state_feature)
        value_loss = self._expectile_loss(target_q - v1).mean() + self._expectile_loss(
            target_q - v2
        ).mean()
        q_loss = (
            F.mse_loss(q1, target_q)
            + F.mse_loss(q2, target_q)
            + F.mse_loss(q3, target_q)
            + F.mse_loss(q4, target_q)
        )
        return q_loss + value_loss, torch.cat([q1, q2, q3, q4], dim=-1)

    def _build_ood_metrics(
        self,
        current_q: torch.Tensor,
        policy_q: torch.Tensor,
        value_s_pi: torch.Tensor,
        value_s_in: torch.Tensor,
        q_comp_target: torch.Tensor,
        action_error: torch.Tensor,
        pred_state_error: torch.Tensor,
        negative_ood_mask: torch.Tensor,
        positive_ood_mask: torch.Tensor,
        ood_action_mask: torch.Tensor,
        reg_loss: torch.Tensor,
        reg_loss_raw: torch.Tensor,
        vc_loss: torch.Tensor,
        vc_loss_raw: torch.Tensor,
        pred_rewards: np.ndarray,
        best_id_rewards: np.ndarray,
    ) -> Dict[str, float]:
        """整理 on-policy critic 辅助 OOD 指标。"""

        total_count = float(len(action_error))
        ood_count = float(ood_action_mask.float().sum().item())
        positive_count = float(positive_ood_mask.float().sum().item())
        negative_count = float(negative_ood_mask.float().sum().item())
        id_count = total_count - ood_count
        safe_total = max(total_count, 1.0)

        return {
            "loss/reg": float(reg_loss.item()),
            "loss/reg_raw": float(reg_loss_raw.item()),
            "loss/vc": float(vc_loss.item()),
            "loss/vc_raw": float(vc_loss_raw.item()),
            "ood/action_error_mean": float(action_error.mean().item()),
            "ood/state_error_mean": float(pred_state_error.mean().item()),
            "ood/total_count": total_count,
            "ood/positive_count": positive_count,
            "ood/negative_count": negative_count,
            "ood/id_ratio": id_count / safe_total,
            "ood/ood_ratio": ood_count / safe_total,
            "ood/positive_ratio": positive_count / safe_total,
            "ood/negative_ratio": negative_count / safe_total,
            "q/current_mean": float(current_q.mean().item()),
            "q/policy_mean": float(policy_q.mean().item()),
            "q/comp_target_mean": float(q_comp_target.mean().item()),
            "value/pi_mean": float(value_s_pi.mean().item()),
            "value/id_mean": float(value_s_in.mean().item()),
            "threshold/action": float(self.action_threshold),
            "threshold/state": float(self.state_threshold),
            "reward/policy_counterfactual_mean": float(np.mean(pred_rewards)),
            "reward/id_counterfactual_mean": float(np.mean(best_id_rewards)),
        }

    def _compute_ood_regularization(
        self,
        minibatch: Batch,
        state_emb: torch.Tensor,
        action_ids: torch.Tensor,
        action_embeddings: torch.Tensor,
        current_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """计算 DOSER 的细粒度 OOD penalty / compensation。"""

        with torch.no_grad():
            action_error = self.ood_helper.compute_action_error(action_embeddings, state_emb)
            normalized_obs = self.ood_helper.normalize_obs_array(minibatch.obs)
            user_ids = normalized_obs[:, 0].astype(np.int64)
            pred_next_state, pred_rewards = self.ood_helper.build_counterfactual_next_state(
                indices=minibatch.indices,
                user_ids=user_ids,
                action_ids=action_ids,
            )
            best_id_action_ids, _, best_id_q = self.ood_helper.select_best_id_action(
                state_emb=state_emb,
                action_mask=getattr(minibatch, "mask", None),
                critic=self.critic,
            )
            best_id_next_state, best_id_rewards = self.ood_helper.build_counterfactual_next_state(
                indices=minibatch.indices,
                user_ids=user_ids,
                action_ids=best_id_action_ids,
            )
            pred_state_error = self.ood_helper.compute_state_error(pred_next_state)
            value_s_pi = self.critic.aux_v_min(pred_next_state)
            value_s_in = self.critic.aux_v_min(best_id_next_state)

        ood_action_mask = action_error > self.action_threshold
        ood_next_state_mask = pred_state_error > self.state_threshold
        negative_value_mask = (value_s_pi < value_s_in).squeeze(-1)
        positive_value_mask = ~negative_value_mask
        negative_ood_mask = (
            ood_action_mask & (ood_next_state_mask | negative_value_mask)
        ).float().unsqueeze(-1)
        positive_ood_mask = (
            ood_action_mask & (~ood_next_state_mask) & positive_value_mask
        ).float().unsqueeze(-1)

        q_min_tensor = torch.full_like(current_q, fill_value=self.q_min)
        reg_loss_raw = (((current_q - q_min_tensor) ** 2) * negative_ood_mask).mean()
        reg_loss = self.beta * reg_loss_raw
        value_diff = (value_s_pi - value_s_in).clamp(min=0.0)
        q_comp_target = self.eta * (best_id_q + value_diff).detach()
        vc_loss_raw = (((current_q - q_comp_target) ** 2) * positive_ood_mask).mean()
        vc_loss = self.lam * vc_loss_raw
        metrics = self._build_ood_metrics(
            current_q=current_q,
            policy_q=current_q,
            value_s_pi=value_s_pi,
            value_s_in=value_s_in,
            q_comp_target=q_comp_target,
            action_error=action_error,
            pred_state_error=pred_state_error,
            negative_ood_mask=negative_ood_mask,
            positive_ood_mask=positive_ood_mask,
            ood_action_mask=ood_action_mask,
            reg_loss=reg_loss,
            reg_loss_raw=reg_loss_raw,
            vc_loss=vc_loss,
            vc_loss_raw=vc_loss_raw,
            pred_rewards=pred_rewards,
            best_id_rewards=best_id_rewards,
        )
        return reg_loss, vc_loss, metrics

    def learn(  # type: ignore[override]
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        """执行保留 A2C actor、增强 critic 的 on-policy 更新。"""

        losses: Dict[str, List[float]] = defaultdict(list)
        env_step = kwargs.get("env_step")
        optim_rl, optim_state = self.optim

        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                self.total_it += 1

                state_emb = self.state_tracker(self._buffer, minibatch.indices, is_obs=True)
                base_value = self.critic(state_emb).flatten()
                result = self(
                    minibatch,
                    self._buffer,
                    indices=minibatch.indices,
                    is_obs=True,
                )
                dist = result.dist
                advantages = to_torch_as(minibatch.adv, base_value)
                returns = to_torch_as(minibatch.returns, base_value)
                action = torch.as_tensor(
                    minibatch.act,
                    dtype=torch.long,
                    device=result.act.device,
                ).view(-1)

                log_prob = dist.log_prob(action)
                log_prob = log_prob.reshape(len(advantages), -1).transpose(0, 1)
                actor_loss = -(log_prob * advantages).mean()
                base_value_loss = F.mse_loss(returns, base_value)

                action_ids = action.view(-1)
                action_embeddings = self.ood_helper.lookup_action_embeddings(action_ids)
                aux_state_feature = self._encode_auxiliary_feature(state_emb)
                aux_critic_loss, current_q = self._compute_auxiliary_critic_loss(
                    state_feature=aux_state_feature,
                    action_embeddings=action_embeddings,
                    target_returns=returns,
                )
                reg_loss, vc_loss, ood_metrics = self._compute_ood_regularization(
                    minibatch=minibatch,
                    state_emb=state_emb,
                    action_ids=action_ids,
                    action_embeddings=action_embeddings,
                    current_q=current_q,
                )
                ent_loss = dist.entropy().mean()
                weighted_aux_critic_loss = self.aux_critic_coef * aux_critic_loss
                total_loss = (
                    actor_loss
                    + self._weight_vf * base_value_loss
                    + weighted_aux_critic_loss
                    + reg_loss
                    + vc_loss
                    - self._weight_ent * ent_loss
                )

                optim_rl.zero_grad()
                optim_state.zero_grad()
                total_loss.backward()
                if self._grad_norm:
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(),
                        max_norm=self._grad_norm,
                    )
                optim_rl.step()
                optim_state.step()

                step_metrics = {
                    "loss": float(total_loss.item()),
                    "loss/actor": float(actor_loss.item()),
                    "loss/vf": float(base_value_loss.item()),
                    "loss/critic_aux": float(aux_critic_loss.item()),
                    "loss/critic_aux_weighted": float(weighted_aux_critic_loss.item()),
                    "loss/ent": float(ent_loss.item()),
                    **ood_metrics,
                }
                for metric_name, metric_value in step_metrics.items():
                    losses[metric_name].append(metric_value)

                if self.total_it % self.log_interval == 0 and env_step is not None:
                    _safe_wandb_log(step_metrics, step=int(env_step))

        return dict(losses)
