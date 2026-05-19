"""On-policy DORL-DOSER 策略实现。

该模块在 Tianshou A2CPolicy 的基础上加入 DOSER OOD critic 正则。
actor、GAE、A2C value loss 和 entropy loss 保持原有语义；新增的
OOD penalty / compensation 仅通过 augmented critic 的辅助 Q/V 分支
参与优化。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Discrete
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy.modelfree.a2c import A2CPolicy
from tianshou.utils.net.common import ActorCritic

from src.core.policy.dorl_doser_impl import (
    CounterfactualRewardModel,
    DiffusionArtifact,
    DORLDOSEROODHelper,
    build_mlp,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_AUX_CRITIC_COEF = 1.0
"""辅助 Q/V 主损失默认权重。"""

DEFAULT_DOSER_BETA = 0.001
"""negative OOD penalty 默认权重。"""

DEFAULT_DOSER_LAM = 0.001
"""positive OOD compensation 默认权重。"""

DEFAULT_DOSER_ETA = 0.9
"""positive OOD compensation 目标值默认缩放系数。"""

DEFAULT_DOSER_EXPECTILE = 0.9
"""辅助 V 分支默认 expectile。"""

DEFAULT_Q_MIN = 0.0
"""negative OOD penalty 的 Q 下界先验。"""

DEFAULT_ACTION_SAMPLES = 10
"""每个状态默认行为扩散候选动作数量。"""

DEFAULT_DIFFUSION_SAMPLE_STEPS = 20
"""行为扩散采样默认步数。"""

DEFAULT_LOG_INTERVAL = 100
"""训练阶段默认指标记录间隔。"""

MIN_ACTION_SAMPLES = 1
"""行为扩散候选动作数量下界。"""

MIN_DIFFUSION_SAMPLE_STEPS = 1
"""扩散采样步数下界。"""


class A2CDOSERAugmentedCritic(nn.Module):
    """带 DOSER 辅助 Q/V 分支的 A2C critic。

    `forward()` 保持 A2C 所需的 `V(s)` 输出；`aux_q()`、`aux_v()` 和
    `q_min()` 为 OOD penalty / compensation 提供 critic 端辅助估计。
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] | List[int],
        device: Any = "cpu",
    ) -> None:
        """初始化 augmented critic。

        Args:
            preprocess_net (nn.Module): 与 actor 共享或同构的状态预处理网络。
            action_dim (int): 动作 embedding 维度。
            hidden_sizes (Tuple[int, ...] | List[int]): 辅助头隐层维度。
            device (Any): 目标设备。

        Raises:
            ValueError: 当 action_dim 非法或 preprocess 输出维度缺失时抛出。
        """

        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")

        self.preprocess = preprocess_net
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        feature_dim = getattr(preprocess_net, "output_dim", None)
        if feature_dim is None:
            raise ValueError("preprocess_net must expose `output_dim`.")
        self.feature_dim = int(feature_dim)

        self.value_head = build_mlp(self.feature_dim, 1, hidden_sizes)
        self.aux_q_head_1 = build_mlp(self.feature_dim + self.action_dim, 1, hidden_sizes)
        self.aux_q_head_2 = build_mlp(self.feature_dim + self.action_dim, 1, hidden_sizes)
        self.aux_v_head_1 = build_mlp(self.feature_dim, 1, hidden_sizes)
        self.aux_v_head_2 = build_mlp(self.feature_dim, 1, hidden_sizes)
        self.to(self.device)

    def encode_state(self, obs: torch.Tensor) -> torch.Tensor:
        """编码推荐状态。

        Args:
            obs (torch.Tensor): state tracker 输出的状态张量。

        Returns:
            torch.Tensor: critic feature 张量。
        """

        feature, _ = self.preprocess(obs, state=None)
        return feature

    def forward(self, obs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """计算 A2C 主 value head。

        Args:
            obs (torch.Tensor): state tracker 输出的状态张量。
            **kwargs (Any): 兼容 Tianshou critic 接口的额外参数。

        Returns:
            torch.Tensor: `V(s)`，形状为 `(batch_size, 1)`。
        """

        feature = self.encode_state(obs)
        return self.value_head(feature)

    def aux_q_from_feature(
        self,
        state_feature: torch.Tensor,
        action_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于已编码状态计算双 Q。

        Args:
            state_feature (torch.Tensor): 状态 feature。
            action_embeddings (torch.Tensor): 动作 embedding。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 两个 Q head 的输出。
        """

        q_input = torch.cat([state_feature, action_embeddings], dim=-1)
        return self.aux_q_head_1(q_input), self.aux_q_head_2(q_input)

    def aux_q(
        self,
        obs: torch.Tensor,
        action_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算辅助双 Q。

        Args:
            obs (torch.Tensor): 状态张量。
            action_embeddings (torch.Tensor): 动作 embedding。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 两个辅助 Q 输出。
        """

        return self.aux_q_from_feature(self.encode_state(obs), action_embeddings)

    def q_min(self, obs: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        """计算 clipped double Q 的最小值。

        Args:
            obs (torch.Tensor): 状态张量。
            action_embeddings (torch.Tensor): 动作 embedding。

        Returns:
            torch.Tensor: `min(Q1, Q2)`，形状为 `(batch_size,)`。
        """

        q_1, q_2 = self.aux_q(obs, action_embeddings)
        return torch.minimum(q_1, q_2).flatten()

    def aux_v_from_feature(self, state_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于已编码状态计算辅助双 V。

        Args:
            state_feature (torch.Tensor): 状态 feature。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 两个辅助 V 输出。
        """

        return self.aux_v_head_1(state_feature), self.aux_v_head_2(state_feature)

    def aux_v(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算辅助双 V。

        Args:
            obs (torch.Tensor): 状态张量。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 两个辅助 V 输出。
        """

        return self.aux_v_from_feature(self.encode_state(obs))

    def aux_v_min(self, obs: torch.Tensor) -> torch.Tensor:
        """计算辅助双 V 的最小值。

        Args:
            obs (torch.Tensor): 状态张量。

        Returns:
            torch.Tensor: `min(V1, V2)`，形状为 `(batch_size,)`。
        """

        v_1, v_2 = self.aux_v(obs)
        return torch.minimum(v_1, v_2).flatten()


class OnPolicyDORLDOSERPolicy(A2CPolicy):
    """A2CPolicy + DOSER critic OOD 正则。

    该策略继承 A2CPolicy 的 forward、process_fn 和 GAE 逻辑，只重写
    `learn()`，在原始 A2C loss 上增加 critic auxiliary loss。
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: A2CDOSERAugmentedCritic,
        optim: torch.optim.Optimizer | Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
        dist_fn: Type[torch.distributions.Distribution],
        state_tracker: nn.Module,
        diffusion_artifact: DiffusionArtifact,
        reward_model: CounterfactualRewardModel,
        discount_factor: float = 0.99,
        gae_lambda: float = 1.0,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: Optional[float] = None,
        reward_normalization: bool = False,
        doser_beta: float = DEFAULT_DOSER_BETA,
        doser_lam: float = DEFAULT_DOSER_LAM,
        doser_eta: float = DEFAULT_DOSER_ETA,
        doser_expectile: float = DEFAULT_DOSER_EXPECTILE,
        doser_q_min: float = DEFAULT_Q_MIN,
        doser_aux_critic_coef: float = DEFAULT_AUX_CRITIC_COEF,
        doser_detach_aux_state: bool = True,
        doser_action_samples: int = DEFAULT_ACTION_SAMPLES,
        diffusion_sample_steps: int = DEFAULT_DIFFUSION_SAMPLE_STEPS,
        log_interval: int = DEFAULT_LOG_INTERVAL,
        **kwargs: Any,
    ) -> None:
        """初始化 on-policy DORL-DOSER 策略。

        Args:
            actor (nn.Module): A2C 离散 actor。
            critic (A2CDOSERAugmentedCritic): augmented critic。
            optim (torch.optim.Optimizer | Tuple[torch.optim.Optimizer, torch.optim.Optimizer]):
                RL 和 state tracker 优化器，沿用项目 A2C 约定。
            dist_fn (Type[torch.distributions.Distribution]): 动作分布类型。
            state_tracker (nn.Module): 推荐系统状态编码器。
            diffusion_artifact (DiffusionArtifact): 预训练扩散产物。
            reward_model (CounterfactualRewardModel): counterfactual reward 近似器。
            discount_factor (float): 折扣因子。
            gae_lambda (float): GAE lambda。
            vf_coef (float): A2C value loss 权重。
            ent_coef (float): entropy loss 权重。
            max_grad_norm (Optional[float]): 梯度裁剪阈值。
            reward_normalization (bool): 是否启用 reward normalization。
            doser_beta (float): negative OOD penalty 权重。
            doser_lam (float): positive OOD compensation 权重。
            doser_eta (float): compensation target 缩放系数。
            doser_expectile (float): 辅助 V expectile。
            doser_q_min (float): Q 下界先验。
            doser_aux_critic_coef (float): 辅助 Q/V 主损失权重。
            doser_detach_aux_state (bool): 是否阻断 OOD 辅助损失到 state tracker。
            doser_action_samples (int): 行为扩散候选动作数。
            diffusion_sample_steps (int): 扩散采样步数。
            log_interval (int): 日志间隔。
            **kwargs (Any): 传给 A2CPolicy 的额外参数。

        Raises:
            ValueError: 当 DOSER 超参数非法时抛出。
        """

        self._validate_hyperparameters(
            doser_expectile=doser_expectile,
            doser_action_samples=doser_action_samples,
            diffusion_sample_steps=diffusion_sample_steps,
            log_interval=log_interval,
        )
        action_dim = int(getattr(actor, "output_dim", 0))
        kwargs.setdefault("action_space", Discrete(action_dim))
        kwargs.setdefault("action_scaling", False)
        super().__init__(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            state_tracker=state_tracker,
            discount_factor=discount_factor,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            reward_normalization=reward_normalization,
            **kwargs,
        )
        self.diffusion_artifact = diffusion_artifact
        self.reward_model = reward_model
        self.doser_beta = float(doser_beta)
        self.doser_lam = float(doser_lam)
        self.doser_eta = float(doser_eta)
        self.doser_expectile = float(doser_expectile)
        self.doser_q_min = float(doser_q_min)
        self.doser_aux_critic_coef = float(doser_aux_critic_coef)
        self.doser_detach_aux_state = bool(doser_detach_aux_state)
        self.doser_action_samples = int(doser_action_samples)
        self.diffusion_sample_steps = int(diffusion_sample_steps)
        self.log_interval = int(log_interval)
        self.learn_step = 0
        self.ood_helper = DORLDOSEROODHelper(
            state_tracker=state_tracker,
            diffusion_artifact=diffusion_artifact,
            reward_model=reward_model,
        )
        self._validate_dimensions()

    @staticmethod
    def _validate_hyperparameters(
        doser_expectile: float,
        doser_action_samples: int,
        diffusion_sample_steps: int,
        log_interval: int,
    ) -> None:
        """校验 DOSER 超参数。

        Args:
            doser_expectile (float): expectile 系数。
            doser_action_samples (int): 候选动作数。
            diffusion_sample_steps (int): 扩散采样步数。
            log_interval (int): 日志间隔。

        Raises:
            ValueError: 当任一参数非法时抛出。
        """

        if not 0.0 < doser_expectile < 1.0:
            raise ValueError("doser_expectile must be in (0, 1).")
        if doser_action_samples < MIN_ACTION_SAMPLES:
            raise ValueError("doser_action_samples must be positive.")
        if diffusion_sample_steps < MIN_DIFFUSION_SAMPLE_STEPS:
            raise ValueError("diffusion_sample_steps must be positive.")
        if log_interval <= 0:
            raise ValueError("log_interval must be positive.")

    def _validate_dimensions(self) -> None:
        """校验 critic/action embedding 与扩散产物维度是否兼容。

        Raises:
            ValueError: 当动作维度不一致时抛出。
        """

        if self.critic.action_dim != self.diffusion_artifact.action_dim:
            raise ValueError(
                "Action embedding dim mismatch between critic and diffusion "
                f"artifact: critic={self.critic.action_dim}, "
                f"artifact={self.diffusion_artifact.action_dim}."
            )

    def _get_aux_state(self, obs_emb: torch.Tensor) -> torch.Tensor:
        """根据配置决定辅助损失是否 detach 状态。

        Args:
            obs_emb (torch.Tensor): state tracker 输出状态。

        Returns:
            torch.Tensor: 用于辅助 critic 的状态张量。
        """

        return obs_emb.detach() if self.doser_detach_aux_state else obs_emb

    def _expectile_loss(
        self,
        diff: torch.Tensor,
        expectile: float,
    ) -> torch.Tensor:
        """计算 expectile loss。

        Args:
            diff (torch.Tensor): 目标值减预测值。
            expectile (float): expectile 系数。

        Returns:
            torch.Tensor: 标量 loss。
        """

        weights = torch.where(diff > 0, expectile, 1.0 - expectile)
        return (weights * diff.pow(2)).mean()

    def _compute_auxiliary_critic_loss(
        self,
        minibatch: Batch,
        obs_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算当前 batch 动作上的辅助 Q/V 主损失。

        Args:
            minibatch (Batch): A2C minibatch。
            obs_emb (torch.Tensor): 当前状态 embedding。

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: 辅助损失和日志指标。
        """

        aux_state = self._get_aux_state(obs_emb)
        action_ids = to_torch_as(minibatch.act, obs_emb).long().view(-1)
        action_embeddings = self.ood_helper.lookup_action_embeddings(action_ids).detach()
        returns = minibatch.returns.detach().to(obs_emb.device, dtype=torch.float32).view(-1)

        q_1, q_2 = self.critic.aux_q(aux_state, action_embeddings)
        v_1, v_2 = self.critic.aux_v(aux_state)
        q_1 = q_1.flatten()
        q_2 = q_2.flatten()
        v_1 = v_1.flatten()
        v_2 = v_2.flatten()

        q_loss = F.mse_loss(q_1, returns) + F.mse_loss(q_2, returns)
        v_loss = self._expectile_loss(
            returns - v_1,
            self.doser_expectile,
        ) + self._expectile_loss(
            returns - v_2,
            self.doser_expectile,
        )
        aux_loss = q_loss + v_loss
        metrics = {
            "loss/doser_aux_q": float(q_loss.detach().cpu().item()),
            "loss/doser_aux_v": float(v_loss.detach().cpu().item()),
            "q/current": float(torch.minimum(q_1, q_2).detach().mean().cpu().item()),
            "v/aux": float(torch.minimum(v_1, v_2).detach().mean().cpu().item()),
        }
        return aux_loss, metrics

    def _policy_action_ids(self, dist: torch.distributions.Distribution) -> torch.Tensor:
        """从当前 actor 分布提取 greedy action ID。

        Args:
            dist (torch.distributions.Distribution): 当前动作分布。

        Returns:
            torch.Tensor: greedy action ID，形状为 `(batch_size,)`。
        """

        probs = getattr(dist, "probs", None)
        if probs is None:
            raise ValueError("OnPolicyDORLDOSERPolicy requires a categorical dist with probs.")
        return torch.argmax(probs.detach(), dim=-1).long().view(-1)

    def _compute_ood_regularization(
        self,
        minibatch: Batch,
        obs_emb: torch.Tensor,
        dist: torch.distributions.Distribution,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算 DOSER OOD penalty 和 compensation。

        Args:
            minibatch (Batch): A2C minibatch。
            obs_emb (torch.Tensor): 当前状态 embedding。
            dist (torch.distributions.Distribution): 当前 actor 动作分布。

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: OOD 正则损失和日志指标。
        """

        aux_state = self._get_aux_state(obs_emb)
        with torch.no_grad():
            policy_action_ids = self._policy_action_ids(dist)
            policy_action_embeddings = self.ood_helper.lookup_action_embeddings(
                policy_action_ids
            )
            cf_next_state = self.ood_helper.build_counterfactual_next_state(
                batch=minibatch,
                buffer=self._buffer,
                indices=minibatch.indices,
                action_ids=policy_action_ids,
                fallback_states=obs_emb,
            )
            action_error = self.ood_helper.compute_action_error(
                policy_action_embeddings,
                obs_emb,
            )
            state_error = self.ood_helper.compute_state_error(
                obs_emb,
                policy_action_embeddings,
                cf_next_state,
            )
            action_ood_mask = action_error > self.diffusion_artifact.action_threshold
            state_ood_mask = state_error > self.diffusion_artifact.state_threshold
            ood_mask = torch.logical_or(action_ood_mask, state_ood_mask)

            best_ids, best_actions, best_q = self.ood_helper.select_best_id_action(
                states=obs_emb,
                critic=self.critic,
                action_samples=self.doser_action_samples,
                diffusion_sample_steps=self.diffusion_sample_steps,
            )
            best_next_state = self.ood_helper.build_counterfactual_next_state(
                batch=minibatch,
                buffer=self._buffer,
                indices=minibatch.indices,
                action_ids=best_ids,
                fallback_states=obs_emb,
            )
            value_s_pi = self.critic.aux_v_min(cf_next_state)
            value_s_in = self.critic.aux_v_min(best_next_state)
            positive_mask = torch.logical_and(
                torch.logical_and(action_ood_mask, torch.logical_not(state_ood_mask)),
                value_s_pi >= value_s_in,
            )
            negative_mask = torch.logical_and(ood_mask, torch.logical_not(positive_mask))
            q_comp_target = self.doser_eta * (best_q + value_s_pi - value_s_in)

        q_pi = self.critic.q_min(aux_state, policy_action_embeddings.detach())
        negative_weight = negative_mask.to(q_pi.dtype)
        positive_weight = positive_mask.to(q_pi.dtype)
        penalty_loss = (
            negative_weight * F.relu(q_pi - self.doser_q_min).pow(2)
        ).mean()
        compensation_loss = (
            positive_weight * (q_pi - q_comp_target.detach()).pow(2)
        ).mean()
        ood_loss = self.doser_beta * penalty_loss + self.doser_lam * compensation_loss

        total_count = float(len(q_pi))
        metrics = {
            "loss/doser_penalty": float(penalty_loss.detach().cpu().item()),
            "loss/doser_compensation": float(compensation_loss.detach().cpu().item()),
            "ood/action_error": float(action_error.detach().mean().cpu().item()),
            "ood/state_error": float(state_error.detach().mean().cpu().item()),
            "ood/ood_ratio": float(ood_mask.float().mean().detach().cpu().item()),
            "ood/positive_ratio": float(positive_mask.float().mean().detach().cpu().item()),
            "ood/negative_ratio": float(negative_mask.float().mean().detach().cpu().item()),
            "ood/total_count": total_count,
            "q/pi": float(q_pi.detach().mean().cpu().item()),
            "q/best_id": float(best_q.detach().mean().cpu().item()),
            "v/pi_next": float(value_s_pi.detach().mean().cpu().item()),
            "v/best_next": float(value_s_in.detach().mean().cpu().item()),
        }
        return ood_loss, metrics

    def _log_training_metrics(self, metrics: Dict[str, float]) -> None:
        """按固定间隔写入 debug 日志。

        Args:
            metrics (Dict[str, float]): 当前 minibatch 指标。
        """

        if self.learn_step % self.log_interval != 0:
            return
        LOGGER.info(
            "DORL-DOSER step=%s loss=%.6f actor=%.6f vf=%.6f "
            "doser=%.6f ood_ratio=%.4f",
            self.learn_step,
            metrics.get("loss", 0.0),
            metrics.get("loss/actor", 0.0),
            metrics.get("loss/vf", 0.0),
            metrics.get("loss/doser_total", 0.0),
            metrics.get("ood/ood_ratio", 0.0),
        )

    def learn(
        self,
        batch: Batch,
        batch_size: int,
        repeat: int,
        **kwargs: Any,
    ) -> Dict[str, List[float]]:
        """执行一次 on-policy A2C + DOSER critic 更新。

        Args:
            batch (Batch): 已由 `process_fn` 计算 returns/adv 的训练 batch。
            batch_size (int): minibatch 大小。
            repeat (int): 对同一 batch 重复更新次数。
            **kwargs (Any): 兼容 Tianshou trainer 的额外参数。

        Returns:
            Dict[str, List[float]]: 训练损失和 OOD 指标序列。
        """

        del kwargs
        losses: List[float] = []
        actor_losses: List[float] = []
        vf_losses: List[float] = []
        ent_losses: List[float] = []
        doser_losses: List[float] = []
        aux_losses: List[float] = []
        penalty_losses: List[float] = []
        compensation_losses: List[float] = []
        ood_ratios: List[float] = []
        positive_ratios: List[float] = []
        negative_ratios: List[float] = []

        optim_rl, optim_state = self.optim
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                dist = self(
                    minibatch,
                    self._buffer,
                    indices=minibatch.indices,
                    is_obs=True,
                ).dist
                log_prob = dist.log_prob(minibatch.act)
                log_prob = log_prob.reshape(len(minibatch.adv), -1).transpose(0, 1)
                actor_loss = -(log_prob * minibatch.adv).mean()

                obs_emb = self.state_tracker(
                    self._buffer,
                    minibatch.indices,
                    is_obs=True,
                )
                value = self.critic(obs_emb).flatten()
                vf_loss = F.mse_loss(minibatch.returns, value)
                ent_loss = dist.entropy().mean()

                aux_loss, aux_metrics = self._compute_auxiliary_critic_loss(
                    minibatch,
                    obs_emb,
                )
                ood_loss, ood_metrics = self._compute_ood_regularization(
                    minibatch,
                    obs_emb,
                    dist,
                )
                doser_loss = self.doser_aux_critic_coef * aux_loss + ood_loss
                loss = (
                    actor_loss
                    + self._weight_vf * vf_loss
                    - self._weight_ent * ent_loss
                    + doser_loss
                )

                optim_rl.zero_grad()
                optim_state.zero_grad()
                loss.backward()
                if self._grad_norm:
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(),
                        max_norm=self._grad_norm,
                    )
                optim_rl.step()
                optim_state.step()

                self.learn_step += 1
                metrics = {
                    "loss": float(loss.detach().cpu().item()),
                    "loss/actor": float(actor_loss.detach().cpu().item()),
                    "loss/vf": float(vf_loss.detach().cpu().item()),
                    "loss/ent": float(ent_loss.detach().cpu().item()),
                    "loss/doser_total": float(doser_loss.detach().cpu().item()),
                    **aux_metrics,
                    **ood_metrics,
                }
                self._log_training_metrics(metrics)

                losses.append(metrics["loss"])
                actor_losses.append(metrics["loss/actor"])
                vf_losses.append(metrics["loss/vf"])
                ent_losses.append(metrics["loss/ent"])
                doser_losses.append(metrics["loss/doser_total"])
                aux_losses.append(metrics["loss/doser_aux_q"] + metrics["loss/doser_aux_v"])
                penalty_losses.append(metrics["loss/doser_penalty"])
                compensation_losses.append(metrics["loss/doser_compensation"])
                ood_ratios.append(metrics["ood/ood_ratio"])
                positive_ratios.append(metrics["ood/positive_ratio"])
                negative_ratios.append(metrics["ood/negative_ratio"])

        return {
            "loss": losses,
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
            "loss/doser": doser_losses,
            "loss/doser_aux": aux_losses,
            "loss/doser_penalty": penalty_losses,
            "loss/doser_compensation": compensation_losses,
            "ood/ood_ratio": ood_ratios,
            "ood/positive_ratio": positive_ratios,
            "ood/negative_ratio": negative_ratios,
        }
