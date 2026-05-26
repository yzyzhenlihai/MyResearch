"""On-policy DORL-DOSER 策略实现。

该模块在 Tianshou A2CPolicy 的基础上加入 DOSER OOD critic 正则。
actor、GAE、A2C value loss 和 entropy loss 保持原有语义；新增的
OOD penalty / compensation 仅通过 augmented critic 的辅助 Q/V 分支
参与优化。
"""

from __future__ import annotations

import logging
import math
import os
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

try:
    import swanlab  # type: ignore
except ImportError:
    swanlab = None


LOGGER = logging.getLogger(__name__)

SWANLAB_DISABLED_VALUES = {"1", "true", "yes", "on"}
"""将环境变量解析为关闭 SwanLab 时认定为真值的集合。"""

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

DEFAULT_ACTOR_TOPK = 64
"""rerank 候选集中来自 actor 原始概率的默认 top-k 数量。"""

DEFAULT_DIFFUSION_CANDIDATES = 64
"""rerank 候选集中来自行为扩散采样的默认候选数量。"""

DEFAULT_RANDOM_CANDIDATES = 16
"""rerank 候选集中用于保持探索的默认随机候选数量。"""

DEFAULT_RERANK_TEMPERATURE = 1.0
"""actor 原始 log-prob 进入 rerank score 前的默认温度。"""

DEFAULT_RERANK_ALPHA_Q = 1.0
"""auxiliary Q 分数在 rerank score 中的默认权重。"""

DEFAULT_RERANK_BETA_ACTION_OOD = 0.5
"""动作 OOD penalty 在 rerank score 中的默认权重。"""

DEFAULT_RERANK_GAMMA_REWARD = 0.2
"""counterfactual reward prior 在 rerank score 中的默认权重。"""

DEFAULT_ACTION_THRESHOLD_SCALE = 1.0
"""行为 OOD 阈值默认缩放系数。"""

DEFAULT_STATE_THRESHOLD_SCALE = 2.0
"""状态 OOD 阈值默认缩放系数，用于缓解在线 counterfactual 状态过度 OOD。"""

NEGATIVE_MASK_VALUE = -1.0e9
"""离散动作被屏蔽时写入 score 张量的稳定负值。"""

MIN_PROBABILITY = 1.0e-8
"""计算 log-prob 和归一化时使用的最小概率。"""

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
        doser_enable_rerank: bool = True,
        doser_actor_topk: int = DEFAULT_ACTOR_TOPK,
        doser_diffusion_candidates: int = DEFAULT_DIFFUSION_CANDIDATES,
        doser_random_candidates: int = DEFAULT_RANDOM_CANDIDATES,
        doser_rerank_temperature: float = DEFAULT_RERANK_TEMPERATURE,
        doser_rerank_alpha_q: float = DEFAULT_RERANK_ALPHA_Q,
        doser_rerank_beta_action_ood: float = DEFAULT_RERANK_BETA_ACTION_OOD,
        doser_rerank_gamma_reward: float = DEFAULT_RERANK_GAMMA_REWARD,
        doser_action_threshold_scale: float = DEFAULT_ACTION_THRESHOLD_SCALE,
        doser_state_threshold_scale: float = DEFAULT_STATE_THRESHOLD_SCALE,
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
            doser_enable_rerank (bool): 是否启用扩散引导 rerank 动作出口。
            doser_actor_topk (int): actor top-k 候选数量。
            doser_diffusion_candidates (int): 行为扩散候选数量。
            doser_random_candidates (int): 随机探索候选数量。
            doser_rerank_temperature (float): actor log-prob 温度，必须大于 0。
            doser_rerank_alpha_q (float): auxiliary Q rerank 权重。
            doser_rerank_beta_action_ood (float): action OOD penalty rerank 权重。
            doser_rerank_gamma_reward (float): reward prior rerank 权重。
            doser_action_threshold_scale (float): action OOD 阈值缩放系数。
            doser_state_threshold_scale (float): state OOD 阈值缩放系数。
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
            doser_actor_topk=doser_actor_topk,
            doser_diffusion_candidates=doser_diffusion_candidates,
            doser_random_candidates=doser_random_candidates,
            doser_rerank_temperature=doser_rerank_temperature,
            doser_action_threshold_scale=doser_action_threshold_scale,
            doser_state_threshold_scale=doser_state_threshold_scale,
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
        self.device = critic.device
        self.doser_beta = float(doser_beta)
        self.doser_lam = float(doser_lam)
        self.doser_eta = float(doser_eta)
        self.doser_expectile = float(doser_expectile)
        self.doser_q_min = float(doser_q_min)
        self.doser_aux_critic_coef = float(doser_aux_critic_coef)
        self.doser_detach_aux_state = bool(doser_detach_aux_state)
        self.doser_action_samples = int(doser_action_samples)
        self.diffusion_sample_steps = int(diffusion_sample_steps)
        self.doser_enable_rerank = bool(doser_enable_rerank)
        self.doser_actor_topk = int(doser_actor_topk)
        self.doser_diffusion_candidates = int(doser_diffusion_candidates)
        self.doser_random_candidates = int(doser_random_candidates)
        self.doser_rerank_temperature = float(doser_rerank_temperature)
        self.doser_rerank_alpha_q = float(doser_rerank_alpha_q)
        self.doser_rerank_beta_action_ood = float(doser_rerank_beta_action_ood)
        self.doser_rerank_gamma_reward = float(doser_rerank_gamma_reward)
        self.doser_action_threshold_scale = float(doser_action_threshold_scale)
        self.doser_state_threshold_scale = float(doser_state_threshold_scale)
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
        doser_actor_topk: int,
        doser_diffusion_candidates: int,
        doser_random_candidates: int,
        doser_rerank_temperature: float,
        doser_action_threshold_scale: float,
        doser_state_threshold_scale: float,
    ) -> None:
        """校验 DOSER 超参数。

        Args:
            doser_expectile (float): expectile 系数。
            doser_action_samples (int): 候选动作数。
            diffusion_sample_steps (int): 扩散采样步数。
            log_interval (int): 日志间隔。
            doser_actor_topk (int): actor top-k 候选数量。
            doser_diffusion_candidates (int): 扩散候选数量。
            doser_random_candidates (int): 随机候选数量。
            doser_rerank_temperature (float): rerank 温度。
            doser_action_threshold_scale (float): action OOD 阈值缩放。
            doser_state_threshold_scale (float): state OOD 阈值缩放。

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
        if doser_actor_topk < 0:
            raise ValueError("doser_actor_topk must be non-negative.")
        if doser_diffusion_candidates < 0:
            raise ValueError("doser_diffusion_candidates must be non-negative.")
        if doser_random_candidates < 0:
            raise ValueError("doser_random_candidates must be non-negative.")
        if doser_actor_topk + doser_diffusion_candidates + doser_random_candidates <= 0:
            raise ValueError("At least one rerank candidate source must be enabled.")
        if doser_rerank_temperature <= 0.0:
            raise ValueError("doser_rerank_temperature must be positive.")
        if doser_action_threshold_scale <= 0.0:
            raise ValueError("doser_action_threshold_scale must be positive.")
        if doser_state_threshold_scale <= 0.0:
            raise ValueError("doser_state_threshold_scale must be positive.")

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

    def _get_action_mask(self, batch: Batch, is_obs: Optional[bool]) -> Optional[torch.Tensor]:
        """读取当前阶段的离散动作 mask。

        Args:
            batch (Batch): collector 或 replay buffer 中的 batch。
            is_obs (Optional[bool]): 是否使用当前状态；`False` 时读取 next mask。

        Returns:
            Optional[torch.Tensor]: 布尔动作 mask，形状为 `(batch_size, action_dim)`；
            当 batch 中没有对应 mask 时返回 `None`。
        """

        if self.action_type != "discrete":
            return None
        mask_name = "mask" if is_obs else "next_mask"
        action_mask = getattr(batch, mask_name, None)
        if action_mask is None:
            return None
        return torch.as_tensor(
            action_mask,
            device=self.device,
            dtype=torch.bool,
        )

    def _normalize_actor_probs(
        self,
        actor_probs: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """归一化 actor 概率并应用合法动作 mask。

        Args:
            actor_probs (torch.Tensor): actor 输出概率，形状为 `(batch_size, action_dim)`。
            action_mask (Optional[torch.Tensor]): 合法动作 mask。

        Returns:
            torch.Tensor: 可用于 `Categorical(probs=...)` 的概率张量。
        """

        probs = actor_probs.clamp_min(0.0)
        if action_mask is not None:
            probs = probs * action_mask.to(probs.dtype)
        row_sum = probs.sum(dim=-1, keepdim=True)
        invalid_rows = row_sum <= MIN_PROBABILITY
        if invalid_rows.any():
            fallback = torch.ones_like(probs)
            if action_mask is not None:
                fallback = fallback * action_mask.to(probs.dtype)
                empty_mask_rows = fallback.sum(dim=-1, keepdim=True) <= 0.0
                fallback = torch.where(
                    empty_mask_rows,
                    torch.ones_like(fallback),
                    fallback,
                )
            fallback_sum = fallback.sum(dim=-1, keepdim=True).clamp_min(1.0)
            fallback = fallback / fallback_sum
            probs = torch.where(invalid_rows, fallback, probs)
            row_sum = probs.sum(dim=-1, keepdim=True).clamp_min(MIN_PROBABILITY)
        return probs / row_sum.clamp_min(MIN_PROBABILITY)

    def _actor_topk_candidate_ids(
        self,
        actor_probs: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """从 actor 原始概率中选取 top-k 候选动作。

        Args:
            actor_probs (torch.Tensor): actor 输出概率。
            action_mask (Optional[torch.Tensor]): 合法动作 mask。

        Returns:
            Optional[torch.Tensor]: 候选 item id，形状为 `(batch_size, k)`。
        """

        if self.doser_actor_topk <= 0:
            return None
        candidate_count = min(self.doser_actor_topk, actor_probs.shape[-1])
        scores = actor_probs
        if action_mask is not None:
            scores = scores.masked_fill(~action_mask, NEGATIVE_MASK_VALUE)
        return torch.topk(scores, k=candidate_count, dim=-1).indices.long()

    def _diffusion_candidate_ids(self, obs_emb: torch.Tensor) -> Optional[torch.Tensor]:
        """通过行为扩散模型采样候选动作并映射到 item id。

        Args:
            obs_emb (torch.Tensor): 当前状态 embedding。

        Returns:
            Optional[torch.Tensor]: 扩散候选 item id，形状为
            `(batch_size, doser_diffusion_candidates)`。
        """

        if self.doser_diffusion_candidates <= 0:
            return None
        with torch.no_grad():
            aligned_states = self.ood_helper.normalize_obs_array(obs_emb)
            sampled_actions = self.diffusion_artifact.diffusion_model.sample(
                self.diffusion_artifact.behavior_model,
                cond=aligned_states,
                action_samples=self.doser_diffusion_candidates,
                n_steps=self.diffusion_sample_steps,
            )
            return self.ood_helper.map_action_embeddings_to_item_ids(
                sampled_actions,
            ).long()

    def _random_candidate_ids(
        self,
        batch_size: int,
        action_dim: int,
        action_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """生成随机探索候选动作。

        Args:
            batch_size (int): batch 大小。
            action_dim (int): 离散动作数量。
            action_mask (Optional[torch.Tensor]): 合法动作 mask。
            device (torch.device): 目标设备。

        Returns:
            Optional[torch.Tensor]: 随机候选 item id。
        """

        if self.doser_random_candidates <= 0:
            return None
        if action_mask is None:
            return torch.randint(
                low=0,
                high=action_dim,
                size=(batch_size, self.doser_random_candidates),
                device=device,
            )
        weights = action_mask.to(dtype=torch.float32, device=device)
        invalid_rows = weights.sum(dim=-1, keepdim=True) <= 0.0
        weights = torch.where(invalid_rows, torch.ones_like(weights), weights)
        return torch.multinomial(
            weights,
            num_samples=self.doser_random_candidates,
            replacement=True,
        ).long()

    def _stored_candidate_ids(
        self,
        batch: Batch,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """读取 replay buffer 中保存的 rerank 候选动作。

        Args:
            batch (Batch): 当前 batch，可能包含 `policy.doser_candidates`。
            device (torch.device): 目标设备。

        Returns:
            Optional[torch.Tensor]: 已保存候选动作；不存在时返回 `None`。
        """

        policy_batch = getattr(batch, "policy", None)
        if policy_batch is None or not hasattr(policy_batch, "doser_candidates"):
            return None
        candidates = getattr(policy_batch, "doser_candidates")
        return torch.as_tensor(candidates, device=device, dtype=torch.long)

    def _required_action_ids(
        self,
        batch: Batch,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """读取 learn 阶段必须保留在候选集中的历史动作。

        Args:
            batch (Batch): 当前 minibatch。
            device (torch.device): 目标设备。

        Returns:
            Optional[torch.Tensor]: 历史动作 id，形状为 `(batch_size, 1)`。
        """

        if "act" not in batch:
            return None

        raw_actions = batch.act
        if isinstance(raw_actions, Batch):
            if raw_actions.is_empty(recurse=True):
                return None
            raise TypeError(
                "batch.act must be tensor-like action ids, got non-empty Batch."
            )

        actions = torch.as_tensor(raw_actions, device=device).long().view(-1, 1)
        return actions

    @staticmethod
    def _is_learning_minibatch(batch: Batch) -> bool:
        """判断当前 batch 是否来自 on-policy learn 阶段。

        Args:
            batch (Batch): 当前输入 batch。

        Returns:
            bool: 当 batch 同时包含 A2C learn 所需的 `adv` 与 `returns`
            时返回 `True`。
        """

        return "adv" in batch and "returns" in batch

    def _merge_candidate_ids(
        self,
        candidate_tensors: List[Optional[torch.Tensor]],
        batch_size: int,
        action_dim: int,
        action_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """合并候选动作并构造候选 mask。

        Args:
            candidate_tensors (List[Optional[torch.Tensor]]): 候选 item id 列表。
            batch_size (int): batch 大小。
            action_dim (int): 动作数量。
            action_mask (Optional[torch.Tensor]): 合法动作 mask。
            device (torch.device): 目标设备。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 合并后的候选 id 和候选 mask。
        """

        valid_tensors = [
            tensor.to(device=device, dtype=torch.long).view(batch_size, -1)
            for tensor in candidate_tensors
            if tensor is not None and tensor.numel() > 0
        ]
        if valid_tensors:
            merged_ids = torch.cat(valid_tensors, dim=1).clamp(0, action_dim - 1)
        else:
            merged_ids = torch.empty(batch_size, 0, device=device, dtype=torch.long)

        candidate_mask = torch.zeros(
            batch_size,
            action_dim,
            device=device,
            dtype=torch.bool,
        )
        if merged_ids.numel() > 0:
            candidate_mask.scatter_(1, merged_ids, True)
        if action_mask is not None:
            candidate_mask = candidate_mask & action_mask

        # 学习阶段必须保证 replay 中实际执行过的动作仍有非零概率。
        required_ids = candidate_tensors[-1]
        if required_ids is not None and required_ids.numel() > 0:
            candidate_mask.scatter_(
                1,
                required_ids.to(
                    device=device,
                    dtype=torch.long,
                ).view(batch_size, -1).clamp(0, action_dim - 1),
                True,
            )

        empty_rows = ~candidate_mask.any(dim=1)
        if empty_rows.any():
            fallback_mask = (
                action_mask.clone()
                if action_mask is not None
                else torch.ones_like(candidate_mask)
            )
            empty_mask_rows = ~fallback_mask.any(dim=1)
            if empty_mask_rows.any():
                fallback_mask[empty_mask_rows] = True
            candidate_mask[empty_rows] = fallback_mask[empty_rows]
        return merged_ids, candidate_mask

    def _extract_user_ids(self, batch: Batch, row_ids: torch.Tensor) -> torch.Tensor:
        """从 batch 观测中提取指定候选行对应的用户 id。

        Args:
            batch (Batch): 当前 batch，`obs[:, 0]` 应为用户 id。
            row_ids (torch.Tensor): 候选动作所在的 batch 行号。

        Returns:
            torch.Tensor: 用户 id，形状为 `(num_candidates,)`。
        """

        obs = torch.as_tensor(batch.obs, device=row_ids.device)
        if obs.ndim == 1:
            user_ids = obs.long()
        else:
            user_ids = obs[:, 0].long()
        return user_ids.index_select(0, row_ids)

    def _rowwise_zscore(
        self,
        values: torch.Tensor,
        row_ids: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """按 batch 行对候选分数做 z-score 标准化。

        Args:
            values (torch.Tensor): 候选分数，形状为 `(num_candidates,)`。
            row_ids (torch.Tensor): 每个候选分数所属 batch 行。
            batch_size (int): batch 大小。

        Returns:
            torch.Tensor: 标准化后的候选分数。
        """

        if values.numel() == 0:
            return values
        ones = torch.ones_like(values)
        counts = torch.zeros(batch_size, device=values.device).scatter_add(
            0,
            row_ids,
            ones,
        ).clamp_min(1.0)
        sums = torch.zeros(batch_size, device=values.device).scatter_add(
            0,
            row_ids,
            values,
        )
        means = sums / counts
        centered = values - means.index_select(0, row_ids)
        variances = torch.zeros(batch_size, device=values.device).scatter_add(
            0,
            row_ids,
            centered.pow(2),
        ) / counts
        std = torch.sqrt(variances.index_select(0, row_ids) + MIN_PROBABILITY)
        zscore = centered / std
        singleton_rows = counts.index_select(0, row_ids) <= 1.0
        return torch.where(singleton_rows, torch.zeros_like(zscore), zscore)

    def _compute_reward_prior(
        self,
        batch: Batch,
        row_ids: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:
        """计算候选动作的 counterfactual reward prior。

        Args:
            batch (Batch): 当前 batch。
            row_ids (torch.Tensor): 候选动作所在行。
            action_ids (torch.Tensor): 候选 item id。

        Returns:
            torch.Tensor: reward prior，形状为 `(num_candidates,)`。
        """

        user_ids = self._extract_user_ids(batch, row_ids)
        rewards = self.reward_model.estimate(
            user_ids.detach().cpu().numpy(),
            action_ids.detach().cpu().numpy(),
        )
        return torch.as_tensor(rewards, device=action_ids.device, dtype=torch.float32)

    def _build_rerank_distribution(
        self,
        batch: Batch,
        obs_emb: torch.Tensor,
        actor_probs: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """构造 DOSER rerank 后的动作概率分布。

        Args:
            batch (Batch): 当前 batch。
            obs_emb (torch.Tensor): state tracker 输出状态。
            actor_probs (torch.Tensor): actor 原始概率。
            action_mask (Optional[torch.Tensor]): 合法动作 mask。

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: rerank 概率和日志指标。
        """

        batch_size, action_dim = actor_probs.shape
        device = actor_probs.device
        stored_ids = self._stored_candidate_ids(batch, device)
        required_ids = (
            self._required_action_ids(batch, device)
            if self._is_learning_minibatch(batch)
            else None
        )
        if stored_ids is not None:
            candidate_sources = [stored_ids, required_ids]
        else:
            candidate_sources = [
                self._actor_topk_candidate_ids(actor_probs, action_mask),
                self._diffusion_candidate_ids(obs_emb),
                self._random_candidate_ids(
                    batch_size,
                    action_dim,
                    action_mask,
                    device,
                ),
                required_ids,
            ]
        merged_ids, candidate_mask = self._merge_candidate_ids(
            candidate_tensors=candidate_sources,
            batch_size=batch_size,
            action_dim=action_dim,
            action_mask=action_mask,
            device=device,
        )

        row_ids, action_ids = candidate_mask.nonzero(as_tuple=True)
        rerank_scores = torch.full_like(actor_probs, NEGATIVE_MASK_VALUE)
        if action_ids.numel() == 0:
            fallback_probs = self._normalize_actor_probs(actor_probs, action_mask)
            return fallback_probs, {"rerank/enabled": 0.0}

        with torch.no_grad():
            candidate_states = obs_emb.index_select(0, row_ids)
            candidate_actions = self.ood_helper.lookup_action_embeddings(action_ids)
            q_values = self.critic.q_min(candidate_states, candidate_actions)
            action_error = self.ood_helper.compute_action_error(
                candidate_actions,
                candidate_states,
            )
            reward_prior = self._compute_reward_prior(batch, row_ids, action_ids)

        actor_log_prob = torch.log(
            actor_probs.index_select(0, row_ids).gather(
                1,
                action_ids.view(-1, 1),
            ).flatten().clamp_min(MIN_PROBABILITY)
        )
        q_score = self._rowwise_zscore(q_values.detach(), row_ids, batch_size)
        reward_score = self._rowwise_zscore(
            reward_prior.detach(),
            row_ids,
            batch_size,
        )
        action_threshold = (
            self.diffusion_artifact.action_threshold
            * self.doser_action_threshold_scale
        )
        action_ood_penalty = F.relu(
            action_error.detach() / max(action_threshold, MIN_PROBABILITY) - 1.0
        )
        final_scores = (
            actor_log_prob / self.doser_rerank_temperature
            + self.doser_rerank_alpha_q * q_score
            + self.doser_rerank_gamma_reward * reward_score
            - self.doser_rerank_beta_action_ood * action_ood_penalty
        )
        rerank_scores.index_put_((row_ids, action_ids), final_scores)
        rerank_probs = torch.softmax(rerank_scores, dim=-1)

        metrics = {
            "rerank/enabled": 1.0,
            "rerank/candidate_size": float(
                candidate_mask.float().sum(dim=1).mean().detach().cpu().item()
            ),
            "rerank/q_score": float(q_values.detach().mean().cpu().item()),
            "rerank/action_error": float(action_error.detach().mean().cpu().item()),
            "rerank/reward_prior": float(reward_prior.detach().mean().cpu().item()),
            "rerank/action_ood_penalty": float(action_ood_penalty.detach().mean().cpu().item()),
        }
        return rerank_probs, metrics

    def forward(
        self,
        batch: Batch,
        buffer: Optional[ReplayBuffer],
        indices: Optional[np.ndarray] = None,
        is_obs: Optional[bool] = None,
        is_train: bool = True,
        state: Optional[Any] = None,
        use_batch_in_statetracker: bool = False,
        **kwargs: Any,
    ) -> Batch:
        """执行 A2C actor 前向，并可选应用 DOSER rerank 动作出口。

        Args:
            batch (Batch): 当前 batch。
            buffer (Optional[ReplayBuffer]): replay buffer。
            indices (Optional[np.ndarray]): batch 在 buffer 中的索引。
            is_obs (Optional[bool]): 是否构造当前状态。
            is_train (bool): 是否处于训练阶段。
            state (Optional[Any]): actor hidden state。
            use_batch_in_statetracker (bool): 是否使用 batch 构造状态。
            **kwargs (Any): 兼容外部调用的额外参数。

        Returns:
            Batch: 包含 `act`、`dist`、`logits` 和可选 rerank policy 信息。
        """

        del kwargs
        obs_emb = self.state_tracker(
            buffer=buffer,
            indices=indices,
            is_obs=is_obs,
            batch=batch,
            is_train=is_train,
            use_batch_in_statetracker=use_batch_in_statetracker,
        )
        batch_info = getattr(batch, "info", {})
        actor_probs, hidden = self.actor(obs_emb, state=state, info=batch_info)
        action_mask = self._get_action_mask(batch, is_obs)
        policy_batch = Batch()
        rerank_metrics: Dict[str, float]

        if self.doser_enable_rerank and self.action_type == "discrete" and is_obs:
            logits, rerank_metrics = self._build_rerank_distribution(
                batch=batch,
                obs_emb=obs_emb,
                actor_probs=actor_probs,
                action_mask=action_mask,
            )
        else:
            logits = self._normalize_actor_probs(actor_probs, action_mask)
            rerank_metrics = {"rerank/enabled": 0.0}

        dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            act = logits.argmax(-1)
        else:
            act = dist.sample()
        return Batch(
            logits=logits,
            act=act,
            state=hidden,
            dist=dist,
            policy=policy_batch,
            doser_rerank_metrics=rerank_metrics,
        )

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
                buffer=getattr(self, "_buffer", None),
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
            action_threshold = (
                self.diffusion_artifact.action_threshold
                * self.doser_action_threshold_scale
            )
            state_threshold = (
                self.diffusion_artifact.state_threshold
                * self.doser_state_threshold_scale
            )
            action_ood_mask = action_error > action_threshold
            state_ood_mask = state_error > state_threshold
            ood_mask = torch.logical_or(action_ood_mask, state_ood_mask)

            best_ids, best_actions, best_q = self.ood_helper.select_best_id_action(
                states=obs_emb,
                critic=self.critic,
                action_samples=self.doser_action_samples,
                diffusion_sample_steps=self.diffusion_sample_steps,
            )
            best_next_state = self.ood_helper.build_counterfactual_next_state(
                batch=minibatch,
                buffer=getattr(self, "_buffer", None),
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
            "ood/action_threshold": float(action_threshold),
            "ood/state_threshold": float(state_threshold),
            "ood/action_ood_ratio": float(
                action_ood_mask.float().mean().detach().cpu().item()
            ),
            "ood/state_ood_ratio": float(
                state_ood_mask.float().mean().detach().cpu().item()
            ),
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

    @staticmethod
    def _is_swanlab_disabled() -> bool:
        """判断当前进程是否关闭 SwanLab 指标写入。

        Returns:
            bool: 当环境变量明确关闭 SwanLab 时返回 `True`。
        """

        mode = os.environ.get("SWANLAB_MODE", "").strip().lower()
        disabled_flag = os.environ.get("SWANLAB_DISABLED", "").strip().lower()
        legacy_mode = os.environ.get("WANDB_MODE", "").strip().lower()
        legacy_disabled_flag = os.environ.get("WANDB_DISABLED", "").strip().lower()
        return (
            mode == "disabled"
            or legacy_mode == "disabled"
            or disabled_flag in SWANLAB_DISABLED_VALUES
            or legacy_disabled_flag in SWANLAB_DISABLED_VALUES
        )

    @staticmethod
    def _to_finite_float(value: Any) -> Optional[float]:
        """把指标值转换为 SwanLab 可记录的有限浮点数。

        Args:
            value (Any): 待转换的原始指标值。

        Returns:
            Optional[float]: 转换后的有限浮点数；无法转换或非有限值时返回 `None`。
        """

        if hasattr(value, "item"):
            value = value.item()
        if not isinstance(value, (float, int, np.number)):
            return None
        scalar = float(value)
        if not math.isfinite(scalar):
            return None
        return scalar

    def _safe_swanlab_log(self, metrics: Dict[str, float]) -> None:
        """安全地把 DORL-DOSER update 指标写入 SwanLab。

        Args:
            metrics (Dict[str, float]): 已聚合的训练标量指标。

        Returns:
            None: SwanLab 不可用或未初始化时静默跳过。
        """

        if swanlab is None or self._is_swanlab_disabled():
            return

        log_fn = getattr(swanlab, "log", None)
        get_run_fn = getattr(swanlab, "get_run", None)
        active_run = (
            get_run_fn()
            if get_run_fn is not None
            else getattr(swanlab, "run", None)
        )
        if log_fn is None or active_run is None:
            return

        scalar_metrics = {
            key: scalar
            for key, value in metrics.items()
            if (scalar := self._to_finite_float(value)) is not None
        }
        if not scalar_metrics:
            return

        step = int(self.learn_step)
        try:
            log_fn(scalar_metrics, step=step)
        except TypeError:
            # 兼容不支持 step 参数的 SwanLab 版本。
            log_fn(scalar_metrics)
        except Exception as exc:  # pragma: no cover - 仅用于保护长训练不中断
            if not getattr(self, "_swanlab_log_warning_emitted", False):
                LOGGER.warning("Skip SwanLab update logging because of error: %s", exc)
                self._swanlab_log_warning_emitted = True

    def _extract_scalar_metrics(self, metrics: Any) -> Dict[str, float]:
        """从 dict 或 Batch 中提取可记录的有限标量。

        Args:
            metrics (Any): 可能为普通字典或 Tianshou Batch 的指标容器。

        Returns:
            Dict[str, float]: 已转换为 Python float 的指标字典。
        """

        if metrics is None:
            return {}
        metric_items = metrics.items() if hasattr(metrics, "items") else []
        return {
            key: scalar
            for key, value in metric_items
            if (scalar := self._to_finite_float(value)) is not None
        }

    def _log_training_metrics(self, metrics: Dict[str, float]) -> None:
        """按固定间隔写入本地日志和 SwanLab 指标。

        Args:
            metrics (Dict[str, float]): 当前 minibatch 指标。
        """

        should_log = self.learn_step == 1 or self.learn_step % self.log_interval == 0
        if not should_log:
            return

        LOGGER.info(
            "DORL-DOSER step=%s env_step=%s loss=%.6f actor=%.6f "
            "vf=%.6f ent=%.6f doser=%.6f aux=%.6f penalty=%.6f "
            "compensation=%.6f ood_ratio=%.4f positive_ratio=%.4f "
            "negative_ratio=%.4f action_ood=%.4f state_ood=%.4f "
            "rerank_candidates=%.2f",
            self.learn_step,
            int(metrics.get("trainer/env_step", -1)),
            metrics.get("loss", 0.0),
            metrics.get("loss/actor", 0.0),
            metrics.get("loss/vf", 0.0),
            metrics.get("loss/ent", 0.0),
            metrics.get("loss/doser", 0.0),
            metrics.get("loss/doser_aux", 0.0),
            metrics.get("loss/doser_penalty", 0.0),
            metrics.get("loss/doser_compensation", 0.0),
            metrics.get("ood/ood_ratio", 0.0),
            metrics.get("ood/positive_ratio", 0.0),
            metrics.get("ood/negative_ratio", 0.0),
            metrics.get("ood/action_ood_ratio", 0.0),
            metrics.get("ood/state_ood_ratio", 0.0),
            metrics.get("rerank/candidate_size", 0.0),
        )
        self._safe_swanlab_log(metrics)

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

        env_step = kwargs.get("env_step", None)
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
        action_ood_ratios: List[float] = []
        state_ood_ratios: List[float] = []
        rerank_candidate_sizes: List[float] = []

        optim_rl, optim_state = self.optim
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                result = self(
                    minibatch,
                    getattr(self, "_buffer", None),
                    indices=minibatch.indices,
                    is_obs=True,
                )
                dist = result.dist
                action_tensor = torch.as_tensor(
                    minibatch.act,
                    device=result.logits.device,
                    dtype=torch.long,
                )
                log_prob = dist.log_prob(action_tensor)
                log_prob = log_prob.reshape(len(minibatch.adv), -1).transpose(0, 1)
                actor_loss = -(log_prob * minibatch.adv).mean()

                obs_emb = self.state_tracker(
                    getattr(self, "_buffer", None),
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
                aux_loss_value = float(aux_loss.detach().cpu().item())
                metrics = {
                    "loss": float(loss.detach().cpu().item()),
                    "loss/actor": float(actor_loss.detach().cpu().item()),
                    "loss/vf": float(vf_loss.detach().cpu().item()),
                    "loss/ent": float(ent_loss.detach().cpu().item()),
                    "loss/doser": float(doser_loss.detach().cpu().item()),
                    "loss/doser_aux": aux_loss_value,
                    **aux_metrics,
                    **ood_metrics,
                }
                rerank_metrics = self._extract_scalar_metrics(
                    getattr(result, "doser_rerank_metrics", {}),
                )
                metrics.update(rerank_metrics)
                if env_step is not None:
                    metrics["trainer/env_step"] = float(env_step)
                self._log_training_metrics(metrics)

                losses.append(metrics["loss"])
                actor_losses.append(metrics["loss/actor"])
                vf_losses.append(metrics["loss/vf"])
                ent_losses.append(metrics["loss/ent"])
                doser_losses.append(metrics["loss/doser"])
                aux_losses.append(metrics["loss/doser_aux"])
                penalty_losses.append(metrics["loss/doser_penalty"])
                compensation_losses.append(metrics["loss/doser_compensation"])
                ood_ratios.append(metrics["ood/ood_ratio"])
                positive_ratios.append(metrics["ood/positive_ratio"])
                negative_ratios.append(metrics["ood/negative_ratio"])
                action_ood_ratios.append(metrics["ood/action_ood_ratio"])
                state_ood_ratios.append(metrics["ood/state_ood_ratio"])
                rerank_candidate_sizes.append(
                    metrics.get("rerank/candidate_size", 0.0)
                )

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
            "ood/action_ood_ratio": action_ood_ratios,
            "ood/state_ood_ratio": state_ood_ratios,
            "rerank/candidate_size": rerank_candidate_sizes,
        }
