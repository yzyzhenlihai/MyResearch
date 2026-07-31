"""DARLR 动态奖励模拟环境。

按照 `docs/DARLR_DORL融合复现方案.md` 中的 P0 修正列表：

* 环境不再私有维护 `previous reward` (由策略侧共享的
  `DynamicRewardStore` 负责);
* 从 `darlr_context` 直接接收策略侧已经计算好的
  `dynamic_reward / previous_reward / dynamic_uncertainty / base_reward
  / similarity_gain / diversity_gain / static_uncertainty`;
* `dynamic_reward_mode=static_dorl` 时严格回退到 DORL 完整公式
  `r̂ - λ_U * V0[u,i] + λ_E * P_E`, 而不是只回退 entropy;
* `dynamic_uncertainty_mode=static` 使用世界模型 ensemble 的
  `V0[u,i]`, 而不是 `abs(dynamic-previous)`;
* selected user id 越界直接断言, 不再用 `np.clip` 静默修正;
* `info` 中记录 `base/dynamic/previous/delta/numerator/denominator/
  raw_rec/env_rec` 等诊断字段, 便于消融/复现。
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from src.core.envs.Simulated_Env.penalty_ent_exp import (  # noqa: E402
    PenaltyEntExpSimulatedEnv,
    get_features_of_last_n_items_features,
)
from src.core.util.utils import clip0  # noqa: E402


class DARLRDynamicRewardEnv(PenaltyEntExpSimulatedEnv):
    """在 DORL 奖励基础上叠加 DARLR 动态奖励 shaping."""

    def __init__(
        self,
        *args: Any,
        lambda_uncertainty: float = 0.05,
        darlr_eps: float = 1.0e-8,
        dynamic_reward_mode: str = "reference_mean",
        dynamic_uncertainty_mode: str = "dynamic",
        **kwargs: Any,
    ) -> None:
        if dynamic_reward_mode not in {"reference_mean", "static_dorl"}:
            raise ValueError("Unsupported dynamic_reward_mode.")
        if dynamic_uncertainty_mode not in {"dynamic", "static", "off"}:
            raise ValueError("Unsupported dynamic_uncertainty_mode.")
        super().__init__(*args, **kwargs)
        self.lambda_uncertainty = float(lambda_uncertainty)
        self.darlr_eps = max(float(darlr_eps), 1.0e-8)
        self.dynamic_reward_mode = dynamic_reward_mode
        self.dynamic_uncertainty_mode = dynamic_uncertainty_mode
        self.darlr_context: Optional[Dict[str, Any]] = None
        self.last_darlr_metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # gymnasium API 集成
    # ------------------------------------------------------------------

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        info.update(self.last_darlr_metrics)
        self.darlr_context = None
        return state, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # DARLR reward 主流程
    # ------------------------------------------------------------------

    def _compute_pred_reward(self, action):
        action_id = int(np.asarray(action).item())
        self._assert_action_id_valid(action_id)
        context = self._normalize_context(self.darlr_context)

        # 分支 1: 完整 DORL 静态公式 (static_dorl 消融) —— 回退到父类，
        # 父类已支持 maxvar_mat + lambda_variance + entropy + exposure。
        if self.dynamic_reward_mode == "static_dorl":
            reward = super()._compute_pred_reward(action_id)
            base_reward = float(self.predicted_mat[self.cur_user, action_id])
            static_variance = self._read_static_variance(action_id)
            self.last_darlr_metrics = {
                "darlr/mode": 0.0,  # 0 = static_dorl
                "darlr/base_reward": base_reward,
                "darlr/dynamic_reward": base_reward,
                "darlr/previous_reward": base_reward,
                "darlr/reward_delta": 0.0,
                "darlr/uncertainty_numerator": 0.0,
                "darlr/uncertainty_denominator": 0.0,
                "darlr/dynamic_uncertainty": 0.0,
                "darlr/static_uncertainty": static_variance,
                "darlr/similarity_gain": 0.0,
                "darlr/diversity_gain": 0.0,
                "darlr/raw_rec_reward": reward + self.MIN_R,
                "darlr/env_rec_reward": reward,
            }
            return reward

        base_reward = float(self.predicted_mat[self.cur_user, action_id])

        # 分支 2: context 缺失, 退回 DORL 静态基线以避免 crash, 并在日志里
        # 显式标记, 便于诊断 (例如测试阶段就应该走这里)。
        if context is None:
            reward = super()._compute_pred_reward(action_id)
            self.last_darlr_metrics = self._empty_metrics(base_reward, raw=reward + self.MIN_R, env=reward, mode_id=-1.0)
            return reward

        # 从策略侧上下文读取已经计算好的动态量
        dynamic_reward = float(context.get("dynamic_reward", base_reward))
        previous_reward = float(context.get("previous_reward", base_reward))
        similarity_gain = float(context.get("similarity_gain", 0.0))
        diversity_gain = float(context.get("diversity_gain", 0.0))
        numerator = float(context.get("uncertainty_numerator", abs(dynamic_reward - previous_reward)))
        denominator = float(context.get("uncertainty_denominator", max(similarity_gain + diversity_gain, self.darlr_eps)))
        dynamic_uncertainty_ctx = float(context.get("dynamic_uncertainty", numerator / max(denominator, self.darlr_eps)))
        static_uncertainty_ctx = float(context.get("static_uncertainty", self._read_static_variance(action_id)))

        # 越界检查 (selected users 用于 dynamic reward 聚合, 若上下文里带过来
        # 就再确认一遍, 而不是像旧实现那样用 np.clip 静默修改)
        selected_users = context.get("selected_users")
        if selected_users is not None:
            selected_users_np = np.asarray(selected_users, dtype=np.int64).reshape(-1)
            num_users = int(self.predicted_mat.shape[0])
            if selected_users_np.size and (
                selected_users_np.min() < 0 or selected_users_np.max() >= num_users
            ):
                raise IndexError(
                    "DARLR selected_users out of range: "
                    f"min={int(selected_users_np.min())}, "
                    f"max={int(selected_users_np.max())}, num_users={num_users}."
                )

        # 组装 uncertainty
        if self.dynamic_uncertainty_mode == "off":
            uncertainty = 0.0
        elif self.dynamic_uncertainty_mode == "static":
            uncertainty = static_uncertainty_ctx  # 使用 V0[u,i]
        else:
            uncertainty = dynamic_uncertainty_ctx

        entropy_bonus = self._compute_entropy_bonus(action_id)
        raw_rec_reward = (
            dynamic_reward
            - self.lambda_uncertainty * uncertainty
            + self.lambda_entropy * entropy_bonus
        )
        shaped_reward = raw_rec_reward - self.MIN_R
        final_reward = self._apply_exposure_and_clip(shaped_reward, action_id)

        self.last_darlr_metrics = {
            "darlr/mode": 1.0,  # 1 = dynamic reference_mean
            "darlr/base_reward": base_reward,
            "darlr/dynamic_reward": dynamic_reward,
            "darlr/previous_reward": previous_reward,
            "darlr/reward_delta": dynamic_reward - previous_reward,
            "darlr/uncertainty_numerator": numerator,
            "darlr/uncertainty_denominator": denominator,
            "darlr/dynamic_uncertainty": dynamic_uncertainty_ctx,
            "darlr/static_uncertainty": static_uncertainty_ctx,
            "darlr/uncertainty_applied": float(uncertainty),
            "darlr/similarity_gain": similarity_gain,
            "darlr/diversity_gain": diversity_gain,
            "darlr/entropy_bonus": float(entropy_bonus),
            "darlr/raw_rec_reward": raw_rec_reward,
            "darlr/env_rec_reward": final_reward,
        }
        return final_reward

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _compute_entropy_bonus(self, action_id: int) -> float:
        entropy = 0.0
        entropy_set = set(self.entropy_window) - {0}
        if not entropy_set:
            return entropy

        action_history = self.history_action[
            max(0, self.total_turn - self.step_n_actions + 1): self.total_turn + 1
        ]
        if hasattr(self.env_task, "lbe_item") and self.env_task.lbe_item:
            action_trans = self.env_task.lbe_item.inverse_transform(action_history)
        else:
            action_trans = action_history

        for window_size in entropy_set:
            if len(action_trans) < window_size:
                entropy += 1.0
                continue

            action_set = (
                tuple(sorted(action_trans[-window_size:]))
                if self.is_sorted
                else tuple(action_trans[-window_size:])
            )
            if self.feature_level:
                entropy += self._compute_feature_entropy(window_size, action_set)
            else:
                entropy += self.entropy_dict["map"].get(action_set, 1.0)
        return float(entropy)

    def _compute_feature_entropy(self, window_size: int, action_set: Tuple[int, ...]) -> float:
        feature_sets = get_features_of_last_n_items_features(
            window_size,
            action_set,
            self.map_item_feat,
            is_sort=self.is_sorted,
        )
        if len(feature_sets) == 0:
            return 1.0
        entropy_sum = 0.0
        for feature_set in feature_sets:
            entropy_sum += self.entropy_dict["map"].get(feature_set, 1.0)
        return float(entropy_sum / len(feature_sets))

    def _apply_exposure_and_clip(self, shaped_reward: float, action_id: int) -> float:
        turn_index = int(self.total_turn)
        if self.use_exposure_intervention:
            exposure_effect = self._compute_exposure_effect(turn_index, action_id)
        else:
            exposure_effect = 0.0
        if turn_index < self.env_task.max_turn:
            self._add_exposure_to_history(turn_index, exposure_effect)

        if self.version == "v1":
            final_reward = clip0(shaped_reward) / (1.0 + exposure_effect)
        else:
            final_reward = clip0(shaped_reward - exposure_effect)
        return max(0.0, float(final_reward))

    def _read_static_variance(self, action_id: int) -> float:
        if self.maxvar_mat is None:
            return 0.0
        return float(self.maxvar_mat[self.cur_user, action_id])

    def _assert_action_id_valid(self, action_id: int) -> None:
        num_items = int(self.predicted_mat.shape[1])
        if action_id < 0 or action_id >= num_items:
            raise IndexError(
                f"DARLR action id out of range: got {action_id}, num_items={num_items}."
            )

    def _empty_metrics(
        self,
        base_reward: float,
        raw: float,
        env: float,
        mode_id: float,
    ) -> Dict[str, float]:
        return {
            "darlr/mode": mode_id,
            "darlr/base_reward": base_reward,
            "darlr/dynamic_reward": base_reward,
            "darlr/previous_reward": base_reward,
            "darlr/reward_delta": 0.0,
            "darlr/uncertainty_numerator": 0.0,
            "darlr/uncertainty_denominator": 0.0,
            "darlr/dynamic_uncertainty": 0.0,
            "darlr/static_uncertainty": 0.0,
            "darlr/uncertainty_applied": 0.0,
            "darlr/similarity_gain": 0.0,
            "darlr/diversity_gain": 0.0,
            "darlr/entropy_bonus": 0.0,
            "darlr/raw_rec_reward": raw,
            "darlr/env_rec_reward": env,
        }

    @staticmethod
    def _normalize_context(context: Optional[Any]) -> Optional[Dict[str, Any]]:
        if context is None:
            return None
        if isinstance(context, dict):
            return context
        if hasattr(context, "keys"):
            return {key: context[key] for key in context.keys()}
        return None
