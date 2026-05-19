"""DARLR 动态奖励模拟环境。"""

import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from src.core.envs.Simulated_Env.penalty_ent_exp import (
    PenaltyEntExpSimulatedEnv,
    get_features_of_last_n_items_features,
)
from src.core.util.utils import clip0


class DARLRDynamicRewardEnv(PenaltyEntExpSimulatedEnv):
    """在 DORL reward 基础上接入 DARLR 动态奖励。

    该环境通过 `darlr_context` 接收 selector 选择出的 reference users，
    在 `_compute_pred_reward()` 中用参考用户集合重估 reward，并按动态
    uncertainty penalty 修正训练奖励。没有 selector context 时完全回退
    到 `PenaltyEntExpSimulatedEnv` 的原始逻辑。

    关键属性：
        darlr_context: Collector 在每个 env.step 前注入的单环境上下文。
        _darlr_previous_reward: `(user, item)` 级别的上一轮动态 reward 缓存。
        last_darlr_metrics: 最近一步的 DARLR 诊断指标，会写入 info。
    """

    def __init__(
        self,
        *args: Any,
        lambda_uncertainty: float = 0.05,
        darlr_eps: float = 1.0e-8,
        dynamic_reward_mode: str = "reference_mean",
        dynamic_uncertainty_mode: str = "dynamic",
        **kwargs: Any,
    ) -> None:
        """初始化 DARLR 动态奖励环境。

        Args:
            *args (Any): 透传给 `PenaltyEntExpSimulatedEnv` 的位置参数。
            lambda_uncertainty (float): 动态不确定性惩罚权重。
            darlr_eps (float): 动态不确定性分母稳定项。
            dynamic_reward_mode (str): 动态奖励模式，支持 `reference_mean`
                和 `static_dorl`。
            dynamic_uncertainty_mode (str): 不确定性模式，支持 `dynamic`、
                `static` 和 `off`。
            **kwargs (Any): 透传给 `PenaltyEntExpSimulatedEnv` 的关键字参数。

        Raises:
            ValueError: 当模式参数非法时抛出。
        """

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
        self._darlr_previous_reward: Dict[Tuple[int, int], float] = {}
        self.last_darlr_metrics: Dict[str, float] = {}

    def step(self, action):
        """执行一步环境交互，并把 DARLR 诊断指标写入 info。

        Args:
            action: 推荐动作，即物品 id。

        Returns:
            tuple: `(state, reward, terminated, truncated, info)`，与
            Gymnasium step API 保持一致。
        """

        state, reward, terminated, truncated, info = super().step(action)
        info.update(self.last_darlr_metrics)
        self.darlr_context = None
        return state, reward, terminated, truncated, info

    def _compute_pred_reward(self, action):
        """计算 DARLR 动态训练奖励。

        Args:
            action: 当前推荐物品 id。

        Returns:
            float: 经过动态 reward、动态 uncertainty、entropy 和 exposure
            修正后的训练 reward。
        """

        action_id = int(np.asarray(action).item())
        context = self._normalize_context(self.darlr_context)
        if context is None or self.dynamic_reward_mode == "static_dorl":
            reward = super()._compute_pred_reward(action_id)
            self.last_darlr_metrics = {
                "darlr/dynamic_reward": float(self.predicted_mat[self.cur_user, action_id]),
                "darlr/dynamic_uncertainty": 0.0,
                "darlr/similarity_gain": 0.0,
                "darlr/diversity_gain": 0.0,
            }
            return reward

        original_reward = float(self.predicted_mat[self.cur_user, action_id])
        selected_users = np.asarray(context.get("selected_users", []), dtype=np.int64)
        selected_users = selected_users.reshape(-1)
        if selected_users.size == 0:
            reward = super()._compute_pred_reward(action_id)
            self.last_darlr_metrics = {
                "darlr/dynamic_reward": original_reward,
                "darlr/dynamic_uncertainty": 0.0,
                "darlr/similarity_gain": 0.0,
                "darlr/diversity_gain": 0.0,
            }
            return reward

        selected_users = np.clip(selected_users, 0, self.predicted_mat.shape[0] - 1)
        dynamic_reward = float(np.mean(self.predicted_mat[selected_users, action_id]))
        similarity_gain = float(context.get("similarity_gain", 0.0))
        diversity_gain = float(context.get("diversity_gain", 0.0))

        cache_key = (int(self.cur_user), action_id)
        previous_reward = self._darlr_previous_reward.get(cache_key, original_reward)
        dynamic_uncertainty = self._compute_dynamic_uncertainty(
            dynamic_reward=dynamic_reward,
            previous_reward=previous_reward,
            similarity_gain=similarity_gain,
            diversity_gain=diversity_gain,
        )
        self._darlr_previous_reward[cache_key] = dynamic_reward

        entropy = self._compute_entropy_bonus(action_id)
        shaped_reward = (
            dynamic_reward
            - self.lambda_uncertainty * dynamic_uncertainty
            + self.lambda_entropy * entropy
            - self.MIN_R
        )
        final_reward = self._apply_exposure_and_clip(shaped_reward, action_id)
        self.last_darlr_metrics = {
            "darlr/dynamic_reward": dynamic_reward,
            "darlr/dynamic_uncertainty": dynamic_uncertainty,
            "darlr/similarity_gain": similarity_gain,
            "darlr/diversity_gain": diversity_gain,
        }
        return final_reward

    def _compute_entropy_bonus(self, action_id: int) -> float:
        """复用 DORL 的历史行为 entropy bonus。

        Args:
            action_id (int): 当前推荐物品 id。

        Returns:
            float: 当前历史窗口下的 entropy bonus。
        """

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
        """计算 feature-level entropy。

        Args:
            window_size (int): 历史窗口长度。
            action_set (Tuple[int, ...]): 历史物品序列。

        Returns:
            float: feature-level entropy 平均值。
        """

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

    def _compute_dynamic_uncertainty(
        self,
        dynamic_reward: float,
        previous_reward: float,
        similarity_gain: float,
        diversity_gain: float,
    ) -> float:
        """计算动态不确定性惩罚项。

        Args:
            dynamic_reward (float): 当前 reference users 聚合 reward。
            previous_reward (float): 该 `(user, item)` 上一次 reward 估计。
            similarity_gain (float): selector 平均相似性增益。
            diversity_gain (float): selector 平均多样性增益。

        Returns:
            float: 动态不确定性数值。
        """

        reward_delta = abs(dynamic_reward - previous_reward)
        if self.dynamic_uncertainty_mode == "off":
            return 0.0
        if self.dynamic_uncertainty_mode == "static":
            return float(reward_delta)
        denominator = max(similarity_gain + diversity_gain, self.darlr_eps)
        return float(reward_delta / denominator)

    def _apply_exposure_and_clip(self, shaped_reward: float, action_id: int) -> float:
        """应用 DORL exposure intervention 与非负裁剪。

        Args:
            shaped_reward (float): 已完成 dynamic reward 和 entropy 计算的奖励。
            action_id (int): 当前推荐物品 id。

        Returns:
            float: 最终环境 reward。
        """

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

    @staticmethod
    def _normalize_context(context: Optional[Any]) -> Optional[Dict[str, Any]]:
        """将 Batch 或 dict 形式的 selector context 统一成字典。

        Args:
            context (Optional[Any]): Collector 注入的上下文。

        Returns:
            Optional[Dict[str, Any]]: 标准字典；无有效上下文时返回 None。
        """

        if context is None:
            return None
        if isinstance(context, dict):
            return context
        if hasattr(context, "keys"):
            return {key: context[key] for key in context.keys()}
        return None
