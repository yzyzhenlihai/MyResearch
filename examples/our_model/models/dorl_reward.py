"""DORL user model reward 封装。"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

from src.core.util.entropy_penalty import compute_step_entropy

LOGGER = logging.getLogger(__name__)


class DORLRewardModel:
    """封装 DORL 已训练 user model 的 reward shaping。

    该类同时支持 DORL entropy bonus 和 MOPO 风格 uncertainty penalty。
    对单步动作的 reward 计算为：

    `pred_reward + lambda_entropy * entropy - lambda_variance * uncertainty - MIN_R`

    其中 `MIN_R` 使用全局最小预测、entropy 下界和 uncertainty 上界做平移，
    最终 reward 再执行非负截断，保持与原模拟环境一致的正 reward 语义。
    """

    def __init__(
        self,
        predicted_mat_path: str,
        raw_user_to_index: Dict[int, int],
        device: torch.device,
        internal_to_raw_item_ids: Optional[Sequence[int]] = None,
        reward_shift: bool = True,
        use_exposure_intervention: bool = False,
        use_entropy_reward: bool = True,
        entropy_map: Optional[Mapping[Tuple[int, ...], float]] = None,
        entropy_window: Sequence[int] = (1, 2),
        lambda_entropy: float = 5.0,
        entropy_min: float = 0.0,
        feature_level: bool = True,
        map_item_feat: Optional[Mapping[int, Sequence[int]]] = None,
        is_sorted: bool = True,
        use_uncertainty_penalty: bool = True,
        maxvar_mat_path: str = "",
        lambda_variance: float = 0.05,
    ) -> None:
        """初始化 reward 模型。

        Args:
            predicted_mat_path (str): DeepFM 预测矩阵 pkl 路径。
            raw_user_to_index (Dict[int, int]): 原始 user id 到 KuaiEnv 内部行号的映射。
            device (torch.device): 计算设备。
            internal_to_raw_item_ids (Optional[Sequence[int]]): 内部 item id 到原始 item id
                的映射；为空时默认二者一致。
            reward_shift (bool): 是否减去预测矩阵全局最小值。
            use_exposure_intervention (bool): 是否启用 exposure intervention；MVP 不支持。
            use_entropy_reward (bool): 是否启用 DORL entropy bonus。
            entropy_map (Optional[Mapping[Tuple[int, ...], float]]): 历史窗口到 entropy 的查表。
            entropy_window (Sequence[int]): DORL entropy 窗口列表。
            lambda_entropy (float): entropy bonus 权重。
            entropy_min (float): entropy 下界，用于 reward shift。
            feature_level (bool): 是否使用 feature-level entropy。
            map_item_feat (Optional[Mapping[int, Sequence[int]]]): 原始 item id 到特征列表的映射。
            is_sorted (bool): entropy 窗口是否排序。
            use_uncertainty_penalty (bool): 是否启用 uncertainty penalty。
            maxvar_mat_path (str): ensemble 最大方差矩阵路径。
            lambda_variance (float): uncertainty penalty 权重。

        Raises:
            FileNotFoundError: 当预测矩阵不存在时抛出。
            NotImplementedError: 当开启 exposure intervention 时抛出。
            ValueError: 当预测矩阵维度不合法时抛出。
        """

        if use_exposure_intervention:
            raise NotImplementedError("DORL-MAC MVP does not implement exposure intervention.")
        if use_entropy_reward and entropy_map is None:
            raise ValueError("entropy_map is required when use_entropy_reward=True.")
        if use_entropy_reward and feature_level and map_item_feat is None:
            raise ValueError("map_item_feat is required for feature-level entropy reward.")
        if lambda_entropy < 0:
            raise ValueError("lambda_entropy must be non-negative.")
        if lambda_variance < 0:
            raise ValueError("lambda_variance must be non-negative.")

        path = Path(predicted_mat_path)
        if not path.exists():
            raise FileNotFoundError(f"Predicted reward matrix does not exist: {path}")
        with path.open("rb") as file_obj:
            predicted_mat = pickle.load(file_obj)
        predicted_tensor = torch.as_tensor(predicted_mat, dtype=torch.float32, device=device)
        if predicted_tensor.ndim != 2:
            raise ValueError("predicted_mat must be a 2D matrix.")

        self.device = device
        self.predicted_mat = predicted_tensor
        self.raw_user_to_index = dict(raw_user_to_index)
        self.reward_shift = bool(reward_shift)
        self.predicted_min = torch.min(predicted_tensor)
        self.use_entropy_reward = bool(use_entropy_reward)
        self.entropy_map = dict(entropy_map or {})
        self.entropy_window = tuple(int(window) for window in entropy_window)
        self.lambda_entropy = float(lambda_entropy)
        self.entropy_min = float(entropy_min)
        self.feature_level = bool(feature_level)
        self.map_item_feat = dict(map_item_feat or {})
        self.is_sorted = bool(is_sorted)
        self.use_uncertainty_penalty = bool(use_uncertainty_penalty)
        self.lambda_variance = float(lambda_variance)

        if internal_to_raw_item_ids is None:
            internal_to_raw_item_ids = list(range(int(predicted_tensor.shape[1])))
        if len(internal_to_raw_item_ids) != int(predicted_tensor.shape[1]):
            raise ValueError("internal_to_raw_item_ids length must match item dimension.")
        self.internal_to_raw_item_ids = torch.as_tensor(
            list(internal_to_raw_item_ids),
            dtype=torch.long,
            device=device,
        )

        self.maxvar_mat = None
        maxvar_max = torch.zeros((), dtype=torch.float32, device=device)
        if self.use_uncertainty_penalty:
            if not maxvar_mat_path:
                raise ValueError("maxvar_mat_path is required when use_uncertainty_penalty=True.")
            maxvar_path = Path(maxvar_mat_path)
            if not maxvar_path.exists():
                raise FileNotFoundError(f"Max variance matrix does not exist: {maxvar_path}")
            with maxvar_path.open("rb") as file_obj:
                maxvar_mat = pickle.load(file_obj)
            maxvar_tensor = torch.as_tensor(maxvar_mat, dtype=torch.float32, device=device)
            if maxvar_tensor.shape != predicted_tensor.shape:
                raise ValueError("maxvar_mat shape must match predicted_mat shape.")
            self.maxvar_mat = maxvar_tensor
            maxvar_max = torch.max(maxvar_tensor)

        if self.reward_shift:
            self.min_reward = (
                self.predicted_min
                + self.lambda_entropy * self.entropy_min
                - self.lambda_variance * maxvar_max
            )
        else:
            self.min_reward = torch.zeros((), dtype=torch.float32, device=device)
        LOGGER.info(
            "DORLRewardModel 加载完成：shape=%s, shift=%s, entropy=%s, uncertainty=%s",
            tuple(predicted_tensor.shape),
            self.reward_shift,
            self.use_entropy_reward,
            self.use_uncertainty_penalty,
        )

    @property
    def num_items(self) -> int:
        """返回 reward 矩阵中的 item 数量。

        Returns:
            int: item 数量。
        """

        return int(self.predicted_mat.shape[1])

    def user_indices(self, raw_user_ids: torch.Tensor) -> torch.Tensor:
        """将原始 user id 转换为 KuaiEnv 内部 user index。

        Args:
            raw_user_ids (torch.Tensor): 原始 user id，形状为 `(B,)` 或 `(B, 1)`。

        Returns:
            torch.Tensor: 内部 user index，形状为 `(B,)`。

        Raises:
            KeyError: 当出现未知 user id 时抛出。
        """

        flat_user_ids = raw_user_ids.detach().cpu().view(-1).tolist()
        indices = []
        for raw_user_id in flat_user_ids:
            raw_user_id = int(raw_user_id)
            if raw_user_id not in self.raw_user_to_index:
                raise KeyError(f"Unknown raw user_id for KuaiEnv: {raw_user_id}")
            indices.append(self.raw_user_to_index[raw_user_id])
        return torch.as_tensor(indices, dtype=torch.long, device=self.device)

    def reward(
        self,
        raw_user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        history_item_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算一批用户-item 对的 DORL predicted reward。

        Args:
            raw_user_ids (torch.Tensor): 原始 user id，形状为 `(B,)` 或 `(B, 1)`。
            item_ids (torch.Tensor): KuaiEnv 内部 item id，形状为 `(B,)`。
            history_item_ids (Optional[torch.Tensor]): 包含当前动作在内的内部 item
                历史，形状为 `(B, H)`；用于 entropy reward。

        Returns:
            torch.Tensor: reward 向量，形状为 `(B,)`。

        Raises:
            ValueError: 当 item id 维度非法时抛出。
        """

        return self.reward_components(raw_user_ids, item_ids, history_item_ids)["reward"]

    def reward_components(
        self,
        raw_user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        history_item_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算 shaped reward 及其组成项。

        Args:
            raw_user_ids (torch.Tensor): 原始 user id，形状为 `(B,)` 或 `(B, 1)`。
            item_ids (torch.Tensor): 内部 item id，形状为 `(B,)`。
            history_item_ids (Optional[torch.Tensor]): 包含当前动作在内的内部 item
                历史，形状为 `(B, H)`。

        Returns:
            Dict[str, torch.Tensor]: 包含 `reward`、`pred_reward`、`entropy` 和
            `uncertainty` 的张量字典。

        Raises:
            ValueError: 当 batch 维度或 item id 非法时抛出。
        """

        item_ids = item_ids.to(device=self.device, dtype=torch.long).view(-1)
        if item_ids.numel() != raw_user_ids.view(-1).numel():
            raise ValueError("raw_user_ids and item_ids must have the same batch size.")
        if torch.any(item_ids < 0) or torch.any(item_ids >= self.num_items):
            raise ValueError("item_ids contain out-of-range values.")
        user_indices = self.user_indices(raw_user_ids)
        pred_rewards = self.predicted_mat[user_indices, item_ids]
        entropy_values = self._compute_entropy_values(history_item_ids, item_ids)
        if self.maxvar_mat is None:
            uncertainty_values = torch.zeros_like(pred_rewards)
        else:
            uncertainty_values = self.maxvar_mat[user_indices, item_ids]

        rewards = (
            pred_rewards
            + self.lambda_entropy * entropy_values
            - self.lambda_variance * uncertainty_values
        )
        if self.reward_shift:
            rewards = rewards - self.min_reward
        rewards = torch.clamp(rewards, min=0.0)
        return {
            "reward": rewards,
            "pred_reward": pred_rewards,
            "entropy": entropy_values,
            "uncertainty": uncertainty_values,
        }

    def _compute_entropy_values(
        self,
        history_item_ids: Optional[torch.Tensor],
        current_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """批量计算 DORL entropy bonus。

        Args:
            history_item_ids (Optional[torch.Tensor]): 包含当前动作在内的内部 item
                历史，形状为 `(B, H)`。
            current_item_ids (torch.Tensor): 当前内部 item id，形状为 `(B,)`。

        Returns:
            torch.Tensor: entropy bonus，形状为 `(B,)`。
        """

        if not self.use_entropy_reward:
            return torch.zeros_like(current_item_ids, dtype=torch.float32, device=self.device)
        if history_item_ids is None:
            raise ValueError("history_item_ids is required when entropy reward is enabled.")

        history_np = history_item_ids.detach().cpu().numpy()
        entropy_values = []
        for history_row, current_item_id in zip(history_np, current_item_ids.detach().cpu().tolist()):
            valid_internal_items = [int(item) for item in history_row if int(item) >= 0]
            if not valid_internal_items or valid_internal_items[-1] != int(current_item_id):
                valid_internal_items.append(int(current_item_id))
            raw_history = [
                int(self.internal_to_raw_item_ids[item].detach().cpu().item())
                for item in valid_internal_items
            ]
            entropy = compute_step_entropy(
                history_with_current=raw_history,
                entropy_dict=self.entropy_map,
                entropy_window=self.entropy_window,
                feature_level=self.feature_level,
                map_item_feat=self.map_item_feat,
                is_sorted=self.is_sorted,
            )
            entropy_values.append(entropy)
        return torch.as_tensor(entropy_values, dtype=torch.float32, device=self.device)
