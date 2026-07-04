"""DORL user model reward 封装。"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict

import torch

LOGGER = logging.getLogger(__name__)


class DORLRewardModel:
    """封装 DORL 已训练 user model 的预测 reward 矩阵。

    第一版只使用 `predicted_mat[cur_user, item_id] - min(predicted_mat)`，
    与 `BaseSimulatedEnv` 的 reward shift 语义保持一致。entropy reward 的配置
    入口保留，但不在 MVP 中计算；exposure intervention 会显式报错。
    """

    def __init__(
        self,
        predicted_mat_path: str,
        raw_user_to_index: Dict[int, int],
        device: torch.device,
        reward_shift: bool = True,
        use_exposure_intervention: bool = False,
        use_entropy_reward: bool = False,
    ) -> None:
        """初始化 reward 模型。

        Args:
            predicted_mat_path (str): DeepFM 预测矩阵 pkl 路径。
            raw_user_to_index (Dict[int, int]): 原始 user id 到 KuaiEnv 内部行号的映射。
            device (torch.device): 计算设备。
            reward_shift (bool): 是否减去预测矩阵全局最小值。
            use_exposure_intervention (bool): 是否启用 exposure intervention；MVP 不支持。
            use_entropy_reward (bool): 是否启用 entropy reward；MVP 暂不计算，仅记录警告。

        Raises:
            FileNotFoundError: 当预测矩阵不存在时抛出。
            NotImplementedError: 当开启 exposure intervention 时抛出。
            ValueError: 当预测矩阵维度不合法时抛出。
        """

        if use_exposure_intervention:
            raise NotImplementedError("DORL-MAC MVP does not implement exposure intervention.")
        if use_entropy_reward:
            LOGGER.warning("MVP 暂不计算 entropy reward，将仅使用 shifted predicted reward。")

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
        self.min_reward = torch.min(predicted_tensor)
        LOGGER.info(
            "DORLRewardModel 加载完成：shape=%s, shift=%s",
            tuple(predicted_tensor.shape),
            self.reward_shift,
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

    def reward(self, raw_user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """计算一批用户-item 对的 DORL predicted reward。

        Args:
            raw_user_ids (torch.Tensor): 原始 user id，形状为 `(B,)` 或 `(B, 1)`。
            item_ids (torch.Tensor): KuaiEnv 内部 item id，形状为 `(B,)`。

        Returns:
            torch.Tensor: reward 向量，形状为 `(B,)`。

        Raises:
            ValueError: 当 item id 维度非法时抛出。
        """

        item_ids = item_ids.to(device=self.device, dtype=torch.long).view(-1)
        if item_ids.numel() != raw_user_ids.view(-1).numel():
            raise ValueError("raw_user_ids and item_ids must have the same batch size.")
        if torch.any(item_ids < 0) or torch.any(item_ids >= self.num_items):
            raise ValueError("item_ids contain out-of-range values.")
        user_indices = self.user_indices(raw_user_ids)
        rewards = self.predicted_mat[user_indices, item_ids]
        if self.reward_shift:
            rewards = rewards - self.min_reward
        return torch.clamp(rewards, min=0.0)

