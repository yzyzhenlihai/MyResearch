"""DORL user model reward 封装。"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
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
        predicted_mat_normalize: str = "none",
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
            predicted_mat_normalize (str): predicted_mat 归一化模式，取值：
                - `"none"`：保持原始 DeepFM 输出（可能是极小值，与真实 CTR 不同尺度）；
                - `"global_minmax"`：整表按全局 min/max 线性缩放到 `[0, 1]`；
                - `"per_user_max"`：每行除以该用户的最大值，把 per-user top-1 拉到 1.0；
                - `"per_user_minmax"`：每行做 min-max 归一化到 `[0, 1]`；
                - `"sigmoid"`：套 sigmoid，把原始 logits 映射到 `(0, 1)` 概率域。

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

        # 可选：对 DeepFM 原始 logits 做归一化。KuaiRec 上 raw predicted_mat 均值 ~ 1e-4，
        # 与真实环境 CTR ~ 0.5 差 3~4 个量级，会导致 pred_reward 信号被 entropy_bonus
        # 完全淹没（λ_entropy × entropy ≫ pred_reward）。归一化能让二者进入同一尺度，
        # 使 Q-learning 真正学到"用户偏好"信号。
        normalize_mode = str(predicted_mat_normalize).lower()
        supported = {"none", "global_minmax", "per_user_max", "per_user_minmax", "sigmoid"}
        if normalize_mode not in supported:
            raise ValueError(
                f"predicted_mat_normalize must be one of {supported}, got {predicted_mat_normalize!r}."
            )
        eps = 1e-8
        if normalize_mode == "global_minmax":
            g_min = torch.min(predicted_tensor)
            g_max = torch.max(predicted_tensor)
            predicted_tensor = (predicted_tensor - g_min) / (g_max - g_min + eps)
        elif normalize_mode == "per_user_max":
            row_abs_max = torch.clamp(torch.abs(predicted_tensor).max(dim=1, keepdim=True).values, min=eps)
            predicted_tensor = predicted_tensor / row_abs_max
        elif normalize_mode == "per_user_minmax":
            row_min = predicted_tensor.min(dim=1, keepdim=True).values
            row_max = predicted_tensor.max(dim=1, keepdim=True).values
            predicted_tensor = (predicted_tensor - row_min) / (row_max - row_min + eps)
        elif normalize_mode == "sigmoid":
            predicted_tensor = torch.sigmoid(predicted_tensor)
        # normalize_mode == "none" 保持原值，向后兼容旧实验。

        self.device = device
        self.predicted_mat = predicted_tensor
        self.predicted_mat_normalize = normalize_mode
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
        # 缓存 CPU 侧 int 数组，用于 entropy 计算时高频查表，避免逐 item 的 GPU→CPU 同步。
        self._internal_to_raw_cpu = [int(x) for x in internal_to_raw_item_ids]
        # user 索引查表用 numpy 数组，比 dict + 逐元素 tolist 快得多。
        max_raw_user_id = max(self.raw_user_to_index.keys()) if self.raw_user_to_index else -1
        self._user_lookup = np.full(max_raw_user_id + 1, -1, dtype=np.int64)
        for raw_uid, internal in self.raw_user_to_index.items():
            self._user_lookup[int(raw_uid)] = int(internal)
        # entropy 计算的 LRU 缓存（key=raw item 序列元组）。
        self._entropy_cache: Dict[Tuple[int, ...], float] = {}
        self._entropy_cache_max = 200_000

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
            "DORLRewardModel 加载完成：shape=%s, normalize=%s, pred_min=%.6f, pred_max=%.6f, "
            "pred_mean=%.6f, shift=%s, entropy=%s, uncertainty=%s",
            tuple(predicted_tensor.shape),
            self.predicted_mat_normalize,
            float(predicted_tensor.min()),
            float(predicted_tensor.max()),
            float(predicted_tensor.mean()),
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

        flat = raw_user_ids.detach().view(-1).cpu().numpy().astype(np.int64, copy=False)
        # 边界检查（少量向量化操作，成本远低于原来的 Python for + dict lookup）。
        if flat.min(initial=0) < 0 or flat.max(initial=0) >= self._user_lookup.shape[0]:
            raise KeyError(f"Unknown raw user_id for KuaiEnv (out of range): {flat.min()}~{flat.max()}")
        mapped = self._user_lookup[flat]
        if (mapped < 0).any():
            bad = int(flat[np.argmax(mapped < 0)])
            raise KeyError(f"Unknown raw user_id for KuaiEnv: {bad}")
        return torch.from_numpy(mapped).to(device=self.device)

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

        # 一次性把 batch 从 GPU 拷回 CPU；不再逐 item .item()。
        history_np = history_item_ids.detach().cpu().numpy()
        current_np = current_item_ids.detach().cpu().numpy()
        internal_to_raw = self._internal_to_raw_cpu
        entropy_values = [0.0] * len(current_np)
        cache = self._entropy_cache
        cache_max = self._entropy_cache_max
        # 展平 -1 padding：只在最大 window 内取尾部，避免每次遍历完整历史。
        max_win = max((int(w) for w in self.entropy_window if int(w) > 0), default=0)
        for row_idx in range(len(current_np)):
            current_item_id = int(current_np[row_idx])
            history_row = history_np[row_idx]
            # 只需要最后 max_win-1 个有效历史（entropy_window 内的最大长度即上限）。
            if max_win > 1:
                tail = history_row[-(max_win - 1):] if history_row.size > 0 else history_row
                valid_internal_items = [int(x) for x in tail.tolist() if int(x) >= 0]
            else:
                valid_internal_items = []
            if not valid_internal_items or valid_internal_items[-1] != current_item_id:
                valid_internal_items.append(current_item_id)
            raw_history_tail = tuple(internal_to_raw[item] for item in valid_internal_items)

            cached = cache.get(raw_history_tail)
            if cached is not None:
                entropy_values[row_idx] = cached
                continue
            entropy = compute_step_entropy(
                history_with_current=raw_history_tail,
                entropy_dict=self.entropy_map,
                entropy_window=self.entropy_window,
                feature_level=self.feature_level,
                map_item_feat=self.map_item_feat,
                is_sorted=self.is_sorted,
            )
            entropy_values[row_idx] = entropy
            if len(cache) < cache_max:
                cache[raw_history_tail] = entropy
        return torch.as_tensor(entropy_values, dtype=torch.float32, device=self.device)
