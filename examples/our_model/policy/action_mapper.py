"""action embedding 到离散 item id 的映射器。"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

MASKED_SCORE = -1.0e9
"""被推荐 mask 屏蔽的 item 使用的极小分数。"""


class ActionMapper:
    """将连续 action embedding 映射到推荐环境可执行的 item id。

    映射逻辑与原 `RecPolicy.get_score` 一致：对 action embedding 和 item
    embedding 做归一化点积，再取 top-k。
    """

    def __init__(self, item_embeddings: torch.Tensor, device: torch.device) -> None:
        """初始化映射器。

        Args:
            item_embeddings (torch.Tensor): item embedding 表，形状为 `(num_items, action_dim)`。
            device (torch.device): 计算设备。

        Raises:
            ValueError: 当 item embedding 维度不合法时抛出。
        """

        if item_embeddings.ndim != 2:
            raise ValueError("item_embeddings must be a 2D tensor.")
        self.device = device
        self.item_embeddings = item_embeddings.detach().to(device=device, dtype=torch.float32)
        self.normalized_item_embeddings = F.normalize(self.item_embeddings, dim=-1)
        self.num_items = int(self.item_embeddings.shape[0])
        self.action_dim = int(self.item_embeddings.shape[1])

    def score(self, action_embeddings: torch.Tensor) -> torch.Tensor:
        """计算 action embedding 对所有 item 的相似度分数。

        Args:
            action_embeddings (torch.Tensor): action embedding，形状为 `(B, action_dim)`。

        Returns:
            torch.Tensor: item 分数矩阵，形状为 `(B, num_items)`。

        Raises:
            ValueError: 当 action 维度不匹配时抛出。
        """

        if action_embeddings.ndim != 2:
            raise ValueError("action_embeddings must be a 2D tensor.")
        if int(action_embeddings.shape[1]) != self.action_dim:
            raise ValueError(
                f"action dim mismatch: expected {self.action_dim}, got {action_embeddings.shape[1]}."
            )
        action_embeddings = action_embeddings.to(device=self.device, dtype=torch.float32)
        normalized_actions = F.normalize(action_embeddings, dim=-1)
        return normalized_actions @ self.normalized_item_embeddings.T

    def map_embeddings(
        self,
        action_embeddings: torch.Tensor,
        recommended_mask: Optional[torch.Tensor] = None,
        topk: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把 action embedding 映射为 top-k item id。

        Args:
            action_embeddings (torch.Tensor): action embedding，形状为 `(B, action_dim)`。
            recommended_mask (Optional[torch.Tensor]): 已推荐 item mask，形状为
                `(B, num_items)`；True 表示该 item 不能再推荐。
            topk (int): 返回的候选数量，必须大于 0。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: item id 和对应分数，形状均为 `(B, topk)`。

        Raises:
            ValueError: 当 topk 或 mask 形状非法时抛出。
        """

        if topk <= 0:
            raise ValueError("topk must be positive.")
        scores = self.score(action_embeddings)
        if recommended_mask is not None:
            recommended_mask = recommended_mask.to(device=self.device, dtype=torch.bool)
            if recommended_mask.shape != scores.shape:
                raise ValueError(
                    f"recommended_mask shape mismatch: expected {scores.shape}, "
                    f"got {recommended_mask.shape}."
                )
            # 若某个样本所有 item 都被 mask，则回退为不屏蔽，避免 topk 全为无效值。
            all_masked = recommended_mask.all(dim=1)
            if all_masked.any():
                recommended_mask = recommended_mask.clone()
                recommended_mask[all_masked] = False
            scores = scores.masked_fill(recommended_mask, MASKED_SCORE)
        top_scores, top_items = torch.topk(scores, k=min(topk, self.num_items), dim=1)
        return top_items, top_scores

