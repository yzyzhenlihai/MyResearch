"""基于 StateTrackerAvg 语义的显式 chunk dynamics。"""

from __future__ import annotations

import torch
from torch import nn


class StateTrackerDynamics(nn.Module):
    """用历史向量追加 action chunk 后重算推荐状态。

    基础状态沿用 `avg_direct`：状态向量由最近 `window_size` 个
    `(item_embedding, reward)` 或 reset dummy 向量求平均得到。为满足与 Q/V
    一起训练的需求，模块额外提供一个零初始化 residual 网络；训练初始时
    完全等价于原 avg dynamics，后续可通过监督 next-state loss 学习修正项。
    """

    def __init__(
        self,
        window_size: int,
        action_dim: int,
        state_dim: int,
        use_trainable_residual: bool = True,
    ) -> None:
        """初始化显式 dynamics。

        Args:
            window_size (int): StateTrackerAvg 历史窗口长度。
            action_dim (int): action embedding 维度。
            state_dim (int): state 维度，当前应为 `action_dim + 1`。
            use_trainable_residual (bool): 是否启用可学习 residual 修正。

        Raises:
            ValueError: 当维度不合法时抛出。
        """

        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if state_dim != action_dim + 1:
            raise ValueError("avg_direct dynamics expects state_dim == action_dim + 1.")
        self.window_size = int(window_size)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.use_trainable_residual = bool(use_trainable_residual)
        if self.use_trainable_residual:
            self.residual = nn.Sequential(
                nn.Linear(self.state_dim, self.state_dim),
                nn.Tanh(),
                nn.Linear(self.state_dim, self.state_dim),
            )
            # 零初始化最后一层，保证初始行为与 avg_direct 完全一致。
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)
        else:
            self.residual = nn.Identity()

    def next_state(
        self,
        history_vectors: torch.Tensor,
        history_valid: torch.Tensor,
        chunk_actions: torch.Tensor,
        chunk_rewards: torch.Tensor,
        chunk_valid: torch.Tensor,
    ) -> torch.Tensor:
        """根据历史和有效 chunk 前缀计算下一状态。

        Args:
            history_vectors (torch.Tensor): 历史状态行，形状为 `(B, W, state_dim)`。
            history_valid (torch.Tensor): 历史有效标记，形状为 `(B, W)`。
            chunk_actions (torch.Tensor): chunk action embedding，形状为 `(B, K, action_dim)`。
            chunk_rewards (torch.Tensor): chunk 每步 reward，形状为 `(B, K)`。
            chunk_valid (torch.Tensor): chunk 中实际执行的步标记，形状为 `(B, K)`。

        Returns:
            torch.Tensor: 重算后的下一状态，形状为 `(B, state_dim)`。

        Raises:
            ValueError: 当输入 shape 不符合预期时抛出。
        """

        self._validate_inputs(history_vectors, history_valid, chunk_actions, chunk_rewards, chunk_valid)
        device = history_vectors.device
        batch_size = int(history_vectors.shape[0])
        window_size = int(self.window_size)
        chunk_vectors = torch.cat([chunk_actions, chunk_rewards.unsqueeze(-1)], dim=-1)
        history_valid_bool = history_valid.bool()
        chunk_valid_bool = chunk_valid.bool()

        # 拼接完整历史 + 本 chunk 的向量与有效标记。
        combined_vectors = torch.cat([history_vectors, chunk_vectors], dim=1)  # (B, W+K, D)
        combined_valid = torch.cat(
            [history_valid_bool, chunk_valid_bool], dim=1
        ).to(dtype=combined_vectors.dtype)  # (B, W+K)
        # 只保留每个位置的向量并把无效位置置零，便于后续 masked mean。
        masked_vectors = combined_vectors * combined_valid.unsqueeze(-1)

        # 为每个 batch 计算「保留最近 window_size 个有效位置」对应的 mask。
        # 方法：从右往左累计有效位置计数（reverse cumsum），值 <= window_size 且原本有效的位置即为"最近的 window_size 个"。
        valid_int = combined_valid.to(torch.int64)
        # reverse cumsum: 每个位置右侧（含自身）的有效数量。
        reverse_cum = torch.flip(torch.cumsum(torch.flip(valid_int, dims=[1]), dim=1), dims=[1])
        recent_mask = (reverse_cum <= window_size) & (valid_int > 0)  # (B, W+K)
        recent_mask_f = recent_mask.to(dtype=combined_vectors.dtype)

        denom = recent_mask_f.sum(dim=1, keepdim=True)  # (B, 1)
        if (denom == 0).any():
            raise ValueError("StateTrackerDynamics received no valid history or chunk rows.")
        numer = (masked_vectors * recent_mask_f.unsqueeze(-1)).sum(dim=1)  # (B, D)
        avg_states = numer / denom
        avg_states = avg_states.to(device=device)
        if not self.use_trainable_residual:
            return avg_states
        return avg_states + self.residual(avg_states)

    def _validate_inputs(
        self,
        history_vectors: torch.Tensor,
        history_valid: torch.Tensor,
        chunk_actions: torch.Tensor,
        chunk_rewards: torch.Tensor,
        chunk_valid: torch.Tensor,
    ) -> None:
        """校验 dynamics 输入张量 shape。

        Args:
            history_vectors (torch.Tensor): 历史向量。
            history_valid (torch.Tensor): 历史有效标记。
            chunk_actions (torch.Tensor): chunk action。
            chunk_rewards (torch.Tensor): chunk reward。
            chunk_valid (torch.Tensor): chunk 有效标记。

        Returns:
            None.

        Raises:
            ValueError: 当 shape 不匹配时抛出。
        """

        if history_vectors.ndim != 3 or history_vectors.shape[-1] != self.state_dim:
            raise ValueError("history_vectors shape is invalid.")
        if history_valid.shape != history_vectors.shape[:2]:
            raise ValueError("history_valid shape mismatch.")
        if chunk_actions.ndim != 3 or chunk_actions.shape[-1] != self.action_dim:
            raise ValueError("chunk_actions shape is invalid.")
        if chunk_rewards.shape != chunk_actions.shape[:2]:
            raise ValueError("chunk_rewards shape mismatch.")
        if chunk_valid.shape != chunk_actions.shape[:2]:
            raise ValueError("chunk_valid shape mismatch.")
