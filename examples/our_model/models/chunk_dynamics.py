"""MAC 风格的直接多步动力学，不读取块内反馈或历史缓存。"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn

from examples.our_model.models.mlp import MLP

DEFAULT_DYNAMICS_HIDDEN_DIMS = (256, 256, 256, 256)
"""直接动力学默认使用四层带 LayerNorm 的 MLP。"""


class ChunkDynamics(nn.Module):
    """预测状态与动作块对应的块末状态。

    `forward` 仅消费当前状态、动作 embedding 及有效前缀掩码。
    掩码用于表示提前退出时实际执行的动作前缀，不包含未来反馈。
    全零前缀返回原状态，保证已经终止的样本不再演化。
    """

    def __init__(
        self, state_dim: int, action_dim: int, chunk_size: int,
        hidden_dims: Iterable[int] = DEFAULT_DYNAMICS_HIDDEN_DIMS,
    ) -> None:
        """初始化直接预测网络。

        Args:
            state_dim (int): 正数状态维度。
            action_dim (int): 正数单步动作维度。
            chunk_size (int): 正数最大块长。
            hidden_dims (Iterable[int]): MLP 隐藏层宽度。

        Raises:
            ValueError: 任一维度非正时抛出。
        """
        super().__init__()
        if min(state_dim, action_dim, chunk_size) <= 0:
            raise ValueError("Dynamics dimensions must be positive.")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.net = MLP(
            self.state_dim + self.chunk_size * (self.action_dim + 1),
            hidden_dims, self.state_dim, layer_norm=True,
        )

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """直接计算有效动作前缀之后的状态，无块内状态递归。

        Args:
            states (torch.Tensor): `(B, state_dim)` 当前状态。
            actions (torch.Tensor): `(B, K, action_dim)` 动作序列。
            valid (Optional[torch.Tensor]): `(B, K)` 连续有效前缀；省略为全块。

        Returns:
            torch.Tensor: `(B, state_dim)` 预测状态。

        Raises:
            ValueError: 形状错误或有效位不是连续前缀时抛出。

        Example:
            >>> model = ChunkDynamics(4, 2, 3, (8,))
            >>> model(torch.zeros(1, 4), torch.zeros(1, 3, 2)).shape
            torch.Size([1, 4])
        """
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError("Invalid dynamics state shape.")
        expected = (len(states), self.chunk_size, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(f"Expected chunk shape {expected}, got {tuple(actions.shape)}.")
        if valid is None:
            valid = torch.ones(actions.shape[:2], device=states.device, dtype=torch.bool)
        if valid.shape != actions.shape[:2]:
            raise ValueError("Invalid chunk validity shape.")
        valid = valid.bool()
        if torch.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError("Chunk validity must be a contiguous prefix.")
        # 未执行动作不参与预测，避免 padding embedding 污染模型输入。
        masked_actions = torch.where(valid.unsqueeze(-1), actions, torch.zeros_like(actions))
        inputs = torch.cat([states, masked_actions.flatten(1), valid.to(states.dtype)], dim=-1)
        predicted = self.net(inputs)
        return torch.where(valid.any(dim=1, keepdim=True), predicted, states)
