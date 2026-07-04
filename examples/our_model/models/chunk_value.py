"""Chunk-level value 与 critic 网络。"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from examples.our_model.models.mlp import MLP


class ChunkValue(nn.Module):
    """状态价值函数 `V(s)`。"""

    def __init__(self, state_dim: int, hidden_dims: Iterable[int], layer_norm: bool = True) -> None:
        """初始化 value 网络。

        Args:
            state_dim (int): 状态维度。
            hidden_dims (Iterable[int]): 隐藏层维度。
            layer_norm (bool): 是否使用 LayerNorm。
        """

        super().__init__()
        self.net = MLP(state_dim, hidden_dims, 1, layer_norm=layer_norm)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """计算状态价值。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。

        Returns:
            torch.Tensor: 标量价值，形状为 `(B,)`。
        """

        return self.net(states).squeeze(-1)


class ChunkCritic(nn.Module):
    """Chunk-level 动作价值函数 `Q(s, a_{t:t+K-1})`。"""

    def __init__(
        self,
        state_dim: int,
        chunk_action_dim: int,
        hidden_dims: Iterable[int],
        layer_norm: bool = True,
    ) -> None:
        """初始化 critic 网络。

        Args:
            state_dim (int): 状态维度。
            chunk_action_dim (int): flatten action chunk 维度。
            hidden_dims (Iterable[int]): 隐藏层维度。
            layer_norm (bool): 是否使用 LayerNorm。
        """

        super().__init__()
        self.net = MLP(state_dim + chunk_action_dim, hidden_dims, 1, layer_norm=layer_norm)

    def forward(self, states: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        """计算 chunk Q 值。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。
            chunks (torch.Tensor): flatten action chunk，形状为 `(B, chunk_action_dim)`。

        Returns:
            torch.Tensor: 标量 Q 值，形状为 `(B,)`。
        """

        return self.net(torch.cat([states, chunks], dim=-1)).squeeze(-1)
