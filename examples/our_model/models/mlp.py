"""DORL-MAC 使用的通用 MLP 网络。"""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn

DEFAULT_LAYER_NORM_EPS = 1e-5
"""LayerNorm 默认数值稳定项。"""


class MLP(nn.Module):
    """可复用的多层感知机。

    该模块用于 actor、critic、value 等网络，保持 PyTorch 实现简洁一致。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        layer_norm: bool = False,
        activate_final: bool = False,
    ) -> None:
        """初始化 MLP。

        Args:
            input_dim (int): 输入特征维度，必须大于 0。
            hidden_dims (Iterable[int]): 隐藏层维度列表。
            output_dim (int): 输出维度，必须大于 0。
            layer_norm (bool): 是否在隐藏层激活后使用 LayerNorm。
            activate_final (bool): 是否对最后一层也添加 GELU 激活。

        Raises:
            ValueError: 当输入或输出维度非法时抛出。
        """

        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        dims: List[int] = [int(input_dim)] + [int(dim) for dim in hidden_dims] + [int(output_dim)]
        if any(dim <= 0 for dim in dims):
            raise ValueError(f"All MLP dimensions must be positive, got {dims}.")

        layers: List[nn.Module] = []
        for layer_index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[layer_index], dims[layer_index + 1]))
            is_final = layer_index == len(dims) - 2
            if not is_final or activate_final:
                layers.append(nn.GELU())
                if layer_norm and not is_final:
                    layers.append(nn.LayerNorm(dims[layer_index + 1], eps=DEFAULT_LAYER_NORM_EPS))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """执行前向传播。

        Args:
            inputs (torch.Tensor): 输入张量，最后一维必须等于初始化时的 `input_dim`。

        Returns:
            torch.Tensor: 网络输出张量。
        """

        return self.net(inputs)

