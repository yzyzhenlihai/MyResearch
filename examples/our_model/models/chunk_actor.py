"""Action chunk actor 网络。"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn

from examples.our_model.models.mlp import MLP


class ChunkFlowActor(nn.Module):
    """Flow matching 速度场 actor。

    输入当前 state、噪声化 action chunk 和时间 `t`，输出从噪声到真实 chunk
    的速度估计。
    """

    def __init__(
        self,
        state_dim: int,
        chunk_action_dim: int,
        hidden_dims: Iterable[int],
        layer_norm: bool = True,
    ) -> None:
        """初始化 flow actor。

        Args:
            state_dim (int): 状态维度。
            chunk_action_dim (int): flatten action chunk 维度。
            hidden_dims (Iterable[int]): 隐藏层维度。
            layer_norm (bool): 是否使用 LayerNorm。
        """

        super().__init__()
        self.state_dim = int(state_dim)
        self.chunk_action_dim = int(chunk_action_dim)
        self.net = MLP(
            input_dim=self.state_dim + self.chunk_action_dim + 1,
            hidden_dims=hidden_dims,
            output_dim=self.chunk_action_dim,
            layer_norm=layer_norm,
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """预测 flow matching 速度。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。
            actions (torch.Tensor): 当前噪声化 chunk，形状为 `(B, chunk_action_dim)`。
            times (torch.Tensor): 时间张量，形状为 `(B, 1)`。

        Returns:
            torch.Tensor: 速度预测，形状为 `(B, chunk_action_dim)`。
        """

        return self.net(torch.cat([states, actions, times], dim=-1))

    @torch.no_grad()
    def sample_flow(self, states: torch.Tensor, noises: torch.Tensor, flow_steps: int) -> torch.Tensor:
        """用 Euler 积分从 flow actor 采样 action chunk。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。
            noises (torch.Tensor): 初始高斯噪声，形状为 `(B, chunk_action_dim)`。
            flow_steps (int): Euler 积分步数，必须大于 0。

        Returns:
            torch.Tensor: 采样得到的 action chunk。

        Raises:
            ValueError: 当 `flow_steps` 非正时抛出。
        """

        if flow_steps <= 0:
            raise ValueError("flow_steps must be positive.")
        actions = noises
        for step_index in range(flow_steps):
            times = torch.full(
                (states.shape[0], 1),
                float(step_index) / float(flow_steps),
                device=states.device,
                dtype=states.dtype,
            )
            velocities = self.forward(states, actions, times)
            actions = actions + velocities / float(flow_steps)
        return actions


class ChunkOneStepActor(nn.Module):
    """one-step action chunk actor。

    该 actor 输入 state 和高斯噪声，直接输出 flatten action chunk。smoke 模式下
    它也作为普通 MLP BC actor 使用。
    """

    def __init__(
        self,
        state_dim: int,
        chunk_action_dim: int,
        hidden_dims: Iterable[int],
        layer_norm: bool = True,
    ) -> None:
        """初始化 one-step actor。

        Args:
            state_dim (int): 状态维度。
            chunk_action_dim (int): flatten action chunk 维度。
            hidden_dims (Iterable[int]): 隐藏层维度。
            layer_norm (bool): 是否使用 LayerNorm。
        """

        super().__init__()
        self.state_dim = int(state_dim)
        self.chunk_action_dim = int(chunk_action_dim)
        self.net = MLP(
            input_dim=self.state_dim + self.chunk_action_dim,
            hidden_dims=hidden_dims,
            output_dim=self.chunk_action_dim,
            layer_norm=layer_norm,
        )

    def forward(self, states: torch.Tensor, noises: Optional[torch.Tensor] = None) -> torch.Tensor:
        """输出 action chunk。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。
            noises (Optional[torch.Tensor]): 高斯噪声，形状为 `(B, chunk_action_dim)`；
                为 `None` 时使用零噪声。

        Returns:
            torch.Tensor: flatten action chunk，形状为 `(B, chunk_action_dim)`。
        """

        if noises is None:
            noises = torch.zeros(
                (states.shape[0], self.chunk_action_dim),
                dtype=states.dtype,
                device=states.device,
            )
        return self.net(torch.cat([states, noises], dim=-1))

