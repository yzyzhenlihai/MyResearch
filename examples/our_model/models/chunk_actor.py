"""Discrete action chunk actor 网络（Categorical BC 版）。"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from examples.our_model.models.mlp import MLP


class CategoricalChunkActor(nn.Module):
    """离散 chunk actor：给定状态输出 chunk 内每步 item id 的 categorical 分布。

    该 actor 用于替换原连续 flow BC actor。它把 chunk 内 `chunk_size` 步的 policy
    近似为「给定当前 state、每步在全体 item 上条件独立的 categorical 分布」，
    从而彻底避开"连续 action → 相似度映射到离散"的坍缩问题。BC 阶段用 chunk 内
    真实 item id 做 K 步 cross entropy 监督；rejection sampling 阶段直接从每步
    categorical 分布采样 item id，再通过 item embedding 表查表得到 chunk 向量。

    Attributes:
        state_dim (int): 状态维度。
        chunk_size (int): action chunk 步数 `K`。
        num_items (int): 全体离散 item 数量 `N`。
    """

    def __init__(
        self,
        state_dim: int,
        chunk_size: int,
        num_items: int,
        hidden_dims: Iterable[int],
        layer_norm: bool = True,
    ) -> None:
        """初始化 categorical chunk actor。

        Args:
            state_dim (int): 状态维度。
            chunk_size (int): action chunk 步数 `K`。
            num_items (int): 全体离散 item 数量 `N`。
            hidden_dims (Iterable[int]): 隐藏层维度。
            layer_norm (bool): 是否使用 LayerNorm。

        Raises:
            ValueError: 当维度参数非正时抛出。
        """

        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if num_items <= 0:
            raise ValueError("num_items must be positive.")
        self.state_dim = int(state_dim)
        self.chunk_size = int(chunk_size)
        self.num_items = int(num_items)
        self.net = MLP(
            input_dim=self.state_dim,
            hidden_dims=hidden_dims,
            output_dim=self.chunk_size * self.num_items,
            layer_norm=layer_norm,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """预测 chunk 内每步的 item logits。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。

        Returns:
            torch.Tensor: chunk logits，形状为 `(B, chunk_size, num_items)`。

        Raises:
            ValueError: 当输入维度不匹配时抛出。
        """

        if states.ndim != 2:
            raise ValueError("states must be a 2D tensor.")
        if int(states.shape[1]) != self.state_dim:
            raise ValueError(
                f"state dim mismatch: expected {self.state_dim}, got {states.shape[1]}."
            )
        raw_logits = self.net(states)
        return raw_logits.view(-1, self.chunk_size, self.num_items)

    def log_prob(self, states: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """计算 chunk 内每步 item id 的 log-probability。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。
            item_ids (torch.Tensor): chunk 内真实 item id，形状为 `(B, chunk_size)`。

        Returns:
            torch.Tensor: 每步 log-prob，形状为 `(B, chunk_size)`。

        Raises:
            ValueError: 当形状不匹配时抛出。
        """

        if item_ids.ndim != 2 or int(item_ids.shape[1]) != self.chunk_size:
            raise ValueError(
                f"item_ids shape mismatch: expected (B, {self.chunk_size}), got {tuple(item_ids.shape)}."
            )
        logits = self.forward(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        item_ids_long = item_ids.to(dtype=torch.long, device=logits.device)
        gathered = log_probs.gather(-1, item_ids_long.unsqueeze(-1)).squeeze(-1)
        return gathered

    def cross_entropy_loss(
        self,
        states: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """chunk 内 K 步 cross entropy BC 损失，返回逐样本平均。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。
            item_ids (torch.Tensor): chunk 内真实 item id，形状为 `(B, chunk_size)`。

        Returns:
            torch.Tensor: 标量 loss（对 batch 与 K 步取平均）。
        """

        log_probs = self.log_prob(states, item_ids)
        return -log_probs.mean()

    @torch.no_grad()
    def sample_item_ids(
        self,
        states: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """从 chunk 每步的 categorical 分布采样 item id。

        每一步在全体 item 上做独立 multinomial 采样，允许同一 chunk 内多个 step
        采到相同 item——该情况由 rollout 阶段的 leave/repeat 规则统一处理。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。
            num_samples (int): 每个状态采样的候选 chunk 数量。

        Returns:
            torch.Tensor: 采样得到的 item id，形状为 `(num_samples, B, chunk_size)`。

        Raises:
            ValueError: 当 `num_samples` 非正时抛出。
        """

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        logits = self.forward(states)  # (B, K, N)
        probs = torch.softmax(logits, dim=-1)
        batch_size = int(probs.shape[0])
        flat_probs = probs.reshape(batch_size * self.chunk_size, self.num_items)
        # multinomial 支持沿最后一维一次性抽多份样本，形状 (B*K, num_samples)。
        sampled_flat = torch.multinomial(
            flat_probs,
            num_samples=num_samples,
            replacement=True,
        )
        # (B, K, num_samples) → 转成 (num_samples, B, K)
        sampled = sampled_flat.view(batch_size, self.chunk_size, num_samples)
        sampled = sampled.permute(2, 0, 1).contiguous()
        return sampled
