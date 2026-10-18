"""Chunk-level value 与 critic 网络。"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from examples.our_model.models.mlp import MLP

CRITIC_BACKBONE_MLP = "mlp"
"""原 MAC 使用的拼接式 MLP Critic。"""

CRITIC_BACKBONE_TRANSFORMER = "transformer"
"""使用状态 token 与 chunk 动作 token 的 Transformer Critic。"""

CRITIC_BACKBONE_ACSAC_TRANSFORMER = "acsac_transformer"
"""ACSAC 风格的因果前缀 Transformer Q。"""

CRITIC_BACKBONE_MULTI_PREFIX_TRANSFORMER = "multi_prefix_transformer"
"""R/T/D/C 四头 ACSAC 因果前缀 Transformer Q。"""

SUPPORTED_CRITIC_BACKBONES = {
    CRITIC_BACKBONE_MLP,
    CRITIC_BACKBONE_TRANSFORMER,
    CRITIC_BACKBONE_ACSAC_TRANSFORMER,
    CRITIC_BACKBONE_MULTI_PREFIX_TRANSFORMER,
}
"""ordinary 架构对照与独立 ACSAC 分支支持的 Critic 类型。"""

DEFAULT_CRITIC_TRANSFORMER_HIDDEN_DIM = 72
"""Transformer Critic 默认 token 维度，使参数量与 `(256,256)` MLP 接近。"""

DEFAULT_CRITIC_TRANSFORMER_LAYERS = 2
"""Transformer Critic 默认编码层数。"""

DEFAULT_CRITIC_TRANSFORMER_HEADS = 4
"""Transformer Critic 默认注意力头数。"""

DEFAULT_CRITIC_TRANSFORMER_DROPOUT = 0.0
"""严格架构对照默认关闭 dropout，与原 MLP Critic 保持一致。"""


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


class TransformerChunkCritic(nn.Module):
    """用 Transformer 编码完整 action chunk 的标量 Critic。

    网络把状态映射为一个 state token，把 K 个 item embedding 分别映射为动作
    token，并使用双向自注意力聚合完整 chunk。输出仍是与原 MAC 完全相同的
    `Q(s, a_{t:t+K-1})` 标量，因此只替换网络结构，不改变 Q/V 训练目标。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = DEFAULT_CRITIC_TRANSFORMER_HIDDEN_DIM,
        num_layers: int = DEFAULT_CRITIC_TRANSFORMER_LAYERS,
        num_heads: int = DEFAULT_CRITIC_TRANSFORMER_HEADS,
        dropout: float = DEFAULT_CRITIC_TRANSFORMER_DROPOUT,
    ) -> None:
        """初始化 Transformer Critic。

        Args:
            state_dim (int): 状态向量维度，必须为正数。
            action_dim (int): 单个 item embedding 维度，必须为正数。
            chunk_size (int): action chunk 长度 K，必须为正数。
            hidden_dim (int): token 隐藏维度，必须能被注意力头数整除。
            num_layers (int): TransformerEncoder 层数，必须为正数。
            num_heads (int): 多头注意力头数，必须为正数。
            dropout (float): dropout 概率，取值范围 `[0, 1)`。

        Raises:
            ValueError: 网络维度或 dropout 配置不合法时抛出。
        """

        super().__init__()
        if min(state_dim, action_dim, chunk_size, hidden_dim, num_layers, num_heads) <= 0:
            raise ValueError("Transformer critic dimensions must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("Transformer critic hidden_dim must be divisible by num_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Transformer critic dropout must lie in [0, 1).")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.chunk_action_dim = self.action_dim * self.chunk_size
        self.state_projection = nn.Linear(self.state_dim, hidden_dim)
        self.action_projection = nn.Linear(self.action_dim, hidden_dim)
        self.position_embedding = nn.Embedding(self.chunk_size + 1, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, states: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        """计算完整 chunk 的标量 Q 值。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。
            chunks (torch.Tensor): 展平的 chunk embedding，形状为
                `(B, chunk_size * action_dim)`。

        Returns:
            torch.Tensor: 标量 Q 值，形状为 `(B,)`。

        Raises:
            ValueError: 输入 batch 或特征维度与初始化配置不一致时抛出。
        """

        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(
                f"states must have shape (B, {self.state_dim}), got {tuple(states.shape)}."
            )
        if chunks.ndim != 2 or chunks.shape[1] != self.chunk_action_dim:
            raise ValueError(
                "chunks must have shape "
                f"(B, {self.chunk_action_dim}), got {tuple(chunks.shape)}."
            )
        if states.shape[0] != chunks.shape[0]:
            raise ValueError("states and chunks must have equal batch sizes.")
        batch_size = states.shape[0]
        action_tokens = self.action_projection(
            chunks.reshape(batch_size, self.chunk_size, self.action_dim)
        )
        state_token = self.state_projection(states).unsqueeze(1)
        tokens = torch.cat((state_token, action_tokens), dim=1)
        positions = self.position_embedding.weight[: self.chunk_size + 1]
        encoded = self.encoder(tokens + positions.unsqueeze(0))
        return self.output(encoded[:, 0]).squeeze(-1)


class CausalTransformerPrefixCritic(nn.Module):
    """对一个候选 chunk 的所有动作前缀并行输出 Q 值。

    输入 token 顺序为 ``[state, a_1, ..., a_K]``。编码器使用严格因果
    attention mask，因此动作位置 ``h`` 的输出只能依赖状态和
    ``a_1...a_h``，对应 ACSAC 的 ``Q(s, a_{1:h})``。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = DEFAULT_CRITIC_TRANSFORMER_HIDDEN_DIM,
        num_layers: int = DEFAULT_CRITIC_TRANSFORMER_LAYERS,
        num_heads: int = DEFAULT_CRITIC_TRANSFORMER_HEADS,
        dropout: float = DEFAULT_CRITIC_TRANSFORMER_DROPOUT,
    ) -> None:
        """初始化因果前缀 Q 网络。

        Args:
            state_dim (int): 状态向量维度。
            action_dim (int): 单步动作 embedding 维度。
            chunk_size (int): 最大候选长度 K。
            hidden_dim (int): Transformer token 维度。
            num_layers (int): 编码层数。
            num_heads (int): 注意力头数。
            dropout (float): dropout 概率。

        Raises:
            ValueError: 网络维度或 dropout 配置不合法时抛出。
        """

        super().__init__()
        if min(state_dim, action_dim, chunk_size, hidden_dim, num_layers, num_heads) <= 0:
            raise ValueError("Causal transformer critic dimensions must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError("Causal transformer hidden_dim must be divisible by num_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Causal transformer dropout must lie in [0, 1).")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.chunk_action_dim = self.action_dim * self.chunk_size
        self.state_projection = nn.Linear(self.state_dim, hidden_dim)
        self.action_projection = nn.Linear(self.action_dim, hidden_dim)
        self.position_embedding = nn.Embedding(self.chunk_size + 1, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, states: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        """计算所有前缀 Q 值。

        Args:
            states (torch.Tensor): 状态，形状 ``(B, state_dim)``。
            chunks (torch.Tensor): 展平或未展平的动作块，形状
                ``(B, K*action_dim)`` 或 ``(B, K, action_dim)``。

        Returns:
            torch.Tensor: 前缀价值，形状 ``(B, K)``，第 h-1 列对应长度 h。

        Raises:
            ValueError: 输入形状与网络配置不一致时抛出。
        """

        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(
                f"states must have shape (B, {self.state_dim}), got {tuple(states.shape)}."
            )
        if chunks.ndim == 2 and chunks.shape[1] == self.chunk_action_dim:
            action_chunks = chunks.reshape(-1, self.chunk_size, self.action_dim)
        elif chunks.ndim == 3 and tuple(chunks.shape[1:]) == (
            self.chunk_size,
            self.action_dim,
        ):
            action_chunks = chunks
        else:
            raise ValueError(
                "chunks must have shape "
                f"(B, {self.chunk_action_dim}) or "
                f"(B, {self.chunk_size}, {self.action_dim}), got {tuple(chunks.shape)}."
            )
        if states.shape[0] != action_chunks.shape[0]:
            raise ValueError("states and chunks must have equal batch sizes.")
        state_token = self.state_projection(states).unsqueeze(1)
        action_tokens = self.action_projection(action_chunks)
        tokens = torch.cat((state_token, action_tokens), dim=1)
        positions = self.position_embedding.weight[: self.chunk_size + 1]
        sequence_length = self.chunk_size + 1
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=states.device,
            ),
            diagonal=1,
        )
        encoded = self.encoder(tokens + positions.unsqueeze(0), mask=causal_mask)
        return self.output(encoded[:, 1:]).squeeze(-1)


class MultiHeadCausalTransformerPrefixCritic(CausalTransformerPrefixCritic):
    """共享因果 Transformer 主干并输出 R/T/D/C 四个前缀 Q。

    网络输入与 :class:`CausalTransformerPrefixCritic` 完全一致。每个动作位置
    共享同一隐藏表示，最后通过四维线性输出层分别预测反馈、展示长度、MCD
    改善和规划成本的长期价值，输出形状为 ``(B,K,4)``。
    """

    num_components = 4
    """固定的 R/T/D/C 价值分量数。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = DEFAULT_CRITIC_TRANSFORMER_HIDDEN_DIM,
        num_layers: int = DEFAULT_CRITIC_TRANSFORMER_LAYERS,
        num_heads: int = DEFAULT_CRITIC_TRANSFORMER_HEADS,
        dropout: float = DEFAULT_CRITIC_TRANSFORMER_DROPOUT,
    ) -> None:
        """初始化四头因果前缀 Q 网络。

        Args:
            state_dim (int): 状态向量维度。
            action_dim (int): 单步动作 embedding 维度。
            chunk_size (int): 最大候选长度 K。
            hidden_dim (int): Transformer token 维度。
            num_layers (int): 编码层数。
            num_heads (int): 注意力头数。
            dropout (float): dropout 概率，范围 ``[0,1)``。

        Raises:
            ValueError: 网络维度或 dropout 配置不合法时由父类抛出。
        """

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.num_components),
        )

    def forward(self, states: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        """计算所有前缀的四分量 Q 值。

        Args:
            states (torch.Tensor): 状态，形状 ``(B,state_dim)``。
            chunks (torch.Tensor): 展平或未展平动作块，形状
                ``(B,K*action_dim)`` 或 ``(B,K,action_dim)``。

        Returns:
            torch.Tensor: 四头前缀价值，形状 ``(B,K,4)``。

        Raises:
            ValueError: 输入形状与父类网络配置不一致时抛出。
        """

        values = super().forward(states, chunks)
        expected_shape = (states.shape[0], self.chunk_size, self.num_components)
        if tuple(values.shape) != expected_shape:
            raise RuntimeError(
                "multi-head prefix critic produced unexpected shape: "
                f"expected {expected_shape}, got {tuple(values.shape)}."
            )
        return values
