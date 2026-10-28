"""ROLeR 官方实验使用的 Transformer 注意力状态编码器。"""

from __future__ import annotations

import math

import torch
from torch import nn

from src.core.state_tracker.base import StateTracker_Base, extract_axis_1


class SinusoidalPositionalEncoding(nn.Module):
    """为批优先序列添加固定正弦位置编码。

    Args:
        dimension: 输入状态维度。
        max_length: 可支持的最大序列长度。
        dropout_rate: 位置编码后的 dropout 概率。
    """

    def __init__(
        self,
        dimension: int,
        max_length: int,
        dropout_rate: float = 0.1,
    ) -> None:
        """初始化并缓存固定位置编码。

        Args:
            dimension: 输入状态维度，必须为正整数。
            max_length: 支持的最大序列长度，必须为正整数。
            dropout_rate: 位置编码后的 dropout 概率。

        Raises:
            ValueError: 维度或最大长度不是正数时抛出。
        """

        super().__init__()
        if dimension <= 0 or max_length <= 0:
            raise ValueError("dimension 与 max_length 必须为正整数。")
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(max_length, dimension, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * divisor)
        cosine = torch.cos(position * divisor)
        encoding[:, 1::2] = cosine[:, : encoding[:, 1::2].shape[1]]
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """把位置编码加到形状为 ``[B, T, D]`` 的输入上。

        Args:
            sequence: 批优先状态序列。

        Returns:
            与输入同形状的编码序列。
        """

        if sequence.shape[1] > self.encoding.shape[1]:
            raise ValueError("输入序列长度超过初始化时的 max_length。")
        return self.dropout(sequence + self.encoding[:, : sequence.shape[1]])


class StateTracker_Attention(StateTracker_Base):
    """使用 TransformerEncoder 汇聚最近交互状态。

    该实现对应 ROLeR 官方 ``StateTrackerAtt`` 的核心结构：物品 embedding 与
    归一化奖励拼接后加入正弦位置编码，再通过两层 Transformer 编码。它适配
    当前 EasyRL4Rec 的统一 buffer/state-tracker 接口，并显式屏蔽补零位置。

    Args:
        user_columns: 用户特征列。
        action_columns: 动作/物品特征列。
        feedback_columns: 反馈特征列。
        dim_model: 物品 embedding 维度。
        train_max: 训练奖励最大值。
        train_min: 训练奖励最小值。
        test_max: 测试奖励最大值。
        test_min: 测试奖励最小值。
        reward_handle: 奖励与物品表示的组合方式。
        saved_embedding: 预训练或 scratch 用户/物品 embedding。
        device: PyTorch 设备。
        use_userEmbedding: 是否拼接用户 embedding。
        window_size: 历史窗口长度。
        dropout_rate: Transformer dropout 概率。
        num_heads: 注意力头数。
        num_layers: TransformerEncoder 层数。
    """

    def __init__(
        self,
        user_columns,
        action_columns,
        feedback_columns,
        dim_model,
        train_max=None,
        train_min=None,
        test_max=None,
        test_min=None,
        reward_handle=None,
        saved_embedding=None,
        device="cpu",
        use_userEmbedding=False,
        window_size=10,
        dropout_rate=0.1,
        num_heads=1,
        num_layers=2,
    ) -> None:
        """初始化 ROLeR 注意力编码器及其 embedding。

        Args:
            user_columns: 用户特征列定义。
            action_columns: 动作/物品特征列定义。
            feedback_columns: 反馈特征列定义。
            dim_model: 物品 embedding 维度。
            train_max: 训练奖励最大值。
            train_min: 训练奖励最小值。
            test_max: 测试奖励最大值。
            test_min: 测试奖励最小值。
            reward_handle: 奖励与物品 embedding 的组合方式。
            saved_embedding: 用户模型保存的验证集 embedding。
            device: PyTorch 计算设备。
            use_userEmbedding: 是否拼接用户 embedding。
            window_size: 最近交互窗口长度。
            dropout_rate: 注意力层 dropout 概率。
            num_heads: 多头注意力头数。
            num_layers: TransformerEncoder 层数。

        Raises:
            ValueError: 最终状态维度不能被注意力头数整除时抛出。
        """

        super().__init__(
            user_columns=user_columns,
            action_columns=action_columns,
            feedback_columns=feedback_columns,
            dim_model=dim_model,
            train_max=train_max,
            train_min=train_min,
            test_max=test_max,
            test_min=test_min,
            reward_handle=reward_handle,
            saved_embedding=saved_embedding,
            device=device,
            window_size=window_size,
            use_userEmbedding=use_userEmbedding,
        )
        if self.hidden_size % num_heads != 0:
            raise ValueError(
                f"状态维度 {self.hidden_size} 必须能被 num_heads={num_heads} 整除。"
            )
        self.position_encoding = SinusoidalPositionalEncoding(
            dimension=self.hidden_size,
            max_length=window_size,
            dropout_rate=dropout_rate,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=num_heads,
            dim_feedforward=self.hidden_size,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.attention_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.final_dim = self.hidden_size

    def forward(
        self,
        buffer=None,
        indices=None,
        is_obs=None,
        batch=None,
        is_train=True,
        use_batch_in_statetracker=False,
        **kwargs,
    ) -> torch.Tensor:
        """把最近交互序列编码为定长用户状态。

        Args:
            buffer: Tianshou 轨迹缓冲区。
            indices: 待编码轨迹在缓冲区中的下标。
            is_obs: 编码当前观测还是下一观测。
            batch: 收集阶段可直接提供的批数据。
            is_train: 是否使用训练奖励范围做归一化。
            use_batch_in_statetracker: 是否从 ``batch`` 而非缓冲区构造首步。
            **kwargs: 兼容策略调用方的扩展参数。

        Returns:
            形状为 ``[B, final_dim]`` 的状态张量。
        """

        del kwargs
        sequence, mask, lengths = self.convert_to_k_state_embedding(
            buffer=buffer,
            indices=indices,
            is_obs=is_obs,
            batch=batch,
            use_batch_in_statetracker=use_batch_in_statetracker,
            is_train=is_train,
        )
        encoded_input = self.position_encoding(sequence)
        padding_mask = ~mask.squeeze(-1).bool()
        encoded = self.attention_encoder(
            encoded_input,
            src_key_padding_mask=padding_mask,
        )
        final_state = extract_axis_1(encoded, lengths - 1).squeeze(1)
        return final_state

