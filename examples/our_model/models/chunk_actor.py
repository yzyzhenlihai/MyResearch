"""离散 action chunk 的交叉熵与 Flow Matching actor。"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from examples.our_model.models.mlp import MLP

DEFAULT_FLOW_STEPS = 10
"""类别 Flow Matching 默认 Euler 积分步数。"""

DEFAULT_FLOW_TEMPERATURE = 0.05
"""Flow 终点转为 categorical 分布时的默认温度。"""

DEFAULT_FLOW_POSITIVE_WEIGHT = 0.0
"""目标 item 的 Flow 损失权重；零表示自动使用 item 数量。"""

DEFAULT_FLOW_SAMPLE_BATCH_SIZE = 4
"""Flow 推理时并行积分的候选数，用于限制完整类别向量的显存占用。"""

DEFAULT_FLOW_DISTILL_WEIGHT = 1.0
"""一步 actor 蒸馏损失在 Flow BC 总损失中的默认权重。"""


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


class CategoricalFlowChunkActor(nn.Module):
    """在类别顶点空间学习条件 Flow Matching 的离散 chunk actor。

    每个 chunk 位置用长度为 `num_items` 的连续向量表示类别状态。训练时把高斯
    噪声与真实 item 的 one-hot 顶点线性插值，并回归恒定条件速度；推理时从高斯
    噪声经 Euler 积分得到终点 logits，再直接采样 categorical item ID。整个过程
    不经过 item embedding 最近邻映射。

    Attributes:
        state_dim (int): 状态维度。
        chunk_size (int): 动作块长度。
        num_items (int): 推荐目录大小。
        flow_steps (int): 推理积分步数。
        temperature (float): 终点 categorical 采样温度。
        positive_weight (float): 真实类别坐标的损失权重。
    """

    def __init__(
        self,
        state_dim: int,
        chunk_size: int,
        num_items: int,
        hidden_dims: Iterable[int],
        flow_steps: int = DEFAULT_FLOW_STEPS,
        temperature: float = DEFAULT_FLOW_TEMPERATURE,
        positive_weight: float = DEFAULT_FLOW_POSITIVE_WEIGHT,
        sample_batch_size: int = DEFAULT_FLOW_SAMPLE_BATCH_SIZE,
    ) -> None:
        """初始化类别 Flow Matching 速度场。

        Args:
            state_dim (int): 状态向量宽度，必须为正数。
            chunk_size (int): chunk 长度，必须为正数。
            num_items (int): 离散 item 数量，必须为正数。
            hidden_dims (Iterable[int]): 速度场隐藏层宽度。
            flow_steps (int): Euler 积分步数，必须为正数。
            temperature (float): 终点 softmax 温度，必须为正数。
            positive_weight (float): 真实类别损失权重；零表示 `num_items`。
            sample_batch_size (int): 单次并行积分的候选数量，必须为正数。

        Raises:
            ValueError: 维度、积分步数、温度或权重非法。
        """
        super().__init__()
        if min(state_dim, chunk_size, num_items, flow_steps, sample_batch_size) <= 0:
            raise ValueError("Flow dimensions, steps and sample batch size must be positive")
        if temperature <= 0 or positive_weight < 0:
            raise ValueError("Flow temperature must be positive and weight non-negative")
        self.state_dim = int(state_dim)
        self.chunk_size = int(chunk_size)
        self.num_items = int(num_items)
        self.flow_steps = int(flow_steps)
        self.temperature = float(temperature)
        self.positive_weight = float(positive_weight or num_items)
        self.sample_batch_size = int(sample_batch_size)
        self.flow_dim = self.chunk_size * self.num_items
        # 完整类别向量进入速度场，使 item 维之间可交换概率质量并表达多峰分布。
        self.vector_field = MLP(
            input_dim=self.state_dim + self.flow_dim + 1,
            hidden_dims=hidden_dims,
            output_dim=self.flow_dim,
            layer_norm=True,
        )

    def forward(
        self,
        states: torch.Tensor,
        category_vectors: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """预测给定时间与类别向量处的条件速度。

        Args:
            states (torch.Tensor): 状态，形状 `(B, state_dim)`。
            category_vectors (torch.Tensor): 类别连续状态，形状 `(B,K,I)`。
            times (torch.Tensor): 每个样本的时间，形状 `(B,1)`，范围 `[0,1]`。

        Returns:
            torch.Tensor: 速度场输出，形状 `(B,K,I)`。

        Raises:
            ValueError: 输入形状、时间范围或有限性不满足契约。
        """
        batch_size = int(states.shape[0]) if states.ndim == 2 else -1
        if tuple(states.shape) != (batch_size, self.state_dim):
            raise ValueError("Flow states shape mismatch")
        if tuple(category_vectors.shape) != (
            batch_size, self.chunk_size, self.num_items,
        ):
            raise ValueError("Flow category vector shape mismatch")
        if tuple(times.shape) != (batch_size, 1):
            raise ValueError("Flow times must have shape (B,1)")
        if (not torch.isfinite(states).all()
                or not torch.isfinite(category_vectors).all()
                or not torch.isfinite(times).all()
                or torch.any(times < 0) or torch.any(times > 1)):
            raise ValueError("Flow inputs must be finite and times lie in [0,1]")
        inputs = torch.cat((states, category_vectors.flatten(1), times), dim=-1)
        return self.vector_field(inputs).reshape(
            batch_size, self.chunk_size, self.num_items,
        )

    def flow_matching_loss(
        self,
        states: torch.Tensor,
        item_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算高斯到真实类别 one-hot 顶点的条件 Flow Matching 损失。

        Args:
            states (torch.Tensor): 状态，形状 `(B,state_dim)`。
            item_ids (torch.Tensor): 真实 chunk item ID，形状 `(B,K)`。
            valid (torch.Tensor): 实际有效位置，形状 `(B,K)`。

        Returns:
            tuple: 标量加权速度 MSE，以及不参与反向的诊断张量字典。

        Raises:
            ValueError: ID、mask 或 batch 形状非法，或没有有效动作。

        Example:
            `loss, metrics = actor.flow_matching_loss(states, ids, valid)`。
        """
        batch_size = int(states.shape[0])
        if tuple(item_ids.shape) != (batch_size, self.chunk_size):
            raise ValueError("Flow item_ids shape mismatch")
        if tuple(valid.shape) != tuple(item_ids.shape) or valid.dtype != torch.bool:
            raise ValueError("Flow valid mask must be boolean and align with item_ids")
        if not valid.any():
            raise ValueError("Flow training batch contains no valid action")
        if torch.any(item_ids[valid] < 0) or torch.any(item_ids[valid] >= self.num_items):
            raise ValueError("Flow target item ID outside catalog")
        safe_ids = item_ids.clamp(min=0, max=self.num_items - 1)
        target = torch.nn.functional.one_hot(
            safe_ids, num_classes=self.num_items,
        ).to(dtype=states.dtype)
        noise = torch.randn_like(target)
        times = torch.rand(batch_size, 1, device=states.device, dtype=states.dtype)
        interpolation = ((1 - times[:, None]) * noise
                         + times[:, None] * target)
        target_velocity = target - noise
        predicted_velocity = self(states, interpolation, times)
        squared_error = (predicted_velocity - target_velocity).square()
        # 稀疏 one-hot 的真实类别坐标使用更高权重，避免其监督被 I-1 个背景坐标稀释。
        coordinate_weights = 1 + target * (self.positive_weight - 1)
        loss_mask = valid[..., None].to(squared_error.dtype)
        denominator = (coordinate_weights * loss_mask).sum().clamp_min(1)
        loss = (squared_error * coordinate_weights * loss_mask).sum() / denominator
        diagnostics = {
            "flow_loss": loss.detach(),
            "velocity_norm": predicted_velocity.detach()[valid].norm(dim=-1).mean(),
            "target_velocity_norm": target_velocity.detach()[valid].norm(dim=-1).mean(),
            "time_mean": times.detach().mean(),
        }
        return loss, diagnostics

    @torch.no_grad()
    def integrate(
        self,
        states: torch.Tensor,
        noises: torch.Tensor,
    ) -> torch.Tensor:
        """从给定高斯噪声积分到 Flow 教师终点。

        Args:
            states (torch.Tensor): 状态，形状 `(B,state_dim)`。
            noises (torch.Tensor): 初始类别噪声，形状 `(B,K,I)`。

        Returns:
            torch.Tensor: 未做温度缩放的 Flow 终点，形状 `(B,K,I)`。

        Raises:
            ValueError: 状态或噪声形状不符合 actor 契约。
        """
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError("Flow integration states shape mismatch")
        if tuple(noises.shape) != (
            int(states.shape[0]), self.chunk_size, self.num_items,
        ):
            raise ValueError("Flow integration noises shape mismatch")
        categories = noises
        step_size = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            times = torch.full(
                (len(categories), 1), step * step_size,
                device=states.device, dtype=states.dtype,
            )
            categories = categories + step_size * self(states, categories, times)
        return categories

    @torch.no_grad()
    def sample_logits(
        self,
        states: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """从高斯基分布积分并返回每个候选的终点 categorical logits。

        Args:
            states (torch.Tensor): 状态，形状 `(B,state_dim)`。
            num_samples (int): 每个状态独立生成的候选数量。

        Returns:
            torch.Tensor: 终点 logits，形状 `(N,B,K,I)`。

        Raises:
            ValueError: 候选数非法或状态形状错误。
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError("Flow sampling states shape mismatch")
        batch_size = int(states.shape[0])
        repeated_states = states.unsqueeze(0).expand(num_samples, -1, -1)
        repeated_states = repeated_states.reshape(num_samples * batch_size, -1)
        noises = torch.randn(
            num_samples * batch_size, self.chunk_size, self.num_items,
            device=states.device, dtype=states.dtype,
        )
        endpoints = self.integrate(repeated_states, noises)
        return (endpoints / self.temperature).reshape(
            num_samples, batch_size, self.chunk_size, self.num_items,
        )


class CategoricalFlowOneStepActor(nn.Module):
    """用一次前向近似多步 categorical Flow 教师终点的蒸馏 actor。"""

    def __init__(
        self,
        state_dim: int,
        chunk_size: int,
        num_items: int,
        hidden_dims: Iterable[int],
        temperature: float = DEFAULT_FLOW_TEMPERATURE,
    ) -> None:
        """初始化一步 Flow 学生网络。

        Args:
            state_dim (int): 状态维度。
            chunk_size (int): 动作块长度。
            num_items (int): item 数量。
            hidden_dims (Iterable[int]): 学生 MLP 隐藏层。
            temperature (float): 推理时 categorical softmax 温度。

        Raises:
            ValueError: 维度或温度非法。
        """
        super().__init__()
        if min(state_dim, chunk_size, num_items) <= 0 or temperature <= 0:
            raise ValueError(
                "One-step Flow actor dimensions and temperature must be positive"
            )
        self.state_dim = int(state_dim)
        self.chunk_size = int(chunk_size)
        self.num_items = int(num_items)
        self.flow_dim = self.chunk_size * self.num_items
        self.temperature = float(temperature)
        self.net = MLP(
            input_dim=self.state_dim + self.flow_dim,
            hidden_dims=hidden_dims,
            output_dim=self.flow_dim,
            layer_norm=True,
        )

    def forward(self, states: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """由状态和初始噪声直接预测未缩放的 Flow 终点。"""
        batch_size = int(states.shape[0]) if states.ndim == 2 else -1
        if tuple(states.shape) != (batch_size, self.state_dim):
            raise ValueError("One-step Flow states shape mismatch")
        if tuple(noises.shape) != (
            batch_size, self.chunk_size, self.num_items,
        ):
            raise ValueError("One-step Flow noises shape mismatch")
        inputs = torch.cat((states, noises.flatten(1)), dim=-1)
        return self.net(inputs).reshape(
            batch_size, self.chunk_size, self.num_items,
        )

    @torch.no_grad()
    def sample_logits(self, states: torch.Tensor, num_samples: int) -> torch.Tensor:
        """一次前向生成多个候选的 categorical logits。"""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError("One-step Flow sampling states shape mismatch")
        batch_size = int(states.shape[0])
        repeated_states = states.unsqueeze(0).expand(num_samples, -1, -1)
        repeated_states = repeated_states.reshape(num_samples * batch_size, -1)
        noises = torch.randn(
            num_samples * batch_size, self.chunk_size, self.num_items,
            device=states.device, dtype=states.dtype,
        )
        endpoints = self(repeated_states, noises)
        return (endpoints / self.temperature).reshape(
            num_samples, batch_size, self.chunk_size, self.num_items,
        )
