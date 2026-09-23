"""共享因果前缀 Q/V、固定尺度和共同动作选择。"""

from dataclasses import dataclass

import torch
from torch import nn

NUM_COMPONENTS = 4
REWARD_QUANTILE = 0.95
MIN_REWARD_SCALE = 1e-6


@dataclass(frozen=True)
class Utility:
    """冻结尺度与业务权重；cost 仅在 scalarize 中取负号。"""

    scales: tuple[float, ...]
    weights: tuple[float, ...] = (1.0, 0.2, 0.2, -0.05)

    def __post_init__(self):
        """检查四个固定正尺度和有限权重；非法配置抛 ValueError。"""
        scales, weights = torch.tensor(self.scales), torch.tensor(self.weights)
        if scales.shape != (NUM_COMPONENTS,) or weights.shape != scales.shape:
            raise ValueError("Utility requires four scales and weights")
        if not torch.isfinite(scales).all() or not torch.all(scales > 0):
            raise ValueError("Scales must be finite and positive")
        if not torch.isfinite(weights).all():
            raise ValueError("Weights must be finite")

    @classmethod
    def from_training_rewards(cls, rewards, max_turn, eta_t=0.2, eta_d=0.2, eta_c=0.05):
        """用训练样本绝对偏好 95% 分位数冻结尺度，返回 Utility。

        空样本、非有限 reward 或非正预算抛 ValueError；不依赖当前候选或 h。
        """
        values = torch.as_tensor(rewards, dtype=torch.float32).flatten()
        if not values.numel() or not torch.isfinite(values).all() or max_turn <= 0:
            raise ValueError("Invalid training rewards or budget")
        reward_scale = max(float(values.abs().quantile(REWARD_QUANTILE)), MIN_REWARD_SCALE)
        return cls((max_turn * reward_scale, float(max_turn), 1.0, float(max_turn)),
                   (1.0, eta_t, eta_d, -eta_c))

    def scalarize(self, components):
        """将末维 (R,T,D,C) 转为固定综合效用；返回去掉末维的张量。"""
        return (components / components.new_tensor(self.scales)
                * components.new_tensor(self.weights)).sum(-1)


class PrefixQV(nn.Module):
    """共享状态编码和因果主干，输出四个线性 Q/V 分量。

    history_projection 保留有序窗口；推荐掩码的 embedding 均值提供全历史摘要。
    backbone='mlp' 是共享的因果前缀 MLP 对照；scalar=True 是仅 R 的单头对照。
    """

    def __init__(self, state_dim, item_embeddings, category_ids, dominant_hits,
                 chunk_size, max_turn, history_window=20, hidden_dim=128,
                 layers=2, heads=4, dropout=0.1, scalar=False, backbone="transformer"):
        """初始化网络；输入类别表按内部 item 对齐，0 为无效类别 padding。

        Args:
            state_dim (int): MAC 状态维度。
            item_embeddings (Tensor): 冻结 item embedding，形状 (I,A)。
            category_ids (Tensor): 正整数类别 ID，形状 (I,L)，0 为 padding。
            dominant_hits (Tensor): item 主导类别命中标记，形状 (I,)。
            chunk_size/max_turn/history_window (int): 正数 K、预算、窗口。
            hidden_dim/layers/heads (int): 主干宽度、层数、注意力头数。
            dropout (float): 训练 dropout 概率。
            scalar (bool): 是否只训练原始反馈 R 的单头价值。
            backbone (str): transformer 或 mlp。

        Raises:
            ValueError: 维度、类别或模型配置非法。
        """
        super().__init__()
        categories = torch.as_tensor(category_ids, dtype=torch.long)
        hits = torch.as_tensor(dominant_hits, dtype=torch.long)
        if min(state_dim, chunk_size, max_turn, history_window, hidden_dim, layers, heads) <= 0:
            raise ValueError("Network dimensions must be positive")
        if hidden_dim % heads or backbone not in {"transformer", "mlp"}:
            raise ValueError("Invalid attention width or backbone")
        if categories.ndim != 2 or len(categories) != len(item_embeddings) or categories.min() < 0:
            raise ValueError("Invalid category table")
        if hits.shape != (len(categories),) or torch.any((hits != 0) & (hits != 1)):
            raise ValueError("Invalid dominant-hit table")
        self.chunk_size, self.max_turn = chunk_size, max_turn
        self.history_window, self.num_items = history_window, len(item_embeddings)
        self.scalar, self.backbone = scalar, backbone
        self.item_embedding = nn.Embedding.from_pretrained(item_embeddings.float(), freeze=True)
        self.item_projection = nn.Linear(item_embeddings.shape[1], hidden_dim)
        self.register_buffer("category_ids", categories)
        self.register_buffer("dominant_hits", hits)
        self.category_embedding = nn.Embedding(int(categories.max()) + 1, hidden_dim, padding_idx=0)
        self.dominant_embedding = nn.Embedding(2, hidden_dim)
        self.position = nn.Embedding(chunk_size + 1, hidden_dim)
        self.feedback_projection = nn.Linear(1, hidden_dim, bias=False)
        self.history_projection = nn.Linear(history_window * hidden_dim, hidden_dim)
        # n、k、F、剩余预算和推荐目录占比显式进入起点 token。
        self.state_projection = nn.Linear(state_dim + 5 + 2 * hidden_dim, hidden_dim)
        if backbone == "transformer":
            layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 4,
                                               dropout, batch_first=True)
            self.trunk = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        else:
            self.trunk = nn.Sequential(
                nn.Linear((chunk_size + 1) * hidden_dim, hidden_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            )
        output_dim = 1 if scalar else NUM_COMPONENTS
        self.q_heads = nn.Linear(hidden_dim, output_dim)
        self.v_heads = nn.Linear(hidden_dim, output_dim)

    def item_tokens(self, ids):
        """编码合法 item 或 -1 padding，返回 (...,hidden_dim)，非法 ID 抛错。"""
        if torch.any(ids < -1) or torch.any(ids >= self.num_items):
            raise ValueError("Candidate item ID outside catalog")
        safe = ids.clamp_min(0)
        categories = self.category_ids[safe]
        category_mask = categories > 0
        category = (self.category_embedding(categories) * category_mask.unsqueeze(-1)).sum(-2)
        category = category / category_mask.sum(-1, keepdim=True).clamp_min(1)
        tokens = (self.item_projection(self.item_embedding(safe)) + category
                  + self.dominant_embedding(self.dominant_hits[safe]))
        return tokens * (ids >= 0).unsqueeze(-1)

    def state_token(self, inputs):
        """从已发生交互构建 StateToken；inputs 为 encode_sessions 输出字典。"""
        history = inputs["history"]
        tokens = self.item_tokens(history)
        tokens = tokens + self.feedback_projection(inputs["feedback"].unsqueeze(-1))
        tokens = tokens * (history >= 0).unsqueeze(-1)
        history_summary = self.history_projection(tokens.flatten(1))
        counts = inputs["counts"].float()
        fraction = torch.where(counts[:, 0] > 0,
                               1 - counts[:, 1] / counts[:, 0].clamp_min(1), 0)
        mask = inputs["recommended"].float()
        # 该摘要覆盖整个会话，但不宣称能无损恢复完整 item 掩码。
        item_table = self.item_tokens(torch.arange(self.num_items, device=mask.device))
        mask_summary = mask @ item_table / mask.sum(-1, keepdim=True).clamp_min(1)
        features = torch.cat((counts / self.max_turn, fraction[:, None],
                              inputs["budget"][:, None] / self.max_turn,
                              mask.mean(-1, keepdim=True)), dim=-1)
        return self.state_projection(torch.cat((inputs["states"], history_summary,
                                                mask_summary, features), dim=-1))

    def _encode(self, tokens):
        """编码 (B,L,D) 序列，因果掩码保证位置 h 不读取后缀。"""
        length = tokens.shape[1]
        if self.backbone == "transformer":
            causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
            return self.trunk(tokens, mask=causal)
        # MLP 对照同样只读取起点与当前位置以前的动作，共享所有 h 的参数。
        padded = tokens.new_zeros((len(tokens), self.chunk_size + 1, tokens.shape[-1]))
        padded[:, :length] = tokens
        outputs = []
        for position in range(length):
            prefix = padded.clone()
            prefix[:, position + 1:] = 0
            outputs.append(self.trunk(prefix.flatten(1)))
        return torch.stack(outputs, dim=1)

    def value(self, inputs):
        """只输入 StateToken 计算 (B,4) 或 (B,1) 的 V，不读取候选。"""
        token = self.state_token(inputs) + self.position.weight[0]
        return self.v_heads(self._encode(token[:, None])[:, 0])

    def chunk_value(self, inputs, candidate_ids, valid_lengths):
        """一次前向计算每条 MLP chunk 的最后有效 Q，不生成前缀 Q。

        Args:
            inputs (dict): 当前真实历史和状态，batch 大小必须与候选一致。
            candidate_ids (Tensor): 形状为 `(B,K)` 的完整候选 item ID。
            valid_lengths (Tensor): 形状为 `(B,)` 的实际执行长度，范围 `[1,K]`；
                自然退出产生的尾 chunk 可以小于 K。

        Returns:
            torch.Tensor: 形状为 `(B,4)` 或 `(B,1)` 的末端 chunk Q。

        Raises:
            ValueError: 主干不是 MLP、输入形状非法、长度越界或有效前缀含 padding
                时抛出。

        Example:
            `q = network.chunk_value(inputs, candidates, lengths)`。
        """
        if self.backbone != "mlp":
            raise ValueError("chunk_value is only available for the ordinary MLP critic")
        if (candidate_ids.ndim != 2
                or candidate_ids.shape[1] != self.chunk_size
                or valid_lengths.shape != (len(candidate_ids),)):
            raise ValueError("Expected candidate IDs (B,K) and valid lengths (B,)")
        lengths = valid_lengths.to(device=candidate_ids.device, dtype=torch.long)
        if torch.any(lengths < 1) or torch.any(lengths > self.chunk_size):
            raise ValueError("Valid chunk lengths must lie in [1,K]")
        positions = torch.arange(self.chunk_size, device=candidate_ids.device)
        executed = positions[None] < lengths[:, None]
        if torch.any((candidate_ids < 0) & executed):
            raise ValueError("An executed chunk prefix cannot contain padding")

        state = self.state_token(inputs) + self.position.weight[0]
        actions = self.item_tokens(candidate_ids) + self.position.weight[1:]
        # 每行只保留实际执行的前 L 个动作；尾 chunk 的未执行后缀不进入 Q。
        actions = actions * executed.unsqueeze(-1)
        tokens = torch.cat((state[:, None], actions), dim=1)
        return self.q_heads(self.trunk(tokens.flatten(1)))

    def forward(self, inputs, candidate_ids):
        """返回候选的 (B,N,K,4) Q，单 R 头对照末维为 1。

        Args:
            inputs (dict): 当前真实历史和状态，不含未来反馈。
            candidate_ids (Tensor): (B,N,K) ID，允许连续后缀 padding=-1。

        Raises:
            ValueError: 形状错误或 padding 不是后缀。
        """
        if candidate_ids.ndim != 3 or candidate_ids.shape[2] != self.chunk_size:
            raise ValueError("Expected candidate IDs with shape (B,N,K)")
        valid = candidate_ids >= 0
        if torch.any(valid[..., 1:] & ~valid[..., :-1]):
            raise ValueError("Candidate padding must be a suffix")
        batch_size, num_candidates, length = candidate_ids.shape
        state = self.state_token(inputs) + self.position.weight[0]
        actions = self.item_tokens(candidate_ids) + self.position.weight[1:length + 1]
        states = state[:, None, None].expand(-1, num_candidates, 1, -1)
        tokens = torch.cat((states, actions), dim=2).flatten(0, 1)
        encoded = self._encode(tokens)
        return self.q_heads(encoded[:, 1:]).reshape(batch_size, num_candidates, length, -1)


def select_prefix(q_values, candidate_ids, budget, utility, fixed_horizon=None, only_h=False):
    """共同选择一个候选及 H，返回 (候选索引,H,已掩码 scores)。

    仅按预算、连续有效长度及显式对照设置限制动作；精确同分优先候选编号再小 H。
    fixed_horizon 在预算或有效长度不足时取剩余可执行长度；only_h 限定候选 0。
    不调用退出规则，也不叠加局部 reward 或 V。无合法前缀抛 ValueError。
    """
    # 单头张量严格表示 Q_R；四头张量才使用业务权重合成决策分数。
    scores = q_values[..., 0] if q_values.shape[-1] == 1 else utility.scalarize(q_values)
    length = candidate_ids.shape[-1]
    horizon = torch.arange(1, length + 1, device=scores.device)
    valid_lengths = (candidate_ids >= 0).long().cumprod(-1).sum(-1)
    limit = torch.minimum(valid_lengths, budget[:, None])
    allowed = horizon <= limit[..., None]
    if fixed_horizon is not None:
        if not 1 <= fixed_horizon <= length:
            raise ValueError("fixed_horizon must lie in [1,K]")
        allowed &= horizon == limit.clamp_max(fixed_horizon)[..., None]
    if only_h:
        allowed[:, 1:] = False
    if not allowed.flatten(1).any(-1).all():
        raise ValueError("No valid prefix for a terminal or empty candidate state")
    if not torch.isfinite(scores[allowed]).all():
        raise ValueError("Non-finite prefix scores")
    scores = scores.masked_fill(~allowed, -torch.inf)
    selected = scores.flatten(1).argmax(-1)
    return selected // length, selected % length + 1, scores
