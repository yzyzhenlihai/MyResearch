"""固定全局锚点的状态递推与完整会话多目标记账。"""

from dataclasses import dataclass, field

import torch

PAD_ITEM = -1
COMPONENTS = ("R", "T", "D", "C")


@dataclass
class Session:
    """保存单环境完整历史和独立于规划边界的 dynamics 锚点。

    anchor 每实际执行 K 步提交一次；pending 保留锚点后实际动作。
    items/feedback 跨 chunk 保留，只有 reset 才创建新对象。
    """

    user_id: int
    anchor: torch.Tensor
    state: torch.Tensor
    max_turn: int
    items: list[int] = field(default_factory=list)
    feedback: list[float] = field(default_factory=list)
    pending: list[int] = field(default_factory=list)
    dominant_count: int = 0
    natural_done: bool = False
    budget_done: bool = False

    @property
    def budget(self) -> int:
        """返回剩余实际展示预算；无参数、无异常。"""
        return self.max_turn - len(self.items)

    @property
    def done(self) -> bool:
        """返回任务是否自然结束或耗尽预算；无参数、无异常。"""
        return self.natural_done or self.budget_done

    @property
    def fraction(self) -> float:
        """返回当前完整会话非主导曝光比例，空历史返回零。"""
        return 1 - self.dominant_count / len(self.items) if self.items else 0.0

    def clone(self) -> "Session":
        """返回可独立推进的快照；张量脱离梯度，历史不共享。"""
        return Session(
            self.user_id, self.anchor.detach().clone(),
            self.state.detach().clone(), self.max_turn, self.items.copy(),
            self.feedback.copy(), self.pending.copy(), self.dominant_count,
            self.natural_done, self.budget_done,
        )


class StateManager:
    """统一训练、V 边界和执行状态；advance 是唯一实际展示入口。"""

    def __init__(self, dynamics, item_embeddings, dominant_hits, max_turn):
        """接收冻结 dynamics、item embedding、命中表和正数预算。

        Raises:
            ValueError: 表大小不匹配或预算非法。
        """
        if max_turn <= 0 or len(item_embeddings) != len(dominant_hits):
            raise ValueError("Invalid state assets or max_turn")
        self.dynamics = dynamics.eval().requires_grad_(False)
        self.embeddings = item_embeddings.detach()
        self.hits = torch.as_tensor(dominant_hits, dtype=torch.bool)
        self.max_turn = int(max_turn)
        self.chunk_size = dynamics.chunk_size

    def reset(self, user_id, initial_state):
        """用用户 ID 和初始一维状态创建空会话，返回 Session。"""
        if initial_state.ndim != 1 or not torch.isfinite(initial_state).all():
            raise ValueError("Initial state must be a finite vector")
        initial = initial_state.detach().clone()
        return Session(int(user_id), initial, initial.clone(), self.max_turn)

    @torch.no_grad()
    def advance(self, session, item_id, raw_reward, new_plan, natural_done=False):
        """实际展示后推进状态并返回原始单位的 (R,T,D,C)。

        Args:
            session (Session): 将原地更新的非终止会话。
            item_id (int): 内部 item ID。
            raw_reward (float): 已获得的有限原始偏好，可为负。
            new_plan (bool): 此步是否为所选前缀首次展示。
            natural_done (bool): 此次实际交互的自然终止结果。

        Returns:
            torch.Tensor: CPU 上四维 reward。

        Raises:
            ValueError: 终止后推进、非法 item 或非有限 reward。
        """
        if session.done or session.budget <= 0:
            raise ValueError("Cannot advance a terminal session")
        if not 0 <= item_id < len(self.hits):
            raise ValueError("Item ID is outside the catalog")
        if not torch.isfinite(torch.tensor(raw_reward)):
            raise ValueError("Non-finite raw preference")
        previous_fraction = session.fraction
        session.items.append(int(item_id))
        session.feedback.append(float(raw_reward))
        session.dominant_count += int(self.hits[item_id])
        session.pending.append(int(item_id))
        # 锚点固定在全局展示数 K 的整数倍，不能根据选定 H 提交。
        length = len(session.pending)
        actions = self.embeddings.new_zeros(
            (1, self.chunk_size, self.embeddings.shape[1]),
        )
        actions[0, :length] = self.embeddings[session.pending]
        valid = torch.arange(self.chunk_size, device=actions.device)[None] < length
        session.state = self.dynamics(session.anchor[None], actions, valid)[0]
        if length == self.chunk_size:
            session.anchor = session.state.clone()
            session.pending.clear()
        session.natural_done = bool(natural_done)
        session.budget_done = session.budget == 0
        return torch.tensor([
            raw_reward, 1.0, session.fraction - previous_fraction, float(new_plan),
        ], dtype=torch.float32)

    @torch.no_grad()
    def advance_batch(self, sessions, item_ids, raw_rewards, new_plan,
                      natural_done, executable):
        """批量执行一次展示，并用单次 dynamics 前向推进所有可执行会话。

        Python 循环仅维护 ``Session`` 的可变日志字段；item embedding 查表、前缀
        mask 和 dynamics 推理均在完整 batch 上执行。这样每个环境步只发生一次
        模型调用，不随 batch size 增加模型调用次数。

        Args:
            sessions (Sequence[Session]): 长度为 B 的会话列表，将被原地推进。
            item_ids (Tensor): 当前候选 item，形状为 `(B,)`。
            raw_rewards (Tensor): 当前标量反馈，形状为 `(B,)`。
            new_plan (bool): 当前展示是否为新 chunk 的第一步。
            natural_done (Tensor): 当前动作后自然终止标记，形状为 `(B,)`。
            executable (Tensor): 本步实际执行标记，形状为 `(B,)`。

        Returns:
            torch.Tensor: CPU 上 `(B,4)` 的 R/T/D/C 即时信号；未执行行为零。

        Raises:
            ValueError: batch 为空、张量形状不匹配、item/reward 非法、会话已终止
                却被标记为可执行，或 pending 超过全局 chunk 长度时抛出。

        Example:
            ``signals = manager.advance_batch(states, items, rewards, True,
            done, valid)``。
        """
        batch_size = len(sessions)
        expected_shape = (batch_size,)
        tensors = (item_ids, raw_rewards, natural_done, executable)
        if batch_size == 0 or any(tuple(tensor.shape) != expected_shape
                                  for tensor in tensors):
            raise ValueError("Batch state and step tensor shapes must agree")

        item_values = item_ids.detach().cpu().tolist()
        reward_values = raw_rewards.detach().cpu().tolist()
        done_values = natural_done.detach().cpu().tolist()
        execute_values = executable.detach().cpu().tolist()
        previous_fractions = [session.fraction for session in sessions]
        for row, session in enumerate(sessions):
            if not execute_values[row]:
                continue
            item_id = int(item_values[row])
            raw_reward = float(reward_values[row])
            if session.done or session.budget <= 0:
                raise ValueError("Cannot advance a terminal session")
            if not 0 <= item_id < len(self.hits):
                raise ValueError("Item ID is outside the catalog")
            if not torch.isfinite(torch.tensor(raw_reward)):
                raise ValueError("Non-finite raw preference")
            session.items.append(item_id)
            session.feedback.append(raw_reward)
            session.dominant_count += int(self.hits[item_id])
            session.pending.append(item_id)
            if len(session.pending) > self.chunk_size:
                raise ValueError("Session pending actions exceed dynamics chunk size")

        device = self.embeddings.device
        actions = self.embeddings.new_zeros(
            (batch_size, self.chunk_size, self.embeddings.shape[1]),
        )
        valid = torch.zeros(
            batch_size, self.chunk_size, dtype=torch.bool, device=device,
        )
        for row, session in enumerate(sessions):
            pending_length = len(session.pending)
            if pending_length:
                pending_ids = torch.as_tensor(
                    session.pending, dtype=torch.long, device=device,
                )
                actions[row, :pending_length] = self.embeddings[pending_ids]
                valid[row, :pending_length] = True
        anchors = torch.stack([session.anchor for session in sessions]).to(device)
        next_states = self.dynamics(anchors, actions, valid)

        signals = torch.zeros(batch_size, len(COMPONENTS), dtype=torch.float32)
        for row, session in enumerate(sessions):
            if not execute_values[row]:
                continue
            session.state = next_states[row].detach().clone()
            if len(session.pending) == self.chunk_size:
                session.anchor = session.state.clone()
                session.pending.clear()
            session.natural_done = bool(done_values[row])
            session.budget_done = session.budget == 0
            signals[row] = torch.tensor([
                reward_values[row],
                1.0,
                session.fraction - previous_fractions[row],
                float(new_plan),
            ], dtype=torch.float32)
        return signals


def encode_sessions(sessions, history_window, num_items, device):
    """将完整会话转为网络输入，保留精确计数和按时间排列的历史。

    Args:
        sessions (Sequence[Session]): 非空起点或边界状态。
        history_window (int): 正数网络历史窗口。
        num_items (int): item 目录大小。
        device (torch.device): 网络所在设备。

    Returns:
        dict[str, torch.Tensor]: 状态、历史反馈、完整已推荐掩码及整数预算/计数。

    Raises:
        ValueError: 空 batch 或非法历史窗口。
    """
    if not sessions or history_window <= 0:
        raise ValueError("A nonempty batch and positive history window are required")
    history = torch.full((len(sessions), history_window), PAD_ITEM, dtype=torch.long)
    feedback = torch.zeros_like(history, dtype=torch.float32)
    recommended = torch.zeros((len(sessions), num_items), dtype=torch.bool)
    for row, session in enumerate(sessions):
        recent = session.items[-history_window:]
        if recent:
            history[row, -len(recent):] = torch.tensor(recent)
            feedback[row, -len(recent):] = torch.tensor(session.feedback[-history_window:])
            recommended[row, session.items] = True
    return {
        "states": torch.stack([s.state for s in sessions]).to(device),
        "history": history.to(device), "feedback": feedback.to(device),
        "recommended": recommended.to(device),
        "counts": torch.tensor([[len(s.items), s.dominant_count] for s in sessions],
                               dtype=torch.long, device=device),
        "budget": torch.tensor([s.budget for s in sessions], device=device),
    }
