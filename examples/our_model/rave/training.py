"""训练域短 rollout、有效前缀 TD 与共同策略的 V 目标。"""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from examples.our_model.models.mac_agent import RolloutHistoryState
from examples.our_model.rave.model import select_prefix
from examples.our_model.rave.state import COMPONENTS, Session, encode_sessions

DEFAULT_GRAD_CLIP = 1.0
REWARD_COMPONENT_INDEX = COMPONENTS.index("R")
"""四分量即时信号中原始反馈 R 的固定索引。"""


@dataclass
class PrefixRollout:
    """训练轨迹中的一个实际执行 chunk。

    candidates 是原始 K 个 item；rewards/endpoints 只包含实际展示步。
    return_to_go 保存从 chunk 起点到 episode 终止的折扣四分量回报，供
    完整 episode 诊断使用。value_target 只写入 imagined rollout 的第一个 chunk，
    保存 MAC chunk-level TD(lambda) value expansion 标签；中间 chunk 仅参与构造
    该标签，不作为独立监督样本。Transformer 的 ACSAC 目标不读取它。
    metadata 记录训练数据源、模型版本、episode 和 chunk 编号。
    """

    start: Session
    candidates: torch.Tensor
    rewards: torch.Tensor
    endpoints: list[Session]
    metadata: dict
    return_to_go: torch.Tensor | None = None
    value_target: torch.Tensor | None = None


@dataclass
class TensorTrajectoryStep:
    """纯 tensor imagined trajectory 中的一个 chunk 决策步。

    Attributes:
        rows (Tensor): 当前仍活跃样本在初始 batch 中的行号，形状 `(A,)`。
        inputs (dict): chunk 起点的 PrefixQV tensor 输入。
        next_inputs (dict): chunk 末状态的 PrefixQV tensor 输入。
        candidates (Tensor): 选中的 item chunk，形状 `(A,K)`。
        reward_chunks (Tensor): chunk 内折扣回报，形状 `(A,)`。
        discounts (Tensor): `gamma^effective_steps`，形状 `(A,)`。
        done (Tensor): chunk 末终止标记，形状 `(A,)`。
        effective_steps (Tensor): 实际展示步数，形状 `(A,)`。
    """

    rows: torch.Tensor
    inputs: dict[str, torch.Tensor]
    next_inputs: dict[str, torch.Tensor]
    candidates: torch.Tensor
    reward_chunks: torch.Tensor
    discounts: torch.Tensor
    done: torch.Tensor
    effective_steps: torch.Tensor


@dataclass
class TensorTrajectoryBatch:
    """ordinary 纯 tensor rollout 及其收集统计。

    Attributes:
        steps (list[TensorTrajectoryStep]): 按 chunk depth 排列的非空步骤。
        batch_size (int): 初始 trajectory 数量。
        completed_episodes (int): rollout 内自然/预算终止的 trajectory 数。
        total_chunks (int): 所有样本实际执行的 chunk 数。
        total_steps (int): 所有样本实际执行的 item 数。
        tail_chunks (int): 实际长度小于 K 的 chunk 数。
        value_target (Tensor | None): MAC TD(lambda) 起点 V 标签 `(B,1)`。
    """

    steps: list[TensorTrajectoryStep]
    batch_size: int
    completed_episodes: int
    total_chunks: int
    total_steps: int
    tail_chunks: int
    value_target: torch.Tensor | None = None


class TrainingSimulator:
    """训练 reward/终止接口，支持逐条参考路径和 batch 向量化路径。

    ``rollout`` 保留为数值等价参考实现；``rollout_batch`` 在每个 item 位置对
    完整 active batch 各调用一次 reward、leave 和 dynamics，是 ordinary 训练的
    默认执行路径。
    """

    def __init__(self, manager, reward_callback, natural_termination,
                 batch_reward_callback=None, batch_natural_termination=None):
        """初始化训练模拟器。

        Args:
            manager (StateManager): 维护 dynamics 状态、历史和终止标记的管理器。
            reward_callback (Callable): 接收 ``(Session, item)`` 并返回标量 R。
            natural_termination (Callable): 接收 ``(Session, item)`` 并返回退出布尔值。
            batch_reward_callback (Callable | None): 接收
                ``(sessions,item_ids,executable)`` 并返回 `(B,)` reward；为空时使用
                标量 callback 的兼容回退。
            batch_natural_termination (Callable | None): 接收相同参数并返回 `(B,)`
                bool；为空时使用标量 callback 的兼容回退。

        Returns:
            None.
        """
        self.manager = manager
        self.reward_callback = reward_callback
        self.natural_termination = natural_termination
        self.batch_reward_callback = batch_reward_callback
        self.batch_natural_termination = batch_natural_termination

    def step(self, session, item, new_plan):
        """执行一个模拟展示并返回四分量信号。

        Args:
            session (Session): 动作执行前的可变会话状态。
            item (int): 当前执行的内部 item ID。
            new_plan (bool): 当前动作是否为新 chunk 的第一步。

        Returns:
            torch.Tensor: ``(R,T,D,C)`` 四分量即时信号。

        Raises:
            ValueError: 状态管理器拒绝终止状态上的额外动作时抛出。
        """
        reward = float(self.reward_callback(session, item))
        natural = bool(self.natural_termination(session, item))
        return self.manager.advance(session, item, reward, new_plan, natural)

    def rollout(self, start, candidate, metadata, max_steps=None):
        """复制训练起点并执行最多 K 步；自然/预算终止后无虚构步骤。

        Args:
            start (Session): 带完整累计历史的非终止起点。
            candidate (Tensor): 一维 K 个 ID，允许后缀 -1 padding。
            metadata (dict): 数据源、模型版本与 seed。
            max_steps (int | None): 收集截断长度，不会伪造任务终止。

        Returns:
            PrefixRollout: 每个有效 h 的边界和即时 reward。

        Raises:
            ValueError: 空 rollout 或非法 padding。
        """
        state = start.clone()
        rewards, endpoints = [], []
        valid = candidate >= 0
        if torch.any(valid[1:] & ~valid[:-1]):
            raise ValueError("Padding must follow every valid item")
        limit = len(candidate) if max_steps is None else max_steps
        for position, item in enumerate(candidate.tolist()):
            if item < 0 or state.done or position >= limit:
                break
            rewards.append(self.step(state, int(item), position == 0))
            endpoints.append(state.clone())
        if not rewards:
            raise ValueError("A rollout must observe at least one step")
        return PrefixRollout(start.clone(), candidate.detach().cpu().clone(),
                             torch.stack(rewards), endpoints, dict(metadata))

    def _batch_step_values(self, sessions, item_ids, executable):
        """批量计算 reward 与自然终止标记，必要时回退到标量参考接口。

        Args:
            sessions (Sequence[Session]): 当前动作执行前的 batch 会话。
            item_ids (Tensor): 当前 item，形状 `(B,)`。
            executable (Tensor): 当前实际执行掩码，形状 `(B,)`。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 与 item device 对齐的 `(B,)` reward
            和 bool 自然终止标记；未执行行严格为零/假。

        Raises:
            ValueError: batch callback 返回形状不正确或 reward 非有限时抛出。
        """
        batch_size = len(sessions)
        device = item_ids.device
        if self.batch_reward_callback is None:
            rewards = torch.zeros(batch_size, dtype=torch.float32, device=device)
            item_values = item_ids.detach().cpu().tolist()
            execute_values = executable.detach().cpu().tolist()
            reward_values = [
                float(self.reward_callback(session, int(item_values[row])))
                if execute_values[row] else 0.0
                for row, session in enumerate(sessions)
            ]
            rewards.copy_(torch.as_tensor(reward_values, device=device))
        else:
            rewards = torch.as_tensor(
                self.batch_reward_callback(sessions, item_ids, executable),
                dtype=torch.float32,
                device=device,
            )
        if self.batch_natural_termination is None:
            item_values = item_ids.detach().cpu().tolist()
            execute_values = executable.detach().cpu().tolist()
            done_values = [
                bool(self.natural_termination(session, int(item_values[row])))
                if execute_values[row] else False
                for row, session in enumerate(sessions)
            ]
            natural_done = torch.as_tensor(
                done_values, dtype=torch.bool, device=device,
            )
        else:
            natural_done = torch.as_tensor(
                self.batch_natural_termination(
                    sessions, item_ids, executable,
                ),
                dtype=torch.bool,
                device=device,
            )
        expected_shape = (batch_size,)
        if rewards.shape != expected_shape or natural_done.shape != expected_shape:
            raise ValueError("Batch reward and termination must have shape (B,)")
        rewards = torch.where(executable, rewards, torch.zeros_like(rewards))
        natural_done = natural_done & executable
        if not torch.isfinite(rewards).all():
            raise ValueError("Batch reward contains non-finite values")
        return rewards, natural_done

    @torch.no_grad()
    def rollout_batch(self, starts, candidates, metadata):
        """在完整 batch 上同步执行一个 action chunk。

        外层仅遍历固定 K 个 item 位置；每个位置的 reward、退出判定和 dynamics
        都对 B 条轨迹一次计算。逐用户操作只负责维护可审计的 ``Session`` 列表和
        构造 ``PrefixRollout``，不再触发逐用户模型调用。

        Args:
            starts (Sequence[Session]): 非空且未终止的 chunk 起点列表。
            candidates (Tensor): 候选 item，形状 `(B,K)`，允许连续后缀 `-1`。
            metadata (Sequence[dict] | dict): 每行 metadata，或所有行共用的字典。

        Returns:
            list[PrefixRollout]: 长度 B 的 rollout 记录，顺序与输入一致。

        Raises:
            ValueError: batch 为空、候选形状非法、padding 不连续、起点终止或某行
                没有执行任何动作时抛出。

        Example:
            ``records = simulator.rollout_batch(states, chunks, metadata_rows)``。
        """
        batch_size = len(starts)
        if batch_size == 0 or candidates.ndim != 2:
            raise ValueError("rollout_batch requires sessions and candidates (B,K)")
        if candidates.shape != (batch_size, self.manager.chunk_size):
            raise ValueError("Candidate batch must have shape (B, chunk_size)")
        if any(session.done for session in starts):
            raise ValueError("rollout_batch cannot start from terminal sessions")
        valid_candidates = candidates >= 0
        if torch.any(valid_candidates[:, 1:] & ~valid_candidates[:, :-1]):
            raise ValueError("Padding must follow every valid item")
        if isinstance(metadata, dict):
            metadata_rows = [dict(metadata) for _ in range(batch_size)]
        else:
            metadata_rows = [dict(row) for row in metadata]
            if len(metadata_rows) != batch_size:
                raise ValueError("Metadata rows must match rollout batch size")

        sessions = [session.clone() for session in starts]
        reward_rows = [[] for _ in sessions]
        endpoint_rows = [[] for _ in sessions]
        device = candidates.device
        for position in range(candidates.shape[1]):
            item_ids = candidates[:, position]
            active = torch.as_tensor(
                [not session.done and session.budget > 0 for session in sessions],
                dtype=torch.bool,
                device=device,
            )
            executable = active & (item_ids >= 0)
            safe_item_ids = torch.where(
                executable, item_ids, torch.zeros_like(item_ids),
            )
            rewards, natural_done = self._batch_step_values(
                sessions, safe_item_ids, executable,
            )
            signals = self.manager.advance_batch(
                sessions=sessions,
                item_ids=safe_item_ids,
                raw_rewards=rewards,
                new_plan=position == 0,
                natural_done=natural_done,
                executable=executable,
            )
            execute_values = executable.detach().cpu().tolist()
            for row, executed in enumerate(execute_values):
                if executed:
                    reward_rows[row].append(signals[row])
                    endpoint_rows[row].append(sessions[row].clone())

        records = []
        for row in range(batch_size):
            if not reward_rows[row]:
                raise ValueError("Every batch rollout row must execute at least one item")
            records.append(PrefixRollout(
                starts[row].clone(),
                candidates[row].detach().cpu().clone(),
                torch.stack(reward_rows[row]),
                endpoint_rows[row],
                metadata_rows[row],
            ))
        return records


class CandidatePolicy:
    """冻结 MAC actor，并用单 R 头或四头综合 Q 选择候选与执行长度。"""

    def __init__(self, mac, network, utility, num_candidates=32,
                 mask_history=False, fixed_horizon=None, only_h=False):
        """配置采样数、共同候选限制和可选对照；非法 N/H 抛 ValueError。"""
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if fixed_horizon is not None and not 1 <= fixed_horizon <= network.chunk_size:
            raise ValueError("fixed_horizon must lie in [1,K]")
        self.mac = mac.eval().requires_grad_(False)
        self.network, self.utility = network, utility
        self.num_candidates, self.mask_history = num_candidates, mask_history
        self.fixed_horizon, self.only_h = fixed_horizon, only_h

    def inputs(self, sessions):
        """将 Session 列表编码到网络设备，返回输入字典。"""
        device = next(self.network.parameters()).device
        return encode_sessions(sessions, self.network.history_window,
                               self.network.num_items, device)

    @torch.no_grad()
    def sample(self, inputs, num_candidates=None, mask_history=None):
        """只调用冻结 actor，返回候选 chunk，不使用原 MAC critic。

        Args:
            inputs (dict[str, torch.Tensor]): `encode_sessions` 生成的策略输入。
            num_candidates (int | None): 每个状态采样的候选数；为空时使用
                `self.num_candidates`。
            mask_history (bool | None): 是否屏蔽当前会话已经展示的 item；
                为空时使用训练配置 `self.mask_history`。DORL 三分支评估会显式
                传入该值，使 FB 与 NX 分支保持原评估语义。

        Returns:
            torch.Tensor: 形状为 `(B, N, K)` 的离散候选 chunk。

        Raises:
            ValueError: 当候选数不是正整数时抛出。
        """
        count = self.num_candidates if num_candidates is None else num_candidates
        if count <= 0:
            raise ValueError("num_candidates must be positive")
        resolved_mask_history = self.mask_history if mask_history is None else bool(mask_history)
        mask = inputs["recommended"] if resolved_mask_history else None
        ids, _ = self.mac.sample_candidate_chunks(inputs["states"], count, mask)
        return ids.permute(1, 0, 2).contiguous()

    @torch.no_grad()
    def choose_inputs(self, inputs, epsilon=0.0, mask_history=None,
                      num_candidates=None, selection_horizon=None):
        """直接从 batch tensor 输入选择 chunk，避免 Session 重新编码。

        Args:
            inputs (dict[str, Tensor]): PrefixQV 输入，各字段首维均为 B。
            epsilon (float): 随机探索概率，必须位于 `[0, 1]`。
            mask_history (bool | None): 是否屏蔽已展示 item；为空时使用
                `self.mask_history`，DORL 评估时由 FB/NX 分支覆盖。
            num_candidates (int | None): 本次选择的候选数；为空时使用部署 N。
            selection_horizon (int | None): 本次选择强制执行的长度。完整 episode
                收集传入 K；为空时沿用部署/消融配置 `self.fixed_horizon`。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 选中的完整候选 chunk、
            执行前缀长度和对应 Q 预测。

        Raises:
            ValueError: 当 `epsilon` 越界或 tensor batch 为空时抛出。

        推理时暂时关闭 dropout，随后恢复原模式。
        """
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must lie in [0,1]")
        batch_size = len(inputs.get("states", ()))
        if batch_size == 0:
            raise ValueError("choose_inputs requires a non-empty tensor batch")
        candidates = self.sample(
            inputs,
            num_candidates=num_candidates,
            mask_history=mask_history,
        )
        fixed_horizon = (
            self.fixed_horizon
            if selection_horizon is None
            else selection_horizon
        )
        was_training = self.network.training
        self.network.eval()
        try:
            values = self.network(inputs, candidates)
            selected, horizons, scores = select_prefix(
                values, candidates, inputs["budget"], self.utility,
                fixed_horizon, self.only_h,
            )
            # 为整个 batch 一次生成 epsilon 探索决策。对每个合法候选/长度赋独立
            # 连续随机优先级，argmax 等价于从该行合法集合均匀抽取一个位置。
            explore = torch.rand(batch_size, device=values.device) < epsilon
            flattened_scores = scores.flatten(1)
            valid_choices = torch.isfinite(flattened_scores)
            if torch.any(~valid_choices.any(dim=1)):
                raise RuntimeError("Every session must have one valid candidate prefix")
            random_priorities = torch.rand_like(flattened_scores).masked_fill(
                ~valid_choices, -1.0,
            )
            random_choices = random_priorities.argmax(dim=1)
            random_selected = random_choices // self.network.chunk_size
            random_horizons = random_choices % self.network.chunk_size + 1
            selected = torch.where(explore, random_selected, selected)
            horizons = torch.where(explore, random_horizons, horizons)
            rows = torch.arange(batch_size, device=values.device)
            return (candidates[rows, selected], horizons,
                    values[rows, selected, horizons - 1])
        finally:
            self.network.train(was_training)

    @torch.no_grad()
    def choose(self, sessions, epsilon=0.0, mask_history=None,
               num_candidates=None, selection_horizon=None):
        """编码 Session 后批量选择 chunk，供评估和 Transformer 收集使用。

        Args:
            sessions (Sequence[Session]): 当前需要规划的非终止会话。
            epsilon (float): 候选级探索概率，范围 `[0,1]`。
            mask_history (bool | None): 是否屏蔽已推荐 item。
            num_candidates (int | None): 每状态候选 chunk 数。
            selection_horizon (int | None): 强制执行前缀长度。

        Returns:
            tuple[Tensor, Tensor, Tensor]: 选中 chunk、执行长度和 Q 预测。

        Raises:
            ValueError: 会话 batch 为空或探索参数非法时抛出。
        """
        if not sessions:
            raise ValueError("choose requires at least one session")
        return self.choose_inputs(
            self.inputs(sessions),
            epsilon=epsilon,
            mask_history=mask_history,
            num_candidates=num_candidates,
            selection_horizon=selection_horizon,
        )


@torch.no_grad()
def prefix_targets(rewards, valid, terminal, endpoint_values, gamma=0.9):
    """计算所有实际 h 的 TD target，终止步计 reward 且续接严格为零。

    Args:
        rewards (Tensor): (B,K,M) 即时分量，成本已按首步一次记账。
        valid/terminal (Tensor): (B,K) 实际展示掩码与边界终止标记。
        endpoint_values (Tensor): (B,K,M) target V，终止位置允许未计算。
        gamma (float): 论文多步回报的折扣率，必须位于 `[0, 1]`。

    Returns:
        Tensor: (B,K,M) 多步目标；padding 为零。

    Raises:
        ValueError: 折扣越界、非连续有效前缀或终止后存在有效步骤。
    """
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0,1]")
    if (rewards.ndim != 3 or rewards.shape != endpoint_values.shape
            or valid.shape != rewards.shape[:2] or terminal.shape != valid.shape
            or valid.dtype != torch.bool or terminal.dtype != torch.bool):
        raise ValueError("Incompatible reward, mask or endpoint-value shapes")
    if torch.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("Valid transitions must be contiguous")
    earlier_terminal = terminal.long().cumsum(1) - terminal.long()
    if torch.any(valid & (earlier_terminal > 0)):
        raise ValueError("Cannot train a transition after termination")
    horizon = torch.arange(rewards.shape[1], device=rewards.device,
                           dtype=rewards.dtype)
    reward_discount = torch.as_tensor(gamma, device=rewards.device,
                                      dtype=rewards.dtype).pow(horizon)
    continuation_discount = torch.as_tensor(
        gamma, device=rewards.device, dtype=rewards.dtype,
    ).pow(horizon + 1)
    immediate = torch.where(
        valid[..., None],
        rewards * reward_discount[None, :, None],
        torch.zeros_like(rewards),
    )
    continuation = torch.where(
        (valid & ~terminal)[..., None],
        endpoint_values * continuation_discount[None, :, None],
        torch.zeros_like(endpoint_values),
    )
    targets = immediate.cumsum(1) + continuation
    return torch.where(valid[..., None], targets, torch.zeros_like(targets))


@torch.no_grad()
def chunk_targets(rewards, valid, terminal, endpoint_values, gamma=0.9):
    """直接计算 ordinary 最后有效 chunk 位置的多步 TD target。

    Args:
        rewards (Tensor): 形状为 `(B,K,M)` 的逐步即时信号。
        valid (Tensor): 形状为 `(B,K)` 的连续有效前缀掩码。
        terminal (Tensor): 形状为 `(B,K)` 的逐步终止标记。
        endpoint_values (Tensor): 形状为 `(B,M)` 的末端状态停止梯度 V；
            终止 chunk 对应行允许为零。
        gamma (float): 每环境步折扣率，范围 `[0,1]`。

    Returns:
        torch.Tensor: 形状为 `(B,M)` 的完整 chunk 多步 TD target。

    Raises:
        ValueError: 输入形状、掩码连续性、终止位置或折扣率非法时抛出。

    Example:
        `targets = chunk_targets(rewards, valid, terminal, values, gamma=0.9)`。
    """
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0,1]")
    if (rewards.ndim != 3 or valid.shape != rewards.shape[:2]
            or terminal.shape != valid.shape
            or endpoint_values.shape != (len(rewards), rewards.shape[-1])
            or valid.dtype != torch.bool or terminal.dtype != torch.bool):
        raise ValueError("Incompatible chunk target shapes")
    if torch.any(valid.sum(1) < 1) or torch.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("Every chunk must contain one contiguous valid prefix")
    earlier_terminal = terminal.long().cumsum(1) - terminal.long()
    if torch.any(valid & (earlier_terminal > 0)):
        raise ValueError("Cannot train a transition after termination")

    lengths = valid.long().sum(1)
    rows = torch.arange(len(rewards), device=rewards.device)
    final_positions = lengths - 1
    discounts = torch.as_tensor(
        gamma, device=rewards.device, dtype=rewards.dtype,
    ).pow(torch.arange(rewards.shape[1], device=rewards.device,
                       dtype=rewards.dtype))
    discounted_rewards = rewards * discounts[None, :, None]
    reward_returns = torch.where(
        valid[..., None], discounted_rewards, torch.zeros_like(discounted_rewards),
    ).sum(1)
    discounted_continuation = torch.as_tensor(
        gamma, device=rewards.device, dtype=rewards.dtype,
    ).pow(lengths).unsqueeze(-1) * endpoint_values
    continuation = torch.where(
        (~terminal[rows, final_positions]).unsqueeze(-1),
        discounted_continuation,
        torch.zeros_like(discounted_continuation),
    )
    return reward_returns + continuation


@torch.no_grad()
def assign_episode_returns(records, gamma):
    """为一个按时间排列的完整 episode 写入 chunk 起点折扣回报。

    Args:
        records (Sequence[PrefixRollout]): 同一 episode 的非空、连续 chunk。
        gamma (float): 每环境步折扣率，必须位于 `[0, 1]`。

    Returns:
        list[PrefixRollout]: 原列表的浅拷贝；每条记录的 `return_to_go` 已更新。

    Raises:
        ValueError: 输入为空、折扣越界、chunk 为空或最后状态未终止时抛出。
    """
    if not records:
        raise ValueError("An episode must contain at least one chunk")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0,1]")
    if not records[-1].endpoints or not records[-1].endpoints[-1].done:
        raise ValueError("Episode collection must end at natural exit or max_turn")
    width = records[0].rewards.shape[-1]
    running = torch.zeros(width, dtype=records[0].rewards.dtype)
    for record in reversed(records):
        count = len(record.endpoints)
        if count == 0 or count != len(record.rewards):
            raise ValueError("Every episode chunk must contain valid transitions")
        discounts = torch.as_tensor(gamma, dtype=record.rewards.dtype).pow(
            torch.arange(count, dtype=record.rewards.dtype),
        )
        chunk_return = (record.rewards * discounts[:, None]).sum(0)
        running = chunk_return + (gamma ** count) * running
        record.return_to_go = running.clone()
    return list(records)


@torch.no_grad()
def collect_complete_episodes(policy, simulator, manager, user_ids,
                              initial_states, metadata, epsilon,
                              num_candidates, gamma):
    """按固定 K 行为策略收集直到自然退出或 `max_turn` 的完整 episodes。

    每个规划边界采样候选并只比较候选的完整可执行 chunk；选中后连续执行 K
    步，除非用户中途退出或预算耗尽。返回的每条记录恰好对应一个实际 chunk，
    最后不足 K 步的尾 chunk 仍被保留。

    Args:
        policy (CandidatePolicy): 冻结 actor 与当前 Q 组成的行为策略。
        simulator (TrainingSimulator): 训练域逐步模拟器。
        manager (StateManager): 会话初始化与状态递推管理器。
        user_ids (Sequence[int]): 本轮 episode 对应的训练用户内部索引。
        initial_states (Tensor): 所有用户的初始状态表。
        metadata (dict): 写入每个 chunk 的公共复现信息。
        epsilon (float): 候选级探索概率，范围 `[0, 1]`。
        num_candidates (int): 每个规划边界采样的候选 chunk 数。
        gamma (float): ordinary 回报与两类 TD 目标共用的每步折扣率。

    Returns:
        tuple[list[PrefixRollout], list[Session]]: 展平后的 chunk replay 记录及
        每个 episode 的终止状态，顺序与 `user_ids` 一致。

    Raises:
        ValueError: 用户列表为空、候选数非法或收集未正常终止时抛出。

    Example:
        `records, terminal_states = collect_complete_episodes(...)`。
    """
    if not user_ids:
        raise ValueError("At least one episode is required for collection")
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    sessions = [
        manager.reset(user_id, initial_states[user_id])
        for user_id in user_ids
    ]
    episodes = [[] for _ in sessions]
    active = list(range(len(sessions)))
    while active:
        active_sessions = [sessions[index] for index in active]
        candidates, horizons, _ = policy.choose(
            active_sessions,
            epsilon=epsilon,
            num_candidates=num_candidates,
            selection_horizon=policy.network.chunk_size,
        )
        for row, episode_index in enumerate(active):
            expected_horizon = min(
                policy.network.chunk_size,
                sessions[episode_index].budget,
            )
            if int(horizons[row]) != expected_horizon:
                raise RuntimeError("Fixed-K behavior selected an invalid horizon")
            chunk_metadata = dict(metadata)
            chunk_metadata.update({
                "episode_index": episode_index,
                "chunk_index": len(episodes[episode_index]),
                "behavior_horizon": int(horizons[row]),
            })
            record = simulator.rollout(
                sessions[episode_index],
                candidates[row],
                chunk_metadata,
            )
            episodes[episode_index].append(record)
            sessions[episode_index] = record.endpoints[-1].clone()
        active = [
            index for index, session in enumerate(sessions)
            if not session.done
        ]
    records = []
    for episode in episodes:
        records.extend(assign_episode_returns(episode, gamma))
    return records, sessions


@torch.no_grad()
def collect_imagined_rollouts(policy, simulator, manager, user_ids,
                              initial_states, metadata, epsilon,
                              num_candidates, rollout_depth):
    """从训练用户初始状态生成固定 chunk 深度的全新 imagined rollouts。

    每次调用都重新 reset 会话并重新采样候选；轨迹不会写入 replay。每个规划边界
    固定执行完整 K，除非用户自然退出或耗尽预算。达到 ``rollout_depth`` 时仅做
    收集截断，不把尾状态伪装成任务终止，供 EMA V bootstrap。

    Args:
        policy (CandidatePolicy): 冻结 actor 与当前 Q/V 组成的行为策略。
        simulator (TrainingSimulator): 使用冻结 ChunkDynamics/user model 的模拟器。
        manager (StateManager): 会话初始化与状态递推管理器。
        user_ids (Sequence[int]): 本轮 fresh rollout 对应的训练用户内部索引。
        initial_states (Tensor): 所有用户的初始状态表。
        metadata (dict): 写入每个 chunk 的公共复现信息。
        epsilon (float): 候选级探索概率，范围 ``[0, 1]``。
        num_candidates (int): 每个规划边界采样的候选 chunk 数。
        rollout_depth (int): 每条 imagined rollout 最多执行的 chunk 数。

    Returns:
        tuple[list[list[PrefixRollout]], list[Session]]: 按 episode 分组的 fresh
        chunks，以及自然终止或固定深度截断后的尾状态。

    Raises:
        ValueError: 用户列表为空、候选数或 rollout 深度不是正整数时抛出。

    Example:
        ``episodes, tails = collect_imagined_rollouts(..., rollout_depth=3)``。
    """
    if not user_ids:
        raise ValueError("At least one imagined rollout is required")
    if num_candidates <= 0 or rollout_depth <= 0:
        raise ValueError("num_candidates and rollout_depth must be positive")
    sessions = [
        manager.reset(user_id, initial_states[user_id])
        for user_id in user_ids
    ]
    episodes = [[] for _ in sessions]
    active = list(range(len(sessions)))
    for depth_index in range(rollout_depth):
        if not active:
            break
        active_sessions = [sessions[index] for index in active]
        candidates, horizons, _ = policy.choose(
            active_sessions,
            epsilon=epsilon,
            num_candidates=num_candidates,
            selection_horizon=policy.network.chunk_size,
        )
        horizon_values = horizons.detach().cpu().tolist()
        metadata_rows = []
        for row, episode_index in enumerate(active):
            expected_horizon = min(
                policy.network.chunk_size,
                sessions[episode_index].budget,
            )
            if int(horizon_values[row]) != expected_horizon:
                raise RuntimeError("Fixed-K behavior selected an invalid horizon")
            chunk_metadata = dict(metadata)
            chunk_metadata.update({
                "episode_index": episode_index,
                "chunk_index": depth_index,
                "behavior_horizon": int(horizon_values[row]),
                "rollout_depth": rollout_depth,
            })
            metadata_rows.append(chunk_metadata)
        batch_records = simulator.rollout_batch(
            active_sessions, candidates, metadata_rows,
        )
        for row, episode_index in enumerate(active):
            record = batch_records[row]
            episodes[episode_index].append(record)
            sessions[episode_index] = record.endpoints[-1].clone()
        active = [
            index for index in active
            if not sessions[index].done
        ]
    if any(not episode for episode in episodes):
        raise RuntimeError("Every imagined rollout must contain at least one chunk")
    return episodes, sessions


def _tensor_state_inputs(state, rows, max_turn):
    """从 ordinary tensor state 提取 PrefixQV batch 输入。

    Args:
        state (dict[str, Tensor]): 完整初始 batch 的当前 rollout 状态。
        rows (Tensor): 需要提取的原始 batch 行号，形状 `(A,)`。
        max_turn (int): episode 最大展示长度，必须为正。

    Returns:
        dict[str, Tensor]: 可直接传入 PrefixQV/CandidatePolicy 的输入。

    Raises:
        ValueError: `rows` 为空、不是一维或 `max_turn` 非正时抛出。
    """
    if rows.ndim != 1 or len(rows) == 0 or max_turn <= 0:
        raise ValueError("Tensor state rows must be non-empty and max_turn positive")
    env_steps = state["env_step"][rows]
    dominant = state["dominant_count"][rows]
    return {
        "states": state["states"][rows],
        "history": state["history"][rows],
        "feedback": state["feedback"][rows],
        "recommended": state["recommended"][rows],
        "counts": torch.stack((env_steps, dominant), dim=1),
        "budget": (max_turn - env_steps).clamp_min(0),
    }


@torch.no_grad()
def collect_tensor_imagined_rollouts(policy, user_ids, initial_states,
                                     raw_users, dominant_hits, max_turn,
                                     epsilon, num_candidates, rollout_depth,
                                     gamma):
    """用原 MAC tensor 内核生成 ordinary fresh imagined trajectories。

    训练热路径不构造 ``Session``、``PrefixRollout`` 或逐 item endpoint。每个
    chunk 边界由 suite Q 选择候选，随后复用 MAC 的 tensor reward/leave/dynamics
    执行器；dynamics 每个 chunk 只前向一次。

    Args:
        policy (CandidatePolicy): 冻结 actor 与 online ordinary Q/V 策略。
        user_ids (Sequence[int]): 训练用户内部索引，长度为 B。
        initial_states (Tensor): 所有用户 reset 状态表 `(U,state_dim)`。
        raw_users (Sequence[int]): 内部用户索引到 reward raw user ID 的映射。
        dominant_hits (Tensor): 每个 item 是否命中主导类别，形状 `(I,)`。
        max_turn (int): episode 最大展示步数。
        epsilon (float): 候选探索概率，范围 `[0,1]`。
        num_candidates (int): 每个边界采样的 chunk 数，必须为正。
        rollout_depth (int): 最大 imagined chunk 深度，必须为正。
        gamma (float): 每 item 折扣率，范围 `[0,1]`。

    Returns:
        TensorTrajectoryBatch: 纯 tensor 轨迹和收集统计。

    Raises:
        ValueError: batch、计数或折扣参数非法时抛出。
        RuntimeError: MAC tensor rollout 所需 reward/leave/dynamics 缺失时抛出。

    Example:
        ``trajectory = collect_tensor_imagined_rollouts(..., rollout_depth=3)``。
    """
    if not user_ids:
        raise ValueError("At least one tensor imagined rollout is required")
    if min(num_candidates, rollout_depth, max_turn) <= 0:
        raise ValueError("Tensor rollout counts must be positive")
    if not 0 <= epsilon <= 1 or not 0 <= gamma <= 1:
        raise ValueError("epsilon and gamma must lie in [0,1]")
    mac = policy.mac
    if (getattr(mac, "reward_model", None) is None
            or getattr(mac, "leave_model", None) is None
            or getattr(mac, "dynamics", None) is None):
        raise RuntimeError("Tensor rollout requires MAC reward, leave and dynamics")

    device = next(policy.network.parameters()).device
    user_index = torch.as_tensor(user_ids, dtype=torch.long, device=device)
    raw_user_ids = torch.as_tensor(
        [raw_users[int(user_id)] for user_id in user_ids],
        dtype=torch.long,
        device=device,
    )
    batch_size = len(user_ids)
    history_window = policy.network.history_window
    num_items = policy.network.num_items
    leave_width = mac.leave_model.num_leave_compute
    state = {
        "states": initial_states[user_index].to(device).detach().clone(),
        "history": torch.full(
            (batch_size, history_window), -1, dtype=torch.long, device=device,
        ),
        "feedback": torch.zeros(
            batch_size, history_window, dtype=torch.float32, device=device,
        ),
        "recommended": torch.zeros(
            batch_size, num_items, dtype=torch.bool, device=device,
        ),
        "leave_history": torch.full(
            (batch_size, leave_width), -1, dtype=torch.long, device=device,
        ),
        "env_step": torch.zeros(batch_size, dtype=torch.long, device=device),
        "dominant_count": torch.zeros(
            batch_size, dtype=torch.long, device=device,
        ),
        "terminated": torch.zeros(
            batch_size, dtype=torch.bool, device=device,
        ),
    }
    hits = torch.as_tensor(dominant_hits, dtype=torch.long, device=device)
    steps = []
    total_chunks = torch.zeros((), dtype=torch.long, device=device)
    total_steps = torch.zeros((), dtype=torch.long, device=device)
    tail_chunks = torch.zeros((), dtype=torch.long, device=device)
    repeat_policy = "mask" if policy.mask_history else "truncate"

    for _ in range(rollout_depth):
        rows = (~state["terminated"]).nonzero(as_tuple=False).flatten()
        if len(rows) == 0:
            break
        inputs = _tensor_state_inputs(state, rows, max_turn)
        candidates, _, _ = policy.choose_inputs(
            inputs,
            epsilon=epsilon,
            num_candidates=num_candidates,
            selection_horizon=policy.network.chunk_size,
        )
        rollout_state = RolloutHistoryState(
            states=state["states"][rows],
            leave_history=state["leave_history"][rows],
            recommended_mask=state["recommended"][rows],
            env_step=state["env_step"][rows],
            terminated=state["terminated"][rows],
        )
        mac._batch_user_ids = raw_user_ids[rows]
        result, next_rollout = mac._rollout_one_chunk(
            candidates,
            rollout_state,
            repeat_policy=repeat_policy,
            leave_policy="terminate",
        )

        history = state["history"][rows].clone()
        feedback = state["feedback"][rows].clone()
        dominant = state["dominant_count"][rows].clone()
        safe_items = result.selected_item_ids.clamp_min(0)
        # K 很小且历史具有时序依赖；循环只执行 tensor shift，不创建 Python 轨迹对象。
        for position in range(policy.network.chunk_size):
            valid = result.chunk_valid[:, position]
            item_column = safe_items[:, position:position + 1]
            reward_column = result.step_rewards[:, position:position + 1]
            shifted_history = torch.cat((history[:, 1:], item_column), dim=1)
            shifted_feedback = torch.cat((feedback[:, 1:], reward_column), dim=1)
            history = torch.where(valid[:, None], shifted_history, history)
            feedback = torch.where(valid[:, None], shifted_feedback, feedback)
            dominant = dominant + hits[safe_items[:, position]] * valid.long()

        state["states"][rows] = next_rollout.states
        state["history"][rows] = history
        state["feedback"][rows] = feedback
        state["recommended"][rows] = next_rollout.recommended_mask
        state["leave_history"][rows] = next_rollout.leave_history
        state["env_step"][rows] = next_rollout.env_step
        state["dominant_count"][rows] = dominant
        state["terminated"][rows] = next_rollout.terminated
        next_inputs = _tensor_state_inputs(state, rows, max_turn)
        discounts = torch.as_tensor(
            gamma, dtype=result.reward_chunks.dtype, device=device,
        ).pow(result.effective_steps)
        steps.append(TensorTrajectoryStep(
            rows=rows,
            inputs=inputs,
            next_inputs=next_inputs,
            candidates=candidates,
            reward_chunks=result.reward_chunks,
            discounts=discounts,
            done=result.done_chunks,
            effective_steps=result.effective_steps,
        ))
        total_chunks += len(rows)
        total_steps += result.effective_steps.sum()
        tail_chunks += (result.effective_steps < policy.network.chunk_size).sum()

    if not steps:
        raise RuntimeError("Tensor rollout produced no trajectory step")
    return TensorTrajectoryBatch(
        steps=steps,
        batch_size=batch_size,
        completed_episodes=int(state["terminated"].sum().cpu()),
        total_chunks=int(total_chunks.cpu()),
        total_steps=int(total_steps.cpu()),
        tail_chunks=int(tail_chunks.cpu()),
    )


class RAVETrainer:
    """按模型类型执行 MAC 起点 Q/V TD 或 ACSAC 全前缀 TD。

    `optimize_all_prefixes=False` 表示 ordinary：每条 fresh imagined trajectory 只把
    第一个 chunk 的 Q 和起点 V 纳入未缩放 MSE，后续 chunk 仅构造 MAC TD(lambda)
    value expansion target。为 True 表示 Transformer：所有有效前缀使用 ACSAC
    多步 Q bootstrap，V head 不参与优化。两种模式共用冻结 Flow actor；ordinary
    不使用跨更新 replay，Transformer 保留完整 episode replay。ACSAC bootstrap
    严格使用在线 critic 的停止梯度输出。
    """

    def __init__(self, policy, learning_rate=3e-4, tau=0.005, alpha_v=1.0,
                 grad_clip=DEFAULT_GRAD_CLIP, optimize_all_prefixes=True,
                 gamma=0.9, lambda_chunk=1.0):
        """接收策略、优化参数和 Q 监督范围；非法配置抛 ValueError。

        Args:
            policy (CandidatePolicy): 共享候选策略与 Q/V 网络。
            learning_rate (float): ordinary Adam 或 Transformer AdamW 的学习率，
                必须为正数。
            tau (float): target 网络 EMA 系数，范围为 `(0, 1]`。
            alpha_v (float): V loss 权重，必须非负。
            grad_clip (float): 梯度范数上限，必须为正数。
            optimize_all_prefixes (bool): 为真时监督全部有效前缀；为假时每条
                imagined trajectory 只监督第一个 chunk 起点的 Q/V。
            gamma (float): MAC/ACSAC 每环境步折扣率，范围 `[0, 1]`。
            lambda_chunk (float): ordinary 的 chunk-level TD(lambda) 系数，范围
                `[0, 1]`；MAC 默认值为 1。

        Raises:
            ValueError: 优化参数或监督模式不合法时抛出。
        """
        if (not 0 < tau <= 1 or min(learning_rate, grad_clip) <= 0
                or alpha_v < 0 or not 0 <= gamma <= 1
                or not 0 <= lambda_chunk <= 1):
            raise ValueError("Invalid optimizer or target-update parameters")
        if not isinstance(optimize_all_prefixes, bool):
            raise ValueError("optimize_all_prefixes must be bool")
        self.policy, self.online = policy, policy.network
        if not optimize_all_prefixes and not self.online.scalar:
            raise ValueError("ordinary Q/V requires a scalar R network")
        if not optimize_all_prefixes and alpha_v != 1.0:
            raise ValueError("MAC ordinary requires alpha_v=1")
        self.target = deepcopy(self.online).eval().requires_grad_(False)
        optimizer_type = torch.optim.AdamW if optimize_all_prefixes else torch.optim.Adam
        self.optimizer = optimizer_type(self.online.parameters(), lr=learning_rate)
        self.tau, self.alpha_v, self.grad_clip = tau, alpha_v, grad_clip
        self.optimize_all_prefixes = optimize_all_prefixes
        self.gamma = float(gamma)
        self.lambda_chunk = float(lambda_chunk)
        self.updates = 0

    @torch.no_grad()
    def critic_targets(self, sessions):
        """按 ACSAC 在在线 Q 上联合选择候选与前缀并返回停止梯度 bootstrap。

        Args:
            sessions (Sequence[Session]): 每个有效数据前缀的下一状态；终止状态
                直接返回零且不调用 actor。

        Returns:
            torch.Tensor: `(B,1)` 或 `(B,4)` 的停止梯度 target Q 分量。
        """
        device = next(self.online.parameters()).device
        width = 1 if self.online.scalar else len(COMPONENTS)
        targets = torch.zeros(len(sessions), width, device=device)
        rows = [row for row, state in enumerate(sessions) if not state.done]
        if not rows:
            return targets
        inputs = self.policy.inputs([sessions[row] for row in rows])
        candidates = self.policy.sample(inputs)
        # ACSAC Appendix B 明确采用 current online critic with gradient stopped。
        # 临时关闭 dropout，使同一状态和候选得到确定的 Bellman 标签；调用完成后
        # 恢复训练模式，主回归分支仍按正常配置使用 dropout。
        was_training = self.online.training
        self.online.eval()
        try:
            evaluated = self.online(inputs, candidates)
            selected, horizon, _ = select_prefix(
                evaluated,
                candidates,
                inputs["budget"],
                self.policy.utility,
                self.policy.fixed_horizon,
                self.policy.only_h,
            )
            indices = torch.arange(len(rows), device=device)
            targets[rows] = evaluated[indices, selected, horizon - 1]
        finally:
            self.online.train(was_training)
        return targets

    @torch.no_grad()
    def value_targets(self, sessions):
        """兼容旧调用名，返回 `critic_targets` 的 ACSAC bootstrap 结果。"""
        return self.critic_targets(sessions)

    @torch.no_grad()
    def assign_imagined_value_targets(self, episodes):
        """为每条 fresh trajectory 的起点写入 MAC TD(lambda) V 标签。

        先在每个 chunk 边界用 EMA target V 构造 TD residual，再按 chunk 的实际
        展示步数和 ``lambda_chunk`` 反向递推。只有第一个 chunk 保存 value_target
        并作为直接 Q/V 监督样本；中间 chunk 只参与 target 构造，不获得梯度。

        Args:
            episodes (Sequence[Sequence[PrefixRollout]]): 同一轮 freshly generated
                imagined rollouts，episode 内 chunk 必须按时间顺序排列。

        Returns:
            list[PrefixRollout]: 每条 trajectory 的第一个 chunk，均已写入
            ``value_target``；列表长度等于输入 trajectory 数。

        Raises:
            ValueError: 在 Transformer 模式调用、输入为空或轨迹结构非法时抛出。

        Example:
            ``fresh_records = trainer.assign_imagined_value_targets(episodes)``。
        """
        if self.optimize_all_prefixes:
            raise ValueError("Imagined V targets are only defined for ordinary Q/V")
        if not episodes or any(not episode for episode in episodes):
            raise ValueError("Imagined rollout episodes must be non-empty")
        for episode in episodes:
            for record in episode:
                count = len(record.endpoints)
                if count == 0 or count != len(record.rewards):
                    raise ValueError(
                        "Every imagined chunk must contain valid transitions"
                    )
        device = next(self.target.parameters()).device
        all_records = [record for episode in episodes for record in episode]
        start_values = self.target.value(
            self.policy.inputs([record.start for record in all_records])
        )
        next_values = torch.zeros(len(all_records), 1, device=device)
        live_rows = [
            row for row, record in enumerate(all_records)
            if not record.endpoints[-1].done
        ]
        if live_rows:
            next_values[live_rows] = self.target.value(self.policy.inputs([
                all_records[row].endpoints[-1] for row in live_rows
            ]))

        first_records = []
        offset = 0
        for episode in episodes:
            episode_size = len(episode)
            episode_starts = start_values[offset:offset + episode_size]
            episode_next = next_values[offset:offset + episode_size]
            gae = torch.zeros(1, device=device)
            # 与 MACAgent._chunk_value_expansion_target 相同：后续 chunk 只进入
            # TD(lambda) residual 链，不成为在线 Q/V 的独立回归样本。
            for index in reversed(range(episode_size)):
                record = episode[index]
                count = len(record.endpoints)
                discounts = torch.as_tensor(
                    self.gamma, device=device, dtype=record.rewards.dtype,
                ).pow(torch.arange(count, device=device,
                                   dtype=record.rewards.dtype))
                rewards = record.rewards[:, REWARD_COMPONENT_INDEX].to(device)
                chunk_return = (rewards * discounts).sum().reshape(1)
                continuation = float(not record.endpoints[-1].done)
                chunk_discount = self.gamma ** count
                delta = (
                    chunk_return
                    + continuation * chunk_discount * episode_next[index]
                    - episode_starts[index]
                )
                gae = (
                    delta
                    + self.lambda_chunk * chunk_discount * continuation * gae
                )
                record.value_target = None
            first_record = episode[0]
            first_record.value_target = (
                gae + episode_starts[0]
            ).detach().cpu()
            first_records.append(first_record)
            offset += episode_size
        return first_records

    @torch.no_grad()
    def assign_tensor_value_target(self, trajectory):
        """直接在纯 tensor trajectory 上构造 MAC 起点 TD(lambda) 标签。

        Args:
            trajectory (TensorTrajectoryBatch): fresh ordinary tensor rollout。

        Returns:
            torch.Tensor: 初始 batch 的 V target，形状 `(B,1)`。

        Raises:
            ValueError: 在 Transformer 模式调用、trajectory 为空或首步未覆盖
                初始 batch 全部行时抛出。

        Example:
            ``targets = trainer.assign_tensor_value_target(trajectory)``。
        """
        if self.optimize_all_prefixes:
            raise ValueError("Tensor V target is only defined for ordinary Q/V")
        if not trajectory.steps:
            raise ValueError("Tensor trajectory must contain at least one step")
        device = next(self.target.parameters()).device
        gae = torch.zeros(trajectory.batch_size, device=device)
        first_values = None
        for step_index in reversed(range(len(trajectory.steps))):
            step = trajectory.steps[step_index]
            current_values = self.target.value(step.inputs).squeeze(-1)
            next_values = self.target.value(step.next_inputs).squeeze(-1)
            continuation = (~step.done).to(dtype=current_values.dtype)
            delta = (
                step.reward_chunks
                + continuation * step.discounts * next_values
                - current_values
            )
            gae[step.rows] = (
                delta
                + self.lambda_chunk * step.discounts * continuation * gae[step.rows]
            )
            if step_index == 0:
                first_values = current_values
        first_step = trajectory.steps[0]
        expected_rows = torch.arange(
            trajectory.batch_size, device=device, dtype=first_step.rows.dtype,
        )
        if not torch.equal(first_step.rows, expected_rows):
            raise ValueError("First tensor rollout step must cover the initial batch")
        trajectory.value_target = (first_values + gae).unsqueeze(-1).detach()
        return trajectory.value_target

    def update_tensor_trajectory(self, trajectory):
        """用纯 tensor trajectory 的首 chunk 执行严格 MAC Q/V 更新。

        Args:
            trajectory (TensorTrajectoryBatch): 本轮新生成且未进入 replay 的轨迹。

        Returns:
            dict[str, float | int]: loss、TD 误差、batch 和更新步指标。

        Raises:
            ValueError: 在 Transformer 模式调用或 trajectory 结构非法时抛出。
            FloatingPointError: loss 或梯度范数非有限时抛出。

        Example:
            ``metrics = trainer.update_tensor_trajectory(trajectory)``。
        """
        if self.optimize_all_prefixes:
            raise ValueError("Tensor trajectory update is only available for ordinary")
        if not trajectory.steps:
            raise ValueError("Tensor trajectory must contain at least one step")
        self.online.train()
        first = trajectory.steps[0]
        value_target = self.assign_tensor_value_target(trajectory)
        with torch.no_grad():
            next_values = self.target.value(first.next_inputs).squeeze(-1)
            continuation = (~first.done).to(dtype=next_values.dtype)
            q_target = (
                first.reward_chunks
                + continuation * first.discounts * next_values
            ).unsqueeze(-1)
        q_prediction = self.online.chunk_value(
            first.inputs,
            first.candidates,
            first.effective_steps,
        )
        value_prediction = self.online.value(first.inputs)
        q_loss = functional.mse_loss(q_prediction, q_target)
        v_loss = functional.mse_loss(value_prediction, value_target)
        loss = q_loss + v_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite tensor ordinary loss")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        squared_norm = sum(
            parameter.grad.detach().pow(2).sum()
            for parameter in self.online.parameters()
            if parameter.grad is not None
        )
        grad_norm = squared_norm.sqrt()
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite tensor ordinary gradient")
        self.optimizer.step()
        with torch.no_grad():
            for target, online in zip(
                    self.target.parameters(), self.online.parameters()):
                target.lerp_(online, self.tau)
            for target, online in zip(self.target.buffers(), self.online.buffers()):
                target.copy_(online)
        self.updates += 1

        metrics = {
            "loss": float(loss.detach()),
            "q_loss": float(q_loss.detach()),
            "v_loss": float(v_loss.detach()),
            "grad_norm": float(grad_norm.detach()),
            "valid_prefixes": trajectory.batch_size,
            "observed_prefixes": int(first.effective_steps.sum().detach().cpu()),
            "bootstrap_states": int((~first.done).sum().detach().cpu()),
            "update": self.updates,
            "v_mae/R": float(
                (value_prediction - value_target).detach().abs().mean()
            ),
        }
        for length in range(1, self.online.chunk_size + 1):
            rows = first.effective_steps == length
            if rows.any():
                metrics[f"td_mae/h{length}/R"] = float(
                    (q_prediction[rows] - q_target[rows]).detach().abs().mean()
                )
        return metrics

    def update(self, records):
        """用非空 PrefixRollout batch 更新 Q/V，返回 loss 与分长度 TD 误差。

        ordinary 输入必须是每条 fresh trajectory 的第一个 chunk，只前向该 chunk
        的 Q 和起点 V，并使用 MAC 未缩放 MSE；
        Transformer 对每个已有前缀监督，尾 chunk 不构造 padding 标签。所有
        bootstrap target 和 actor 候选采样均不参与反向传播。
        """
        if not records:
            raise ValueError("Empty replay batch")
        self.online.train()
        starts = [record.start for record in records]
        inputs = self.policy.inputs(starts)
        device = inputs["states"].device
        candidates = torch.stack([record.candidates for record in records]).to(device)
        batch_size, length = candidates.shape
        rewards = torch.zeros(batch_size, length, len(COMPONENTS), device=device)
        valid = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
        terminal = valid.clone()
        continuations, positions = [], []
        for row, record in enumerate(records):
            count = len(record.endpoints)
            if count != len(record.rewards) or not 0 < count <= length:
                raise ValueError("Invalid rollout transitions")
            rewards[row, :count] = record.rewards.to(device)
            # 每条记录被解释为一次开环前缀，不能沿用日志内部的旧规划事件。
            rewards[row, :count, COMPONENTS.index("C")] = 0
            rewards[row, 0, COMPONENTS.index("C")] = 1
            valid[row, :count] = True
            for step, endpoint in enumerate(record.endpoints):
                terminal[row, step] = endpoint.done
                if self.optimize_all_prefixes and not endpoint.done:
                    continuations.append(endpoint)
                    positions.append((row, step))
            if not self.optimize_all_prefixes and not record.endpoints[-1].done:
                # MAC ordinary 只需要完整 chunk 末端的 V，不计算中间前缀 V。
                continuations.append(record.endpoints[-1])
                positions.append((row, count - 1))
        if self.online.scalar:
            # 单头消融只学习原始反馈 R，不允许 T/D/C 经权重混入标签。
            rewards = rewards[..., REWARD_COMPONENT_INDEX:REWARD_COMPONENT_INDEX + 1]
        valid_lengths = valid.long().sum(1)
        with torch.no_grad():
            if self.optimize_all_prefixes:
                endpoint_values = torch.zeros(
                    batch_size, length, rewards.shape[-1], device=device,
                )
                if continuations:
                    # ACSAC Eq. (5)：每个数据前缀在下一状态重新采样候选，
                    # 用在线 critic 的停止梯度输出在候选与长度上做联合 argmax。
                    values = self.critic_targets(continuations)
                    rows, steps = zip(*positions)
                    endpoint_values[list(rows), list(steps)] = values
                targets = prefix_targets(
                    rewards, valid, terminal, endpoint_values, gamma=self.gamma,
                )
            else:
                endpoint_values = torch.zeros(
                    batch_size, rewards.shape[-1], device=device,
                )
                if continuations:
                    # MAC chunk TD：只在实际 chunk 末端接停止梯度的 EMA V。
                    values = self.target.value(self.policy.inputs(continuations))
                    rows, _ = zip(*positions)
                    endpoint_values[list(rows)] = values
                targets = chunk_targets(
                    rewards, valid, terminal, endpoint_values, gamma=self.gamma,
                )
        if self.optimize_all_prefixes:
            predictions = self.online(inputs, candidates[:, None])[:, 0]
            q_supervision = valid
            supervised_residuals = (
                predictions - targets
            )[q_supervision]
        else:
            predictions = self.online.chunk_value(
                inputs, candidates, valid_lengths,
            )
            q_supervision = torch.ones(batch_size, dtype=torch.bool, device=device)
            supervised_residuals = predictions - targets
        value_prediction = None
        value_target = None
        if self.optimize_all_prefixes:
            scales = predictions.new_tensor(self.policy.utility.scales)
            normalized_residuals = supervised_residuals / scales
            q_loss = functional.smooth_l1_loss(
                normalized_residuals,
                torch.zeros_like(normalized_residuals),
                reduction="none",
            ).sum(-1).mean()
            # ACSAC 不定义独立 V；保留数值为零方便训练日志横向比较。
            v_loss = q_loss.new_zeros(())
            loss = q_loss
        else:
            if any(record.value_target is None for record in records):
                raise ValueError(
                    "MAC ordinary requires fresh imagined value_target labels"
                )
            value_target = torch.stack([
                record.value_target for record in records
            ]).to(device)
            value_prediction = self.online.value(inputs)
            # 原 MAC qv_update 对 Q(s0,a0) 与 V(s0) 均使用未缩放 MSE。
            q_loss = functional.mse_loss(predictions, targets)
            v_loss = functional.mse_loss(value_prediction, value_target)
            loss = q_loss + v_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite RAVE loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.optimize_all_prefixes:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.online.parameters(), self.grad_clip,
            )
        else:
            # 原 MAC 的 Adam 更新不额外裁剪 Q/V 梯度。
            squared_norm = sum(
                parameter.grad.detach().pow(2).sum()
                for parameter in self.online.parameters()
                if parameter.grad is not None
            )
            grad_norm = squared_norm.sqrt()
        self.optimizer.step()
        with torch.no_grad():
            for target, online in zip(self.target.parameters(), self.online.parameters()):
                target.lerp_(online, self.tau)
            for target, online in zip(self.target.buffers(), self.online.buffers()):
                target.copy_(online)
        self.updates += 1
        metrics = {"loss": float(loss.detach()), "q_loss": float(q_loss.detach()),
                   "v_loss": float(v_loss.detach()), "grad_norm": float(grad_norm),
                   "valid_prefixes": int(q_supervision.sum()),
                   "observed_prefixes": int(valid.sum()),
                   "bootstrap_states": len(continuations), "update": self.updates}
        names = ("R",) if self.online.scalar else COMPONENTS
        for component, name in enumerate(names):
            if value_prediction is not None:
                metrics[f"v_mae/{name}"] = float(
                    (value_prediction - value_target)[:, component]
                    .detach().abs().mean()
                )
            for step in range(length):
                rows = (q_supervision[:, step] if self.optimize_all_prefixes
                        else valid_lengths == step + 1)
                if rows.any():
                    if self.optimize_all_prefixes:
                        error = (predictions[rows, step, component]
                                 - targets[rows, step, component])
                    else:
                        error = predictions[rows, component] - targets[rows, component]
                    metrics[f"td_mae/h{step + 1}/{name}"] = float(error.detach().abs().mean())
        return metrics
