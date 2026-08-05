"""缓存 action chunk 长期失效与延迟重规划代价的配对反事实评估。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """保存推荐环境分叉评估所需的最小可变状态。

    Attributes:
        cur_user (int): 当前内部用户编号。
        action (int): 最近一次执行的内部物品编号。
        cum_reward (float): 当前 episode 累计奖励。
        total_turn (int): 当前 episode 已执行步数。
        history_action (dict[int, int]): 环境按时间索引保存的动作历史。
        sequence_action (list[int]): 环境按执行顺序保存的动作历史。
        max_history (int): 环境动作历史长度。
    """

    cur_user: int
    action: int
    cum_reward: float
    total_turn: int
    history_action: dict[int, int]
    sequence_action: list[int]
    max_history: int


@dataclass(frozen=True)
class DelayEvaluation:
    """一个重规划延迟分支的局部干预与完整未来轨迹结果。

    Attributes:
        delay_steps (int): 重新规划前额外执行的缓存动作数。
        local_reward_sum (float): 长度为原缓存后缀长度的局部窗口回报。
        future_reward_sum (float): 从分支状态到 episode 结束的完整未来回报。
        local_executed_steps (int): 局部窗口内实际执行步数。
        future_executed_steps (int): 从分支状态到 episode 结束的实际步数。
        local_terminated (bool): episode 是否在局部窗口内结束。
        terminated (bool): 完整未来 rollout 是否到达自然退出或最大步数。
    """

    delay_steps: int
    local_reward_sum: float
    future_reward_sum: float
    local_executed_steps: int
    future_executed_steps: int
    local_terminated: bool
    terminated: bool


@dataclass(frozen=True)
class BranchPointResult:
    """同一中间状态上多个重规划延迟分支的配对结果。

    Attributes:
        planned_steps (int): 原 chunk 剩余动作数，也是局部干预窗口长度。
        evaluations (tuple[DelayEvaluation, ...]): 按延迟从小到大排列的结果。
    """

    planned_steps: int
    evaluations: tuple[DelayEvaluation, ...]

    def _get_evaluation(self, delay_steps: int) -> DelayEvaluation:
        """取得指定延迟分支。

        Args:
            delay_steps (int): 目标分支的重规划延迟。

        Returns:
            DelayEvaluation: 与延迟匹配的评估结果。

        Raises:
            RuntimeError: 当结果中缺少目标分支时抛出。
        """

        for evaluation in self.evaluations:
            if evaluation.delay_steps == delay_steps:
                return evaluation
        raise RuntimeError(
            f"BranchPointResult is missing delay_steps={delay_steps}."
        )

    @property
    def immediate_evaluation(self) -> DelayEvaluation:
        """返回立即重规划分支。

        Returns:
            DelayEvaluation: `delay_steps=0` 分支。
        """

        return self._get_evaluation(0)

    @property
    def continue_evaluation(self) -> DelayEvaluation:
        """返回完整执行缓存后缀的 Continue 分支。

        Returns:
            DelayEvaluation: `delay_steps=planned_steps` 分支。
        """

        return self._get_evaluation(self.planned_steps)

    @property
    def long_term_replanning_advantage(self) -> float:
        """计算立即重规划相对完整 Continue 的长期回报优势。

        Returns:
            float: 两分支从当前状态到 episode 结束的未来回报之差。
        """

        return (
            self.immediate_evaluation.future_reward_sum
            - self.continue_evaluation.future_reward_sum
        )

    @property
    def short_term_replanning_gain(self) -> float:
        """计算局部窗口内每个计划步的重规划收益，仅用于机制诊断。

        Returns:
            float: 局部回报差除以固定局部窗口长度。
        """

        return (
            self.immediate_evaluation.local_reward_sum
            - self.continue_evaluation.local_reward_sum
        ) / self.planned_steps


Planner = Callable[[Sequence[int], Sequence[float]], Sequence[int]]
DownstreamPlanner = Callable[
    [Sequence[int], Sequence[float], int],
    Sequence[int],
]


def capture_environment(env: Any) -> EnvironmentSnapshot:
    """捕获 BaseEnv/KuaiEnv 的可变 episode 状态。

    Args:
        env (Any): 具有 BaseEnv 状态字段的推荐环境。

    Returns:
        EnvironmentSnapshot: 可用于精确恢复的环境快照。

    Raises:
        AttributeError: 当环境缺少分叉评估所需字段时抛出。
    """

    required_fields = (
        "cur_user",
        "action",
        "cum_reward",
        "total_turn",
        "history_action",
        "sequence_action",
        "max_history",
        "max_turn",
    )
    missing_fields = [name for name in required_fields if not hasattr(env, name)]
    if missing_fields:
        raise AttributeError(
            "Environment does not support state branching; missing fields: "
            f"{missing_fields}."
        )
    return EnvironmentSnapshot(
        cur_user=int(env.cur_user),
        action=int(env.action),
        cum_reward=float(env.cum_reward),
        total_turn=int(env.total_turn),
        history_action=copy.deepcopy(env.history_action),
        sequence_action=copy.deepcopy(env.sequence_action),
        max_history=int(env.max_history),
    )


def restore_environment(env: Any, snapshot: EnvironmentSnapshot) -> None:
    """把推荐环境恢复到指定 episode 快照。

    Args:
        env (Any): 待恢复的 BaseEnv/KuaiEnv 环境。
        snapshot (EnvironmentSnapshot): `capture_environment` 生成的快照。

    Returns:
        None.
    """

    env.cur_user = snapshot.cur_user
    env.action = snapshot.action
    env.cum_reward = snapshot.cum_reward
    env.total_turn = snapshot.total_turn
    env.history_action = copy.deepcopy(snapshot.history_action)
    env.sequence_action = copy.deepcopy(snapshot.sequence_action)
    env.max_history = snapshot.max_history


def _step_branch(
    env: Any,
    action: int,
    history_items: list[int],
    history_rewards: list[float],
) -> tuple[float, bool]:
    """执行一个分支动作并同步策略状态历史。

    Args:
        env (Any): 当前反事实环境分支。
        action (int): 待执行内部物品编号。
        history_items (list[int]): 会被原地追加的物品历史。
        history_rewards (list[float]): 会被原地追加的奖励历史。

    Returns:
        tuple[float, bool]: 当前步奖励与 episode 结束标记。
    """

    _, reward, terminated, truncated, _ = env.step(int(action))
    reward_float = float(reward)
    history_items.append(int(action))
    history_rewards.append(reward_float)
    reached_max_turn = int(env.total_turn) >= int(env.max_turn)
    return reward_float, bool(terminated or truncated or reached_max_turn)


def _roll_out_downstream(
    env: Any,
    history_items: list[int],
    history_rewards: list[float],
    downstream_planner: DownstreamPlanner,
    initial_terminated: bool,
) -> tuple[float, int, bool]:
    """用相同下游策略从局部干预窗口末端滚动到 episode 结束。

    Args:
        env (Any): 已处于局部干预窗口末端的环境分支。
        history_items (list[int]): 当前分支的完整物品历史。
        history_rewards (list[float]): 当前分支的完整奖励历史。
        downstream_planner (DownstreamPlanner): 每次生成一个完整后续 chunk
            的固定策略回调；第三个参数是该分支内从零开始的规划编号。
        initial_terminated (bool): 局部窗口是否已经终止 episode。

    Returns:
        tuple[float, int, bool]: 下游回报、下游执行步数与最终结束标记。

    Raises:
        ValueError: 当下游 planner 返回空动作序列时抛出。
        RuntimeError: 当环境超过自身最大步数仍未结束时抛出。
    """

    downstream_reward = 0.0
    downstream_steps = 0
    terminated = bool(initial_terminated)
    planning_index = 0
    maximum_followup_steps = max(int(env.max_turn) - int(env.total_turn), 0)

    while not terminated and downstream_steps < maximum_followup_steps:
        actions = list(
            downstream_planner(
                history_items,
                history_rewards,
                planning_index,
            )
        )
        if not actions:
            raise ValueError("downstream_planner must return at least one action.")
        planning_index += 1
        for action in actions:
            reward, terminated = _step_branch(
                env,
                int(action),
                history_items,
                history_rewards,
            )
            downstream_reward += reward
            downstream_steps += 1
            if terminated or downstream_steps >= maximum_followup_steps:
                break

    if not terminated and int(env.total_turn) >= int(env.max_turn):
        terminated = True
    if not terminated and downstream_steps >= maximum_followup_steps:
        raise RuntimeError(
            "Downstream rollout exhausted env.max_turn without termination."
        )
    return downstream_reward, downstream_steps, terminated


def _evaluate_delay(
    env: Any,
    snapshot: EnvironmentSnapshot,
    history_items: Sequence[int],
    history_rewards: Sequence[float],
    cached_suffix: Sequence[int],
    delay_steps: int,
    planner: Planner,
    downstream_planner: DownstreamPlanner,
) -> DelayEvaluation:
    """评估一个延迟分支的局部干预和完整未来轨迹。

    局部窗口长度固定为缓存后缀长度：先执行 `delay_steps` 个旧动作，
    再用最新状态重规划并补足窗口。窗口结束后，无论局部分支采用何种
    干预，都切换到同一个下游策略，直到自然退出或达到 `max_turn`。

    Args:
        env (Any): 可恢复状态的推荐环境。
        snapshot (EnvironmentSnapshot): 分支共同起点。
        history_items (Sequence[int]): 起点时策略看到的完整物品历史。
        history_rewards (Sequence[float]): 起点时策略看到的完整奖励历史。
        cached_suffix (Sequence[int]): 原 chunk 尚未执行的缓存动作。
        delay_steps (int): 重规划前继续执行的缓存动作数。
        planner (Planner): 在局部窗口内根据最新历史生成新 chunk 的回调。
        downstream_planner (DownstreamPlanner): 局部窗口后统一使用的策略。

    Returns:
        DelayEvaluation: 局部回报与完整未来轨迹回报。

    Raises:
        ValueError: 当历史、缓存后缀、延迟或 planner 输出非法时抛出。
    """

    if len(history_items) != len(history_rewards):
        raise ValueError("history_items and history_rewards must have equal length.")
    planned_steps = len(cached_suffix)
    if planned_steps <= 0:
        raise ValueError("cached_suffix must contain at least one action.")
    if not 0 <= delay_steps <= planned_steps:
        raise ValueError(
            f"delay_steps must be in [0, {planned_steps}], got {delay_steps}."
        )

    restore_environment(env, snapshot)
    branch_items = list(history_items)
    branch_rewards = list(history_rewards)
    local_reward_sum = 0.0
    local_executed_steps = 0
    terminated = False

    # 先执行指定数量的旧缓存动作；一旦自然退出，该分支不再重规划。
    for action in cached_suffix[:delay_steps]:
        reward, terminated = _step_branch(
            env,
            int(action),
            branch_items,
            branch_rewards,
        )
        local_reward_sum += reward
        local_executed_steps += 1
        if terminated:
            break

    remaining_steps = planned_steps - local_executed_steps
    if not terminated and remaining_steps > 0:
        replanned_actions = list(planner(branch_items, branch_rewards))
        if len(replanned_actions) < remaining_steps:
            raise ValueError(
                "planner returned fewer actions than the remaining local window: "
                f"needed={remaining_steps}, got={len(replanned_actions)}."
            )
        for action in replanned_actions[:remaining_steps]:
            reward, terminated = _step_branch(
                env,
                int(action),
                branch_items,
                branch_rewards,
            )
            local_reward_sum += reward
            local_executed_steps += 1
            if terminated:
                break

    local_terminated = terminated
    downstream_reward, downstream_steps, terminated = _roll_out_downstream(
        env=env,
        history_items=branch_items,
        history_rewards=branch_rewards,
        downstream_planner=downstream_planner,
        initial_terminated=local_terminated,
    )
    return DelayEvaluation(
        delay_steps=delay_steps,
        local_reward_sum=local_reward_sum,
        future_reward_sum=local_reward_sum + downstream_reward,
        local_executed_steps=local_executed_steps,
        future_executed_steps=local_executed_steps + downstream_steps,
        local_terminated=local_terminated,
        terminated=terminated,
    )


def evaluate_branch_point(
    env: Any,
    history_items: Sequence[int],
    history_rewards: Sequence[float],
    cached_suffix: Sequence[int],
    planner: Planner,
    downstream_planner: DownstreamPlanner,
    delay_values: Optional[Sequence[int]] = None,
) -> BranchPointResult:
    """在同一状态上比较不同延迟分支的完整未来轨迹回报。

    每个分支都从同一环境快照开始。局部干预窗口结束后，各分支使用
    相同的下游规划策略滚动到 episode 结束。函数结束后恢复主轨迹环境。

    Args:
        env (Any): 当前主轨迹环境。
        history_items (Sequence[int]): 当前策略物品历史，包含 reset dummy。
        history_rewards (Sequence[float]): 当前策略奖励历史。
        cached_suffix (Sequence[int]): 原 chunk 未执行的动作。
        planner (Planner): 局部窗口内根据分支最新历史生成动作的回调。
        downstream_planner (DownstreamPlanner): 局部窗口后统一执行的策略。
        delay_values (Optional[Sequence[int]]): 要评估的延迟集合。为空时
            评估 `0..L`；若指定，仍必须包含立即重规划 `0` 和完整继续 `L`。

    Returns:
        BranchPointResult: 指定延迟分支的长期配对结果。

    Raises:
        ValueError: 当缓存后缀为空、延迟集合非法或底层输入非法时抛出。
    """

    if not cached_suffix:
        raise ValueError("cached_suffix must contain at least one action.")
    planned_steps = len(cached_suffix)
    selected_delays = (
        list(range(planned_steps + 1))
        if delay_values is None
        else sorted(set(int(value) for value in delay_values))
    )
    if not selected_delays:
        raise ValueError("delay_values must not be empty.")
    if any(value < 0 or value > planned_steps for value in selected_delays):
        raise ValueError(
            f"delay_values must be within [0, {planned_steps}]."
        )
    if 0 not in selected_delays or planned_steps not in selected_delays:
        raise ValueError(
            "delay_values must include both 0 and the full-continue delay."
        )

    snapshot = capture_environment(env)
    evaluations: list[DelayEvaluation] = []
    try:
        for delay_steps in selected_delays:
            evaluations.append(
                _evaluate_delay(
                    env=env,
                    snapshot=snapshot,
                    history_items=history_items,
                    history_rewards=history_rewards,
                    cached_suffix=cached_suffix,
                    delay_steps=delay_steps,
                    planner=planner,
                    downstream_planner=downstream_planner,
                )
            )
    finally:
        restore_environment(env, snapshot)
    return BranchPointResult(
        planned_steps=planned_steps,
        evaluations=tuple(evaluations),
    )
