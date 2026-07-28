"""原 chunk 失效与延迟重规划代价的配对反事实评估。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Sequence


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
    """一个重规划延迟分支的固定窗口结果。

    Attributes:
        delay_steps (int): 重新规划前额外执行的缓存动作数。
        reward_sum (float): 固定计划窗口内获得的奖励总和。
        executed_steps (int): 因自然退出可能小于计划窗口的实际步数。
        terminated (bool): 分支是否在窗口结束前自然退出。
    """

    delay_steps: int
    reward_sum: float
    executed_steps: int
    terminated: bool


@dataclass(frozen=True)
class BranchPointResult:
    """同一中间状态上全部重规划延迟分支的配对结果。

    Attributes:
        planned_steps (int): 原 chunk 剩余动作数，也是所有分支的计划窗口。
        evaluations (tuple[DelayEvaluation, ...]): 按延迟从小到大排列的结果。
    """

    planned_steps: int
    evaluations: tuple[DelayEvaluation, ...]

    @property
    def immediate_reward(self) -> float:
        """返回立即重规划分支的奖励。

        Returns:
            float: `delay_steps=0` 分支的奖励总和。

        Raises:
            RuntimeError: 当结果中缺少立即重规划分支时抛出。
        """

        for evaluation in self.evaluations:
            if evaluation.delay_steps == 0:
                return evaluation.reward_sum
        raise RuntimeError("BranchPointResult is missing delay_steps=0.")

    @property
    def continue_reward(self) -> float:
        """返回完整执行原 chunk 剩余动作的奖励。

        Returns:
            float: `delay_steps=planned_steps` 分支的奖励总和。

        Raises:
            RuntimeError: 当结果中缺少完整继续分支时抛出。
        """

        for evaluation in self.evaluations:
            if evaluation.delay_steps == self.planned_steps:
                return evaluation.reward_sum
        raise RuntimeError("BranchPointResult is missing the full-continue branch.")

    @property
    def replanning_gain(self) -> float:
        """计算每个计划步的立即重规划收益。

        未执行步仍保留在分母中，避免自然退出分支通过缩短实际轨迹获得
        人为偏高的单步奖励。

        Returns:
            float: `(立即重规划奖励 - 完整继续奖励) / 计划窗口长度`。
        """

        return (self.immediate_reward - self.continue_reward) / self.planned_steps


Planner = Callable[[Sequence[int], Sequence[float]], Sequence[int]]


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
        tuple[float, bool]: 当前步奖励与自然终止标记。
    """

    _, reward, terminated, truncated, _ = env.step(int(action))
    reward_float = float(reward)
    history_items.append(int(action))
    history_rewards.append(reward_float)
    return reward_float, bool(terminated or truncated)


def _evaluate_delay(
    env: Any,
    snapshot: EnvironmentSnapshot,
    history_items: Sequence[int],
    history_rewards: Sequence[float],
    cached_suffix: Sequence[int],
    delay_steps: int,
    planner: Planner,
) -> DelayEvaluation:
    """评估“延迟若干缓存动作后再重规划”的单个分支。

    Args:
        env (Any): 可恢复状态的推荐环境。
        snapshot (EnvironmentSnapshot): 分支共同起点。
        history_items (Sequence[int]): 起点时策略看到的完整物品历史。
        history_rewards (Sequence[float]): 起点时策略看到的完整奖励历史。
        cached_suffix (Sequence[int]): 原 chunk 尚未执行的缓存动作。
        delay_steps (int): 重规划前继续执行的缓存动作数，范围
            `[0, len(cached_suffix)]`。
        planner (Planner): 根据最新历史生成新 chunk 的回调。

    Returns:
        DelayEvaluation: 固定计划窗口内的分支结果。

    Raises:
        ValueError: 当历史长度不一致、缓存后缀为空、延迟越界或 planner
            返回的新动作不足时抛出。
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
    reward_sum = 0.0
    executed_steps = 0
    terminated = False

    # 先执行指定数量的旧缓存动作；一旦自然退出，该分支不再重规划。
    for action in cached_suffix[:delay_steps]:
        reward, terminated = _step_branch(
            env,
            int(action),
            branch_items,
            branch_rewards,
        )
        reward_sum += reward
        executed_steps += 1
        if terminated:
            break

    remaining_steps = planned_steps - executed_steps
    if not terminated and remaining_steps > 0:
        replanned_actions = list(planner(branch_items, branch_rewards))
        if len(replanned_actions) < remaining_steps:
            raise ValueError(
                "planner returned fewer actions than the remaining evaluation window: "
                f"needed={remaining_steps}, got={len(replanned_actions)}."
            )
        for action in replanned_actions[:remaining_steps]:
            reward, terminated = _step_branch(
                env,
                int(action),
                branch_items,
                branch_rewards,
            )
            reward_sum += reward
            executed_steps += 1
            if terminated:
                break

    return DelayEvaluation(
        delay_steps=delay_steps,
        reward_sum=reward_sum,
        executed_steps=executed_steps,
        terminated=terminated,
    )


def evaluate_branch_point(
    env: Any,
    history_items: Sequence[int],
    history_rewards: Sequence[float],
    cached_suffix: Sequence[int],
    planner: Planner,
) -> BranchPointResult:
    """在同一中间状态上评估从立即重规划到完整继续的全部延迟。

    每个 delay 分支都从相同环境快照开始，且总计划窗口始终等于原
    chunk 的剩余长度。函数结束后会恢复调用前的主轨迹环境状态。

    Args:
        env (Any): 当前主轨迹环境。
        history_items (Sequence[int]): 当前策略物品历史，包含 reset dummy。
        history_rewards (Sequence[float]): 当前策略奖励历史。
        cached_suffix (Sequence[int]): 原 chunk 未执行的动作。
        planner (Planner): 根据分支最新历史生成动作序列的回调。

    Returns:
        BranchPointResult: 全部延迟分支的配对结果。

    Raises:
        ValueError: 当缓存后缀为空或底层分支输入不合法时抛出。

    Example:
        >>> class ToyEnv:
        ...     pass
        >>> # 实际使用时传入 KuaiEnv 和 DORL-MAC planner。
    """

    if not cached_suffix:
        raise ValueError("cached_suffix must contain at least one action.")
    snapshot = capture_environment(env)
    evaluations: list[DelayEvaluation] = []
    try:
        for delay_steps in range(len(cached_suffix) + 1):
            evaluations.append(
                _evaluate_delay(
                    env=env,
                    snapshot=snapshot,
                    history_items=history_items,
                    history_rewards=history_rewards,
                    cached_suffix=cached_suffix,
                    delay_steps=delay_steps,
                    planner=planner,
                )
            )
    finally:
        restore_environment(env, snapshot)
    return BranchPointResult(
        planned_steps=len(cached_suffix),
        evaluations=tuple(evaluations),
    )
