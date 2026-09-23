"""真实会话指标与冻结当前策略的训练域 MC 校准。"""

from collections import Counter

import numpy as np
import torch

from examples.our_model.rave.state import COMPONENTS


@torch.no_grad()
def monte_carlo_calibration(policy, simulator, records, snapshot_version):
    """逐 h 执行指定前缀后交回同一策略，返回真实 MC 误差和标签。

    Args:
        policy (CandidatePolicy): 校准期间保持冻结的当前策略。
        simulator (TrainingSimulator): 仅训练域，可访问原始训练偏好。
        records (Sequence[PrefixRollout]): 少量训练起点和候选。
        snapshot_version (str): 标签对应的策略版本。

    Returns:
        dict: 每个 h/分量的误差及含起点 F 的 MC 标签；不复用为新策略的无偏标签。
    """
    errors, samples = {}, []
    was_training = policy.network.training
    policy.network.eval()
    try:
        for record in records:
            inputs = policy.inputs([record.start])
            candidate = record.candidates.to(inputs["states"].device)
            prediction = policy.network(inputs, candidate[None, None])[0, 0]
            limit = min(len(candidate), record.start.budget,
                        int((candidate >= 0).sum()))
            for horizon in range(1, limit + 1):
                state = record.start.clone()
                returns = torch.zeros(len(COMPONENTS))
                for step, item in enumerate(candidate[:horizon].tolist()):
                    if state.done:
                        break
                    returns += simulator.step(state, item, step == 0)
                while not state.done:
                    selected, horizons, _ = policy.choose([state])
                    for step, item in enumerate(selected[0, :int(horizons[0])].tolist()):
                        if state.done:
                            break
                        returns += simulator.step(state, item, step == 0)
                # 单头 checkpoint 的唯一校准标签是原始反馈回报 R。
                target = returns[:1] if policy.network.scalar else returns
                names = ("R",) if policy.network.scalar else COMPONENTS
                for component, name in enumerate(names):
                    key = f"mc_mae/h{horizon}/{name}"
                    errors.setdefault(key, []).append(
                        abs(float(prediction[horizon - 1, component]) - float(target[component])))
                samples.append({"horizon": horizon, "user_id": state.user_id,
                                "start_count": len(record.start.items),
                                "start_fraction": record.start.fraction,
                                "end_fraction": state.fraction,
                                "returns": returns.tolist(),
                                "prediction": prediction[horizon - 1].cpu().tolist(),
                                "policy_version": snapshot_version})
    finally:
        policy.network.train(was_training)
    return {"errors": {key: float(np.mean(value)) for key, value in errors.items()},
            "samples": samples, "policy_version": snapshot_version}


def episode_metrics(sessions, category_ids, plans, latencies, horizons,
                    recent_window=20, matched_length=5, state_update_latencies=()):
    """汇总真实完整会话，区分 pooled/macro MCD 和缺失类别曝光。

    Args:
        sessions (Sequence[Session]): 已完成、非空会话。
        category_ids (array): 内部 item 的类别表，0 为缺失/padding。
        plans (int): 实际规划事件数。
        latencies (Sequence[float]): 秒为单位的单步决策耗时。
        horizons (Sequence[int]): 实际选择 H。
        recent_window (int): 重复率及相邻推荐 Jaccard 的历史窗口。
        matched_length (int): 覆盖率对照使用的共同展示长度。
        state_update_latencies (Sequence[float]): 展示后 dynamics 状态维护耗时。

    Returns:
        dict: 未加展示修正的原始指标；无足够样本的指标为 None。
    """
    if not sessions or any(not s.done or not s.items for s in sessions):
        raise ValueError("Metrics require complete, nonempty episodes")
    if min(recent_window, matched_length) <= 0:
        raise ValueError("Diagnostic windows must be positive")
    categories = [set(int(v) for v in row if v > 0) for row in category_ids]
    repeated, missing, jaccard = 0, 0, []
    coverage, matched = [], []
    for session in sessions:
        exposed = set()
        for position, item in enumerate(session.items):
            previous = session.items[max(0, position - recent_window):position]
            repeated += int(item in previous)
            missing += int(not categories[item])
            for earlier in previous:
                union = categories[item] | categories[earlier]
                if union:
                    jaccard.append(1 - len(categories[item] & categories[earlier]) / len(union))
            exposed.update(categories[item])
        coverage.append(len(exposed))
        if len(session.items) >= matched_length:
            matched.append(len(set().union(*(categories[item] for item in session.items[:matched_length]))))
    lengths = np.asarray([len(s.items) for s in sessions])
    counts = np.asarray([s.dominant_count for s in sessions])
    totals = np.asarray([sum(s.feedback) for s in sessions])
    displays = int(lengths.sum())
    return {
        "episodes": len(sessions), "displays": displays,
        "raw_return_mean": float(totals.mean()),
        "raw_feedback_per_display": float(totals.sum() / displays),
        "length_mean": float(lengths.mean()),
        "natural_exit_rate": float(np.mean([s.natural_done for s in sessions])),
        "budget_reached_rate": float(np.mean([s.budget_done for s in sessions])),
        "mcd_pooled": float(counts.sum() / displays),
        "mcd_macro": float(np.mean(counts / lengths)),
        "missing_category_exposure_rate": missing / displays,
        "recent_repeat_rate": repeated / displays,
        "window_jaccard_difference": float(np.mean(jaccard)) if jaccard else None,
        "category_coverage_mean": float(np.mean(coverage)),
        "matched_category_coverage_mean": float(np.mean(matched)) if matched else None,
        "matched_length": matched_length, "matched_episodes": len(matched),
        "planning_events": plans, "plans_per_100_displays": 100 * plans / displays,
        "decision_seconds": float(sum(latencies)),
        "state_update_seconds": float(sum(state_update_latencies)),
        "inference_seconds": float(sum(latencies) + sum(state_update_latencies)),
        "latency_p50_seconds": float(np.quantile(latencies, 0.50)) if latencies else None,
        "latency_p95_seconds": float(np.quantile(latencies, 0.95)) if latencies else None,
        "selected_horizons": dict(sorted(Counter(horizons).items())),
    }
