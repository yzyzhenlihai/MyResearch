from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.common import minmax_normalize_rows, stable_argsort_desc


def compute_cost_analysis(
    pred_score_mat: np.ndarray,
    decision_cost_reward_mat: np.ndarray,
    risk_signal_mats: Dict[str, np.ndarray],
    item_raw_ids: np.ndarray,
    config: Dict[str, Any],
    rollout_payload: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    topk = min(int(config.get("topk", 20)), pred_score_mat.shape[1])
    lambdas = [float(x) for x in config.get("cost_lambdas", [0.0, 0.1, 0.2, 0.5, 1.0])]
    compare_modes = config.get("cost_compare_modes", [config.get("cost_mode", "soft_penalty")])
    compare_modes = [str(mode) for mode in compare_modes]
    budget_quantiles = [float(x) for x in config.get("cost_budget_quantiles", [0.05, 0.10, 0.20])]
    exposure_weights = build_exposure_weights(topk, scheme=str(config.get("exposure_weight_scheme", "log_discount")))

    rows: List[Dict[str, Any]] = []
    base_result = evaluate_one_signal(
        signal_name="baseline",
        pred_score_mat=pred_score_mat,
        decision_cost_reward_mat=decision_cost_reward_mat,
        risk_norm_mat=np.zeros_like(pred_score_mat, dtype=np.float32),
        risk_apply_mask=np.zeros_like(pred_score_mat, dtype=bool),
        item_raw_ids=item_raw_ids,
        topk=topk,
        penalty_lambda=0.0,
        cost_mode="soft_penalty",
        exposure_weights=exposure_weights,
        rollout_payload=rollout_payload,
        budget_quantile=0.0,
    )
    rows.append(base_result)

    for signal_name, risk_mat in risk_signal_mats.items():
        risk_norm = minmax_normalize_rows(np.asarray(risk_mat, dtype=np.float32))
        for budget_quantile in budget_quantiles:
            risk_apply_mask, masked_rate_mean = build_top_risk_mask(risk_norm, budget_quantile)
            if "soft_penalty" in compare_modes:
                for penalty_lambda in lambdas:
                    rows.append(
                        evaluate_one_signal(
                            signal_name=signal_name,
                            pred_score_mat=pred_score_mat,
                            decision_cost_reward_mat=decision_cost_reward_mat,
                            risk_norm_mat=risk_norm,
                            risk_apply_mask=risk_apply_mask,
                            item_raw_ids=item_raw_ids,
                            topk=topk,
                            penalty_lambda=float(penalty_lambda),
                            cost_mode="soft_penalty",
                            exposure_weights=exposure_weights,
                            rollout_payload=rollout_payload,
                            budget_quantile=budget_quantile,
                            masked_rate_mean=masked_rate_mean,
                        )
                    )
            if "hard_mask" in compare_modes:
                rows.append(
                    evaluate_one_signal(
                        signal_name=signal_name,
                        pred_score_mat=pred_score_mat,
                        decision_cost_reward_mat=decision_cost_reward_mat,
                        risk_norm_mat=risk_norm,
                        risk_apply_mask=risk_apply_mask,
                        item_raw_ids=item_raw_ids,
                        topk=topk,
                        penalty_lambda=1.0,
                        cost_mode="hard_mask",
                        exposure_weights=exposure_weights,
                        rollout_payload=rollout_payload,
                        budget_quantile=budget_quantile,
                        masked_rate_mean=masked_rate_mean,
                    )
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["is_best_for_source"] = False
    for signal_name, sub in df.groupby("signal_name", observed=False):
        best_idx = sub["topk_oracle_reward_loss_mean"].idxmin()
        df.loc[best_idx, "is_best_for_source"] = True
    return df.sort_values(["signal_name", "budget_quantile", "cost_mode", "penalty_lambda"]).reset_index(drop=True)


def build_top_risk_mask(risk_norm_mat: np.ndarray, budget_quantile: float) -> Tuple[np.ndarray, float]:
    if budget_quantile <= 0:
        mask = np.zeros_like(risk_norm_mat, dtype=bool)
        return mask, 0.0
    budget_quantile = min(max(budget_quantile, 0.0), 1.0)
    thresholds = np.quantile(risk_norm_mat, 1.0 - budget_quantile, axis=1, keepdims=True)
    mask = risk_norm_mat >= thresholds
    return mask.astype(bool), float(mask.mean())


def build_exposure_weights(topk: int, scheme: str = "log_discount") -> np.ndarray:
    if scheme == "uniform":
        return np.ones(topk, dtype=np.float32)
    if scheme == "log_discount":
        denom = np.log2(np.arange(2, topk + 2))
        return (1.0 / denom).astype(np.float32)
    raise ValueError(f"Unsupported exposure weight scheme: {scheme}")


def evaluate_one_signal(
    signal_name: str,
    pred_score_mat: np.ndarray,
    decision_cost_reward_mat: np.ndarray,
    risk_norm_mat: np.ndarray,
    risk_apply_mask: np.ndarray,
    item_raw_ids: np.ndarray,
    topk: int,
    penalty_lambda: float,
    cost_mode: str,
    exposure_weights: np.ndarray,
    rollout_payload: Optional[Dict[str, Any]],
    budget_quantile: float,
    masked_rate_mean: float = 0.0,
) -> Dict[str, Any]:
    oracle_losses: List[float] = []
    exposure_costs: List[float] = []
    adjusted_rewards: List[float] = []
    oracle_rewards: List[float] = []
    rollout_returns: List[float] = []

    for u in range(pred_score_mat.shape[0]):
        pred_scores = pred_score_mat[u]
        oracle_scores = decision_cost_reward_mat[u]
        risk = risk_norm_mat[u]
        risk_mask = risk_apply_mask[u]

        adjusted_scores = apply_penalty(pred_scores, risk, risk_mask, penalty_lambda, cost_mode)
        oracle_order = stable_argsort_desc(oracle_scores, item_raw_ids)
        adjusted_order = stable_argsort_desc(adjusted_scores, item_raw_ids)

        oracle_topk = oracle_order[:topk]
        adjusted_topk = adjusted_order[:topk]
        best_reward = float(oracle_scores[oracle_topk].sum())
        adjusted_reward = float(oracle_scores[adjusted_topk].sum())
        oracle_rewards.append(best_reward)
        adjusted_rewards.append(adjusted_reward)
        oracle_losses.append(best_reward - adjusted_reward)

        exposure_cost = np.sum(exposure_weights * (oracle_scores[oracle_topk] - oracle_scores[adjusted_topk]))
        exposure_costs.append(float(exposure_cost))

        if rollout_payload is not None and rollout_payload.get("enabled", False):
            rollout_returns.append(
                simulate_static_ranking_return(
                    oracle_reward_mat=rollout_payload["oracle_reward_mat"],
                    list_feat_small=rollout_payload["list_feat_small"],
                    ranking=adjusted_order,
                    leave_threshold=int(rollout_payload["leave_threshold"]),
                    num_leave_compute=int(rollout_payload["num_leave_compute"]),
                    max_turn=int(rollout_payload["max_turn"]),
                    user_index=u,
                )
            )

    return {
        "signal_name": signal_name,
        "penalty_lambda": float(penalty_lambda),
        "cost_mode": cost_mode,
        "budget_quantile": float(budget_quantile),
        "budget_label": f"top_{int(round(budget_quantile * 100)):02d}pct" if budget_quantile > 0 else "baseline",
        "masked_rate_mean": float(masked_rate_mean),
        "topk_oracle_reward_mean": float(np.mean(adjusted_rewards)),
        "oracle_topk_reward_mean": float(np.mean(oracle_rewards)),
        "topk_oracle_reward_loss_mean": float(np.mean(oracle_losses)),
        "topk_oracle_reward_loss_std": float(np.std(oracle_losses)),
        "exposure_weighted_cost_mean": float(np.mean(exposure_costs)),
        "exposure_weighted_cost_std": float(np.std(exposure_costs)),
        "rollout_return_mean": float(np.mean(rollout_returns)) if rollout_returns else float("nan"),
        "n_users": int(pred_score_mat.shape[0]),
    }


def apply_penalty(
    pred_scores: np.ndarray,
    risk: np.ndarray,
    risk_mask: np.ndarray,
    penalty_lambda: float,
    cost_mode: str,
) -> np.ndarray:
    if cost_mode == "soft_penalty":
        applied_risk = risk * risk_mask.astype(np.float32)
        return pred_scores - penalty_lambda * applied_risk
    if cost_mode == "hard_mask":
        adjusted = pred_scores.copy()
        adjusted[risk_mask] = adjusted.min() - penalty_lambda - 1.0
        return adjusted
    raise ValueError(f"Unsupported cost_mode: {cost_mode}")


def simulate_static_ranking_return(
    oracle_reward_mat: np.ndarray,
    list_feat_small: List[List[int]],
    ranking: np.ndarray,
    leave_threshold: int,
    num_leave_compute: int,
    max_turn: int,
    user_index: int,
) -> float:
    history: List[int] = []
    total_reward = 0.0
    for t, action in enumerate(ranking[:max_turn]):
        terminated = False
        if t > 0:
            window_actions = history[max(0, t - num_leave_compute):t]
            hist_counts: Dict[int, int] = {}
            for act in window_actions:
                for feat in list_feat_small[int(act)]:
                    hist_counts[feat] = hist_counts.get(feat, 0) + 1
            for feat in list_feat_small[int(action)]:
                if hist_counts.get(feat, 0) > leave_threshold:
                    terminated = True
                    break
        if t >= max_turn - 1:
            terminated = True
        history.append(int(action))
        total_reward += float(oracle_reward_mat[user_index, int(action)])
        if terminated:
            break
    return total_reward
