from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.common import (
    bucketize_by_edges,
    quantile_edges,
    ranks_from_order,
    safe_average_precision,
    safe_roc_auc,
    safe_spearman,
    sanitize_column_token,
    split_matrix_regions,
    stable_argsort_desc,
)
from analysis.compute_support_score import SUPPORT_LABELS

try:
    from numba import njit
except Exception:  # pragma: no cover - fallback path
    njit = None


MARGIN_LABELS = np.asarray(["small", "medium", "large"])


if njit is not None:
    @njit
    def _fenwick_update(tree: np.ndarray, idx: int, delta: int) -> None:
        while idx < tree.shape[0]:
            tree[idx] += delta
            idx += idx & -idx


    @njit
    def _fenwick_query(tree: np.ndarray, idx: int) -> int:
        res = 0
        while idx > 0:
            res += tree[idx]
            idx -= idx & -idx
        return res


    @njit
    def _item_inversion_counts_numba(oracle_pos_in_pred_order: np.ndarray) -> np.ndarray:
        n = oracle_pos_in_pred_order.shape[0]
        left = np.zeros(n, dtype=np.int64)
        right = np.zeros(n, dtype=np.int64)
        tree = np.zeros(n + 2, dtype=np.int64)

        for i in range(n):
            rank = int(oracle_pos_in_pred_order[i]) + 1
            left[i] = i - _fenwick_query(tree, rank)
            _fenwick_update(tree, rank, 1)

        tree[:] = 0
        for i in range(n - 1, -1, -1):
            rank = int(oracle_pos_in_pred_order[i]) + 1
            right[i] = _fenwick_query(tree, rank - 1)
            _fenwick_update(tree, rank, 1)

        return left + right
else:
    def _item_inversion_counts_numba(oracle_pos_in_pred_order: np.ndarray) -> np.ndarray:
        return _item_inversion_counts_python(oracle_pos_in_pred_order)


def _item_inversion_counts_python(oracle_pos_in_pred_order: np.ndarray) -> np.ndarray:
    n = len(oracle_pos_in_pred_order)
    left = np.zeros(n, dtype=np.int64)
    right = np.zeros(n, dtype=np.int64)
    tree = np.zeros(n + 2, dtype=np.int64)

    def update(idx: int, delta: int) -> None:
        while idx < len(tree):
            tree[idx] += delta
            idx += idx & -idx

    def query(idx: int) -> int:
        total = 0
        while idx > 0:
            total += tree[idx]
            idx -= idx & -idx
        return total

    for i, rank0 in enumerate(oracle_pos_in_pred_order):
        rank = int(rank0) + 1
        left[i] = i - query(rank)
        update(rank, 1)

    tree[:] = 0
    for i in range(n - 1, -1, -1):
        rank = int(oracle_pos_in_pred_order[i]) + 1
        right[i] = query(rank - 1)
        update(rank, 1)
    return left + right


def compute_ranking_metric_matrices(
    pred_score_mat: np.ndarray,
    oracle_rank_score_mat: np.ndarray,
    item_raw_ids: np.ndarray,
    topk: int,
    shortlist_topl: int,
) -> Dict[str, np.ndarray]:
    n_user, n_item = pred_score_mat.shape
    kk = min(int(topk), n_item)
    ll = min(int(shortlist_topl), n_item)

    pred_rank_mat = np.zeros((n_user, n_item), dtype=np.int32)
    oracle_rank_mat = np.zeros((n_user, n_item), dtype=np.int32)
    margin_mat = np.zeros((n_user, n_item), dtype=np.float32)
    topk_flip_mat = np.zeros((n_user, n_item), dtype=np.uint8)
    global_pairwise_flip_mat = np.zeros((n_user, n_item), dtype=np.float32)
    local_pairwise_flip_mat = np.zeros((n_user, n_item), dtype=np.float32)
    local_pairwise_flip_raw_mat = np.zeros((n_user, n_item), dtype=np.float32)
    boundary_crossing_score_mat = np.zeros((n_user, n_item), dtype=np.float32)
    boundary_weight_mat = np.zeros((n_user, n_item), dtype=np.float32)
    shortlist_mask_mat = np.zeros((n_user, n_item), dtype=bool)
    pred_topk_mask_mat = np.zeros((n_user, n_item), dtype=bool)
    oracle_topk_mask_mat = np.zeros((n_user, n_item), dtype=bool)

    for u in range(n_user):
        pred_scores = pred_score_mat[u]
        oracle_scores = oracle_rank_score_mat[u]
        pred_order = stable_argsort_desc(pred_scores, item_raw_ids)
        oracle_order = stable_argsort_desc(oracle_scores, item_raw_ids)

        pred_ranks = ranks_from_order(pred_order)
        oracle_ranks = ranks_from_order(oracle_order)
        pred_rank_mat[u] = pred_ranks
        oracle_rank_mat[u] = oracle_ranks

        cutoff_item = pred_order[kk - 1]
        cutoff_score = pred_scores[cutoff_item]
        margin = np.abs(pred_scores - cutoff_score).astype(np.float32)
        margin_mat[u] = margin

        pred_topk = np.zeros(n_item, dtype=bool)
        oracle_topk = np.zeros(n_item, dtype=bool)
        pred_topk[pred_order[:kk]] = True
        oracle_topk[oracle_order[:kk]] = True
        pred_topk_mask_mat[u] = pred_topk
        oracle_topk_mask_mat[u] = oracle_topk
        topk_flip = np.logical_xor(pred_topk, oracle_topk)
        topk_flip_mat[u] = topk_flip.astype(np.uint8)

        oracle_pos_in_pred_order = oracle_ranks[pred_order] - 1
        counts_pred_order = _item_inversion_counts_numba(oracle_pos_in_pred_order.astype(np.int32))
        counts_in_item_order = np.zeros(n_item, dtype=np.float32)
        counts_in_item_order[pred_order] = counts_pred_order.astype(np.float32)
        denom = max(n_item - 1, 1)
        global_pairwise_flip_mat[u] = counts_in_item_order / float(denom)

        shortlist_mask = np.zeros(n_item, dtype=bool)
        shortlist_mask[pred_order[:ll]] = True
        shortlist_mask[oracle_order[:ll]] = True
        shortlist_mask_mat[u] = shortlist_mask
        shortlist_items = pred_order[shortlist_mask[pred_order]]

        if len(shortlist_items) > 0:
            shortlist_margin = margin[shortlist_items]
            boundary_scale = float(np.median(shortlist_margin))
            boundary_scale = max(boundary_scale, 1e-8)
            boundary_weight = np.exp(-margin / boundary_scale).astype(np.float32)
            boundary_weight_mat[u] = boundary_weight
            boundary_crossing_score_mat[u] = (topk_flip.astype(np.float32) * boundary_weight).astype(np.float32)

            if len(shortlist_items) > 1:
                oracle_short_order_local = stable_argsort_desc(
                    oracle_scores[shortlist_items],
                    item_raw_ids[shortlist_items],
                )
                oracle_short_ranks_local = ranks_from_order(oracle_short_order_local)
                oracle_pos_in_pred_short = oracle_short_ranks_local - 1
                local_counts = _item_inversion_counts_numba(oracle_pos_in_pred_short.astype(np.int32)).astype(np.float32)
                local_ratio = local_counts / float(max(len(shortlist_items) - 1, 1))
                weighted_local_ratio = local_ratio * boundary_weight[shortlist_items]
                local_pairwise_flip_raw_mat[u, shortlist_items] = local_ratio.astype(np.float32)
                local_pairwise_flip_mat[u, shortlist_items] = weighted_local_ratio.astype(np.float32)

    return {
        "pred_rank_mat": pred_rank_mat,
        "oracle_rank_mat": oracle_rank_mat,
        "margin_mat": margin_mat,
        "topk_flip_mat": topk_flip_mat,
        "global_pairwise_flip_mat": global_pairwise_flip_mat,
        "local_pairwise_flip_mat": local_pairwise_flip_mat,
        "local_pairwise_flip_raw_mat": local_pairwise_flip_raw_mat,
        "boundary_crossing_score_mat": boundary_crossing_score_mat,
        "boundary_weight_mat": boundary_weight_mat,
        "shortlist_mask_mat": shortlist_mask_mat,
        "pred_topk_mask_mat": pred_topk_mask_mat,
        "oracle_topk_mask_mat": oracle_topk_mask_mat,
    }


def compute_margin_region_payload(
    margin_mat: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    quantiles = config.get("margin_quantiles", [0.33, 0.66])
    codes_by_mode: Dict[str, np.ndarray] = {}
    stats_by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in ["global", "per_user"]:
        codes, stats = split_matrix_regions(
            mat=margin_mat,
            quantiles=quantiles,
            mode=mode,
            labels=MARGIN_LABELS.tolist(),
        )
        codes_by_mode[mode] = codes
        stats_by_mode[mode] = stats
    active_mode = str(config.get("margin_quantile_mode", "global"))
    if active_mode not in codes_by_mode:
        active_mode = "global"
    return {
        "active_mode": active_mode,
        "codes_by_mode": codes_by_mode,
        "stats_by_mode": stats_by_mode,
        "active_codes": codes_by_mode[active_mode],
    }


def build_primary_slice_mask(
    ranking_metrics: Mapping[str, np.ndarray],
    config: Dict[str, Any],
) -> Tuple[str, np.ndarray]:
    slice_name = str(config.get("evaluation_slice", "all_items"))
    if slice_name == "all_items":
        return slice_name, np.ones_like(ranking_metrics["shortlist_mask_mat"], dtype=bool)
    if slice_name == "shortlist_union":
        return slice_name, np.asarray(ranking_metrics["shortlist_mask_mat"], dtype=bool)
    raise ValueError(f"Unsupported evaluation_slice: {slice_name}")


def build_sample_metrics_frame(
    user_raw_ids: np.ndarray,
    item_raw_ids: np.ndarray,
    pred_score_mat: np.ndarray,
    uncertainty_mats: Mapping[str, np.ndarray],
    primary_uncertainty_source: str,
    oracle_norm_mat: np.ndarray,
    oracle_raw_mat: np.ndarray,
    error_oracle_mat: np.ndarray,
    support_score_mat: np.ndarray,
    support_region_codes_by_mode: Mapping[str, np.ndarray],
    margin_region_codes_by_mode: Mapping[str, np.ndarray],
    primary_slice_mask: np.ndarray,
    ranking_metrics: Mapping[str, np.ndarray],
    row_metadata: Optional[Mapping[str, np.ndarray]] = None,
) -> pd.DataFrame:
    n_user, n_item = pred_score_mat.shape
    flat_user = np.repeat(user_raw_ids.astype(np.int64), n_item)
    flat_item = np.tile(item_raw_ids.astype(np.int64), n_user)
    primary_source_token = sanitize_column_token(primary_uncertainty_source)

    data: Dict[str, Any] = {
        "user_id": flat_user,
        "item_id": flat_item,
        "pred_score": pred_score_mat.reshape(-1).astype(np.float32),
        "oracle_norm": oracle_norm_mat.reshape(-1).astype(np.float32),
        "oracle_raw": oracle_raw_mat.reshape(-1).astype(np.float32),
        "pointwise_error": np.abs(pred_score_mat - error_oracle_mat).reshape(-1).astype(np.float32),
        "support_score": support_score_mat.reshape(-1).astype(np.float32),
        "pred_rank": ranking_metrics["pred_rank_mat"].reshape(-1).astype(np.int32),
        "oracle_rank": ranking_metrics["oracle_rank_mat"].reshape(-1).astype(np.int32),
        "margin": ranking_metrics["margin_mat"].reshape(-1).astype(np.float32),
        "topk_flip": ranking_metrics["topk_flip_mat"].reshape(-1).astype(np.uint8),
        "global_pairwise_flip": ranking_metrics["global_pairwise_flip_mat"].reshape(-1).astype(np.float32),
        "local_pairwise_flip": ranking_metrics["local_pairwise_flip_mat"].reshape(-1).astype(np.float32),
        "local_pairwise_flip_raw": ranking_metrics["local_pairwise_flip_raw_mat"].reshape(-1).astype(np.float32),
        "boundary_crossing_score": ranking_metrics["boundary_crossing_score_mat"].reshape(-1).astype(np.float32),
        "boundary_weight": ranking_metrics["boundary_weight_mat"].reshape(-1).astype(np.float32),
        "in_shortlist": ranking_metrics["shortlist_mask_mat"].reshape(-1).astype(bool),
        "in_primary_slice": primary_slice_mask.reshape(-1).astype(bool),
        "is_pred_topk": ranking_metrics["pred_topk_mask_mat"].reshape(-1).astype(bool),
        "is_oracle_topk": ranking_metrics["oracle_topk_mask_mat"].reshape(-1).astype(bool),
    }

    for source, mat in uncertainty_mats.items():
        token = sanitize_column_token(source)
        data[f"uncertainty_{token}"] = np.asarray(mat, dtype=np.float32).reshape(-1)
    data["uncertainty"] = data[f"uncertainty_{primary_source_token}"]

    for mode, codes in support_region_codes_by_mode.items():
        labels = SUPPORT_LABELS[codes.reshape(-1)]
        data[f"support_region_{mode}"] = pd.Categorical(
            labels,
            categories=SUPPORT_LABELS.tolist(),
            ordered=True,
        )
    data["support_region"] = data[f"support_region_per_user"] if "support_region_per_user" in data else data["support_region_global"]

    for mode, codes in margin_region_codes_by_mode.items():
        labels = MARGIN_LABELS[codes.reshape(-1)]
        data[f"margin_region_{mode}"] = pd.Categorical(
            labels,
            categories=MARGIN_LABELS.tolist(),
            ordered=True,
        )
    data["margin_region"] = data[f"margin_region_per_user"] if "margin_region_per_user" in data else data["margin_region_global"]

    if row_metadata is not None:
        for key, values in row_metadata.items():
            value_arr = np.asarray(values)
            if value_arr.shape[0] != n_user:
                raise ValueError(f"Row metadata {key} has incompatible length {value_arr.shape[0]} (expected {n_user})")
            if value_arr.dtype.kind in {"U", "S", "O"}:
                repeated = np.repeat(value_arr.astype(object), n_item)
            else:
                repeated = np.repeat(value_arr, n_item)
            data[key] = repeated

    return pd.DataFrame(data)


def filter_sample_df(sample_df: pd.DataFrame, mask_col: str = "in_primary_slice") -> pd.DataFrame:
    if mask_col not in sample_df.columns:
        return sample_df.copy()
    return sample_df[sample_df[mask_col].to_numpy().astype(bool)].copy()


def activate_region_columns(
    sample_df: pd.DataFrame,
    support_mode: str,
    margin_mode: str,
) -> pd.DataFrame:
    work_df = sample_df.copy()
    support_col = f"support_region_{support_mode}"
    margin_col = f"margin_region_{margin_mode}"
    if support_col in work_df.columns:
        work_df["support_region"] = work_df[support_col]
    if margin_col in work_df.columns:
        work_df["margin_region"] = work_df[margin_col]
    return work_df


def compute_correlation_summary(sample_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    high_error_q = float(config.get("high_error_quantile", 0.8))
    high_flip_q = float(config.get("high_flip_quantile", 0.8))
    thresholds = {
        "pointwise_error": float(np.quantile(sample_df["pointwise_error"], high_error_q)),
        "local_pairwise_flip": float(np.quantile(sample_df["local_pairwise_flip"], high_flip_q)),
        "global_pairwise_flip": float(np.quantile(sample_df["global_pairwise_flip"], high_flip_q)),
        "boundary_crossing_score": float(np.quantile(sample_df["boundary_crossing_score"], high_flip_q)),
    }

    rows = [
        _make_target_summary_row(
            sample_df=sample_df,
            target_col="pointwise_error",
            positive_mask=(sample_df["pointwise_error"].to_numpy() >= thresholds["pointwise_error"]).astype(np.uint8),
            threshold=thresholds["pointwise_error"],
            positive_rule=f">={high_error_q:.2f} quantile",
        ),
        _make_target_summary_row(
            sample_df=sample_df,
            target_col="topk_flip",
            positive_mask=sample_df["topk_flip"].to_numpy().astype(np.uint8),
            threshold=1.0,
            positive_rule="binary topk flip",
        ),
        _make_target_summary_row(
            sample_df=sample_df,
            target_col="local_pairwise_flip",
            positive_mask=(sample_df["local_pairwise_flip"].to_numpy() >= thresholds["local_pairwise_flip"]).astype(np.uint8),
            threshold=thresholds["local_pairwise_flip"],
            positive_rule=f">={high_flip_q:.2f} quantile",
        ),
        _make_target_summary_row(
            sample_df=sample_df,
            target_col="boundary_crossing_score",
            positive_mask=(sample_df["boundary_crossing_score"].to_numpy() >= thresholds["boundary_crossing_score"]).astype(np.uint8),
            threshold=thresholds["boundary_crossing_score"],
            positive_rule=f">={high_flip_q:.2f} quantile",
        ),
        _make_target_summary_row(
            sample_df=sample_df,
            target_col="global_pairwise_flip",
            positive_mask=(sample_df["global_pairwise_flip"].to_numpy() >= thresholds["global_pairwise_flip"]).astype(np.uint8),
            threshold=thresholds["global_pairwise_flip"],
            positive_rule=f">={high_flip_q:.2f} quantile",
        ),
    ]
    threshold_stats = {
        "high_error_quantile": high_error_q,
        "high_flip_quantile": high_flip_q,
        **{f"{key}_threshold": value for key, value in thresholds.items()},
    }
    return pd.DataFrame(rows), threshold_stats


def _make_target_summary_row(
    sample_df: pd.DataFrame,
    target_col: str,
    positive_mask: np.ndarray,
    threshold: float,
    positive_rule: str,
) -> Dict[str, Any]:
    scores = sample_df["uncertainty"].to_numpy()
    target = sample_df[target_col].to_numpy()
    return {
        "target": target_col,
        "n_samples": int(len(sample_df)),
        "spearman_rho": safe_spearman(scores, target),
        "auc": safe_roc_auc(positive_mask, scores),
        "average_precision": safe_average_precision(positive_mask, scores),
        "positive_rate": float(np.mean(positive_mask)),
        "positive_threshold": float(threshold),
        "positive_rule": positive_rule,
    }


def compute_bucket_statistics(sample_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
    uncertainty = sample_df["uncertainty"].to_numpy()
    edges = quantile_edges(uncertainty, int(config.get("bucket_count", 10)))
    bucket_ids = bucketize_by_edges(uncertainty, edges)
    work_df = sample_df.copy()
    work_df["uncertainty_bucket"] = bucket_ids

    bucket_rows: List[Dict[str, Any]] = []
    for bucket_id in np.sort(work_df["uncertainty_bucket"].unique()):
        sub = work_df[work_df["uncertainty_bucket"] == bucket_id]
        lower = float(edges[min(bucket_id, len(edges) - 2)])
        upper = float(edges[min(bucket_id + 1, len(edges) - 1)])
        bucket_rows.append(
            {
                "uncertainty_bucket": int(bucket_id),
                "bucket_label": f"B{bucket_id + 1}",
                "bucket_lower": lower,
                "bucket_upper": upper,
                "count": int(len(sub)),
                "mean_uncertainty": float(sub["uncertainty"].mean()),
                "mean_pointwise_error": float(sub["pointwise_error"].mean()),
                "mean_topk_flip": float(sub["topk_flip"].mean()),
                "mean_local_pairwise_flip": float(sub["local_pairwise_flip"].mean()),
                "mean_boundary_crossing_score": float(sub["boundary_crossing_score"].mean()),
                "mean_global_pairwise_flip": float(sub["global_pairwise_flip"].mean()),
            }
        )
    return pd.DataFrame(bucket_rows), edges


def compute_support_stratified_summary(
    sample_df: pd.DataFrame,
    config: Dict[str, Any],
    analysis_slices: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    high_error_q = float(config.get("high_error_quantile", 0.8))
    high_flip_q = float(config.get("high_flip_quantile", 0.8))
    rows: List[Dict[str, Any]] = []

    for slice_name, slice_df in analysis_slices.items():
        if slice_df.empty:
            continue
        error_threshold = float(np.quantile(slice_df["pointwise_error"], high_error_q))
        local_pairwise_threshold = float(np.quantile(slice_df["local_pairwise_flip"], high_flip_q))
        boundary_crossing_threshold = float(np.quantile(slice_df["boundary_crossing_score"], high_flip_q))

        for region in SUPPORT_LABELS:
            sub = slice_df[slice_df["support_region"] == region]
            if len(sub) == 0:
                continue
            uncertainty = sub["uncertainty"].to_numpy()
            pointwise_error = sub["pointwise_error"].to_numpy()
            topk_flip = sub["topk_flip"].to_numpy()
            local_pairwise = sub["local_pairwise_flip"].to_numpy()
            boundary_crossing = sub["boundary_crossing_score"].to_numpy()
            global_pairwise = sub["global_pairwise_flip"].to_numpy()

            rho_error = safe_spearman(uncertainty, pointwise_error)
            rho_topk = safe_spearman(uncertainty, topk_flip)
            rho_local_pairwise = safe_spearman(uncertainty, local_pairwise)
            rho_boundary_crossing = safe_spearman(uncertainty, boundary_crossing)

            rows.append(
                {
                    "analysis_slice": slice_name,
                    "support_region": region,
                    "n_samples": int(len(sub)),
                    "rho_uncertainty_error": rho_error,
                    "rho_uncertainty_topk_flip": rho_topk,
                    "rho_uncertainty_local_pairwise_flip": rho_local_pairwise,
                    "rho_uncertainty_boundary_crossing": rho_boundary_crossing,
                    "rho_uncertainty_global_pairwise_flip": safe_spearman(uncertainty, global_pairwise),
                    "alignment_gap_topk": rho_error - rho_topk if np.isfinite(rho_error) and np.isfinite(rho_topk) else np.nan,
                    "alignment_gap_local_pairwise": rho_error - rho_local_pairwise if np.isfinite(rho_error) and np.isfinite(rho_local_pairwise) else np.nan,
                    "alignment_gap_boundary_crossing": rho_error - rho_boundary_crossing if np.isfinite(rho_error) and np.isfinite(rho_boundary_crossing) else np.nan,
                    "auc_error": safe_roc_auc((pointwise_error >= error_threshold).astype(np.uint8), uncertainty),
                    "auc_topk_flip": safe_roc_auc(topk_flip.astype(np.uint8), uncertainty),
                    "auc_local_pairwise_flip": safe_roc_auc((local_pairwise >= local_pairwise_threshold).astype(np.uint8), uncertainty),
                    "auc_boundary_crossing": safe_roc_auc((boundary_crossing >= boundary_crossing_threshold).astype(np.uint8), uncertainty),
                    "ap_error": safe_average_precision((pointwise_error >= error_threshold).astype(np.uint8), uncertainty),
                    "ap_topk_flip": safe_average_precision(topk_flip.astype(np.uint8), uncertainty),
                    "ap_local_pairwise_flip": safe_average_precision((local_pairwise >= local_pairwise_threshold).astype(np.uint8), uncertainty),
                    "ap_boundary_crossing": safe_average_precision((boundary_crossing >= boundary_crossing_threshold).astype(np.uint8), uncertainty),
                    "mean_uncertainty": float(sub["uncertainty"].mean()),
                    "mean_pointwise_error": float(pointwise_error.mean()),
                    "mean_topk_flip": float(topk_flip.mean()),
                    "mean_local_pairwise_flip": float(local_pairwise.mean()),
                    "mean_boundary_crossing": float(boundary_crossing.mean()),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["analysis_slice"] = pd.Categorical(
            out["analysis_slice"],
            categories=["all", "small_margin"],
            ordered=True,
        )
        out["support_region"] = pd.Categorical(
            out["support_region"],
            categories=SUPPORT_LABELS.tolist(),
            ordered=True,
        )
        out = out.sort_values(["analysis_slice", "support_region"]).reset_index(drop=True)
    return out


def compute_margin_summary(
    sample_df: pd.DataFrame,
    config: Dict[str, Any],
    uncertainty_edges: np.ndarray,
) -> pd.DataFrame:
    bucket_ids = bucketize_by_edges(sample_df["uncertainty"].to_numpy(), uncertainty_edges)
    work_df = sample_df.copy()
    work_df["uncertainty_bucket"] = bucket_ids

    rows: List[Dict[str, Any]] = []
    for margin_region, sub_margin in work_df.groupby("margin_region", observed=False):
        for bucket_id, sub in sub_margin.groupby("uncertainty_bucket", observed=False):
            rows.append(
                {
                    "margin_region": margin_region,
                    "uncertainty_bucket": int(bucket_id),
                    "count": int(len(sub)),
                    "mean_uncertainty": float(sub["uncertainty"].mean()),
                    "mean_pointwise_error": float(sub["pointwise_error"].mean()),
                    "mean_topk_flip": float(sub["topk_flip"].mean()),
                    "mean_local_pairwise_flip": float(sub["local_pairwise_flip"].mean()),
                    "mean_boundary_crossing_score": float(sub["boundary_crossing_score"].mean()),
                    "mean_margin": float(sub["margin"].mean()),
                }
            )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["margin_region"] = pd.Categorical(
            summary["margin_region"],
            categories=MARGIN_LABELS.tolist(),
            ordered=True,
        )
        summary = summary.sort_values(["margin_region", "uncertainty_bucket"]).reset_index(drop=True)
    return summary
