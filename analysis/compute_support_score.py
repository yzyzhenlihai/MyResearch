from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.common import (
    CoreAssets,
    build_item_row_mapping,
    load_train_item_embedding_mean,
    normalize_rows,
    pair_to_key,
    split_matrix_regions,
)


SUPPORT_LABELS = np.asarray(["far_OOD", "boundary_support", "in_support"])


def compute_support_matrices(
    assets: CoreAssets,
    config: Dict[str, Any],
    user_raw_ids: Optional[np.ndarray] = None,
    item_raw_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    target_user_raw_ids = np.asarray(user_raw_ids if user_raw_ids is not None else assets.user_raw_ids, dtype=np.int64)
    target_item_raw_ids = np.asarray(item_raw_ids if item_raw_ids is not None else assets.item_raw_ids, dtype=np.int64)
    item_emb = load_train_item_embedding_mean(
        args=assets.args,
        read_message=str(config["read_message"]),
        env_name=str(config["env"]),
        user_model_name=str(config["user_model_name"]),
    )
    _, item_to_row, mapping_stats = build_item_row_mapping(assets.dataset, item_emb)

    support_rows_by_user, overlap_stats = scan_big_logs_for_support_rows(
        user_raw_ids=target_user_raw_ids,
        item_to_row=item_to_row,
        big_chunksize=int(config.get("big_chunksize", 2_000_000)),
        verify_overlap=bool(config.get("verify_ood", True)),
    )

    distance_mat, distance_stats = compute_user_knn_distance_matrix(
        user_raw_ids=target_user_raw_ids,
        item_raw_ids=target_item_raw_ids,
        item_emb=item_emb,
        item_to_row=item_to_row,
        support_rows_by_user=support_rows_by_user,
        knn_k=int(config.get("knn_k", 20)),
    )
    tau_support = config.get("tau_support", "auto")
    finite_distance = distance_mat[np.isfinite(distance_mat)]
    if tau_support == "auto":
        tau_value = float(np.median(finite_distance)) if len(finite_distance) > 0 else 1.0
    else:
        tau_value = float(tau_support)
    tau_value = max(tau_value, 1e-8)

    support_score = np.zeros_like(distance_mat, dtype=np.float32)
    valid = np.isfinite(distance_mat)
    support_score[valid] = np.exp(-distance_mat[valid] / tau_value).astype(np.float32)

    region_codes_by_mode: Dict[str, np.ndarray] = {}
    region_stats_by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in ["global", "per_user"]:
        codes, stats = split_support_regions(
            support_score=support_score,
            quantiles=config.get("support_quantiles", [0.2, 0.8]),
            mode=mode,
        )
        region_codes_by_mode[mode] = codes
        region_stats_by_mode[mode] = stats
    active_mode = str(config.get("support_quantile_mode", "global"))
    if active_mode not in region_codes_by_mode:
        active_mode = "global"

    return {
        "item_embedding": item_emb,
        "support_rows_by_user": support_rows_by_user,
        "distance_mat": distance_mat,
        "support_score_mat": support_score,
        "support_region_codes": region_codes_by_mode[active_mode],
        "support_region_codes_by_mode": region_codes_by_mode,
        "support_region_labels": SUPPORT_LABELS,
        "stats": {
            "mapping": mapping_stats,
            "overlap": overlap_stats,
            "distance": distance_stats,
            "tau_support": float(tau_value),
            "region_active_mode": active_mode,
            "region_by_mode": region_stats_by_mode,
        },
    }


def scan_big_logs_for_support_rows(
    user_raw_ids: np.ndarray,
    item_to_row: Dict[int, int],
    big_chunksize: int,
    verify_overlap: bool,
) -> Tuple[Dict[int, np.ndarray], Dict[str, Optional[float]]]:
    support_rows_by_user = defaultdict(set)
    small_pair_keys_sorted: Optional[np.ndarray] = None
    seen_small: Optional[np.ndarray] = None

    if verify_overlap:
        df_small = pd.read_csv(
            "data/KuaiRec/data_raw/small_matrix_processed.csv",
            usecols=["user_id", "item_id"],
        )
        df_small = df_small[df_small["user_id"].isin(set(user_raw_ids.tolist()))]
        small_pair_keys_sorted = np.unique(
            pair_to_key(df_small["user_id"].to_numpy(), df_small["item_id"].to_numpy())
        )
        seen_small = np.zeros(len(small_pair_keys_sorted), dtype=bool)

    user_set = set(user_raw_ids.tolist())
    for chunk in pd.read_csv(
        "data/KuaiRec/data_raw/big_matrix_processed.csv",
        usecols=["user_id", "item_id"],
        chunksize=big_chunksize,
    ):
        if seen_small is not None and small_pair_keys_sorted is not None:
            big_keys = np.unique(pair_to_key(chunk["user_id"].to_numpy(), chunk["item_id"].to_numpy()))
            pos = np.searchsorted(small_pair_keys_sorted, big_keys)
            valid = pos < len(small_pair_keys_sorted)
            if valid.any():
                pos_valid = pos[valid]
                matched = small_pair_keys_sorted[pos_valid] == big_keys[valid]
                if matched.any():
                    seen_small[pos_valid[matched]] = True

        sub = chunk[chunk["user_id"].isin(user_set)]
        if len(sub) == 0:
            continue
        row_ids = sub["item_id"].map(item_to_row)
        sub = sub.assign(item_row=row_ids)
        sub = sub[sub["item_row"].notna()]
        if len(sub) == 0:
            continue
        for uid, rows in sub.groupby("user_id", observed=False)["item_row"]:
            support_rows_by_user[int(uid)].update(rows.astype(int).to_numpy().tolist())

    overlap_stats = {
        "seen_small_pairs_in_big": None,
        "unseen_small_pairs_in_big": None,
        "unseen_ratio_vs_big": None,
    }
    if seen_small is not None:
        seen = int(seen_small.sum())
        total = int(len(seen_small))
        unseen = total - seen
        overlap_stats = {
            "seen_small_pairs_in_big": seen,
            "unseen_small_pairs_in_big": unseen,
            "unseen_ratio_vs_big": float(unseen / max(total, 1)),
        }

    support_rows_by_user_np = {
        uid: np.asarray(sorted(list(rows)), dtype=np.int32)
        for uid, rows in support_rows_by_user.items()
    }
    return support_rows_by_user_np, overlap_stats


def compute_user_knn_distance_matrix(
    user_raw_ids: np.ndarray,
    item_raw_ids: np.ndarray,
    item_emb: np.ndarray,
    item_to_row: Dict[int, int],
    support_rows_by_user: Dict[int, np.ndarray],
    knn_k: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    missing_items = [int(item) for item in item_raw_ids if int(item) not in item_to_row]
    if missing_items:
        raise ValueError(f"Some candidate items are not found in the embedding map. Missing count={len(missing_items)}")

    candidate_rows = np.asarray([item_to_row[int(item)] for item in item_raw_ids], dtype=np.int32)
    emb_norm = normalize_rows(np.asarray(item_emb, dtype=np.float32))
    cand_emb_norm = emb_norm[candidate_rows]
    distance_mat = np.full((len(user_raw_ids), len(item_raw_ids)), np.nan, dtype=np.float32)

    users_with_empty_support = 0
    users_with_short_support = 0

    for local_uid, raw_uid in enumerate(tqdm(user_raw_ids, desc="Computing support kNN distance")):
        support_rows = support_rows_by_user.get(int(raw_uid), None)
        if support_rows is None or len(support_rows) == 0:
            users_with_empty_support += 1
            continue

        supp_emb_norm = emb_norm[support_rows]
        sims = np.matmul(cand_emb_norm, supp_emb_norm.T)
        kk = min(knn_k, supp_emb_norm.shape[0])
        if kk < knn_k:
            users_with_short_support += 1
        if kk == supp_emb_norm.shape[0]:
            topk_mean_sim = sims.mean(axis=1)
        else:
            topk = np.partition(sims, supp_emb_norm.shape[0] - kk, axis=1)[:, -kk:]
            topk_mean_sim = topk.mean(axis=1)
        distance_mat[local_uid] = (1.0 - topk_mean_sim).astype(np.float32)

    stats = {
        "distance_metric": "cosine_knn_mean",
        "knn_k": int(knn_k),
        "users_computed": int(len(user_raw_ids)),
        "users_with_empty_support": int(users_with_empty_support),
        "users_with_short_support": int(users_with_short_support),
        "distance_non_nan_ratio_grid": float(np.isfinite(distance_mat).mean()),
    }
    return distance_mat, stats


def split_support_regions(
    support_score: np.ndarray,
    quantiles: Sequence[float],
    mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    return split_matrix_regions(
        mat=support_score,
        quantiles=quantiles,
        mode=mode,
        labels=SUPPORT_LABELS.tolist(),
    )
