from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from analysis.common import (
    CoreAssets,
    get_appendix_uncertainty_sources,
    get_primary_uncertainty_source,
    matrix_slice,
)


def compute_uncertainty_payload(
    assets: CoreAssets,
    config: Dict[str, Any],
    user_raw_ids: Optional[np.ndarray] = None,
    item_raw_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    return compute_uncertainty_payload_for_targets(
        assets,
        config,
        user_raw_ids=user_raw_ids,
        item_raw_ids=item_raw_ids,
    )


def compute_uncertainty_payload_for_targets(
    assets: CoreAssets,
    config: Dict[str, Any],
    user_raw_ids: Optional[np.ndarray] = None,
    item_raw_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    primary_source = get_primary_uncertainty_source(config)
    appendix_sources = [source for source in get_appendix_uncertainty_sources(config) if source != primary_source]
    sources = [primary_source] + appendix_sources
    matrices: Dict[str, np.ndarray] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        matrix, source_meta = compute_uncertainty_matrix_for_source(
            assets,
            source,
            config,
            user_raw_ids=user_raw_ids,
            item_raw_ids=item_raw_ids,
        )
        matrices[source] = matrix
        meta[source] = source_meta
    return {
        "primary_source": primary_source,
        "matrices": matrices,
        "meta": meta,
        "primary_matrix": matrices[primary_source],
    }


def compute_uncertainty_matrix(assets: CoreAssets, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    payload = compute_uncertainty_payload(assets, config)
    return payload["primary_matrix"], payload["meta"][payload["primary_source"]]


def compute_uncertainty_matrix_for_source(
    assets: CoreAssets,
    source: str,
    config: Dict[str, Any],
    user_raw_ids: Optional[np.ndarray] = None,
    item_raw_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    target_user_raw_ids = np.asarray(user_raw_ids if user_raw_ids is not None else assets.user_raw_ids, dtype=np.int64)
    target_item_raw_ids = np.asarray(item_raw_ids if item_raw_ids is not None else assets.item_raw_ids, dtype=np.int64)
    if source == "variance_head":
        user_index = assets.env.lbe_user.transform(target_user_raw_ids)
        item_index = assets.env.lbe_item.transform(target_item_raw_ids)
        uncertainty = np.asarray(assets.var_mat[np.ix_(user_index, item_index)], dtype=np.float32)
        meta = {
            "uncertainty_source": source,
            "description": "Saved variance-head output from matsVar; not strict ensemble variance.",
            "alignment_check": forward_alignment_check(assets, config),
        }
        return uncertainty.astype(np.float32), meta
    if source == "ensemble_disagreement":
        uncertainty = compute_ensemble_disagreement(
            assets,
            config,
            user_raw_ids=target_user_raw_ids,
            item_raw_ids=target_item_raw_ids,
        )
        meta = {
            "uncertainty_source": source,
            "description": "Variance across ensemble mean predictions on selected user-item grid.",
            "alignment_check": None,
        }
        return uncertainty.astype(np.float32), meta
    raise ValueError(f"Unsupported uncertainty_source: {source}")


def forward_alignment_check(assets: CoreAssets, config: Dict[str, Any]) -> Dict[str, Any]:
    n_users = int(config.get("forward_check_users", 3))
    n_items = int(config.get("forward_check_items", 64))
    atol = float(config.get("forward_allclose_atol", 1e-5))
    if n_users <= 0 or n_items <= 0:
        return {"enabled": False}

    df_val, df_user_val, df_item_val, _ = assets.dataset.get_val_data()
    user_features, item_features, _ = assets.dataset.get_features(assets.args.is_userinfo)
    user_ids = np.unique(df_val["user_id"].to_numpy())[:n_users]
    item_ids = np.unique(df_val["item_id"].to_numpy())[:n_items]

    x_np = build_positive_feature_block(
        df_user_val=df_user_val,
        df_item_val=df_item_val,
        user_features=user_features,
        item_features=item_features,
        user_raw_ids=user_ids,
        item_raw_ids=item_ids,
    )

    sigma2_direct = extract_sigma2_from_forward(assets, x_np).reshape(len(user_ids), len(item_ids))
    sigma2_saved = assets.var_mat[np.ix_(
        assets.env.lbe_user.transform(user_ids),
        assets.env.lbe_item.transform(item_ids),
    )]
    diff = np.abs(sigma2_direct - sigma2_saved)
    return {
        "enabled": True,
        "subset_shape": [int(len(user_ids)), int(len(item_ids))],
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "allclose": bool(np.allclose(sigma2_direct, sigma2_saved, atol=atol, rtol=atol)),
        "atol": float(atol),
    }


def extract_sigma2_from_forward(assets: CoreAssets, x_np: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    vars_each_model: List[np.ndarray] = []
    with torch.no_grad():
        for model in assets.ensemble.user_models:
            model.eval()
            model_vars: List[np.ndarray] = []
            for start in range(0, len(x_np), batch_size):
                xb = torch.tensor(x_np[start:start + batch_size], dtype=torch.float32, device=model.device)
                _, log_var = model.forward(xb)
                model_vars.append(torch.exp(log_var).cpu().numpy().reshape(-1))
            vars_each_model.append(np.concatenate(model_vars, axis=0))
    return np.stack(vars_each_model, axis=0).max(axis=0)


def compute_ensemble_disagreement(
    assets: CoreAssets,
    config: Dict[str, Any],
    user_raw_ids: Optional[np.ndarray] = None,
    item_raw_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    df_val, df_user_val, df_item_val, _ = assets.dataset.get_val_data()
    user_features, item_features, _ = assets.dataset.get_features(assets.args.is_userinfo)

    batch_user_size = int(config.get("ensemble_user_batch_size", 32))
    batch_size = int(config.get("ensemble_eval_batch_size", 8192))
    user_raw_ids = np.asarray(user_raw_ids if user_raw_ids is not None else assets.user_raw_ids, dtype=np.int64)
    item_raw_ids = np.asarray(item_raw_ids if item_raw_ids is not None else assets.item_raw_ids, dtype=np.int64)
    uncertainty = np.zeros((len(user_raw_ids), len(item_raw_ids)), dtype=np.float32)

    for user_start in range(0, len(user_raw_ids), batch_user_size):
        user_block = user_raw_ids[user_start:user_start + batch_user_size]
        x_np = build_positive_feature_block(
            df_user_val=df_user_val,
            df_item_val=df_item_val,
            user_features=user_features,
            item_features=item_features,
            user_raw_ids=user_block,
            item_raw_ids=item_raw_ids,
        )
        preds = predict_each_model(assets, x_np, batch_size=batch_size)
        block_uncertainty = np.var(preds, axis=0, dtype=np.float32)
        uncertainty[user_start:user_start + len(user_block)] = block_uncertainty.reshape(len(user_block), len(item_raw_ids))
    return uncertainty


def predict_each_model(assets: CoreAssets, x_np: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for model in assets.ensemble.user_models:
            model.eval()
            one_model: List[np.ndarray] = []
            for start in range(0, len(x_np), batch_size):
                xb = torch.tensor(x_np[start:start + batch_size], dtype=torch.float32, device=model.device)
                pred, _ = model.forward(xb)
                one_model.append(pred.detach().cpu().numpy().reshape(-1))
            outputs.append(np.concatenate(one_model, axis=0))
    return np.stack(outputs, axis=0)


def build_positive_feature_block(
    df_user_val: pd.DataFrame,
    df_item_val: pd.DataFrame,
    user_features: List[str],
    item_features: List[str],
    user_raw_ids: np.ndarray,
    item_raw_ids: np.ndarray,
) -> np.ndarray:
    user_block = df_user_val.loc[user_raw_ids].reset_index()[user_features].to_numpy()
    item_block = df_item_val.loc[item_raw_ids].reset_index()[item_features].to_numpy()
    user_rep = np.repeat(user_block, len(item_raw_ids), axis=0)
    item_rep = np.tile(item_block, (len(user_raw_ids), 1))
    return np.concatenate([user_rep, item_rep], axis=1).astype(np.float32)
