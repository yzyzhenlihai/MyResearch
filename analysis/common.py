import ast
import json
import os
import pickle
import random
import sys
from argparse import Namespace
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.extend([".", "./src", "./examples/policy", "./src/DeepCTR-Torch", "./src/tianshou"])

from examples.policy.policy_utils import prepare_user_model
from src.core.envs.KuaiRec.KuaiData import KuaiData
from src.core.util.data import get_env_args, get_true_env


RUNTIME_PREFIX = (
    "PYTHONPATH=.:./src:./src/DeepCTR-Torch MPLCONFIGDIR=/tmp/mpl "
    "conda run -n easyrl4rec"
)


@dataclass
class OutputPaths:
    root: str
    figures: str
    artifacts: str
    tables: str


@dataclass
class CoreAssets:
    config: Dict[str, Any]
    args: Namespace
    ensemble: Any
    env: Any
    dataset: Any
    pred_mat: np.ndarray
    var_mat: np.ndarray
    oracle_norm_mat: np.ndarray
    oracle_raw_mat: np.ndarray
    user_indices: np.ndarray
    item_indices: np.ndarray
    user_raw_ids: np.ndarray
    item_raw_ids: np.ndarray
    output_paths: OutputPaths


def ensure_runtime_paths() -> None:
    for path in [".", "./src", "./examples/policy", "./src/DeepCTR-Torch", "./src/tianshou"]:
        if path not in sys.path:
            sys.path.append(path)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def create_output_paths(output_dir: str) -> OutputPaths:
    root = ensure_dir(output_dir)
    return OutputPaths(
        root=root,
        figures=ensure_dir(os.path.join(root, "figures")),
        artifacts=ensure_dir(os.path.join(root, "artifacts")),
        tables=ensure_dir(os.path.join(root, "tables")),
    )


def save_json(data: Mapping[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def save_yaml(data: Mapping[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_jsonable(data), f, sort_keys=False, allow_unicode=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    return value


def load_config(config_path: str, overrides: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if overrides:
        apply_overrides(config, overrides)
    return config


def parse_override_value(raw_value: str) -> Any:
    try:
        return yaml.safe_load(raw_value)
    except Exception:
        pass
    try:
        return ast.literal_eval(raw_value)
    except Exception:
        return raw_value


def apply_overrides(config: MutableMapping[str, Any], overrides: Sequence[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have key=value form: {override}")
        key, raw_value = override.split("=", 1)
        value = parse_override_value(raw_value)
        cursor: MutableMapping[str, Any] = config
        pieces = key.split(".")
        for piece in pieces[:-1]:
            if piece not in cursor or not isinstance(cursor[piece], MutableMapping):
                cursor[piece] = {}
            cursor = cursor[piece]
        cursor[pieces[-1]] = value


def build_args_from_config(config: Mapping[str, Any]) -> Namespace:
    seed = int(config.get("seed", 2023))
    args_dict = dict(config)
    args_dict.setdefault("cpu", True)
    args_dict.setdefault("cuda", 0)
    args_dict.setdefault("random_init", True)
    args_dict.setdefault("seed", seed)
    args = Namespace(**args_dict)
    return get_env_args(args)


def subset_axis(values: np.ndarray, max_count: int) -> np.ndarray:
    if max_count <= 0 or max_count >= len(values):
        return values.copy()
    return values[:max_count].copy()


def load_core_assets(config: Dict[str, Any]) -> CoreAssets:
    ensure_runtime_paths()
    set_global_seed(int(config.get("seed", 2023)))

    output_paths = create_output_paths(str(config["output_dir"]))
    args = build_args_from_config(config)
    ensemble = prepare_user_model(args)
    env, dataset, _ = get_true_env(args)

    with open(ensemble.PREDICTION_MAT_PATH, "rb") as f:
        pred_mat = pickle.load(f)
    with open(ensemble.VAR_MAT_PATH, "rb") as f:
        var_mat = pickle.load(f)

    oracle_raw_mat, _, _ = KuaiData.load_mat()
    small_df = pd.read_csv(
        "data/KuaiRec/data_raw/small_matrix_processed.csv",
        usecols=["user_id", "item_id", "watch_ratio_normed"],
    )
    rows = env.lbe_user.transform(small_df["user_id"].to_numpy())
    cols = env.lbe_item.transform(small_df["item_id"].to_numpy())
    oracle_norm_mat = csr_matrix(
        (small_df["watch_ratio_normed"].to_numpy(), (rows, cols)),
        shape=pred_mat.shape,
    ).toarray()

    pred_mat = np.asarray(pred_mat, dtype=np.float32)
    var_mat = np.asarray(var_mat, dtype=np.float32)
    oracle_raw_mat = np.asarray(oracle_raw_mat, dtype=np.float32)
    oracle_norm_mat = np.asarray(oracle_norm_mat, dtype=np.float32)

    if pred_mat.shape != var_mat.shape or pred_mat.shape != oracle_raw_mat.shape or pred_mat.shape != oracle_norm_mat.shape:
        raise ValueError(
            "Shape mismatch among prediction / variance / oracle matrices: "
            f"pred={pred_mat.shape}, var={var_mat.shape}, raw={oracle_raw_mat.shape}, norm={oracle_norm_mat.shape}"
        )

    user_indices = subset_axis(np.arange(pred_mat.shape[0], dtype=np.int32), int(config.get("max_users", 0)))
    item_indices = subset_axis(np.arange(pred_mat.shape[1], dtype=np.int32), int(config.get("subset_items", 0)))
    user_raw_ids = env.lbe_user.inverse_transform(user_indices)
    item_raw_ids = env.lbe_item.inverse_transform(item_indices)

    save_yaml(config, os.path.join(output_paths.root, "resolved_config.yaml"))

    return CoreAssets(
        config=config,
        args=args,
        ensemble=ensemble,
        env=env,
        dataset=dataset,
        pred_mat=pred_mat,
        var_mat=var_mat,
        oracle_norm_mat=oracle_norm_mat,
        oracle_raw_mat=oracle_raw_mat,
        user_indices=user_indices,
        item_indices=item_indices,
        user_raw_ids=np.asarray(user_raw_ids, dtype=np.int64),
        item_raw_ids=np.asarray(item_raw_ids, dtype=np.int64),
        output_paths=output_paths,
    )


def matrix_slice(matrix: np.ndarray, assets: CoreAssets) -> np.ndarray:
    return np.asarray(matrix[np.ix_(assets.user_indices, assets.item_indices)], dtype=np.float32)


def pair_to_key(user_ids: np.ndarray, item_ids: np.ndarray, factor: int = 10_000_000) -> np.ndarray:
    return user_ids.astype(np.int64) * factor + item_ids.astype(np.int64)


def load_train_item_embedding_mean(args: Namespace, read_message: str, env_name: str, user_model_name: str) -> np.ndarray:
    emb_dir = os.path.join("saved_models", env_name, user_model_name, "embeddings")
    emb_files = sorted(
        [
            os.path.join(emb_dir, name)
            for name in os.listdir(emb_dir)
            if name.startswith(f"[{read_message}]_emb_item_M") and name.endswith(".pt")
        ]
    )
    if not emb_files:
        raise FileNotFoundError(
            f"No training item embeddings found in {emb_dir} for read_message={read_message}"
        )

    emb_list: List[np.ndarray] = []
    for path in emb_files:
        emb = torch.load(path, map_location="cpu")
        if isinstance(emb, torch.Tensor):
            emb_np = emb.detach().cpu().numpy()
        elif isinstance(emb, np.ndarray):
            emb_np = emb
        elif isinstance(emb, dict):
            tensor_keys = [k for k, v in emb.items() if isinstance(v, torch.Tensor)]
            if not tensor_keys:
                raise ValueError(f"Unsupported embedding payload at {path}")
            emb_np = emb[tensor_keys[0]].detach().cpu().numpy()
        else:
            raise TypeError(f"Unsupported embedding type at {path}: {type(emb)}")
        emb_list.append(np.asarray(emb_np, dtype=np.float32))
    return np.mean(np.stack(emb_list, axis=0), axis=0)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def build_item_row_mapping(dataset: Any, item_emb: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], Dict[str, int]]:
    df_item_train = dataset.load_item_feat(only_small=False)
    item_raw_ids = df_item_train.index.to_numpy().astype(int)
    if len(item_raw_ids) != item_emb.shape[0]:
        if item_emb.shape[0] == item_raw_ids.max() + 1:
            item_raw_ids = np.arange(item_emb.shape[0], dtype=int)
        else:
            raise ValueError(
                "Cannot align item embedding rows and item ids: "
                f"emb_rows={item_emb.shape[0]}, item_ids={len(item_raw_ids)}"
            )

    item_to_row = {int(raw): int(row) for row, raw in enumerate(item_raw_ids)}
    stats = {
        "embedding_item_rows": int(item_emb.shape[0]),
        "train_item_id_count": int(len(item_raw_ids)),
        "train_item_id_min": int(item_raw_ids.min()),
        "train_item_id_max": int(item_raw_ids.max()),
    }
    return item_raw_ids, item_to_row, stats


def stable_argsort_desc(scores: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    return np.lexsort((item_ids, -scores)).astype(np.int32)


def ranks_from_order(order: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return ranks


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def quantile_edges(values: np.ndarray, num_buckets: int) -> np.ndarray:
    if num_buckets <= 1:
        return np.asarray([values.min(), values.max()], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, num_buckets + 1)
    edges = np.quantile(values, quantiles)
    edges = np.asarray(edges, dtype=np.float64)
    edges[0] = np.min(values)
    edges[-1] = np.max(values)
    unique = np.unique(edges)
    if len(unique) <= 1:
        unique = np.asarray([values.min(), values.max()], dtype=np.float64)
    return unique


def bucketize_by_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=np.int32)
    bucket_ids = np.digitize(values, edges[1:-1], right=False)
    return bucket_ids.astype(np.int32)


def minmax_normalize_rows(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    row_min = np.min(mat, axis=1, keepdims=True)
    row_max = np.max(mat, axis=1, keepdims=True)
    denom = row_max - row_min
    out = np.zeros_like(mat, dtype=np.float32)
    valid = denom > eps
    out[valid[:, 0]] = ((mat[valid[:, 0]] - row_min[valid[:, 0]]) / denom[valid[:, 0]]).astype(np.float32)
    return out


def dataframe_to_markdown(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "| Empty |\n| --- |\n| No rows |"
    fmt_df = df.copy()
    for col in fmt_df.columns:
        if pd.api.types.is_float_dtype(fmt_df[col]):
            fmt_df[col] = fmt_df[col].map(lambda x: "nan" if pd.isna(x) else f"{x:.{float_digits}f}")
    header = "| " + " | ".join(map(str, fmt_df.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(fmt_df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in fmt_df.to_numpy()]
    return "\n".join([header, sep] + rows)


def choose_matrix_by_name(name: str, norm_mat: np.ndarray, raw_mat: np.ndarray) -> np.ndarray:
    mapping = {
        "watch_ratio_normed": norm_mat,
        "norm": norm_mat,
        "oracle_norm": norm_mat,
        "watch_ratio": raw_mat,
        "raw": raw_mat,
        "oracle_raw": raw_mat,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported oracle matrix name: {name}")
    return mapping[name]


def get_primary_uncertainty_source(config: Mapping[str, Any]) -> str:
    return str(config.get("primary_uncertainty_source", config.get("uncertainty_source", "variance_head")))


def get_appendix_uncertainty_sources(config: Mapping[str, Any]) -> List[str]:
    sources = config.get("appendix_uncertainty_sources", [])
    if sources is None:
        return []
    if isinstance(sources, str):
        return [sources]
    return [str(source) for source in sources]


def sanitize_column_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.strip()).strip("_").lower()


def split_matrix_regions(
    mat: np.ndarray,
    quantiles: Sequence[float],
    mode: str,
    labels: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if len(quantiles) != 2:
        raise ValueError(f"quantiles must contain exactly two values, got {quantiles}")
    if len(labels) != 3:
        raise ValueError(f"labels must contain exactly three values, got {labels}")
    q_low, q_high = sorted(float(q) for q in quantiles)
    values = np.asarray(mat, dtype=np.float32)
    codes = np.ones(values.shape, dtype=np.int8)

    if mode == "global":
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            return codes, {"mode": mode, "q_low": q_low, "q_high": q_high, "threshold_low": None, "threshold_high": None}
        threshold_low = float(np.quantile(valid, q_low))
        threshold_high = float(np.quantile(valid, q_high))
        codes[values <= threshold_low] = 0
        codes[values >= threshold_high] = 2
        stats = {
            "mode": mode,
            "q_low": q_low,
            "q_high": q_high,
            "threshold_low": threshold_low,
            "threshold_high": threshold_high,
        }
    elif mode == "per_user":
        row_low = np.quantile(values, q_low, axis=1, keepdims=True)
        row_high = np.quantile(values, q_high, axis=1, keepdims=True)
        codes[values <= row_low] = 0
        codes[values >= row_high] = 2
        stats = {
            "mode": mode,
            "q_low": q_low,
            "q_high": q_high,
            "threshold_low_mean": float(np.mean(row_low)),
            "threshold_high_mean": float(np.mean(row_high)),
        }
    else:
        raise ValueError(f"Unsupported quantile split mode: {mode}")

    stats["region_counts"] = {str(labels[idx]): int((codes == idx).sum()) for idx in range(len(labels))}
    return codes, stats


def choose_active_mask(mask_name: str, mask_map: Mapping[str, np.ndarray], default_name: str) -> Tuple[str, np.ndarray]:
    active_name = mask_name if mask_name in mask_map else default_name
    return active_name, np.asarray(mask_map[active_name], dtype=bool)


def build_manifest(config: Mapping[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "runtime_prefix": RUNTIME_PREFIX,
        "config": to_jsonable(config),
        "extra": to_jsonable(extra),
    }
