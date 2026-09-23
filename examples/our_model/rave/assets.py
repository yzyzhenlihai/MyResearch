"""训练划分主导类别冻结、资产指纹与 MAC checkpoint 接入。"""

import hashlib

import numpy as np
import torch

from src.core.evaluation.utils import get_dominant_values, get_feat_dominate_dict

DOMINATION_VERSION = "BaseData_positive_frequency_eval_cutoff_v1"


def file_sha256(path):
    """分块读取本地文件并返回 SHA256；不存在时透传 FileNotFoundError。"""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_categories(dataset, env, env_name, train_raw_users, top_rate=0.8):
    """重新读取训练数据并复用 BaseData 统计，完全绕过旧 domination 缓存。

    Args:
        dataset (BaseData): KuaiRec/KuaiRand 数据源。
        env (BaseEnv): 只读取内部 item ID 映射。
        env_name (str): KuaiEnv-v0 或 KuaiRand-v0。
        train_raw_users (Sequence[int]): 当前训练划分用户，验证/测试用户不参与统计。
        top_rate (float): 与评估相同的主导类别截断阈值。

    Returns:
        tuple: 类别表、命中表以及包含划分/映射/阈值/指纹的元数据。

    Raises:
        ValueError: 不支持的数据集或当前划分无正反馈类别统计。
    """
    if env_name == "KuaiEnv-v0":
        source, feedback, threshold = "big_matrix_processed.csv", "watch_ratio_normed", 0.6
    elif env_name == "KuaiRand-v0":
        source, feedback, threshold = "train_processed.csv", "is_click", 1.0
    else:
        raise ValueError("RAVE assets currently support KuaiRec and KuaiRand")
    interactions, _, items, _ = dataset.get_df(source)
    interactions = interactions[interactions["user_id"].isin(train_raw_users)]
    statistics = dataset.get_sorted_domination_features(
        interactions, items, is_multi_hot=True, yname=feedback, threshold=threshold)
    dominant = get_dominant_values(statistics["feat"], top_rate)
    encoder = getattr(env, "lbe_item", None)
    raw_items = np.asarray(encoder.classes_ if encoder is not None else np.arange(env.mat.shape[1]))
    categories = items.loc[raw_items].filter(regex="^feat", axis=1).to_numpy().astype(np.int64)
    categories[categories <= 0] = 0
    hits = np.isin(categories, dominant).any(axis=1)
    # 直接与既有评估函数核对 pooled 命中语义，防止类别编码偏移。
    reference = get_feat_dominate_dict(items, raw_items, statistics, top_rate)["ifeat_feat"]
    if not np.isclose(hits.mean(), reference):
        raise ValueError("Frozen MCD hits disagree with existing evaluator")
    metadata = {
        "version": DOMINATION_VERSION, "source": source,
        "feedback": feedback, "positive_threshold": threshold,
        "top_rate": top_rate, "dominant_categories": dominant.tolist(),
        "train_raw_users": [int(user) for user in train_raw_users],
        "training_interactions": len(interactions),
        "raw_item_ids": raw_items.tolist(),
        "category_sha256": hashlib.sha256(categories.tobytes()).hexdigest(),
        "training_rows_sha256": hashlib.sha256(
            interactions[["user_id", "item_id", feedback]].to_numpy().tobytes()).hexdigest(),
        "missing_category_catalog_rate": float((categories == 0).all(axis=1).mean()),
        "statistics": [[int(category), float(weight)] for category, weight in statistics["feat"]],
    }
    return torch.from_numpy(categories), torch.from_numpy(hits), metadata
