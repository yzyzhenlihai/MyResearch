from __future__ import annotations

from collections import Counter
import math
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple


DEFAULT_ENTROPY_WHEN_HISTORY_TOO_SHORT = 1.0
"""当历史长度不足时，沿用 DORL 原始实现返回的默认熵值。"""


def get_entropy(mylist, need_count: bool = True) -> float:
    """计算归一化熵。

    Args:
        mylist: 当 `need_count=True` 时为可迭代样本列表；当 `need_count=False`
            时为“取值 -> 计数”的映射。
        need_count (bool): 是否需要先对 `mylist` 做频次统计。

    Returns:
        float: 归一化后的熵值，理论范围约为 `[0, 1]`。

    Raises:
        ValueError: 当输入频次和为 0 时，底层除法会失败。
    """
    if len(mylist) <= 1:
        return DEFAULT_ENTROPY_WHEN_HISTORY_TOO_SHORT
    if need_count:
        cnt_dict = Counter(mylist)
    else:
        cnt_dict = mylist
    prob = [v / sum(cnt_dict.values()) for v in cnt_dict.values()]
    log_prob = [0.0 if p <= 0 else math.log2(p) for p in prob]
    entropy = -sum(lp * p for lp, p in zip(log_prob, prob)) / math.log2(len(cnt_dict))
    return float(entropy)


def get_features_of_last_n_items_features(
    n: int,
    hist_tra: Sequence[int],
    map_item_feat: Mapping[int, Sequence[int]],
    is_sort: bool,
):
    """枚举最近 `n` 个物品的所有特征组合。

    Args:
        n (int): 组合长度，必须大于 0。
        hist_tra (Sequence[int]): 物品历史序列。
        map_item_feat (Mapping[int, Sequence[int]]): 物品到离散特征列表的映射。
        is_sort (bool): 是否对组合结果排序后再作为键使用。

    Returns:
        set[tuple[int, ...]] | list[list[int]]: 当历史不足时返回 `[[]]`，
            否则返回所有可能的特征组合集合。

    Raises:
        KeyError: 当历史中的物品在 `map_item_feat` 中不存在时抛出。
    """
    if len(hist_tra) < n or n <= 0:
        return [[]]
    target_item = hist_tra[-1]
    target_features = map_item_feat[target_item]
    last_lists = get_features_of_last_n_items_features(
        n - 1, hist_tra[:-1], map_item_feat, is_sort
    )
    res = set()
    for feat_list in last_lists:
        feat_list = list(feat_list)
        for feat in target_features:
            new_list = feat_list.copy()
            new_list.append(feat)
            if is_sort:
                new_list = sorted(new_list)
            res.add(tuple(new_list))
    return res


def update_entropy_count_map(
    map_hist_count: MutableMapping[Tuple[int, ...], MutableMapping[int, int]],
    hist_tra: Sequence[int],
    item: int,
    require_len: int,
    is_sort: bool = True,
) -> None:
    """将一个“历史窗口 -> 下一动作”的出现次数累加到计数字典。

    Args:
        map_hist_count (MutableMapping[Tuple[int, ...], MutableMapping[int, int]]):
            历史窗口到动作计数的嵌套映射。
        hist_tra (Sequence[int]): 当前动作之前的历史序列。
        item (int): 当前动作或当前特征。
        require_len (int): 需要截取的历史窗口长度。
        is_sort (bool): 是否对历史窗口排序。

    Returns:
        None
    """
    if len(hist_tra) < require_len:
        return
    history = tuple(sorted(hist_tra[-require_len:]) if is_sort else hist_tra[-require_len:])
    map_hist_count[history][item] += 1


def accumulate_entropy_counts_for_row(
    map_hist_count: MutableMapping[Tuple[int, ...], MutableMapping[int, int]],
    hist_tra: Sequence[int],
    item: int,
    entropy_window: Iterable[int],
    feature_level: bool,
    map_item_feat: Mapping[int, Sequence[int]] | None,
    is_sorted: bool,
) -> None:
    """按 DORL 口径累计构造熵统计所需的历史计数。

    Args:
        map_hist_count: 历史窗口计数字典，会被原地更新。
        hist_tra (Sequence[int]): 当前动作之前的历史物品序列。
        item (int): 当前动作对应的物品 ID。
        entropy_window (Iterable[int]): 需要统计的窗口长度集合。
        feature_level (bool): 是否切换到特征级熵统计。
        map_item_feat (Mapping[int, Sequence[int]] | None): 物品到特征列表的映射。
        is_sorted (bool): 是否对窗口内容排序。

    Returns:
        None

    Raises:
        ValueError: 当 `feature_level=True` 但未提供 `map_item_feat` 时抛出。
    """
    entropy_terms = set(entropy_window) - {0}
    if not entropy_terms:
        return

    if feature_level:
        if map_item_feat is None:
            raise ValueError("map_item_feat is required when feature_level=True")
        features = map_item_feat[item]
        for require_len in entropy_terms:
            hist_feats = get_features_of_last_n_items_features(
                require_len, hist_tra, map_item_feat, is_sort=is_sorted
            )
            for hist_feat_list in hist_feats:
                for feat in features:
                    update_entropy_count_map(
                        map_hist_count, hist_feat_list, feat, require_len, is_sort=is_sorted
                    )
    else:
        for require_len in entropy_terms:
            update_entropy_count_map(
                map_hist_count, hist_tra, item, require_len, is_sort=is_sorted
            )


def finalize_entropy_map(
    map_hist_count: Mapping[Tuple[int, ...], Mapping[int, int]]
) -> Dict[Tuple[int, ...], float]:
    """将计数字典转换为“历史窗口 -> 熵值”的查表字典。

    Args:
        map_hist_count (Mapping[Tuple[int, ...], Mapping[int, int]]): 历史窗口到动作计数的映射。

    Returns:
        Dict[Tuple[int, ...], float]: 历史窗口到归一化熵的映射。
    """
    return {k: get_entropy(v, need_count=False) for k, v in map_hist_count.items()}


def compute_step_entropy(
    history_with_current: Sequence[int],
    entropy_dict: Mapping[Tuple[int, ...], float],
    entropy_window: Iterable[int],
    feature_level: bool,
    map_item_feat: Mapping[int, Sequence[int]] | None,
    is_sorted: bool,
) -> float:
    """计算某一步动作对应的 DORL 熵项。

    Args:
        history_with_current (Sequence[int]): 包含当前动作在内的历史序列。
        entropy_dict (Mapping[Tuple[int, ...], float]): 预先构造好的熵查表。
        entropy_window (Iterable[int]): 需要累加的窗口长度集合。
        feature_level (bool): 是否使用特征级熵统计。
        map_item_feat (Mapping[int, Sequence[int]] | None): 物品到特征列表的映射。
        is_sorted (bool): 是否对窗口内容排序。

    Returns:
        float: 当前步的累计熵项。

    Raises:
        ValueError: 当 `feature_level=True` 但未提供 `map_item_feat` 时抛出。
    """
    entropy = 0.0
    entropy_terms = set(entropy_window) - {0}
    if not entropy_terms:
        return entropy

    for k in entropy_terms:
        if len(history_with_current) < k:
            entropy += DEFAULT_ENTROPY_WHEN_HISTORY_TOO_SHORT
            continue

        action_set = tuple(
            sorted(history_with_current[-k:]) if is_sorted else history_with_current[-k:]
        )
        if feature_level:
            if map_item_feat is None:
                raise ValueError("map_item_feat is required when feature_level=True")
            feat_set = get_features_of_last_n_items_features(
                k, action_set, map_item_feat, is_sort=is_sorted
            )
            if len(feat_set) == 0:
                entropy += DEFAULT_ENTROPY_WHEN_HISTORY_TOO_SHORT
                continue
            ans_feat = 0.0
            for feat in feat_set:
                ans_feat += float(entropy_dict.get(feat, DEFAULT_ENTROPY_WHEN_HISTORY_TOO_SHORT))
            entropy += ans_feat / len(feat_set)
        else:
            entropy += float(entropy_dict.get(action_set, DEFAULT_ENTROPY_WHEN_HISTORY_TOO_SHORT))
    return float(entropy)


def compute_mopo_penalized_reward(
    pred_reward: float,
    max_var: float,
    lambda_variance: float,
    predicted_min: float,
    maxvar_max: float,
) -> float:
    """按 MOPO 环境实现还原平移后的惩罚 reward。

    Args:
        pred_reward (float): user model 预测 reward。
        max_var (float): 对应 `(user, item)` 的不确定性上界。
        lambda_variance (float): 不确定性惩罚系数。
        predicted_min (float): `predicted_mat` 的全局最小值。
        maxvar_max (float): `maxvar_mat` 的全局最大值。

    Returns:
        float: 与 `PenaltyVarSimulatedEnv` 一致的平移后 reward。
    """
    min_r = predicted_min - lambda_variance * maxvar_max
    return float(pred_reward - lambda_variance * max_var - min_r)
