import numpy as np


DEFAULT_MCD_SEGMENT_LENGTH = 30
"""分段 MCD 默认使用的连续轨迹步数。"""


def get_dominant_values(sorted_items, top_rate=0.8):
    """复用评估的累计截断口径，返回冻结主导类别值。

    Args:
        sorted_items (Sequence): 按频次降序排列的 `(类别, 正权重)`。
        top_rate (float): 累计阈值，范围 `(0, 1]`。

    Returns:
        np.ndarray: 不纳入首次越过阈值的类别，至少保留一个。

    Raises:
        ValueError: 空统计、非有限/非正权重或非法阈值。
    """
    if not sorted_items or not 0 < top_rate <= 1:
        raise ValueError("Nonempty category statistics and valid top_rate required")
    values = np.asarray([pair[1] for pair in sorted_items], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Category weights must be finite and positive")
    cumulative = (values / values.sum()).cumsum()
    count = 0
    for value in cumulative:
        if value > top_rate:
            break
        count += 1
    return np.asarray([pair[0] for pair in sorted_items])[:max(1, count)]


def get_feat_dominate_dict(
    df_item_val,
    all_acts_origin,
    item_feat_domination,
    top_rate=0.8,
    draw_bar=False,
):
    """计算全部曝光按步数汇总的旧版 MCD 指标。

    Args:
        df_item_val (pandas.DataFrame): 以原始 item id 为索引的物品特征表。
        all_acts_origin (Sequence): 需要统计的原始 item id 序列，不能为空。
        item_feat_domination (dict | None): 训练数据得到的各特征频次降序表；
            为 ``None`` 时返回空字典。
        top_rate (float): 主导类别累计频率阈值，范围为 ``(0, 1]``。
        draw_bar (bool): 是否同时返回用于绘图的有效类别数组。

    Returns:
        dict: 旧版 pooled MCD 指标；KuaiRec/KuaiRand 的键为
        ``ifeat_feat``。

    Raises:
        ValueError: 当曝光序列为空或 ``top_rate`` 非法时抛出。
        KeyError: 当曝光 item 不在 ``df_item_val`` 索引中时抛出。
    """
    if item_feat_domination is None:  # for yahoo
        return dict()
    if not 0 < top_rate <= 1:
        raise ValueError("top_rate must lie in (0, 1].")
    if len(all_acts_origin) == 0:
        raise ValueError("all_acts_origin must not be empty.")
    # if need_transform:
    #     all_acts_origin = lbe_item.inverse_transform(all_acts)
    # else:
    #     all_acts_origin = all_acts

    feat_dominate_dict = {}
    recommended_item_features = df_item_val.loc[all_acts_origin]

    if "feat" in item_feat_domination:  # for kuairec and kuairand
        sorted_items = item_feat_domination["feat"]
        dominated_values = get_dominant_values(sorted_items, top_rate)

        # dominated_value = sorted_items[0][0]
        recommended_item_features = recommended_item_features.filter(
            regex="^feat",
            axis=1,
        )
        feat_numpy = recommended_item_features.to_numpy().astype(int)

        dominate_array = np.zeros([len(feat_numpy)], dtype=bool)
        for value in dominated_values:
            equal_mat = (feat_numpy == value)
            has_dominate = equal_mat.sum(axis=1)
            dominate_array = dominate_array | has_dominate

        rate = dominate_array.sum() / len(recommended_item_features)
        feat_dominate_dict["ifeat_feat"] = rate

        ####################################
        #Todo: visual bar plot
        if draw_bar:
            feat_predicted = feat_numpy
            cats_predicted = feat_predicted.reshape(-1)
            pos_cat_predicted = cats_predicted[cats_predicted > 0]

            feat_dominate_dict["all_feats"] = pos_cat_predicted

        ####################################

    else:  # for coat
        for feat_name, sorted_items in item_feat_domination.items():
            dominated_values = get_dominant_values(sorted_items, top_rate)

            feat_numpy = recommended_item_features[feat_name].to_numpy().astype(int)

            dominate_array = np.zeros([len(feat_numpy)], dtype=bool)
            for value in dominated_values:
                has_dominate = (feat_numpy == value)
                # has_dominate = equal_mat
                dominate_array = dominate_array | has_dominate

            rate = dominate_array.sum() / len(recommended_item_features)

            feat_dominate_dict["ifeat_" + feat_name] = rate

    return feat_dominate_dict


def get_trajectory_feat_dominate_dict(
    df_item_val,
    trajectory_actions,
    item_feat_domination,
    top_rate=0.8,
    segment_length=DEFAULT_MCD_SEGMENT_LENGTH,
):
    """计算轨迹等权的整轨 MCD 与分段 MCD。

    每条轨迹先独立计算整轨 MCD；分段指标则将轨迹按连续
    ``segment_length`` 步切分，包含不足一个完整窗口的尾段。轨迹内各分段
    等权平均，再对所有轨迹等权平均。该函数不会替代旧版 pooled MCD。

    Args:
        df_item_val (pandas.DataFrame): 以原始 item id 为索引的物品特征表。
        trajectory_actions (Sequence[Sequence]): 各条非空轨迹的原始 item id
            序列。
        item_feat_domination (dict | None): 训练数据得到的各特征频次降序表；
            为 ``None`` 时返回空字典。
        top_rate (float): 主导类别累计频率阈值，范围为 ``(0, 1]``。
        segment_length (int): 每个连续分段的最大步数，必须大于 0。

    Returns:
        dict: 对每个旧版 MCD 键新增 ``_trajectory_mean`` 和
        ``_segment_mean_<segment_length>`` 两种轨迹等权指标。

    Raises:
        ValueError: 当分段长度非法、轨迹集合为空或包含空轨迹时抛出。
        KeyError: 当曝光 item 不在 ``df_item_val`` 索引中时抛出。

    Example:
        >>> trajectories = [[0, 1, 2], [3, 4]]
        >>> metrics = get_trajectory_feat_dominate_dict(
        ...     df_item_val, trajectories, domination, segment_length=2
        ... )
    """
    if item_feat_domination is None:
        return {}
    if not isinstance(segment_length, (int, np.integer)) or segment_length <= 0:
        raise ValueError("segment_length must be a positive integer.")
    if len(trajectory_actions) == 0:
        raise ValueError("trajectory_actions must not be empty.")

    trajectory_metric_values = {}
    segmented_metric_values = {}
    for trajectory_index, actions in enumerate(trajectory_actions):
        action_array = np.asarray(actions).reshape(-1)
        if action_array.size == 0:
            raise ValueError(
                f"trajectory_actions[{trajectory_index}] must not be empty."
            )

        whole_metrics = get_feat_dominate_dict(
            df_item_val,
            action_array,
            item_feat_domination,
            top_rate=top_rate,
        )
        segment_metrics = []
        for start in range(0, action_array.size, int(segment_length)):
            segment_metrics.append(
                get_feat_dominate_dict(
                    df_item_val,
                    action_array[start:start + int(segment_length)],
                    item_feat_domination,
                    top_rate=top_rate,
                )
            )

        for metric_name, metric_value in whole_metrics.items():
            trajectory_metric_values.setdefault(metric_name, []).append(
                float(metric_value)
            )
            segment_values = [
                float(metrics[metric_name]) for metrics in segment_metrics
            ]
            segmented_metric_values.setdefault(metric_name, []).append(
                float(np.mean(segment_values))
            )

    results = {}
    for metric_name, values in trajectory_metric_values.items():
        results[f"{metric_name}_trajectory_mean"] = float(np.mean(values))
        results[
            f"{metric_name}_segment_mean_{int(segment_length)}"
        ] = float(np.mean(segmented_metric_values[metric_name]))
    return results
