"""KuaiRec 高奖励用户筛选统计的回归测试。"""

import numpy as np
import pytest

from analysis.kuai_user_selection import (
    build_reward_selection_summary,
    compute_item_reward_statistics,
    compute_user_reward_statistics,
)


def test_rank_users_by_reward_threshold_rate_then_mean() -> None:
    """验证主排序使用阈值占比、并以均值作为稳定次排序。"""

    reward_matrix = np.array(
        [
            [1.26, 1.30, 0.00, 0.00],
            [2.00, 1.40, 0.20, 0.20],
            [1.27, 0.10, 0.10, 0.10],
        ],
        dtype=np.float64,
    )
    statistics = compute_user_reward_statistics(
        reward_matrix,
        raw_user_ids=[30, 20, 10],
        target_reward=1.26,
    )

    assert statistics["user_id"].tolist() == [20, 30, 10]
    assert statistics["rank"].tolist() == [1, 2, 3]
    assert statistics.iloc[1]["reward_exact_target_count"] == 1
    assert statistics.iloc[0]["reward_ge_target_rate"] == pytest.approx(0.5)


def test_summary_uses_selected_matrix_rows() -> None:
    """验证 Top-K 子矩阵摘要与排序后的矩阵行严格对应。"""

    reward_matrix = np.array(
        [[0.0, 0.0], [1.3, 1.4], [1.26, 0.0]],
        dtype=np.float64,
    )
    statistics = compute_user_reward_statistics(
        reward_matrix,
        raw_user_ids=[100, 200, 300],
        target_reward=1.26,
    )
    summary = build_reward_selection_summary(
        reward_matrix,
        statistics,
        top_k=1,
    )

    assert summary["top_k_matrix_mean_reward"] == pytest.approx(1.35)
    assert summary["top_k_reward_ge_target_rate"] == pytest.approx(1.0)


def test_reject_non_finite_reward_matrix() -> None:
    """验证非有限 reward 会被显式拒绝。"""

    reward_matrix = np.array([[0.0, np.nan]], dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        compute_user_reward_statistics(reward_matrix, raw_user_ids=[1])


def test_rank_items_by_selected_user_mean_reward() -> None:
    """验证 item 排序只使用指定用户，并以真实 reward 均值为主键。"""

    reward_matrix = np.array(
        [
            [9.0, 0.0, 0.0],
            [0.2, 1.3, 1.1],
            [0.2, 1.1, 1.5],
        ],
        dtype=np.float64,
    )
    statistics = compute_item_reward_statistics(
        reward_matrix,
        raw_item_ids=[100, 200, 300],
        selected_user_indexes=[1, 2],
    )

    assert statistics["item_id"].tolist() == [300, 200, 100]
    assert statistics["rank"].tolist() == [1, 2, 3]
    assert statistics.iloc[0]["mean_reward"] == pytest.approx(1.3)
