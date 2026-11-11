"""分段 MCD 聚合口径的单元测试。"""

import unittest

import numpy as np
import pandas as pd

from src.core.evaluation.utils import (
    get_feat_dominate_dict,
    get_trajectory_feat_dominate_dict,
)


class SegmentedMCDTest(unittest.TestCase):
    """验证 pooled、整轨等权和分段等权 MCD 的差异。"""

    def setUp(self) -> None:
        """构造一个主导类别 item 与一个非主导类别 item。"""
        self.item_features = pd.DataFrame(
            {"feat0": [1, 2]},
            index=[0, 1],
        )
        self.domination = {"feat": [(1, 80), (2, 20)]}

    def test_tail_segment_and_trajectories_are_equal_weighted(self) -> None:
        """不足 30 步的尾段应与完整分段、其他轨迹保持等权。"""
        first_trajectory = np.asarray(
            [0] * 30 + [1] * 30 + [1] * 30 + [0] * 10
        )
        second_trajectory = np.asarray([1] * 30)
        trajectories = [first_trajectory, second_trajectory]

        legacy_metrics = get_feat_dominate_dict(
            self.item_features,
            np.concatenate(trajectories),
            self.domination,
        )
        trajectory_metrics = get_trajectory_feat_dominate_dict(
            self.item_features,
            trajectories,
            self.domination,
            segment_length=30,
        )

        self.assertAlmostEqual(legacy_metrics["ifeat_feat"], 40 / 130)
        self.assertAlmostEqual(
            trajectory_metrics["ifeat_feat_trajectory_mean"],
            0.2,
        )
        self.assertAlmostEqual(
            trajectory_metrics["ifeat_feat_segment_mean_30"],
            0.25,
        )

    def test_empty_trajectory_is_rejected(self) -> None:
        """空轨迹必须显式报错，避免静默产生 NaN。"""
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            get_trajectory_feat_dominate_dict(
                self.item_features,
                [[0], []],
                self.domination,
                segment_length=30,
            )

    def test_nonpositive_segment_length_is_rejected(self) -> None:
        """非正分段长度必须在进入聚合前被拒绝。"""
        with self.assertRaisesRegex(ValueError, "positive integer"):
            get_trajectory_feat_dominate_dict(
                self.item_features,
                [[0]],
                self.domination,
                segment_length=0,
            )


if __name__ == "__main__":
    unittest.main()
