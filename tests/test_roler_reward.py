"""ROLeR 近邻奖励核心逻辑的单元测试。"""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from examples.advance.prepare_ROLeR import shape_reward
from src.core.envs.Simulated_Env.roler_reward import (
    batched_cosine_topk,
    compute_knn_reward,
    minmax_normalize,
)
from src.core.envs.Simulated_Env.roler_kuairec import (
    compute_official_kuairec_reward,
)


class BatchedCosineTopKTest(unittest.TestCase):
    """验证分批余弦近邻搜索的排序和自身排除行为。"""

    def setUp(self) -> None:
        """构造具有明确近邻关系的三用户矩阵。"""

        self.matrix = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32
        )

    def test_keeps_self_for_official_fidelity(self) -> None:
        """官方兼容模式应把用户自身作为第一近邻。"""

        indices, similarities = batched_cosine_topk(
            self.matrix, k=2, batch_size=1
        )
        np.testing.assert_array_equal(indices[:, 0], np.arange(3))
        np.testing.assert_allclose(similarities[:, 0], 1.0, atol=1e-6)
        self.assertEqual(indices[0, 1], 1)

    def test_can_exclude_self(self) -> None:
        """消融模式应从候选近邻中排除同下标用户。"""

        indices, _ = batched_cosine_topk(
            self.matrix, k=2, batch_size=2, exclude_self=True
        )
        for row, neighbors in enumerate(indices):
            self.assertNotIn(row, neighbors)


class KNNRewardTest(unittest.TestCase):
    """验证 ROLeR 奖励、不确定性和塑形公式。"""

    def setUp(self) -> None:
        """构造搜索指标与奖励来源不同的合成数据。"""

        self.indicator = np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32
        )
        self.reward = np.asarray(
            [[1.0, 3.0], [3.0, 7.0], [9.0, 11.0]], dtype=np.float32
        )

    def test_item_variance_matches_neighbor_feedback(self) -> None:
        """I-var 模式应输出近邻均值和逐物品方差。"""

        result = compute_knn_reward(
            reference_indicator_matrix=self.indicator,
            reference_reward_matrix=self.reward,
            k=2,
            uncertainty_type="I-var",
            batch_size=1,
            aggregation_batch_size=1,
        )
        np.testing.assert_allclose(result.reward[0], [2.0, 5.0], atol=1e-6)
        np.testing.assert_allclose(result.uncertainty[0], [1.0, 4.0], atol=1e-6)

    def test_weighted_normalized_reward_uses_cosine_weights(self) -> None:
        """II-weight-norm 应返回相似度归一化加权奖励且不重复惩罚。"""

        result = compute_knn_reward(
            reference_indicator_matrix=self.indicator,
            reference_reward_matrix=self.reward,
            k=2,
            uncertainty_type="II-weight-norm",
        )
        cosine = 0.8 / np.sqrt(0.8**2 + 0.2**2)
        expected = (self.reward[0] + cosine * self.reward[1]) / (1.0 + cosine)
        np.testing.assert_allclose(result.reward[0], expected, atol=1e-6)
        np.testing.assert_array_equal(result.uncertainty, 0.0)

    def test_paper_formula_uses_normalized_uncertainty(self) -> None:
        """论文公式应以归一化不确定性衰减近邻奖励。"""

        uncertainty = np.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
        reward = np.full_like(uncertainty, 10.0)
        shaped = shape_reward(reward, uncertainty, "paper", 1.0)
        expected = reward * (1.0 - minmax_normalize(uncertainty))
        np.testing.assert_allclose(shaped, expected, atol=1e-6)
        self.assertAlmostEqual(float(shaped.max()), 10.0)
        self.assertAlmostEqual(float(shaped.min()), 0.0)

    def test_rejects_unsupported_uncertainty_type(self) -> None:
        """未知的不确定性类型应立即失败，避免静默采用错误公式。"""

        with self.assertRaisesRegex(ValueError, "uncertainty_type"):
            compute_knn_reward(self.indicator, k=1, uncertainty_type="unknown")


class KuaiRecBlockTest(unittest.TestCase):
    """验证 KuaiRec 使用训练用户到测试用户的跨块近邻，而非测试集自近邻。"""

    def test_builds_official_cross_user_blocks(self) -> None:
        """测试用户应从训练用户反馈中继承测试物品奖励。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"user_id": [0, 1, 2]}).to_csv(
                root / "user_features.csv", index=False
            )
            pd.DataFrame({"video_duration": [1.0, 1.0, 1.0]}).to_csv(
                root / "video_duration.csv", index=False
            )
            pd.DataFrame({"user_id_small": [2]}).to_csv(
                root / "user_id_small.csv", index=False
            )
            pd.DataFrame({"item_id_small": [2]}).to_csv(
                root / "item_id_small.csv", index=False
            )
            pd.DataFrame(
                {
                    "user_id": [0, 0, 0, 1, 1, 1, 2, 2],
                    "item_id": [0, 1, 2, 0, 1, 2, 0, 1],
                    "watch_ratio": [1.0, 0.0, 3.0, 0.0, 1.0, 5.0, 0.9, 0.1],
                }
            ).to_csv(root / "big_matrix_processed.csv", index=False)

            result = compute_official_kuairec_reward(
                data_root=root,
                k=1,
                uncertainty_type="II-weight-norm",
                search_batch_size=1,
                aggregation_batch_size=1,
                csv_chunk_size=3,
            )
            self.assertEqual(result.reward.shape, (1, 1))
            np.testing.assert_allclose(result.reward, [[3.0]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()

