"""DARLR selector 训练指标聚合与回调的单元测试。"""

import math
import unittest

from src.core.darlr.selector_metrics import (
    SelectorTrainingMetricsCallback,
    WeightedScalarAccumulator,
)


class WeightedScalarAccumulatorTest(unittest.TestCase):
    """验证 selector 标量聚合器的加权、极值和异常值处理。"""

    def test_aggregates_epoch_metrics(self) -> None:
        """验证普通指标加权平均、极值指标和计数指标的聚合语义。"""

        accumulator = WeightedScalarAccumulator()
        accumulator.update(
            {
                "reward_mean": 2.0,
                "reward_std": 1.0,
                "reward_min": 1.0,
                "reward_max": 3.0,
            },
            weight=2,
        )
        accumulator.update(
            {
                "reward_mean": 4.0,
                "reward_std": 0.0,
                "reward_min": 0.5,
                "reward_max": 5.0,
            },
            weight=1,
        )

        summary = accumulator.summary()

        self.assertAlmostEqual(summary["selector/train/reward_mean"], 8.0 / 3.0)
        self.assertAlmostEqual(
            summary["selector/train/reward_std"],
            math.sqrt(14.0 / 9.0),
        )
        self.assertAlmostEqual(summary["selector/train/reward_min"], 0.5)
        self.assertAlmostEqual(summary["selector/train/reward_max"], 5.0)
        self.assertAlmostEqual(summary["selector/train/minibatch_count"], 2.0)
        self.assertAlmostEqual(summary["selector/train/sample_count"], 3.0)
        self.assertAlmostEqual(
            summary["selector/train/nonfinite_value_count"],
            0.0,
        )

    def test_tracks_nonfinite_values(self) -> None:
        """验证非有限指标不会污染曲线且能够被单独计数。"""

        accumulator = WeightedScalarAccumulator()
        accumulator.update(
            {
                "finite_metric": 1.0,
                "nan_metric": math.nan,
                "infinite_metric": math.inf,
            },
            weight=4,
        )

        summary = accumulator.summary()

        self.assertAlmostEqual(summary["selector/train/finite_metric"], 1.0)
        self.assertNotIn("selector/train/nan_metric", summary)
        self.assertNotIn("selector/train/infinite_metric", summary)
        self.assertAlmostEqual(
            summary["selector/train/nonfinite_value_count"],
            2.0,
        )

    def test_rejects_invalid_weight(self) -> None:
        """验证非法样本权重会立即失败，避免产生错误的 epoch 均值。"""

        accumulator = WeightedScalarAccumulator()

        with self.assertRaisesRegex(ValueError, "positive"):
            accumulator.update({"reward_mean": 1.0}, weight=0)


class SelectorTrainingMetricsCallbackTest(unittest.TestCase):
    """验证 selector 训练指标回调的 epoch 生命周期。"""

    def test_resets_and_exports_metrics(self) -> None:
        """验证 epoch 回调能够重置并导出策略侧的 selector 统计。"""

        class FakeSelectorPolicy:
            """提供回调测试所需最小指标接口的伪策略。"""

            def __init__(self) -> None:
                """初始化重置计数器。"""

                self.reset_count = 0

            def reset_selector_training_metrics(self) -> None:
                """记录回调触发的重置次数。"""

                self.reset_count += 1

            def get_selector_training_metrics(self):
                """返回固定的 selector 诊断指标。"""

                return {"selector/train/entropy": 0.25}

        selector_policy = FakeSelectorPolicy()
        callback = SelectorTrainingMetricsCallback(selector_policy)

        callback.on_train_begin()
        callback.on_epoch_begin(epoch=3)
        metrics = callback.on_epoch_end(epoch=3, results={})

        self.assertEqual(selector_policy.reset_count, 2)
        self.assertEqual(metrics, {"selector/train/entropy": 0.25})


if __name__ == "__main__":
    unittest.main()
