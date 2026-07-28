"""DORL-MAC 开环消融 CCR@W 与汇总逻辑测试。"""

from __future__ import annotations

import unittest

import numpy as np

from examples.our_model.runners.evaluation_utils import (
    build_open_loop_metrics,
    compute_window_completion_metrics,
)


class WindowCompletionMetricsTest(unittest.TestCase):
    """验证可配置固定窗口 CCR@W 的计数与删失语义。"""

    def test_natural_termination_counts_partial_window_as_failure(self) -> None:
        """验证自然退出造成的未完成尾窗口进入 CCR 分母。"""

        metrics = compute_window_completion_metrics(
            episode_lengths=np.asarray([12, 5, 3]),
            completion_window=5,
            censor_limit=100,
        )

        self.assertAlmostEqual(metrics["rate"], 0.6)
        self.assertEqual(metrics["completed_windows"], 3.0)
        self.assertEqual(metrics["eligible_windows"], 5.0)
        self.assertEqual(metrics["censored_windows"], 0.0)

    def test_max_turn_partial_window_is_right_censored(self) -> None:
        """验证因评估上限结束的尾窗口不进入 CCR 分母。"""

        metrics = compute_window_completion_metrics(
            episode_lengths=np.asarray([12]),
            completion_window=5,
            censor_limit=12,
        )

        self.assertEqual(metrics["rate"], 1.0)
        self.assertEqual(metrics["completed_windows"], 2.0)
        self.assertEqual(metrics["eligible_windows"], 2.0)
        self.assertEqual(metrics["censored_windows"], 1.0)

    def test_dynamic_window_and_adr_are_merged_per_branch(self) -> None:
        """验证动态 CCR@W key 与 Collector policy hook ADR 按分支合并。"""

        results = {
            "lens": np.asarray([8, 3]),
            "NX_0_lens": np.asarray([4, 4]),
            "NX_20_lens": np.asarray([20]),
            "open_loop/FB/ADR": 0.75,
            "open_loop/FB/ADR_exposed": 0.5,
        }
        metrics = build_open_loop_metrics(
            results=results,
            completion_window=4,
            max_turn=20,
            force_length=20,
        )

        self.assertAlmostEqual(metrics["open_loop/FB/CCR@4"], 2.0 / 3.0)
        self.assertEqual(metrics["open_loop/NX_0/CCR@4"], 1.0)
        self.assertEqual(metrics["open_loop/NX_20/CCR@4"], 1.0)
        self.assertEqual(metrics["open_loop/FB/ADR"], 0.75)
        self.assertEqual(metrics["open_loop/FB/ADR_exposed"], 0.5)

    def test_invalid_completion_window_is_rejected(self) -> None:
        """验证非正窗口和小于窗口的删失上限被拒绝。"""

        with self.assertRaisesRegex(ValueError, "completion_window must be positive"):
            compute_window_completion_metrics(
                episode_lengths=np.asarray([1]),
                completion_window=0,
                censor_limit=10,
            )
        with self.assertRaisesRegex(ValueError, "censor_limit"):
            compute_window_completion_metrics(
                episode_lengths=np.asarray([1]),
                completion_window=5,
                censor_limit=4,
            )


if __name__ == "__main__":
    unittest.main()
