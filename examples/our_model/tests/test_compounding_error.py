"""K=1 长程累计误差实验核心逻辑测试。"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from examples.our_model.runners.eval_compounding_error import (
    evaluate_dynamics_batch,
    paired_user_bootstrap,
    valid_start_indices,
)


class AdditiveBiasDynamics:
    """每步在真实加一转移上额外加入固定偏差的测试 dynamics。"""

    def __init__(self, bias: float) -> None:
        """初始化固定单步偏差。

        Args:
            bias (float): 每次递归额外加入的状态偏差。
        """

        self.bias = float(bias)

    def __call__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """执行每步 `state + 1 + bias` 的确定性转移。

        Args:
            states (torch.Tensor): 当前状态 `(B,1)`。
            actions (torch.Tensor): 占位动作 `(B,1,1)`。
            valid (torch.Tensor): 有效动作标记 `(B,1)`。

        Returns:
            torch.Tensor: 加入固定偏差后的下一状态。
        """

        del actions
        increment = torch.where(
            valid[:, :1],
            torch.full_like(states, 1.0 + self.bias),
            torch.zeros_like(states),
        )
        return states + increment


class CompoundingErrorTest(unittest.TestCase):
    """验证合法窗口、累计误差和配对 bootstrap。"""

    def test_valid_starts_do_not_cross_terminal(self) -> None:
        """验证长程窗口不会越过中间 terminal。"""

        starts = valid_start_indices(
            terminals=np.array([False, True, False, False]),
            trajectory_length=4,
            max_horizon=2,
            min_start=0,
        )

        np.testing.assert_array_equal(starts, np.array([0, 2]))

    def test_free_running_accumulates_but_teacher_forcing_stays_local(self) -> None:
        """验证递归自反馈累积固定偏差，真实状态输入只保留局部误差。"""

        teacher_states = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
        targets = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        actions = torch.zeros((1, 4, 1))
        result = evaluate_dynamics_batch(
            dynamics=AdditiveBiasDynamics(bias=0.1),
            teacher_states=teacher_states,
            actions=actions,
            targets=targets,
            state_scale=torch.ones(1),
            valid_dimensions=torch.ones(1, dtype=torch.bool),
        )

        np.testing.assert_allclose(
            result["free_nrmse"][0], [0.1, 0.2, 0.3, 0.4], atol=1.0e-6,
        )
        np.testing.assert_allclose(
            result["teacher_nrmse"][0], [0.1, 0.1, 0.1, 0.1], atol=1.0e-6,
        )

    def test_paired_bootstrap_preserves_curve_shapes(self) -> None:
        """验证用户配对 bootstrap 输出维度和正传播增量。"""

        free_curves = np.array([
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        ])
        teacher_curves = np.ones_like(free_curves)
        result = paired_user_bootstrap(
            free_curves=free_curves,
            teacher_curves=teacher_curves,
            bootstrap_samples=20,
            bootstrap_seed=7,
        )

        self.assertEqual(result["mean"].shape, (2, 3))
        self.assertEqual(result["ci_low"].shape, (2, 3))
        self.assertTrue(np.all(result["bootstrap_gap_slopes"] > 0.0))


if __name__ == "__main__":
    unittest.main()
