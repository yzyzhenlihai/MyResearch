"""DARLR selector 标准化工具的单元测试。"""

import unittest

import torch

from src.core.darlr.normalization import standardize_tensor


NORMALIZATION_EPS = 1.0e-8
"""单元测试采用的标准化数值稳定项。"""

STATISTIC_TOLERANCE = 1.0e-6
"""标准化后均值和标准差断言使用的绝对误差。"""


class StandardizeTensorTest(unittest.TestCase):
    """验证 selector 奖励与优势共享标准化函数的数值语义。"""

    def test_standardizes_nonconstant_tensor(self) -> None:
        """验证非恒定张量被转换为零均值、单位总体标准差。"""

        values = torch.tensor(
            [[-2.0, 0.0, 2.0], [4.0, 6.0, 8.0]],
            dtype=torch.float32,
        )

        standardized = standardize_tensor(values, eps=NORMALIZATION_EPS)

        self.assertEqual(standardized.shape, values.shape)
        self.assertEqual(standardized.dtype, values.dtype)
        self.assertAlmostEqual(
            float(standardized.mean().item()),
            0.0,
            delta=STATISTIC_TOLERANCE,
        )
        self.assertAlmostEqual(
            float(standardized.std(unbiased=False).item()),
            1.0,
            delta=STATISTIC_TOLERANCE,
        )

    def test_constant_tensor_returns_finite_zeros(self) -> None:
        """验证零方差输入不会产生 NaN，并退化为全零张量。"""

        values = torch.full((2, 3), 7.0, dtype=torch.float32)

        standardized = standardize_tensor(values, eps=NORMALIZATION_EPS)

        self.assertTrue(bool(torch.isfinite(standardized).all().item()))
        self.assertTrue(
            torch.equal(standardized, torch.zeros_like(standardized))
        )

    def test_preserves_autograd_connection(self) -> None:
        """验证标准化不会截断 selector actor 所需的自动微分图。"""

        values = torch.tensor(
            [1.0, 2.0, 5.0],
            dtype=torch.float32,
            requires_grad=True,
        )

        standardized = standardize_tensor(values, eps=NORMALIZATION_EPS)
        loss = standardized.square().sum()
        loss.backward()

        self.assertTrue(standardized.requires_grad)
        self.assertIsNotNone(values.grad)
        self.assertTrue(bool(torch.isfinite(values.grad).all().item()))

    def test_rejects_nonpositive_epsilon(self) -> None:
        """验证非正稳定项被明确拒绝，避免隐藏数值配置错误。"""

        values = torch.tensor([1.0, 2.0], dtype=torch.float32)

        for invalid_eps in (0.0, -NORMALIZATION_EPS):
            with self.subTest(invalid_eps=invalid_eps):
                with self.assertRaises(ValueError):
                    standardize_tensor(values, eps=invalid_eps)


if __name__ == "__main__":
    unittest.main()
