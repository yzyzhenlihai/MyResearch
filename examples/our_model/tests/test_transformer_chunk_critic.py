"""严格 Q/V 架构对照使用的 Transformer Chunk Critic 测试。"""

import unittest

import torch

from examples.our_model.models.chunk_value import (
    CausalTransformerPrefixCritic,
    ChunkCritic,
    TransformerChunkCritic,
)


class TransformerChunkCriticTest(unittest.TestCase):
    """验证 Transformer Critic 的形状、梯度与输入边界检查。"""

    def setUp(self) -> None:
        """固定随机数和 CPU 线程，保证小型测试可复现。"""

        torch.manual_seed(31)
        torch.set_num_threads(1)

    def test_forward_shape_and_gradients(self) -> None:
        """完整 chunk 前向应输出逐样本标量并产生有限梯度。"""

        critic = TransformerChunkCritic(
            state_dim=3,
            action_dim=4,
            chunk_size=2,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        states = torch.randn(5, 3)
        chunks = torch.randn(5, 8)
        values = critic(states, chunks)
        self.assertEqual(tuple(values.shape), (5,))
        self.assertTrue(torch.isfinite(values).all())
        values.square().mean().backward()
        gradients = [parameter.grad for parameter in critic.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_invalid_inputs_and_dimensions_fail_early(self) -> None:
        """非法网络配置或输入形状应给出明确 ValueError。"""

        with self.assertRaises(ValueError):
            TransformerChunkCritic(
                state_dim=3,
                action_dim=4,
                chunk_size=2,
                hidden_dim=15,
                num_layers=1,
                num_heads=4,
            )
        critic = TransformerChunkCritic(
            state_dim=3,
            action_dim=4,
            chunk_size=2,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        with self.assertRaises(ValueError):
            critic(torch.randn(2, 4), torch.randn(2, 8))
        with self.assertRaises(ValueError):
            critic(torch.randn(2, 3), torch.randn(2, 7))

    def test_default_capacity_matches_mlp_control(self) -> None:
        """默认 Transformer 与原 MLP Critic 参数量差异应控制在 3% 内。"""

        mlp_critic = ChunkCritic(
            state_dim=42,
            chunk_action_dim=41 * 5,
            hidden_dims=(256, 256),
        )
        transformer_critic = TransformerChunkCritic(
            state_dim=42,
            action_dim=41,
            chunk_size=5,
        )
        mlp_parameters = sum(parameter.numel() for parameter in mlp_critic.parameters())
        transformer_parameters = sum(
            parameter.numel() for parameter in transformer_critic.parameters()
        )
        relative_difference = abs(transformer_parameters - mlp_parameters) / mlp_parameters
        self.assertLess(relative_difference, 0.03)

    def test_causal_prefix_outputs_ignore_future_actions(self) -> None:
        """修改后缀动作不得影响更短前缀的 Q 值。"""

        critic = CausalTransformerPrefixCritic(
            state_dim=3,
            action_dim=4,
            chunk_size=4,
            hidden_dim=16,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        ).eval()
        states = torch.randn(2, 3)
        chunks = torch.randn(2, 4, 4)
        altered = chunks.clone()
        altered[:, 2:] = torch.randn_like(altered[:, 2:]) * 100.0
        original_values = critic(states, chunks)
        altered_values = critic(states, altered)
        self.assertEqual(tuple(original_values.shape), (2, 4))
        torch.testing.assert_close(original_values[:, :2], altered_values[:, :2])
        self.assertFalse(torch.allclose(original_values[:, 2:], altered_values[:, 2:]))


if __name__ == "__main__":
    unittest.main()
