"""模型化 ACSAC 因果 Transformer Q 分支测试。"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for relative_path in [".", "src", "src/tianshou"]:
    resolved_path = str(PROJECT_ROOT / relative_path)
    if resolved_path not in sys.path:
        sys.path.insert(0, resolved_path)

from examples.our_model.models.acsac_agent import (
    ACSAC_CHECKPOINT_FORMAT,
    ACSACTransformerAgent,
)
from examples.our_model.models.mac_agent import RolloutHistoryState
from examples.our_model.policy.action_mapper import ActionMapper
from examples.our_model.tests.test_chunk_dynamics import build_fixture
from examples.our_model.tests.test_mac_agent_unique_sampling import (
    attach_rollout_dependencies,
)


def build_agent() -> ACSACTransformerAgent:
    """构造小型 CPU ACSAC agent。"""

    item_embeddings = torch.eye(4)
    mapper = ActionMapper(item_embeddings, torch.device("cpu"))
    agent = ACSACTransformerAgent(
        state_dim=2,
        action_dim=4,
        chunk_size=3,
        actor_hidden_dims=(8,),
        value_hidden_dims=(8,),
        gamma=0.9,
        device=torch.device("cpu"),
        action_mapper=mapper,
        critic_transformer_hidden_dim=16,
        critic_transformer_layers=1,
        critic_transformer_heads=4,
        critic_transformer_dropout=0.0,
    )
    attach_rollout_dependencies(agent, max_turn=9)
    agent.observation_spec = {
        "representation": "precomputed_avg_observation_v1",
        "state_dim": 2,
    }
    return agent


class FixedPrefixCritic(nn.Module):
    """按输入顺序返回预设的候选前缀价值。"""

    def __init__(self, values: torch.Tensor) -> None:
        """保存预设价值表。"""

        super().__init__()
        self.register_buffer("values", values)

    def forward(self, states: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        """忽略输入内容并按 batch 截取价值。"""

        del states, chunks
        return self.values


class LearnablePrefixCritic(nn.Module):
    """为每个前缀维护一个独立标量参数的测试 Q 网络。"""

    def __init__(self, initial_values: torch.Tensor) -> None:
        """初始化可学习的前缀价值。

        Args:
            initial_values (torch.Tensor): 形状 ``(K,)`` 的初始前缀价值。
        """

        super().__init__()
        self.prefix_values = nn.Parameter(initial_values.clone().float())

    def forward(self, states: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        """将独立前缀参数扩展到输入 batch。

        Args:
            states (torch.Tensor): 状态 batch，仅用于确定 batch size。
            chunks (torch.Tensor): chunk batch，仅用于接口兼容。

        Returns:
            torch.Tensor: 形状 ``(B,K)`` 的可学习前缀价值。
        """

        del chunks
        return self.prefix_values.unsqueeze(0).expand(states.shape[0], -1)


class ZeroPrefixCritic(nn.Module):
    """返回动态 batch 形状全零价值的固定 target Q 网络。"""

    def __init__(self, chunk_size: int) -> None:
        """保存输出前缀数量。

        Args:
            chunk_size (int): 输出前缀数量 K，必须为正数。

        Raises:
            ValueError: 当 ``chunk_size`` 非正时抛出。
        """

        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        self.chunk_size = chunk_size

    def forward(self, states: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        """返回与状态 batch 对齐的零价值。

        Args:
            states (torch.Tensor): 状态 batch，形状 ``(B,state_dim)``。
            chunks (torch.Tensor): chunk batch，仅用于保持接口一致。

        Returns:
            torch.Tensor: 形状 ``(B,K)`` 的零张量。
        """

        del chunks
        return states.new_zeros((states.shape[0], self.chunk_size))


class ItemIdRewardModel:
    """用 ``item_id + 1`` 生成确定性 step reward 的测试模型。"""

    def reward_components(
        self,
        raw_user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """返回便于手算各前缀累计回报的 reward 分量。

        Args:
            raw_user_ids (torch.Tensor): 用户 id，本测试不使用。
            item_ids (torch.Tensor): 当前 item id，形状 ``(B,)``。
            history_item_ids (torch.Tensor): 历史 item id，本测试不使用。

        Returns:
            dict[str, torch.Tensor]: reward、预测、熵和不确定性分量。
        """

        del raw_user_ids, history_item_ids
        reward = item_ids.float() + 1.0
        zeros = torch.zeros_like(reward)
        return {
            "reward": reward,
            "pred_reward": reward,
            "entropy": zeros,
            "uncertainty": zeros,
        }


class ACSACTransformerAgentTest(unittest.TestCase):
    """验证联合选择、可变长度 rollout、Q-only 更新和 checkpoint。"""

    def setUp(self) -> None:
        """固定随机性与 CPU 线程。"""

        torch.manual_seed(41)
        torch.set_num_threads(1)

    def test_joint_candidate_and_horizon_selection(self) -> None:
        """argmax 应在候选数与 K 的笛卡尔积上进行。"""

        agent = build_agent()
        candidate_ids = torch.tensor([[[0, 1, 2]], [[3, 2, 1]]])
        candidate_chunks = agent._item_ids_to_chunk_vectors(candidate_ids)

        def fixed_candidates(
            self: ACSACTransformerAgent,
            states: torch.Tensor,
            num_samples: int,
            recommended_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del self, states, num_samples, recommended_mask
            return candidate_ids, candidate_chunks

        agent.sample_candidate_chunks = types.MethodType(fixed_candidates, agent)
        agent.critic = FixedPrefixCritic(torch.tensor([
            [1.0, 2.0, 3.0],
            [0.0, 8.0, 7.0],
        ]))
        selected_ids, _, lengths = agent.select_adaptive_chunks(
            torch.zeros(1, 2),
            num_samples=2,
        )
        self.assertEqual(selected_ids.tolist(), [[3, 2, 1]])
        self.assertEqual(lengths.tolist(), [2])

    def test_selected_horizon_controls_rollout_prefix(self) -> None:
        """所选 H 必须同时约束有效动作、reward 步数和 Dynamics valid mask。"""

        agent = build_agent()
        agent._batch_user_ids = torch.zeros(2, dtype=torch.long)
        history = RolloutHistoryState(
            states=torch.zeros(2, 2),
            leave_history=torch.full((2, 1), -1, dtype=torch.long),
            recommended_mask=torch.zeros(2, 4, dtype=torch.bool),
            env_step=torch.zeros(2, dtype=torch.long),
            terminated=torch.zeros(2, dtype=torch.bool),
        )
        result, _ = agent._rollout_one_chunk(
            torch.tensor([[0, 1, 2], [3, 2, 1]]),
            history,
            repeat_policy="truncate",
            execution_lengths=torch.tensor([1, 2]),
        )
        self.assertEqual(result.effective_steps.tolist(), [1, 2])
        self.assertEqual(
            result.chunk_valid.tolist(),
            [[True, False, False], [True, True, False]],
        )

    def test_q_only_td_lambda_update_and_checkpoint(self) -> None:
        """训练应只含 Q/target-Q，并产生有限的 MSE 与自适应 H 指标。"""

        _, _, batch = build_fixture()
        agent = build_agent()
        self.assertFalse(hasattr(agent, "value"))
        self.assertFalse(hasattr(agent, "target_value"))
        optimizer = torch.optim.Adam(agent.qv_parameters(), lr=0.001)
        metrics = agent.qv_update(
            batch,
            optimizer,
            num_samples_train=2,
            rollout_depth=2,
            lambda_chunk=0.7,
        )
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        self.assertIn("rollout/selected_h_mean", metrics)
        self.assertNotIn("value/value_loss", metrics)
        checkpoint = agent.checkpoint_state({"critic_backbone": "acsac_transformer"})
        self.assertEqual(checkpoint["format"], ACSAC_CHECKPOINT_FORMAT)
        self.assertNotIn("value", checkpoint)
        restored = build_agent()
        restored.load_checkpoint_state(checkpoint)
        for source, target in zip(agent.critic.parameters(), restored.critic.parameters()):
            torch.testing.assert_close(source, target)

    def test_all_prefix_auxiliary_loss_updates_every_horizon(self) -> None:
        """辅助一步 TD 应让未被 argmax 选中的合法 H 也获得梯度。"""

        _, _, batch = build_fixture()
        agent = build_agent()
        agent.invalid_action_penalty = 0.0
        agent.critic = LearnablePrefixCritic(torch.tensor([3.0, 2.0, 1.0]))
        agent.target_critic = ZeroPrefixCritic(agent.chunk_size)
        optimizer = torch.optim.SGD(
            agent.qv_parameters(include_dynamics=False),
            lr=0.01,
        )
        before = agent.critic.prefix_values.detach().clone()

        metrics = agent.qv_update(
            batch,
            optimizer,
            num_samples_train=2,
            rollout_depth=2,
            lambda_chunk=0.7,
        )

        after = agent.critic.prefix_values.detach()
        self.assertTrue(bool(torch.all(after < before)))
        self.assertAlmostEqual(metrics["critic/selected_td_lambda_loss"], 9.0)
        self.assertAlmostEqual(
            metrics["critic/all_prefix_aux_loss"],
            14.0 / 3.0,
            places=5,
        )
        self.assertAlmostEqual(
            metrics["critic/critic_loss"],
            9.0 + 14.0 / 3.0,
            places=5,
        )
        self.assertEqual(metrics["rollout/legal_prefixes_mean"], 3.0)

    def test_all_prefix_targets_use_individual_discounted_rewards(self) -> None:
        """每个 H 应使用自身累计 reward，而不是复制 selected-H 的目标。"""

        agent = build_agent()
        agent.reward_model = ItemIdRewardModel()
        agent.invalid_action_penalty = 0.0
        agent.target_critic = ZeroPrefixCritic(agent.chunk_size)
        batch = {
            "observations": torch.zeros(1, agent.state_dim),
            "leave_history_item_ids": torch.full((1, 1), -1, dtype=torch.long),
            "recommended_mask": torch.zeros(
                1,
                agent.num_items,
                dtype=torch.bool,
            ),
            "env_step": torch.zeros(1, dtype=torch.long),
            "initial_terminals": torch.zeros(1, dtype=torch.bool),
            "user_id": torch.zeros(1, dtype=torch.long),
        }
        first_step = types.SimpleNamespace(
            selected_item_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        )

        target = agent._all_prefix_one_step_target(
            batch=batch,
            first_step=first_step,
            num_samples=2,
            repeat_policy="truncate",
            leave_policy="penalty",
        )

        expected = torch.tensor([[1.0, 2.8, 5.23]])
        torch.testing.assert_close(target.values, expected)
        self.assertTrue(bool(target.valid.all()))
        self.assertEqual(target.effective_steps.tolist(), [[1, 2, 3]])

    def test_zero_aux_weight_skips_all_prefix_rollout(self) -> None:
        """权重为零时应恢复 selected-H 路径且不构造额外前缀目标。"""

        _, _, batch = build_fixture()
        agent = build_agent()
        agent.all_prefix_aux_weight = 0.0

        def fail_if_called(*args: object, **kwargs: object) -> None:
            """在被意外调用时立即使测试失败。

            Args:
                *args (object): 被替换方法的位置参数。
                **kwargs (object): 被替换方法的关键字参数。

            Raises:
                AssertionError: 本函数任何时候都不应被调用。
            """

            del args, kwargs
            raise AssertionError("all-prefix rollout should be skipped")

        agent._all_prefix_one_step_target = fail_if_called
        optimizer = torch.optim.Adam(
            agent.qv_parameters(include_dynamics=False),
            lr=0.001,
        )

        metrics = agent.qv_update(
            batch,
            optimizer,
            num_samples_train=2,
            rollout_depth=1,
        )

        self.assertEqual(metrics["critic/all_prefix_aux_loss"], 0.0)
        self.assertEqual(metrics["critic/all_prefix_aux_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
