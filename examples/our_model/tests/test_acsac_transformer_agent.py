"""模型化 ACSAC 因果 Transformer Q 分支测试。"""

from __future__ import annotations

import types
import unittest
import sys
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


if __name__ == "__main__":
    unittest.main()
