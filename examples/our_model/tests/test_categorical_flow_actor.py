"""类别空间 Flow Matching actor 的训练、采样与 checkpoint 测试。"""

from copy import deepcopy
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for relative_path in ["src/tianshou", "src", "."]:
    resolved_path = str(PROJECT_ROOT / relative_path)
    if resolved_path in sys.path:
        sys.path.remove(resolved_path)
    sys.path.insert(0, resolved_path)

from examples.our_model.models.chunk_actor import CategoricalFlowChunkActor
from examples.our_model.models.mac_agent import (
    ACTOR_TYPE_CATEGORICAL_FLOW,
    FLOW_SAMPLING_ONESTEP,
    FLOW_SAMPLING_TEACHER,
    MACAgent,
)
from examples.our_model.policy.action_mapper import ActionMapper

TEST_SEED = 23
"""类别 Flow 单元测试随机种子。"""


def build_flow_agent(
    distill_weight: float = 1.0,
    sampling_mode: str = FLOW_SAMPLING_ONESTEP,
) -> MACAgent:
    """构造小目录上的 categorical Flow MACAgent。

    Returns:
        MACAgent: CPU 上的两步 Flow actor agent。
    """
    embeddings = torch.eye(5)
    return MACAgent(
        state_dim=3,
        action_dim=5,
        chunk_size=2,
        actor_hidden_dims=(16,),
        value_hidden_dims=(8,),
        gamma=0.9,
        device=torch.device("cpu"),
        action_mapper=ActionMapper(embeddings, torch.device("cpu")),
        actor_type=ACTOR_TYPE_CATEGORICAL_FLOW,
        flow_steps=2,
        flow_temperature=0.1,
        flow_distill_weight=distill_weight,
        flow_sampling_mode=sampling_mode,
    )


class CategoricalFlowActorTest(unittest.TestCase):
    """验证 Flow 速度回归、直接离散采样、历史 mask 与类型契约。"""

    def setUp(self) -> None:
        """固定随机种子，避免候选采样测试发生非确定漂移。"""
        torch.manual_seed(TEST_SEED)
        torch.set_num_threads(1)

    def test_flow_loss_has_finite_gradients(self) -> None:
        """Flow Matching loss 应为有限标量并更新完整速度场参数。"""
        actor = CategoricalFlowChunkActor(
            state_dim=3, chunk_size=2, num_items=5,
            hidden_dims=(16,), flow_steps=2, temperature=0.1,
        )
        states = torch.randn(4, 3)
        item_ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]])
        valid = torch.tensor([[True, True], [True, False],
                              [True, True], [True, True]])
        loss, metrics = actor.flow_matching_loss(states, item_ids, valid)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [parameter.grad for parameter in actor.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(float(metrics["target_velocity_norm"]), 0)

    def test_flow_agent_update_and_masked_sampling(self) -> None:
        """Agent 应记录 Flow 指标并在每个独立候选终点屏蔽历史 item。"""
        agent = build_flow_agent()
        student_before = [
            parameter.detach().clone()
            for parameter in agent.actor_onestep.parameters()
        ]
        batch = {
            "observations": torch.randn(4, 3),
            "chunk_item_ids": torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]]),
            "chunk_valid": torch.ones(4, 2, dtype=torch.bool),
        }
        optimizer = torch.optim.Adam(agent.actor_parameters(), lr=1e-3)
        metrics = agent.pretrain_actor_update(batch, optimizer)
        self.assertIn("actor/flow_loss", metrics)
        self.assertIn("actor/distill_loss", metrics)
        self.assertGreater(metrics["actor/distill_loss"], 0)
        self.assertNotIn("actor/bc_loss", metrics)
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(student_before, agent.actor_onestep.parameters())
        ))
        history_mask = torch.tensor([[True, False, False, False, False]])
        with mock.patch.object(
            agent.actor, "sample_logits",
            side_effect=AssertionError("teacher sampling should not run"),
        ):
            candidate_ids, candidate_chunks = agent.sample_candidate_chunks(
                torch.zeros(1, 3), num_samples=8,
                recommended_mask=history_mask,
            )
        self.assertEqual(tuple(candidate_ids.shape), (8, 1, 2))
        self.assertEqual(tuple(candidate_chunks.shape), (8, 1, 10))
        self.assertFalse((candidate_ids == 0).any())

    def test_checkpoint_records_and_validates_actor_type(self) -> None:
        """Flow checkpoint 应显式记录 actor 类型并拒绝 categorical agent 误加载。"""
        agent = build_flow_agent()
        checkpoint = agent.checkpoint_state({"actor_type": ACTOR_TYPE_CATEGORICAL_FLOW})
        self.assertEqual(checkpoint["actor_type"], ACTOR_TYPE_CATEGORICAL_FLOW)
        self.assertIsNotNone(checkpoint["actor_onestep"])
        restored = build_flow_agent()
        restored.load_checkpoint_state(checkpoint)
        for source, target in zip(agent.actor.parameters(), restored.actor.parameters()):
            torch.testing.assert_close(source, target)
        for source, target in zip(
            agent.actor_onestep.parameters(), restored.actor_onestep.parameters(),
        ):
            torch.testing.assert_close(source, target)
        legacy_teacher = deepcopy(checkpoint)
        legacy_teacher["actor_onestep"] = None
        teacher_only = build_flow_agent(
            distill_weight=0.0, sampling_mode=FLOW_SAMPLING_TEACHER,
        )
        teacher_only.load_checkpoint_state(legacy_teacher)
        with self.assertRaises(ValueError):
            restored.load_checkpoint_state(legacy_teacher)
        incompatible = deepcopy(checkpoint)
        incompatible["actor_type"] = "categorical"
        with self.assertRaises(ValueError):
            restored.load_checkpoint_state(incompatible)


if __name__ == "__main__":
    unittest.main()
