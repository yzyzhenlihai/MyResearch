"""验证 MAC_origin 的预编码状态监督、直接转移与 checkpoint 契约。"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for relative_path in [".", "src", "src/tianshou"]:
    resolved_path = str(PROJECT_ROOT / relative_path)
    if resolved_path not in sys.path:
        sys.path.insert(0, resolved_path)

from examples.our_model.data.observed_chunk_dataset import ObservedChunkDataset
from examples.our_model.data.trajectory_loader import TrajectoryBundle
from examples.our_model.models.chunk_dynamics import ChunkDynamics
from examples.our_model.models.mac_agent import MACAgent
from examples.our_model.policy.action_mapper import ActionMapper
from examples.our_model.tests.test_mac_agent_unique_sampling import attach_rollout_dependencies

TEST_SEED = 7
"""固定小型监督与采样实验的随机种子。"""


def build_fixture() -> tuple[torch.Tensor, ObservedChunkDataset, dict[str, torch.Tensor]]:
    """构造包含内部终止与显式预编码状态的微型轨迹。

    Returns:
        tuple[torch.Tensor, ObservedChunkDataset, dict[str, torch.Tensor]]:
            item embedding、数据集与完整 batch。
    """

    items = torch.eye(4)
    observations = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 1.0]], dtype=np.float32,
    )
    next_observations = np.asarray(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 1.0]], dtype=np.float32,
    )
    trajectory = {
        "user_id": 10,
        "actions": items.numpy(),
        "rewards": np.array([0.0, 1.0, 0.5, 0.2], dtype=np.float32),
        "terminals": np.array([False, True, False, True]),
        "observations": observations,
        "next_observations": next_observations,
    }
    bundle = TrajectoryBundle(
        [trajectory], 2, 4, 1, 4, "synthetic", {10: observations[0]},
    )
    dataset = ObservedChunkDataset(
        bundle, items, chunk_size=3, gamma=0.9, window_size=2,
    )
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    return items, dataset, batch


def build_agent(items: torch.Tensor, state_dim: int = 2) -> MACAgent:
    """构造使用真实 ChunkDynamics 的小型测试 agent。

    Args:
        items (torch.Tensor): item embedding 表。
        state_dim (int): pkl observation 维度。

    Returns:
        MACAgent: 具有规则奖励依赖的 CPU agent。
    """

    mapper = ActionMapper(items, torch.device("cpu"))
    agent = MACAgent(
        state_dim, 4, 3, (16,), (16,), 0.9, torch.device("cpu"), mapper,
    )
    attach_rollout_dependencies(agent, max_turn=9)
    agent.dynamics = ChunkDynamics(state_dim, 4, 3, (32,))
    agent.observation_spec = {
        "representation": "precomputed_avg_observation_v1",
        "state_dim": state_dim,
    }
    return agent


class IncrementDynamics(nn.Module):
    """记录输入并每块增加一，检测下一块是否使用预测状态。"""

    def __init__(self) -> None:
        """初始化调用记录。"""

        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        """记录输入，并让包含有效动作的行增加一。

        Args:
            states (torch.Tensor): 当前状态。
            actions (torch.Tensor): 动作块，仅为接口兼容。
            valid (torch.Tensor): 实际执行前缀。

        Returns:
            torch.Tensor: 推进后的状态。
        """

        del actions
        self.inputs.append(states.clone())
        return states + valid.any(-1, keepdim=True).float()


class ChunkDynamicsTest(unittest.TestCase):
    """覆盖预编码状态、模型学习和递归 rollout 数据契约。"""

    def setUp(self) -> None:
        """固定随机数与 CPU 线程并构造测试数据。"""

        torch.manual_seed(TEST_SEED)
        torch.set_num_threads(1)
        self.items, self.dataset, self.batch = build_fixture()

    def test_pkl_states_prefix_targets_and_episode_boundaries(self) -> None:
        """验证状态原样读取、前缀目标对齐且 chunk 不跨终止边界。"""

        self.assertEqual(self.batch["chunk_valid"].sum(-1).tolist(), [2, 1, 2, 1])
        torch.testing.assert_close(
            self.batch["observations"],
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 1.0]]),
        )
        torch.testing.assert_close(
            self.batch["prefix_next_observations"][0, :2],
            torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        )
        torch.testing.assert_close(
            self.batch["next_observations"][2], torch.tensor([4.0, 1.0]),
        )
        self.assertAlmostEqual(self.batch["rewards"][0].item(), 0.9, places=6)
        self.assertNotIn("history_vectors", self.batch)

    def test_prefix_mask_excludes_unexecuted_actions(self) -> None:
        """验证 padding 动作不影响预测、零步恒等且有效位必须连续。"""

        model = ChunkDynamics(self.dataset.state_dim, 4, 3, (16,))
        states = self.batch["observations"][:1]
        actions = self.batch["actions"][:1].reshape(1, 3, 4)
        valid = torch.tensor([[True, False, False]])
        expected = model(states, actions, valid)
        altered = actions.clone()
        altered[:, 1:] = 100
        torch.testing.assert_close(expected, model(states, altered, valid))
        torch.testing.assert_close(states, model(states, actions, torch.zeros_like(valid)))
        with self.assertRaises(ValueError):
            model(states, actions, torch.tensor([[False, True, False]]))

    def test_supervised_model_learns_precomputed_prefixes(self) -> None:
        """验证前缀监督降低预测误差，BC 与 Q/V 均可更新一次。"""

        agent = build_agent(self.items, self.dataset.state_dim)
        initial = agent.compute_dynamics_mse(self.batch)
        optimizer = torch.optim.Adam(agent.dynamics.parameters(), lr=0.01)
        for _ in range(100):
            agent.dynamics_update(self.batch, optimizer)
        self.assertLess(agent.compute_dynamics_mse(self.batch), initial * 0.2)
        agent.pretrain_actor_update(
            self.batch, torch.optim.Adam(agent.actor_parameters(), lr=0.001),
        )
        metrics = agent.qv_update(
            self.batch,
            torch.optim.Adam(agent.qv_parameters(), lr=0.001),
            num_samples_train=2,
            rollout_depth=2,
        )
        self.assertTrue(np.isfinite(list(metrics.values())).all())

    def test_predictions_recur_without_state_reencoding(self) -> None:
        """验证 imagined rollout 的下一块输入严格使用上一块预测。"""

        agent = build_agent(self.items, self.dataset.state_dim)
        agent.dynamics = IncrementDynamics()
        with torch.no_grad():
            rollout = agent.rollout_trajectory(
                self.batch, num_samples=2, rollout_depth=3,
            )
        self.assertTrue(rollout.steps[0].active.all())
        for step in range(3):
            torch.testing.assert_close(
                rollout.steps[step].states, self.batch["observations"] + step,
            )
        torch.testing.assert_close(
            agent.dynamics.inputs[1], rollout.steps[0].rollout.next_states,
        )

    def test_checkpoint_roundtrip_and_legacy_rejection(self) -> None:
        """验证 checkpoint 保存 dynamics 并拒绝旧格式与观测契约错配。"""

        agent = build_agent(self.items, self.dataset.state_dim)
        state = agent.checkpoint_state({})
        restored = build_agent(self.items, self.dataset.state_dim)
        restored.load_checkpoint_state(state)
        self.assertEqual(
            agent.compute_dynamics_mse(self.batch),
            restored.compute_dynamics_mse(self.batch),
        )
        legacy = copy.deepcopy(state)
        legacy.pop("format")
        with self.assertRaises(ValueError):
            restored.load_checkpoint_state(legacy, strict=False)
        state["observation_spec"] = {"wrong_state": True}
        with self.assertRaises(ValueError):
            restored.load_checkpoint_state(state, strict=False)


if __name__ == "__main__":
    unittest.main()
