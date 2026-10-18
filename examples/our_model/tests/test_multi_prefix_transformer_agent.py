"""R/T/D/C 四头 ACSAC Transformer agent 测试。"""

from __future__ import annotations

import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for relative_path in [".", "src", "src/tianshou"]:
    resolved_path = str(PROJECT_ROOT / relative_path)
    if resolved_path not in sys.path:
        sys.path.insert(0, resolved_path)

from examples.our_model.models.multi_prefix_agent import (  # noqa: E402
    MULTI_PREFIX_CHECKPOINT_FORMAT,
    MultiPrefixTransformerAgent,
)
from examples.our_model.models.mac_agent import RolloutHistoryState  # noqa: E402
from examples.our_model.policy.action_mapper import ActionMapper  # noqa: E402
from examples.our_model.rave.model import Utility  # noqa: E402
from examples.our_model.runners.run_dorl_mac import build_common_argv  # noqa: E402
from examples.our_model.tests.test_chunk_dynamics import build_fixture  # noqa: E402
from examples.our_model.tests.test_mac_agent_unique_sampling import (  # noqa: E402
    attach_rollout_dependencies,
)


def build_multi_agent() -> MultiPrefixTransformerAgent:
    """构造小型 CPU 四头 agent。

    Returns:
        MultiPrefixTransformerAgent: K=3、无 dropout 的确定性测试 agent。
    """

    item_embeddings = torch.eye(4)
    mapper = ActionMapper(item_embeddings, torch.device("cpu"))
    agent = MultiPrefixTransformerAgent(
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
        utility=Utility(
            scales=(1.0, 1.0, 1.0, 1.0),
            weights=(1.0, 0.2, 0.2, -0.05),
        ),
        dominant_hits=torch.tensor([True, False, True, False]),
    )
    attach_rollout_dependencies(agent, max_turn=9)
    agent.invalid_action_penalty = 0.0
    agent.observation_spec = {
        "representation": "precomputed_avg_observation_v1",
        "state_dim": 2,
    }
    return agent


class FixedMultiCritic(nn.Module):
    """按输入顺序返回预设 R/T/D/C 前缀价值。"""

    def __init__(self, values: torch.Tensor) -> None:
        """保存形状为 ``(B,K,4)`` 的固定价值。

        Args:
            values (torch.Tensor): 预设四头前缀价值。
        """

        super().__init__()
        self.register_buffer("values", values)

    def forward(self, states: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        """忽略输入内容并返回固定价值。

        Args:
            states (torch.Tensor): 状态 batch，仅用于接口兼容。
            chunks (torch.Tensor): chunk batch，仅用于接口兼容。

        Returns:
            torch.Tensor: 初始化时保存的四头价值。
        """

        del states, chunks
        return self.values


class ZeroMultiCritic(nn.Module):
    """返回动态 batch 大小全零 R/T/D/C 价值的 target Q。"""

    def __init__(self, chunk_size: int) -> None:
        """保存输出前缀数。

        Args:
            chunk_size (int): 前缀数 K，必须为正。

        Raises:
            ValueError: 当 K 非正时抛出。
        """

        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        self.chunk_size = int(chunk_size)

    def forward(self, states: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
        """返回 ``(B,K,4)`` 全零价值。

        Args:
            states (torch.Tensor): 状态 batch。
            chunks (torch.Tensor): chunk batch，仅用于接口兼容。

        Returns:
            torch.Tensor: 与输入设备和 dtype 一致的零价值。
        """

        del chunks
        return states.new_zeros((states.shape[0], self.chunk_size, 4))


class MultiPrefixTransformerAgentTest(unittest.TestCase):
    """验证四头网络、共同决策、目标语义、更新和 checkpoint。"""

    def setUp(self) -> None:
        """固定随机性和 CPU 线程数。"""

        torch.manual_seed(47)
        torch.set_num_threads(1)

    def test_joint_selection_uses_scalarized_four_heads(self) -> None:
        """候选与 H 必须由四头综合分数的唯一 argmax 决定。"""

        agent = build_multi_agent()
        agent.utility = Utility(
            scales=(1.0, 1.0, 1.0, 1.0),
            weights=(1.0, 0.0, 1.0, 0.0),
        )
        candidate_ids = torch.tensor([[[0, 1, 2]], [[3, 2, 1]]])
        candidate_chunks = agent._item_ids_to_chunk_vectors(candidate_ids)

        def fixed_candidates(
            self: MultiPrefixTransformerAgent,
            states: torch.Tensor,
            num_samples: int,
            recommended_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """返回两条固定候选。

            Args:
                self (MultiPrefixTransformerAgent): 被绑定的测试 agent。
                states (torch.Tensor): 输入状态。
                num_samples (int): 候选数。
                recommended_mask (torch.Tensor | None): 历史屏蔽。

            Returns:
                tuple[torch.Tensor, torch.Tensor]: 固定 item 与 embedding 候选。
            """

            del self, states, num_samples, recommended_mask
            return candidate_ids, candidate_chunks

        agent.sample_candidate_chunks = types.MethodType(fixed_candidates, agent)
        values = torch.zeros(2, 3, 4)
        values[0, 0, 0] = 5.0
        values[1, 1, 2] = 8.0
        agent.critic = FixedMultiCritic(values)

        selected_ids, _, selected_lengths = agent.select_adaptive_chunks(
            torch.zeros(1, 2),
            num_samples=2,
        )

        self.assertEqual(selected_ids.tolist(), [[3, 2, 1]])
        self.assertEqual(selected_lengths.tolist(), [2])

    def test_target_bootstrap_uses_all_heads_from_one_action(self) -> None:
        """target bootstrap 不得为每个头分别选择不同动作。"""

        agent = build_multi_agent()
        agent.utility = Utility(
            scales=(1.0, 1.0, 1.0, 1.0),
            weights=(1.0, 0.0, 1.0, 0.0),
        )
        candidate_ids = torch.tensor([[[0, 1, 2]], [[3, 2, 1]]])
        candidate_chunks = agent._item_ids_to_chunk_vectors(candidate_ids)

        def fixed_candidates(
            self: MultiPrefixTransformerAgent,
            states: torch.Tensor,
            num_samples: int,
            recommended_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """返回用于 bootstrap 的两条固定候选。

            Args:
                self (MultiPrefixTransformerAgent): 被绑定的测试 agent。
                states (torch.Tensor): 输入状态。
                num_samples (int): 候选数。
                recommended_mask (torch.Tensor | None): 历史屏蔽。

            Returns:
                tuple[torch.Tensor, torch.Tensor]: 固定 item 与 embedding 候选。
            """

            del self, states, num_samples, recommended_mask
            return candidate_ids, candidate_chunks

        agent.sample_candidate_chunks = types.MethodType(fixed_candidates, agent)
        values = torch.zeros(2, 3, 4)
        values[0, 0] = torch.tensor([10.0, 1.0, 0.0, 4.0])
        values[1, 1] = torch.tensor([0.0, 2.0, 20.0, 3.0])
        agent.target_critic = FixedMultiCritic(values)
        history = RolloutHistoryState(
            states=torch.zeros(1, agent.state_dim),
            leave_history=torch.full((1, 9), -1, dtype=torch.long),
            recommended_mask=torch.zeros(1, agent.num_items, dtype=torch.bool),
            env_step=torch.zeros(1, dtype=torch.long),
            terminated=torch.zeros(1, dtype=torch.bool),
        )

        bootstrap = agent._target_state_values(
            history,
            num_samples=2,
            repeat_policy="truncate",
        )

        torch.testing.assert_close(
            bootstrap,
            torch.tensor([[0.0, 2.0, 20.0, 3.0]]),
        )

    def test_arbitrary_start_restores_full_session_mcd_counts(self) -> None:
        """离线中间起点应从完整前缀历史恢复 n/k，而非清零。"""

        agent = build_multi_agent()
        batch = {
            "leave_history_item_ids": torch.tensor([
                [-1, -1, -1, -1, -1, -1, -1, 0, 1],
                [-1, -1, -1, -1, -1, -1, 0, 2, 3],
            ]),
        }

        item_counts, dominant_counts = agent._initial_component_counts(batch)

        self.assertEqual(item_counts.tolist(), [2, 3])
        self.assertEqual(dominant_counts.tolist(), [1, 2])

    def test_smoke_runner_forwards_multi_critic_configuration(self) -> None:
        """smoke 编排必须把四头 critic 及 Transformer 结构传给子阶段。"""

        arguments = Namespace(
            env="KuaiEnv-v0",
            user_model_name="DeepFM",
            read_message="pointneg",
            dataset_path="dataset.pkl",
            which_tracker="none",
            reward_handle="cat",
            dynamics_pretrain_steps=1,
            dynamics_lr=3e-4,
            window_size=3,
            chunk_size=2,
            rollout_depth=2,
            lambda_chunk=1.0,
            all_prefix_aux_weight=1.0,
            critic_backbone="multi_prefix_transformer",
            critic_transformer_hidden_dim=16,
            critic_transformer_layers=1,
            critic_transformer_heads=4,
            critic_transformer_dropout=0.0,
            eta_t=0.2,
            eta_d=0.3,
            eta_c=0.1,
            top_rate=0.8,
            scale_samples=8,
            leave_policy="terminate",
            predicted_mat_normalize="none",
            gamma=0.9,
            seed=1,
            device="cpu",
            cuda="0",
            batch_size=2,
            max_trajectories=2,
            max_chunks=4,
            num_leave_compute=1,
            leave_threshold=0,
            max_turn=5,
            force_length=5,
            invalid_action_penalty=0.0,
            lambda_entropy=0.5,
            lambda_variance=1.0,
            dynamics_loss_weight=0.0,
            swanlab_project="test",
            run_name="test",
            execution_horizon=None,
            completion_window=2,
            entropy_window=[1, 2],
            use_entropy_reward=False,
            use_uncertainty_penalty=False,
            feature_level=True,
            is_sorted=True,
            enable_open_loop_diagnostics=False,
            dynamics_hidden_dims=[8],
            item_embedding_path="",
            predicted_mat_path="",
            maxvar_mat_path="",
        )

        common_argv = build_common_argv(arguments)

        critic_index = common_argv.index("--critic_backbone")
        self.assertEqual(common_argv[critic_index + 1], "multi_prefix_transformer")
        self.assertIn("--critic_transformer_hidden_dim", common_argv)

    def test_all_prefix_targets_have_distinct_rtdc_returns(self) -> None:
        """每个合法 H 应拥有自己的 T/D/C 累计值和同一策略 bootstrap。"""

        agent = build_multi_agent()
        agent.target_critic = ZeroMultiCritic(agent.chunk_size)
        batch = {
            "observations": torch.zeros(1, agent.state_dim),
            "leave_history_item_ids": torch.full((1, 9), -1, dtype=torch.long),
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

        expected = torch.tensor([
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.9, 0.45, 1.0],
                [0.0, 2.71, 0.315, 1.0],
            ]
        ])
        torch.testing.assert_close(target.values, expected)
        self.assertTrue(bool(target.valid.all()))
        self.assertEqual(target.effective_steps.tolist(), [[1, 2, 3]])

    def test_qv_update_and_checkpoint_round_trip(self) -> None:
        """四头联合更新应有限，并完整保存效用、命中表和 target Q。"""

        _, _, batch = build_fixture()
        agent = build_multi_agent()
        optimizer = torch.optim.Adam(agent.qv_parameters(), lr=0.001)

        metrics = agent.qv_update(
            batch=batch,
            optimizer=optimizer,
            num_samples_train=2,
            rollout_depth=2,
            lambda_chunk=0.7,
        )

        self.assertTrue(np.isfinite(list(metrics.values())).all())
        for component in ("R", "T", "D", "C"):
            self.assertIn(f"critic/q_mean/{component}", metrics)
            self.assertIn(f"critic/all_prefix_aux_loss/{component}", metrics)
        checkpoint = agent.checkpoint_state({
            "critic_backbone": "multi_prefix_transformer",
        })
        self.assertEqual(checkpoint["format"], MULTI_PREFIX_CHECKPOINT_FORMAT)
        restored = build_multi_agent()
        restored.load_checkpoint_state(checkpoint)
        self.assertEqual(restored.utility, agent.utility)
        torch.testing.assert_close(restored.dominant_hits, agent.dominant_hits)
        for source, target in zip(
            agent.critic.parameters(),
            restored.critic.parameters(),
        ):
            torch.testing.assert_close(source, target)


if __name__ == "__main__":
    unittest.main()
