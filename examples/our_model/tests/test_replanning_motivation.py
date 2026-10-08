"""原 chunk 失效动机实验核心逻辑测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from analysis.plot_replanning_motivation import (
    build_plot_summaries,
)
from examples.our_model.evaluation.replanning_motivation import (
    capture_environment,
    compute_session_mcd_counts,
    evaluate_branch_point,
)
from examples.our_model.runners.eval_replanning_motivation import (
    build_recommended_mask,
    build_state_from_history,
    parse_checkpoint_specs,
)


class ToyBranchEnvironment:
    """提供确定性奖励和 BaseEnv 状态字段的轻量测试环境。"""

    def __init__(self) -> None:
        """初始化一个已经执行一步的中间状态。"""

        self.cur_user = 7
        self.action = 9
        self.cum_reward = 1.0
        self.total_turn = 1
        self.history_action = {0: 9}
        self.sequence_action = [9]
        self.max_history = 1
        self.max_turn = 4
        self.reward_by_action = {0: 0.0, 1: 1.0, 2: 2.0}

    def step(self, action: int) -> tuple[None, float, bool, bool, dict[str, float]]:
        """执行动作并更新与 BaseEnv 一致的可变字段。

        Args:
            action (int): 离散动作编号。

        Returns:
            tuple[None, float, bool, bool, dict[str, float]]: Gymnasium 风格结果。
        """

        reward = self.reward_by_action[int(action)]
        terminated = self.total_turn >= self.max_turn - 1
        self.action = int(action)
        self.history_action[self.total_turn] = int(action)
        self.sequence_action.append(int(action))
        self.max_history += 1
        self.total_turn += 1
        self.cum_reward += reward
        return None, reward, terminated, False, {"cum_reward": self.cum_reward}


class ReplanningBranchTest(unittest.TestCase):
    """验证同状态分叉、延迟语义和环境恢复。"""

    def test_long_term_return_uses_complete_future_trajectory(self) -> None:
        """验证主指标包含局部窗口之后直到 episode 结束的回报。"""

        env = ToyBranchEnvironment()
        original_snapshot = capture_environment(env)

        result = evaluate_branch_point(
            env=env,
            history_items=[-1, 9],
            history_rewards=[0.0, 1.0],
            cached_suffix=[0, 0],
            planner=lambda items, rewards: [2, 2],
            downstream_planner=lambda items, rewards, index: [2, 2],
            dominant_item_mask=[True] + [False] * 9,
        )

        self.assertEqual(result.planned_steps, 2)
        self.assertAlmostEqual(result.long_term_replanning_advantage, 4.0)
        self.assertAlmostEqual(result.short_term_replanning_gain, 2.0)
        local_rewards = {
            evaluation.delay_steps: evaluation.local_reward_sum
            for evaluation in result.evaluations
        }
        future_returns = {
            evaluation.delay_steps: evaluation.future_reward_sum
            for evaluation in result.evaluations
        }
        self.assertEqual(local_rewards, {0: 4.0, 1: 2.0, 2: 0.0})
        self.assertEqual(future_returns, {0: 6.0, 1: 4.0, 2: 2.0})
        self.assertTrue(
            all(evaluation.terminated for evaluation in result.evaluations)
        )
        self.assertAlmostEqual(result.immediate_evaluation.final_mcd, 0.0)
        self.assertAlmostEqual(result.continue_evaluation.final_mcd, 0.5)
        self.assertAlmostEqual(result.diversity_replanning_advantage, 0.5)
        self.assertEqual(capture_environment(env), original_snapshot)

    def test_delay_subset_keeps_immediate_and_continue_endpoints(self) -> None:
        """验证非首位置可只评估两个端点并拒绝缺端点的配置。"""

        env = ToyBranchEnvironment()
        kwargs = {
            "env": env,
            "history_items": [-1, 9],
            "history_rewards": [0.0, 1.0],
            "cached_suffix": [0, 0],
            "planner": lambda items, rewards: [2, 2],
            "downstream_planner": (
                lambda items, rewards, index: [2, 2]
            ),
            "dominant_item_mask": [True] + [False] * 9,
        }

        result = evaluate_branch_point(**kwargs, delay_values=[0, 2])

        self.assertEqual(
            [evaluation.delay_steps for evaluation in result.evaluations],
            [0, 2],
        )
        self.assertAlmostEqual(result.long_term_replanning_advantage, 4.0)
        with self.assertRaisesRegex(ValueError, "must include both"):
            evaluate_branch_point(**kwargs, delay_values=[0, 1])


class MotivationInputTest(unittest.TestCase):
    """验证 checkpoint 规格与推荐历史 mask 边界。"""

    def test_checkpoint_specs_allow_same_k_but_reject_duplicate_path(self) -> None:
        """验证多个 seed 可共享 K，但同一路径不能重复输入。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "first.pt"
            second_path = Path(temp_dir) / "second.pt"
            checkpoint_path.touch()
            second_path.touch()
            with self.assertRaisesRegex(ValueError, "K > 1"):
                parse_checkpoint_specs([f"1={checkpoint_path}"])
            parsed = parse_checkpoint_specs(
                [f"5={checkpoint_path}", f"5={second_path}"]
            )
            self.assertEqual([chunk_size for chunk_size, _ in parsed], [5, 5])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                parse_checkpoint_specs(
                    [
                        f"5={checkpoint_path}",
                        f"5={checkpoint_path}",
                    ]
                )

    def test_session_mcd_counts_include_public_prefix(self) -> None:
        """验证最终 MCD 统计包含公共前缀且忽略 reset dummy。"""

        dominant_count, exposure_count = compute_session_mcd_counts(
            history_items=[-1, 0, 2, 1],
            dominant_item_mask=[True, False, True],
        )

        self.assertEqual((dominant_count, exposure_count), (2, 3))

    def test_recommended_mask_ignores_reset_dummy_and_deduplicates(self) -> None:
        """验证 reset dummy 不进入 mask，重复历史只屏蔽一次。"""

        mask = build_recommended_mask(
            history_items=[-1, 1, 1, 3],
            num_items=5,
            device=torch.device("cpu"),
        )

        self.assertEqual(mask.shape, (1, 5))
        self.assertEqual(mask[0].tolist(), [False, True, False, True, False])

    def test_explicit_history_uses_chunk_dynamics_from_initial_state(self) -> None:
        """动机实验应从离线初始状态按历史动作递推 dynamics。"""

        class SumDynamics:
            """把有效动作 embedding 之和加到状态各维。"""

            def __call__(
                self,
                states: torch.Tensor,
                actions: torch.Tensor,
                valid: torch.Tensor,
            ) -> torch.Tensor:
                """返回动作前缀递推后的测试状态。

                Args:
                    states (torch.Tensor): 当前状态。
                    actions (torch.Tensor): 动作 embedding。
                    valid (torch.Tensor): 有效前缀标记。

                Returns:
                    torch.Tensor: 加上有效动作 embedding 总和后的状态。
                """

                return states + (actions * valid.unsqueeze(-1)).sum((1, 2), keepdim=False).unsqueeze(1)

        agent = SimpleNamespace(
            chunk_size=2,
            action_dim=1,
            action_mapper=SimpleNamespace(
                item_embeddings=torch.tensor([[10.0], [20.0], [30.0]]),
            ),
            dynamics=SumDynamics(),
        )
        state = build_state_from_history(
            agent=agent,
            initial_states=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            user_id=1,
            history_items=[-1, 0, 2],
            history_rewards=[0.0, 0.5, 1.5],
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(
            state,
            torch.tensor([[43.0, 44.0]]),
        )


class MotivationSummaryTest(unittest.TestCase):
    """验证绘图汇总不重复计算同一状态的长期重规划优势。"""

    def test_summary_uses_full_continue_row_for_gain(self) -> None:
        """验证左图只取 delay 等于剩余步数的记录。"""

        base_record = {
            "model_seed": 2023,
            "chunk_size": 3,
            "episode_id": 0,
            "user_id": 0,
            "chunk_index": 0,
            "chunk_position": 1,
            "remaining_steps": 2,
            "immediate_local_reward": 4.0,
            "delayed_local_reward": 4.0,
            "immediate_future_return": 6.0,
            "delayed_future_return": 6.0,
            "immediate_local_executed_steps": 2,
            "delayed_local_executed_steps": 2,
            "immediate_future_executed_steps": 3,
            "delayed_future_executed_steps": 3,
            "immediate_local_terminated": 0,
            "delayed_local_terminated": 0,
            "immediate_terminated": 0,
            "delayed_terminated": 0,
            "long_term_replanning_advantage": 4.0,
            "long_term_delay_cost": 0.0,
            "short_term_replanning_gain": 2.0,
            "short_term_delay_loss": 0.0,
            "immediate_final_mcd": 0.25,
            "delayed_final_mcd": 0.25,
            "continue_final_mcd": 0.50,
            "diversity_replanning_advantage": 0.25,
            "diversity_delay_cost": 0.0,
            "return_noninferior_diversity_improved": 1,
            "weighted_replanning_advantage": 29.0,
            "immediate_dominant_count": 1,
            "immediate_exposure_count": 4,
            "delayed_dominant_count": 1,
            "delayed_exposure_count": 4,
            "continue_dominant_count": 2,
            "continue_exposure_count": 4,
        }
        records = []
        for delay_steps, (local_reward, future_return) in enumerate(
            [(4.0, 6.0), (2.0, 4.0), (0.0, 2.0)]
        ):
            record = dict(base_record)
            record["delay_steps"] = delay_steps
            record["delayed_local_reward"] = local_reward
            record["delayed_future_return"] = future_return
            record["long_term_delay_cost"] = 6.0 - future_return
            record["short_term_delay_loss"] = (
                4.0 - local_reward
            ) / 2.0
            record["delayed_final_mcd"] = 0.25 + 0.125 * delay_steps
            record["diversity_delay_cost"] = 0.125 * delay_steps
            records.append(record)

        summaries = build_plot_summaries(
            records=records,
            bootstrap_samples=20,
            seed=0,
        )

        self.assertEqual(
            len(summaries["long_term_replanning_advantage"]),
            1,
        )
        gain_summary = summaries["long_term_replanning_advantage"][0]
        self.assertEqual(gain_summary["state_count"], 1)
        self.assertAlmostEqual(float(gain_summary["mean"]), 4.0)
        self.assertEqual(len(summaries["long_term_delay_cost"]), 3)
        self.assertEqual(len(summaries["diversity_delay_cost"]), 3)
        self.assertTrue(summaries["delay_optima"][0]["shared_optimum_exists"])


if __name__ == "__main__":
    unittest.main()
