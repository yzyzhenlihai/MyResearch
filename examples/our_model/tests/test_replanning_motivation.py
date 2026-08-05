"""原 chunk 失效动机实验核心逻辑测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from analysis.plot_replanning_motivation import (
    build_plot_summaries,
)
from examples.our_model.evaluation.replanning_motivation import (
    capture_environment,
    evaluate_branch_point,
)
from examples.our_model.runners.eval_replanning_motivation import (
    build_recommended_mask,
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

    def test_checkpoint_specs_reject_duplicate_or_small_k(self) -> None:
        """验证 K 必须大于一且不能重复。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "latest.pt"
            checkpoint_path.touch()
            with self.assertRaisesRegex(ValueError, "K > 1"):
                parse_checkpoint_specs([f"1={checkpoint_path}"])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                parse_checkpoint_specs(
                    [
                        f"3={checkpoint_path}",
                        f"3={checkpoint_path}",
                    ]
                )

    def test_recommended_mask_ignores_reset_dummy_and_deduplicates(self) -> None:
        """验证 reset dummy 不进入 mask，重复历史只屏蔽一次。"""

        mask = build_recommended_mask(
            history_items=[-1, 1, 1, 3],
            num_items=5,
            device=torch.device("cpu"),
        )

        self.assertEqual(mask.shape, (1, 5))
        self.assertEqual(mask[0].tolist(), [False, True, False, True, False])


class MotivationSummaryTest(unittest.TestCase):
    """验证绘图汇总不重复计算同一状态的长期重规划优势。"""

    def test_summary_uses_full_continue_row_for_gain(self) -> None:
        """验证左图只取 delay 等于剩余步数的记录。"""

        base_record = {
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


if __name__ == "__main__":
    unittest.main()
