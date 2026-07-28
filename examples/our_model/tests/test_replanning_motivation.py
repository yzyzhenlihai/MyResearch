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
        self.reward_by_action = {0: 0.0, 1: 1.0, 2: 2.0}

    def step(self, action: int) -> tuple[None, float, bool, bool, dict[str, float]]:
        """执行动作并更新与 BaseEnv 一致的可变字段。

        Args:
            action (int): 离散动作编号。

        Returns:
            tuple[None, float, bool, bool, dict[str, float]]: Gymnasium 风格结果。
        """

        reward = self.reward_by_action[int(action)]
        self.action = int(action)
        self.history_action[self.total_turn] = int(action)
        self.sequence_action.append(int(action))
        self.max_history += 1
        self.total_turn += 1
        self.cum_reward += reward
        return None, reward, False, False, {"cum_reward": self.cum_reward}


class ReplanningBranchTest(unittest.TestCase):
    """验证同状态分叉、延迟语义和环境恢复。"""

    def test_replanning_gain_and_delay_loss_use_paired_state(self) -> None:
        """验证立即重规划优于缓存动作时指标数值正确。"""

        env = ToyBranchEnvironment()
        original_snapshot = capture_environment(env)

        result = evaluate_branch_point(
            env=env,
            history_items=[-1, 9],
            history_rewards=[0.0, 1.0],
            cached_suffix=[0, 0],
            planner=lambda items, rewards: [2, 2],
        )

        self.assertEqual(result.planned_steps, 2)
        self.assertAlmostEqual(result.immediate_reward, 4.0)
        self.assertAlmostEqual(result.continue_reward, 0.0)
        self.assertAlmostEqual(result.replanning_gain, 2.0)
        delay_rewards = {
            evaluation.delay_steps: evaluation.reward_sum
            for evaluation in result.evaluations
        }
        self.assertEqual(delay_rewards, {0: 4.0, 1: 2.0, 2: 0.0})
        self.assertEqual(capture_environment(env), original_snapshot)


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
    """验证绘图汇总不重复计算同一状态的 replanning gain。"""

    def test_summary_uses_full_continue_row_for_gain(self) -> None:
        """验证左图只取 delay 等于剩余步数的记录。"""

        base_record = {
            "chunk_size": 3,
            "episode_id": 0,
            "user_id": 0,
            "chunk_index": 0,
            "chunk_position": 1,
            "remaining_steps": 2,
            "immediate_reward": 4.0,
            "delayed_reward": 4.0,
            "immediate_executed_steps": 2,
            "delayed_executed_steps": 2,
            "immediate_terminated": 0,
            "delayed_terminated": 0,
            "replanning_gain": 2.0,
            "delay_loss": 0.0,
        }
        records = []
        for delay_steps, delayed_reward in enumerate([4.0, 2.0, 0.0]):
            record = dict(base_record)
            record["delay_steps"] = delay_steps
            record["delayed_reward"] = delayed_reward
            record["delay_loss"] = (4.0 - delayed_reward) / 2.0
            records.append(record)

        summaries = build_plot_summaries(
            records=records,
            bootstrap_samples=20,
            seed=0,
        )

        self.assertEqual(len(summaries["replanning_gain"]), 1)
        gain_summary = summaries["replanning_gain"][0]
        self.assertEqual(gain_summary["state_count"], 1)
        self.assertAlmostEqual(float(gain_summary["mean"]), 2.0)
        self.assertEqual(len(summaries["delay_loss"]), 3)


if __name__ == "__main__":
    unittest.main()
