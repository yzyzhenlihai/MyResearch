"""KuaiRand 用户级离线强化学习轨迹构造测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from examples.our_model.data.kuairand_trajectory_builder import (
    build_kuairand_trajectories,
    compute_avg_states,
    validate_trajectories,
)


class KuaiRandTrajectoryBuilderTest(unittest.TestCase):
    """验证 KuaiRand 用户轨迹构造的数值和结构语义。"""

    def test_compute_avg_states_matches_state_tracker_avg_semantics(self) -> None:
        """验证首步 padding、滑动窗口和 next state 的平均语义。"""

        actions = np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        )
        rewards = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        padding_state = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)

        observations, next_observations = compute_avg_states(
            actions=actions,
            rewards=rewards,
            padding_state=padding_state,
            window_size=2,
        )

        expected_observations = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [2.0, 3.0, 0.5],
            ],
            dtype=np.float32,
        )
        expected_next_observations = np.asarray(
            [
                [0.5, 1.0, 0.0],
                [2.0, 3.0, 0.5],
                [4.0, 5.0, 0.5],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(observations, expected_observations)
        np.testing.assert_allclose(
            next_observations,
            expected_next_observations,
        )

    def test_build_trajectories_groups_users_and_marks_terminal(self) -> None:
        """验证每位用户仅生成一条轨迹且最后一步为 terminal。"""

        interactions = pd.DataFrame(
            {
                "user_id": [0, 0, 1],
                "item_id": [1, 0, 2],
                "time_ms": [10, 20, 15],
                "is_click": [1.0, 0.0, 1.0],
            }
        )
        item_embeddings = np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 2.0],
            ],
            dtype=np.float32,
        )

        trajectories = build_kuairand_trajectories(
            interactions=interactions,
            item_embeddings=item_embeddings,
            window_size=3,
            seed=2023,
        )
        statistics = validate_trajectories(
            trajectories,
            expected_action_dim=2,
        )

        self.assertEqual(
            [trajectory["user_id"] for trajectory in trajectories],
            [0, 1],
        )
        self.assertEqual(
            trajectories[0]["terminals"].tolist(),
            [False, True],
        )
        self.assertEqual(trajectories[1]["terminals"].tolist(), [True])
        np.testing.assert_array_equal(
            trajectories[0]["actions"],
            item_embeddings[[1, 0]],
        )
        self.assertEqual(statistics["num_trajectories"], 2)
        self.assertEqual(statistics["num_transitions"], 3)
        self.assertEqual(statistics["action_dim"], 2)
        self.assertEqual(statistics["state_dim"], 3)


if __name__ == "__main__":
    unittest.main()
