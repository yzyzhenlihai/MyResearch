"""KuaiRand 用户级离线强化学习轨迹构造测试。"""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from examples.diffusion.build_kuairec_trajectory_dataset import (
    resolve_dataset_config,
)
from examples.diffusion.build_kuairand_trajectory_dataset import (
    resolve_dataset_paths,
)
from examples.our_model.data.kuairand_trajectory_builder import (
    build_and_save_trajectory_dataset,
    build_kuairand_trajectories,
    compute_avg_states,
    load_interactions,
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

    def test_generic_loader_supports_kuairec_columns_and_stable_sort(self) -> None:
        """验证通用加载器支持 KuaiRec 字段并保持同时间记录原始顺序。"""

        frame = pd.DataFrame(
            {
                "user_id": [1, 0, 0, 0],
                "item_id": [3, 2, 1, 0],
                "timestamp": [5.0, 2.0, 1.0, 1.0],
                "watch_ratio_normed": [0.4, 0.3, 0.2, 0.1],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "big_matrix_processed.csv"
            frame.to_csv(input_path, index=False)
            interactions = load_interactions(
                input_path=input_path,
                time_column="timestamp",
                reward_column="watch_ratio_normed",
                dataset_name="KuaiRec",
            )

        self.assertEqual(interactions["user_id"].tolist(), [0, 0, 0, 1])
        self.assertEqual(interactions["item_id"].tolist(), [1, 0, 2, 3])
        self.assertEqual(interactions["watch_ratio_normed"].dtype, np.float32)

    def test_build_and_save_pipeline_writes_valid_pickle_and_summary(self) -> None:
        """验证通用构建流程写出可校验轨迹、哈希摘要并保护已有文件。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "train_processed.csv"
            embedding_path = root / "embedding.pt"
            output_path = root / "trajectory.pkl"
            pd.DataFrame(
                {
                    "user_id": [1, 0, 0],
                    "item_id": [2, 1, 0],
                    "time_ms": [3, 2, 1],
                    "is_click": [1, 0, 1],
                }
            ).to_csv(input_path, index=False)
            torch.save(
                torch.tensor(
                    [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]],
                    dtype=torch.float32,
                ),
                embedding_path,
            )

            summary = build_and_save_trajectory_dataset(
                dataset_name="KuaiRand",
                split="train",
                input_path=input_path,
                embedding_path=embedding_path,
                output_path=output_path,
                summary_path=None,
                time_column="time_ms",
                reward_column="is_click",
                window_size=3,
                seed=2023,
                padding_std=0.01,
            )
            with output_path.open("rb") as file_obj:
                trajectories = pickle.load(file_obj)
            with output_path.with_suffix(".pkl.summary.json").open(
                "r", encoding="utf-8",
            ) as file_obj:
                saved_summary = json.load(file_obj)

            self.assertEqual(summary["num_trajectories"], 2)
            self.assertEqual(saved_summary["num_transitions"], 3)
            self.assertEqual(trajectories[0]["user_id"], 0)
            self.assertEqual(trajectories[0]["terminals"].tolist(), [False, True])
            with self.assertRaises(FileExistsError):
                build_and_save_trajectory_dataset(
                    dataset_name="KuaiRand",
                    split="train",
                    input_path=input_path,
                    embedding_path=embedding_path,
                    output_path=output_path,
                    summary_path=None,
                    time_column="time_ms",
                    reward_column="is_click",
                    window_size=3,
                    seed=2023,
                    padding_std=0.01,
                )

    def test_split_configs_select_matching_item_spaces(self) -> None:
        """验证 KuaiRec big 与 KuaiRand train 默认选择正确的 embedding。"""

        kuairec_train = resolve_dataset_config("train")
        kuairand_train = resolve_dataset_paths("train")

        self.assertEqual(kuairec_train.input_path.name, "big_matrix_processed.csv")
        self.assertEqual(
            kuairec_train.embedding_path.name,
            "[pointneg]_emb_item_M0.pt",
        )
        self.assertEqual(kuairand_train.input_path.name, "train_processed.csv")
        self.assertEqual(
            kuairand_train.embedding_path.name,
            "[pointneg]_emb_item_val_M0.pt",
        )


if __name__ == "__main__":
    unittest.main()
