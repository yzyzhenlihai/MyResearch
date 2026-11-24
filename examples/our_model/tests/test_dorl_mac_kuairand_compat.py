"""DORL-MAC KuaiRec/KuaiRand 双数据集兼容测试。"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

import src.core.envs.KuaiRand_Pure.KuaiRandData as kuairand_data_module
from examples.our_model.models.leave_model import RuleBasedLeaveModel
from examples.our_model.runners.common import (
    SUPPORTED_KUAI_DATASET_PATHS,
    build_reward_and_leave,
    resolve_common_paths,
)
from examples.our_model.runners.pretrain_categorical_bc import (
    build_pretraining_datasets,
)
from src.core.envs.KuaiRand_Pure.KuaiRandData import KuaiRandData


class DORLMACKuaiRandCompatTest(unittest.TestCase):
    """验证环境级路径分发以及原始 ID 映射。"""

    def test_resolve_common_paths_selects_dataset_from_env(self) -> None:
        """未显式传路径时，应根据环境选择对应的轨迹和模型资产。"""

        for env_name, dataset_path in SUPPORTED_KUAI_DATASET_PATHS.items():
            with self.subTest(env=env_name):
                args = argparse.Namespace(
                    env=env_name,
                    user_model_name="DeepFM",
                    read_message="pointneg",
                    dataset_path="",
                    dynamics_dataset_path="",
                    item_embedding_path="",
                    dynamics_item_embedding_path="",
                    predicted_mat_path="",
                    maxvar_mat_path="",
                )
                resolve_common_paths(args)
                self.assertEqual(args.dataset_path, dataset_path)
                self.assertEqual(args.dynamics_dataset_path, dataset_path)
                self.assertIn(f"saved_models/{env_name}/DeepFM", args.item_embedding_path)
                self.assertEqual(
                    args.dynamics_item_embedding_path,
                    args.item_embedding_path,
                )
                self.assertTrue(args.predicted_mat_path.endswith("_matPre.pickle"))
                self.assertTrue(args.maxvar_mat_path.endswith("_matVar.pickle"))

    def test_pretraining_builds_independent_dynamics_dataset(self) -> None:
        """独立 Dynamics 路径必须使用第二个 loader 且保留 Actor mapper。"""

        actor_dataset = SimpleNamespace(state_dim=42, action_dim=41)
        dynamics_dataset = SimpleNamespace(state_dim=42, action_dim=41)
        actor_mapper = object()
        args = argparse.Namespace(
            dataset_path="small.pkl",
            item_embedding_path="small.pt",
            dynamics_dataset_path="big.pkl",
            dynamics_item_embedding_path="big.pt",
        )
        with mock.patch(
            "examples.our_model.runners.pretrain_categorical_bc."
            "build_dataset_and_mapper",
            side_effect=[
                (actor_dataset, actor_mapper, object()),
                (dynamics_dataset, object(), object()),
            ],
        ) as build_dataset:
            actual_actor, actual_dynamics, actual_mapper = (
                build_pretraining_datasets(args, torch.device("cpu"))
            )

        self.assertIs(actual_actor, actor_dataset)
        self.assertIs(actual_dynamics, dynamics_dataset)
        self.assertIs(actual_mapper, actor_mapper)
        self.assertEqual(build_dataset.call_count, 2)
        dynamics_args = build_dataset.call_args_list[1].args[0]
        self.assertEqual(dynamics_args.dataset_path, "big.pkl")
        self.assertEqual(dynamics_args.item_embedding_path, "big.pt")

    def test_pretraining_reuses_policy_dataset_for_legacy_mode(self) -> None:
        """两组路径相同时应只加载一次，保持旧模式内存行为。"""

        actor_dataset = SimpleNamespace(state_dim=18, action_dim=17)
        actor_mapper = object()
        args = argparse.Namespace(
            dataset_path="test.pkl",
            item_embedding_path="item.pt",
            dynamics_dataset_path="test.pkl",
            dynamics_item_embedding_path="item.pt",
        )
        with mock.patch(
            "examples.our_model.runners.pretrain_categorical_bc."
            "build_dataset_and_mapper",
            return_value=(actor_dataset, actor_mapper, object()),
        ) as build_dataset:
            _, dynamics_dataset, _ = build_pretraining_datasets(
                args,
                torch.device("cpu"),
            )

        self.assertIs(dynamics_dataset, actor_dataset)
        build_dataset.assert_called_once()

    def test_kuairand_reward_and_leave_use_identity_ids(self) -> None:
        """无 LabelEncoder 的 KuaiRand 应直接使用连续原始 user/item ID。"""

        env = SimpleNamespace(
            mat=np.zeros((2, 3), dtype=np.float32),
            list_feat=[[1], [2], [1, 2]],
        )
        args = SimpleNamespace(
            predicted_mat_path="prediction.pickle",
            maxvar_mat_path="variance.pickle",
            use_exposure_intervention=False,
            use_entropy_reward=False,
            entropy_window=[1, 2],
            lambda_entropy=0.5,
            feature_level=True,
            is_sorted=True,
            use_uncertainty_penalty=False,
            lambda_variance=1.0,
            predicted_mat_normalize="none",
            num_leave_compute=10,
            leave_threshold=0,
            max_turn=30,
        )
        with (
            mock.patch(
                "examples.our_model.runners.common.build_entropy_reward_assets",
                return_value=({}, {}, 0.0),
            ),
            mock.patch(
                "examples.our_model.runners.common.DORLRewardModel"
            ) as reward_model_cls,
        ):
            reward_model, leave_model = build_reward_and_leave(
                args=args,
                env=env,
                dataset=object(),
                device=torch.device("cpu"),
            )

        self.assertIs(reward_model, reward_model_cls.return_value)
        reward_kwargs = reward_model_cls.call_args.kwargs
        self.assertEqual(reward_kwargs["raw_user_to_index"], {0: 0, 1: 1})
        self.assertEqual(reward_kwargs["internal_to_raw_item_ids"], [0, 1, 2])
        self.assertEqual(leave_model.list_feat_small, env.list_feat)

    def test_negative_missing_category_does_not_collide(self) -> None:
        """KuaiRand 的 -1 缺失标签不能与最大正类别共享向量列。"""

        leave_model = RuleBasedLeaveModel(
            list_feat_small=[[-1], [5], [-1]],
            num_leave_compute=2,
            leave_threshold=0,
            max_turn=30,
        )
        violation_steps = leave_model.first_violation_steps(
            leave_history_item_ids=torch.tensor([[1]], dtype=torch.long),
            chunk_item_ids=torch.tensor([[2]], dtype=torch.long),
        )
        self.assertEqual(int(violation_steps.item()), -1)

    def test_load_video_duration_uses_pandas_two_compatible_concat(self) -> None:
        """首次统计视频时长时应兼容 pandas 2.x 并正确计算 item 均值。"""

        small_duration = pd.DataFrame(
            {
                "item_id": [0, 0, 1],
                "duration_normed": [1.0, 3.0, 2.0],
            }
        )
        big_duration = pd.DataFrame(
            {
                "item_id": [0, 1],
                "duration_normed": [5.0, 4.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch.object(kuairand_data_module, "PRODATAPATH", temp_dir),
                mock.patch.object(
                    kuairand_data_module,
                    "get_df_data",
                    side_effect=[small_duration, big_duration],
                ),
            ):
                video_duration = KuaiRandData.load_video_duration()

            cache_path = Path(temp_dir) / "video_duration_normed.csv"
            self.assertTrue(cache_path.is_file())

        np.testing.assert_allclose(
            video_duration.to_numpy(),
            np.asarray([3.0, 3.0]),
        )
        self.assertEqual(video_duration.index.name, "item_id")

    def test_load_category_avoids_chained_assignment(self) -> None:
        """类别解析应支持缺失值且不依赖 pandas 链式赋值。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            feature_path = Path(temp_dir) / "video_features_basic_pure.csv"
            pd.DataFrame({"tag": [None, "1,2", "3"]}).to_csv(
                feature_path,
                index=False,
            )
            with mock.patch.object(kuairand_data_module, "DATAPATH", temp_dir):
                categories, item_features = KuaiRandData.load_category()

        self.assertEqual(categories, [[-1], [1, 2], [3]])
        self.assertEqual(item_features["tags"].tolist(), categories)
        self.assertEqual(tuple(item_features.shape), (3, 4))


if __name__ == "__main__":
    unittest.main()
