"""DORL-MAC 单进程 execution-horizon sweep 参数与资源测试。"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import examples.our_model.runners.eval_dorl_mac as eval_module
from examples.our_model.runners.eval_dorl_mac import (
    close_evaluator_environments,
    resolve_eval_episodes,
    resolve_execution_horizons,
)
from examples.our_model.runners.evaluation_utils import (
    resolve_evaluation_collection_config,
)


TEST_CHUNK_SIZE = 5
"""测试使用的规划长度 K。"""


def build_horizon_args(
    execution_horizon: int | None = None,
    execution_horizons: list[int] | None = None,
) -> argparse.Namespace:
    """构造 execution horizon 解析所需最小参数。

    Args:
        execution_horizon (int | None): 单值 H。
        execution_horizons (list[int] | None): 多值 H。

    Returns:
        argparse.Namespace: 包含 K、单值 H 和多值 H 的参数对象。
    """

    return argparse.Namespace(
        chunk_size=TEST_CHUNK_SIZE,
        execution_horizon=execution_horizon,
        execution_horizons=execution_horizons,
    )


class FakeVectorEnvironment:
    """记录 `close()` 是否被调用的向量环境测试替身。"""

    def __init__(self) -> None:
        """初始化未关闭状态。"""

        self.closed = False

    def close(self) -> None:
        """记录环境已关闭。

        Returns:
            None.
        """

        self.closed = True


class HorizonResolutionTest(unittest.TestCase):
    """验证单值和多值 H 的解析、校验与默认行为。"""

    def test_default_horizon_uses_chunk_size(self) -> None:
        """验证未设置 H 时保持 `H=K` 的原单次评估语义。"""

        args = build_horizon_args()
        self.assertEqual(resolve_execution_horizons(args), [TEST_CHUNK_SIZE])

    def test_multiple_horizons_preserve_input_order(self) -> None:
        """验证多 H sweep 保留用户给定顺序。"""

        args = build_horizon_args(execution_horizons=[1, 3, 2, 5])
        self.assertEqual(resolve_execution_horizons(args), [1, 3, 2, 5])

    def test_single_and_multiple_horizons_conflict(self) -> None:
        """验证单值和多值 H 不能同时设置。"""

        args = build_horizon_args(
            execution_horizon=2,
            execution_horizons=[1, 2],
        )
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            resolve_execution_horizons(args)

    def test_duplicate_or_out_of_range_horizons_are_rejected(self) -> None:
        """验证重复 H 和越界 H 都会被拒绝。"""

        duplicate_args = build_horizon_args(execution_horizons=[1, 3, 3])
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            resolve_execution_horizons(duplicate_args)

        invalid_args = build_horizon_args(execution_horizons=[0, 6])
        with self.assertRaisesRegex(ValueError, "1 <= H <= chunk_size"):
            resolve_execution_horizons(invalid_args)

    def test_eval_episodes_are_independent_from_parallel_environment_count(self) -> None:
        """验证轨迹总数由 eval_episodes 控制，而非 test_num。"""

        args = argparse.Namespace(eval_episodes=9, test_num=2)

        self.assertEqual(resolve_eval_episodes(args), 9)

    def test_zero_eval_episodes_preserves_legacy_test_num_fallback(self) -> None:
        """验证旧调用中 0 仍回退为 test_num。"""

        args = argparse.Namespace(eval_episodes=0, test_num=3)

        self.assertEqual(resolve_eval_episodes(args), 3)

    def test_close_evaluator_environments_closes_every_branch(self) -> None:
        """验证进入下一个 H 前关闭 FB/NX 的全部向量环境。"""

        environments = [FakeVectorEnvironment() for _ in range(3)]
        collectors = {
            f"branch_{index}": argparse.Namespace(env=environment)
            for index, environment in enumerate(environments)
        }
        evaluator = argparse.Namespace(
            collector_set=argparse.Namespace(collector_dict=collectors)
        )

        close_evaluator_environments(evaluator)

        self.assertTrue(all(environment.closed for environment in environments))

    def test_multi_horizon_main_loads_shared_assets_once(self) -> None:
        """验证多 H 主流程只加载一次数据、模型和 checkpoint。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            checkpoint_path = temp_path / "latest.pt"
            checkpoint_path.touch()
            output_path = temp_path / "outputs"
            fake_agent = mock.Mock()
            fake_env = argparse.Namespace()

            with (
                mock.patch.object(eval_module, "configure_logging"),
                mock.patch.object(eval_module, "set_mpl_cache_to_tmp"),
                mock.patch.object(eval_module, "resolve_common_paths"),
                mock.patch.object(
                    eval_module,
                    "resolve_device",
                    return_value=torch.device("cpu"),
                ),
                mock.patch.object(
                    eval_module,
                    "build_env_assets",
                    return_value=(fake_env, object(), {"shared": True}),
                ) as build_env_assets,
                mock.patch.object(
                    eval_module,
                    "build_dataset_and_mapper",
                    return_value=(argparse.Namespace(state_dim=2), object(), torch.zeros(4, 3)),
                ) as build_dataset_and_mapper,
                mock.patch.object(
                    eval_module,
                    "build_reward_and_leave",
                    return_value=(object(), object()),
                ) as build_reward_and_leave,
                mock.patch.object(
                    eval_module,
                    "build_agent",
                    return_value=fake_agent,
                ) as build_agent,
                mock.patch.object(
                    eval_module.torch,
                    "load",
                    return_value={"format": "mac_origin_precomputed_state_v2", "config": {
                        "chunk_size": 3, "window_size": 3, "gamma": 0.9,
                        "actor_hidden_dims": [8], "value_hidden_dims": [8],
                        "dynamics_hidden_dims": [8],
                        "state_representation": "precomputed_avg_observation_v1",
                        "state_dim": 2,
                    }},
                ) as load_checkpoint,
                mock.patch.object(
                    eval_module,
                    "build_initial_state_table",
                    return_value=torch.zeros((2, 2)),
                ) as build_initial_state_table,
                mock.patch.object(
                    eval_module,
                    "evaluate_one_horizon",
                    side_effect=lambda **kwargs: (
                        kwargs["save_dir"] / "summary_metrics.json"
                    ),
                ) as evaluate_one_horizon,
            ):
                result_path = eval_module.main(
                    [
                        "--mac_ckpt",
                        str(checkpoint_path),
                        "--chunk_size",
                        "3",
                        "--execution_horizons",
                        "1",
                        "2",
                        "3",
                        "--eval_save_dir",
                        str(output_path),
                        "--eval_episodes",
                        "9",
                        "--device",
                        "cpu",
                    ]
                )

            self.assertEqual(result_path, output_path)
            build_env_assets.assert_called_once()
            build_dataset_and_mapper.assert_called_once()
            build_reward_and_leave.assert_called_once()
            build_agent.assert_called_once()
            load_checkpoint.assert_called_once()
            build_initial_state_table.assert_called_once()
            self.assertEqual(evaluate_one_horizon.call_count, 3)
            evaluated_horizons = [
                call.kwargs["execution_horizon"]
                for call in evaluate_one_horizon.call_args_list
            ]
            self.assertEqual(evaluated_horizons, [1, 2, 3])
            self.assertTrue(
                all(
                    call.kwargs["eval_episodes"] == 9
                    for call in evaluate_one_horizon.call_args_list
                )
            )


class EvaluationCollectionConfigTest(unittest.TestCase):
    """验证多轨迹评估的环境并行度和 buffer 容量。"""

    def test_default_buffer_retains_every_trajectory(self) -> None:
        """验证默认容量按全部最长轨迹预留，而非仅按并行环境数预留。"""

        parallel_envs, buffer_size = resolve_evaluation_collection_config(
            eval_episodes=10,
            requested_env_num=3,
            max_turn=100,
            force_length=80,
            buffer_size=0,
        )

        self.assertEqual(parallel_envs, 3)
        self.assertEqual(buffer_size, 4000)

    def test_explicit_small_buffer_is_rejected(self) -> None:
        """验证显式 buffer 无法保存全部轨迹时直接报错。"""

        with self.assertRaisesRegex(ValueError, "too small"):
            resolve_evaluation_collection_config(
                eval_episodes=4,
                requested_env_num=1,
                max_turn=10,
                force_length=10,
                buffer_size=39,
            )


if __name__ == "__main__":
    unittest.main()
