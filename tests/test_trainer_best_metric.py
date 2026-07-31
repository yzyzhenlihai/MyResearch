"""Trainer 按 NX_0 指标选择最佳 checkpoint 的回归测试。"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""仓库根目录。"""

LOCAL_TIANSHOU_PATH = REPOSITORY_ROOT / "src" / "tianshou"
"""仓库内置 Tianshou 包目录。"""

if str(LOCAL_TIANSHOU_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_TIANSHOU_PATH))

PREVIOUS_SWANLAB_MODE = os.environ.get("SWANLAB_MODE")
os.environ["SWANLAB_MODE"] = "disabled"
from tianshou.trainer import base as trainer_base  # noqa: E402

if PREVIOUS_SWANLAB_MODE is None:
    os.environ.pop("SWANLAB_MODE", None)
else:
    os.environ["SWANLAB_MODE"] = PREVIOUS_SWANLAB_MODE


NX0_METRIC_KEY = "NX_0_rew"
"""论文 KuaiRec 主表对应的无重复推荐 reward key。"""


class DummyPolicy:
    """提供 BaseTrainer 测试所需最小策略接口。"""

    def __init__(self) -> None:
        """初始化空 callback 列表和训练模式状态。"""

        self.callbacks = []
        self.is_training = True

    def train(self) -> None:
        """把策略切换到训练模式。"""

        self.is_training = True

    def eval(self) -> None:
        """把策略切换到评估模式。"""

        self.is_training = False


class MinimalTrainer(trainer_base.BaseTrainer):
    """实现抽象更新接口的最小 trainer 测试替身。"""

    def policy_update_fn(self, data, result) -> None:
        """满足 BaseTrainer 抽象接口，本测试不执行策略更新。

        Args:
            data (dict): 训练进度数据，本测试不使用。
            result (dict): collector 结果，本测试不使用。

        Returns:
            None: 不执行任何更新。
        """

        del data, result


def build_trainer(
    best_metric_key: str = "rew",
    stop_fn=None,
    save_best_fn=None,
) -> MinimalTrainer:
    """构造可直接调用 `reset()` 和 `test_step()` 的最小 trainer。

    Args:
        best_metric_key (str): 最佳 checkpoint 使用的评估指标 key。
        stop_fn (Callable, optional): 提前停止回调。
        save_best_fn (Callable, optional): 保存最佳策略的回调。

    Returns:
        MinimalTrainer: 配置完成的轻量 trainer。
    """

    return MinimalTrainer(
        learning_type="onpolicy",
        policy=DummyPolicy(),
        max_epoch=2,
        batch_size=2,
        test_collector=mock.Mock(),
        episode_per_test=2,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_model_fn=mock.Mock(),
        best_metric_key=best_metric_key,
        verbose=False,
        show_progress=False,
    )


class TrainerBestMetricTest(unittest.TestCase):
    """验证 reset、best 更新和 stop 均使用配置的主评估指标。"""

    def test_default_metric_preserves_feedback_reward_behavior(self) -> None:
        """验证默认 `rew` 仍读取配套的 `rew_std`。"""

        trainer = build_trainer()
        trainer.epoch = 1
        result = {
            "rew": 8.0,
            "rew_std": 0.8,
            "NX_0_rew": 3.0,
            "NX_0_rew_std": 0.3,
        }

        with mock.patch.object(
            trainer_base,
            "test_episode",
            return_value=result,
        ):
            trainer.test_step()

        self.assertAlmostEqual(trainer.best_reward, 8.0)
        self.assertAlmostEqual(trainer.best_reward_std, 0.8)

    def test_nx0_metric_wins_when_feedback_reward_declines(self) -> None:
        """验证 FB 与 NX_0 方向相反时按 NX_0 更新最佳 epoch。"""

        save_best_fn = mock.Mock()
        trainer = build_trainer(
            best_metric_key=NX0_METRIC_KEY,
            save_best_fn=save_best_fn,
        )
        epoch_results = (
            {
                "rew": 100.0,
                "rew_std": 10.0,
                "NX_0_rew": 5.0,
                "NX_0_rew_std": 0.5,
            },
            {
                "rew": 90.0,
                "rew_std": 9.0,
                "NX_0_rew": 6.0,
                "NX_0_rew_std": 0.6,
            },
        )

        with mock.patch.object(
            trainer_base,
            "test_episode",
            side_effect=epoch_results,
        ):
            trainer.epoch = 1
            trainer.test_step()
            trainer.epoch = 2
            trainer.test_step()

        self.assertEqual(trainer.best_epoch, 2)
        self.assertAlmostEqual(trainer.best_reward, 6.0)
        self.assertAlmostEqual(trainer.best_reward_std, 0.6)
        self.assertEqual(save_best_fn.call_count, 2)

    def test_reset_initializes_best_reward_from_nx0(self) -> None:
        """验证训练前初次评估同样使用 NX_0，而不是只修复 epoch 更新。"""

        trainer = build_trainer(best_metric_key=NX0_METRIC_KEY)
        initial_result = {
            "rew": 50.0,
            "rew_std": 5.0,
            "NX_0_rew": 4.0,
            "NX_0_rew_std": 0.4,
        }

        with mock.patch.object(
            trainer_base,
            "test_episode",
            return_value=initial_result,
        ):
            trainer.reset()

        self.assertAlmostEqual(trainer.best_reward, 4.0)
        self.assertAlmostEqual(trainer.best_reward_std, 0.4)
        self.assertEqual(trainer.best_epoch, 0)

    def test_stop_function_receives_nx0_metric(self) -> None:
        """验证提前停止阈值不会继续读取允许重复推荐的 FB reward。"""

        observed_rewards = []

        def stop_fn(reward: float) -> bool:
            """记录 trainer 传入的主指标并保持训练。

            Args:
                reward (float): trainer 当前最佳主指标。

            Returns:
                bool: 固定返回 False。
            """

            observed_rewards.append(reward)
            return False

        trainer = build_trainer(
            best_metric_key=NX0_METRIC_KEY,
            stop_fn=stop_fn,
        )
        trainer.epoch = 1
        result = {
            "rew": 99.0,
            "rew_std": 9.9,
            "NX_0_rew": 7.0,
            "NX_0_rew_std": 0.7,
        }

        with mock.patch.object(
            trainer_base,
            "test_episode",
            return_value=result,
        ):
            trainer.test_step()

        self.assertEqual(observed_rewards, [7.0])

    def test_missing_configured_metric_fails_loudly(self) -> None:
        """验证缺失 NX_0 时不允许静默回退到 FB reward。"""

        trainer = build_trainer(best_metric_key=NX0_METRIC_KEY)
        trainer.epoch = 1
        result_without_nx0 = {
            "rew": 10.0,
            "rew_std": 1.0,
        }

        with mock.patch.object(
            trainer_base,
            "test_episode",
            return_value=result_without_nx0,
        ):
            with self.assertRaises(KeyError):
                trainer.test_step()


if __name__ == "__main__":
    unittest.main()
