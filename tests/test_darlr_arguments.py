"""DARLR 稳定化参数与 NX_0 主指标参数的解析测试。"""

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""仓库根目录。"""

ADVANCE_EXAMPLES_PATH = REPOSITORY_ROOT / "examples" / "advance"
"""DARLR 训练入口所在目录。"""

POLICY_EXAMPLES_PATH = REPOSITORY_ROOT / "examples" / "policy"
"""通用训练参数模块所在目录。"""

for import_path in (ADVANCE_EXAMPLES_PATH, POLICY_EXAMPLES_PATH):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

PREVIOUS_SWANLAB_MODE = os.environ.get("SWANLAB_MODE")
os.environ["SWANLAB_MODE"] = "disabled"
run_darlr = importlib.import_module("run_DARLR")
policy_utils = importlib.import_module("policy_utils")
if PREVIOUS_SWANLAB_MODE is None:
    os.environ.pop("SWANLAB_MODE", None)
else:
    os.environ["SWANLAB_MODE"] = PREVIOUS_SWANLAB_MODE


class DarlrArgumentParsingTest(unittest.TestCase):
    """验证稳定化参数能够从命令行进入 DARLR 配置。"""

    @staticmethod
    def parse_darlr_args(*arguments: str):
        """调用 DARLR 独立参数解析器。

        Args:
            *arguments (str): 传入训练入口的命令行参数。

        Returns:
            argparse.Namespace: DARLR 参数解析结果。

        Raises:
            SystemExit: 当参数值不满足解析器约束时抛出。
        """

        with mock.patch.object(
            sys,
            "argv",
            ["run_DARLR.py", *arguments],
        ):
            return run_darlr.get_args_DARLR()

    @staticmethod
    def parse_common_args(*arguments: str):
        """调用通用训练参数解析器。

        Args:
            *arguments (str): 除必需环境参数外的命令行参数。

        Returns:
            argparse.Namespace: 通用参数解析结果。

        Raises:
            SystemExit: 当参数值不满足解析器约束时抛出。
        """

        with mock.patch.object(
            sys,
            "argv",
            [
                "run_DARLR.py",
                "--env",
                "KuaiEnv-v0",
                *arguments,
            ],
        ):
            return policy_utils.get_args_all("onpolicy")

    def test_parses_selector_stabilization_options(self) -> None:
        """验证 selector 独立熵和两种标准化开关均可显式设置。"""

        args = self.parse_darlr_args(
            "--selector_ent_coef",
            "0.01",
            "--selector_reward_normalization",
            "--selector_advantage_normalization",
            "--selector_normalization_eps",
            "1e-6",
        )

        self.assertAlmostEqual(args.selector_ent_coef, 0.01)
        self.assertTrue(args.selector_reward_normalization)
        self.assertTrue(args.selector_advantage_normalization)
        self.assertAlmostEqual(args.selector_normalization_eps, 1.0e-6)

    def test_rejects_invalid_selector_stabilization_values(self) -> None:
        """验证负 entropy coefficient 和非正 epsilon 会解析失败。"""

        invalid_argument_sets = (
            ("--selector_ent_coef", "-0.01"),
            ("--selector_normalization_eps", "0"),
            ("--selector_normalization_eps", "-1e-8"),
        )
        for invalid_arguments in invalid_argument_sets:
            with self.subTest(arguments=invalid_arguments):
                with self.assertRaises(SystemExit):
                    self.parse_darlr_args(*invalid_arguments)

    def test_parses_nx0_best_metric(self) -> None:
        """验证通用 trainer 主指标可切换为论文使用的 NX_0 reward。"""

        args = self.parse_common_args("--best-metric", "NX_0")

        self.assertEqual(args.best_metric, "NX_0")

    def test_best_metric_defaults_to_feedback_reward(self) -> None:
        """验证未显式设置时保持原有 FB reward 选择语义。"""

        args = self.parse_common_args()

        self.assertEqual(args.best_metric, "FB")

    def test_rejects_unknown_best_metric(self) -> None:
        """验证拼错主指标时立即失败，避免静默选择错误 checkpoint。"""

        with self.assertRaises(SystemExit):
            self.parse_common_args("--best-metric", "NX_0_rew")


if __name__ == "__main__":
    unittest.main()
