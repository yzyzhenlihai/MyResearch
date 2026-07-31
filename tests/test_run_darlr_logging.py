"""DARLR 实验记录收尾逻辑的回归测试。"""

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""仓库根目录。"""

ADVANCE_EXAMPLES_PATH = REPOSITORY_ROOT / "examples" / "advance"
"""高级示例脚本目录。"""

PREVIOUS_SWANLAB_MODE = os.environ.get("SWANLAB_MODE")
os.environ["SWANLAB_MODE"] = "disabled"
sys.path.insert(0, str(ADVANCE_EXAMPLES_PATH))
run_darlr = importlib.import_module("run_DARLR")
if PREVIOUS_SWANLAB_MODE is None:
    os.environ.pop("SWANLAB_MODE", None)
else:
    os.environ["SWANLAB_MODE"] = PREVIOUS_SWANLAB_MODE


class FinishWandbTest(unittest.TestCase):
    """验证 SwanLab 收尾故障不会中断 DARLR 实验进程。"""

    def test_finish_wandb_calls_backend_when_enabled(self) -> None:
        """启用记录且后端正常时应调用一次 ``finish``。"""

        backend = mock.Mock()
        with mock.patch.object(run_darlr, "wandb", backend), mock.patch.object(
            run_darlr, "_SWANLAB_DISABLE_LOGIN", False
        ):
            run_darlr.finish_wandb()

        backend.finish.assert_called_once_with()

    def test_finish_wandb_suppresses_backend_failure(self) -> None:
        """远程收尾失败时应记录警告并正常返回。"""

        backend = mock.Mock()
        backend.finish.side_effect = RuntimeError("remote API unavailable")
        with mock.patch.object(run_darlr, "wandb", backend), mock.patch.object(
            run_darlr, "_SWANLAB_DISABLE_LOGIN", False
        ), mock.patch.object(run_darlr.logzero.logger, "warning") as warning:
            run_darlr.finish_wandb()

        backend.finish.assert_called_once_with()
        warning.assert_called_once()
        self.assertTrue(warning.call_args.kwargs["exc_info"])

    def test_finish_wandb_skips_disabled_backend(self) -> None:
        """显式禁用记录时不应调用后端。"""

        backend = mock.Mock()
        with mock.patch.object(run_darlr, "wandb", backend), mock.patch.object(
            run_darlr, "_SWANLAB_DISABLE_LOGIN", True
        ):
            run_darlr.finish_wandb()

        backend.finish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
