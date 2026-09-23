"""SwanLab 日志器的本地持久化与在线指标过滤测试。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from examples.our_model.logging.swanlab_logger import SwanLabLogger


class TestSwanLabLogger(unittest.TestCase):
    """验证评估元数据不会被误传为 SwanLab 图表指标。"""

    def test_log_keeps_metadata_locally_and_uploads_numeric_metrics(self) -> None:
        """本地保留字符串元数据，在线 payload 只包含数值标量。"""

        with tempfile.TemporaryDirectory() as temporary_dir:
            log_path = Path(temporary_dir) / "metrics.jsonl"
            logger = SwanLabLogger(
                project="test-project",
                run_name="test-run",
                config={},
                log_path=str(log_path),
                mode="disabled",
            )
            swanlab_mock = Mock()
            logger._swanlab = swanlab_mock

            logger.log(
                {
                    "NX_0_rew": 12.5,
                    "evaluation/execution_horizon_mode": "adaptive",
                    "evaluation/adr_comparison": "category_overlap",
                    "evaluation/adr_enabled": True,
                    "episode_lengths": [3, 4],
                    "missing_metric": None,
                },
                step=7,
            )
            logger.finish()

            swanlab_mock.log.assert_called_once_with(
                {
                    "NX_0_rew": 12.5,
                    "evaluation/adr_enabled": 1,
                },
                step=7,
            )
            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            logged_metrics = records[-1]["metrics"]
            self.assertEqual(
                logged_metrics["evaluation/execution_horizon_mode"],
                "adaptive",
            )
            self.assertEqual(
                logged_metrics["evaluation/adr_comparison"],
                "category_overlap",
            )
            self.assertEqual(logged_metrics["episode_lengths"], [3, 4])


if __name__ == "__main__":
    unittest.main()
