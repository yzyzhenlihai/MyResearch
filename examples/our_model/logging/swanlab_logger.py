"""SwanLab 可选日志封装。"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

DISABLED_MODES = {"offline", "disabled"}
"""不尝试在线 SwanLab 初始化的模式集合。"""


class SwanLabLogger:
    """统一封装 SwanLab 与本地 JSONL 日志。

    若环境未安装 `swanlab`，或 `SWANLAB_MODE=offline/disabled`，该类会自动
    降级为本地 JSONL 记录，不要求 API key。
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        config: Dict[str, Any],
        log_path: str,
        mode: Optional[str] = None,
    ) -> None:
        """初始化日志器。

        Args:
            project (str): SwanLab 项目名。
            run_name (str): run 名称。
            config (Dict[str, Any]): 配置快照。
            log_path (str): 本地 JSONL 日志路径。
            mode (Optional[str]): 显式日志模式；为 `None` 时读取 `SWANLAB_MODE`。
        """

        self.project = project
        self.run_name = run_name
        self.config = dict(config)
        self.mode = (mode or os.environ.get("SWANLAB_MODE", "offline")).lower()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_obj = self.log_path.open("a", encoding="utf-8")
        self._swanlab = None
        self._run = None
        self._init_online_if_available()
        self.log({"event/config_saved": 1.0}, step=0)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """记录一组指标。

        Args:
            metrics (Dict[str, Any]): 指标字典。
            step (Optional[int]): 全局步数。

        Returns:
            None.
        """

        payload = {"step": step, "metrics": self._to_jsonable(metrics)}
        self._file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file_obj.flush()
        if self._swanlab is not None:
            self._swanlab.log(metrics, step=step)

    def finish(self) -> None:
        """结束日志会话。

        Returns:
            None.
        """

        if self._swanlab is not None:
            self._swanlab.finish()
        self._file_obj.close()

    def _init_online_if_available(self) -> None:
        """在允许时尝试初始化 SwanLab 在线日志。

        Returns:
            None.
        """

        if self.mode in DISABLED_MODES:
            LOGGER.info("SwanLab 在线日志关闭，使用本地 JSONL：%s", self.log_path)
            return
        try:
            import swanlab  # type: ignore
        except ImportError:
            LOGGER.warning("未安装 swanlab，使用本地 JSONL：%s", self.log_path)
            return
        self._swanlab = swanlab
        self._run = swanlab.init(
            project=self.project,
            name=self.run_name,
            config=self.config,
            mode=self.mode,
        )
        LOGGER.info("SwanLab 初始化完成：project=%s, run=%s", self.project, self.run_name)

    @staticmethod
    def _to_jsonable(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """把指标转换为 JSON 可序列化对象。

        Args:
            metrics (Dict[str, Any]): 原始指标。

        Returns:
            Dict[str, Any]: JSON 友好的指标。
        """

        jsonable = {}
        for key, value in metrics.items():
            if hasattr(value, "item"):
                jsonable[key] = value.item()
            else:
                jsonable[key] = value
        return jsonable

