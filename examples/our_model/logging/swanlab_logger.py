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
        # 在指标日志开头写入一条 event=config 记录，把本次训练的完整超参配置直接
        # 存到 metrics_*.jsonl 头部；无需再额外查 resolved_config.json，便于事后
        # 定位与复现。写在 SwanLab 在线初始化之后，兼顾 SwanLab 面板的 config 面板。
        self._write_config_header()

    def _write_config_header(self) -> None:
        """把本次训练的配置作为 metrics 日志首行写入。

        Returns:
            None.
        """

        header = {
            "step": 0,
            "event": "config",
            "project": self.project,
            "run_name": self.run_name,
            "mode": self.mode,
            "config": self._to_jsonable(self.config),
        }
        self._file_obj.write(json.dumps(header, ensure_ascii=False) + "\n")
        self._file_obj.flush()
        # 兼容旧下游解析：保留一条 metrics 事件表明 config 已落盘。
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

    @classmethod
    def _to_jsonable(cls, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """把指标/配置转换为 JSON 可序列化对象。

        Args:
            metrics (Dict[str, Any]): 原始指标或配置。

        Returns:
            Dict[str, Any]: JSON 友好的字典。
        """

        return {key: cls._json_safe(value) for key, value in metrics.items()}

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """递归把任意值转换为 JSON 可序列化对象。

        Args:
            value (Any): 原始值。

        Returns:
            Any: JSON 友好的值；对无法序列化的对象退化为其 `str(...)`。
        """

        # 0 维张量 / numpy 标量 / torch 标量都实现了 .item()
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [cls._json_safe(v) for v in value]
        # numpy array 等
        if hasattr(value, "tolist"):
            try:
                return cls._json_safe(value.tolist())
            except (TypeError, ValueError):
                pass
        # Path、torch.device、其它自定义对象降级为字符串
        return str(value)

