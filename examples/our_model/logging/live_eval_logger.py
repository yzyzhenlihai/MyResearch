"""在独立进程中实时上传 RAVE-H 的逐 epoch 评估指标。"""

import logging
from pathlib import Path
from typing import Any

from examples.our_model.logging.swanlab_logger import SwanLabLogger

LOGGER = logging.getLogger(__name__)


def run_live_eval_logger(
    metric_queue: Any,
    project: str,
    run_name: str,
    config: dict,
    log_path: str | Path,
) -> None:
    """消费父进程发送的评估指标并按 epoch 写入独立 SwanLab run。

    该函数设计为 ``multiprocessing`` 的 spawn 进程入口，使训练 run 和
    RAVE-H-Eval run 分属不同进程，绕开 SwanLab 单进程单 run 的限制。

    Args:
        metric_queue (Any): 支持 ``get()`` 的进程安全队列；``None`` 为结束信号。
        project (str): 独立评估项目名。
        run_name (str): 独立评估 run 名。
        config (dict): 完整实验配置快照，必须可 pickle。
        log_path (str | Path): 本地评估指标 JSONL 路径。

    Returns:
        None.

    Raises:
        ValueError: 当队列消息缺少合法 epoch 或 metrics 字典时抛出。
    """
    tracker = SwanLabLogger(
        project=project,
        run_name=run_name,
        config=config,
        log_path=str(log_path),
    )
    try:
        while True:
            record = metric_queue.get()
            if record is None:
                break
            if not isinstance(record, dict):
                raise ValueError("Live evaluation record must be a dictionary")
            epoch = record.get("epoch")
            metrics = record.get("metrics")
            if not isinstance(epoch, int) or epoch <= 0:
                raise ValueError("Live evaluation epoch must be a positive integer")
            if not isinstance(metrics, dict):
                raise ValueError("Live evaluation metrics must be a dictionary")
            tracker.log(metrics, step=epoch)
            LOGGER.info("RAVE-H-Eval 实时上传完成：epoch=%s", epoch)
    finally:
        tracker.finish()
