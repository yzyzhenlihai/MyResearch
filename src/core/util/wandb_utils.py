"""Weights & Biases 可选依赖的加载工具。"""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType
from typing import Optional, Union


def load_wandb(
    repo_root: Optional[Union[str, Path]] = None,
) -> Optional[ModuleType]:
    """按需加载 ``wandb``，未安装时返回 ``None``。

    训练器把实验跟踪视为可选能力，因此基础模块导入不应因为本机未安装
    ``wandb`` 而失败。调用方可通过返回值是否为 ``None`` 决定是否记录。

    Args:
        repo_root: 保留给历史调用方的项目根目录。当前实现不修改导入路径，
            该参数仅用于兼容训练器原有接口。

    Returns:
        成功导入时返回 ``wandb`` 模块，否则返回 ``None``。
    """

    del repo_root
    try:
        import wandb
    except ImportError:
        logging.getLogger(__name__).info(
            "未安装 wandb，将禁用在线实验跟踪。"
        )
        return None
    return wandb

