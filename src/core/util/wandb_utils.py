"""W&B 可选导入工具。

该仓库训练时常在项目根目录下生成 `wandb/` 日志目录。Python 在从当前工作目录
导入模块时，会优先把这个目录识别为同名 namespace package，从而遮蔽真正的
 `wandb` 第三方包。这里提供一个更稳健的加载函数，优先返回具备官方 W&B
 常用接口的真实模块；若本机未安装，则返回 `None`。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional, Set


_REQUIRED_WANDB_ATTRS = ("init", "log", "finish")
"""判断导入结果是否为可用 wandb 模块所需的最小接口集合。"""


def _is_valid_wandb_module(module: Optional[ModuleType]) -> bool:
    """判断模块对象是否满足训练所需的 wandb 接口。"""

    if module is None:
        return False
    return all(hasattr(module, attr_name) for attr_name in _REQUIRED_WANDB_ATTRS)


def _build_blocked_paths(repo_root: Optional[Path] = None) -> Set[str]:
    """构造需要临时从 `sys.path` 中剔除的路径集合。"""

    blocked_paths = {"", str(Path.cwd().resolve())}
    if repo_root is not None:
        blocked_paths.add(str(repo_root.resolve()))
    return blocked_paths


def load_wandb(repo_root: Optional[Path] = None) -> Optional[ModuleType]:
    """稳健地加载 wandb。

    首先尝试正常导入；若导入到的是被仓库内 `wandb/` 目录遮蔽的 namespace
    package，则临时移除当前工作目录与仓库根目录后重新导入。

    Args:
        repo_root (Optional[Path]): 仓库根目录，用于屏蔽本地 `wandb/` 运行目录。

    Returns:
        Optional[ModuleType]: 可用的 wandb 模块；若环境中未安装则返回 `None`。
    """

    try:
        import wandb as imported_wandb
    except ImportError:
        imported_wandb = None

    if _is_valid_wandb_module(imported_wandb):
        return imported_wandb

    blocked_paths = _build_blocked_paths(repo_root=repo_root)
    original_sys_path = list(sys.path)
    previous_module = sys.modules.pop("wandb", None)
    resolved_wandb: Optional[ModuleType] = None

    try:
        sys.path = [path for path in original_sys_path if path not in blocked_paths]
        resolved_wandb = importlib.import_module("wandb")
    except ImportError:
        resolved_wandb = None
    finally:
        sys.path = original_sys_path
        if not _is_valid_wandb_module(resolved_wandb):
            if previous_module is not None:
                sys.modules["wandb"] = previous_module

    if _is_valid_wandb_module(resolved_wandb):
        return resolved_wandb
    return None
