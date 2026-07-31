"""DARLR selector 训练使用的数值归一化工具。"""

from __future__ import annotations

import torch


def standardize_tensor(
    values: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """对非空浮点张量执行总体标准化，并保留自动求导关系。

    Args:
        values (torch.Tensor): 待标准化的非空浮点张量。
        eps (float): 标准差分母的正下界，防止常量张量产生除零。

    Returns:
        torch.Tensor: 与输入形状、设备和 dtype 相同的标准化张量。常量
        输入会稳定地映射为全零。

    Raises:
        TypeError: 当输入不是浮点张量时抛出。
        ValueError: 当输入为空、包含非有限值或 ``eps`` 非正时抛出。
    """

    if not values.is_floating_point():
        raise TypeError("values must be a floating-point tensor.")
    if values.numel() == 0:
        raise ValueError("values must not be empty.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("values must contain only finite numbers.")

    centered_values = values - values.mean()
    denominator = values.std(unbiased=False).clamp_min(float(eps))
    return centered_values / denominator
