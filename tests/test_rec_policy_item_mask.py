"""推荐策略静态 item 候选掩码的回归测试。"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""仓库根目录。"""

VENDORED_TIANSHOU_DIR = REPOSITORY_ROOT / "src" / "tianshou"
"""仓库内置 Tianshou 包目录。"""

if str(VENDORED_TIANSHOU_DIR) not in sys.path:
    sys.path.insert(0, str(VENDORED_TIANSHOU_DIR))

from src.core.policy.RecPolicy import RecPolicy


class DummyDiscretePolicy(nn.Module):
    """提供 RecPolicy 构造所需最小接口的离散策略。"""

    action_type = "discrete"


class DummyStateTracker:
    """提供固定 item 数和嵌入维度的最小状态追踪器。"""

    num_item = 4
    emb_dim = 2


def build_policy() -> RecPolicy:
    """构造仅用于候选掩码测试的 RecPolicy。

    Returns:
        RecPolicy: CPU 上包含四个 item 的最小策略包装器。
    """

    arguments = SimpleNamespace(device="cpu", remove_recommended_ids=False)
    return RecPolicy(arguments, DummyDiscretePolicy(), DummyStateTracker())


def test_static_allowed_items_are_combined_with_recommend_mask() -> None:
    """验证静态候选集合同时作用于当前和下一状态掩码。"""

    policy = build_policy()
    policy.set_allowed_item_indexes(np.array([1, 3], dtype=np.int64))
    observation_mask, next_mask = policy._get_recommend_mask(
        False,
        batch_size=2,
        buffer=[],
        indices=None,
    )

    expected = torch.tensor([[False, True, False, True]]).repeat(2, 1)
    assert torch.equal(observation_mask.cpu(), expected)
    assert torch.equal(next_mask.cpu(), expected)


def test_static_allowed_items_reject_duplicates_and_out_of_range() -> None:
    """验证重复候选和越界候选会被显式拒绝。"""

    policy = build_policy()
    with pytest.raises(ValueError, match="duplicates"):
        policy.set_allowed_item_indexes([1, 1])
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        policy.set_allowed_item_indexes([4])
