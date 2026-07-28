"""DORL-MAC 评估期动作块缓存与滚动重规划测试。"""

from __future__ import annotations

import unittest
from typing import Optional, Sequence

import torch

from examples.our_model.policy.dorl_mac_policy import (
    ADR_COMPARISON_CATEGORY_OVERLAP,
    DORLMACPolicyAdapter,
)


TEST_CHUNK_SIZE = 5
"""测试用规划长度 K。"""

TEST_NUM_ITEMS = 100
"""测试用离散 item 数量。"""


class FakeAgent:
    """生成可预测 chunk 的最小 agent 测试替身。

    Attributes:
        chunk_size (int): 每次规划生成的 chunk 长度。
        select_call_count (int): `select_chunks` 的累计调用次数。
    """

    def __init__(self, chunk_size: int) -> None:
        """初始化测试 agent。

        Args:
            chunk_size (int): 动作块长度，必须大于 0。

        Raises:
            ValueError: 当 `chunk_size` 不大于 0 时抛出。
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        self.chunk_size = int(chunk_size)
        self.select_call_count = 0

    def select_chunks(
        self,
        states: torch.Tensor,
        num_samples: int,
        recommended_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按调用次数生成不同的确定性 chunk。

        Args:
            states (torch.Tensor): 当前状态，形状为 `(B, state_dim)`。
            num_samples (int): 兼容真实 agent 的候选数参数。
            recommended_mask (Optional[torch.Tensor]): 兼容真实 agent 的屏蔽矩阵。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: item id chunk 与占位 chunk 向量。
        """

        del num_samples, recommended_mask
        batch_size = int(states.shape[0])
        chunk_start = self.select_call_count * 10
        self.select_call_count += 1
        item_ids = torch.arange(
            chunk_start,
            chunk_start + self.chunk_size,
            dtype=torch.long,
            device=states.device,
        ).repeat(batch_size, 1)
        chunk_vectors = item_ids.to(dtype=torch.float32)
        return item_ids, chunk_vectors


class FakeActionMapper:
    """只提供适配器构造所需属性的 action mapper 测试替身。"""

    def __init__(self, num_items: int) -> None:
        """初始化 action mapper 测试替身。

        Args:
            num_items (int): 离散 item 数量，必须大于 0。

        Raises:
            ValueError: 当 `num_items` 不大于 0 时抛出。
        """

        if num_items <= 0:
            raise ValueError("num_items must be positive.")
        self.num_items = int(num_items)
        self.action_dim = 1


def build_test_adapter(
    execution_horizon: Optional[int],
    enable_open_loop_diagnostics: bool = False,
    item_categories: Optional[Sequence[Sequence[int]]] = None,
    provide_item_categories: bool = True,
) -> tuple[DORLMACPolicyAdapter, FakeAgent]:
    """构造 CPU 上的最小策略适配器。

    Args:
        execution_horizon (Optional[int]): 每次规划后连续执行步数；
            `None` 表示使用完整 chunk。
        enable_open_loop_diagnostics (bool): 是否启用 ADR shadow replan。
        item_categories (Optional[Sequence[Sequence[int]]]): 可选的测试
            item 多标签类别映射；未提供时为每个 item 分配独立类别。
        provide_item_categories (bool): 是否向适配器传入类别映射，用于
            测试诊断开启但映射缺失的异常分支。

    Returns:
        tuple[DORLMACPolicyAdapter, FakeAgent]: 策略适配器与可检查调用次数的 agent。
    """

    agent = FakeAgent(chunk_size=TEST_CHUNK_SIZE)
    resolved_item_categories = item_categories
    if resolved_item_categories is None:
        resolved_item_categories = [
            [item_id] for item_id in range(TEST_NUM_ITEMS)
        ]
    adapter = DORLMACPolicyAdapter(
        agent=agent,  # type: ignore[arg-type]
        state_tracker=torch.nn.Identity(),
        action_mapper=FakeActionMapper(TEST_NUM_ITEMS),  # type: ignore[arg-type]
        num_samples_test=4,
        device=torch.device("cpu"),
        execution_horizon=execution_horizon,
        enable_open_loop_diagnostics=enable_open_loop_diagnostics,
        item_categories=(
            resolved_item_categories if provide_item_categories else None
        ),
    )
    return adapter, agent


class DORLMACPolicyExecutionHorizonTest(unittest.TestCase):
    """验证 execution horizon 对 chunk 缓存重规划时机的控制。"""

    def test_default_execution_horizon_consumes_full_chunk(self) -> None:
        """验证未显式设置 H 时完整执行 K 个动作。"""

        adapter, agent = build_test_adapter(execution_horizon=None)
        states = torch.zeros(1, 3)
        reset_mask = torch.zeros(1, dtype=torch.bool)
        outputs = [
            int(adapter._next_chunk_item_ids(states, reset_mask, None).item())
            for _ in range(TEST_CHUNK_SIZE)
        ]

        self.assertEqual(outputs, [0, 1, 2, 3, 4])
        self.assertEqual(agent.select_call_count, 1)
        self.assertEqual(adapter.execution_horizon, TEST_CHUNK_SIZE)

    def test_short_execution_horizon_discards_cached_suffix(self) -> None:
        """验证 H=2 时每执行两步就丢弃后缀并重新规划。"""

        adapter, agent = build_test_adapter(execution_horizon=2)
        states = torch.zeros(1, 3)
        reset_mask = torch.zeros(1, dtype=torch.bool)
        outputs = [
            int(adapter._next_chunk_item_ids(states, reset_mask, None).item())
            for _ in range(TEST_CHUNK_SIZE)
        ]

        self.assertEqual(outputs, [0, 1, 10, 11, 20])
        self.assertEqual(agent.select_call_count, 3)

    def test_episode_reset_replans_only_reset_rows(self) -> None:
        """验证并行环境中 episode reset 只使对应行重新规划。"""

        adapter, agent = build_test_adapter(execution_horizon=3)
        states = torch.zeros(2, 3)
        no_reset = torch.zeros(2, dtype=torch.bool)
        first_output = adapter._next_chunk_item_ids(states, no_reset, None)
        second_output = adapter._next_chunk_item_ids(
            states,
            torch.tensor([True, False]),
            None,
        )

        self.assertEqual(first_output.tolist(), [0, 0])
        self.assertEqual(second_output.tolist(), [10, 1])
        self.assertEqual(agent.select_call_count, 2)

    def test_invalid_execution_horizon_is_rejected(self) -> None:
        """验证 H 超出 `[1, K]` 时立即抛出明确异常。"""

        with self.assertRaisesRegex(ValueError, "1 <= H <= chunk_size"):
            build_test_adapter(execution_horizon=0)
        with self.assertRaisesRegex(ValueError, "1 <= H <= chunk_size"):
            build_test_adapter(execution_horizon=TEST_CHUNK_SIZE + 1)

    def test_adr_counts_cached_tail_disagreements(self) -> None:
        """验证类别无交集时 ADR 统计分歧、暴露和位置指标。"""

        adapter, _ = build_test_adapter(
            execution_horizon=2,
            enable_open_loop_diagnostics=True,
        )
        states = torch.zeros(1, 3)
        reset_mask = torch.zeros(1, dtype=torch.bool)
        first_item = adapter._next_chunk_item_ids(states, reset_mask, None)
        second_item = adapter._next_chunk_item_ids(states, reset_mask, None)
        metrics = adapter.get_open_loop_diagnostics()

        self.assertEqual(first_item.tolist(), [0])
        self.assertEqual(second_item.tolist(), [1])
        self.assertEqual(metrics["ADR"], 1.0)
        self.assertEqual(metrics["ADR_exposed"], 0.5)
        self.assertEqual(metrics["ADR_disagreements"], 1.0)
        self.assertEqual(metrics["ADR_cached_tail_steps"], 1.0)
        self.assertEqual(metrics["ADR_total_steps"], 2.0)
        self.assertEqual(metrics["ADR@position_2"], 1.0)

    def test_adr_treats_shared_category_as_agreement(self) -> None:
        """验证不同 item 只要共享一个类别就不计入 ADR 分歧。"""

        item_categories = [
            [item_id] for item_id in range(TEST_NUM_ITEMS)
        ]
        # 第 2 步缓存 item=1，shadow replan 首 item=10；二者共享类别 9。
        item_categories[1] = [7, 9]
        item_categories[10] = [9, 11]
        adapter, _ = build_test_adapter(
            execution_horizon=2,
            enable_open_loop_diagnostics=True,
            item_categories=item_categories,
        )
        states = torch.zeros(1, 3)
        reset_mask = torch.zeros(1, dtype=torch.bool)
        adapter._next_chunk_item_ids(states, reset_mask, None)
        adapter._next_chunk_item_ids(states, reset_mask, None)
        metrics = adapter.get_open_loop_diagnostics()

        self.assertEqual(metrics["ADR"], 0.0)
        self.assertEqual(metrics["ADR_exposed"], 0.0)
        self.assertEqual(metrics["ADR_disagreements"], 0.0)
        self.assertEqual(metrics["ADR_cached_tail_steps"], 1.0)
        self.assertEqual(metrics["ADR@position_2"], 0.0)
        self.assertEqual(
            adapter.adr_comparison,
            ADR_COMPARISON_CATEGORY_OVERLAP,
        )
        self.assertEqual(adapter.adr_num_categories, TEST_NUM_ITEMS)

    def test_adr_requires_item_categories(self) -> None:
        """验证启用 ADR 但未提供类别映射时立即报错。"""

        with self.assertRaisesRegex(ValueError, "item_categories is required"):
            build_test_adapter(
                execution_horizon=2,
                enable_open_loop_diagnostics=True,
                provide_item_categories=False,
            )

    def test_h1_reports_no_conditional_adr(self) -> None:
        """验证 H=1 无缓存后缀时 ADR 为 N/A 且暴露率为零。"""

        adapter, _ = build_test_adapter(
            execution_horizon=1,
            enable_open_loop_diagnostics=True,
        )
        states = torch.zeros(1, 3)
        reset_mask = torch.zeros(1, dtype=torch.bool)
        for _ in range(3):
            adapter._next_chunk_item_ids(states, reset_mask, None)
        metrics = adapter.get_open_loop_diagnostics()

        self.assertIsNone(metrics["ADR"])
        self.assertEqual(metrics["ADR_exposed"], 0.0)
        self.assertEqual(metrics["ADR_cached_tail_steps"], 0.0)
        self.assertEqual(metrics["ADR_total_steps"], 3.0)


if __name__ == "__main__":
    unittest.main()
