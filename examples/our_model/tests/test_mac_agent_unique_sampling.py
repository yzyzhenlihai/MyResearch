"""DORL-MAC 候选 chunk 严格去重采样测试。"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from examples.our_model.models.mac_agent import MACAgent, RolloutHistoryState
from examples.our_model.policy.action_mapper import ActionMapper


STATE_DIM = 2
"""测试使用的状态维度。"""

ACTION_DIM = 3
"""测试使用的 item embedding 维度。"""

CHUNK_SIZE = 3
"""测试使用的 chunk 长度。"""

NUM_ITEMS = 6
"""测试使用的 item 数量。"""

NUM_CANDIDATES = 16
"""每个状态生成的候选 chunk 数量。"""


class PreferredItemActor(nn.Module):
    """输出固定 logits、强烈偏好 item 0 的测试 actor。

    该 actor 用于稳定验证两种采样语义：未传 mask 时保留允许重复的
    旧行为；传入 mask 时，即使所有位置都偏好同一 item，动态 mask
    仍会强制候选 chunk 内 item 唯一。
    """

    def __init__(self, chunk_size: int, num_items: int) -> None:
        """初始化固定 logits actor。

        Args:
            chunk_size (int): chunk 长度，必须大于 0。
            num_items (int): item 数量，必须大于 0。

        Raises:
            ValueError: 当参数非正时抛出。
        """

        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if num_items <= 0:
            raise ValueError("num_items must be positive.")
        logits = torch.full((chunk_size, num_items), -100.0)
        logits[:, 0] = 100.0
        self.register_buffer("fixed_logits", logits)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """为 batch 中每个状态返回相同 logits。

        Args:
            states (torch.Tensor): 状态张量，形状为 `(B, state_dim)`。

        Returns:
            torch.Tensor: 固定 logits，形状为
            `(B, chunk_size, num_items)`。
        """

        return self.fixed_logits.unsqueeze(0).expand(
            int(states.shape[0]),
            -1,
            -1,
        )


def build_test_agent(
    chunk_size: int = CHUNK_SIZE,
    num_items: int = NUM_ITEMS,
) -> MACAgent:
    """构造仅用于候选采样的最小 CPU agent。

    Args:
        chunk_size (int): chunk 长度，必须大于 0。
        num_items (int): item 数量，必须大于 0。

    Returns:
        MACAgent: 使用固定 logits actor 的测试 agent。

    Raises:
        ValueError: 当 `chunk_size` 或 `num_items` 非正时，由模型构造器抛出。
    """

    device = torch.device("cpu")
    item_embeddings = torch.arange(
        num_items * ACTION_DIM,
        dtype=torch.float32,
    ).view(num_items, ACTION_DIM)
    action_mapper = ActionMapper(item_embeddings=item_embeddings, device=device)
    agent = MACAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        chunk_size=chunk_size,
        actor_hidden_dims=[8],
        value_hidden_dims=[8],
        gamma=0.9,
        device=device,
        action_mapper=action_mapper,
    )
    agent.actor = PreferredItemActor(
        chunk_size=chunk_size,
        num_items=num_items,
    )
    return agent


class MACAgentUniqueChunkSamplingTest(unittest.TestCase):
    """验证 mask 模式下历史去重和 chunk 内去重语义。"""

    def test_masked_sampling_excludes_history_and_chunk_duplicates(self) -> None:
        """验证每个候选均不含历史 item，且 chunk 内 item 两两不同。"""

        torch.manual_seed(2023)
        agent = build_test_agent()
        states = torch.zeros((2, STATE_DIM), dtype=torch.float32)
        recommended_mask = torch.tensor(
            [
                [True, True, False, False, False, False],
                [False, True, True, False, False, False],
            ],
            dtype=torch.bool,
        )

        sampled_ids, chunk_vectors = agent.sample_candidate_chunks(
            states=states,
            num_samples=NUM_CANDIDATES,
            recommended_mask=recommended_mask,
        )

        self.assertEqual(
            tuple(sampled_ids.shape),
            (NUM_CANDIDATES, 2, CHUNK_SIZE),
        )
        self.assertEqual(
            tuple(chunk_vectors.shape),
            (NUM_CANDIDATES, 2, CHUNK_SIZE * ACTION_DIM),
        )
        expanded_history_mask = recommended_mask.unsqueeze(0).expand(
            NUM_CANDIDATES,
            -1,
            -1,
        )
        selected_history_flags = torch.gather(
            expanded_history_mask,
            dim=2,
            index=sampled_ids,
        )
        self.assertFalse(bool(selected_history_flags.any()))
        sorted_ids = torch.sort(sampled_ids, dim=-1).values
        adjacent_duplicates = sorted_ids[..., 1:] == sorted_ids[..., :-1]
        self.assertFalse(bool(adjacent_duplicates.any()))

    def test_unmasked_sampling_preserves_repeat_allowed_behavior(self) -> None:
        """验证未传 mask 时仍允许同一 chunk 内重复，保持 FB 兼容性。"""

        agent = build_test_agent()
        states = torch.zeros((1, STATE_DIM), dtype=torch.float32)

        sampled_ids, _ = agent.sample_candidate_chunks(
            states=states,
            num_samples=4,
            recommended_mask=None,
        )

        self.assertTrue(bool((sampled_ids == 0).all()))

    def test_unique_sampling_rejects_insufficient_available_items(self) -> None:
        """验证剩余 item 少于 K 时明确失败，而不是静默解除 mask。"""

        agent = build_test_agent(chunk_size=CHUNK_SIZE, num_items=4)
        states = torch.zeros((1, STATE_DIM), dtype=torch.float32)
        recommended_mask = torch.tensor(
            [[True, True, False, False]],
            dtype=torch.bool,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Not enough unmasked items",
        ):
            agent.sample_candidate_chunks(
                states=states,
                num_samples=2,
                recommended_mask=recommended_mask,
            )

    def test_exhausted_rollout_rows_use_inactive_placeholder_sampling(self) -> None:
        """验证候选耗尽行被终止，且占位采样不会解除活跃行的历史 mask。"""

        agent = build_test_agent(chunk_size=1, num_items=NUM_ITEMS)
        history_state = RolloutHistoryState(
            history_vectors=torch.zeros((2, 1, STATE_DIM)),
            history_valid=torch.zeros((2, 1)),
            leave_history=torch.full((2, 1), -1, dtype=torch.long),
            recommended_mask=torch.tensor(
                [
                    [True, True, True, True, True, True],
                    [True, False, True, True, True, True],
                ],
                dtype=torch.bool,
            ),
            env_step=torch.zeros(2, dtype=torch.long),
            terminated=torch.zeros(2, dtype=torch.bool),
        )

        active_rows, sampling_mask = agent._prepare_rollout_sampling_mask(history_state)

        self.assertEqual(active_rows.tolist(), [False, True])
        self.assertEqual(history_state.terminated.tolist(), [True, False])
        self.assertFalse(bool(sampling_mask[0].any()))
        self.assertEqual(sampling_mask[1].tolist(), [True, False, True, True, True, True])

    def test_local_rollout_mask_ignores_offline_prefix_items(self) -> None:
        """验证局部 imagined rollout 不屏蔽离线轨迹前缀中的 item。"""

        agent = build_test_agent(chunk_size=1, num_items=NUM_ITEMS)
        history_state = RolloutHistoryState(
            history_vectors=torch.zeros((1, 1, STATE_DIM)),
            history_valid=torch.zeros((1, 1)),
            leave_history=torch.full((1, 1), -1, dtype=torch.long),
            recommended_mask=torch.ones((1, NUM_ITEMS), dtype=torch.bool),
            env_step=torch.zeros(1, dtype=torch.long),
            terminated=torch.zeros(1, dtype=torch.bool),
        )

        agent._reset_local_rollout_recommended_mask(history_state)

        self.assertFalse(bool(history_state.recommended_mask.any()))


if __name__ == "__main__":
    unittest.main()
