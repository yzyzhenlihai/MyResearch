"""DORL-MAC 候选 chunk 跨 chunk 去重采样测试。"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from examples.our_model.models.leave_model import RuleBasedLeaveModel
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

    该 actor 用于稳定验证采样语义：未传 mask 时保留允许重复的旧行为；
    传入 mask 时只屏蔽 chunk 起点前的历史 item，chunk 内各位置仍可重复。
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
    state_dim: int = STATE_DIM,
) -> MACAgent:
    """构造仅用于候选采样的最小 CPU agent。

    Args:
        chunk_size (int): chunk 长度，必须大于 0。
        num_items (int): item 数量，必须大于 0。
        state_dim (int): 状态维度，rollout 测试需满足 `action_dim + 1`。

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
        state_dim=state_dim,
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


class ZeroDynamics(nn.Module):
    """返回零向量 next state 的测试 dynamics。"""

    def __init__(self, state_dim: int) -> None:
        """记录输出状态维度。

        Args:
            state_dim (int): 输出 next state 的维度。
        """

        super().__init__()
        self.state_dim = int(state_dim)

    def next_state(
        self,
        history_vectors: torch.Tensor,
        history_valid: torch.Tensor,
        chunk_actions: torch.Tensor,
        chunk_rewards: torch.Tensor,
        chunk_valid: torch.Tensor,
    ) -> torch.Tensor:
        """按 batch 大小返回零状态。

        Args:
            history_vectors (torch.Tensor): 历史向量，形状为 `(B, W, state_dim)`。
            history_valid (torch.Tensor): 历史有效标记，未在测试逻辑中使用。
            chunk_actions (torch.Tensor): chunk action embedding，未在测试逻辑中使用。
            chunk_rewards (torch.Tensor): chunk 每步 reward，未在测试逻辑中使用。
            chunk_valid (torch.Tensor): chunk 有效标记，未在测试逻辑中使用。

        Returns:
            torch.Tensor: 零 next state，形状为 `(B, state_dim)`。
        """

        del history_valid, chunk_actions, chunk_rewards, chunk_valid
        return torch.zeros(
            int(history_vectors.shape[0]),
            self.state_dim,
            dtype=history_vectors.dtype,
            device=history_vectors.device,
        )


class ZeroRewardModel:
    """返回零 reward 组成项的测试 reward model。"""

    def reward_components(
        self,
        raw_user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """构造与 DORLRewardModel 相同 key 的零 reward 字典。

        Args:
            raw_user_ids (torch.Tensor): 原始 user id，未在测试逻辑中使用。
            item_ids (torch.Tensor): 当前 item id，形状为 `(B,)`。
            history_item_ids (torch.Tensor): 历史 item id，未在测试逻辑中使用。

        Returns:
            dict[str, torch.Tensor]: 包含 `reward`、`pred_reward`、`entropy`
            和 `uncertainty` 的零张量。
        """

        del raw_user_ids, history_item_ids
        zeros = torch.zeros_like(item_ids, dtype=torch.float32)
        return {
            "reward": zeros,
            "pred_reward": zeros,
            "entropy": zeros,
            "uncertainty": zeros,
        }


def attach_rollout_dependencies(agent: MACAgent, max_turn: int = 10) -> None:
    """为采样测试 agent 挂载最小 rollout 依赖。

    Args:
        agent (MACAgent): 需要补齐 dependencies 的测试 agent。
        max_turn (int): rollout 最大底层步数。

    Returns:
        None.
    """

    agent.dynamics = ZeroDynamics(agent.state_dim)
    agent.reward_model = ZeroRewardModel()
    agent.leave_model = RuleBasedLeaveModel(
        list_feat_small=[[] for _ in range(agent.num_items)],
        num_leave_compute=1,
        leave_threshold=1.0,
        max_turn=max_turn,
    )
    agent._batch_user_ids = torch.zeros(1, dtype=torch.long)


def build_rollout_history_state(
    agent: MACAgent,
    recommended_mask: torch.Tensor,
) -> RolloutHistoryState:
    """构造单样本 rollout history state。

    Args:
        agent (MACAgent): 测试 agent。
        recommended_mask (torch.Tensor): chunk 起点前的历史 item mask。

    Returns:
        RolloutHistoryState: 可直接传给 `_rollout_one_chunk` 的历史缓存。
    """

    return RolloutHistoryState(
        history_vectors=torch.zeros((1, 1, agent.state_dim), dtype=torch.float32),
        history_valid=torch.zeros((1, 1), dtype=torch.float32),
        leave_history=torch.full((1, 1), -1, dtype=torch.long),
        recommended_mask=recommended_mask,
        env_step=torch.zeros(1, dtype=torch.long),
        terminated=torch.zeros(1, dtype=torch.bool),
    )


class MACAgentCrossChunkMaskSamplingTest(unittest.TestCase):
    """验证 mask 模式下跨 chunk 去重、chunk 内允许重复的语义。"""

    def test_masked_sampling_excludes_history_but_allows_chunk_duplicates(self) -> None:
        """验证每个候选均不含历史 item，但 chunk 内可以重复同一 item。"""

        torch.manual_seed(2023)
        agent = build_test_agent()
        states = torch.zeros((2, STATE_DIM), dtype=torch.float32)
        recommended_mask = torch.tensor(
            [
                [False, True, True, False, False, False],
                [False, True, False, True, False, False],
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
        self.assertTrue(bool((sampled_ids == 0).all()))

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

    def test_masked_sampling_allows_single_available_item_across_chunk(self) -> None:
        """验证剩余 1 个可选 item 时仍可采出长度为 K 的重复 chunk。"""

        agent = build_test_agent(chunk_size=CHUNK_SIZE, num_items=4)
        states = torch.zeros((1, STATE_DIM), dtype=torch.float32)
        recommended_mask = torch.tensor(
            [[True, True, False, True]],
            dtype=torch.bool,
        )

        sampled_ids, _ = agent.sample_candidate_chunks(
            states=states,
            num_samples=2,
            recommended_mask=recommended_mask,
        )

        self.assertTrue(bool((sampled_ids == 2).all()))

    def test_rollout_exact_repeat_ignores_duplicates_inside_same_chunk(self) -> None:
        """验证 chunk 内重复不会触发 exact repeat penalty。"""

        agent = build_test_agent(state_dim=ACTION_DIM + 1)
        attach_rollout_dependencies(agent)
        history_state = build_rollout_history_state(
            agent,
            recommended_mask=torch.zeros((1, NUM_ITEMS), dtype=torch.bool),
        )

        result, next_history_state = agent._rollout_one_chunk(
            torch.tensor([[0, 0, 1]], dtype=torch.long),
            history_state,
            repeat_policy="mask",
        )

        self.assertEqual(float(result.exact_repeat_ratio.item()), 0.0)
        self.assertEqual(result.selected_item_ids.cpu().tolist(), [[0, 0, 1]])
        self.assertTrue(bool(next_history_state.recommended_mask[0, 0]))
        self.assertTrue(bool(next_history_state.recommended_mask[0, 1]))

    def test_rollout_exact_repeat_detects_items_from_previous_chunks(self) -> None:
        """验证 chunk 起点前的历史 item 仍会触发 exact repeat penalty。"""

        agent = build_test_agent(state_dim=ACTION_DIM + 1)
        attach_rollout_dependencies(agent)
        history_state = build_rollout_history_state(
            agent,
            recommended_mask=torch.tensor(
                [[True, False, False, False, False, False]],
                dtype=torch.bool,
            ),
        )

        result, _ = agent._rollout_one_chunk(
            torch.tensor([[0, 0, 1]], dtype=torch.long),
            history_state,
            repeat_policy="mask",
        )

        self.assertEqual(float(result.exact_repeat_ratio.item()), 1.0)

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
