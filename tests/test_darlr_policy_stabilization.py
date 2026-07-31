"""DARLR selector 稳定化损失的轻量行为测试。"""

import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""仓库根目录。"""

LOCAL_TIANSHOU_PATH = REPOSITORY_ROOT / "src" / "tianshou"
"""仓库内置 Tianshou 包目录。"""

if str(LOCAL_TIANSHOU_PATH) not in sys.path:
    sys.path.insert(0, str(LOCAL_TIANSHOU_PATH))

from tianshou.data import Batch  # noqa: E402

from src.core.policy.darlr import DARLRPolicy  # noqa: E402


SELECTOR_ENTROPY_COEFFICIENT = 0.25
"""测试 selector 独立熵正则使用的系数。"""

RECOMMENDER_ENTROPY_COEFFICIENT = 99.0
"""用于确认 selector 不再误用推荐器熵系数的哨兵值。"""

VALUE_LOSS_COEFFICIENT = 0.5
"""selector critic 损失测试使用的权重。"""

FLOAT_TOLERANCE = 1.0e-6
"""损失标量比较的绝对误差。"""


class ConstantSelectorActor(nn.Module):
    """返回均匀二分类 logits 的轻量 selector actor。"""

    def __init__(self) -> None:
        """初始化一个可训练哨兵参数，满足策略模块接口。"""

        super().__init__()
        self.logit_offset = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        recommender_state: torch.Tensor,
        current_preference: torch.Tensor,
        selected_state: torch.Tensor,
        candidate_preferences: torch.Tensor,
    ) -> torch.Tensor:
        """生成与候选用户数量匹配的均匀 logits。

        Args:
            recommender_state (torch.Tensor): 推荐器状态，首维为批大小。
            current_preference (torch.Tensor): 当前用户偏好，本测试不使用。
            selected_state (torch.Tensor): 已选用户状态，本测试不使用。
            candidate_preferences (torch.Tensor): 候选用户偏好，第二维为
                候选数量。

        Returns:
            torch.Tensor: 形状为 `(batch_size, candidate_size)` 的 logits。
        """

        del current_preference, selected_state
        batch_size = recommender_state.shape[0]
        candidate_size = candidate_preferences.shape[1]
        return self.logit_offset.expand(batch_size, candidate_size)


class ZeroSelectorCritic(nn.Module):
    """始终返回零价值估计的轻量 selector critic。"""

    def __init__(self) -> None:
        """初始化一个可训练零值参数，满足 critic 模块接口。"""

        super().__init__()
        self.value_offset = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        recommender_state: torch.Tensor,
        current_preference: torch.Tensor,
        selected_state: torch.Tensor,
    ) -> torch.Tensor:
        """返回每个样本的零价值估计。

        Args:
            recommender_state (torch.Tensor): 推荐器状态，首维为批大小。
            current_preference (torch.Tensor): 当前用户偏好，本测试不使用。
            selected_state (torch.Tensor): 已选用户状态，本测试不使用。

        Returns:
            torch.Tensor: 形状为 `(batch_size,)` 的零价值张量。
        """

        del current_preference, selected_state
        return self.value_offset.expand(recommender_state.shape[0])


class ZeroStateEncoder(nn.Module):
    """将任意已选用户序列编码为零向量的测试替身。"""

    def forward(
        self,
        selected_embeddings: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回与偏好维度一致的零状态。

        Args:
            selected_embeddings (torch.Tensor): 已选用户嵌入。
            selected_mask (torch.Tensor): 已选位置掩码，本测试不使用。

        Returns:
            torch.Tensor: 形状为 `(batch_size, preference_dim)` 的零张量。
        """

        del selected_mask
        return torch.zeros(
            selected_embeddings.shape[0],
            selected_embeddings.shape[-1],
            dtype=selected_embeddings.dtype,
            device=selected_embeddings.device,
        )


def build_lightweight_policy(
    reward_normalization: bool,
    advantage_normalization: bool,
) -> DARLRPolicy:
    """绕过完整推荐器构造，建立 selector 损失测试所需最小策略。

    Args:
        reward_normalization (bool): 是否启用 selector reward 标准化。
        advantage_normalization (bool): 是否启用 selector advantage 标准化。

    Returns:
        DARLRPolicy: 仅配置 `_compute_selector_loss()` 所需属性的策略。
    """

    policy = DARLRPolicy.__new__(DARLRPolicy)
    nn.Module.__init__(policy)
    policy.predicted_mat = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    policy.preference_encoder = nn.Identity()
    policy.selector_state_encoder = ZeroStateEncoder()
    policy.selector_actor = ConstantSelectorActor()
    policy.selector_critic = ZeroSelectorCritic()
    policy.selector_k = 1
    policy.selector_candidate_size = 2
    policy.selector_policy_mode = "learned"
    policy.selector_lambda_s = 1.0
    policy.selector_lambda_d = 0.05
    policy.selector_discount_factor = 0.9
    policy._weight_vf = VALUE_LOSS_COEFFICIENT
    policy._weight_ent = RECOMMENDER_ENTROPY_COEFFICIENT
    policy.selector_ent_coef = SELECTOR_ENTROPY_COEFFICIENT
    policy.selector_reward_normalization = reward_normalization
    policy.selector_advantage_normalization = advantage_normalization
    policy.selector_normalization_eps = 1.0e-8
    return policy


def build_minibatch() -> Batch:
    """构造两样本、单选择步的确定性 selector 上下文。

    Returns:
        Batch: selector reward 分别为 1 和 3 的最小训练批。
    """

    context = Batch(
        current_users=torch.tensor([0, 1], dtype=torch.long),
        candidate_users=torch.tensor([[1, 2], [0, 2]], dtype=torch.long),
        selected_users=torch.tensor([[1], [2]], dtype=torch.long),
        selector_actions=torch.tensor([[0], [1]], dtype=torch.long),
        selector_rewards=torch.tensor([[1.0], [3.0]], dtype=torch.float32),
        base_reward=torch.tensor([0.1, 0.2], dtype=torch.float32),
        similarity_per_step=torch.tensor(
            [[0.2], [0.4]],
            dtype=torch.float32,
        ),
        diversity_per_step=torch.tensor(
            [[0.0], [0.5]],
            dtype=torch.float32,
        ),
        dynamic_reward=torch.tensor([0.2, 0.3], dtype=torch.float32),
        dynamic_uncertainty=torch.tensor([0.01, 0.02], dtype=torch.float32),
    )
    return Batch(policy=Batch(darlr_context=context))


class SelectorStabilizationLossTest(unittest.TestCase):
    """验证 selector 独立熵系数及两种标准化开关的实际损失语义。"""

    def compute_loss(
        self,
        reward_normalization: bool,
        advantage_normalization: bool,
    ):
        """执行一次轻量 selector 损失计算。

        Args:
            reward_normalization (bool): 是否启用 reward 标准化。
            advantage_normalization (bool): 是否启用 advantage 标准化。

        Returns:
            tuple: `_compute_selector_loss()` 返回的五元组。
        """

        policy = build_lightweight_policy(
            reward_normalization=reward_normalization,
            advantage_normalization=advantage_normalization,
        )
        recommender_state = torch.zeros((2, 2), dtype=torch.float32)
        return policy._compute_selector_loss(
            minibatch=build_minibatch(),
            recommender_state=recommender_state,
        )

    def test_uses_selector_specific_entropy_coefficient(self) -> None:
        """验证 selector 总损失不再复用推荐器的 entropy coefficient。"""

        total_loss, actor_loss, value_loss, entropy, _ = self.compute_loss(
            reward_normalization=False,
            advantage_normalization=False,
        )
        expected_total = (
            actor_loss
            + VALUE_LOSS_COEFFICIENT * value_loss
            - SELECTOR_ENTROPY_COEFFICIENT * entropy
        )

        self.assertAlmostEqual(
            float(total_loss.item()),
            float(expected_total.item()),
            delta=FLOAT_TOLERANCE,
        )
        self.assertLess(abs(float(total_loss.item())), 10.0)

    def test_reward_normalization_changes_return_and_value_scale(self) -> None:
        """验证 reward 标准化发生在折扣回报和 critic 损失计算之前。"""

        _, _, raw_value_loss, _, _ = self.compute_loss(
            reward_normalization=False,
            advantage_normalization=False,
        )
        _, _, normalized_value_loss, _, _ = self.compute_loss(
            reward_normalization=True,
            advantage_normalization=False,
        )

        self.assertAlmostEqual(
            float(raw_value_loss.item()),
            5.0,
            delta=FLOAT_TOLERANCE,
        )
        self.assertAlmostEqual(
            float(normalized_value_loss.item()),
            1.0,
            delta=FLOAT_TOLERANCE,
        )

    def test_advantage_normalization_changes_actor_scale_only(self) -> None:
        """验证 advantage 标准化作用于 actor，而不改变 critic 目标。"""

        _, raw_actor_loss, raw_value_loss, _, _ = self.compute_loss(
            reward_normalization=False,
            advantage_normalization=False,
        )
        _, normalized_actor_loss, normalized_value_loss, _, _ = (
            self.compute_loss(
                reward_normalization=False,
                advantage_normalization=True,
            )
        )

        self.assertAlmostEqual(
            float(raw_actor_loss.item()),
            2.0 * math.log(2.0),
            delta=FLOAT_TOLERANCE,
        )
        self.assertAlmostEqual(
            float(normalized_actor_loss.item()),
            0.0,
            delta=FLOAT_TOLERANCE,
        )
        self.assertAlmostEqual(
            float(normalized_value_loss.item()),
            float(raw_value_loss.item()),
            delta=FLOAT_TOLERANCE,
        )


if __name__ == "__main__":
    unittest.main()
