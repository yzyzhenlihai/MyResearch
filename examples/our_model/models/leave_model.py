"""KuaiEnv 规则退出模型。"""

from __future__ import annotations

from collections import Counter
from typing import List, Sequence

import numpy as np
import torch

INVALID_ITEM_ID = -1
"""padding item id。"""


class RuleBasedLeaveModel:
    """复刻 KuaiEnv 的类别重复退出规则。

    该模型在 chunk rollout 内部逐步执行：每一步根据当前推荐前的最近
    `num_leave_compute` 个 item 统计类别，当候选 item 的任一类别出现次数
    大于 `leave_threshold` 时判定用户离开。
    """

    def __init__(
        self,
        list_feat_small: Sequence[Sequence[int]],
        num_leave_compute: int,
        leave_threshold: float,
        max_turn: int,
    ) -> None:
        """初始化规则退出模型。

        Args:
            list_feat_small (Sequence[Sequence[int]]): KuaiEnv 内部 item id 对应的类别列表。
            num_leave_compute (int): 退出规则观察的最近推荐数。
            leave_threshold (float): 类别出现次数阈值。
            max_turn (int): episode 最大长度。

        Raises:
            ValueError: 当参数非法时抛出。
        """

        if num_leave_compute <= 0:
            raise ValueError("num_leave_compute must be positive.")
        if max_turn <= 0:
            raise ValueError("max_turn must be positive.")
        self.list_feat_small = [list(features) for features in list_feat_small]
        self.num_leave_compute = int(num_leave_compute)
        self.leave_threshold = float(leave_threshold)
        self.max_turn = int(max_turn)

    def should_leave_batch(
        self,
        leave_history_item_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        env_steps: torch.Tensor,
    ) -> torch.Tensor:
        """批量判断候选 item 是否触发 KuaiEnv leave。

        Args:
            leave_history_item_ids (torch.Tensor): 最近历史 item，形状为 `(B, H)`。
            candidate_item_ids (torch.Tensor): 候选 item id，形状为 `(B,)`。
            env_steps (torch.Tensor): 当前 episode 已执行步数，形状为 `(B,)`。

        Returns:
            torch.Tensor: bool 标记，形状为 `(B,)`，True 表示当前步后终止。
        """

        history_np = leave_history_item_ids.detach().cpu().numpy()
        candidates_np = candidate_item_ids.detach().cpu().numpy()
        steps_np = env_steps.detach().cpu().numpy()
        done_flags: List[bool] = []
        for history, item_id, env_step in zip(history_np, candidates_np, steps_np):
            leave_done = self._should_leave_one(history, int(item_id), int(env_step))
            max_turn_done = int(env_step) >= self.max_turn - 1
            done_flags.append(bool(leave_done or max_turn_done))
        return torch.as_tensor(done_flags, dtype=torch.bool, device=candidate_item_ids.device)

    def append_items(
        self,
        leave_history_item_ids: torch.Tensor,
        item_ids: torch.Tensor,
        append_mask: torch.Tensor,
    ) -> torch.Tensor:
        """将已执行 item 追加到固定长度 leave 历史。

        Args:
            leave_history_item_ids (torch.Tensor): 历史 item，形状为 `(B, H)`。
            item_ids (torch.Tensor): 本步执行的 item id，形状为 `(B,)`。
            append_mask (torch.Tensor): 哪些样本需要追加，形状为 `(B,)`。

        Returns:
            torch.Tensor: 更新后的 leave 历史，形状为 `(B, H)`。
        """

        updated = leave_history_item_ids.clone()
        for batch_index in range(updated.shape[0]):
            if not bool(append_mask[batch_index].item()):
                continue
            updated[batch_index, :-1] = updated[batch_index, 1:].clone()
            updated[batch_index, -1] = item_ids[batch_index]
        return updated

    def first_violation_steps(
        self,
        leave_history_item_ids: torch.Tensor,
        chunk_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """判断完整 action chunk 中第一次触发退出规则的位置。

        Args:
            leave_history_item_ids (torch.Tensor): chunk 执行前的历史 item，
                形状为 `(B, H)`，无效位置为 `-1`。
            chunk_item_ids (torch.Tensor): 当前 chunk 内 item，形状为 `(B, K)`。

        Returns:
            torch.Tensor: 每个样本第一次触发退出规则的 step，下标从 0 开始；
            未触发时返回 `-1`。
        """

        history_np = leave_history_item_ids.detach().cpu().numpy()
        chunk_np = chunk_item_ids.detach().cpu().numpy()
        violation_steps: List[int] = []
        for history, chunk_items in zip(history_np, chunk_np):
            valid_history = [int(item) for item in history if int(item) != INVALID_ITEM_ID]
            first_step = -1
            for step_index, item_id in enumerate(chunk_items):
                item_id = int(item_id)
                if item_id == INVALID_ITEM_ID:
                    continue
                if self._should_leave_by_history(valid_history, item_id):
                    first_step = int(step_index)
                    break
                valid_history.append(item_id)
            violation_steps.append(first_step)
        return torch.as_tensor(violation_steps, dtype=torch.long, device=chunk_item_ids.device)

    def _should_leave_one(self, history: np.ndarray, item_id: int, env_step: int) -> bool:
        """判断单个样本是否触发离开。

        Args:
            history (np.ndarray): 最近历史 item id。
            item_id (int): 当前候选 item id。
            env_step (int): 当前 episode 步数。

        Returns:
            bool: 是否触发 leave。
        """

        if env_step == 0:
            return False
        valid_history = [int(item) for item in history if int(item) != INVALID_ITEM_ID]
        window_actions = valid_history[-self.num_leave_compute :]
        hist_categories: List[int] = []
        for historical_item in window_actions:
            hist_categories.extend(self.list_feat_small[historical_item])
        hist_counter = Counter(hist_categories)
        for category in self.list_feat_small[item_id]:
            if hist_counter[category] > self.leave_threshold:
                return True
        return False

    def _should_leave_by_history(self, valid_history: Sequence[int], item_id: int) -> bool:
        """基于完整有效历史判断候选 item 是否违反退出规则。

        Args:
            valid_history (Sequence[int]): 已执行 item 历史。
            item_id (int): 当前候选 item id。

        Returns:
            bool: 是否触发退出规则。
        """

        window_actions = list(valid_history)[-self.num_leave_compute :]
        hist_categories: List[int] = []
        for historical_item in window_actions:
            hist_categories.extend(self.list_feat_small[historical_item])
        hist_counter = Counter(hist_categories)
        for category in self.list_feat_small[item_id]:
            if hist_counter[category] > self.leave_threshold:
                return True
        return False
