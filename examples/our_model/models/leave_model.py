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

        # 预构造 (num_items, num_categories) 的稠密 bool 矩阵，用于 O(1) 批量查表。
        num_items = len(self.list_feat_small)
        all_categories = set()
        for features in self.list_feat_small:
            all_categories.update(int(f) for f in features)
        self._num_categories = (max(all_categories) + 1) if all_categories else 0
        # bool 矩阵：item_feat_mat[item_id, category] = True 表示该 item 拥有该 category。
        # 训练期实际使用时会 lazily 迁移到 GPU（见 first_violation_steps_batch）。
        if self._num_categories > 0:
            item_feat_mat = torch.zeros(num_items, self._num_categories, dtype=torch.bool)
            for item_id, features in enumerate(self.list_feat_small):
                for feat in features:
                    item_feat_mat[item_id, int(feat)] = True
        else:
            item_feat_mat = torch.zeros(num_items, 0, dtype=torch.bool)
        self._item_feat_mat_cpu = item_feat_mat
        # 设备缓存，避免每次都搬运。
        self._item_feat_mat_by_device: dict = {}

    def _item_feat_matrix(self, device: torch.device) -> torch.Tensor:
        """按设备缓存 item→category bool 矩阵。

        Args:
            device (torch.device): 目标设备。

        Returns:
            torch.Tensor: `(num_items, num_categories)` bool 张量。
        """

        key = str(device)
        mat = self._item_feat_mat_by_device.get(key)
        if mat is None:
            mat = self._item_feat_mat_cpu.to(device=device)
            self._item_feat_mat_by_device[key] = mat
        return mat

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

        if leave_history_item_ids.shape[1] == 0:
            return leave_history_item_ids.clone()
        # 全量向量化：一次性对整个 batch 做左移一位并把新 item 填到末尾。
        item_ids_col = item_ids.to(dtype=leave_history_item_ids.dtype).view(-1, 1)
        shifted = torch.cat([leave_history_item_ids[:, 1:], item_ids_col], dim=1)
        mask = append_mask.to(dtype=torch.bool).view(-1, 1)
        return torch.where(mask, shifted, leave_history_item_ids)

    def first_violation_steps(
        self,
        leave_history_item_ids: torch.Tensor,
        chunk_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """判断完整 action chunk 中第一次触发退出规则的位置。

        向量化实现：把 `list_feat_small` 转成 `(num_items+1, num_categories)` 的 bool
        矩阵（`-1` 映射为全零 dummy 行），窗口内每类别的出现次数就是 window 的
        one-hot bool 求和。对 chunk 内的每个 step 做一次 batch tensor 判定并滑动窗口。

        Args:
            leave_history_item_ids (torch.Tensor): chunk 执行前的历史 item，
                形状为 `(B, H)`，无效位置为 `-1`。
            chunk_item_ids (torch.Tensor): 当前 chunk 内 item，形状为 `(B, K)`。

        Returns:
            torch.Tensor: 每个样本第一次触发退出规则的 step，下标从 0 开始；
            未触发时返回 `-1`。
        """

        device = chunk_item_ids.device
        batch_size = int(chunk_item_ids.shape[0])
        chunk_steps = int(chunk_item_ids.shape[1])
        if batch_size == 0:
            return torch.full((0,), -1, dtype=torch.long, device=device)
        if self._num_categories == 0 or chunk_steps == 0:
            return torch.full((batch_size,), -1, dtype=torch.long, device=device)

        item_feat = self._item_feat_matrix(device)  # (num_items, C) bool
        num_items = int(item_feat.shape[0])
        dummy_row = torch.zeros(1, self._num_categories, dtype=torch.bool, device=device)
        item_feat_ext = torch.cat([item_feat, dummy_row], dim=0)  # (num_items+1, C)
        invalid_id = num_items

        window_len = self.num_leave_compute
        history = leave_history_item_ids.to(device=device, dtype=torch.long)
        if history.shape[1] >= window_len:
            window = history[:, -window_len:].clone()
        else:
            pad = torch.full(
                (batch_size, window_len - history.shape[1]),
                INVALID_ITEM_ID,
                dtype=torch.long,
                device=device,
            )
            window = torch.cat([pad, history], dim=1)
        window = torch.where(
            window < 0,
            torch.full_like(window, invalid_id),
            window,
        )
        # (B, C) 每类别在窗口中的出现次数（float 便于与浮点阈值比较）。
        cat_count = item_feat_ext[window].sum(dim=1).to(torch.float32)
        threshold_val = float(self.leave_threshold)

        result = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        chunk_items_long = chunk_item_ids.to(device=device, dtype=torch.long)

        for step_index in range(chunk_steps):
            item_k = chunk_items_long[:, step_index]
            valid_item = item_k != INVALID_ITEM_ID
            item_k_safe = torch.where(item_k < 0, torch.full_like(item_k, invalid_id), item_k)
            cand_feats = item_feat_ext[item_k_safe]  # (B, C) bool
            cat_over_threshold = cat_count > threshold_val  # (B, C) bool
            violation = (cand_feats & cat_over_threshold).any(dim=1) & valid_item & active
            result = torch.where(
                violation,
                torch.full_like(result, step_index),
                result,
            )
            active = active & (~violation)
            if step_index == chunk_steps - 1:
                break
            append_mask = active & valid_item
            oldest_feats = item_feat_ext[window[:, 0]].to(torch.float32)
            cand_feats_f = cand_feats.to(torch.float32)
            new_cat_count = cat_count - oldest_feats + cand_feats_f
            cat_count = torch.where(append_mask.unsqueeze(1), new_cat_count, cat_count)
            new_window = torch.cat([window[:, 1:], item_k_safe.view(-1, 1)], dim=1)
            window = torch.where(append_mask.unsqueeze(1), new_window, window)
        return result

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
