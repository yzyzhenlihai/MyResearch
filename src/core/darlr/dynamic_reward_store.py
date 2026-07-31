"""DARLR 动态奖励的共享 previous-reward store.

论文公式 P'_U = |r̂_t - r̂_{t-1}| / (sim + div + eps) 中的 `r̂_{t-1}`
表示同一 `(user, item)` 上一次的动态奖励估计。原有实现让每个模拟环境
实例各自维护一个字典，会导致 N 个并行环境形成 N 份互相不一致的
上一轮奖励，且 checkpoint 时无法恢复。此模块提供整个训练 run 共享
的 store：

* `read(users, actions)`: 若 `(u,i)` 存在则返回上次动态奖励，否则
  回退到 world model 静态预测 `R0[u,i]` (第一次访问)。
* `stage_and_commit(users, actions, dynamic_rewards)`: 一个 batch
  内出现的重复 `(u,i)` 会按均值确定性提交，避免顺序敏感行为。
* `state_dict()/load_state_dict()`: 支持 checkpoint 与 resume。
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, Iterable, Tuple

import numpy as np


class DynamicRewardStore:
    """训练级共享 previous-reward store."""

    def __init__(self, initial_matrix: np.ndarray) -> None:
        """使用世界模型预测矩阵初始化。

        Args:
            initial_matrix (np.ndarray): 形状 `(num_users, num_items)` 的
                初始预测矩阵 `R0`。第一次访问 `(u,i)` 时用该值填充。
        """
        if initial_matrix is None or initial_matrix.ndim != 2:
            raise ValueError("initial_matrix must be a 2-D numpy array.")
        self._initial = np.asarray(initial_matrix, dtype=np.float32)
        self._store: Dict[Tuple[int, int], float] = {}
        self._duplicate_updates: int = 0
        self._total_updates: int = 0

    @property
    def shape(self) -> Tuple[int, int]:
        """初始矩阵形状。"""
        return self._initial.shape

    @property
    def duplicate_updates(self) -> int:
        """batch 内重复 `(u,i)` 更新次数（累计）。"""
        return self._duplicate_updates

    def read(self, users: Iterable[int], actions: Iterable[int]) -> np.ndarray:
        """向量化读取 `(u,i)` 的上一轮动态奖励。

        Args:
            users (Iterable[int]): 用户 id 序列。
            actions (Iterable[int]): 物品 id 序列，长度与 users 相同。

        Returns:
            np.ndarray: 一维 float32 数组，长度等于 users。
        """
        users_np = np.asarray(list(users), dtype=np.int64)
        actions_np = np.asarray(list(actions), dtype=np.int64)
        if users_np.shape != actions_np.shape:
            raise ValueError("users/actions must have the same length.")
        self._assert_indices_in_bounds(users_np, actions_np)
        result = np.empty(users_np.shape[0], dtype=np.float32)
        for idx in range(users_np.shape[0]):
            u = int(users_np[idx])
            i = int(actions_np[idx])
            cached = self._store.get((u, i))
            if cached is None:
                result[idx] = float(self._initial[u, i])
            else:
                result[idx] = float(cached)
        return result

    def stage_and_commit(
        self,
        users: Iterable[int],
        actions: Iterable[int],
        dynamic_rewards: Iterable[float],
    ) -> None:
        """在一个 batch 结束时确定性地提交动态奖励更新。

        同一 batch 内同一 `(u,i)` 出现多次时，按均值提交，避免顺序
        依赖。

        Args:
            users (Iterable[int]): 用户 id 序列。
            actions (Iterable[int]): 物品 id 序列。
            dynamic_rewards (Iterable[float]): 每个 `(u,i)` 对应的当前
                动态奖励估计。
        """
        users_np = np.asarray(list(users), dtype=np.int64)
        actions_np = np.asarray(list(actions), dtype=np.int64)
        rewards_np = np.asarray(list(dynamic_rewards), dtype=np.float32)
        if not (users_np.shape == actions_np.shape == rewards_np.shape):
            raise ValueError("users/actions/dynamic_rewards must share shape.")
        self._assert_indices_in_bounds(users_np, actions_np)

        aggregated: Dict[Tuple[int, int], list] = defaultdict(list)
        for u, i, r in zip(users_np.tolist(), actions_np.tolist(), rewards_np.tolist()):
            aggregated[(int(u), int(i))].append(float(r))

        for key, values in aggregated.items():
            if len(values) > 1:
                self._duplicate_updates += 1
            self._store[key] = float(np.mean(values))
            self._total_updates += 1

    def reset(self) -> None:
        """清空 store（一般用于新 seed 启动时）。"""
        self._store.clear()
        self._duplicate_updates = 0
        self._total_updates = 0

    def state_dict(self) -> Dict[str, Any]:
        """导出 checkpoint 需要的状态。"""
        return {
            "store": {f"{u}:{i}": v for (u, i), v in self._store.items()},
            "duplicate_updates": self._duplicate_updates,
            "total_updates": self._total_updates,
            "shape": tuple(self._initial.shape),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """从 checkpoint 恢复状态。"""
        raw_store = state.get("store", {})
        self._store.clear()
        for key, value in raw_store.items():
            if isinstance(key, tuple):
                u, i = key
            else:
                u_str, i_str = str(key).split(":")
                u, i = int(u_str), int(i_str)
            self._store[(int(u), int(i))] = float(value)
        self._duplicate_updates = int(state.get("duplicate_updates", 0))
        self._total_updates = int(state.get("total_updates", 0))
        expected_shape = state.get("shape")
        if expected_shape is not None:
            expected_shape = tuple(expected_shape)
            if expected_shape != tuple(self._initial.shape):
                raise ValueError(
                    "DynamicRewardStore shape mismatch on load: "
                    f"checkpoint={expected_shape}, current={self._initial.shape}."
                )

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: Tuple[int, int]) -> bool:
        return (int(key[0]), int(key[1])) in self._store

    def snapshot(self) -> Dict[Tuple[int, int], float]:
        """返回 store 的浅拷贝，用于调试或诊断。"""
        return copy.copy(self._store)

    def _assert_indices_in_bounds(self, users: np.ndarray, actions: np.ndarray) -> None:
        num_users, num_items = self._initial.shape
        if users.size and (users.min() < 0 or users.max() >= num_users):
            raise IndexError(
                f"DynamicRewardStore user id out of bounds: "
                f"min={int(users.min())}, max={int(users.max())}, "
                f"num_users={num_users}."
            )
        if actions.size and (actions.min() < 0 or actions.max() >= num_items):
            raise IndexError(
                f"DynamicRewardStore action id out of bounds: "
                f"min={int(actions.min())}, max={int(actions.max())}, "
                f"num_items={num_items}."
            )
