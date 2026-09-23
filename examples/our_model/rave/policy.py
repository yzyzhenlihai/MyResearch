"""每环境独立 H 的开环执行适配器，不使用 shadow replan。"""

from time import perf_counter

import torch


class RAVEPolicyAdapter:
    """通过稳定 env ID 管理 Session、待执行前缀和待接收反馈。

    reset/act/observe 可由并行收集器调用；batch 重排或缩减不影响缓存归属。
    每次 act 后必须 observe 实际反馈；终止时清除该环境缓存。
    """

    def __init__(self, policy, manager):
        """接收 CandidatePolicy 和统一 StateManager，创建空环境字典。"""
        self.policy, self.manager = policy, manager
        self.sessions, self.cache, self.waiting = {}, {}, {}
        self.plans = 0
        self.horizons = []
        self.latencies = []
        self.state_update_latencies = []

    def reset(self, env_id, user_id, initial_state):
        """重置指定环境的新会话；其他环境的历史和缓存保持不变。"""
        self.sessions[env_id] = self.manager.reset(user_id, initial_state)
        self.cache[env_id] = []
        self.waiting.pop(env_id, None)

    @torch.no_grad()
    def act(self, env_ids):
        """为一组唯一 env ID 返回 item 列表，仅对空缓存环境批量规划。

        已终止、重复 ID 或上步反馈尚未提交时抛 ValueError。
        GPU 同步确保记录包含真实计算时间；批次耗时均摊为每步延迟。
        """
        if not env_ids or len(set(env_ids)) != len(env_ids):
            raise ValueError("act requires nonempty unique environment IDs")
        for env_id in env_ids:
            if env_id not in self.sessions or self.sessions[env_id].done or env_id in self.waiting:
                raise ValueError("Reset and observe are required before act")
        device = next(self.policy.network.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = perf_counter()
        refill = [env_id for env_id in env_ids if not self.cache[env_id]]
        if refill:
            candidates, horizons, _ = self.policy.choose([self.sessions[key] for key in refill])
            for row, env_id in enumerate(refill):
                horizon = int(horizons[row])
                self.cache[env_id] = candidates[row, :horizon].tolist()
                self.horizons.append(horizon)
            self.plans += len(refill)
        actions = []
        for env_id in env_ids:
            item = self.cache[env_id].pop(0)
            self.waiting[env_id] = (item, env_id in refill)
            actions.append(item)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.latencies.extend([(perf_counter() - begin) / len(env_ids)] * len(env_ids))
        return actions

    def observe(self, env_id, raw_reward, natural_done=False, truncated=False):
        """提交实际展示反馈，返回四分量；采样截断不等于任务终止。

        Args:
            env_id (Hashable): 已调用 act 的环境。
            raw_reward (float): 本步可用原始偏好。
            natural_done (bool): 环境自然终止，不包括采样器截断。
            truncated (bool): 仅提示收集暂停；保留 Session 与缓存供续采。

        Raises:
            ValueError: 尚未发出动作。
        """
        if env_id not in self.waiting:
            raise ValueError("observe requires an outstanding action")
        device = next(self.policy.network.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = perf_counter()
        item, new_plan = self.waiting[env_id]
        reward = self.manager.advance(self.sessions[env_id], item, raw_reward,
                                      new_plan, natural_done)
        self.waiting.pop(env_id)
        if self.sessions[env_id].done:
            self.cache[env_id].clear()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.state_update_latencies.append(perf_counter() - begin)
        return reward
