"""RAVE-H 的手算语义、因果性、训练与独立缓存集成测试。"""

from argparse import Namespace
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import call, Mock, patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from examples.our_model.runners.run_rave import (
    apply_ordinary_dorl_reward_config,
    batch_session_training_reward,
    build_parser,
    calibration_log_metrics,
    default_eval_run_name,
    evaluation_log_metrics,
    resolve_epoch_schedule,
    resolve_output_dir,
    resolve_qv_type,
    set_collection_seed,
    session_training_reward,
    train,
    validation_log_metrics,
)
from examples.our_model.logging.live_eval_logger import run_live_eval_logger
from examples.our_model.models.mac_agent import RolloutHistoryState, RolloutResult
from examples.our_model.rave.assets import freeze_categories
from examples.our_model.rave.dorl_policy import RAVEDORLPolicyAdapter
from examples.our_model.rave.evaluation import episode_metrics, monte_carlo_calibration
from examples.our_model.rave.model import PrefixQV, Utility, select_prefix
from examples.our_model.rave.policy import RAVEPolicyAdapter
from examples.our_model.rave.state import StateManager
from examples.our_model.rave.training import (
    CandidatePolicy,
    RAVETrainer,
    TrainingSimulator,
    assign_episode_returns,
    chunk_targets,
    collect_complete_episodes,
    collect_imagined_rollouts,
    collect_tensor_imagined_rollouts,
    prefix_targets,
)
from src.core.evaluation.utils import get_dominant_values
from src.tianshou.tianshou.data import Batch

TEST_SEED = 17
CATALOG_SIZE = 6
CHUNK_SIZE = 3


class ToyDynamics(nn.Module):
    """非结合转移用于检测错误的按 H 提交；真实动作和固定锚点共同决定状态。"""

    chunk_size = CHUNK_SIZE

    def forward(self, states, actions, valid):
        """接收状态、动作与前缀 mask，返回刻意非结合的两维状态。"""
        return states * 2 + (actions * valid[..., None]).sum((1, 2))[:, None]


class ToyActor(nn.Module):
    """确定性候选生成器，记录采样数以检查部署与 V target 一致。"""

    def __init__(self):
        """初始化采样计数列表。"""
        super().__init__()
        self.counts = []
        self.reward_model = object()
        self.leave_model = Mock(num_leave_compute=1, max_turn=6)
        self.dynamics = ToyDynamics()
        self.gamma = 0.9
        self._batch_user_ids = None

    def sample_candidate_chunks(self, states, count, mask):
        """返回 (N,B,K) 候选和占位 embedding，不使用任何 critic。"""
        self.counts.append(count)
        ids = torch.arange(count * CHUNK_SIZE, device=states.device)
        ids = (ids % CATALOG_SIZE).reshape(count, 1, CHUNK_SIZE)
        return ids.expand(-1, len(states), -1), None

    def _rollout_one_chunk(self, chunk_item_ids, history_state,
                           repeat_policy, leave_policy):
        """以纯 tensor 方式执行 toy chunk，模拟 MAC rollout 内核。"""

        del repeat_policy, leave_policy
        batch_size = len(chunk_item_ids)
        device = chunk_item_ids.device
        positions = torch.arange(CHUNK_SIZE, device=device)
        budget = (6 - history_state.env_step).clamp_min(0)
        valid = positions[None] < budget[:, None]
        effective = valid.long().sum(1)
        safe_items = chunk_item_ids.clamp_min(0)
        step_rewards = (safe_items.float() - 2.0) * valid
        discounts = torch.as_tensor(0.9, device=device).pow(positions)
        reward_chunks = (step_rewards * discounts[None]).sum(1)
        embeddings = torch.eye(CATALOG_SIZE, device=device)[safe_items]
        next_states = self.dynamics(history_state.states, embeddings, valid)
        next_steps = history_state.env_step + effective
        done = next_steps >= 6
        recommended = history_state.recommended_mask.clone()
        rows = torch.arange(batch_size, device=device)
        for position in range(CHUNK_SIZE):
            recommended[rows, safe_items[:, position]] |= valid[:, position]
        leave_history = history_state.leave_history.clone()
        for position in range(CHUNK_SIZE):
            appended = safe_items[:, position:position + 1]
            leave_history = torch.where(
                valid[:, position:position + 1], appended, leave_history,
            )
        result = RolloutResult(
            next_states=next_states,
            reward_chunks=reward_chunks,
            done_chunks=done,
            effective_steps=effective,
            selected_item_ids=torch.where(
                valid, chunk_item_ids, torch.full_like(chunk_item_ids, -1),
            ),
            step_rewards=step_rewards,
            chunk_valid=valid,
            pred_rewards=step_rewards,
            entropy_rewards=torch.zeros_like(step_rewards),
            uncertainty_penalties=torch.zeros_like(step_rewards),
            repeat_ratio=torch.zeros((), device=device),
            exact_repeat_ratio=torch.zeros((), device=device),
        )
        next_history = RolloutHistoryState(
            states=next_states,
            leave_history=leave_history,
            recommended_mask=recommended,
            env_step=next_steps,
            terminated=done,
        )
        return result, next_history


def fixture():
    """固定 CPU 随机数，构建无外部数据依赖的小型训练系统。"""
    torch.manual_seed(TEST_SEED)
    torch.set_num_threads(1)
    embeddings = torch.eye(CATALOG_SIZE)
    categories = torch.tensor([[1], [2], [1], [2], [3], [0]])
    hits = torch.tensor([1, 0, 1, 0, 0, 0])
    manager = StateManager(ToyDynamics(), embeddings, hits, max_turn=6)
    network = PrefixQV(2, embeddings, categories, hits, CHUNK_SIZE, 6,
                       history_window=2, hidden_dim=8, layers=1, heads=2, dropout=0)
    policy = CandidatePolicy(ToyActor(), network, Utility((6, 6, 1, 6)), num_candidates=2)
    return manager, policy


def raw_feedback(state, item):
    """返回可为负的训练偏好，方便检查原始反馈未截断。"""
    return float(item - 2)


def no_termination(state, item):
    """非预算终止回调恒为 False，用于完整会话测试。"""
    return False



class RAVETest(unittest.TestCase):
    """无需新增依赖的 RAVE-H 语义与集成回归测试。"""

    def setUp(self):
        """构建独立模型与临时文件目录，测试结束自动清理。"""
        self.fixture = fixture()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)


    def test_mcd_telescoping_and_boundary_invariance(self):
        """手算 (1,0,1) 的 D，分段只改变 C，跨全局锚点仍保持相同状态。"""
        fixture = self.fixture
        manager, _ = fixture
        first = manager.reset(0, torch.ones(2))
        second = first.clone()
        signals = []
        for step, item in enumerate([0, 1, 2, 3, 4, 0]):
            one = manager.advance(first, item, -1.0, step % 3 == 0)
            two = manager.advance(second, item, -1.0, step % 2 == 0)
            torch.testing.assert_close(one[:3], two[:3])
            torch.testing.assert_close(first.state, second.state)
            assert first.pending == second.pending
            signals.append(one)
        rewards = torch.stack(signals)
        torch.testing.assert_close(rewards[:3, 2], torch.tensor([0., 0.5, -1 / 6]))
        self.assertAlmostEqual(rewards[:, 2].sum().item(), first.fraction, places=6)
        self.assertAlmostEqual(rewards[2:, 2].sum().item(), first.fraction - 0.5, places=6)
        assert rewards[:, 3].sum() == 2
        assert first.done and first.budget == 0
        with self.assertRaises(ValueError):
            manager.advance(first, 0, 0, True)

    def test_batch_rollout_matches_scalar_and_batches_model_calls(self):
        """batch rollout 数值等价于逐轨迹参考路径且模型调用次数只随 K 增长。"""

        scalar_manager, _ = fixture()
        batch_manager, _ = fixture()
        candidates = torch.tensor([[0, 1, 2], [3, 4, 5]])
        scalar_starts = [
            scalar_manager.reset(0, torch.ones(2)),
            scalar_manager.reset(0, torch.ones(2)),
        ]
        batch_starts = [
            batch_manager.reset(0, torch.ones(2)),
            batch_manager.reset(0, torch.ones(2)),
        ]

        def terminate_on_item_zero(state, item):
            """仅 item 0 触发自然结束，用于覆盖 batch 内不同有效长度。"""

            del state
            return item == 0

        scalar_simulator = TrainingSimulator(
            scalar_manager, raw_feedback, terminate_on_item_zero,
        )
        expected = [
            scalar_simulator.rollout(state, candidates[row], {"row": row})
            for row, state in enumerate(scalar_starts)
        ]

        def batch_reward(states, item_ids, executable):
            """向量化复刻 raw_feedback，并屏蔽不可执行行。"""

            del states
            values = item_ids.float() - 2.0
            return torch.where(executable, values, torch.zeros_like(values))

        def batch_termination(states, item_ids, executable):
            """向量化复刻 terminate_on_item_zero。"""

            del states
            return (item_ids == 0) & executable

        batch_reward_mock = Mock(side_effect=batch_reward)
        batch_termination_mock = Mock(side_effect=batch_termination)
        batch_simulator = TrainingSimulator(
            batch_manager,
            Mock(side_effect=AssertionError("scalar reward path was used")),
            Mock(side_effect=AssertionError("scalar leave path was used")),
            batch_reward_callback=batch_reward_mock,
            batch_natural_termination=batch_termination_mock,
        )
        with patch.object(
            batch_manager.dynamics,
            "forward",
            wraps=batch_manager.dynamics.forward,
        ) as dynamics_forward:
            actual = batch_simulator.rollout_batch(
                batch_starts,
                candidates,
                [{"row": 0}, {"row": 1}],
            )

        self.assertEqual(batch_reward_mock.call_count, CHUNK_SIZE)
        self.assertEqual(batch_termination_mock.call_count, CHUNK_SIZE)
        self.assertEqual(dynamics_forward.call_count, CHUNK_SIZE)
        self.assertEqual(
            [len(record.endpoints) for record in actual],
            [len(record.endpoints) for record in expected],
        )
        for actual_record, expected_record in zip(actual, expected):
            torch.testing.assert_close(actual_record.rewards, expected_record.rewards)
            self.assertEqual(actual_record.metadata, expected_record.metadata)
            for actual_state, expected_state in zip(
                    actual_record.endpoints, expected_record.endpoints):
                torch.testing.assert_close(actual_state.state, expected_state.state)
                torch.testing.assert_close(actual_state.anchor, expected_state.anchor)
                self.assertEqual(actual_state.items, expected_state.items)
                self.assertEqual(actual_state.feedback, expected_state.feedback)
                self.assertEqual(actual_state.pending, expected_state.pending)
                self.assertEqual(actual_state.done, expected_state.done)

    def test_causality_padding_and_value(self):
        """相同前缀不同后缀具有相同 Q，V 与候选无关，历史顺序与计数仍可见。"""
        fixture = self.fixture
        for backbone in ['transformer', 'mlp']:
            manager, policy = fixture
            model = policy.network
            if backbone == 'mlp':
                model = PrefixQV(2, torch.eye(6), model.category_ids, model.dominant_hits,
                                 3, 6, 2, 8, 1, 2, 0, backbone=backbone)
            model.eval()
            state = manager.reset(0, torch.zeros(2))
            for item in [0, 1, 2]:
                manager.advance(state, item, float(item), True)
            inputs = policy.inputs([state])
            assert inputs['counts'].tolist() == [[3, 2]]
            assert inputs['history'].tolist() == [[1, 2]]
            candidates = torch.tensor([[[0, 1, 2], [0, 1, 4], [0, 1, -1]]])
            values = model(inputs, candidates)
            torch.testing.assert_close(values[:, 0, :2], values[:, 1, :2])
            torch.testing.assert_close(values[:, 0, :2], values[:, 2, :2])
            token = model.state_token(inputs) + model.position.weight[0]
            actions = model.item_tokens(candidates[:, 0]) + model.position.weight[1:]
            encoded = model._encode(torch.cat((token[:, None], actions), dim=1))
            torch.testing.assert_close(model.value(inputs), model.v_heads(encoded[:, 0]))
            with self.assertRaises(ValueError):
                model(inputs, torch.tensor([[[0, -1, 2]]]))

    def test_selection_budget_padding_and_ties(self):
        """同分优先候选再小 H，超预算与 padding 前缀永不可选。"""
        ids = torch.tensor([[[0, 1, -1], [2, 3, 4]]])
        values = torch.zeros(1, 2, 3, 4)
        utility = Utility((1, 1, 1, 1))
        selected, horizon, _ = select_prefix(values, ids, torch.tensor([2]), utility)
        assert (selected.item(), horizon.item()) == (0, 1)
        values[0, 0, 2, 0] = 1000
        values[0, 1, 2, 0] = 1000
        values[0, 1, 1, 0] = 2
        selected, horizon, _ = select_prefix(values, ids, torch.tensor([2]), utility)
        assert (selected.item(), horizon.item()) == (1, 2)
        _, horizon, _ = select_prefix(values, ids, torch.tensor([1]), utility, fixed_horizon=3)
        assert horizon.item() == 1
        with self.assertRaises(ValueError):
            select_prefix(values, ids, torch.tensor([0]), utility)

    def test_td_terminal_padding_and_stop_gradient(self):
        """终止步保留即时反馈，终止与 padding 的 NaN 续接被排除，标签不传梯度。"""
        rewards = torch.tensor([[[2., 1, 0, 1], [-1, 1, .5, 0], [float('nan')] * 4]])
        valid = torch.tensor([[True, True, False]])
        terminal = torch.tensor([[False, True, False]])
        continuation = torch.tensor([[[3., 2, -.2, 1], [float('nan')] * 4,
                                      [float('nan')] * 4]], requires_grad=True)
        targets = prefix_targets(rewards, valid, terminal, continuation, gamma=1.0)
        torch.testing.assert_close(targets, torch.tensor([[[5., 3, -.2, 2], [1., 2, .5, 1], [0.] * 4]]))
        assert not targets.requires_grad
        with self.assertRaises(ValueError):
            prefix_targets(
                rewards,
                torch.ones_like(valid),
                terminal,
                continuation,
                gamma=1.0,
            )
        discounted = prefix_targets(
            rewards,
            valid,
            terminal,
            continuation,
            gamma=0.5,
        )
        torch.testing.assert_close(
            discounted,
            torch.tensor([[[3.5, 2., -.1, 1.5], [1.5, 1.5, .25, 1.], [0.] * 4]]),
        )
        with self.assertRaises(ValueError):
            prefix_targets(rewards, valid, terminal, continuation, gamma=1.1)

    def test_ordinary_chunk_target_matches_final_prefix_target(self):
        """ordinary 直接 chunk target 与旧全前缀实现的最后有效标签完全一致。"""

        rewards = torch.tensor([
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [float("nan")]],
        ])
        valid = torch.tensor([
            [True, True, True],
            [True, True, False],
        ])
        terminal = torch.tensor([
            [False, False, False],
            [False, True, False],
        ])
        final_values = torch.tensor([[7.0], [float("nan")]], requires_grad=True)
        prefix_values = torch.zeros_like(rewards)
        prefix_values[0, 2] = final_values[0]
        prefix_values[1, 1] = final_values[1]
        old_targets = prefix_targets(
            rewards, valid, terminal, prefix_values, gamma=0.5,
        )
        direct_targets = chunk_targets(
            rewards, valid, terminal, final_values, gamma=0.5,
        )
        final_positions = valid.long().sum(1) - 1
        rows = torch.arange(len(rewards))
        torch.testing.assert_close(
            direct_targets, old_targets[rows, final_positions],
        )
        self.assertFalse(direct_targets.requires_grad)

    def test_ordinary_chunk_value_matches_gathered_prefix_q(self):
        """MLP 末端专用前向与旧实现按实际长度提取的前缀 Q 数值一致。"""

        manager, base_policy = self.fixture
        base_network = base_policy.network
        network = PrefixQV(
            2,
            torch.eye(CATALOG_SIZE),
            base_network.category_ids,
            base_network.dominant_hits,
            CHUNK_SIZE,
            6,
            2,
            8,
            1,
            2,
            0,
            scalar=True,
            backbone="mlp",
        ).eval()
        first = manager.reset(0, torch.zeros(2))
        second = manager.reset(1, torch.ones(2))
        inputs = base_policy.inputs([first, second])
        candidates = torch.tensor([[0, 1, 2], [3, 4, 5]])
        lengths = torch.tensor([3, 1])
        prefix_q = network(inputs, candidates[:, None])[:, 0, :, 0]
        direct_q = network.chunk_value(inputs, candidates, lengths)[:, 0]
        rows = torch.arange(len(candidates))
        torch.testing.assert_close(direct_q, prefix_q[rows, lengths - 1])

    def test_rollout_truncation_is_not_termination(self):
        """收集截断允许 V bootstrap，自然终止后不再调用偏好回调。"""
        fixture = self.fixture
        manager, _ = fixture
        state = manager.reset(0, torch.zeros(2))
        simulator = TrainingSimulator(manager, raw_feedback, no_termination)
        record = simulator.rollout(state, torch.tensor([0, 1, 2]), {}, max_steps=1)
        assert len(record.endpoints) == 1 and not record.endpoints[-1].done
        assert not state.items
        with patch.object(simulator, 'natural_termination', return_value=True):
            record = simulator.rollout(state, torch.tensor([0, 1, 2]), {})
        assert len(record.endpoints) == 1 and record.endpoints[0].done
        assert record.rewards[0, 1].item() == 1

    def test_acsac_bootstrap_uses_online_critic_stop_gradient(self):
        """ACSAC 目标来自在线 Q 联合选中的前缀，停止梯度且终止行不采样。"""
        fixture = self.fixture
        manager, policy = fixture
        trainer = RAVETrainer(policy)
        live = manager.reset(0, torch.zeros(2))
        terminal = live.clone()
        terminal.natural_done = True
        target = torch.zeros(1, 2, 3, 4)
        target[0, 1, 1] = torch.tensor([10., 2., .5, 0.])
        target[0, 0, 0] = torch.tensor([1., 1., 0., 0.])
        self.assertTrue(trainer.online.training)
        with patch.object(trainer.online, 'forward', return_value=target) as online:
            values = trainer.critic_targets([live, terminal])
        torch.testing.assert_close(values[0], target[0, 1, 1])
        torch.testing.assert_close(values[1], torch.zeros(4))
        self.assertFalse(values.requires_grad)
        self.assertTrue(trainer.online.training)
        online.assert_called_once()
        assert policy.mac.counts == [policy.num_candidates]

    def test_joint_training_and_complete_target_ema(self):
        """ACSAC 四头与仅 R 单头均更新全前缀，冻结 MAC 并保持 EMA 副本。"""
        fixture = self.fixture
        for scalar in [False, True]:
            manager, policy = fixture
            if scalar:
                policy.network = PrefixQV(2, torch.eye(6), policy.network.category_ids,
                                          policy.network.dominant_hits, 3, 6, 2, 8, 1, 2, 0, scalar=True)
            simulator = TrainingSimulator(manager, raw_feedback, no_termination)
            state = manager.reset(0, torch.ones(2))
            record = simulator.rollout(state, torch.tensor([0, 1, 2]), {})
            trainer = RAVETrainer(policy, tau=.5)
            before = deepcopy(trainer.target.state_dict())
            # 检查来自离线多块轨迹的旧成本在构造前缀时被重新记账。
            record.rewards[:, 3] = 9
            captured = []
            original = prefix_targets
            def capture(*args, **kwargs):
                """记录传入 TD 的即时信号并委托原实现，返回目标张量。"""
                captured.append(args[0].clone())
                return original(*args, **kwargs)
            with patch('examples.our_model.rave.training.prefix_targets', side_effect=capture):
                metrics = trainer.update([record])
            assert np.isfinite(list(metrics.values())).all()
            assert metrics['valid_prefixes'] == 3
            if scalar:
                torch.testing.assert_close(
                    captured[0][0, :, 0], torch.tensor([-2., -1., 0.]),
                )
                self.assertIn('td_mae/h1/R', metrics)
                self.assertNotIn('td_mae/h1/utility', metrics)
            else:
                torch.testing.assert_close(captured[0][0, :, 3], torch.tensor([1., 0, 0]))
            changed = False
            for name, parameter in policy.network.named_parameters():
                expected = before[name].lerp(parameter.detach(), .5)
                torch.testing.assert_close(trainer.target.state_dict()[name], expected)
                changed |= not torch.equal(before[name], parameter)
            assert changed
            assert all(parameter.grad is None for parameter in trainer.target.parameters())
            assert all(not parameter.requires_grad for parameter in manager.dynamics.parameters())

    def test_ordinary_optimizes_only_trajectory_start(self):
        """ordinary 只用 imagined trajectory 的首个 chunk 监督起点 Q/V。"""

        manager, base_policy = self.fixture
        base_network = base_policy.network
        network = PrefixQV(
            2,
            torch.eye(6),
            base_network.category_ids,
            base_network.dominant_hits,
            3,
            6,
            2,
            8,
            1,
            2,
            0,
            scalar=True,
            backbone="mlp",
        )
        policy = CandidatePolicy(
            base_policy.mac,
            network,
            base_policy.utility,
            num_candidates=base_policy.num_candidates,
            fixed_horizon=3,
        )
        simulator = TrainingSimulator(manager, raw_feedback, no_termination)
        episodes, tail_states = collect_imagined_rollouts(
            policy=policy,
            simulator=simulator,
            manager=manager,
            user_ids=[0],
            initial_states=torch.ones(1, 2),
            metadata={},
            epsilon=0.0,
            num_candidates=2,
            rollout_depth=1,
        )
        self.assertFalse(tail_states[0].done)
        self.assertEqual(len(episodes[0]), 1)
        trainer = RAVETrainer(
            policy,
            optimize_all_prefixes=False,
            gamma=0.9,
        )
        with patch.object(
            trainer.target,
            "value",
            return_value=torch.tensor([[10.0]]),
        ) as target_value:
            records = trainer.assign_imagined_value_targets(episodes)
        expected_v = sum(
            (0.9 ** index) * reward[0]
            for index, reward in enumerate(records[0].rewards)
        ) + (0.9 ** 3) * 10.0
        torch.testing.assert_close(records[0].value_target, expected_v.reshape(1))
        self.assertEqual(target_value.call_count, 2)
        self.assertIsInstance(trainer.optimizer, torch.optim.Adam)
        with patch(
            "examples.our_model.rave.training.functional.mse_loss",
            wraps=torch.nn.functional.mse_loss,
        ) as mse_loss, patch(
            "examples.our_model.rave.training.functional.smooth_l1_loss",
            wraps=torch.nn.functional.smooth_l1_loss,
        ) as huber_loss, patch(
            "examples.our_model.rave.training.torch.nn.utils.clip_grad_norm_",
            wraps=torch.nn.utils.clip_grad_norm_,
        ) as clip_grad:
            metrics = trainer.update([records[0]])
        self.assertEqual(mse_loss.call_count, 2)
        huber_loss.assert_not_called()
        clip_grad.assert_not_called()
        self.assertEqual(metrics["observed_prefixes"], 3)
        self.assertEqual(metrics["valid_prefixes"], 1)
        self.assertEqual(metrics["bootstrap_states"], 1)
        self.assertNotIn("td_mae/h1/R", metrics)
        self.assertNotIn("td_mae/h2/R", metrics)
        self.assertIn("td_mae/h3/R", metrics)
        self.assertIn("v_mae/R", metrics)

    def test_ordinary_middle_chunk_only_builds_mac_td_lambda_target(self):
        """中间 chunk 影响 MAC TD(lambda) 标签，但不成为独立监督样本。"""

        manager, base_policy = self.fixture
        base_network = base_policy.network
        network = PrefixQV(
            2, torch.eye(CATALOG_SIZE), base_network.category_ids,
            base_network.dominant_hits, CHUNK_SIZE, 6, 2, 8, 1, 2, 0,
            scalar=True, backbone="mlp",
        )
        policy = CandidatePolicy(
            base_policy.mac, network, base_policy.utility,
            num_candidates=2, fixed_horizon=CHUNK_SIZE,
        )
        episodes, _ = collect_imagined_rollouts(
            policy=policy,
            simulator=TrainingSimulator(manager, raw_feedback, no_termination),
            manager=manager,
            user_ids=[0],
            initial_states=torch.ones(1, 2),
            metadata={},
            epsilon=0.0,
            num_candidates=2,
            rollout_depth=2,
        )
        trainer = RAVETrainer(
            policy,
            optimize_all_prefixes=False,
            gamma=0.9,
            lambda_chunk=0.5,
        )
        def constant_target_value(inputs):
            """按输入 batch 大小返回常数 target V，便于手算 TD(lambda)。"""

            return torch.full((len(inputs["states"]), 1), 10.0)

        with patch.object(
            trainer.target,
            "value",
            side_effect=constant_target_value,
        ):
            records = trainer.assign_imagined_value_targets(episodes)
        first_chunk, second_chunk = episodes[0]
        first_return = sum(
            (0.9 ** index) * reward[0]
            for index, reward in enumerate(first_chunk.rewards)
        )
        second_return = sum(
            (0.9 ** index) * reward[0]
            for index, reward in enumerate(second_chunk.rewards)
        )
        second_continuation = float(not second_chunk.endpoints[-1].done)
        first_continuation = float(not first_chunk.endpoints[-1].done)
        second_delta = (
            second_return + second_continuation * (0.9 ** 3) * 10.0 - 10.0
        )
        first_delta = (
            first_return + first_continuation * (0.9 ** 3) * 10.0 - 10.0
        )
        expected = (
            10.0 + first_delta
            + 0.5 * (0.9 ** 3) * first_continuation * second_delta
        )
        self.assertEqual(records, [first_chunk])
        torch.testing.assert_close(
            first_chunk.value_target,
            torch.as_tensor([expected]),
        )
        self.assertIsNone(second_chunk.value_target)

    def test_pure_tensor_ordinary_matches_mac_return_and_one_dynamics_per_chunk(self):
        """纯 tensor ordinary 不建 endpoint，且 lambda=1 回报与 MAC 展开一致。"""

        _, base_policy = self.fixture
        base_network = base_policy.network
        network = PrefixQV(
            2, torch.eye(CATALOG_SIZE), base_network.category_ids,
            base_network.dominant_hits, CHUNK_SIZE, 6, 2, 8, 1, 2, 0,
            scalar=True, backbone="mlp",
        )
        policy = CandidatePolicy(
            base_policy.mac, network, base_policy.utility,
            num_candidates=2, fixed_horizon=CHUNK_SIZE,
        )
        with patch.object(
            policy.mac.dynamics,
            "forward",
            wraps=policy.mac.dynamics.forward,
        ) as dynamics_forward:
            trajectory = collect_tensor_imagined_rollouts(
                policy=policy,
                user_ids=[0, 0],
                initial_states=torch.ones(1, 2),
                raw_users=[10],
                dominant_hits=network.dominant_hits,
                max_turn=6,
                epsilon=0.0,
                num_candidates=2,
                rollout_depth=2,
                gamma=0.9,
            )
        self.assertEqual(dynamics_forward.call_count, 2)
        self.assertEqual(len(trajectory.steps), 2)
        self.assertEqual(trajectory.batch_size, 2)
        self.assertEqual(trajectory.total_chunks, 4)
        self.assertEqual(trajectory.total_steps, 12)
        self.assertEqual(trajectory.completed_episodes, 2)
        trainer = RAVETrainer(
            policy,
            optimize_all_prefixes=False,
            gamma=0.9,
            lambda_chunk=1.0,
        )
        value_target = trainer.assign_tensor_value_target(trajectory)
        expected = (
            trajectory.steps[0].reward_chunks
            + trajectory.steps[0].discounts
            * trajectory.steps[1].reward_chunks
        ).unsqueeze(-1)
        torch.testing.assert_close(value_target, expected)
        metrics = trainer.update_tensor_trajectory(trajectory)
        self.assertEqual(metrics["valid_prefixes"], 2)
        self.assertEqual(metrics["observed_prefixes"], 6)
        self.assertTrue(np.isfinite(list(metrics.values())).all())

    def test_complete_episode_collection_keeps_natural_tail(self):
        """Transformer 完整 episode 收集的自然退出尾 chunk 只保留有效前缀。"""

        embeddings = torch.eye(CATALOG_SIZE)
        categories = torch.tensor([[1], [2], [1], [2], [3], [0]])
        hits = torch.tensor([1, 0, 1, 0, 0, 0])
        manager = StateManager(ToyDynamics(), embeddings, hits, max_turn=6)
        network = PrefixQV(
            2, embeddings, categories, hits, CHUNK_SIZE, 6,
            history_window=2, hidden_dim=8, layers=1, heads=2, dropout=0,
        )
        policy = CandidatePolicy(
            ToyActor(), network, Utility((6, 6, 1, 6)), num_candidates=4,
        )

        def terminate_in_second_chunk(state, item):
            """在累计执行第四步时自然退出，形成长度一的尾 chunk。"""
            del item
            return len(state.items) == CHUNK_SIZE

        simulator = TrainingSimulator(manager, raw_feedback, terminate_in_second_chunk)
        records, terminal_states = collect_complete_episodes(
            policy=policy,
            simulator=simulator,
            manager=manager,
            user_ids=[0],
            initial_states=torch.zeros(1, 2),
            metadata={"source": "unit_test"},
            epsilon=0.0,
            num_candidates=2,
            gamma=0.5,
        )
        self.assertEqual([len(record.endpoints) for record in records], [3, 1])
        self.assertEqual([record.metadata["behavior_horizon"] for record in records], [3, 3])
        self.assertTrue(terminal_states[0].natural_done)
        self.assertEqual(len(terminal_states[0].items), 4)
        expected_first = (
            records[0].rewards[0]
            + 0.5 * records[0].rewards[1]
            + 0.25 * records[0].rewards[2]
            + 0.125 * records[1].rewards[0]
        )
        torch.testing.assert_close(records[0].return_to_go, expected_first)
        torch.testing.assert_close(records[1].return_to_go, records[1].rewards[0])
        self.assertEqual(policy.mac.counts, [2, 2])

    def test_assign_episode_returns_requires_terminal_episode(self):
        """MAC 完整轨迹回报拒绝未自然退出且未耗尽预算的截断数据。"""

        manager, _ = self.fixture
        simulator = TrainingSimulator(manager, raw_feedback, no_termination)
        record = simulator.rollout(
            manager.reset(0, torch.zeros(2)),
            torch.tensor([0, 1, 2]),
            {},
        )
        with self.assertRaises(ValueError):
            assign_episode_returns([record], gamma=0.9)

    def test_imagined_rollout_depth_truncates_without_replay_semantics(self):
        """fresh collector 严格限制 chunk 深度，截断尾状态保持可 bootstrap。"""

        manager, policy = self.fixture
        policy.fixed_horizon = CHUNK_SIZE
        simulator = TrainingSimulator(manager, raw_feedback, no_termination)
        with patch.object(
            simulator, "rollout_batch", wraps=simulator.rollout_batch,
        ) as rollout_batch, patch.object(
            simulator,
            "rollout",
            side_effect=AssertionError("imagined collector used scalar rollout"),
        ):
            episodes, tail_states = collect_imagined_rollouts(
                policy=policy,
                simulator=simulator,
                manager=manager,
                user_ids=[0, 0],
                initial_states=torch.zeros(1, 2),
                metadata={"source": "unit_test"},
                epsilon=0.0,
                num_candidates=2,
                rollout_depth=1,
            )
        rollout_batch.assert_called_once()
        self.assertEqual([len(episode) for episode in episodes], [1, 1])
        self.assertTrue(all(not state.done for state in tail_states))
        self.assertEqual(
            [episode[0].metadata["rollout_depth"] for episode in episodes],
            [1, 1],
        )
        self.assertEqual(policy.mac.counts, [2])
        with self.assertRaises(ValueError):
            collect_imagined_rollouts(
                policy, simulator, manager, [0], torch.zeros(1, 2), {},
                epsilon=0.0, num_candidates=2, rollout_depth=0,
            )

    def test_ordinary_train_collects_fresh_rollout_every_update(self):
        """ordinary runner 每个 optimizer step 都新收集，并始终报告 replay 为零。"""

        manager, base_policy = self.fixture
        base_network = base_policy.network
        network = PrefixQV(
            2, torch.eye(CATALOG_SIZE), base_network.category_ids,
            base_network.dominant_hits, CHUNK_SIZE, 6, 2, 8, 1, 2, 0,
            scalar=True, backbone="mlp",
        )
        policy = CandidatePolicy(
            base_policy.mac, network, base_policy.utility,
            num_candidates=2, fixed_horizon=CHUNK_SIZE,
        )
        simulator = TrainingSimulator(manager, raw_feedback, no_termination)
        args = Namespace(
            learning_rate=3e-4, tau=0.005, alpha_v=1.0,
            qv_type="ordinary", gamma=0.9, steps=2, epochs=1,
            chunk_size=CHUNK_SIZE, max_turn=6,
            replay_capacity=16, seed=TEST_SEED, episodes_per_collect=2,
            epsilon=0.0, rollout_candidates=2, ordinary_rollout_depth=1,
            lambda_chunk=1.0,
            collect_every=99, batch_size=8, log_every=1,
            eval_every_epochs=1, mc_samples=0,
        )
        assets = {
            "policy": policy,
            "simulator": simulator,
            "manager": manager,
            "splits": {"train": [0]},
            "initial": torch.zeros(1, 2),
            "fingerprints": {"predicted_mat_path": "unit-test"},
            "utility": policy.utility,
            "mac": {},
            "categories": network.category_ids,
            "hits": network.dominant_hits,
            "category_metadata": {},
            "training_reward_rule": "dorl_shaped_reward_v1",
            "reward_config": {},
            "raw_users": [10],
        }
        tracker = Mock()
        evaluator = Mock()
        evaluator.evaluate.return_value = {}
        original_collector = collect_tensor_imagined_rollouts
        with patch(
            "examples.our_model.runners.run_rave.collect_tensor_imagined_rollouts",
            wraps=original_collector,
        ) as collector, patch("torch.save"):
            train(args, assets, self.tmp_path, tracker, evaluator)
        self.assertEqual(collector.call_count, 2)
        lines = (self.tmp_path / "training_metrics.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            metrics = json.loads(line)
            self.assertEqual(metrics["replay_size"], 0)
            self.assertGreater(metrics["fresh_batch_size"], 0)

    def test_independent_cache_reorder_and_early_exit(self):
        """不同环境 H 独立缓存，反馈不重规划，重排/提前结束不污染其他环境。"""
        fixture = self.fixture
        manager, policy = fixture
        adapter = RAVEPolicyAdapter(policy, manager)
        adapter.reset('a', 0, torch.zeros(2))
        adapter.reset('b', 1, torch.zeros(2))
        choice = (torch.tensor([[0, 1, 2], [3, 4, 5]]), torch.tensor([3, 2]), None)
        with patch.object(policy, 'choose', return_value=choice) as choose:
            assert adapter.act(['a', 'b']) == [0, 3]
            with self.assertRaises(ValueError):
                adapter.act(['a'])
            adapter.observe('a', -1)
            adapter.observe('b', 2, truncated=True)
            assert adapter.act(['b', 'a']) == [4, 1]
            adapter.observe('b', 1, natural_done=True)
            adapter.observe('a', 0)
            assert adapter.act(['a']) == [2]
            adapter.observe('a', 1)
            assert choose.call_count == 1
        assert adapter.cache['b'] == [] and adapter.sessions['b'].done
        assert adapter.sessions['a'].items == [0, 1, 2]
        assert adapter.plans == 2

    def test_dorl_collector_adapter_reuses_transformer_q_only_for_selection(self):
        """共享 DORL Collector 时保留真实反馈、分支 mask 和独立会话缓存。"""

        manager, policy = self.fixture
        adapter = RAVEDORLPolicyAdapter(
            candidate_policy=policy,
            manager=manager,
            initial_states=torch.zeros(2, 2),
            device=torch.device("cpu"),
        )
        first_choice = (
            torch.tensor([[0, 1, 2], [3, 4, 5]]),
            torch.tensor([3, 2]),
            torch.zeros(2, 4),
        )
        reset_choice = (
            torch.tensor([[5, 0, 1]]),
            torch.tensor([1]),
            torch.zeros(1, 4),
        )
        first_batch = Batch(
            obs=np.array([[0, CATALOG_SIZE], [1, CATALOG_SIZE]]),
            rew_prev=np.zeros(2),
            is_start=np.ones(2, dtype=bool),
        )
        continuation_batch = Batch(
            obs=np.array([[0, 0], [1, 3]]),
            rew_prev=np.array([-1.0, 2.0]),
            is_start=np.zeros(2, dtype=bool),
        )
        partial_reset_batch = Batch(
            obs=np.array([[0, 1], [1, CATALOG_SIZE]]),
            rew_prev=np.array([0.5, 0.0]),
            is_start=np.array([False, True]),
        )

        with patch.object(
            policy,
            "choose",
            side_effect=[first_choice, reset_choice],
        ) as choose:
            first = adapter(
                first_batch,
                buffer=None,
                state=None,
                remove_recommended_ids=False,
            )
            second = adapter(
                continuation_batch,
                buffer=None,
                state=first.state,
                remove_recommended_ids=False,
            )
            third = adapter(
                partial_reset_batch,
                buffer=None,
                state=second.state,
                remove_recommended_ids=True,
            )

        self.assertEqual(first.act.tolist(), [0, 3])
        self.assertEqual(second.act.tolist(), [1, 4])
        self.assertEqual(third.act.tolist(), [2, 5])
        self.assertEqual(choose.call_count, 2)
        self.assertFalse(choose.call_args_list[0].kwargs["mask_history"])
        self.assertTrue(choose.call_args_list[1].kwargs["mask_history"])
        first_session_id = int(first.state[0])
        self.assertEqual(adapter._sessions[first_session_id].items, [0, 1])
        self.assertEqual(adapter._sessions[first_session_id].feedback, [-1.0, 0.5])
        metadata = adapter.get_evaluation_metadata()
        self.assertEqual(metadata["evaluation/selected_horizon/h3"], 1)
        self.assertEqual(metadata["evaluation/selected_horizon/h2"], 1)
        self.assertEqual(metadata["evaluation/selected_horizon/h1"], 1)
        self.assertEqual(metadata["evaluation/selected_horizon_total"], 3)
        self.assertEqual(metadata["evaluation/selected_horizon_mean"], 2.0)
        self.assertAlmostEqual(
            metadata["evaluation/selected_horizon_fraction/h1"],
            1 / 3,
        )
        branch_metrics = adapter.get_open_loop_diagnostics()
        self.assertEqual(branch_metrics["selected_horizon_total"], 3)
        adapter.reset_open_loop_diagnostics()
        self.assertEqual(
            adapter.get_open_loop_diagnostics()["selected_horizon_total"],
            0,
        )
        self.assertEqual(
            adapter.get_evaluation_metadata()["evaluation/selected_horizon_total"],
            3,
        )

    def test_mcd_aggregation_and_mc_mid_session(self):
        """验证 macro/pooled 分离及中途 MC 标签以起点比例为基线。"""
        fixture = self.fixture
        manager, policy = fixture
        first = manager.reset(0, torch.zeros(2))
        manager.advance(first, 0, 1, True, natural_done=True)
        second = manager.reset(1, torch.zeros(2))
        for item in [1, 3, 4]:
            manager.advance(second, item, 1, True, natural_done=item == 4)
        metrics = episode_metrics([first, second], policy.network.category_ids.numpy(),
                                  4, [.01] * 4, [1] * 4, state_update_latencies=[.02] * 4)
        assert metrics['mcd_pooled'] == .25
        assert metrics['mcd_macro'] == .5
        self.assertAlmostEqual(metrics['inference_seconds'], .12, places=6)
        state = manager.reset(0, torch.zeros(2))
        manager.advance(state, 1, 1, True)
        simulator = TrainingSimulator(manager, raw_feedback, no_termination)
        record = simulator.rollout(state, torch.tensor([0, 1, 2]), {})
        calibration = monte_carlo_calibration(policy, simulator, [record], 'test_snapshot')
        for sample in calibration['samples']:
            self.assertAlmostEqual(sample['returns'][2], sample['end_fraction'] - 1, places=6)
            assert 1 <= sample['returns'][3] <= sample['returns'][1] <= state.budget
            assert sample['policy_version'] == 'test_snapshot'

    def test_fixed_scale_and_original_category_cutoff(self):
        """固定尺度由训练绝对反馈得出，类别沿用首次越界不纳入的原口径。"""
        utility = Utility.from_training_rewards(torch.tensor([-2., 2.]), 10)
        assert utility.scales == (20, 10, 1, 10)
        assert get_dominant_values([(1, .6), (2, .3), (3, .1)]).tolist() == [1]
        assert get_dominant_values([(1, .9), (2, .1)]).tolist() == [1]
        with self.assertRaises(ValueError):
            Utility((0, 1, 1, 1))

    def test_category_freezing_uses_training_users_only(self):
        """测试用户的高频类别不能进入冻结集合，多标签与缺失值对齐现有评估。"""
        from types import SimpleNamespace
        from src.core.envs.BaseData import BaseData

        items = pd.DataFrame({'feat0': [1, 2, 0], 'feat1': [3, 0, 0]}, index=[0, 1, 2])
        interactions = pd.DataFrame({
            'user_id': [10, 10, 20, 20, 20], 'item_id': [0, 0, 1, 1, 1],
            'watch_ratio_normed': [1., 1., 1., 1., 1.],
            'feat0': [1, 1, 2, 2, 2], 'feat1': [3, 3, 0, 0, 0],
        })
        dataset = SimpleNamespace(
            get_df=lambda source: (interactions, None, items, None),
            get_sorted_domination_features=BaseData().get_sorted_domination_features,
        )
        env = SimpleNamespace(lbe_item=SimpleNamespace(classes_=np.arange(3)))
        categories, hits, metadata = freeze_categories(dataset, env, 'KuaiEnv-v0', [10])
        assert metadata['dominant_categories'] == [1]
        assert metadata['training_interactions'] == 2
        assert hits.tolist() == [True, False, False]
        self.assertAlmostEqual(metadata['missing_category_catalog_rate'], 1 / 3, places=6)
        assert categories.tolist() == [[1, 3], [2, 0], [0, 0]]

    def test_raw_preference_bypasses_old_reward_shaping(self):
        """Transformer 的 raw preference 接口仍不混入 DORL reward shaping。"""
        tmp_path = self.tmp_path
        import pickle
        from examples.our_model.models.dorl_reward import DORLRewardModel

        predicted = tmp_path / 'predicted.pkl'
        variance = tmp_path / 'variance.pkl'
        predicted.write_bytes(pickle.dumps(np.array([[-2., 3.]], dtype=np.float32)))
        variance.write_bytes(pickle.dumps(np.array([[1., 1.]], dtype=np.float32)))
        reward = DORLRewardModel(str(predicted), {10: 0}, torch.device('cpu'),
                                 feature_level=False, entropy_map={(0,): 8.},
                                 maxvar_mat_path=str(variance))
        torch.testing.assert_close(reward.raw_preference(torch.tensor([10, 10]),
                                                        torch.tensor([0, 1])),
                                   torch.tensor([-2., 3.]))

    def test_ordinary_restores_and_calls_mac_dorl_reward(self):
        """ordinary 从 MAC checkpoint 恢复 reward 配置并传入含当前 item 的历史。"""

        args = Namespace()
        expected = {
            "use_entropy_reward": True,
            "use_uncertainty_penalty": True,
            "lambda_entropy": 0.5,
            "lambda_variance": 1.0,
            "predicted_mat_normalize": "per_user_max",
            "entropy_window": [1, 2],
            "feature_level": True,
            "is_sorted": True,
        }
        restored = apply_ordinary_dorl_reward_config(
            args, {"config": expected.copy()},
        )
        self.assertEqual(restored, expected)
        self.assertEqual(
            {name: getattr(args, name) for name in expected}, expected,
        )
        with self.assertRaises(ValueError):
            apply_ordinary_dorl_reward_config(
                Namespace(), {"config": {"use_entropy_reward": True}},
            )

        manager, _ = self.fixture
        state = manager.reset(0, torch.zeros(2))
        state.items.extend([1, 2])
        reward_model = Mock()
        reward_model.reward.return_value = torch.tensor([4.25])
        shaped = session_training_reward(
            reward_model, [10], state, 3, torch.device("cpu"), True,
        )
        self.assertEqual(shaped, 4.25)
        reward_model.reward.assert_called_once()
        reward_call = reward_model.reward.call_args
        torch.testing.assert_close(reward_call.args[0], torch.tensor([10]))
        torch.testing.assert_close(reward_call.args[1], torch.tensor([3]))
        torch.testing.assert_close(
            reward_call.kwargs["history_item_ids"], torch.tensor([[1, 2, 3]]),
        )
        reward_model.raw_preference.assert_not_called()

        second_state = manager.reset(0, torch.zeros(2))
        second_state.items.extend([4])
        reward_model.reset_mock()
        reward_model.entropy_window = [1, 2]
        reward_model.reward.return_value = torch.tensor([4.25, 5.5])
        batch_rewards = batch_session_training_reward(
            reward_model=reward_model,
            raw_users=[10],
            states=[state, second_state],
            item_ids=torch.tensor([3, 5]),
            executable=torch.tensor([True, True]),
            device=torch.device("cpu"),
            use_dorl_reward=True,
        )
        torch.testing.assert_close(batch_rewards, torch.tensor([4.25, 5.5]))
        reward_model.reward.assert_called_once()
        batch_call = reward_model.reward.call_args
        torch.testing.assert_close(batch_call.args[0], torch.tensor([10, 10]))
        torch.testing.assert_close(batch_call.args[1], torch.tensor([3, 5]))
        torch.testing.assert_close(
            batch_call.kwargs["history_item_ids"],
            torch.tensor([[2, 3], [4, 5]]),
        )

    def test_unified_qv_type_resolution(self):
        """统一类型正确映射 ordinary、单头及四头 Q/V 语义。"""

        ordinary = build_parser().parse_args(["--qv_type", "ordinary"])
        self.assertEqual(resolve_qv_type(ordinary), "ordinary")
        self.assertTrue(ordinary.scalar_q)
        self.assertEqual(ordinary.backbone, "mlp")
        self.assertEqual(ordinary.fixed_horizon, ordinary.chunk_size)
        self.assertEqual((ordinary.eta_t, ordinary.eta_d, ordinary.eta_c), (0., 0., 0.))

        scalar = build_parser().parse_args(["--qv_type", "transformer"])
        self.assertEqual(resolve_qv_type(scalar), "transformer")
        self.assertTrue(scalar.scalar_q)
        self.assertEqual(scalar.backbone, "transformer")
        self.assertEqual((scalar.eta_t, scalar.eta_d, scalar.eta_c), (0., 0., 0.))

        multi = build_parser().parse_args(["--qv_type", "multi_prefix_transformer"])
        self.assertEqual(resolve_qv_type(multi), "multi_prefix_transformer")
        self.assertFalse(multi.scalar_q)
        self.assertEqual((multi.eta_t, multi.eta_d, multi.eta_c), (.2, .2, .05))

        weighted = build_parser().parse_args([
            "--qv_type", "multi_prefix_transformer",
            "--eta_t", "0.1", "--eta_d", "0.3", "--eta_c", "0.02",
        ])
        self.assertEqual(resolve_qv_type(weighted), "multi_prefix_transformer")
        self.assertEqual(
            (weighted.eta_t, weighted.eta_d, weighted.eta_c), (.1, .3, .02),
        )

        legacy = build_parser().parse_args(["--scalar_q"])
        self.assertEqual(resolve_qv_type(legacy), "transformer")
        with self.assertRaises(ValueError):
            invalid_weights = build_parser().parse_args([
                "--qv_type", "transformer", "--eta_t", "0.2",
            ])
            resolve_qv_type(invalid_weights)
        with self.assertRaises(ValueError):
            conflict = build_parser().parse_args([
                "--qv_type", "multi_prefix_transformer", "--scalar_q",
            ])
            resolve_qv_type(conflict)
        with self.assertRaises(ValueError):
            invalid_ordinary = build_parser().parse_args([
                "--qv_type", "ordinary", "--chunk_size", "5",
                "--fixed_horizon", "2",
            ])
            resolve_qv_type(invalid_ordinary)

    def test_epoch_schedule_matches_dorl_evaluation_axis(self):
        """RAVE 总更新均匀映射到 100 个 epoch，非法非整除配置及时失败。"""
        self.assertEqual(resolve_epoch_schedule(10000, 100), 100)
        self.assertEqual(build_parser().parse_args([]).epochs, 100)
        self.assertEqual(build_parser().parse_args([]).eval_every_epochs, 1)
        self.assertEqual(build_parser().parse_args([]).ordinary_rollout_depth, 3)
        with self.assertRaises(ValueError):
            resolve_epoch_schedule(10001, 100)
        with self.assertRaises(ValueError):
            resolve_epoch_schedule(10000, 0)

    def test_collection_seed_is_independent_from_model_rng_consumption(self):
        """同 seed/step 的收集随机流不应受此前网络随机数消耗影响。"""

        set_collection_seed(2023, 11)
        expected = torch.rand(8)
        _ = torch.rand(137)
        set_collection_seed(2023, 11)
        actual = torch.rand(8)
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            set_collection_seed(2023, 0)

    def test_live_eval_logger_streams_epoch_metrics(self):
        """独立日志进程按 epoch 实时消费指标，并在结束信号后关闭 tracker。"""
        metric_queue = Mock()
        metric_queue.get.side_effect = [
            {"epoch": 1, "metrics": {"NX_0_rew": 1.5}},
            {"epoch": 2, "metrics": {"NX_0_rew": 2.5}},
            None,
        ]
        with patch(
            "examples.our_model.logging.live_eval_logger.SwanLabLogger"
        ) as logger_class:
            run_live_eval_logger(
                metric_queue=metric_queue,
                project="RAVE-H-Eval",
                run_name="seed-2023-eval",
                config={"seed": 2023},
                log_path="metrics.jsonl",
            )
        logger = logger_class.return_value
        logger.log.assert_has_calls([
            call({"NX_0_rew": 1.5}, step=1),
            call({"NX_0_rew": 2.5}, step=2),
        ])
        logger.finish.assert_called_once_with()

    def test_saved_model_directory_and_swanlab_metric_projection(self):
        """默认训练目录位于 saved_models，SwanLab 只接收扁平数值指标。"""
        args = Namespace(
            output_dir="", output_root="", mode="train", save_root="saved_models",
            env="KuaiEnv-v0", run_name="", chunk_size=5, seed=2023,
            qv_type="multi_prefix_transformer", scalar_q=False,
            rave_checkpoint="", eval_split="validation",
        )
        self.assertEqual(
            resolve_output_dir(args, "20260913_000000"),
            Path("saved_models/KuaiEnv-v0/QV_suite/K5/"
                 "multi_prefix_transformer/seed-2023"),
        )
        args.output_root = "saved_models/custom-suite/K5"
        self.assertEqual(
            resolve_output_dir(args, "20260913_000000"),
            Path("saved_models/custom-suite/K5/"
                 "multi_prefix_transformer/seed-2023"),
        )
        validation = validation_log_metrics({
            "mcd_macro": np.float64(0.4), "mcd_pooled": np.float64(0.3),
            "raw_return_mean": np.float64(8.0),
            "raw_feedback_per_display": np.float64(0.8),
            "length_mean": np.float64(10.0), "episodes": 2,
            "selected_horizons": {1: 3, 2: 4},
            "episodes_detail": [{"items": [1, 2]}], "missing": None,
        })
        self.assertEqual(validation["validation/mcd_macro"], 0.4)
        self.assertEqual(validation["validation/selected_horizon/h2"], 4)
        self.assertEqual(validation["NX_0_rew"], 8.0)
        self.assertEqual(validation["NX_0_len"], 10.0)
        self.assertEqual(validation["NX_0_ctr"], 0.8)
        self.assertEqual(validation["NX_0_ifeat_feat"], 0.3)
        self.assertEqual(validation["raw/NX_0_rew"], 8.0)
        self.assertNotIn("validation/episodes_detail", validation)
        test_metrics = evaluation_log_metrics({
            "mcd_pooled": 0.3, "raw_return_mean": 8.0,
            "raw_feedback_per_display": 0.8, "length_mean": 10.0,
            "episodes": 2, "selected_horizons": {1: 2},
        }, "test")
        self.assertEqual(test_metrics["test/episodes"], 2)
        self.assertEqual(test_metrics["test/selected_horizon/h1"], 2)
        self.assertEqual(test_metrics["NX_0_rew"], 8.0)
        with self.assertRaises(ValueError):
            evaluation_log_metrics({}, "train")
        eval_args = Namespace(
            run_name="", env="KuaiEnv-v0", chunk_size=5,
            qv_type="transformer", eval_split="test", seed=2023,
        )
        self.assertEqual(
            default_eval_run_name(eval_args),
            "rave-h-eval-KuaiEnv-v0-K5-transformer-test-seed2023",
        )
        calibration = calibration_log_metrics({"errors": {"mc_mae/h1/R": 0.2}})
        self.assertEqual(calibration, {"calibration/mc_mae/h1/R": 0.2})


class RAVEShellRoutingTest(unittest.TestCase):
    """验证统一脚本保持前缀 Q 的自适应 H 与类型路由语义。"""

    @classmethod
    def setUpClass(cls) -> None:
        """定位仓库根目录，供子进程调用真实 shell 入口。"""

        cls.repo_root = Path(__file__).resolve().parents[3]

    def test_rave_wrapper_defaults_to_adaptive_horizon(self) -> None:
        """未显式设置固定 H 时，RAVE wrapper 不得传入 `--fixed_horizon`。"""

        environment = os.environ.copy()
        environment.update({
            "PYTHON_BIN": "/bin/echo",
            "MODE": "train",
            "MAC_CHECKPOINT": "/tmp/shared-flow-bc.pt",
            "QV_TYPE": "transformer",
            "CHUNK_SIZE": "5",
            "DEVICE": "cpu",
        })
        result = subprocess.run(
            ["bash", "script/run_rave_kuai.sh"],
            cwd=self.repo_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--qv_type transformer", result.stdout)
        self.assertNotIn("--fixed_horizon", result.stdout)

        environment["FIXED_HORIZON"] = "3"
        fixed_result = subprocess.run(
            ["bash", "script/run_rave_kuai.sh"],
            cwd=self.repo_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--fixed_horizon 3", fixed_result.stdout)

    def test_suite_routes_ordinary_to_mac_and_transformers_to_rave(self) -> None:
        """ordinary 应走冻结 Dynamics 的 MAC，Transformer 保持 RAVE。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = subprocess.run(
                [
                    "bash", "script/run_kuai_qv_suite.sh",
                    "--stage", "qv",
                    "--qv-type", "all",
                    "--seeds", "2023 2024 2025 2026 2027",
                    "--gpus", "0",
                    "--output-root", temporary_directory,
                    "--serial",
                    "--dry-run",
                ],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn("QV_TYPE=transformer", result.stdout)
        self.assertIn("QV_TYPE=multi_prefix_transformer", result.stdout)
        self.assertNotIn("QV_TYPE=ordinary", result.stdout)
        self.assertIn("ROLLOUT_DEPTH=3", result.stdout)
        self.assertIn("TRAIN_STAGES=qv", result.stdout)
        self.assertIn("FREEZE_DYNAMICS_DURING_QV=1", result.stdout)
        self.assertIn("DYNAMICS_LOSS_WEIGHT=0", result.stdout)
        self.assertIn("run_dorl_mac_kuai_train.sh", result.stdout)
        self.assertIn("CRITIC_BACKBONE=mlp", result.stdout)
        self.assertNotIn("FIXED_HORIZON=", result.stdout)


if __name__ == "__main__":
    unittest.main()
