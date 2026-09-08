"""用完整未来轨迹回报验证 chunk 失效与及时重规划必要性。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.plot_replanning_motivation import plot_motivation_figure
from examples.our_model.evaluation import evaluate_branch_point
from examples.our_model.runners.common import (
    add_common_args,
    build_agent,
    build_dataset_and_mapper,
    build_env_assets,
    configure_logging,
    ensure_dir,
    namespace_to_dict,
    resolve_common_paths,
    resolve_device,
    save_resolved_config,
    set_mpl_cache_to_tmp,
    set_seed,
)
from examples.our_model.runners.evaluation_utils import (
    build_initial_state_table,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_EVAL_EPISODES = 100
"""动机实验默认用户轨迹数量。"""

DEFAULT_NUM_SAMPLES = 32
"""测试阶段 rejection sampling 的默认候选 chunk 数。"""

DEFAULT_BOOTSTRAP_SAMPLES = 2000
"""可视化置信区间的默认 episode-level bootstrap 次数。"""

FULL_DELAY_DIAGNOSTIC_POSITION = 1
"""只在 chunk 第一个分支位置枚举全部延迟，控制长期 rollout 计算量。"""

RECORDS_SCHEMA_VERSION = 2
"""完整未来轨迹回报评估所使用的 CSV schema 版本。"""

MAX_TORCH_SEED = 2**63 - 1
"""PyTorch Generator 接受的稳定非负种子上界。"""

INITIAL_DUMMY_ITEM = -1
"""在 episode reset 时表示空历史的虚拟物品编号。"""

SHARED_CHECKPOINT_FIELDS = (
    "env",
    "user_model_name",
    "read_message",
    "dataset_path",
    "which_tracker",
    "reward_handle",
    "window_size",
    "state_representation",
    "gamma",
    "num_leave_compute",
    "leave_threshold",
    "max_turn",
    "force_length",
    "item_embedding_path",
)
"""不同 K checkpoint 必须一致、且可从 checkpoint 恢复的公共配置。"""

AGENT_CHECKPOINT_FIELDS = (
    "actor_hidden_dims",
    "value_hidden_dims",
    "gamma",
    "target_tau",
    "invalid_action_penalty",
    "dynamics_loss_weight",
    "dynamics_hidden_dims",
)
"""构造每个 K agent 时从 checkpoint 恢复的结构与数值配置。"""

RECORD_FIELDNAMES = (
    "chunk_size",
    "episode_id",
    "user_id",
    "chunk_index",
    "chunk_position",
    "remaining_steps",
    "delay_steps",
    "immediate_local_reward",
    "delayed_local_reward",
    "immediate_future_return",
    "delayed_future_return",
    "immediate_local_executed_steps",
    "delayed_local_executed_steps",
    "immediate_future_executed_steps",
    "delayed_future_executed_steps",
    "immediate_local_terminated",
    "delayed_local_terminated",
    "immediate_terminated",
    "delayed_terminated",
    "long_term_replanning_advantage",
    "long_term_delay_cost",
    "short_term_replanning_gain",
    "short_term_delay_loss",
)
"""完整未来轨迹配对反事实记录的稳定 CSV 字段顺序。"""


@dataclass(frozen=True)
class TorchRNGSnapshot:
    """保存一次 MAC 规划前的 PyTorch 随机数状态。

    Attributes:
        cpu_state (torch.Tensor): CPU 随机数生成器状态。
        cuda_state (Optional[torch.Tensor]): 当前 CUDA 设备随机数状态；
            CPU 实验时为 `None`。
    """

    cpu_state: torch.Tensor
    cuda_state: Optional[torch.Tensor]


def build_parser() -> argparse.ArgumentParser:
    """构造动机实验命令行解析器。

    Returns:
        argparse.ArgumentParser: 已注册公共环境参数和动机实验参数的解析器。
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paired Continue-vs-Replan branches from identical "
            "DORL-MAC intermediate states."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--mac-ckpts",
        "--mac_ckpts",
        dest="mac_ckpts",
        type=str,
        nargs="+",
        required=True,
        metavar="K=PATH",
        help="一个或多个 K=checkpoint 路径，例如 3=.../K3/mac_agent/latest.pt。",
    )
    parser.add_argument(
        "--eval-episodes",
        "--eval_episodes",
        dest="eval_episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="每个 K 独立采样的主轨迹数。",
    )
    parser.add_argument(
        "--num-samples-test",
        "--num_samples_test",
        dest="num_samples_test",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="每次规划时 rejection sampling 的候选 chunk 数。",
    )
    parser.add_argument(
        "--max-chunks-per-episode",
        "--max_chunks_per_episode",
        dest="max_chunks_per_episode",
        type=int,
        default=0,
        help="每条主轨迹最多诊断的 chunk 数；0 表示直到 episode 结束。",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="results/replanning_motivation",
        help="原始记录、汇总和双子图输出目录。",
    )
    parser.add_argument(
        "--bootstrap-samples",
        "--bootstrap_samples",
        dest="bootstrap_samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="绘图时 episode-level bootstrap 次数。",
    )
    parser.add_argument(
        "--no-remove-recommended",
        dest="remove_recommended",
        action="store_false",
        help="重规划时不屏蔽已经推荐过的物品。",
    )
    parser.set_defaults(remove_recommended=True)
    return parser


def parse_checkpoint_specs(specs: Sequence[str]) -> dict[int, Path]:
    """解析并校验 `K=PATH` checkpoint 列表。

    Args:
        specs (Sequence[str]): 命令行传入的 checkpoint 规格。

    Returns:
        dict[int, Path]: 按 K 升序插入的 checkpoint 路径映射。

    Raises:
        ValueError: 当格式非法、K 非正、K 重复或列表为空时抛出。
        FileNotFoundError: 当任一 checkpoint 文件不存在时抛出。

    Example:
        >>> parse_checkpoint_specs(["3=/tmp/k3.pt"])  # doctest: +SKIP
        {3: PosixPath('/tmp/k3.pt')}
    """

    if not specs:
        raise ValueError("At least one --mac-ckpts K=PATH entry is required.")
    parsed: dict[int, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid checkpoint spec {spec!r}; expected K=PATH."
            )
        chunk_text, path_text = spec.split("=", maxsplit=1)
        try:
            chunk_size = int(chunk_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid chunk size in checkpoint spec {spec!r}."
            ) from error
        if chunk_size <= 1:
            raise ValueError(
                "Motivation branching requires chunk size K > 1, "
                f"got K={chunk_size}."
            )
        if chunk_size in parsed:
            raise ValueError(f"Duplicate checkpoint chunk size: K={chunk_size}.")
        checkpoint_path = Path(path_text).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint for K={chunk_size} does not exist: {checkpoint_path}"
            )
        parsed[chunk_size] = checkpoint_path
    return dict(sorted(parsed.items()))


def load_checkpoint_payloads(
    checkpoint_paths: dict[int, Path],
) -> dict[int, dict[str, Any]]:
    """加载并校验全部 MAC checkpoint。

    Args:
        checkpoint_paths (dict[int, Path]): K 到 checkpoint 路径的映射。

    Returns:
        dict[int, dict[str, Any]]: K 到 checkpoint 字典的映射。

    Raises:
        ValueError: 当 checkpoint 不是字典、缺少 config，或 config 中 K
            与命令行规格不一致时抛出。
    """

    payloads: dict[int, dict[str, Any]] = {}
    for chunk_size, checkpoint_path in checkpoint_paths.items():
        payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(
            payload.get("config"), dict
        ):
            raise ValueError(
                f"Checkpoint must contain a dict config: {checkpoint_path}"
            )
        configured_chunk_size = int(payload["config"].get("chunk_size", -1))
        if configured_chunk_size != chunk_size:
            raise ValueError(
                "Checkpoint chunk size does not match its K=PATH spec: "
                f"spec={chunk_size}, config={configured_chunk_size}, "
                f"path={checkpoint_path}."
            )
        payloads[chunk_size] = payload
    return payloads


def apply_shared_checkpoint_config(
    args: argparse.Namespace,
    checkpoint_payloads: dict[int, dict[str, Any]],
) -> None:
    """从 checkpoint 恢复公共实验配置并检查不同 K 的可比性。

    Args:
        args (argparse.Namespace): 会被原地更新的命令行参数。
        checkpoint_payloads (dict[int, dict[str, Any]]): 已加载 checkpoint。

    Returns:
        None.

    Raises:
        ValueError: 当不同 K 的关键环境或观测配置不一致时抛出。
    """

    configs = {
        chunk_size: payload["config"]
        for chunk_size, payload in checkpoint_payloads.items()
    }
    first_chunk_size = next(iter(configs))
    reference_config = configs[first_chunk_size]
    for field_name in SHARED_CHECKPOINT_FIELDS:
        reference_value = reference_config.get(field_name)
        if reference_value is None:
            continue
        inconsistent_values = {
            chunk_size: config.get(field_name)
            for chunk_size, config in configs.items()
            if config.get(field_name) != reference_value
        }
        if inconsistent_values:
            raise ValueError(
                "Checkpoints are not comparable because shared config differs: "
                f"field={field_name}, reference={reference_value!r}, "
                f"mismatches={inconsistent_values}."
            )
        setattr(args, field_name, reference_value)


def build_state_from_history(
    agent: torch.nn.Module,
    initial_states: torch.Tensor,
    user_id: int,
    history_items: Sequence[int],
    history_rewards: Sequence[float],
    device: torch.device,
) -> torch.Tensor:
    """从离线初始 observation 与已执行动作递推当前模型状态。

    Args:
        agent (torch.nn.Module): 含 chunk dynamics 与 action mapper 的 MAC agent。
        initial_states (torch.Tensor): 与环境内部用户 ID 对齐的初始状态表。
        user_id (int): 环境内部用户编号，与 user embedding 行号一致。
        history_items (Sequence[int]): 含 reset dummy 的物品历史。
        history_rewards (Sequence[float]): 与物品历史逐位置对齐的奖励历史。
        device (torch.device): 返回状态所在设备。

    Returns:
        torch.Tensor: 形状为 `(1, state_dim)` 的当前状态。

    Raises:
        ValueError: 当历史为空或物品、奖励长度不一致时抛出。
    """

    if not history_items or len(history_items) != len(history_rewards):
        raise ValueError(
            "State history must be non-empty and item/reward lengths must match."
        )
    if not 0 <= user_id < initial_states.shape[0]:
        raise ValueError(f"user_id is outside initial state table: {user_id}.")
    state = initial_states[user_id:user_id + 1].to(
        device=device, dtype=torch.float32,
    )
    executed_items = [int(item_id) for item_id in history_items if int(item_id) >= 0]
    with torch.no_grad():
        for start in range(0, len(executed_items), int(agent.chunk_size)):
            prefix = executed_items[start:start + int(agent.chunk_size)]
            prefix_ids = torch.as_tensor(prefix, dtype=torch.long, device=device)
            actions = torch.zeros(
                (1, int(agent.chunk_size), int(agent.action_dim)),
                dtype=torch.float32,
                device=device,
            )
            actions[0, :len(prefix)] = agent.action_mapper.item_embeddings[prefix_ids]
            valid = torch.zeros(
                (1, int(agent.chunk_size)), dtype=torch.bool, device=device,
            )
            valid[0, :len(prefix)] = True
            state = agent.dynamics(state, actions, valid)
    return state


def build_recommended_mask(
    history_items: Sequence[int],
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    """根据主轨迹或分支历史构造已推荐物品 mask。

    Args:
        history_items (Sequence[int]): 含 reset dummy 的物品历史。
        num_items (int): 环境物品总数。
        device (torch.device): mask 所在设备。

    Returns:
        torch.Tensor: 形状为 `(1, num_items)` 的 bool mask。

    Raises:
        ValueError: 当物品数非正或历史中出现大于等于 `num_items` 的编号时抛出。
    """

    if num_items <= 0:
        raise ValueError("num_items must be positive.")
    valid_item_ids = [int(item_id) for item_id in history_items if int(item_id) >= 0]
    if any(item_id >= num_items for item_id in valid_item_ids):
        raise ValueError("history_items contains an out-of-range item id.")
    mask = torch.zeros((1, num_items), dtype=torch.bool, device=device)
    if valid_item_ids:
        item_tensor = torch.as_tensor(
            sorted(set(valid_item_ids)),
            dtype=torch.long,
            device=device,
        )
        mask[0, item_tensor] = True
    return mask


def capture_torch_rng(device: torch.device) -> TorchRNGSnapshot:
    """捕获当前规划使用的 CPU/CUDA 随机数状态。

    Args:
        device (torch.device): MAC agent 所在设备。

    Returns:
        TorchRNGSnapshot: 可用于公共随机数配对规划的状态副本。
    """

    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device).clone()
    return TorchRNGSnapshot(
        cpu_state=torch.random.get_rng_state().clone(),
        cuda_state=cuda_state,
    )


def restore_torch_rng(
    rng_snapshot: TorchRNGSnapshot,
    device: torch.device,
) -> None:
    """恢复指定 CPU/CUDA 随机数状态。

    Args:
        rng_snapshot (TorchRNGSnapshot): `capture_torch_rng` 返回的状态。
        device (torch.device): MAC agent 所在设备。

    Returns:
        None.

    Raises:
        ValueError: 当 CUDA 设备与快照类型不匹配时抛出。
    """

    torch.random.set_rng_state(rng_snapshot.cpu_state)
    if device.type == "cuda":
        if rng_snapshot.cuda_state is None:
            raise ValueError("CUDA planning requires a CUDA RNG snapshot.")
        torch.cuda.set_rng_state(rng_snapshot.cuda_state, device)


def build_seeded_torch_rng(
    seed: int,
    device: torch.device,
) -> TorchRNGSnapshot:
    """从显式种子构造独立 RNG 快照，不改动进程全局随机状态。

    Args:
        seed (int): 位于 `[0, MAX_TORCH_SEED]` 的随机种子。
        device (torch.device): MAC agent 所在设备。

    Returns:
        TorchRNGSnapshot: CPU 与可选 CUDA 规划随机数状态。

    Raises:
        ValueError: 当种子越界时抛出。
    """

    if not 0 <= seed <= MAX_TORCH_SEED:
        raise ValueError(
            f"seed must be in [0, {MAX_TORCH_SEED}], got {seed}."
        )
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    cuda_state = None
    if device.type == "cuda":
        cuda_generator = torch.Generator(device=device)
        cuda_generator.manual_seed(seed)
        cuda_state = cuda_generator.get_state().clone()
    return TorchRNGSnapshot(
        cpu_state=cpu_generator.get_state().clone(),
        cuda_state=cuda_state,
    )


def derive_downstream_seed(
    base_seed: int,
    chunk_size: int,
    episode_id: int,
    chunk_index: int,
    chunk_position: int,
    planning_index: int,
) -> int:
    """为同一分支点的第 n 次下游规划派生稳定公共随机数种子。

    延迟编号不参与种子计算，因此所有 Continue/Replan 分支在相同的
    下游规划编号上使用同一候选采样随机流；状态不同仍可产生不同动作。

    Args:
        base_seed (int): 实验全局随机种子。
        chunk_size (int): 当前 checkpoint 的规划长度。
        episode_id (int): 当前主轨迹编号。
        chunk_index (int): 当前 chunk 编号。
        chunk_position (int): 当前 chunk 内一基分支位置。
        planning_index (int): 局部干预结束后的零基规划编号。

    Returns:
        int: 可传给 `build_seeded_torch_rng` 的稳定种子。
    """

    identity = (
        f"{base_seed}|{chunk_size}|{episode_id}|{chunk_index}|"
        f"{chunk_position}|{planning_index}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % MAX_TORCH_SEED


def select_chunk(
    agent: torch.nn.Module,
    initial_states: torch.Tensor,
    user_id: int,
    history_items: Sequence[int],
    history_rewards: Sequence[float],
    num_samples_test: int,
    remove_recommended: bool,
    device: torch.device,
    preserve_rng: bool,
    planning_rng: Optional[TorchRNGSnapshot] = None,
) -> list[int]:
    """根据显式最新历史调用同一个 MAC agent 生成动作 chunk。

    Args:
        agent (torch.nn.Module): 已加载 checkpoint 的 `MACAgent`。
        initial_states (torch.Tensor): 与环境内部用户 ID 对齐的初始状态表。
        user_id (int): 环境内部用户编号。
        history_items (Sequence[int]): 当前物品历史。
        history_rewards (Sequence[float]): 当前奖励历史。
        num_samples_test (int): rejection sampling 候选数。
        remove_recommended (bool): 是否屏蔽历史物品。
        device (torch.device): agent 所在设备。
        preserve_rng (bool): 是否在规划后恢复 PyTorch RNG；shadow 分支应为
            True，主轨迹规划应为 False。
        planning_rng (Optional[TorchRNGSnapshot]): 可选的公共规划随机数起点。
            传入后，shadow replan 会复用原 chunk 规划前的同一组随机数，
            排除候选采样噪声造成的伪 Replanning Gain。

    Returns:
        list[int]: 长度为 agent chunk size 的内部物品编号。

    Raises:
        ValueError: 当候选数非正时抛出。
    """

    if num_samples_test <= 0:
        raise ValueError("num_samples_test must be positive.")
    state = build_state_from_history(
        agent=agent,
        initial_states=initial_states,
        user_id=user_id,
        history_items=history_items,
        history_rewards=history_rewards,
        device=device,
    )
    recommended_mask = None
    if remove_recommended:
        recommended_mask = build_recommended_mask(
            history_items=history_items,
            num_items=int(agent.num_items),
            device=device,
        )

    caller_rng = capture_torch_rng(device) if preserve_rng else None
    if planning_rng is not None:
        restore_torch_rng(planning_rng, device=device)
    try:
        with torch.no_grad():
            item_ids, _ = agent.select_chunks(
                state,
                num_samples=num_samples_test,
                recommended_mask=recommended_mask,
            )
    finally:
        if caller_rng is not None:
            restore_torch_rng(caller_rng, device=device)
    return [int(item_id) for item_id in item_ids[0].detach().cpu().tolist()]


def _append_branch_records(
    records: list[dict[str, Any]],
    result: Any,
    chunk_size: int,
    episode_id: int,
    user_id: int,
    chunk_index: int,
    chunk_position: int,
) -> None:
    """把一个中间状态的全部延迟分支展开为 CSV 行。

    Args:
        records (list[dict[str, Any]]): 会被原地追加的记录列表。
        result (Any): `evaluate_branch_point` 返回结果。
        chunk_size (int): 当前 agent 的规划长度 K。
        episode_id (int): 当前主轨迹编号。
        user_id (int): 当前环境内部用户编号。
        chunk_index (int): 当前 chunk 在主轨迹中的编号。
        chunk_position (int): 已执行动作在 chunk 中的一基位置。

    Returns:
        None.
    """

    immediate_evaluation = result.immediate_evaluation
    long_term_advantage = float(result.long_term_replanning_advantage)
    short_term_gain = float(result.short_term_replanning_gain)
    for evaluation in result.evaluations:
        records.append(
            {
                "chunk_size": chunk_size,
                "episode_id": episode_id,
                "user_id": user_id,
                "chunk_index": chunk_index,
                "chunk_position": chunk_position,
                "remaining_steps": result.planned_steps,
                "delay_steps": evaluation.delay_steps,
                "immediate_local_reward": (
                    immediate_evaluation.local_reward_sum
                ),
                "delayed_local_reward": evaluation.local_reward_sum,
                "immediate_future_return": (
                    immediate_evaluation.future_reward_sum
                ),
                "delayed_future_return": evaluation.future_reward_sum,
                "immediate_local_executed_steps": (
                    immediate_evaluation.local_executed_steps
                ),
                "delayed_local_executed_steps": (
                    evaluation.local_executed_steps
                ),
                "immediate_future_executed_steps": (
                    immediate_evaluation.future_executed_steps
                ),
                "delayed_future_executed_steps": (
                    evaluation.future_executed_steps
                ),
                "immediate_local_terminated": int(
                    immediate_evaluation.local_terminated
                ),
                "delayed_local_terminated": int(evaluation.local_terminated),
                "immediate_terminated": int(immediate_evaluation.terminated),
                "delayed_terminated": int(evaluation.terminated),
                "long_term_replanning_advantage": long_term_advantage,
                "long_term_delay_cost": (
                    immediate_evaluation.future_reward_sum
                    - evaluation.future_reward_sum
                ),
                "short_term_replanning_gain": short_term_gain,
                "short_term_delay_loss": (
                    immediate_evaluation.local_reward_sum
                    - evaluation.local_reward_sum
                )
                / result.planned_steps,
            }
        )


def evaluate_checkpoint(
    args: argparse.Namespace,
    env: Any,
    initial_states: torch.Tensor,
    agent: torch.nn.Module,
    chunk_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """采集一个 K checkpoint 的配对反事实动机实验记录。

    Args:
        args (argparse.Namespace): 动机实验配置。
        env (Any): KuaiEnv 主轨迹环境。
        initial_states (torch.Tensor): 共享的离线用户初始状态表。
        agent (torch.nn.Module): 当前 K 的 MAC agent。
        chunk_size (int): 当前规划长度 K。
        device (torch.device): 模型设备。

    Returns:
        list[dict[str, Any]]: 可直接写入稳定 CSV schema 的原始记录。

    Raises:
        ValueError: 当评估 episode 数或 chunk 上限非法时抛出。
    """

    if args.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive.")
    if args.max_chunks_per_episode < 0:
        raise ValueError("max_chunks_per_episode must be non-negative.")

    set_seed(args.seed)
    agent.eval()
    records: list[dict[str, Any]] = []
    for episode_id in range(args.eval_episodes):
        observation, _ = env.reset()
        user_id = int(np.asarray(observation).reshape(-1)[0])
        history_items = [INITIAL_DUMMY_ITEM]
        history_rewards = [0.0]
        terminated = False
        chunk_index = 0

        while not terminated:
            if (
                args.max_chunks_per_episode > 0
                and chunk_index >= args.max_chunks_per_episode
            ):
                break
            # 所有 shadow replan 从原 chunk 规划前的同一随机数状态开始。
            # 若用户状态未变化，它们应复现相同 chunk；状态变化后的差异
            # 才能归因于新反馈，而非 rejection sampling 的随机候选差异。
            base_planning_rng = capture_torch_rng(device)
            base_chunk = select_chunk(
                agent=agent,
                initial_states=initial_states,
                user_id=user_id,
                history_items=history_items,
                history_rewards=history_rewards,
                num_samples_test=args.num_samples_test,
                remove_recommended=args.remove_recommended,
                device=device,
                preserve_rng=False,
            )
            replayed_base_chunk = select_chunk(
                agent=agent,
                initial_states=initial_states,
                user_id=user_id,
                history_items=history_items,
                history_rewards=history_rewards,
                num_samples_test=args.num_samples_test,
                remove_recommended=args.remove_recommended,
                device=device,
                preserve_rng=True,
                planning_rng=base_planning_rng,
            )
            if replayed_base_chunk != base_chunk:
                raise RuntimeError(
                    "Common-random-number control failed to reproduce the "
                    f"original chunk for K={chunk_size}."
                )
            for zero_based_position, action in enumerate(base_chunk):
                _, reward, terminated_flag, truncated, _ = env.step(int(action))
                reward_float = float(reward)
                history_items.append(int(action))
                history_rewards.append(reward_float)
                terminated = bool(terminated_flag or truncated)
                chunk_position = zero_based_position + 1
                cached_suffix = base_chunk[chunk_position:]

                if not terminated and cached_suffix:
                    def local_planner(
                        items: Sequence[int],
                        rewards: Sequence[float],
                    ) -> list[int]:
                        """用原 chunk 的候选随机流执行局部重规划。"""

                        return select_chunk(
                            agent=agent,
                            initial_states=initial_states,
                            user_id=user_id,
                            history_items=items,
                            history_rewards=rewards,
                            num_samples_test=args.num_samples_test,
                            remove_recommended=args.remove_recommended,
                            device=device,
                            preserve_rng=True,
                            planning_rng=base_planning_rng,
                        )

                    def downstream_planner(
                        items: Sequence[int],
                        rewards: Sequence[float],
                        planning_index: int,
                    ) -> list[int]:
                        """用分支间对齐的随机流生成一个下游完整 chunk。"""

                        downstream_seed = derive_downstream_seed(
                            base_seed=args.seed,
                            chunk_size=chunk_size,
                            episode_id=episode_id,
                            chunk_index=chunk_index,
                            chunk_position=chunk_position,
                            planning_index=planning_index,
                        )
                        downstream_rng = build_seeded_torch_rng(
                            seed=downstream_seed,
                            device=device,
                        )
                        return select_chunk(
                            agent=agent,
                            initial_states=initial_states,
                            user_id=user_id,
                            history_items=items,
                            history_rewards=rewards,
                            num_samples_test=args.num_samples_test,
                            remove_recommended=args.remove_recommended,
                            device=device,
                            preserve_rng=True,
                            planning_rng=downstream_rng,
                        )

                    if chunk_position == FULL_DELAY_DIAGNOSTIC_POSITION:
                        delay_values = list(range(len(cached_suffix) + 1))
                    else:
                        # 左图只需要立即重规划与完整 Continue 两个端点。
                        delay_values = [0, len(cached_suffix)]
                    result = evaluate_branch_point(
                        env=env,
                        history_items=history_items,
                        history_rewards=history_rewards,
                        cached_suffix=cached_suffix,
                        planner=local_planner,
                        downstream_planner=downstream_planner,
                        delay_values=delay_values,
                    )
                    _append_branch_records(
                        records=records,
                        result=result,
                        chunk_size=chunk_size,
                        episode_id=episode_id,
                        user_id=user_id,
                        chunk_index=chunk_index,
                        chunk_position=chunk_position,
                    )
                if terminated:
                    break
            chunk_index += 1
        LOGGER.info(
            "K=%s 动机实验进度：episode=%s/%s, records=%s",
            chunk_size,
            episode_id + 1,
            args.eval_episodes,
            len(records),
        )
    return records


def save_records(records: Sequence[dict[str, Any]], output_path: Path) -> Path:
    """保存配对反事实原始记录。

    Args:
        records (Sequence[dict[str, Any]]): `evaluate_checkpoint` 生成的记录。
        output_path (Path): 目标 CSV 路径。

    Returns:
        Path: 已写入的 CSV 路径。

    Raises:
        ValueError: 当记录为空时抛出。
    """

    if not records:
        raise ValueError(
            "No valid branch points were collected. Increase eval episodes or "
            "check whether episodes terminate at the first action."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=RECORD_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def main(argv: Optional[list[str]] = None) -> Path:
    """运行多 K 配对反事实实验并生成双子图。

    Args:
        argv (Optional[list[str]]): 可选命令行参数；为空时读取进程参数。

    Returns:
        Path: 输出目录。

    Raises:
        ValueError: 当 checkpoint 配置不可比或未采集到分支点时抛出。
        FileNotFoundError: 当 checkpoint 或环境资产不存在时抛出。
    """

    configure_logging()
    set_mpl_cache_to_tmp()
    args = build_parser().parse_args(argv)
    checkpoint_paths = parse_checkpoint_specs(args.mac_ckpts)
    checkpoint_payloads = load_checkpoint_payloads(checkpoint_paths)
    apply_shared_checkpoint_config(args, checkpoint_payloads)
    resolve_common_paths(args)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    set_seed(args.seed)

    resolved_config = namespace_to_dict(args)
    resolved_config["mac_ckpts"] = {
        str(chunk_size): str(path)
        for chunk_size, path in checkpoint_paths.items()
    }
    save_resolved_config(output_dir, resolved_config)

    LOGGER.info(
        "开始构造动机实验共享环境：K=%s, episodes=%s, output=%s",
        list(checkpoint_paths),
        args.eval_episodes,
        output_dir,
    )
    env, _, _ = build_env_assets(args)
    offline_dataset, action_mapper, _ = build_dataset_and_mapper(args, device=device)
    initial_states = build_initial_state_table(offline_dataset, env, device)

    all_records: list[dict[str, Any]] = []
    for chunk_size, checkpoint_path in checkpoint_paths.items():
        checkpoint = checkpoint_payloads[chunk_size]
        agent_args = argparse.Namespace(**vars(args))
        agent_args.chunk_size = chunk_size
        checkpoint_config = checkpoint["config"]
        for field_name in AGENT_CHECKPOINT_FIELDS:
            if field_name in checkpoint_config:
                setattr(agent_args, field_name, checkpoint_config[field_name])
        agent = build_agent(
            agent_args,
            device=device,
            action_mapper=action_mapper,
            state_dim=offline_dataset.state_dim,
        )
        agent.load_checkpoint_state(checkpoint, strict=True)
        agent.eval()
        LOGGER.info(
            "开始评估 K=%s checkpoint=%s",
            chunk_size,
            checkpoint_path,
        )
        all_records.extend(
            evaluate_checkpoint(
                args=args,
                env=env,
                initial_states=initial_states,
                agent=agent,
                chunk_size=chunk_size,
                device=device,
            )
        )

    records_path = save_records(
        all_records,
        output_dir / "paired_counterfactual_records.csv",
    )
    figure_paths, summary_path = plot_motivation_figure(
        records_path=records_path,
        output_dir=output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    manifest = {
        "status": "completed",
        "records_schema_version": RECORDS_SCHEMA_VERSION,
        "primary_metric_definition": (
            "undiscounted_complete_future_trajectory_return_from_branch_state"
        ),
        "downstream_execution_horizon": "checkpoint_chunk_size",
        "full_delay_diagnostic_position": FULL_DELAY_DIAGNOSTIC_POSITION,
        "records": str(records_path),
        "summary": str(summary_path),
        "figures": [str(path) for path in figure_paths],
        "num_records": len(all_records),
        "chunk_sizes": list(checkpoint_paths),
    }
    manifest_path = output_dir / "experiment_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, indent=2, ensure_ascii=False)
    LOGGER.info("动机实验完成：manifest=%s", manifest_path)
    return output_dir


if __name__ == "__main__":
    main()
