"""DORL-MAC 推荐环境评估入口。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.our_model.logging import SwanLabLogger
from examples.our_model.policy import ActionMapper
from examples.our_model.runners.common import (
    add_common_args,
    build_agent,
    apply_checkpoint_model_config,
    build_dataset_and_mapper,
    build_env_assets,
    build_reward_and_leave,
    configure_logging,
    default_run_name,
    ensure_dir,
    namespace_to_dict,
    resolve_common_paths,
    resolve_device,
    save_resolved_config,
    set_mpl_cache_to_tmp,
    set_seed,
)
from examples.our_model.runners.evaluation_utils import (
    build_dorl_mac_evaluator,
    build_initial_state_table,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_EVAL_EPISODES = 100
"""独立开环评估默认采样的轨迹数量。"""


def build_parser() -> argparse.ArgumentParser:
    """构造评估参数解析器。

    Returns:
        argparse.ArgumentParser: 参数解析器。
    """

    parser = argparse.ArgumentParser(description="Evaluate DORL-MAC in DORL environments.")
    add_common_args(parser)
    parser.add_argument(
        "--execution-horizons",
        "--execution_horizons",
        dest="execution_horizons",
        type=int,
        nargs="+",
        default=None,
        help=(
            "在同一 Python 进程中依次评估多个 H；数据集、user model、"
            "离线初始状态和 checkpoint 只加载一次。设置后不能同时传入"
            "单值 --execution_horizon。"
        ),
    )
    parser.add_argument("--mac_ckpt", type=str, required=True)
    parser.add_argument("--num_samples_test", type=int, default=32)
    parser.add_argument(
        "--test-num",
        "--test_num",
        dest="test_num",
        type=int,
        default=1,
        help="并行测试环境数量；不决定总评估轨迹数。",
    )
    parser.add_argument(
        "--eval-episodes",
        "--eval_episodes",
        dest="eval_episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="每个 H、每个评估分支采样的独立轨迹数量，指标按轨迹汇总。",
    )
    parser.add_argument("--eval_save_dir", type=str, default="")
    parser.add_argument("--buffer-size", "--buffer_size", dest="buffer_size", type=int, default=0)
    return parser


def resolve_execution_horizons(args: argparse.Namespace) -> list[int]:
    """解析并校验单值或多值 execution horizon。

    Args:
        args (argparse.Namespace): 必须包含 `chunk_size`、
            `execution_horizon` 和 `execution_horizons`。

    Returns:
        list[int]: 保持用户输入顺序且不重复的 H 列表。

    Raises:
        ValueError: 当同时设置单值和多值 H、H 重复或 H 超出
            `[1, chunk_size]` 时抛出。

    Example:
        `chunk_size=5, execution_horizons=[1, 3, 5]` 返回
        `[1, 3, 5]`。
    """

    single_horizon = getattr(args, "execution_horizon", None)
    multiple_horizons = getattr(args, "execution_horizons", None)
    if single_horizon is not None and multiple_horizons is not None:
        raise ValueError(
            "--execution_horizon and --execution_horizons cannot be used together."
        )
    if multiple_horizons is None:
        resolved_horizons = [
            int(args.chunk_size if single_horizon is None else single_horizon)
        ]
    else:
        resolved_horizons = [int(horizon) for horizon in multiple_horizons]
    if len(set(resolved_horizons)) != len(resolved_horizons):
        raise ValueError(
            f"execution horizons must not contain duplicates: {resolved_horizons}."
        )
    invalid_horizons = [
        horizon
        for horizon in resolved_horizons
        if not 1 <= horizon <= int(args.chunk_size)
    ]
    if invalid_horizons:
        raise ValueError(
            "Each execution horizon must satisfy "
            f"1 <= H <= chunk_size ({args.chunk_size}), "
            f"got {invalid_horizons}."
        )
    return resolved_horizons


def resolve_eval_episodes(args: argparse.Namespace) -> int:
    """解析并校验独立评估轨迹数量。

    `eval_episodes` 是每个 FB/NX 分支实际采样的 episode 数，和
    `test_num`（并行环境数量）相互独立。保留值 0 的兼容语义：它会
    回退为 `test_num`，以兼容旧命令行调用。

    Args:
        args (argparse.Namespace): 必须包含 `eval_episodes` 与 `test_num`。

    Returns:
        int: 每个评估分支应采样的正整数轨迹数量。

    Raises:
        ValueError: 当 `eval_episodes` 为负数，或回退后的轨迹数非正时抛出。
    """

    requested_episodes = int(args.eval_episodes)
    if requested_episodes < 0:
        raise ValueError("eval_episodes must be non-negative.")
    resolved_episodes = (
        requested_episodes if requested_episodes > 0 else int(args.test_num)
    )
    if resolved_episodes <= 0:
        raise ValueError("test_num must be positive when eval_episodes is 0.")
    return resolved_episodes


def close_evaluator_environments(evaluator: Any) -> None:
    """关闭一个 H 对应的所有向量测试环境。

    单进程 sweep 在进入下一个 H 前显式关闭当前 CollectorSet 的
    FB/NX 环境，避免循环持有 worker 或环境资源。

    Args:
        evaluator (Any): 包含 `collector_set.collector_dict` 的评估器。

    Returns:
        None.
    """

    collector_dict = getattr(
        getattr(evaluator, "collector_set", None),
        "collector_dict",
        {},
    )
    for collector in collector_dict.values():
        close_method = getattr(getattr(collector, "env", None), "close", None)
        if callable(close_method):
            close_method()


def evaluate_one_horizon(
    args: argparse.Namespace,
    execution_horizon: int,
    save_dir: Path,
    run_name: str,
    env: Any,
    env_dataset: Any,
    kwargs_um: dict[str, Any],
    agent: Any,
    action_mapper: ActionMapper,
    initial_states: torch.Tensor,
    device: torch.device,
    eval_episodes: int,
) -> Path:
    """使用共享模型资产评估一个 execution horizon。

    每次调用都会复制参数、恢复相同随机种子、创建独立 Policy/Collector
    和测试环境，并在结束后关闭该组环境。大型数据和模型对象由调用方共享。

    Args:
        args (argparse.Namespace): 已解析并完成公共路径解析的参数。
        execution_horizon (int): 当前评估的 H。
        save_dir (Path): 当前 H 的输出目录。
        run_name (str): 当前 H 的日志运行名。
        env (Any): 共享的真实环境模板。
        env_dataset (Any): 共享的 DORL 数据集对象。
        kwargs_um (dict[str, Any]): 测试环境构造参数。
        agent (Any): 已加载 checkpoint 的共享 MAC agent。
        action_mapper (ActionMapper): 共享动作映射器。
        initial_states (torch.Tensor): 共享的离线用户初始状态表。
        device (torch.device): 评估设备。
        eval_episodes (int): 当前 H 的评估 episode 数。

    Returns:
        Path: 当前 H 的 `summary_metrics.json` 路径。
    """

    horizon_args = argparse.Namespace(**vars(args))
    horizon_args.execution_horizon = int(execution_horizon)
    horizon_args.eval_save_dir = str(save_dir)
    horizon_args.run_name = run_name
    set_seed(horizon_args.seed)
    config = namespace_to_dict(horizon_args)
    save_resolved_config(save_dir, config)

    LOGGER.info(
        "开始单进程 H 评估：H=%s, seed=%s, output=%s",
        execution_horizon,
        horizon_args.seed,
        save_dir,
    )
    evaluator = build_dorl_mac_evaluator(
        args=horizon_args,
        env=env,
        dataset=env_dataset,
        kwargs_um=kwargs_um,
        agent=agent,
        action_mapper=action_mapper,
        initial_states=initial_states,
        num_samples_test=horizon_args.num_samples_test,
        execution_horizon=execution_horizon,
        completion_window=horizon_args.completion_window,
        enable_open_loop_diagnostics=horizon_args.enable_open_loop_diagnostics,
        device=device,
        eval_episodes=eval_episodes,
        save_dir=save_dir,
        buffer_size=horizon_args.buffer_size,
    )
    try:
        summary = evaluator.evaluate(epoch=0, global_step=0)
        summary.setdefault("evaluation/eval_episodes", int(eval_episodes))
        summary.setdefault(
            "evaluation/parallel_test_envs",
            min(int(horizon_args.test_num), int(eval_episodes)),
        )
        summary.setdefault("evaluation/trajectory_aggregation", "mean_or_pooled_rate")
        configured_horizons = horizon_args.execution_horizons
        summary.setdefault(
            "evaluation/single_process_sweep",
            int(configured_horizons is not None),
        )
        summary.setdefault(
            "evaluation/sweep_size",
            len(configured_horizons) if configured_horizons is not None else 1,
        )
        summary_path = save_dir / "summary_metrics.json"
        with summary_path.open("w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2, ensure_ascii=False)
        logger = SwanLabLogger(
            project=horizon_args.swanlab_project,
            run_name=default_run_name("eval", horizon_args),
            config=config,
            log_path=str(save_dir / "metrics.jsonl"),
        )
        logger.log(
            {
                key: value
                for key, value in summary.items()
                if isinstance(value, (float, int))
            },
            step=0,
        )
        logger.finish()
    finally:
        close_evaluator_environments(evaluator)
    LOGGER.info(
        "单进程 H 评估完成：H=%s, summary=%s",
        execution_horizon,
        summary_path,
    )
    return summary_path


def main(argv: Optional[list[str]] = None) -> Path:
    """执行单 H 或单进程多 H 的 DORL-MAC 环境评估。

    Args:
        argv (Optional[list[str]]): 可选命令行参数列表。

    Returns:
        Path: 单 H 时返回 summary JSON；多 H 时返回输出根目录。

    Raises:
        FileNotFoundError: 当 checkpoint 不存在时抛出。
        ValueError: 当评估 episode 数或 execution horizons 非法时抛出。
    """

    configure_logging()
    set_mpl_cache_to_tmp()
    parser = build_parser()
    args = parser.parse_args(argv)
    is_multi_horizon = args.execution_horizons is not None
    mac_ckpt = Path(args.mac_ckpt)
    if not mac_ckpt.is_file():
        raise FileNotFoundError(f"mac_ckpt file does not exist: {mac_ckpt}")
    checkpoint = torch.load(mac_ckpt, map_location="cpu")
    apply_checkpoint_model_config(args, checkpoint)
    execution_horizons = resolve_execution_horizons(args)
    effective_eval_episodes = resolve_eval_episodes(args)

    resolve_common_paths(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_base = ensure_dir(
        args.eval_save_dir
        or Path(args.save_root) / args.env / "MAC_origin" / "eval"
    )

    LOGGER.info(
        "开始加载单进程共享评估资产：H=%s, episodes=%s, parallel_envs=%s, checkpoint=%s",
        execution_horizons,
        effective_eval_episodes,
        min(int(args.test_num), effective_eval_episodes),
        mac_ckpt,
    )
    env, env_dataset, kwargs_um = build_env_assets(args)
    offline_dataset, action_mapper, _ = build_dataset_and_mapper(args, device=device)
    initial_states = build_initial_state_table(offline_dataset, env, device)
    reward_model, leave_model = build_reward_and_leave(
        args,
        env=env,
        dataset=env_dataset,
        device=device,
    )
    agent = build_agent(
        args,
        device=device,
        action_mapper=action_mapper,
        state_dim=offline_dataset.state_dim,
        reward_model=reward_model,
        leave_model=leave_model,
    )
    agent.load_checkpoint_state(checkpoint, strict=True)
    agent.eval()
    LOGGER.info(
        "共享评估资产加载完成；后续 %d 个 H 不再重载数据集、模型或 checkpoint。",
        len(execution_horizons),
    )

    original_run_name = args.run_name
    summary_paths: list[Path] = []
    for execution_horizon in execution_horizons:
        if is_multi_horizon:
            save_dir = ensure_dir(output_base / f"H{execution_horizon}")
            run_name_base = original_run_name or f"open-loop-K{args.chunk_size}"
            run_name = f"{run_name_base}-H{execution_horizon}"
        else:
            save_dir = output_base
            run_name = original_run_name
        summary_paths.append(
            evaluate_one_horizon(
                args=args,
                execution_horizon=execution_horizon,
                save_dir=save_dir,
                run_name=run_name,
                env=env,
                env_dataset=env_dataset,
                kwargs_um=kwargs_um,
                agent=agent,
                action_mapper=action_mapper,
                initial_states=initial_states,
                device=device,
                eval_episodes=effective_eval_episodes,
            )
        )

    if is_multi_horizon:
        LOGGER.info(
            "单进程多 H 评估全部完成：H=%s, output=%s",
            execution_horizons,
            output_base,
        )
        return output_base
    return summary_paths[0]


if __name__ == "__main__":
    main()
