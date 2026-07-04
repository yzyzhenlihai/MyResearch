"""DORL-MAC chunk-level Q/V 训练入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.our_model.logging import SwanLabLogger
from examples.our_model.models.mac_agent import REPEAT_POLICY_MASK, REPEAT_POLICY_TRUNCATE
from examples.our_model.runners.common import (
    add_common_args,
    build_agent,
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
    set_seed,
)
from examples.our_model.runners.evaluation_utils import build_dorl_mac_evaluator
from examples.our_model.runners.pretrain_flow_bc import cycle_dataloader

LOGGER = logging.getLogger(__name__)

DEFAULT_TRAIN_STEPS = 500_000
"""正式 Q/V 训练默认步数。"""

LEGACY_EPOCH_SENTINEL = 0
"""未显式指定 epoch 调度时的兼容占位值。"""

DEFAULT_EVAL_EPISODES_SENTINEL = 0
"""评估 episode 为 0 时复用 DORL 的 `test_num` 语义。"""


def build_parser() -> argparse.ArgumentParser:
    """构造 Q/V 训练参数解析器。

    Returns:
        argparse.ArgumentParser: 参数解析器。
    """

    parser = argparse.ArgumentParser(description="Train DORL-MAC chunk Q/V.")
    add_common_args(parser)
    parser.add_argument("--flow_actor_ckpt", type=str, required=True)
    parser.add_argument("--train_steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--epoch", type=int, default=LEGACY_EPOCH_SENTINEL)
    parser.add_argument(
        "--step-per-epoch",
        "--step_per_epoch",
        dest="step_per_epoch",
        type=int,
        default=LEGACY_EPOCH_SENTINEL,
    )
    parser.add_argument("--qv_lr", type=float, default=3e-4)
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--num_samples_train", type=int, default=8)
    parser.add_argument(
        "--repeat_policy",
        choices=[REPEAT_POLICY_TRUNCATE, REPEAT_POLICY_MASK],
        default=REPEAT_POLICY_TRUNCATE,
    )
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=DEFAULT_EVAL_EPISODES_SENTINEL)
    parser.add_argument("--num_samples_test", type=int, default=32)
    parser.add_argument("--test-num", "--test_num", dest="test_num", type=int, default=1)
    parser.add_argument("--eval_save_dir", type=str, default="")
    parser.add_argument("--buffer-size", "--buffer_size", dest="buffer_size", type=int, default=0)
    return parser


def resolve_epoch_schedule(args: argparse.Namespace) -> tuple[int, int, int]:
    """解析 DORL 风格的 epoch 调度。

    Args:
        args (argparse.Namespace): Q/V runner 参数。

    Returns:
        tuple[int, int, int]: `(epoch, step_per_epoch, total_steps)`。

    Raises:
        ValueError: 当调度参数非法时抛出。
    """

    if args.train_steps <= 0:
        raise ValueError("train_steps must be positive.")
    if args.epoch < 0 or args.step_per_epoch < 0:
        raise ValueError("epoch and step_per_epoch must be non-negative.")
    if args.epoch == LEGACY_EPOCH_SENTINEL and args.step_per_epoch == LEGACY_EPOCH_SENTINEL:
        epoch = 1
        step_per_epoch = int(args.train_steps)
    elif args.epoch > 0 and args.step_per_epoch > 0:
        epoch = int(args.epoch)
        step_per_epoch = int(args.step_per_epoch)
    else:
        raise ValueError("epoch and step_per_epoch must be provided together.")
    total_steps = epoch * step_per_epoch
    args.epoch = epoch
    args.step_per_epoch = step_per_epoch
    args.train_steps = total_steps
    return epoch, step_per_epoch, total_steps


def main(argv: Optional[list[str]] = None) -> Path:
    """执行 Q/V 训练。

    Args:
        argv (Optional[list[str]]): 可选命令行参数列表。

    Returns:
        Path: 最终 checkpoint 路径。

    Raises:
        FileNotFoundError: 当 actor checkpoint 不存在时抛出。
        ValueError: 当训练步数非法时抛出。
    """

    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    epoch, step_per_epoch, total_steps = resolve_epoch_schedule(args)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.eval_episodes < 0:
        raise ValueError("eval_episodes must be non-negative.")
    if args.eval_interval > 0:
        LOGGER.warning("eval_interval is ignored in DORL-style epoch evaluation.")
    flow_actor_ckpt = Path(args.flow_actor_ckpt)
    if not flow_actor_ckpt.exists():
        raise FileNotFoundError(f"flow_actor_ckpt does not exist: {flow_actor_ckpt}")
    effective_eval_episodes = args.eval_episodes if args.eval_episodes > 0 else args.test_num
    if effective_eval_episodes <= 0:
        raise ValueError("test_num must be positive when eval_episodes is 0.")

    resolve_common_paths(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.device = str(device)
    save_dir = ensure_dir(
        args.save_dir
        or Path(args.save_root) / args.env / "DORL_MAC" / "mac_agent"
    )
    config = namespace_to_dict(args)
    save_resolved_config(save_dir, config)

    dataset, action_mapper, _ = build_dataset_and_mapper(args, device=device)
    env, env_dataset, kwargs_um = build_env_assets(args)
    reward_model, leave_model = build_reward_and_leave(args, env=env, device=device)
    agent = build_agent(
        args,
        device=device,
        action_mapper=action_mapper,
        reward_model=reward_model,
        leave_model=leave_model,
    )
    checkpoint = torch.load(flow_actor_ckpt, map_location=device)
    agent.load_checkpoint_state(checkpoint, strict=False)
    eval_save_dir = ensure_dir(args.eval_save_dir or save_dir / "eval_during_train")
    periodic_evaluator = build_dorl_mac_evaluator(
        args=args,
        env=env,
        dataset=env_dataset,
        kwargs_um=kwargs_um,
        agent=agent,
        action_mapper=action_mapper,
        device=device,
        num_samples_test=args.num_samples_test,
        eval_episodes=effective_eval_episodes,
        save_dir=eval_save_dir,
        buffer_size=args.buffer_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(agent.qv_parameters(), lr=args.qv_lr)
    logger = SwanLabLogger(
        project=args.swanlab_project,
        run_name=default_run_name("qv", args),
        config=config,
        log_path=str(save_dir / "metrics.jsonl"),
    )

    LOGGER.info(
        "开始 DORL-MAC Q/V 训练：epoch=%s, step_per_epoch=%s, total_steps=%s",
        epoch,
        step_per_epoch,
        total_steps,
    )
    batch_iterator = cycle_dataloader(dataloader)
    last_metrics = {}
    global_step = 0
    best_reward = float("-inf")
    for current_epoch in range(1, epoch + 1):
        LOGGER.info("开始 Q/V epoch=%s/%s", current_epoch, epoch)
        for _ in range(step_per_epoch):
            global_step += 1
            batch = next(batch_iterator)
            last_metrics = agent.qv_update(
                batch=batch,
                optimizer=optimizer,
                num_samples_train=args.num_samples_train,
                repeat_policy=args.repeat_policy,
            )
            if global_step == 1 or global_step % args.log_interval == 0 or global_step == total_steps:
                train_metrics = dict(last_metrics)
                train_metrics["trainer/epoch"] = current_epoch
                train_metrics["trainer/global_step"] = global_step
                logger.log(train_metrics, step=global_step)
                LOGGER.info("qv step=%s metrics=%s", global_step, train_metrics)

        eval_summary = periodic_evaluator.evaluate(epoch=current_epoch, global_step=global_step)
        if "rew" in eval_summary:
            best_reward = max(best_reward, float(eval_summary["rew"]))
            eval_summary["best_reward"] = best_reward
        logger.log(eval_summary, step=current_epoch)

    checkpoint_path = save_dir / "latest.pt"
    checkpoint_config = dict(config)
    checkpoint_config.update(
        {
            "state_dim": agent.state_dim,
            "action_dim": agent.action_dim,
            "chunk_action_dim": agent.chunk_action_dim,
        }
    )
    torch.save(agent.checkpoint_state(checkpoint_config), checkpoint_path)
    logger.log({"artifact/latest_checkpoint": str(checkpoint_path)}, step=total_steps)
    logger.finish()
    LOGGER.info("Q/V 训练完成，checkpoint=%s", checkpoint_path)
    return checkpoint_path


if __name__ == "__main__":
    main()
