"""DORL-MAC 离散 Categorical chunk actor 离线预训练入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.our_model.logging import SwanLabLogger
from examples.our_model.runners.common import (
    add_common_args,
    build_agent,
    build_dataset_and_mapper,
    configure_logging,
    default_run_name,
    ensure_dir,
    namespace_to_dict,
    resolve_common_paths,
    resolve_device,
    save_resolved_config,
    set_seed,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_PRETRAIN_STEPS = 200_000
"""正式 actor 预训练默认步数。"""

DEFAULT_BC_SUBDIR = "categorical_bc"
"""actor checkpoint 默认保存子目录。"""


def build_parser() -> argparse.ArgumentParser:
    """构造预训练参数解析器。

    Returns:
        argparse.ArgumentParser: 参数解析器。
    """

    parser = argparse.ArgumentParser(description="Pretrain DORL-MAC discrete categorical chunk actor.")
    add_common_args(parser)
    parser.add_argument("--pretrain_steps", type=int, default=DEFAULT_PRETRAIN_STEPS)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--log_interval", type=int, default=100)
    return parser


def cycle_dataloader(dataloader: DataLoader) -> Iterable[dict]:
    """无限循环 DataLoader。

    Args:
        dataloader (DataLoader): PyTorch 数据加载器。

    Returns:
        Iterable[dict]: 无限 batch 迭代器。
    """

    while True:
        yield from dataloader


def main(argv: Optional[list[str]] = None) -> Path:
    """执行 actor 预训练。

    Args:
        argv (Optional[list[str]]): 可选命令行参数列表，便于 smoke runner 复用。

    Returns:
        Path: 最终 checkpoint 路径。

    Raises:
        ValueError: 当训练步数或 batch size 非法时抛出。
    """

    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pretrain_steps <= 0:
        raise ValueError("pretrain_steps must be positive.")
    if args.dynamics_pretrain_steps <= 0:
        raise ValueError("dynamics_pretrain_steps must be positive for MAC_origin.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    resolve_common_paths(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.device = str(device)

    save_dir = ensure_dir(
        args.save_dir
        or Path(args.save_root) / args.env / "MAC_origin" / DEFAULT_BC_SUBDIR
    )
    config = namespace_to_dict(args)
    save_resolved_config(save_dir, config)
    dataset, action_mapper, _ = build_dataset_and_mapper(args, device=device)
    dataloader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    agent = build_agent(
        args,
        device=device,
        action_mapper=action_mapper,
        state_dim=dataset.state_dim,
    )
    optimizer = torch.optim.Adam(agent.actor_parameters(), lr=args.actor_lr)
    logger = SwanLabLogger(
        project=args.swanlab_project,
        run_name=default_run_name("categorical-bc", args),
        config=config,
        log_path=str(save_dir / "metrics.jsonl"),
    )

    dynamics_optimizer = torch.optim.Adam(agent.dynamics.parameters(), lr=args.dynamics_lr)
    batch_iterator = cycle_dataloader(dataloader)
    LOGGER.info("开始直接 chunk dynamics 预训练：steps=%s", args.dynamics_pretrain_steps)
    for dynamics_step in range(1, args.dynamics_pretrain_steps + 1):
        metrics = agent.dynamics_update(next(batch_iterator), dynamics_optimizer)
        if dynamics_step == 1 or dynamics_step % args.log_interval == 0 or dynamics_step == args.dynamics_pretrain_steps:
            logger.log(metrics, step=dynamics_step)
            LOGGER.info("dynamics step=%s metrics=%s", dynamics_step, metrics)

    LOGGER.info("开始 DORL-MAC categorical BC 预训练：steps=%s", args.pretrain_steps)
    batch_iterator = cycle_dataloader(dataloader)
    last_metrics = {}
    for step in range(1, args.pretrain_steps + 1):
        batch = next(batch_iterator)
        last_metrics = agent.pretrain_actor_update(
            batch=batch,
            optimizer=optimizer,
        )
        if step == 1 or step % args.log_interval == 0 or step == args.pretrain_steps:
            logger.log(last_metrics, step=args.dynamics_pretrain_steps + step)
            LOGGER.info("pretrain step=%s metrics=%s", step, last_metrics)

    checkpoint_path = save_dir / "latest.pt"
    checkpoint_config = dict(config)
    checkpoint_config.update(
        {
            "state_dim": agent.state_dim,
            "action_dim": agent.action_dim,
            "chunk_action_dim": agent.chunk_action_dim,
            "num_items": agent.num_items,
        }
    )
    torch.save(agent.checkpoint_state(checkpoint_config), checkpoint_path)
    logger.log({"artifact/latest_checkpoint": str(checkpoint_path)}, step=args.dynamics_pretrain_steps + args.pretrain_steps)
    logger.finish()
    LOGGER.info("actor 预训练完成，checkpoint=%s", checkpoint_path)
    return checkpoint_path


if __name__ == "__main__":
    main()
