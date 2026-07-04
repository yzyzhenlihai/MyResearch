"""DORL-MAC action chunk actor 离线预训练入口。"""

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
from examples.our_model.models.mac_agent import ACTOR_BACKEND_FLOW, ACTOR_BACKEND_MLP_BC
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

DEFAULT_FLOW_STEPS = 10
"""flow actor Euler 采样默认步数。"""


def build_parser() -> argparse.ArgumentParser:
    """构造预训练参数解析器。

    Returns:
        argparse.ArgumentParser: 参数解析器。
    """

    parser = argparse.ArgumentParser(description="Pretrain DORL-MAC chunk actor.")
    add_common_args(parser)
    parser.add_argument("--actor_backend", choices=[ACTOR_BACKEND_FLOW, ACTOR_BACKEND_MLP_BC], default=ACTOR_BACKEND_FLOW)
    parser.add_argument("--pretrain_steps", type=int, default=DEFAULT_PRETRAIN_STEPS)
    parser.add_argument("--flow_steps", type=int, default=DEFAULT_FLOW_STEPS)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--bc_weight", type=float, default=1.0)
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
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    resolve_common_paths(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.device = str(device)

    save_dir = ensure_dir(
        args.save_dir
        or Path(args.save_root) / args.env / "DORL_MAC" / "flow_bc"
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
    agent = build_agent(args, device=device, action_mapper=action_mapper)
    optimizer = torch.optim.Adam(agent.actor_parameters(), lr=args.actor_lr)
    logger = SwanLabLogger(
        project=args.swanlab_project,
        run_name=default_run_name("flow-bc", args),
        config=config,
        log_path=str(save_dir / "metrics.jsonl"),
    )

    LOGGER.info("开始 DORL-MAC actor 预训练：steps=%s", args.pretrain_steps)
    batch_iterator = cycle_dataloader(dataloader)
    last_metrics = {}
    for step in range(1, args.pretrain_steps + 1):
        batch = next(batch_iterator)
        last_metrics = agent.pretrain_actor_update(
            batch=batch,
            optimizer=optimizer,
            actor_backend=args.actor_backend,
            flow_steps=args.flow_steps,
            bc_weight=args.bc_weight,
        )
        if step == 1 or step % args.log_interval == 0 or step == args.pretrain_steps:
            logger.log(last_metrics, step=step)
            LOGGER.info("pretrain step=%s metrics=%s", step, last_metrics)

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
    logger.log({"artifact/latest_checkpoint": str(checkpoint_path)}, step=args.pretrain_steps)
    logger.finish()
    LOGGER.info("actor 预训练完成，checkpoint=%s", checkpoint_path)
    return checkpoint_path


if __name__ == "__main__":
    main()
