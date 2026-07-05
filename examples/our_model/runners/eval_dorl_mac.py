"""DORL-MAC 推荐环境评估入口。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.our_model.logging import SwanLabLogger
from examples.our_model.policy import ActionMapper
from examples.our_model.runners.common import (
    add_common_args,
    build_agent,
    build_env_assets,
    build_reward_and_leave,
    configure_logging,
    default_run_name,
    ensure_dir,
    load_item_embeddings,
    namespace_to_dict,
    resolve_common_paths,
    resolve_device,
    save_resolved_config,
    set_mpl_cache_to_tmp,
    set_seed,
)
from examples.our_model.runners.evaluation_utils import build_dorl_mac_evaluator

LOGGER = logging.getLogger(__name__)

def build_parser() -> argparse.ArgumentParser:
    """构造评估参数解析器。

    Returns:
        argparse.ArgumentParser: 参数解析器。
    """

    parser = argparse.ArgumentParser(description="Evaluate DORL-MAC in DORL environments.")
    add_common_args(parser)
    parser.add_argument("--mac_ckpt", type=str, required=True)
    parser.add_argument("--num_samples_test", type=int, default=32)
    parser.add_argument("--test-num", "--test_num", dest="test_num", type=int, default=1)
    parser.add_argument("--eval_episodes", type=int, default=0)
    parser.add_argument("--eval_save_dir", type=str, default="")
    parser.add_argument("--buffer-size", "--buffer_size", dest="buffer_size", type=int, default=0)
    return parser


def main(argv: Optional[list[str]] = None) -> Path:
    """执行 DORL-MAC 环境评估。

    Args:
        argv (Optional[list[str]]): 可选命令行参数列表。

    Returns:
        Path: 评估 summary JSON 路径。

    Raises:
        FileNotFoundError: 当 checkpoint 不存在时抛出。
        ValueError: 当评估 episode 数非法时抛出。
    """

    configure_logging()
    set_mpl_cache_to_tmp()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.eval_episodes < 0:
        raise ValueError("eval_episodes must be non-negative.")
    mac_ckpt = Path(args.mac_ckpt)
    if not mac_ckpt.exists():
        raise FileNotFoundError(f"mac_ckpt does not exist: {mac_ckpt}")
    effective_eval_episodes = args.eval_episodes if args.eval_episodes > 0 else args.test_num
    if effective_eval_episodes <= 0:
        raise ValueError("test_num must be positive when eval_episodes is 0.")

    resolve_common_paths(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    save_dir = ensure_dir(
        args.eval_save_dir
        or Path(args.save_root) / args.env / "DORL_MAC" / "eval"
    )
    config = namespace_to_dict(args)
    save_resolved_config(save_dir, config)

    env, env_dataset, kwargs_um = build_env_assets(args)
    item_embeddings = load_item_embeddings(args.item_embedding_path)
    action_mapper = ActionMapper(item_embeddings=item_embeddings, device=device)
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
        reward_model=reward_model,
        leave_model=leave_model,
    )
    checkpoint = torch.load(mac_ckpt, map_location=device)
    agent.load_checkpoint_state(checkpoint, strict=False)
    agent.eval()
    evaluator = build_dorl_mac_evaluator(
        args=args,
        env=env,
        dataset=env_dataset,
        kwargs_um=kwargs_um,
        agent=agent,
        action_mapper=action_mapper,
        num_samples_test=args.num_samples_test,
        device=device,
        eval_episodes=effective_eval_episodes,
        save_dir=save_dir,
        buffer_size=args.buffer_size,
    )
    summary = evaluator.evaluate(epoch=0, global_step=0)
    summary_path = save_dir / "summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)
    logger = SwanLabLogger(
        project=args.swanlab_project,
        run_name=default_run_name("eval", args),
        config=config,
        log_path=str(save_dir / "metrics.jsonl"),
    )
    logger.log({key: value for key, value in summary.items() if isinstance(value, (float, int))}, step=0)
    logger.finish()
    LOGGER.info("DORL-MAC 评估完成，summary=%s", summary_path)
    return summary_path


if __name__ == "__main__":
    main()
