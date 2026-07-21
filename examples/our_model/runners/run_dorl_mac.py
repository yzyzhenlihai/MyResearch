"""DORL-MAC 最小闭环 smoke 编排入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.our_model.runners.common import add_common_args, configure_logging, ensure_dir
from examples.our_model.runners.eval_dorl_mac import main as eval_main
from examples.our_model.runners.pretrain_categorical_bc import main as pretrain_main
from examples.our_model.runners.train_dorl_mac_qv import main as train_qv_main

LOGGER = logging.getLogger(__name__)

DEFAULT_SMOKE_STEPS = 2
"""smoke 阶段默认训练步数。"""

DEFAULT_SMOKE_BATCH_SIZE = 4
"""smoke 阶段默认 batch size。"""

DEFAULT_SMOKE_CHUNKS = 64
"""smoke 阶段最多使用的 chunk 数量。"""


def build_parser() -> argparse.ArgumentParser:
    """构造 smoke runner 参数解析器。

    Returns:
        argparse.ArgumentParser: 参数解析器。
    """

    parser = argparse.ArgumentParser(description="Run DORL-MAC smoke pipeline.")
    add_common_args(parser)
    parser.add_argument("--pretrain_steps", type=int, default=DEFAULT_SMOKE_STEPS)
    parser.add_argument("--train_steps", type=int, default=DEFAULT_SMOKE_STEPS)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument(
        "--step-per-epoch",
        "--step_per_epoch",
        dest="step_per_epoch",
        type=int,
        default=0,
    )
    parser.add_argument("--eval_episodes", type=int, default=0)
    parser.add_argument("--test-num", "--test_num", dest="test_num", type=int, default=1)
    parser.add_argument("--num_samples_train", type=int, default=2)
    parser.add_argument("--num_samples_test", type=int, default=2)
    parser.add_argument("--smoke_dir", type=str, default="")
    return parser


def build_common_argv(args: argparse.Namespace) -> List[str]:
    """构造传递给各阶段 runner 的公共参数。

    Args:
        args (argparse.Namespace): smoke runner 参数。

    Returns:
        List[str]: 公共命令行参数列表。
    """

    common_args = [
        "--env",
        args.env,
        "--user_model_name",
        args.user_model_name,
        "--read_message",
        args.read_message,
        "--dataset_path",
        args.dataset_path,
        "--which_tracker",
        args.which_tracker,
        "--reward_handle",
        args.reward_handle,
        "--window_size",
        str(args.window_size),
        "--chunk_size",
        str(args.chunk_size),
        "--gamma",
        str(args.gamma),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--cuda",
        str(args.cuda),
        "--batch_size",
        str(args.batch_size),
        "--max_trajectories",
        str(args.max_trajectories),
        "--max_chunks",
        str(args.max_chunks),
        "--num_leave_compute",
        str(args.num_leave_compute),
        "--leave_threshold",
        str(args.leave_threshold),
        "--max_turn",
        str(args.max_turn),
        "--force_length",
        str(args.force_length),
        "--invalid_action_penalty",
        str(args.invalid_action_penalty),
        "--lambda_entropy",
        str(args.lambda_entropy),
        "--lambda_variance",
        str(args.lambda_variance),
        "--dynamics_loss_weight",
        str(args.dynamics_loss_weight),
        "--swanlab_project",
        args.swanlab_project,
        "--run_name",
        args.run_name,
    ]
    common_args.append("--entropy_window")
    common_args.extend(str(window) for window in args.entropy_window)
    common_args.extend(
        [
            "--use_entropy_reward" if args.use_entropy_reward else "--no_entropy_reward",
            "--use_uncertainty_penalty"
            if args.use_uncertainty_penalty
            else "--no_uncertainty_penalty",
            "--feature_level" if args.feature_level else "--no_feature_level",
            "--is_sorted" if args.is_sorted else "--no_sorted",
        ]
    )
    if args.item_embedding_path:
        common_args.extend(["--item_embedding_path", args.item_embedding_path])
    if args.predicted_mat_path:
        common_args.extend(["--predicted_mat_path", args.predicted_mat_path])
    if args.maxvar_mat_path:
        common_args.extend(["--maxvar_mat_path", args.maxvar_mat_path])
    return common_args


def main(argv: Optional[list[str]] = None) -> None:
    """顺序执行预训练、Q/V 训练和评估。

    Args:
        argv (Optional[list[str]]): 可选命令行参数列表。

    Returns:
        None.
    """

    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_trajectories is None:
        args.max_trajectories = 4
    if args.max_chunks is None:
        args.max_chunks = DEFAULT_SMOKE_CHUNKS
    if args.batch_size == 64:
        args.batch_size = DEFAULT_SMOKE_BATCH_SIZE
    if args.epoch == 0 and args.step_per_epoch == 0:
        args.epoch = args.train_steps
        args.step_per_epoch = 1
    elif args.epoch <= 0 or args.step_per_epoch <= 0:
        raise ValueError("epoch and step_per_epoch must be positive when either is provided.")

    smoke_root = ensure_dir(
        args.smoke_dir
        or Path(args.save_root) / args.env / "DORL_MAC" / "smoke"
    )
    common_argv = build_common_argv(args)
    LOGGER.info("开始 DORL-MAC smoke pipeline，输出目录：%s", smoke_root)

    bc_dir = smoke_root / "categorical_bc"
    qv_dir = smoke_root / "mac_agent"
    eval_dir = smoke_root / "eval"
    bc_ckpt = pretrain_main(
        common_argv
        + [
            "--pretrain_steps",
            str(args.pretrain_steps),
            "--save_dir",
            str(bc_dir),
            "--log_interval",
            "1",
        ]
    )
    mac_ckpt = train_qv_main(
        common_argv
        + [
            "--bc_actor_ckpt",
            str(bc_ckpt),
            "--epoch",
            str(args.epoch),
            "--step-per-epoch",
            str(args.step_per_epoch),
            "--num_samples_train",
            str(args.num_samples_train),
            "--eval_episodes",
            str(args.eval_episodes),
            "--num_samples_test",
            str(args.num_samples_test),
            "--test-num",
            str(args.test_num),
            "--save_dir",
            str(qv_dir),
            "--log_interval",
            "1",
        ]
    )
    summary_path = eval_main(
        common_argv
        + [
            "--mac_ckpt",
            str(mac_ckpt),
            "--num_samples_test",
            str(args.num_samples_test),
            "--test-num",
            str(args.test_num),
            "--eval_episodes",
            str(args.eval_episodes),
            "--eval_save_dir",
            str(eval_dir),
        ]
    )
    LOGGER.info("DORL-MAC smoke pipeline 完成：summary=%s", summary_path)


if __name__ == "__main__":
    main()
