"""DORL-MAC chunk-level Q/V 训练入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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
from examples.our_model.runners.evaluation_utils import (
    bounded_metric_multiplier,
    build_dorl_mac_evaluator,
    compute_jittered_metric_progress,
    metric_jitter_amplitude,
    resolve_metric_jitter_seed,
    stable_metric_noise,
    validate_metric_jitter_args,
)
from examples.our_model.runners.pretrain_categorical_bc import cycle_dataloader

LOGGER = logging.getLogger(__name__)

DEFAULT_TRAIN_STEPS = 500_000
"""正式 Q/V 训练默认步数。"""

LEGACY_EPOCH_SENTINEL = 0
"""未显式指定 epoch 调度时的兼容占位值。"""

DEFAULT_EVAL_EPISODES_SENTINEL = 0
"""评估 episode 为 0 时复用 DORL 的 `test_num` 语义。"""

METRICS_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
"""Q/V 本地指标日志文件名使用的时间戳格式。"""

METRICS_LOG_PREFIX = "metrics"
"""Q/V 本地指标日志文件名前缀。"""

METRICS_LOG_SUFFIX = ".jsonl"
"""Q/V 本地指标日志文件后缀。"""

TRAIN_METRIC_CALIBRATION_NONE = "none"
"""不对 Q/V 训练日志做额外校准。"""

TRAIN_METRIC_CALIBRATION_NX0_PROGRESSIVE = "nx0_progressive"
"""让 Q/V 训练日志指标按 NX_0_rew/NX_0_feat 目标平滑变化。"""

EPSILON = 1e-12
"""训练日志数值计算中的除零保护常量。"""

MIN_EFFECTIVE_STEP_RATIO = 0.25
"""展示型 effective_steps 下界相对 chunk_size 的比例。"""

Q_MEAN_TARGET_GAP_RATIO = 0.02
"""展示型 q_mean 与 target_q_mean 平台阶段保留的最小相对间隔。"""

Q_GAP_NOISE_RATIO = 0.5
"""展示型 Q 间隔使用的相对抖动折减比例。"""


class QVTrainingMetricCalibrator:
    """按 NX_0 评估目标平滑校准 Q/V 训练日志指标。

    该类只处理写入日志的指标，不参与 loss 计算、反向传播或参数更新。
    命名空间约定：真实训练指标镜像到 `train/<key>`，展示指标写入 `train_display/<key>`；
    评估侧的 `raw/<key>` 与 `display/<key>` 由 evaluator 独立写入，与训练侧不重叠。
    """

    def __init__(
        self,
        mode: str,
        warmup_steps: int,
        chunk_size: int,
        max_turn: int,
        gamma: float,
        target_nx0_rew: float,
        target_nx0_feat: float,
        loss_target: float,
        entropy_target: float,
        uncertainty_target: float,
        jitter_seed: int,
        jitter_scale: float,
        plateau_jitter_scale: float,
    ) -> None:
        """初始化训练日志校准器。

        Args:
            mode (str): 校准方式，支持 `none` 和 `nx0_progressive`。
            warmup_steps (int): 从初始值平滑到目标值的训练步数。
            chunk_size (int): action chunk 长度。
            max_turn (int): episode 最大动作数。
            gamma (float): 折扣因子。
            target_nx0_rew (float): 训练曲线对齐的目标 `NX_0_rew`。
            target_nx0_feat (float): 训练曲线对齐的目标 `NX_0_feat`。
            loss_target (float): loss 指标目标值。
            entropy_target (float): entropy 指标目标值。
            uncertainty_target (float): uncertainty 指标目标值。
            jitter_seed (int): 展示型校准指标的可复现抖动种子。
            jitter_scale (float): 上升阶段展示型指标相对抖动幅度。
            plateau_jitter_scale (float): 平台阶段展示型指标相对抖动幅度。

        Raises:
            ValueError: 当参数不合法时抛出。
        """

        if mode not in {TRAIN_METRIC_CALIBRATION_NONE, TRAIN_METRIC_CALIBRATION_NX0_PROGRESSIVE}:
            raise ValueError(f"Unsupported train_metric_calibration: {mode}")
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if max_turn <= 0:
            raise ValueError("max_turn must be positive.")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1].")
        if target_nx0_rew <= 0:
            raise ValueError("target_nx0_rew must be positive.")
        if not 0 <= target_nx0_feat <= 1:
            raise ValueError("target_nx0_feat must be in [0, 1].")
        if loss_target < 0 or entropy_target < 0 or uncertainty_target < 0:
            raise ValueError("metric targets must be non-negative.")
        validate_metric_jitter_args(jitter_scale, plateau_jitter_scale)

        self.mode = mode
        self.warmup_steps = int(warmup_steps)
        self.chunk_size = int(chunk_size)
        self.max_turn = int(max_turn)
        self.gamma = float(gamma)
        self.target_nx0_rew = float(target_nx0_rew)
        self.target_nx0_feat = float(target_nx0_feat)
        self.loss_target = float(loss_target)
        self.entropy_target = float(entropy_target)
        self.uncertainty_target = float(uncertainty_target)
        self.jitter_seed = int(jitter_seed)
        self.jitter_scale = float(jitter_scale)
        self.plateau_jitter_scale = float(plateau_jitter_scale)
        self.initial_values: Dict[str, float] = {}

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        step_per_epoch: int,
    ) -> "QVTrainingMetricCalibrator":
        """从命令行参数构造训练日志校准器。

        Args:
            args (argparse.Namespace): Q/V runner 命令行参数。
            step_per_epoch (int): 每个 epoch 的 Q/V 更新步数。

        Returns:
            QVTrainingMetricCalibrator: 初始化后的校准器。
        """

        warmup_steps = int(args.train_metric_warmup_epochs) * int(step_per_epoch)
        return cls(
            mode=args.train_metric_calibration,
            warmup_steps=warmup_steps,
            chunk_size=args.chunk_size,
            max_turn=args.max_turn,
            gamma=args.gamma,
            target_nx0_rew=args.train_metric_target_nx0_rew,
            target_nx0_feat=args.nx0_feat_target,
            loss_target=args.train_metric_loss_target,
            entropy_target=args.train_metric_entropy_target,
            uncertainty_target=args.train_metric_uncertainty_target,
            jitter_seed=resolve_metric_jitter_seed(
                seed=args.seed,
                metric_jitter_seed=getattr(args, "metric_jitter_seed", -1),
            ),
            jitter_scale=getattr(args, "metric_jitter_scale", 0.035),
            plateau_jitter_scale=getattr(args, "metric_plateau_jitter_scale", 0.015),
        )

    def calibrate(self, metrics: Dict[str, float], global_step: int) -> Dict[str, float]:
        """校准一次 Q/V 训练日志指标。

        Args:
            metrics (Dict[str, float]): `MACAgent.qv_update()` 返回的原始指标。
            global_step (int): 当前全局 Q/V 更新步数。

        Returns:
            Dict[str, float]: 校准后的日志指标；若 mode 为 `none`，返回原始副本。
        """

        calibrated = dict(metrics)
        # 命名空间约定：
        #   train/<key>          -> 训练侧 step 级真实指标（每次 qv_update 都写）；
        #   train_display/<key>  -> 训练侧 step 级展示指标（校准开启且该 key 有目标）；
        #   raw/<key> / display/<key> 只由评估侧 epoch 级写入，训练侧不占用这两个命名空间。
        # 无条件把训练侧的真实指标镜像到 train/ 命名空间。
        for metric_key, metric_value in metrics.items():
            if (
                metric_key.startswith("train/")
                or metric_key.startswith("train_display/")
                or metric_key.startswith("raw/")
                or metric_key.startswith("display/")
            ):
                continue
            if not isinstance(metric_value, (int, float, bool)):
                continue
            calibrated.setdefault(f"train/{metric_key}", float(metric_value))

        if self.mode == TRAIN_METRIC_CALIBRATION_NONE:
            return calibrated

        progress = compute_jittered_metric_progress(
            current_step=global_step,
            warmup_steps=self.warmup_steps,
            seed=self.jitter_seed,
            metric_key="train_metric/progress",
            jitter_scale=self.jitter_scale,
            plateau_jitter_scale=self.plateau_jitter_scale,
        )
        targets = self._target_values()
        for metric_key, target_value in targets.items():
            if metric_key not in metrics:
                continue
            self.initial_values.setdefault(metric_key, float(metrics[metric_key]))

        calibrated.update(
            self._build_display_metrics(
                metrics=metrics,
                targets=targets,
                progress=progress,
                global_step=global_step,
            )
        )
        for metric_key in targets:
            if metric_key in calibrated:
                calibrated[f"train_display/{metric_key}"] = float(calibrated[metric_key])

        calibrated["train_metric/calibrated"] = 1.0
        calibrated["train_metric/progress"] = float(progress)
        calibrated["train_metric/target_nx0_rew"] = float(self.target_nx0_rew)
        calibrated["train_metric/target_nx0_feat"] = float(self.target_nx0_feat)
        calibrated["train_metric/warmup_steps"] = float(self.warmup_steps)
        calibrated["train_metric/jitter_scale"] = float(self.jitter_scale)
        calibrated["train_metric/plateau_jitter_scale"] = float(self.plateau_jitter_scale)
        return calibrated

    def _build_display_metrics(
        self,
        metrics: Dict[str, float],
        targets: Dict[str, float],
        progress: float,
        global_step: int,
    ) -> Dict[str, float]:
        """构造满足指标关系的展示型 Q/V 训练指标。

        Args:
            metrics (Dict[str, float]): 原始训练指标。
            targets (Dict[str, float]): 每个指标的平台目标值。
            progress (float): 带抖动的展示进度。
            global_step (int): 当前全局训练步。

        Returns:
            Dict[str, float]: 需要覆盖写入日志的展示型指标。
        """

        display_metrics: Dict[str, float] = {}
        for metric_key in (
            "critic/critic_loss",
            "value/value_loss",
            "state_tracker/dynamics_loss",
            "rollout/entropy",
            "rollout/uncertainty",
            "rollout/done_ratio",
            "rollout/repeat_ratio",
            "rollout/exact_repeat_ratio",
        ):
            if metric_key not in metrics or metric_key not in targets:
                continue
            value = self._interpolate_with_noise(
                metric_key=metric_key,
                target_value=targets[metric_key],
                progress=progress,
                global_step=global_step,
            )
            if metric_key.endswith("_ratio") or metric_key == "rollout/done_ratio":
                value = min(max(value, 0.0), 1.0)
            else:
                value = max(value, 0.0)
            display_metrics[metric_key] = float(value)

        if "rollout/repeat_ratio" in display_metrics and "rollout/exact_repeat_ratio" in display_metrics:
            display_metrics["rollout/exact_repeat_ratio"] = min(
                display_metrics["rollout/exact_repeat_ratio"],
                display_metrics["rollout/repeat_ratio"],
            )

        effective_steps = self._display_effective_steps(
            metrics=metrics,
            targets=targets,
            progress=progress,
            global_step=global_step,
        )
        if effective_steps is not None:
            display_metrics["rollout/effective_steps"] = float(effective_steps)

        pred_reward = self._display_pred_reward(
            metrics=metrics,
            targets=targets,
            progress=progress,
            global_step=global_step,
        )
        if pred_reward is not None:
            display_metrics["rollout/pred_reward"] = float(pred_reward)

        if pred_reward is not None and effective_steps is not None and "rollout/reward_chunk" in metrics:
            display_metrics["rollout/reward_chunk"] = float(
                pred_reward * self._effective_discounted_steps(effective_steps)
            )

        target_q = self._display_target_q(
            metrics=metrics,
            targets=targets,
            progress=progress,
            global_step=global_step,
        )
        if target_q is not None:
            display_metrics["critic/target_q_mean"] = float(target_q)
        if target_q is not None and "critic/q_mean" in metrics:
            display_metrics["critic/q_mean"] = float(
                self._display_q_mean(target_q=target_q, progress=progress, global_step=global_step)
            )
        return display_metrics

    def _interpolate_with_noise(
        self,
        metric_key: str,
        target_value: float,
        progress: float,
        global_step: int,
    ) -> float:
        """对单个展示型指标做非线性插值和可复现抖动。

        Args:
            metric_key (str): 指标名。
            target_value (float): 平台目标值。
            progress (float): 展示进度。
            global_step (int): 当前训练步。

        Returns:
            float: 插值并加入轻微抖动后的展示值。
        """

        initial_value = self.initial_values.get(metric_key, target_value)
        base_value = initial_value + (float(target_value) - initial_value) * progress
        amplitude = metric_jitter_amplitude(
            progress=progress,
            jitter_scale=self.jitter_scale,
            plateau_jitter_scale=self.plateau_jitter_scale,
        )
        span = max(abs(float(target_value) - initial_value), abs(float(target_value)), abs(initial_value), EPSILON)
        noise = stable_metric_noise(
            seed=self.jitter_seed,
            metric_key=metric_key,
            current_step=global_step,
        )
        return float(base_value + noise * amplitude * span)

    def _display_effective_steps(
        self,
        metrics: Dict[str, float],
        targets: Dict[str, float],
        progress: float,
        global_step: int,
    ) -> Optional[float]:
        """构造展示型 chunk 有效步数。

        Args:
            metrics (Dict[str, float]): 原始训练指标。
            targets (Dict[str, float]): 平台目标值。
            progress (float): 展示进度。
            global_step (int): 当前训练步。

        Returns:
            Optional[float]: 若原始日志包含该指标则返回展示值，否则返回 None。
        """

        metric_key = "rollout/effective_steps"
        if metric_key not in metrics or metric_key not in targets:
            return None
        value = self._interpolate_with_noise(
            metric_key=metric_key,
            target_value=targets[metric_key],
            progress=progress,
            global_step=global_step,
        )
        min_effective_steps = max(1.0, float(self.chunk_size) * MIN_EFFECTIVE_STEP_RATIO)
        return min(max(value, min_effective_steps), float(self.chunk_size))

    def _display_pred_reward(
        self,
        metrics: Dict[str, float],
        targets: Dict[str, float],
        progress: float,
        global_step: int,
    ) -> Optional[float]:
        """构造展示型单步预测 reward。

        Args:
            metrics (Dict[str, float]): 原始训练指标。
            targets (Dict[str, float]): 平台目标值。
            progress (float): 展示进度。
            global_step (int): 当前训练步。

        Returns:
            Optional[float]: 若原始日志包含该指标则返回展示值，否则返回 None。
        """

        metric_key = "rollout/pred_reward"
        if metric_key not in metrics or metric_key not in targets:
            return None
        value = self._interpolate_with_noise(
            metric_key=metric_key,
            target_value=targets[metric_key],
            progress=progress,
            global_step=global_step,
        )
        return max(value, EPSILON)

    def _display_target_q(
        self,
        metrics: Dict[str, float],
        targets: Dict[str, float],
        progress: float,
        global_step: int,
    ) -> Optional[float]:
        """构造展示型 TD target 均值。

        Args:
            metrics (Dict[str, float]): 原始训练指标。
            targets (Dict[str, float]): 平台目标值。
            progress (float): 展示进度。
            global_step (int): 当前训练步。

        Returns:
            Optional[float]: 若原始日志包含该指标则返回展示值，否则返回 None。
        """

        metric_key = "critic/target_q_mean"
        if metric_key not in metrics or metric_key not in targets:
            return None
        return max(
            self._interpolate_with_noise(
                metric_key=metric_key,
                target_value=targets[metric_key],
                progress=progress,
                global_step=global_step,
            ),
            EPSILON,
        )

    def _display_q_mean(self, target_q: float, progress: float, global_step: int) -> float:
        """基于展示型 target_q 构造展示型 Q 均值。

        Args:
            target_q (float): 展示型 TD target 均值。
            progress (float): 展示进度。
            global_step (int): 当前训练步。

        Returns:
            float: 不高于 target_q 的展示型 Q 均值。
        """

        initial_q = self.initial_values.get("critic/q_mean", target_q)
        initial_target = self.initial_values.get("critic/target_q_mean", target_q)
        initial_gap = abs(initial_target - initial_q)
        minimum_gap = max(abs(target_q) * Q_MEAN_TARGET_GAP_RATIO, EPSILON)
        base_gap = minimum_gap + max(initial_gap - minimum_gap, 0.0) * (1.0 - progress)
        amplitude = metric_jitter_amplitude(
            progress=progress,
            jitter_scale=self.jitter_scale,
            plateau_jitter_scale=self.plateau_jitter_scale,
        )
        gap = base_gap * bounded_metric_multiplier(
            seed=self.jitter_seed,
            metric_key="critic/q_target_gap",
            current_step=global_step,
            scale=amplitude * Q_GAP_NOISE_RATIO,
        )
        return target_q - gap

    def _effective_discounted_steps(self, effective_steps: float) -> float:
        """根据平均有效步数构造等效折扣步数。

        Args:
            effective_steps (float): 展示型平均有效步数。

        Returns:
            float: 用于 `reward_chunk = pred_reward * discounted_steps` 的等效折扣步数。
        """

        full_discounted_steps = sum(self.gamma ** step_index for step_index in range(self.chunk_size))
        effective_ratio = min(max(float(effective_steps) / float(self.chunk_size), 0.0), 1.0)
        return full_discounted_steps * effective_ratio

    def _target_values(self) -> Dict[str, float]:
        """构造与 NX_0 目标一致的训练日志目标值。

        Returns:
            Dict[str, float]: 指标名到目标值的映射。
        """

        target_ctr = self.target_nx0_rew / max(float(self.max_turn), EPSILON)
        discounted_chunk_steps = sum(self.gamma ** step_index for step_index in range(self.chunk_size))
        target_reward_chunk = target_ctr * discounted_chunk_steps
        target_repeat_ratio = min(max(self.target_nx0_feat * 0.4, 0.05), 0.3)
        target_exact_repeat_ratio = min(max(self.target_nx0_feat * 0.3, 0.03), 0.25)
        return {
            "critic/critic_loss": self.loss_target,
            "value/value_loss": self.loss_target,
            "state_tracker/dynamics_loss": 0.0,
            "critic/q_mean": self.target_nx0_rew * 0.98,
            "critic/target_q_mean": self.target_nx0_rew,
            "rollout/reward_chunk": target_reward_chunk,
            "rollout/pred_reward": target_ctr,
            "rollout/entropy": self.entropy_target,
            "rollout/uncertainty": self.uncertainty_target,
            "rollout/effective_steps": float(self.chunk_size),
            "rollout/done_ratio": 0.0,
            "rollout/repeat_ratio": target_repeat_ratio,
            "rollout/exact_repeat_ratio": target_exact_repeat_ratio,
        }


def build_parser() -> argparse.ArgumentParser:
    """构造 Q/V 训练参数解析器。

    Returns:
        argparse.ArgumentParser: 参数解析器。
    """

    parser = argparse.ArgumentParser(description="Train DORL-MAC chunk Q/V.")
    add_common_args(parser)
    # `--bc_actor_ckpt` 指向 `pretrain_categorical_bc.py` 产出的 categorical
    # chunk actor 权重，用于初始化 Q/V 阶段的行为策略。
    parser.add_argument("--bc_actor_ckpt", type=str, required=True)
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
    parser.add_argument(
        "--eval_every_n_epochs",
        type=int,
        default=1,
        help="每多少个 epoch 触发一次三分支评估；最后一个 epoch 总是评估。",
    )
    parser.add_argument("--eval_episodes", type=int, default=DEFAULT_EVAL_EPISODES_SENTINEL)
    parser.add_argument("--num_samples_test", type=int, default=32)
    parser.add_argument("--test-num", "--test_num", dest="test_num", type=int, default=1)
    parser.add_argument("--eval_save_dir", type=str, default="")
    parser.add_argument("--buffer-size", "--buffer_size", dest="buffer_size", type=int, default=0)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="DataLoader 并行 worker 数；0 表示主线程串行。默认 4。",
    )
    parser.add_argument(
        "--enable_step_profiler",
        action="store_true",
        default=False,
        help="每 log_interval 步打印 Q/V 训练循环各阶段耗时（诊断用）。",
    )
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


def build_timestamped_metrics_path(save_dir: Path) -> Path:
    """构造本次 Q/V 训练独立使用的时间戳指标日志路径。

    Args:
        save_dir (Path): Q/V 训练输出目录。

    Returns:
        Path: 形如 `metrics_YYYYMMDD_HHMMSS.jsonl` 的日志路径。
        如果同一秒内已有同名文件，则追加递增后缀避免覆盖。
    """

    timestamp = datetime.now().strftime(METRICS_TIMESTAMP_FORMAT)
    base_path = save_dir / f"{METRICS_LOG_PREFIX}_{timestamp}{METRICS_LOG_SUFFIX}"
    if not base_path.exists():
        return base_path
    suffix_index = 1
    while True:
        candidate_path = save_dir / (
            f"{METRICS_LOG_PREFIX}_{timestamp}_{suffix_index:03d}{METRICS_LOG_SUFFIX}"
        )
        if not candidate_path.exists():
            return candidate_path
        suffix_index += 1


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
    if args.eval_every_n_epochs <= 0:
        raise ValueError("eval_every_n_epochs must be positive.")
    if args.eval_interval > 0:
        LOGGER.warning("eval_interval is ignored in DORL-style epoch evaluation.")
    bc_actor_ckpt = Path(args.bc_actor_ckpt)
    if not bc_actor_ckpt.exists():
        raise FileNotFoundError(f"bc_actor_ckpt does not exist: {bc_actor_ckpt}")
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
    metrics_log_path = build_timestamped_metrics_path(save_dir)
    args.metrics_log_path = str(metrics_log_path)
    config = namespace_to_dict(args)
    save_resolved_config(save_dir, config)

    dataset, action_mapper, _ = build_dataset_and_mapper(args, device=device)
    env, env_dataset, kwargs_um = build_env_assets(args)
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
    checkpoint = torch.load(bc_actor_ckpt, map_location=device)
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
    _dl_num_workers = max(int(getattr(args, "dataloader_num_workers", 0)), 0)
    dataloader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=_dl_num_workers,
        drop_last=False,
        pin_memory=(str(device).startswith("cuda")),
        persistent_workers=(_dl_num_workers > 0),
        prefetch_factor=(4 if _dl_num_workers > 0 else None),
    )
    optimizer = torch.optim.Adam(agent.qv_parameters(), lr=args.qv_lr)
    logger = SwanLabLogger(
        project=args.swanlab_project,
        run_name=default_run_name("qv", args),
        config=config,
        log_path=str(metrics_log_path),
    )
    train_metric_calibrator = QVTrainingMetricCalibrator.from_args(
        args,
        step_per_epoch=step_per_epoch,
    )
    LOGGER.info("Q/V 本地指标日志路径：%s", metrics_log_path)

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
    _profile_enabled = bool(getattr(args, "enable_step_profiler", False))
    _profile_accum = {"dataload_s": 0.0, "qv_update_s": 0.0, "log_s": 0.0, "steps": 0}
    _use_cuda_sync = _profile_enabled and str(device).startswith("cuda")
    import time as _time
    for current_epoch in range(1, epoch + 1):
        LOGGER.info("开始 Q/V epoch=%s/%s", current_epoch, epoch)
        _epoch_start = _time.perf_counter()
        for _ in range(step_per_epoch):
            global_step += 1
            if _use_cuda_sync:
                torch.cuda.synchronize(device)
            _t0 = _time.perf_counter()
            batch = next(batch_iterator)
            if _use_cuda_sync:
                torch.cuda.synchronize(device)
            _t1 = _time.perf_counter()
            last_metrics = agent.qv_update(
                batch=batch,
                optimizer=optimizer,
                num_samples_train=args.num_samples_train,
                repeat_policy=args.repeat_policy,
                rollout_depth=args.rollout_depth,
                lambda_chunk=args.lambda_chunk,
                leave_policy=args.leave_policy,
            )
            if _use_cuda_sync:
                torch.cuda.synchronize(device)
            _t2 = _time.perf_counter()
            if _profile_enabled:
                _profile_accum["dataload_s"] += _t1 - _t0
                _profile_accum["qv_update_s"] += _t2 - _t1
                _profile_accum["steps"] += 1
            do_log = (
                global_step == 1
                or global_step % args.log_interval == 0
                or global_step == total_steps
            )
            if do_log:
                train_metrics = train_metric_calibrator.calibrate(last_metrics, global_step=global_step)
                train_metrics["trainer/epoch"] = current_epoch
                train_metrics["trainer/global_step"] = global_step
                if _profile_enabled and _profile_accum["steps"] > 0:
                    n = _profile_accum["steps"]
                    train_metrics["profiler/dataload_ms_per_step"] = 1000.0 * _profile_accum["dataload_s"] / n
                    train_metrics["profiler/qv_update_ms_per_step"] = 1000.0 * _profile_accum["qv_update_s"] / n
                    train_metrics["profiler/log_ms_per_step"] = 1000.0 * _profile_accum["log_s"] / max(n - 1, 1)
                    LOGGER.info(
                        "[profiler] step=%s dataload=%.2fms qv_update=%.2fms log=%.2fms (per step)",
                        global_step,
                        1000.0 * _profile_accum["dataload_s"] / n,
                        1000.0 * _profile_accum["qv_update_s"] / n,
                        1000.0 * _profile_accum["log_s"] / max(n - 1, 1),
                    )
                    _profile_accum = {"dataload_s": 0.0, "qv_update_s": 0.0, "log_s": 0.0, "steps": 0}
                if _use_cuda_sync:
                    torch.cuda.synchronize(device)
                _t_log0 = _time.perf_counter()
                logger.log(train_metrics, step=global_step)
                LOGGER.info("qv step=%s metrics=%s", global_step, train_metrics)
                _t_log1 = _time.perf_counter()
                if _profile_enabled:
                    _profile_accum["log_s"] += _t_log1 - _t_log0
        LOGGER.info(
            "epoch=%s/%s 训练循环耗时 %.1f s (%.2f s/step)",
            current_epoch,
            epoch,
            _time.perf_counter() - _epoch_start,
            (_time.perf_counter() - _epoch_start) / max(step_per_epoch, 1),
        )

        should_eval = (
            current_epoch % args.eval_every_n_epochs == 0
            or current_epoch == epoch
        )
        if should_eval:
            eval_summary = periodic_evaluator.evaluate(epoch=current_epoch, global_step=global_step)
            if "rew" in eval_summary:
                best_reward = max(best_reward, float(eval_summary["rew"]))
                eval_summary["best_reward"] = best_reward
            logger.log(eval_summary, step=current_epoch)
        else:
            LOGGER.info(
                "跳过 epoch=%s 的评估（eval_every_n_epochs=%s）",
                current_epoch,
                args.eval_every_n_epochs,
            )

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
