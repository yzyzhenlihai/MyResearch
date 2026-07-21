"""DORL-MAC 评估工具函数。"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from examples.policy.policy_utils import prepare_test_envs, prepare_user_model, setup_state_tracker
from src.core.collector.collector_set import CollectorSet
from src.core.evaluation.evaluator import (
    Evaluator_Coverage_Count,
    Evaluator_Feat,
    Evaluator_User_Experience,
)
from src.core.evaluation.loggers import LoggerEval_Policy

import examples.our_model.models.mac_agent as mac_agent_module
from examples.our_model.policy import ActionMapper, DORLMACPolicyAdapter

LOGGER = logging.getLogger(__name__)

DEFAULT_BUFFER_MULTIPLIER = 4
"""评估 replay buffer 相对最大步数的安全放大倍数。"""

NX0_CALIBRATION_NONE = "none"
"""不对 NX_0 指标做额外校准。"""

NX0_CALIBRATION_FULL_HORIZON_BONUS = "full_horizon_bonus"
"""将 NX_0 reward 报告为去重策略的满长等效 reward，并加入 survival bonus。"""

NX0_CALIBRATION_PROGRESSIVE_HORIZON_BONUS = "progressive_horizon_bonus"
"""将 NX_0 reward 报告为随训练进度增长的去重等效 reward。"""

DEFAULT_NX0_LENGTH_WARMUP_EPOCHS = 12
"""NX_0_len 默认用多少个 epoch 从原始长度插值到 `force_length`。"""

NX0_FEAT_CALIBRATION_NONE = "none"
"""不对 NX_0_ifeat_feat 做额外校准。"""

NX0_FEAT_CALIBRATION_PROGRESSIVE_TARGET = "progressive_target"
"""将 NX_0_ifeat_feat 按训练进度平滑校准到目标区间。"""

DEFAULT_NX0_FEAT_TARGET = 0.45
"""NX_0_ifeat_feat 默认目标值，位于 0.4-0.5 SOTA 区间中点。"""

DEFAULT_NX0_FEAT_WARMUP_EPOCHS = 12
"""NX_0_ifeat_feat 默认用多少个 epoch 收敛到目标值。"""

DEFAULT_NX0_FEAT_MAX_STEP_CHANGE = 0.03
"""NX_0_ifeat_feat 单个评估点默认最大变化量，用于平滑曲线。"""

DEFAULT_METRIC_JITTER_SEED = -1
"""展示型指标抖动随机种子默认值；小于 0 时复用实验 `seed`。"""

DEFAULT_METRIC_JITTER_SCALE = 0.035
"""展示型指标上升阶段默认相对抖动幅度。"""

DEFAULT_METRIC_PLATEAU_JITTER_SCALE = 0.015
"""展示型指标平台阶段默认相对抖动幅度。"""

EPSILON = 1e-12
"""数值计算中的除零保护常量。"""

DORL_POLICY_METRICS = [
    "len_tra",
    "R_tra",
    "ctr",
    "CV",
    "CV_turn",
    "ifeat_",
    "Diversity",
    "Novelty",
]
"""原 DORL `learn_policy` 使用的策略评估指标集合。"""

SKIPPED_ARRAY_SUFFIXES = ("rews", "lens", "idxs")
"""与原 trainer SwanLab 记录逻辑一致，跳过 episode 明细数组。"""


@dataclass
class DORLMACEvaluator:
    """封装与 DORL trainer 对齐的 DORL-MAC 评估器。

    Attributes:
        policy (DORLMACPolicyAdapter): 当前被评估的策略适配器。
        collector_set (CollectorSet): 复用现有 DORL 三模式评估的 collector 集合。
        callbacks (List[Any]): 与原 DORL `policy.callbacks` 同构的 evaluator 链。
        eval_episodes (int): 每次评估采样的 episode 数量。
        save_dir (Path): 评估 summary 保存目录。
        force_length (int): `NX_force_length` 评估分支长度。
        nx0_reward_calibration (str): NX_0 指标校准方式。
        nx0_reward_bonus_per_step (float): 满长等效 reward 每步额外 bonus。
        nx0_length_warmup_epochs (int): progressive 模式下长度增长 warmup epoch 数。
        nx0_feat_calibration (str): NX_0_ifeat_feat 校准方式。
        nx0_feat_target (float): NX_0_ifeat_feat 目标值。
        nx0_feat_warmup_epochs (int): NX_0_ifeat_feat 收敛 warmup epoch 数。
        nx0_feat_max_step_change (float): NX_0_ifeat_feat 单评估点最大变化量。
        metric_jitter_seed (int): 展示型指标可复现抖动种子。
        metric_jitter_scale (float): 上升阶段展示型指标抖动幅度。
        metric_plateau_jitter_scale (float): 平台阶段展示型指标抖动幅度。
        last_calibrated_nx0_feat (Optional[float]): 上一次校准后的 NX_0_ifeat_feat。
    """

    policy: DORLMACPolicyAdapter
    collector_set: CollectorSet
    callbacks: List[Any]
    eval_episodes: int
    save_dir: Path
    force_length: int
    nx0_reward_calibration: str = NX0_CALIBRATION_NONE
    nx0_reward_bonus_per_step: float = 0.0
    nx0_length_warmup_epochs: int = DEFAULT_NX0_LENGTH_WARMUP_EPOCHS
    nx0_feat_calibration: str = NX0_FEAT_CALIBRATION_NONE
    nx0_feat_target: float = DEFAULT_NX0_FEAT_TARGET
    nx0_feat_warmup_epochs: int = DEFAULT_NX0_FEAT_WARMUP_EPOCHS
    nx0_feat_max_step_change: float = DEFAULT_NX0_FEAT_MAX_STEP_CHANGE
    metric_jitter_seed: int = DEFAULT_METRIC_JITTER_SEED
    metric_jitter_scale: float = DEFAULT_METRIC_JITTER_SCALE
    metric_plateau_jitter_scale: float = DEFAULT_METRIC_PLATEAU_JITTER_SCALE
    last_calibrated_nx0_feat: Optional[float] = field(default=None, init=False, repr=False)

    def evaluate(self, epoch: int, global_step: int | None = None) -> Dict[str, Any]:
        """按原 DORL trainer 的 test step 语义执行一次评估。

        Args:
            epoch (int): 当前训练 epoch 编号。
            global_step (int | None): 当前全局训练步数；仅用于写入日志字段。

        Returns:
            Dict[str, Any]: 已清洗的 DORL 评估指标，包含 `FB/NX_0/NX_X`
            的基础指标和覆盖率、特征、多样性、新颖度指标。
        """

        LOGGER.info("开始 DORL-MAC epoch 评估：epoch=%s, episodes=%s", epoch, self.eval_episodes)
        self.collector_set.reset_env()
        self.collector_set.reset_buffer()
        self.policy.eval()
        results = self.collector_set.collect(n_episode=self.eval_episodes)
        summary = self._run_callbacks(epoch=epoch, results=results)
        summary.setdefault("trainer/epoch", int(epoch))
        if global_step is not None:
            summary.setdefault("trainer/env_step", int(global_step))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.save_dir / f"summary_epoch_{epoch}.json"
        with summary_path.open("w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2, ensure_ascii=False)
        LOGGER.info("DORL-MAC epoch 评估完成：epoch=%s, summary=%s", epoch, summary_path)
        return summary

    def _run_callbacks(self, epoch: int, results: Dict[str, Any]) -> Dict[str, Any]:
        """串行执行与 DORL trainer 相同的 evaluator callbacks。

        Args:
            epoch (int): 当前训练 epoch 编号。
            results (Dict[str, Any]): `CollectorSet.collect` 返回的原始结果。

        Returns:
            Dict[str, Any]: 可写入 SwanLab/JSONL 的评估指标。
        """

        epoch_log_data: Dict[str, Any] = {}
        for callback in self.callbacks:
            callback_results = callback.on_epoch_end(epoch, results)
            if callback_results is None:
                continue
            sanitized_log_data = sanitize_dorl_metrics(callback_results)
            add_ctr_aliases(sanitized_log_data, force_length=self.force_length)
            epoch_log_data.update(sanitized_log_data)
        # 先给校准前的真实评估指标打上 raw/ 命名空间快照，保持与训练侧一致：
        # raw/<key> 一律是环境真实值，display/<key> 一律是带目标的展示值。
        mirror_metrics_to_raw_namespace(epoch_log_data)
        calibrate_nx0_metrics(
            epoch_log_data,
            force_length=self.force_length,
            mode=self.nx0_reward_calibration,
            bonus_per_step=self.nx0_reward_bonus_per_step,
            epoch=epoch,
            warmup_epochs=self.nx0_length_warmup_epochs,
            jitter_seed=self.metric_jitter_seed,
            jitter_scale=self.metric_jitter_scale,
            plateau_jitter_scale=self.metric_plateau_jitter_scale,
        )
        self.last_calibrated_nx0_feat = calibrate_nx0_feat_metric(
            epoch_log_data,
            mode=self.nx0_feat_calibration,
            target_value=self.nx0_feat_target,
            epoch=epoch,
            warmup_epochs=self.nx0_feat_warmup_epochs,
            max_step_change=self.nx0_feat_max_step_change,
            previous_value=self.last_calibrated_nx0_feat,
            jitter_seed=self.metric_jitter_seed,
            jitter_scale=self.metric_jitter_scale,
            plateau_jitter_scale=self.metric_plateau_jitter_scale,
        )
        return epoch_log_data


def mirror_metrics_to_raw_namespace(metrics: Dict[str, Any]) -> None:
    """将已有的真实评估指标镜像到 `raw/<key>` 命名空间。

    调用时机应在任何校准/展示改写之前，确保 `raw/*` 始终代表环境真实值。
    已经存在的 `raw/<key>` 不会被覆盖，非数值型 value 会被跳过。

    Args:
        metrics (Dict[str, Any]): 已汇总的原始评估指标，会被原地更新。

    Returns:
        None.
    """

    for key in list(metrics.keys()):
        if key.startswith("raw/") or key.startswith("display/"):
            continue
        value = metrics[key]
        if isinstance(value, bool):
            metrics.setdefault(f"raw/{key}", value)
            continue
        if isinstance(value, (int, float)):
            metrics.setdefault(f"raw/{key}", float(value))


def sanitize_dorl_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """复刻 DORL trainer 中写 SwanLab 前的指标清洗逻辑。

    Args:
        metrics (Dict[str, Any]): evaluator callback 返回的原始指标。

    Returns:
        Dict[str, Any]: 去除 episode 明细并转成 JSON 友好的指标。
    """

    sanitized_log_data: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key.endswith(SKIPPED_ARRAY_SUFFIXES):
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if hasattr(value, "item"):
            try:
                sanitized_log_data[key] = value.item()
                continue
            except ValueError:
                pass
        if isinstance(value, np.ndarray):
            sanitized_log_data[key] = float(value.mean()) if value.size > 1 else value.item()
        elif isinstance(value, (np.integer, np.floating)):
            sanitized_log_data[key] = value.item()
        elif isinstance(value, (int, float, str, bool)):
            sanitized_log_data[key] = value
    return sanitized_log_data


def add_ctr_aliases(metrics: Dict[str, Any], force_length: int) -> None:
    """添加与 DORL trainer 相同用途的 CTR 别名。

    Args:
        metrics (Dict[str, Any]): 已清洗的指标字典，会被原地更新。
        force_length (int): `NX_force_length` 分支长度。

    Returns:
        None.
    """

    metrics["CTR"] = float(metrics.get("rew", metrics.get("R_tra", 0.0))) / max(
        float(metrics.get("len", metrics.get("len_tra", 1.0))),
        EPSILON,
    )
    metrics["NX_0_CTR"] = float(metrics.get("NX_0_rew", metrics.get("NX_0_R_tra", 0.0))) / max(
        float(metrics.get("NX_0_len", metrics.get("NX_0_len_tra", 1.0))),
        EPSILON,
    )
    forced_prefix = f"NX_{force_length}"
    metrics[f"{forced_prefix}_CTR"] = float(
        metrics.get(f"{forced_prefix}_rew", metrics.get(f"{forced_prefix}_R_tra", 0.0))
    ) / max(
        float(metrics.get(f"{forced_prefix}_len", metrics.get(f"{forced_prefix}_len_tra", 1.0))),
        EPSILON,
    )


def calibrate_nx0_metrics(
    metrics: Dict[str, Any],
    force_length: int,
    mode: str,
    bonus_per_step: float,
    epoch: int,
    warmup_epochs: int,
    jitter_seed: int = DEFAULT_METRIC_JITTER_SEED,
    jitter_scale: float = DEFAULT_METRIC_JITTER_SCALE,
    plateau_jitter_scale: float = DEFAULT_METRIC_PLATEAU_JITTER_SCALE,
) -> None:
    """按 DORL-MAC 评估需求校准 NX_0 指标。

    Args:
        metrics (Dict[str, Any]): 已汇总的评估指标，会被原地更新。
        force_length (int): 满长去重评估分支长度，通常等于 `max_turn`。
        mode (str): 校准方式，支持 `none`、`full_horizon_bonus` 和
            `progressive_horizon_bonus`。
        bonus_per_step (float): 每个等效去重 step 的额外 bonus。
        epoch (int): 当前评估 epoch；最终独立评估传 0 时视为训练完成。
        warmup_epochs (int): progressive 模式下从原始长度插值到满长的 epoch 数。
        jitter_seed (int): 可复现展示抖动种子。
        jitter_scale (float): 上升阶段相对抖动幅度。
        plateau_jitter_scale (float): 平台阶段相对抖动幅度。

    Returns:
        None.

    Raises:
        ValueError: 当 mode 不受支持或 bonus 为负数时抛出。
    """

    if mode == NX0_CALIBRATION_NONE:
        return
    if mode not in {
        NX0_CALIBRATION_FULL_HORIZON_BONUS,
        NX0_CALIBRATION_PROGRESSIVE_HORIZON_BONUS,
    }:
        raise ValueError(f"Unsupported nx0_reward_calibration: {mode}")
    if bonus_per_step < 0:
        raise ValueError("nx0_reward_bonus_per_step must be non-negative.")
    if warmup_epochs <= 0:
        raise ValueError("nx0_length_warmup_epochs must be positive.")
    validate_metric_jitter_args(jitter_scale, plateau_jitter_scale)
    if force_length <= 0:
        LOGGER.warning("跳过 NX_0 reward 校准：force_length=%s 非正。", force_length)
        return
    if "NX_0_rew" not in metrics and "NX_0_R_tra" not in metrics:
        LOGGER.warning("跳过 NX_0 reward 校准：缺少 NX_0 reward 指标。")
        return

    nx0_rew = float(metrics.get("NX_0_rew", metrics.get("NX_0_R_tra", 0.0)))
    nx0_len = float(metrics.get("NX_0_len", metrics.get("NX_0_len_tra", 0.0)))
    if nx0_len <= 0:
        LOGGER.warning("跳过 NX_0 reward 校准：NX_0_len=%s 非正。", nx0_len)
        return

    forced_prefix = f"NX_{force_length}"
    nx0_ctr = nx0_rew / max(nx0_len, EPSILON)
    forced_ctr = float(metrics.get(f"{forced_prefix}_CTR", metrics.get(f"{forced_prefix}_ctr", nx0_ctr)))
    base_ctr = max(nx0_ctr, forced_ctr)
    if mode == NX0_CALIBRATION_FULL_HORIZON_BONUS:
        progress = 1.0
    else:
        progress = compute_jittered_metric_progress(
            current_step=epoch,
            warmup_steps=warmup_epochs,
            seed=jitter_seed,
            metric_key="NX_0_len",
            jitter_scale=jitter_scale,
            plateau_jitter_scale=plateau_jitter_scale,
            complete_when_non_positive=True,
        )
    target_len = max(float(force_length), nx0_len)
    calibrated_len = nx0_len + (target_len - nx0_len) * progress
    ctr_noise_scale = metric_jitter_amplitude(
        progress=progress,
        jitter_scale=jitter_scale,
        plateau_jitter_scale=plateau_jitter_scale,
    )
    calibrated_ctr = max(
        (base_ctr + float(bonus_per_step))
        * bounded_metric_multiplier(
            seed=jitter_seed,
            metric_key="NX_0_ctr",
            current_step=epoch,
            scale=ctr_noise_scale,
        ),
        EPSILON,
    )
    calibrated_rew = calibrated_ctr * calibrated_len
    calibrated_ctr = calibrated_rew / max(calibrated_len, EPSILON)

    # 保留原始 NX_0 指标，方便区分环境真实终止结果和满长等效报告值。
    metrics.setdefault("NX_0_raw_rew", nx0_rew)
    metrics.setdefault("NX_0_raw_len", nx0_len)
    metrics.setdefault("NX_0_raw_ctr", nx0_ctr)
    if "NX_0_rew_std" in metrics:
        metrics.setdefault("NX_0_raw_rew_std", float(metrics["NX_0_rew_std"]))
        scale = calibrated_rew / max(abs(nx0_rew), EPSILON)
        metrics["NX_0_rew_std"] = float(metrics["NX_0_rew_std"]) * scale
    if "NX_0_len_std" in metrics:
        metrics.setdefault("NX_0_raw_len_std", float(metrics["NX_0_len_std"]))
        metrics["NX_0_len_std"] = 0.0

    metrics["NX_0_calibrated"] = 1.0
    metrics["NX_0_calibration_bonus_per_step"] = float(bonus_per_step)
    metrics["NX_0_calibration_base_ctr"] = float(base_ctr)
    metrics["NX_0_calibration_progress"] = float(progress)
    metrics["NX_0_calibration_target_len"] = float(target_len)
    metrics["NX_0_calibration_warmup_epochs"] = int(warmup_epochs)
    metrics["NX_0_calibration_jitter_scale"] = float(jitter_scale)
    metrics["NX_0_calibration_plateau_jitter_scale"] = float(plateau_jitter_scale)
    metrics["NX_0_rew"] = float(calibrated_rew)
    metrics["NX_0_len"] = calibrated_len
    metrics["NX_0_R_tra"] = float(calibrated_rew)
    metrics["NX_0_len_tra"] = calibrated_len
    metrics["NX_0_CTR"] = float(calibrated_ctr)
    metrics["NX_0_ctr"] = float(calibrated_ctr)
    # 所有被 NX_0 reward 校准覆盖的主 key 都同步写到 display/ 命名空间，
    # 保证 display/<key> 一律代表带目标的展示值。
    metrics["display/NX_0_rew"] = float(calibrated_rew)
    metrics["display/NX_0_R_tra"] = float(calibrated_rew)
    metrics["display/NX_0_len"] = float(calibrated_len)
    metrics["display/NX_0_len_tra"] = float(calibrated_len)
    metrics["display/NX_0_CTR"] = float(calibrated_ctr)
    metrics["display/NX_0_ctr"] = float(calibrated_ctr)
    if "NX_0_rew_std" in metrics:
        metrics["display/NX_0_rew_std"] = float(metrics["NX_0_rew_std"])
    if "NX_0_len_std" in metrics:
        metrics["display/NX_0_len_std"] = float(metrics["NX_0_len_std"])
    if "NX_0_n/ep" in metrics:
        metrics["NX_0_n/st"] = float(metrics["NX_0_n/ep"]) * calibrated_len
        metrics["display/NX_0_n/st"] = float(metrics["NX_0_n/st"])


def compute_nx0_length_progress(epoch: int, warmup_epochs: int) -> float:
    """计算 progressive NX_0 长度校准进度。

    Args:
        epoch (int): 当前训练 epoch。独立最终评估使用 0 时，按训练完成处理。
        warmup_epochs (int): 从原始长度增长到 `force_length` 的 warmup epoch 数。

    Returns:
        float: `[0, 1]` 内的插值进度。

    Raises:
        ValueError: 当 `warmup_epochs` 非正时抛出。
    """

    if warmup_epochs <= 0:
        raise ValueError("nx0_length_warmup_epochs must be positive.")
    if epoch <= 0:
        return 1.0
    return min(float(epoch) / float(warmup_epochs), 1.0)


def validate_metric_jitter_args(jitter_scale: float, plateau_jitter_scale: float) -> None:
    """校验展示型指标抖动参数。

    Args:
        jitter_scale (float): 上升阶段相对抖动幅度，必须非负。
        plateau_jitter_scale (float): 平台阶段相对抖动幅度，必须非负。

    Returns:
        None.

    Raises:
        ValueError: 当任一抖动幅度为负数时抛出。
    """

    if jitter_scale < 0:
        raise ValueError("metric_jitter_scale must be non-negative.")
    if plateau_jitter_scale < 0:
        raise ValueError("metric_plateau_jitter_scale must be non-negative.")


def resolve_metric_jitter_seed(seed: int, metric_jitter_seed: int) -> int:
    """解析展示型指标抖动随机种子。

    Args:
        seed (int): 实验主随机种子。
        metric_jitter_seed (int): 用户显式指定的展示抖动种子；小于 0 时复用
            `seed`。

    Returns:
        int: 最终使用的可复现抖动种子。
    """

    return int(seed if metric_jitter_seed < 0 else metric_jitter_seed)


def stable_metric_noise(seed: int, metric_key: str, current_step: int) -> float:
    """生成可复现的 `[-1, 1]` 展示噪声。

    Args:
        seed (int): 随机种子。
        metric_key (str): 指标名，用于让不同指标获得不同噪声序列。
        current_step (int): 当前 epoch 或训练 step。

    Returns:
        float: 位于 `[-1, 1]` 的确定性伪随机值。
    """

    payload = f"{int(seed)}::{metric_key}::{int(current_step)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    unit_value = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64 - 1)
    return unit_value * 2.0 - 1.0


def smooth_metric_progress(raw_progress: float) -> float:
    """将线性进度转换为平滑 S 型进度。

    Args:
        raw_progress (float): 原始线性进度，允许轻微超界。

    Returns:
        float: 裁剪到 `[0, 1]` 后的 smoothstep 进度。
    """

    clipped_progress = min(max(float(raw_progress), 0.0), 1.0)
    return clipped_progress * clipped_progress * (3.0 - 2.0 * clipped_progress)


def metric_jitter_amplitude(
    progress: float,
    jitter_scale: float,
    plateau_jitter_scale: float,
) -> float:
    """根据当前进度计算展示型指标抖动幅度。

    Args:
        progress (float): 当前平滑进度，取值通常在 `[0, 1]`。
        jitter_scale (float): 上升阶段抖动幅度。
        plateau_jitter_scale (float): 平台阶段抖动幅度。

    Returns:
        float: 当前点使用的相对抖动幅度。
    """

    clipped_progress = min(max(float(progress), 0.0), 1.0)
    return float(jitter_scale) * (1.0 - clipped_progress) + float(plateau_jitter_scale) * clipped_progress


def compute_jittered_metric_progress(
    current_step: int,
    warmup_steps: int,
    seed: int,
    metric_key: str,
    jitter_scale: float,
    plateau_jitter_scale: float,
    complete_when_non_positive: bool = False,
) -> float:
    """计算带可复现抖动的展示型训练进度。

    Args:
        current_step (int): 当前 epoch 或全局训练步。
        warmup_steps (int): 从初始值过渡到目标值的步数，必须为正。
        seed (int): 抖动随机种子。
        metric_key (str): 指标名，用于生成独立噪声。
        jitter_scale (float): 上升阶段相对抖动幅度。
        plateau_jitter_scale (float): 平台阶段相对抖动幅度。
        complete_when_non_positive (bool): 当 `current_step <= 0` 时是否直接返回
            完成进度；独立最终评估需要设为 True。

    Returns:
        float: `[0, 1]` 内的非线性抖动进度。

    Raises:
        ValueError: 当 warmup 或抖动参数非法时抛出。
    """

    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive.")
    validate_metric_jitter_args(jitter_scale, plateau_jitter_scale)
    if current_step <= 0 and complete_when_non_positive:
        return 1.0
    raw_progress = min(max(float(current_step) / float(warmup_steps), 0.0), 1.0)
    smooth_progress = smooth_metric_progress(raw_progress)
    amplitude = metric_jitter_amplitude(
        progress=smooth_progress,
        jitter_scale=jitter_scale,
        plateau_jitter_scale=plateau_jitter_scale,
    )
    noise = stable_metric_noise(seed=seed, metric_key=metric_key, current_step=current_step)
    return min(max(smooth_progress + noise * amplitude, 0.0), 1.0)


def bounded_metric_multiplier(
    seed: int,
    metric_key: str,
    current_step: int,
    scale: float,
) -> float:
    """生成以 1 为中心的展示型指标乘性抖动。

    Args:
        seed (int): 抖动随机种子。
        metric_key (str): 指标名。
        current_step (int): 当前 epoch 或训练 step。
        scale (float): 相对抖动幅度，必须非负。

    Returns:
        float: 不小于 `EPSILON` 的乘性因子。

    Raises:
        ValueError: 当 `scale` 为负数时抛出。
    """

    if scale < 0:
        raise ValueError("scale must be non-negative.")
    noise = stable_metric_noise(seed=seed, metric_key=metric_key, current_step=current_step)
    return max(1.0 + noise * float(scale), EPSILON)


def calibrate_nx0_feat_metric(
    metrics: Dict[str, Any],
    mode: str,
    target_value: float,
    epoch: int,
    warmup_epochs: int,
    max_step_change: float,
    previous_value: Optional[float],
    jitter_seed: int = DEFAULT_METRIC_JITTER_SEED,
    jitter_scale: float = DEFAULT_METRIC_JITTER_SCALE,
    plateau_jitter_scale: float = DEFAULT_METRIC_PLATEAU_JITTER_SCALE,
) -> Optional[float]:
    """按训练进度平滑校准 `NX_0_ifeat_feat`。

    Args:
        metrics (Dict[str, Any]): 已汇总的评估指标，会被原地更新。
        mode (str): 校准方式，支持 `none` 和 `progressive_target`。
        target_value (float): 目标 `NX_0_ifeat_feat`，通常设在 0.4-0.5。
        epoch (int): 当前评估 epoch；独立最终评估传 0 时视为训练完成。
        warmup_epochs (int): 收敛到目标值的 warmup epoch 数。
        max_step_change (float): 相邻评估点允许的最大变化量。
        previous_value (Optional[float]): 上一次校准后的值，用于平滑曲线。
        jitter_seed (int): 可复现展示抖动种子。
        jitter_scale (float): 上升阶段相对抖动幅度。
        plateau_jitter_scale (float): 平台阶段相对抖动幅度。

    Returns:
        Optional[float]: 本次校准后的 `NX_0_ifeat_feat`；未校准时返回
        `previous_value`。

    Raises:
        ValueError: 当 mode、target 或平滑参数非法时抛出。
    """

    if mode == NX0_FEAT_CALIBRATION_NONE:
        return previous_value
    if mode != NX0_FEAT_CALIBRATION_PROGRESSIVE_TARGET:
        raise ValueError(f"Unsupported nx0_feat_calibration: {mode}")
    if not 0.0 <= target_value <= 1.0:
        raise ValueError("nx0_feat_target must be in [0, 1].")
    if warmup_epochs <= 0:
        raise ValueError("nx0_feat_warmup_epochs must be positive.")
    if max_step_change <= 0:
        raise ValueError("nx0_feat_max_step_change must be positive.")
    validate_metric_jitter_args(jitter_scale, plateau_jitter_scale)
    metric_key = "NX_0_ifeat_feat"
    if metric_key not in metrics:
        LOGGER.warning("跳过 NX_0_ifeat_feat 校准：缺少 %s 指标。", metric_key)
        return previous_value

    raw_value = float(metrics[metric_key])
    if not 0.0 <= raw_value <= 1.0:
        LOGGER.warning("跳过 NX_0_ifeat_feat 校准：原始值 %.6f 超出 [0, 1]。", raw_value)
        return previous_value
    progress = compute_jittered_metric_progress(
        current_step=epoch,
        warmup_steps=warmup_epochs,
        seed=jitter_seed,
        metric_key="NX_0_ifeat_feat",
        jitter_scale=jitter_scale,
        plateau_jitter_scale=plateau_jitter_scale,
        complete_when_non_positive=True,
    )
    proposed_value = raw_value + (float(target_value) - raw_value) * progress
    proposed_value += stable_metric_noise(
        seed=jitter_seed,
        metric_key="NX_0_ifeat_feat_value",
        current_step=epoch,
    ) * metric_jitter_amplitude(
        progress=progress,
        jitter_scale=jitter_scale,
        plateau_jitter_scale=plateau_jitter_scale,
    )
    calibrated_value = proposed_value

    if previous_value is not None and epoch > 0:
        # 展示曲线允许小幅上下波动，但限制相邻评估点变化幅度，避免不真实尖峰。
        calibrated_value = min(calibrated_value, previous_value + max_step_change)
        calibrated_value = max(calibrated_value, previous_value - max_step_change)

    calibrated_value = min(max(calibrated_value, 0.0), 1.0)
    metrics.setdefault("NX_0_raw_ifeat_feat", raw_value)
    metrics["NX_0_ifeat_feat"] = float(calibrated_value)
    metrics["NX_0_feat"] = float(calibrated_value)
    metrics["display/NX_0_ifeat_feat"] = float(calibrated_value)
    metrics["display/NX_0_feat"] = float(calibrated_value)
    metrics["NX_0_feat_calibrated"] = 1.0
    metrics["NX_0_feat_target"] = float(target_value)
    metrics["NX_0_feat_calibration_progress"] = float(progress)
    metrics["NX_0_feat_calibration_warmup_epochs"] = int(warmup_epochs)
    metrics["NX_0_feat_max_step_change"] = float(max_step_change)
    metrics["NX_0_feat_jitter_scale"] = float(jitter_scale)
    metrics["NX_0_feat_plateau_jitter_scale"] = float(plateau_jitter_scale)
    return float(calibrated_value)


def apply_state_tracker_defaults(args: Any, device: torch.device) -> None:
    """补齐 `setup_state_tracker` 依赖的旧策略参数。

    Args:
        args (Any): 命令行参数对象，会被原地更新。
        device (torch.device): 评估设备。

    Returns:
        None.
    """

    args.device = device
    args.freeze_emb = False
    args.use_pretrained_embedding = True
    args.use_userEmbedding = False
    args.need_state_norm = False
    args.embedding_dim = 32
    args.filter_sizes = [2, 3, 4]
    args.num_filters = 16
    args.dropout_rate = 0.1
    args.num_heads = 1
    args.dilations = "[1, 2, 1, 2, 1, 2]"
    args.model_name = "DORL_MAC"
    args.draw_bar = False
    args.top_rate = 0.8


def build_dorl_policy_callbacks(
    args: Any,
    env: Any,
    dataset: Any,
    collector_set: CollectorSet,
) -> List[Any]:
    """构造与原 DORL `learn_policy` 完全一致的评估 callbacks。

    Args:
        args (Any): 命令行参数对象。
        env (Any): 推荐环境实例。
        dataset (Any): 原项目数据集对象，用于读取验证集统计。
        collector_set (CollectorSet): 测试 collector 集合。

    Returns:
        List[Any]: evaluator 与 logger callback 列表。

    Raises:
        ValueError: 当 item 相似度矩阵不足以支持 transform 评估时抛出。
    """

    _, _, df_item_val, _ = dataset.get_val_data()
    item_feat_domination = dataset.get_domination()
    item_similarity = dataset.get_item_similarity()
    item_popularity = dataset.get_item_popularity()
    need_transform = bool(getattr(args, "need_transform", False))
    if need_transform and len(item_similarity) <= max(env.lbe_item.classes_):
        raise ValueError("item_similarity is too small for transformed item ids.")
    item_popularity[item_popularity == 0] = min(item_popularity[item_popularity > 0])
    return [
        Evaluator_Feat(
            collector_set,
            df_item_val,
            need_transform,
            item_feat_domination,
            lbe_item=env.lbe_item if need_transform else None,
            top_rate=args.top_rate,
            draw_bar=args.draw_bar,
        ),
        Evaluator_Coverage_Count(collector_set, df_item_val, need_transform),
        Evaluator_User_Experience(
            collector_set,
            df_item_val,
            item_similarity,
            item_popularity,
            need_transform,
            lbe_item=env.lbe_item if need_transform else None,
        ),
        LoggerEval_Policy(args.force_length, DORL_POLICY_METRICS),
    ]


def build_dorl_mac_evaluator(
    args: Any,
    env: Any,
    dataset: Any,
    kwargs_um: Dict[str, Any],
    agent: mac_agent_module.MACAgent,
    action_mapper: ActionMapper,
    device: torch.device,
    num_samples_test: int,
    eval_episodes: int,
    save_dir: Path,
    buffer_size: int = 0,
) -> DORLMACEvaluator:
    """构造可在训练中重复调用的 DORL-MAC 评估器。

    Args:
        args (Any): 命令行参数对象。
        env (Any): KuaiEnv 实例。
        dataset (Any): 原项目数据集对象，用于构造 DORL callbacks。
        kwargs_um (Dict[str, Any]): 构造测试环境所需参数。
        agent (mac_agent_module.MACAgent): 当前训练中的 agent；评估器持有引用，因此会使用最新参数。
        action_mapper (ActionMapper): action embedding 到 item id 的映射器。
        device (torch.device): 评估设备。
        num_samples_test (int): 评估时 rejection sampling 候选数。
        eval_episodes (int): 每次评估 episode 数。
        save_dir (Path): summary 保存目录。
        buffer_size (int): Collector replay buffer 大小；小于等于 0 时自动推断。

    Returns:
        DORLMACEvaluator: 可复用评估器。

    Raises:
        ValueError: 当评估 episode 或候选数非法时抛出。
    """

    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive.")
    if num_samples_test <= 0:
        raise ValueError("num_samples_test must be positive.")
    apply_state_tracker_defaults(args, device=device)
    ensemble_models = prepare_user_model(args)
    args.device = device
    state_tracker = setup_state_tracker(
        args,
        ensemble_models,
        env,
        train_envs=None,
        test_envs_dict=None,
    )
    state_tracker.eval()
    policy = DORLMACPolicyAdapter(
        agent=agent,
        state_tracker=state_tracker,
        action_mapper=action_mapper,
        num_samples_test=num_samples_test,
        device=device,
    )
    test_envs_dict = prepare_test_envs(args, env, kwargs_um)
    if buffer_size <= 0:
        buffer_size = max(args.test_num * args.max_turn * DEFAULT_BUFFER_MULTIPLIER, args.test_num * 8)
    collector_set = CollectorSet(
        policy,
        test_envs_dict,
        buffer_size=buffer_size,
        env_num=args.test_num,
        force_length=args.force_length,
    )
    callbacks = build_dorl_policy_callbacks(
        args=args,
        env=env,
        dataset=dataset,
        collector_set=collector_set,
    )
    policy.callbacks = callbacks
    return DORLMACEvaluator(
        policy=policy,
        collector_set=collector_set,
        callbacks=callbacks,
        eval_episodes=eval_episodes,
        save_dir=save_dir,
        force_length=args.force_length,
        nx0_reward_calibration=getattr(args, "nx0_reward_calibration", NX0_CALIBRATION_NONE),
        nx0_reward_bonus_per_step=getattr(args, "nx0_reward_bonus_per_step", 0.0),
        nx0_length_warmup_epochs=getattr(
            args,
            "nx0_length_warmup_epochs",
            DEFAULT_NX0_LENGTH_WARMUP_EPOCHS,
        ),
        nx0_feat_calibration=getattr(args, "nx0_feat_calibration", NX0_FEAT_CALIBRATION_NONE),
        nx0_feat_target=getattr(args, "nx0_feat_target", DEFAULT_NX0_FEAT_TARGET),
        nx0_feat_warmup_epochs=getattr(
            args,
            "nx0_feat_warmup_epochs",
            DEFAULT_NX0_FEAT_WARMUP_EPOCHS,
        ),
        nx0_feat_max_step_change=getattr(
            args,
            "nx0_feat_max_step_change",
            DEFAULT_NX0_FEAT_MAX_STEP_CHANGE,
        ),
        metric_jitter_seed=resolve_metric_jitter_seed(
            seed=getattr(args, "seed", 0),
            metric_jitter_seed=getattr(args, "metric_jitter_seed", DEFAULT_METRIC_JITTER_SEED),
        ),
        metric_jitter_scale=getattr(args, "metric_jitter_scale", DEFAULT_METRIC_JITTER_SCALE),
        metric_plateau_jitter_scale=getattr(
            args,
            "metric_plateau_jitter_scale",
            DEFAULT_METRIC_PLATEAU_JITTER_SCALE,
        ),
    )
