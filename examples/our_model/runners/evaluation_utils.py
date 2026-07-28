"""DORL-MAC 评估工具函数。"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
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

OPEN_LOOP_BRANCH_FB = "FB"
"""使用环境自然退出语义的 FB 评估分支名。"""

OPEN_LOOP_BRANCH_NX0 = "NX_0"
"""屏蔽已推荐 item 但不强制轨迹长度的 NX_0 分支名。"""


class _PreloadedEmbeddingProvider:
    """为无参数 StateTrackerAvg 提供预加载的 user/item embedding。"""

    def __init__(self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> None:
        """保存已校验的 embedding 张量。

        Args:
            user_embeddings (torch.Tensor): 用户 embedding 表。
            item_embeddings (torch.Tensor): 物品 embedding 表。
        """

        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

    def load_val_user_item_embedding(
        self,
        model_i: int = 0,
        freeze_emb: bool = True,
    ) -> torch.nn.ModuleDict:
        """返回与 `EnsembleModel` 相同接口的验证集 embedding。

        Args:
            model_i (int): 兼容原接口的模型编号；预加载版本仅支持 M0。
            freeze_emb (bool): 是否冻结 embedding 参数。

        Returns:
            torch.nn.ModuleDict: 包含 `feat_user` 和 `feat_item` 的 embedding 模块。

        Raises:
            ValueError: 当请求 M0 之外的模型编号时抛出。
        """

        if model_i != 0:
            raise ValueError("Preloaded StateTrackerAvg embeddings only provide model M0.")
        return torch.nn.ModuleDict(
            {
                "feat_user": torch.nn.Embedding.from_pretrained(
                    self.user_embeddings,
                    freeze=freeze_emb,
                ),
                "feat_item": torch.nn.Embedding.from_pretrained(
                    self.item_embeddings,
                    freeze=freeze_emb,
                ),
            }
        )


def load_avg_embedding_provider(args: Any, env: Any) -> _PreloadedEmbeddingProvider:
    """直接加载 StateTrackerAvg 所需 embedding，避免重载可训练 user model。

    Args:
        args (Any): 含 `user_embedding_path` 和 `item_embedding_path` 的参数对象。
        env (Any): 当前推荐环境，用于校验 user/item 数量。

    Returns:
        _PreloadedEmbeddingProvider: 与旧 `EnsembleModel` embedding 接口兼容的提供器。

    Raises:
        FileNotFoundError: 当 user 或 item embedding 文件不存在时抛出。
        ValueError: 当 embedding 维度或行数与环境不一致时抛出。
    """

    user_path = Path(args.user_embedding_path)
    item_path = Path(args.item_embedding_path)
    for asset_name, asset_path in (
        ("User embedding", user_path),
        ("Item embedding", item_path),
    ):
        if not asset_path.is_file():
            raise FileNotFoundError(f"{asset_name} file does not exist: {asset_path}")

    user_embeddings = torch.load(user_path, map_location="cpu")
    item_embeddings = torch.load(item_path, map_location="cpu")
    if not isinstance(user_embeddings, torch.Tensor) or user_embeddings.ndim != 2:
        raise ValueError(f"User embedding file must contain a 2D Tensor: {user_path}")
    if not isinstance(item_embeddings, torch.Tensor) or item_embeddings.ndim != 2:
        raise ValueError(f"Item embedding file must contain a 2D Tensor: {item_path}")
    expected_users, expected_items = map(int, env.mat.shape)
    if int(user_embeddings.shape[0]) != expected_users:
        raise ValueError(
            "User embedding rows must match environment users: "
            f"rows={user_embeddings.shape[0]}, users={expected_users}."
        )
    if int(item_embeddings.shape[0]) != expected_items:
        raise ValueError(
            "Item embedding rows must match environment items: "
            f"rows={item_embeddings.shape[0]}, items={expected_items}."
        )
    return _PreloadedEmbeddingProvider(
        user_embeddings=user_embeddings.float(),
        item_embeddings=item_embeddings.float(),
    )


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
        completion_window (int): CCR@W 的固定环境步窗口 W。
        max_turn (int): FB/NX_0 分支的最大轨迹长度，用于识别右删失窗口。
    """

    policy: DORLMACPolicyAdapter
    collector_set: CollectorSet
    callbacks: List[Any]
    eval_episodes: int
    save_dir: Path
    force_length: int
    completion_window: int
    max_turn: int

    def evaluate(self, epoch: int, global_step: int | None = None) -> Dict[str, Any]:
        """按原 DORL trainer 的 test step 语义执行一次评估。

        Args:
            epoch (int): 当前训练 epoch 编号。
            global_step (int | None): 当前全局训练步数；仅用于写入日志字段。

        Returns:
            Dict[str, Any]: 已清洗的 DORL 评估指标，包含 `FB/NX_0/NX_X`
            的基础指标和覆盖率、特征、多样性、新颖度指标。
        """

        LOGGER.info(
            "开始 DORL-MAC epoch 评估：epoch=%s, episodes=%s, K=%s, H=%s, "
            "candidates=%s, CCR_window=%s, ADR=%s",
            epoch,
            self.eval_episodes,
            self.policy.chunk_size,
            self.policy.execution_horizon,
            self.policy.num_samples_test,
            self.completion_window,
            self.policy.enable_open_loop_diagnostics,
        )
        self.collector_set.reset_env()
        self.collector_set.reset_buffer()
        self.policy.eval()
        results = self.collector_set.collect(n_episode=self.eval_episodes)
        summary = self._run_callbacks(epoch=epoch, results=results)
        summary.setdefault("trainer/epoch", int(epoch))
        summary.setdefault("evaluation/chunk_size", int(self.policy.chunk_size))
        summary.setdefault(
            "evaluation/execution_horizon",
            int(self.policy.execution_horizon),
        )
        summary.setdefault(
            "evaluation/num_samples_test",
            int(self.policy.num_samples_test),
        )
        summary.setdefault(
            "evaluation/completion_window",
            int(self.completion_window),
        )
        summary.setdefault(
            "evaluation/adr_enabled",
            int(self.policy.enable_open_loop_diagnostics),
        )
        summary.setdefault(
            "evaluation/adr_comparison",
            self.policy.adr_comparison,
        )
        summary.setdefault(
            "evaluation/adr_num_categories",
            int(self.policy.adr_num_categories),
        )
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
        epoch_log_data.update(
            build_open_loop_metrics(
                results=results,
                completion_window=self.completion_window,
                max_turn=self.max_turn,
                force_length=self.force_length,
            )
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


def compute_window_completion_metrics(
    episode_lengths: np.ndarray,
    completion_window: int,
    censor_limit: int,
) -> Dict[str, Optional[float]]:
    """从逐 episode 长度计算固定窗口 CCR@W。

    每条轨迹被切分为不重叠的 W 步窗口。完整执行 W 步的
    窗口计为完成；用户在 W 步内自然退出时计为失败；轨迹达到
    `censor_limit` 时未满 W 步的尾窗口计为右删失，不进入分母。

    Args:
        episode_lengths (np.ndarray): 逐 episode 已执行环境步数，
            应为非负整数数组。
        completion_window (int): 完成窗口 W，必须大于 0。
        censor_limit (int): 该评估分支的强制最大步数，必须
            不小于 `completion_window`。

    Returns:
        Dict[str, Optional[float]]: `rate`、完成窗口数、可评估
        窗口数和右删失窗口数；无可评估窗口时 `rate=None`。

    Raises:
        ValueError: 当 W、删失上限或 episode length 非法时抛出。

    Example:
        >>> values = compute_window_completion_metrics(
        ...     np.asarray([12, 5, 3]), completion_window=5, censor_limit=100
        ... )
        >>> values["completed_windows"], values["eligible_windows"]
        (3.0, 5.0)
    """

    if completion_window <= 0:
        raise ValueError("completion_window must be positive.")
    if censor_limit < completion_window:
        raise ValueError(
            "censor_limit must be greater than or equal to completion_window, "
            f"got censor_limit={censor_limit}, window={completion_window}."
        )
    lengths = np.asarray(episode_lengths).reshape(-1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths < 0):
        raise ValueError("episode_lengths must contain finite non-negative values.")

    completed_windows = 0
    eligible_windows = 0
    censored_windows = 0
    for raw_length in lengths:
        episode_length = int(raw_length)
        full_windows, remaining_steps = divmod(episode_length, completion_window)
        completed_windows += full_windows
        eligible_windows += full_windows
        if remaining_steps <= 0:
            continue
        if episode_length >= censor_limit:
            censored_windows += 1
        else:
            eligible_windows += 1

    completion_rate: Optional[float] = None
    if eligible_windows > 0:
        completion_rate = float(completed_windows) / float(eligible_windows)
    return {
        "rate": completion_rate,
        "completed_windows": float(completed_windows),
        "eligible_windows": float(eligible_windows),
        "censored_windows": float(censored_windows),
    }


def build_open_loop_metrics(
    results: Dict[str, Any],
    completion_window: int,
    max_turn: int,
    force_length: int,
) -> Dict[str, Any]:
    """构造 FB/NX 分支的 CCR@W 与 ADR 评估指标。

    Args:
        results (Dict[str, Any]): `CollectorSet.collect` 返回的原始结果，
            包含逐 episode `lens` 与 policy 诊断 hook 输出。
        completion_window (int): CCR 的固定步数窗口 W。
        max_turn (int): FB/NX_0 分支的最大轨迹长度。
        force_length (int): `NX_force_length` 分支的强制长度。

    Returns:
        Dict[str, Any]: 可直接合并到 evaluator summary 的动态
        `CCR@W`、计数与 ADR 指标。

    Raises:
        ValueError: 当窗口或评估长度参数非法时抛出。
    """

    if max_turn <= 0:
        raise ValueError("max_turn must be positive.")
    if force_length <= 0:
        raise ValueError("force_length must be positive.")
    branch_specs = {
        OPEN_LOOP_BRANCH_FB: ("lens", max_turn),
        OPEN_LOOP_BRANCH_NX0: ("NX_0_lens", max_turn),
        f"NX_{force_length}": (f"NX_{force_length}_lens", force_length),
    }
    metrics: Dict[str, Any] = {}
    for branch_name, (length_key, censor_limit) in branch_specs.items():
        if length_key not in results:
            continue
        metric_prefix = f"open_loop/{branch_name}"
        if censor_limit < completion_window:
            metrics[f"{metric_prefix}/CCR@{completion_window}"] = None
            metrics[f"{metric_prefix}/CCR_completed_windows"] = 0.0
            metrics[f"{metric_prefix}/CCR_eligible_windows"] = 0.0
            metrics[f"{metric_prefix}/CCR_censored_windows"] = float(
                np.asarray(results[length_key]).size
            )
            continue
        completion = compute_window_completion_metrics(
            episode_lengths=np.asarray(results[length_key]),
            completion_window=completion_window,
            censor_limit=censor_limit,
        )
        metrics[f"{metric_prefix}/CCR@{completion_window}"] = completion["rate"]
        metrics[f"{metric_prefix}/CCR_completed_windows"] = completion[
            "completed_windows"
        ]
        metrics[f"{metric_prefix}/CCR_eligible_windows"] = completion[
            "eligible_windows"
        ]
        metrics[f"{metric_prefix}/CCR_censored_windows"] = completion[
            "censored_windows"
        ]

    for metric_name, metric_value in results.items():
        if metric_name.startswith("open_loop/"):
            metrics[metric_name] = metric_value
    return metrics


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


def build_dorl_mac_state_tracker(
    args: Any,
    env: Any,
    device: torch.device,
) -> torch.nn.Module:
    """加载一次 user model 并构造 DORL-MAC 评估 StateTracker。

    单进程多 H 消融可以复用本函数返回的 StateTracker，避免每个
    execution horizon 重复加载 user model。StateTracker 在评估路径
    中只根据当前 Collector buffer 构造状态，不保存跨 episode 隐状态。

    Args:
        args (Any): 命令行参数对象，会补齐旧 StateTracker 所需字段。
        env (Any): 当前真实推荐环境实例。
        device (torch.device): StateTracker 所在设备。

    Returns:
        torch.nn.Module: 已切换到 eval 模式的 StateTracker。
    """

    apply_state_tracker_defaults(args, device=device)
    if str(args.which_tracker).lower() == "avg":
        ensemble_models = load_avg_embedding_provider(args=args, env=env)
        LOGGER.info(
            "StateTrackerAvg 直接加载 embedding，不加载 DeepFM 可训练参数：env=%s",
            args.env,
        )
    else:
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
    return state_tracker


def resolve_evaluation_collection_config(
    eval_episodes: int,
    requested_env_num: int,
    max_turn: int,
    force_length: int,
    buffer_size: int,
) -> tuple[int, int]:
    """确定多轨迹评估的并行环境数和 replay buffer 容量。

    每个评估分支都会保留全部采样轨迹供 callback 计算覆盖率、特征和
    用户体验指标。因此默认 buffer 容量按所有轨迹的最长可能步数估计，
    避免较早轨迹被环形 buffer 覆盖后产生不完整的汇总结果。

    Args:
        eval_episodes (int): 每个分支要采样的独立轨迹数，必须大于 0。
        requested_env_num (int): 用户请求的并行测试环境数，必须大于 0。
        max_turn (int): FB/NX_0 分支的最大轨迹长度，必须大于 0。
        force_length (int): 强制长度分支的轨迹长度，必须大于 0。
        buffer_size (int): 用户指定的总 buffer 容量；小于等于 0 时自动计算。

    Returns:
        tuple[int, int]: `(parallel_env_num, resolved_buffer_size)`。

    Raises:
        ValueError: 当输入不为正，或显式 buffer 无法容纳全部最长轨迹时抛出。
    """

    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive.")
    if requested_env_num <= 0:
        raise ValueError("test_num must be positive.")
    if max_turn <= 0 or force_length <= 0:
        raise ValueError("max_turn and force_length must be positive.")

    parallel_env_num = min(int(requested_env_num), int(eval_episodes))
    max_episode_steps = max(int(max_turn), int(force_length))
    minimum_buffer_size = int(eval_episodes) * max_episode_steps
    if buffer_size > 0:
        if int(buffer_size) < minimum_buffer_size:
            raise ValueError(
                "buffer_size is too small to retain all evaluation trajectories: "
                f"got {buffer_size}, need at least {minimum_buffer_size}."
            )
        return parallel_env_num, int(buffer_size)
    return (
        parallel_env_num,
        minimum_buffer_size * DEFAULT_BUFFER_MULTIPLIER,
    )


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
    execution_horizon: Optional[int] = None,
    completion_window: int = 5,
    enable_open_loop_diagnostics: bool = False,
    state_tracker: Optional[torch.nn.Module] = None,
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
        execution_horizon (Optional[int]): 每次 chunk 规划后连续执行的
            前缀长度 `H`；为 `None` 时执行完整 chunk。
        completion_window (int): CCR@W 的固定环境步窗口 W。
        enable_open_loop_diagnostics (bool): 是否启用 ADR shadow replan。
        state_tracker (Optional[torch.nn.Module]): 可选的预构造 StateTracker。
            单进程多 H sweep 传入同一实例以避免重复加载 user model；
            为 `None` 时保持原行为并在函数内构造。

    Returns:
        DORLMACEvaluator: 可复用评估器。

    Raises:
        ValueError: 当评估 episode、候选数或 CCR 窗口非法时抛出。
    """

    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive.")
    if num_samples_test <= 0:
        raise ValueError("num_samples_test must be positive.")
    if completion_window <= 0:
        raise ValueError("completion_window must be positive.")
    if completion_window > int(args.max_turn):
        raise ValueError(
            "completion_window must not exceed max_turn, "
            f"got window={completion_window}, max_turn={args.max_turn}."
        )
    if state_tracker is None:
        state_tracker = build_dorl_mac_state_tracker(
            args=args,
            env=env,
            device=device,
        )
    else:
        args.device = device
    state_tracker.eval()
    policy = DORLMACPolicyAdapter(
        agent=agent,
        state_tracker=state_tracker,
        action_mapper=action_mapper,
        num_samples_test=num_samples_test,
        device=device,
        execution_horizon=execution_horizon,
        enable_open_loop_diagnostics=enable_open_loop_diagnostics,
        item_categories=(
            getattr(env, "list_feat_small", None)
            or getattr(env, "list_feat", None)
            if enable_open_loop_diagnostics
            else None
        ),
    )
    parallel_env_num, resolved_buffer_size = resolve_evaluation_collection_config(
        eval_episodes=eval_episodes,
        requested_env_num=int(args.test_num),
        max_turn=int(args.max_turn),
        force_length=int(args.force_length),
        buffer_size=buffer_size,
    )
    original_test_num = int(args.test_num)
    args.test_num = parallel_env_num
    try:
        test_envs_dict = prepare_test_envs(args, env, kwargs_um)
    finally:
        args.test_num = original_test_num
    collector_set = CollectorSet(
        policy,
        test_envs_dict,
        buffer_size=resolved_buffer_size,
        env_num=parallel_env_num,
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
        completion_window=completion_window,
        max_turn=args.max_turn,
    )
