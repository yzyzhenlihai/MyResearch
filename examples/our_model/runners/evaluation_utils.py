"""DORL-MAC 评估工具函数。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
    """

    policy: DORLMACPolicyAdapter
    collector_set: CollectorSet
    callbacks: List[Any]
    eval_episodes: int
    save_dir: Path
    force_length: int

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
        return epoch_log_data


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
        1e-12,
    )
    metrics["NX_0_CTR"] = float(metrics.get("NX_0_rew", metrics.get("NX_0_R_tra", 0.0))) / max(
        float(metrics.get("NX_0_len", metrics.get("NX_0_len_tra", 1.0))),
        1e-12,
    )
    forced_prefix = f"NX_{force_length}"
    metrics[f"{forced_prefix}_CTR"] = float(
        metrics.get(f"{forced_prefix}_rew", metrics.get(f"{forced_prefix}_R_tra", 0.0))
    ) / max(
        float(metrics.get(f"{forced_prefix}_len", metrics.get(f"{forced_prefix}_len_tra", 1.0))),
        1e-12,
    )


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
    )
