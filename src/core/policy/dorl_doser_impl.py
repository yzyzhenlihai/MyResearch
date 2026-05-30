"""DORL-DOSER 的扩散产物加载与通用 OOD 工具。

该模块为推荐系统版 DORL-DOSER 提供三类能力：加载 KuaiEnv-v0
扩散预训练产物、构造 counterfactual reward 近似值、计算 DOSER
OOD 识别所需的动作和状态重构误差。实现重点服务 on-policy A2C
训练路径，off-policy 类仅保留兼容入口。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Discrete
from torch import nn

from tianshou.data import Batch, to_numpy

from src.core.diffusion_doser.karras import DiffusionModel
from src.core.diffusion_doser.mlps import ScoreNetwork


LOGGER = logging.getLogger(__name__)

DEFAULT_HIDDEN_DIM = 256
"""扩散预训练脚本默认使用的 MLP 隐层宽度。"""

DEFAULT_TIME_EMBED_DIM = 16
"""扩散预训练脚本默认使用的时间嵌入维度。"""

DEFAULT_NUM_HIDDEN_LAYERS = 4
"""扩散预训练脚本默认使用的 MLP 隐层数量。"""

DEFAULT_SIGMA_DATA = 0.5
"""Karras diffusion 默认数据噪声尺度。"""

DEFAULT_SIGMA_MIN = 0.002
"""Karras diffusion 默认最小噪声尺度。"""

DEFAULT_SIGMA_MAX = 80.0
"""Karras diffusion 默认最大噪声尺度。"""

DEFAULT_OOD_LEVELS = 1
"""运行期 OOD 误差默认采样噪声层次数。"""

DEFAULT_OOD_BATCH_SIZE = 256
"""运行期 OOD 误差默认分批大小。"""

DEFAULT_THRESHOLD = float("inf")
"""缺少阈值时的保守默认值，避免误把全部样本判为 OOD。"""

MIN_REWARD_VALUE = 0.0
"""counterfactual reward 的非负裁剪下界。"""

UNKNOWN_ENTROPY_VALUE = 1.0
"""真实模拟环境中未知历史组合使用的 entropy 回退值。"""

OBS_LAST_ACTION_COLUMN = 1
"""推荐环境观测中上一动作 item id 所在的列。"""

STATE_MODEL_CONDITIONAL = "conditional"
"""新格式状态扩散模型类型：p(s_next | s, a)。"""

STATE_MODEL_UNCONDITIONAL = "unconditional"
"""旧格式状态扩散模型类型：p(s)。"""


@dataclass
class DiffusionArtifact:
    """封装 DOSER 运行期所需的扩散模型产物。

    Attributes:
        artifact_dir (Path): 实际加载的扩散产物目录。
        meta (Dict[str, Any]): 合并后的配置和指标字典。
        diffusion_model (DiffusionModel): Karras diffusion 工具对象。
        behavior_model (ScoreNetwork): 行为扩散模型，学习 `p(a | s)`。
        state_distribution (ScoreNetwork): 状态扩散模型，新格式学习
            `p(s_next | s, a)`，旧格式为无条件状态分布。
        state_dim (int): 扩散状态向量维度。
        action_dim (int): 扩散动作向量维度。
        state_threshold (float): 状态 OOD 阈值。
        action_threshold (float): 行为 OOD 阈值。
        state_model_kind (str): 状态模型类型，区分新旧 artifact。
    """

    artifact_dir: Path
    meta: Dict[str, Any]
    diffusion_model: DiffusionModel
    behavior_model: ScoreNetwork
    state_distribution: ScoreNetwork
    state_dim: int
    action_dim: int
    state_threshold: float
    action_threshold: float
    state_model_kind: str = STATE_MODEL_CONDITIONAL


@dataclass
class RewardModelConfig:
    """counterfactual reward 近似器的配置。

    该配置与 `examples/our_model/dorl_doser.py` 中训练环境配置保持字段兼容。
    当前运行期只需要 `predicted_mat`、环境名和若干 reward shaping 开关；
    其他字段保留用于后续更精细的推荐 reward 复现。
    """

    env_name: str
    predicted_mat: Any
    real_env: Any = None
    prefer_real_env_reward: bool = True
    version: str = "v1"
    tau: float = 0.0
    use_exposure_intervention: bool = False
    gamma_exposure: float = 10.0
    alpha_u: Any = None
    beta_i: Any = None
    entropy_dict: Optional[Dict[str, Any]] = None
    entropy_window: Optional[Sequence[int]] = None
    lambda_entropy: float = 5.0
    step_n_actions: int = 0
    entropy_min: float = 0.0
    entropy_max: float = 0.0
    feature_level: bool = True
    map_item_feat: Any = None
    is_sorted: bool = True


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int],
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """构造简单 MLP 网络。

    Args:
        input_dim (int): 输入特征维度，必须大于 0。
        output_dim (int): 输出特征维度，必须大于 0。
        hidden_sizes (Sequence[int]): 隐层维度序列。
        activation (type[nn.Module]): 隐层激活函数类型。

    Returns:
        nn.Sequential: 构造完成的前馈网络。

    Raises:
        ValueError: 当输入或输出维度非法时抛出。
    """

    if input_dim <= 0 or output_dim <= 0:
        raise ValueError(
            f"Invalid MLP dimensions: input_dim={input_dim}, output_dim={output_dim}."
        )

    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        if hidden_dim <= 0:
            raise ValueError(f"Hidden dimension must be positive, got {hidden_dim}.")
        layers.extend([nn.Linear(last_dim, hidden_dim), activation()])
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def _load_json_file(path: Path) -> Dict[str, Any]:
    """读取 JSON 文件。

    Args:
        path (Path): JSON 文件路径。

    Returns:
        Dict[str, Any]: 解析得到的字典。

    Raises:
        FileNotFoundError: 当路径不存在时抛出。
        ValueError: 当 JSON 顶层不是字典时抛出。
    """

    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be a dict: {path}")
    return data


def _candidate_artifact_dirs(
    save_root: str,
    env_name: str,
    artifact_name: Optional[str],
) -> List[Path]:
    """按优先级生成扩散产物候选目录。

    Args:
        save_root (str): 模型保存根目录。
        env_name (str): 环境或数据集名称。
        artifact_name (Optional[str]): 可选 artifact 子目录名。

    Returns:
        List[Path]: 候选目录列表，按优先级排列。
    """

    root = Path(save_root)
    env_dir = root / env_name
    candidates = [env_dir]
    if artifact_name and artifact_name.strip():
        name = artifact_name.strip()
        candidates.extend(
            [
                env_dir / name,
                env_dir / "DOSER" / "diffusion" / name,
            ]
        )
    return candidates


def _is_new_artifact_dir(path: Path) -> bool:
    """判断目录是否为新格式扩散产物目录。

    Args:
        path (Path): 候选目录。

    Returns:
        bool: 若包含新格式模型文件则返回 `True`。
    """

    return (path / "behavior_diffusion.pt").exists() and (
        path / "state_diffusion.pt"
    ).exists()


def _is_old_artifact_dir(path: Path) -> bool:
    """判断目录是否为旧格式扩散产物目录。

    Args:
        path (Path): 候选目录。

    Returns:
        bool: 若包含旧格式模型文件则返回 `True`。
    """

    return (path / "behavior_model.pth").exists() and (
        path / "state_distribution.pth"
    ).exists()


def resolve_diffusion_artifact_dir(
    save_root: str,
    env_name: str,
    artifact_name: Optional[str] = None,
) -> Path:
    """解析实际可用的扩散产物目录。

    Args:
        save_root (str): 模型保存根目录。
        env_name (str): 环境或数据集名称。
        artifact_name (Optional[str]): 可选 artifact 子目录名。

    Returns:
        Path: 包含扩散模型文件的目录。

    Raises:
        FileNotFoundError: 当所有候选目录都不存在有效产物时抛出。
    """

    checked_paths: List[str] = []
    for candidate in _candidate_artifact_dirs(save_root, env_name, artifact_name):
        checked_paths.append(str(candidate))
        if _is_new_artifact_dir(candidate) or _is_old_artifact_dir(candidate):
            return candidate

    searched = "\n  - ".join(checked_paths)
    raise FileNotFoundError(
        "Cannot find diffusion artifact. Please run diffusion pretraining first "
        "or pass --diffusion_artifact_name explicitly. Searched:\n"
        f"  - {searched}"
    )


def _resolve_checkpoint_path(
    artifact_dir: Path,
    meta: Dict[str, Any],
    key: str,
    default_name: str,
) -> Path:
    """从 metadata 与默认文件名中解析 checkpoint 路径。

    Args:
        artifact_dir (Path): artifact 目录。
        meta (Dict[str, Any]): 配置字典。
        key (str): `checkpoints` 中的字段名。
        default_name (str): 默认文件名。

    Returns:
        Path: checkpoint 路径。
    """

    checkpoints = meta.get("checkpoints", {})
    raw_path = checkpoints.get(key)
    if raw_path:
        path = Path(raw_path)
        if path.exists():
            return path
        candidate = artifact_dir / path.name
        if candidate.exists():
            return candidate
    return artifact_dir / default_name


def _merge_new_metadata(artifact_dir: Path) -> Dict[str, Any]:
    """合并新格式配置与指标。

    Args:
        artifact_dir (Path): 新格式 artifact 目录。

    Returns:
        Dict[str, Any]: 合并后的 metadata。

    Raises:
        FileNotFoundError: 当配置或指标文件缺失时抛出。
    """

    config = _load_json_file(artifact_dir / "pretrain_config.json")
    metrics_path = artifact_dir / "training_metrics.json"
    metrics = _load_json_file(metrics_path) if metrics_path.exists() else {}
    merged = dict(config)
    merged.update(metrics)
    merged["artifact_format"] = "new_flat"
    return merged


def _load_old_metadata(artifact_dir: Path) -> Dict[str, Any]:
    """加载旧格式 metadata。

    Args:
        artifact_dir (Path): 旧格式 artifact 目录。

    Returns:
        Dict[str, Any]: metadata 字典。

    Raises:
        FileNotFoundError: 当 `pretrain_meta.json` 缺失时抛出。
    """

    meta = _load_json_file(artifact_dir / "pretrain_meta.json")
    meta["artifact_format"] = "old_doser"
    return meta


def _get_training_arg(
    meta: Dict[str, Any],
    name: str,
    default: float,
) -> float:
    """从 metadata 中读取扩散训练参数。

    Args:
        meta (Dict[str, Any]): 扩散 metadata。
        name (str): 参数名。
        default (float): 缺省值。

    Returns:
        float: 解析后的浮点参数。
    """

    training_args = meta.get("training_args", {})
    value = training_args.get(name, default)
    return float(value)


def _build_behavior_model(
    state_dim: int,
    action_dim: int,
    device: torch.device,
) -> ScoreNetwork:
    """构造行为扩散模型网络。

    Args:
        state_dim (int): 条件状态维度。
        action_dim (int): 动作扩散目标维度。
        device (torch.device): 模型设备。

    Returns:
        ScoreNetwork: 条件行为扩散网络。
    """

    return ScoreNetwork(
        x_dim=action_dim,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        time_embed_dim=DEFAULT_TIME_EMBED_DIM,
        cond_dim=state_dim,
        cond_mask_prob=0.0,
        num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS,
        output_dim=action_dim,
        device=str(device),
        cond_conditional=True,
    ).to(device)


def _build_state_model(
    state_dim: int,
    action_dim: int,
    device: torch.device,
    state_model_kind: str,
) -> ScoreNetwork:
    """构造状态扩散模型网络。

    Args:
        state_dim (int): 状态扩散目标维度。
        action_dim (int): 动作条件维度。
        device (torch.device): 模型设备。
        state_model_kind (str): 状态模型类型，新格式为条件模型。

    Returns:
        ScoreNetwork: 状态扩散网络。

    Raises:
        ValueError: 当状态模型类型未知时抛出。
    """

    if state_model_kind == STATE_MODEL_CONDITIONAL:
        cond_dim = state_dim + action_dim
        cond_conditional = True
    elif state_model_kind == STATE_MODEL_UNCONDITIONAL:
        cond_dim = 0
        cond_conditional = False
    else:
        raise ValueError(f"Unknown state_model_kind: {state_model_kind}")

    return ScoreNetwork(
        x_dim=state_dim,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        time_embed_dim=DEFAULT_TIME_EMBED_DIM,
        cond_dim=cond_dim,
        cond_mask_prob=0.0,
        num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS,
        output_dim=state_dim,
        device=str(device),
        cond_conditional=cond_conditional,
    ).to(device)


def load_diffusion_artifact(
    save_root: str,
    env_name: str,
    artifact_name: Optional[str],
    device: Any,
) -> DiffusionArtifact:
    """加载 DOSER 运行期扩散产物。

    Args:
        save_root (str): 扩散模型保存根目录。
        env_name (str): 环境或数据集名称。
        artifact_name (Optional[str]): 可选 artifact 子目录名。
        device (Any): 目标设备，可为字符串或 `torch.device`。

    Returns:
        DiffusionArtifact: 加载完成的扩散产物对象。

    Raises:
        FileNotFoundError: 当模型文件或 metadata 缺失时抛出。
        ValueError: 当 metadata 中缺少维度信息时抛出。
        RuntimeError: 当 checkpoint 与网络结构不匹配时由 PyTorch 抛出。
    """

    torch_device = torch.device(device)
    artifact_dir = resolve_diffusion_artifact_dir(save_root, env_name, artifact_name)
    is_new_format = _is_new_artifact_dir(artifact_dir)
    if is_new_format:
        meta = _merge_new_metadata(artifact_dir)
        state_model_kind = STATE_MODEL_CONDITIONAL
        behavior_path = _resolve_checkpoint_path(
            artifact_dir, meta, "behavior_diffusion", "behavior_diffusion.pt"
        )
        state_path = _resolve_checkpoint_path(
            artifact_dir, meta, "state_diffusion", "state_diffusion.pt"
        )
    else:
        meta = _load_old_metadata(artifact_dir)
        state_model_kind = STATE_MODEL_UNCONDITIONAL
        behavior_path = _resolve_checkpoint_path(
            artifact_dir, meta, "behavior_model", "behavior_model.pth"
        )
        state_path = _resolve_checkpoint_path(
            artifact_dir, meta, "state_distribution", "state_distribution.pth"
        )

    state_dim = int(meta.get("state_dim", 0))
    action_dim = int(meta.get("action_dim", 0))
    if state_dim <= 0 or action_dim <= 0:
        raise ValueError(
            f"Invalid diffusion dimensions in {artifact_dir}: "
            f"state_dim={state_dim}, action_dim={action_dim}."
        )

    diffusion_model = DiffusionModel(
        sigma_data=_get_training_arg(meta, "sigma_data", DEFAULT_SIGMA_DATA),
        sigma_min=_get_training_arg(meta, "sigma_min", DEFAULT_SIGMA_MIN),
        sigma_max=_get_training_arg(meta, "sigma_max", DEFAULT_SIGMA_MAX),
        device=str(torch_device),
    )
    behavior_model = _build_behavior_model(state_dim, action_dim, torch_device)
    state_distribution = _build_state_model(
        state_dim,
        action_dim,
        torch_device,
        state_model_kind=state_model_kind,
    )

    behavior_model.load_state_dict(torch.load(behavior_path, map_location=torch_device))
    state_distribution.load_state_dict(torch.load(state_path, map_location=torch_device))
    behavior_model.eval()
    state_distribution.eval()

    state_threshold = float(meta.get("state_threshold", DEFAULT_THRESHOLD))
    action_threshold = float(meta.get("action_threshold", DEFAULT_THRESHOLD))
    LOGGER.info(
        "Loaded diffusion artifact: dir=%s, format=%s, state_dim=%s, "
        "action_dim=%s, state_threshold=%s, action_threshold=%s",
        artifact_dir,
        meta.get("artifact_format"),
        state_dim,
        action_dim,
        state_threshold,
        action_threshold,
    )
    return DiffusionArtifact(
        artifact_dir=artifact_dir,
        meta=meta,
        diffusion_model=diffusion_model,
        behavior_model=behavior_model,
        state_distribution=state_distribution,
        state_dim=state_dim,
        action_dim=action_dim,
        state_threshold=state_threshold,
        action_threshold=action_threshold,
        state_model_kind=state_model_kind,
    )


def _as_numpy_1d(values: Any, dtype: Any = np.int64) -> np.ndarray:
    """将输入转换为一维 numpy 数组。

    Args:
        values (Any): Tensor、numpy 数组或列表。
        dtype (Any): 目标 numpy dtype。

    Returns:
        np.ndarray: 一维数组。
    """

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    return array.reshape(-1).astype(dtype, copy=False)


def _get_feature_histories(
    window_size: int,
    item_history: Sequence[int],
    map_item_feat: Any,
    is_sort: bool,
) -> Sequence[Tuple[int, ...]]:
    """递归展开最近若干 item 对应的特征组合。

    Args:
        window_size (int): 需要展开的历史窗口长度，必须为非负整数。
        item_history (Sequence[int]): 使用原始 item id 表示的历史 item 序列。
        map_item_feat (Any): 原始 item id 到特征列表的映射。
        is_sort (bool): 是否对窗口内特征组合排序。

    Returns:
        Sequence[Tuple[int, ...]]: 特征组合集合。若 item 特征缺失则返回空集合，
        由调用方按真实环境的未知组合回退逻辑处理。
    """

    if len(item_history) < window_size or window_size <= 0:
        return [tuple()]

    target_item = int(item_history[-1])
    if map_item_feat is None or target_item not in map_item_feat:
        return []

    target_features = map_item_feat[target_item]
    previous_histories = _get_feature_histories(
        window_size - 1,
        item_history[:-1],
        map_item_feat,
        is_sort,
    )
    result = set()
    for feature_history in previous_histories:
        for feature in target_features:
            new_history = list(feature_history)
            new_history.append(int(feature))
            if is_sort:
                new_history = sorted(new_history)
            result.add(tuple(new_history))
    return result


class CounterfactualRewardModel:
    """推荐系统 counterfactual reward 的轻量近似器。

    当前实现优先使用真实推荐环境 `real_env.mat[user, item]`，使 rerank
    reward prior 更贴近最终评估中的 CTR/reward。若后续数据集没有提供真实
    reward 矩阵，则回退到离线预测矩阵加 entropy bonus 的训练模拟环境近似。
    exposure intervention 依赖完整曝光历史，当前仍按关闭曝光时的主训练配置处理。
    """

    def __init__(self, config: RewardModelConfig) -> None:
        """初始化 reward 近似器。

        Args:
            config (RewardModelConfig): reward 计算配置。

        Raises:
            ValueError: 当 `predicted_mat` 为空时抛出。
        """

        if config.predicted_mat is None:
            raise ValueError("RewardModelConfig.predicted_mat must not be None.")
        self.config = config
        self.predicted_mat = np.asarray(config.predicted_mat)
        if self.predicted_mat.ndim != 2:
            raise ValueError(
                "predicted_mat must be a 2D matrix, got "
                f"shape={self.predicted_mat.shape}."
            )
        self.real_reward_mat = self._resolve_real_reward_matrix()
        self.use_real_reward = self.real_reward_mat is not None
        entropy_offset = config.lambda_entropy * float(config.entropy_min)
        self.min_reward = float(np.min(self.predicted_mat) + entropy_offset)
        self.entropy_map = self._get_entropy_map()
        self.entropy_windows = self._get_entropy_windows()
        self._entropy_cache: Dict[Tuple[int, ...], float] = {}

    def _resolve_real_reward_matrix(self) -> Optional[np.ndarray]:
        """解析真实环境 reward 矩阵。

        Returns:
            Optional[np.ndarray]: 若 `real_env.mat` 可用且与预测矩阵形状一致，
            返回二维 numpy 矩阵；否则返回 `None` 并回退到模拟 reward 近似。
        """

        if not self.config.prefer_real_env_reward:
            return None
        real_env = self.config.real_env
        real_mat = getattr(real_env, "mat", None)
        if real_mat is None:
            return None
        real_reward_mat = np.asarray(real_mat)
        if real_reward_mat.ndim != 2:
            LOGGER.warning(
                "Ignore real_env.mat for reward prior because it is not 2D: shape=%s",
                real_reward_mat.shape,
            )
            return None
        if real_reward_mat.shape != self.predicted_mat.shape:
            LOGGER.warning(
                "Ignore real_env.mat for reward prior because shape mismatch: "
                "real_shape=%s, predicted_shape=%s",
                real_reward_mat.shape,
                self.predicted_mat.shape,
            )
            return None
        LOGGER.info(
            "CounterfactualRewardModel uses real_env.mat as reward prior source: "
            "shape=%s, min=%.6f, max=%.6f",
            real_reward_mat.shape,
            float(np.min(real_reward_mat)),
            float(np.max(real_reward_mat)),
        )
        return real_reward_mat

    def _get_entropy_map(self) -> Dict[Any, float]:
        """读取 entropy map。

        Returns:
            Dict[Any, float]: 历史组合到 entropy 值的映射。缺少配置时返回空字典。
        """

        entropy_dict = self.config.entropy_dict or {}
        entropy_map = entropy_dict.get("map", {})
        return entropy_map if isinstance(entropy_map, dict) else {}

    def _get_entropy_windows(self) -> Tuple[int, ...]:
        """整理需要参与 reward shaping 的 entropy 窗口。

        Returns:
            Tuple[int, ...]: 去重并排序后的正整数窗口。
        """

        windows = self.config.entropy_window or []
        return tuple(sorted({int(window) for window in windows if int(window) > 0}))

    def _decode_action_history(self, encoded_history: Sequence[int]) -> Tuple[int, ...]:
        """将环境内部 item id 解码为原始 item id。

        Args:
            encoded_history (Sequence[int]): 环境内部连续编码 item 序列。

        Returns:
            Tuple[int, ...]: 原始 item id 序列。若无法访问 LabelEncoder，则返回
            裁剪后的环境内部编码。
        """

        history = np.asarray(encoded_history, dtype=np.int64).reshape(-1)
        real_env = self.config.real_env
        lbe_item = getattr(real_env, "lbe_item", None)
        if lbe_item is None:
            return tuple(int(action_id) for action_id in history)

        classes = getattr(lbe_item, "classes_", None)
        if classes is not None and len(classes) > 0:
            history = np.clip(history, 0, len(classes) - 1)

        try:
            decoded = lbe_item.inverse_transform(history)
        except Exception as exc:  # pragma: no cover - 防御外部编码器异常
            LOGGER.debug(
                "Failed to inverse-transform item ids for reward prior: %s",
                exc,
            )
            return tuple(int(action_id) for action_id in history)
        return tuple(int(action_id) for action_id in decoded)

    def _estimate_entropy_for_history(self, encoded_history: Sequence[int]) -> float:
        """估计单条候选动作历史的 entropy bonus。

        Args:
            encoded_history (Sequence[int]): 环境内部 item id 历史，最后一个元素应为
                当前候选动作。

        Returns:
            float: 与 `PenaltyEntExpSimulatedEnv._compute_pred_reward()` 对齐的
            entropy 累加值。
        """

        if not self.entropy_windows:
            return 0.0
        if not self.entropy_map:
            return 0.0

        decoded_history = self._decode_action_history(encoded_history)
        if decoded_history in self._entropy_cache:
            return self._entropy_cache[decoded_history]

        entropy = 0.0
        for window_size in self.entropy_windows:
            if len(decoded_history) < window_size:
                entropy += UNKNOWN_ENTROPY_VALUE
                continue

            action_window = decoded_history[-window_size:]
            action_key = (
                tuple(sorted(action_window))
                if self.config.is_sorted
                else tuple(action_window)
            )
            if self.config.feature_level:
                feature_histories = _get_feature_histories(
                    window_size,
                    action_key,
                    self.config.map_item_feat,
                    self.config.is_sorted,
                )
                if not feature_histories:
                    entropy += UNKNOWN_ENTROPY_VALUE
                    continue
                feature_entropy = [
                    float(self.entropy_map.get(feature_key, UNKNOWN_ENTROPY_VALUE))
                    for feature_key in feature_histories
                ]
                entropy += float(np.mean(feature_entropy))
            else:
                entropy += float(
                    self.entropy_map.get(action_key, UNKNOWN_ENTROPY_VALUE)
                )

        self._entropy_cache[decoded_history] = entropy
        return entropy

    def _build_action_histories(
        self,
        actions: np.ndarray,
        history_actions: Optional[Any],
    ) -> np.ndarray:
        """构造 reward 估计所需的候选动作历史。

        Args:
            actions (np.ndarray): 当前候选动作数组，形状为 `(batch_size,)`。
            history_actions (Optional[Any]): 可选历史动作数组，形状应可整理为
                `(batch_size, history_len)`，并且最后一列建议为当前候选动作。

        Returns:
            np.ndarray: 环境内部 item id 历史，形状为 `(batch_size, history_len)`。

        Raises:
            ValueError: 当历史动作行数与候选动作数量不一致时抛出。
        """

        if history_actions is None:
            histories = actions.reshape(-1, 1)
        else:
            histories = np.asarray(history_actions, dtype=np.int64)
            if histories.ndim == 1:
                histories = histories.reshape(-1, 1)
            elif histories.ndim > 2:
                histories = histories.reshape(histories.shape[0], -1)
            if histories.shape[0] != actions.shape[0]:
                raise ValueError(
                    "history_actions must have the same first dimension as "
                    f"action_ids, got {histories.shape[0]} and {actions.shape[0]}."
                )
        return np.clip(histories, 0, self.predicted_mat.shape[1] - 1)

    def estimate(
        self,
        user_ids: Any,
        action_ids: Any,
        history_actions: Optional[Any] = None,
    ) -> np.ndarray:
        """估计一批 `(user, item)` 的 counterfactual reward。

        Args:
            user_ids (Any): 用户 ID，形状可展平为 `(batch_size,)`。
            action_ids (Any): 物品 ID，形状可展平为 `(batch_size,)`。
            history_actions (Optional[Any]): 可选动作历史，最后一个元素应为当前
                候选动作；缺省时只用当前候选动作近似真实环境历史。

        Returns:
            np.ndarray: 非负 reward 数组，形状为 `(batch_size,)`。
        """

        active_reward_mat = (
            self.real_reward_mat if self.use_real_reward else self.predicted_mat
        )
        users = _as_numpy_1d(user_ids, dtype=np.int64)
        actions = _as_numpy_1d(action_ids, dtype=np.int64)
        users = np.clip(users, 0, active_reward_mat.shape[0] - 1)
        actions = np.clip(actions, 0, active_reward_mat.shape[1] - 1)
        if self.use_real_reward:
            rewards = active_reward_mat[users, actions].astype(np.float32)
            return np.maximum(rewards, MIN_REWARD_VALUE).astype(np.float32)

        histories = self._build_action_histories(actions, history_actions)
        entropy_values = np.asarray(
            [self._estimate_entropy_for_history(history) for history in histories],
            dtype=np.float32,
        )
        rewards = (
            self.predicted_mat[users, actions].astype(np.float32)
            + float(self.config.lambda_entropy) * entropy_values
            - self.min_reward
        )
        return np.maximum(rewards, MIN_REWARD_VALUE).astype(np.float32)


class DORLDOSEROODHelper:
    """DORL-DOSER 运行期 OOD 识别和候选动作工具。

    该类集中处理 item embedding 查询、扩散重构误差、候选动作映射、
    counterfactual next-state 构造等逻辑，避免策略类中混入过多数据适配代码。
    """

    def __init__(
        self,
        state_tracker: nn.Module,
        diffusion_artifact: DiffusionArtifact,
        reward_model: CounterfactualRewardModel,
        action_levels: int = DEFAULT_OOD_LEVELS,
        state_levels: int = DEFAULT_OOD_LEVELS,
        batch_size: int = DEFAULT_OOD_BATCH_SIZE,
    ) -> None:
        """初始化 OOD helper。

        Args:
            state_tracker (nn.Module): 推荐系统状态编码器。
            diffusion_artifact (DiffusionArtifact): 扩散预训练产物。
            reward_model (CounterfactualRewardModel): counterfactual reward 近似器。
            action_levels (int): 动作重构误差噪声采样次数。
            state_levels (int): 状态重构误差噪声采样次数。
            batch_size (int): 误差计算分批大小。

        Raises:
            ValueError: 当采样次数或批大小非法时抛出。
        """

        if action_levels <= 0 or state_levels <= 0:
            raise ValueError("OOD levels must be positive.")
        if batch_size <= 0:
            raise ValueError("OOD batch_size must be positive.")

        self.state_tracker = state_tracker
        self.artifact = diffusion_artifact
        self.reward_model = reward_model
        self.action_levels = action_levels
        self.state_levels = state_levels
        self.batch_size = batch_size
        self.device = next(diffusion_artifact.behavior_model.parameters()).device
        self._cached_item_embeddings: Optional[torch.Tensor] = None
        self._warned_state_alignment = False
        self._warned_counterfactual_fallback = False

    def get_all_item_embeddings(self) -> torch.Tensor:
        """读取全部 item/action embedding。

        Returns:
            torch.Tensor: 归一化前的 item embedding，形状为
            `(num_items, action_dim)`。
        """

        if self._cached_item_embeddings is None:
            item_index = np.expand_dims(np.arange(self.state_tracker.num_item), -1)
            embeddings = self.state_tracker.get_embedding(item_index, "action")
            self._cached_item_embeddings = embeddings.detach().to(
                self.device, dtype=torch.float32
            )
        return self._cached_item_embeddings

    def lookup_action_embeddings(self, action_ids: Any) -> torch.Tensor:
        """根据 item ID 查询动作 embedding。

        Args:
            action_ids (Any): item ID 张量或数组，形状可展平为 `(batch_size,)`。

        Returns:
            torch.Tensor: 动作 embedding，形状为 `(batch_size, action_dim)`。
        """

        ids = torch.as_tensor(action_ids, device=self.device, dtype=torch.long).view(-1)
        ids = torch.clamp(ids, 0, self.state_tracker.num_item - 1)
        return self.get_all_item_embeddings().index_select(0, ids)

    def normalize_obs_array(self, states: torch.Tensor) -> torch.Tensor:
        """将策略状态维度对齐到扩散状态维度。

        Args:
            states (torch.Tensor): 原始状态张量，形状为 `(batch_size, dim)`。

        Returns:
            torch.Tensor: 维度对齐后的状态张量，形状为
            `(batch_size, diffusion_state_dim)`。
        """

        states = states.to(self.device, dtype=torch.float32)
        if states.ndim != 2:
            states = states.view(states.shape[0], -1)
        current_dim = states.shape[-1]
        target_dim = self.artifact.state_dim
        if current_dim == target_dim:
            return states

        if not self._warned_state_alignment:
            LOGGER.warning(
                "Align state dim for diffusion OOD: current_dim=%s, target_dim=%s.",
                current_dim,
                target_dim,
            )
            self._warned_state_alignment = True

        if current_dim > target_dim:
            return states[:, :target_dim]

        padding = torch.zeros(
            states.shape[0],
            target_dim - current_dim,
            device=states.device,
            dtype=states.dtype,
        )
        return torch.cat([states, padding], dim=-1)

    def normalize_action_array(self, actions: torch.Tensor) -> torch.Tensor:
        """将动作 embedding 维度对齐到扩散动作维度。

        Args:
            actions (torch.Tensor): 动作张量，形状为 `(batch_size, dim)`。

        Returns:
            torch.Tensor: 维度对齐后的动作张量。
        """

        actions = actions.to(self.device, dtype=torch.float32)
        current_dim = actions.shape[-1]
        target_dim = self.artifact.action_dim
        if current_dim == target_dim:
            return actions
        if current_dim > target_dim:
            return actions[:, :target_dim]
        padding = torch.zeros(
            actions.shape[0],
            target_dim - current_dim,
            device=actions.device,
            dtype=actions.dtype,
        )
        return torch.cat([actions, padding], dim=-1)

    def _compute_reconstruction_error(
        self,
        model: ScoreNetwork,
        targets: torch.Tensor,
        conditions: Optional[torch.Tensor],
        levels: int,
    ) -> torch.Tensor:
        """计算 Karras 扩散模型的逐样本重构误差。

        Args:
            model (ScoreNetwork): score network。
            targets (torch.Tensor): 扩散目标，形状为 `(batch_size, dim)`。
            conditions (Optional[torch.Tensor]): 条件输入或 `None`。
            levels (int): 噪声采样次数。

        Returns:
            torch.Tensor: 逐样本 L2 重构误差，形状为 `(batch_size,)`。
        """

        targets = targets.to(self.device, dtype=torch.float32)
        if conditions is not None:
            conditions = conditions.to(self.device, dtype=torch.float32)

        errors = torch.zeros(targets.shape[0], device=self.device)
        model_was_training = model.training
        model.eval()
        with torch.no_grad():
            for _ in range(levels):
                noise_scale = self.artifact.diffusion_model.make_sample_density()(
                    shape=(targets.shape[0],),
                    device=self.device,
                )
                noise = torch.randn_like(targets)
                noisy_targets = targets + noise * noise_scale.view(-1, 1)
                c_skip, c_out, c_in = [
                    scale.view(-1, 1)
                    for scale in self.artifact.diffusion_model.get_diffusion_scalings(
                        noise_scale
                    )
                ]
                model_output = model(
                    noisy_targets * c_in,
                    conditions,
                    torch.log(noise_scale) / 4,
                )
                denoised_targets = c_skip * noisy_targets + c_out * model_output
                errors += torch.norm(denoised_targets - targets, dim=1)
        if model_was_training:
            model.train()
        return errors / float(levels)

    def compute_action_error(
        self,
        actions: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """计算行为扩散动作重构误差。

        Args:
            actions (torch.Tensor): 动作 embedding，形状为 `(batch_size, action_dim)`。
            states (torch.Tensor): 当前状态，形状为 `(batch_size, state_dim)`。

        Returns:
            torch.Tensor: 逐样本动作 OOD 误差。
        """

        aligned_actions = self.normalize_action_array(actions)
        aligned_states = self.normalize_obs_array(states)
        return self._compute_reconstruction_error(
            model=self.artifact.behavior_model,
            targets=aligned_actions,
            conditions=aligned_states,
            levels=self.action_levels,
        )

    def compute_state_error(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """计算状态扩散下一状态重构误差。

        Args:
            states (torch.Tensor): 当前状态。
            actions (torch.Tensor): 当前动作 embedding。
            next_states (torch.Tensor): 下一状态或 counterfactual 下一状态。

        Returns:
            torch.Tensor: 逐样本状态 OOD 误差。
        """

        aligned_states = self.normalize_obs_array(states)
        aligned_actions = self.normalize_action_array(actions)
        aligned_next_states = self.normalize_obs_array(next_states)
        if self.artifact.state_model_kind == STATE_MODEL_CONDITIONAL:
            conditions = torch.cat([aligned_states, aligned_actions], dim=-1)
        else:
            conditions = None
        return self._compute_reconstruction_error(
            model=self.artifact.state_distribution,
            targets=aligned_next_states,
            conditions=conditions,
            levels=self.state_levels,
        )

    def map_action_embeddings_to_item_ids(
        self,
        action_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """将连续动作 embedding 映射到最近邻 item ID。

        Args:
            action_embeddings (torch.Tensor): 动作 embedding，形状为
                `(batch_size, action_dim)` 或 `(batch_size, samples, action_dim)`。

        Returns:
            torch.Tensor: 最近邻 item ID，形状匹配前缀维度。
        """

        original_shape = action_embeddings.shape[:-1]
        flat_actions = action_embeddings.reshape(-1, action_embeddings.shape[-1])
        flat_actions = self.normalize_action_array(flat_actions)
        item_embeddings = self.normalize_action_array(self.get_all_item_embeddings())
        flat_actions = F.normalize(flat_actions, dim=-1)
        item_embeddings = F.normalize(item_embeddings, dim=-1)
        scores = torch.matmul(flat_actions, item_embeddings.transpose(0, 1))
        item_ids = torch.argmax(scores, dim=-1)
        return item_ids.view(*original_shape)

    def select_best_id_action(
        self,
        states: torch.Tensor,
        critic: nn.Module,
        action_samples: int,
        diffusion_sample_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从行为扩散采样动作中选择 critic 评分最高的 ID 动作。

        Args:
            states (torch.Tensor): 当前状态，形状为 `(batch_size, state_dim)`。
            critic (nn.Module): 带有 `q_min` 方法的 augmented critic。
            action_samples (int): 每个状态采样的候选动作数量。
            diffusion_sample_steps (int): 扩散采样步数。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 最优 item ID、动作
            embedding 和对应 Q 值。
        """

        aligned_states = self.normalize_obs_array(states)
        sampled_actions = self.artifact.diffusion_model.sample(
            self.artifact.behavior_model,
            cond=aligned_states,
            action_samples=action_samples,
            n_steps=diffusion_sample_steps,
        )
        sampled_ids = self.map_action_embeddings_to_item_ids(sampled_actions)
        sampled_embs = self.lookup_action_embeddings(sampled_ids.reshape(-1)).view(
            sampled_ids.shape[0],
            sampled_ids.shape[1],
            -1,
        )
        repeated_states = states.unsqueeze(1).expand(
            -1, sampled_ids.shape[1], -1
        ).reshape(-1, states.shape[-1])
        flat_actions = sampled_embs.reshape(-1, sampled_embs.shape[-1])
        q_values = critic.q_min(repeated_states, flat_actions).view(
            sampled_ids.shape[0],
            sampled_ids.shape[1],
        )
        best_indices = torch.argmax(q_values, dim=1)
        row_indices = torch.arange(sampled_ids.shape[0], device=self.device)
        best_ids = sampled_ids[row_indices, best_indices]
        best_actions = sampled_embs[row_indices, best_indices]
        best_q = q_values[row_indices, best_indices]
        return best_ids, best_actions, best_q

    def build_counterfactual_next_state(
        self,
        batch: Batch,
        buffer: Any,
        indices: Any,
        action_ids: torch.Tensor,
        fallback_states: torch.Tensor,
    ) -> torch.Tensor:
        """通过 state tracker 构造 counterfactual 下一状态。

        Args:
            batch (Batch): 当前训练 minibatch。
            buffer (Any): replay buffer。
            indices (Any): minibatch 对应 buffer 索引。
            action_ids (torch.Tensor): counterfactual item ID。
            fallback_states (torch.Tensor): 构造失败时使用的回退状态。

        Returns:
            torch.Tensor: counterfactual 下一状态。
        """

        try:
            obs = to_numpy(batch.obs)
            if obs.ndim < 2 or obs.shape[1] < 1:
                raise ValueError(f"Cannot parse user ids from obs shape={obs.shape}.")
            user_ids = obs[:, 0].astype(np.int64, copy=False)
            action_np = action_ids.detach().cpu().numpy().reshape(-1).astype(np.int64)
            cf_obs = np.stack([user_ids, action_np], axis=1)
            if obs.shape[1] > OBS_LAST_ACTION_COLUMN:
                reward_histories = np.stack(
                    [
                        obs[:, OBS_LAST_ACTION_COLUMN].astype(np.int64, copy=False),
                        action_np,
                    ],
                    axis=1,
                )
            else:
                reward_histories = action_np.reshape(-1, 1)
            rewards = self.reward_model.estimate(
                user_ids,
                action_np,
                history_actions=reward_histories,
            )
            cf_batch = Batch(obs=cf_obs, rew_prev=rewards, info=getattr(batch, "info", Batch()))
            next_states = self.state_tracker(
                buffer=buffer,
                indices=indices,
                is_obs=True,
                batch=cf_batch,
                is_train=True,
                use_batch_in_statetracker=True,
            )
            return next_states.to(self.device, dtype=torch.float32)
        except Exception as exc:  # pragma: no cover - 仅作为训练时防御分支
            if not self._warned_counterfactual_fallback:
                LOGGER.warning(
                    "Failed to build counterfactual next state; fallback to "
                    "current states. error=%s",
                    exc,
                )
                self._warned_counterfactual_fallback = True
            return fallback_states.detach()


class DiscreteActorNetwork(nn.Module):
    """off-policy 兼容用离散 actor 网络。

    该类保留 `examples/our_model/dorl_doser.py` 的导入兼容性；本轮主训练
    路径使用 on-policy A2C actor。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        """初始化离散 actor。

        Args:
            state_dim (int): 状态维度。
            action_dim (int): 离散动作数量。
            hidden_sizes (Sequence[int]): 隐层维度。
        """

        super().__init__()
        self.output_dim = action_dim
        self.net = build_mlp(state_dim, action_dim, hidden_sizes)

    def forward(
        self,
        obs: torch.Tensor,
        state: Any = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """计算动作概率。

        Args:
            obs (torch.Tensor): 状态张量。
            state (Any): 兼容 Tianshou 的隐藏状态。
            info (Optional[Dict[str, Any]]): 兼容 Tianshou 的信息字段。

        Returns:
            Tuple[torch.Tensor, Any]: 动作概率与隐藏状态。
        """

        return torch.softmax(self.net(obs), dim=-1), state


class DORLCriticNetwork(nn.Module):
    """off-policy 兼容用 Q/V critic 网络。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        """初始化 critic。

        Args:
            state_dim (int): 状态维度。
            action_dim (int): 动作 embedding 维度。
            hidden_sizes (Sequence[int]): 隐层维度。
        """

        super().__init__()
        self.q_net = build_mlp(state_dim + action_dim, 1, hidden_sizes)
        self.v_net = build_mlp(state_dim, 1, hidden_sizes)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """计算 Q 值。

        Args:
            states (torch.Tensor): 状态张量。
            actions (torch.Tensor): 动作 embedding。

        Returns:
            torch.Tensor: Q 值张量。
        """

        return self.q_net(torch.cat([states, actions], dim=-1))

    def value(self, states: torch.Tensor) -> torch.Tensor:
        """计算 V 值。

        Args:
            states (torch.Tensor): 状态张量。

        Returns:
            torch.Tensor: V 值张量。
        """

        return self.v_net(states)


class DORLDOSERPolicy(nn.Module):
    """off-policy DORL-DOSER 兼容壳。

    本轮实现目标是 on-policy A2C 版本，因此该类仅保证旧入口导入和
    forward 行为不崩溃；若调用 `learn` 会给出明确错误。
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        optim: Any,
        state_tracker: nn.Module,
        diffusion_artifact: DiffusionArtifact,
        reward_model: CounterfactualRewardModel,
        action_dim: int,
        **kwargs: Any,
    ) -> None:
        """初始化兼容 policy。

        Args:
            actor (nn.Module): 离散 actor。
            critic (nn.Module): critic 网络。
            optim (Any): 优化器对象。
            state_tracker (nn.Module): 状态编码器。
            diffusion_artifact (DiffusionArtifact): 扩散产物。
            reward_model (CounterfactualRewardModel): reward 近似器。
            action_dim (int): 动作 embedding 维度。
            **kwargs (Any): 兼容旧参数。
        """

        super().__init__()
        self.actor = actor
        self.critic = critic
        self.optim = optim
        self.state_tracker = state_tracker
        self.diffusion_artifact = diffusion_artifact
        self.reward_model = reward_model
        self.action_dim = action_dim
        self.action_type = "discrete"
        self.action_space = Discrete(getattr(actor, "output_dim", 1))

    def forward(
        self,
        batch: Batch,
        buffer: Any = None,
        indices: Any = None,
        is_obs: bool = True,
        state: Any = None,
        is_train: bool = True,
        use_batch_in_statetracker: bool = False,
        **kwargs: Any,
    ) -> Batch:
        """计算离散动作。

        Args:
            batch (Batch): 输入 batch。
            buffer (Any): replay buffer。
            indices (Any): buffer 索引。
            is_obs (bool): 是否使用当前 obs。
            state (Any): 隐藏状态。
            is_train (bool): 是否训练阶段。
            use_batch_in_statetracker (bool): 是否使用 batch 构造状态。
            **kwargs (Any): 兼容额外参数。

        Returns:
            Batch: 包含 logits、act、dist 的 batch。
        """

        obs_emb = self.state_tracker(
            buffer=buffer,
            indices=indices,
            is_obs=is_obs,
            batch=batch,
            is_train=is_train,
            use_batch_in_statetracker=use_batch_in_statetracker,
        )
        logits, hidden = self.actor(obs_emb, state=state, info=batch.info)
        dist = torch.distributions.Categorical(logits)
        act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def process_fn(self, batch: Batch, buffer: Any, indices: Any) -> Batch:
        """兼容 Tianshou 的数据预处理接口。"""

        self._buffer = buffer
        self._indices = indices
        return batch

    def post_process_fn(self, batch: Batch, buffer: Any, indices: Any) -> None:
        """兼容 Tianshou 的后处理接口。"""

        return None

    def learn(self, *args: Any, **kwargs: Any) -> Dict[str, List[float]]:
        """阻止误用未实现的 off-policy 训练路径。

        Raises:
            NotImplementedError: 始终抛出，提示使用 on-policy 入口。
        """

        raise NotImplementedError(
            "Off-policy DORLDOSERPolicy is not implemented in this revision. "
            "Use examples/our_model/dorl_doser_onpolicy.py for A2C on-policy training."
        )
