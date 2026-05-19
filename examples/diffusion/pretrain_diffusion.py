"""DOSER diffusion 预训练的通用入口脚本。

该脚本同时支持两类数据后端：

1. `trajectory_pkl`
   直接读取推荐系统离线轨迹 `pickle` 文件，使用其中的
   `observations`、`actions` 与 `next_observations` 训练行为扩散模型
   和条件状态扩散模型。
2. `d4rl`
   兼容原始 DOSER 的 D4RL 训练方式，用于连续控制环境。

与原始实现相比，本版本有三项核心变化：

1. 训练改为基于 `DataLoader` 的小批量优化，但保留原有
   `--pretrain_epochs` 作为“优化步数”语义，避免默认超参数失真。
2. 默认将产物保存到 `saved_models/<env_name>/`，并为未来数据集保留
   `--env_name`、`--dataset_path`、`--save_root` 与 `--artifact_name` 接口。
3. 训练结束后额外计算并保存 `state_threshold`、`action_threshold`、
   `pretrain_config.json` 与 `training_metrics.json`。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.diffusion_doser.karras import DiffusionModel
from src.core.diffusion_doser.mlps import ScoreNetwork
from src.core.diffusion_doser.utils import append_dims

LOGGER = logging.getLogger(__name__)

DEFAULT_ENV_NAME = "KuaiEnv-v0"
"""默认环境名。当前主流程面向 KuaiEnv-v0。"""

DEFAULT_TRAJECTORY_DATASET_PATH = PROJECT_ROOT / "data/KuaiRec/data_processed/DM_KuaiEnv-v0_small_data.pkl"
"""默认 Kuai 轨迹数据路径。"""

DEFAULT_SAVE_ROOT = PROJECT_ROOT / "saved_models"
"""默认模型保存根目录。"""

DEFAULT_SWANLAB_PROJECT = "DOSER-Diffusion"
"""SwanLab 实验日志后端的默认项目名。"""

DEFAULT_HIDDEN_DIM = 256
"""扩散网络的默认隐藏层维度。"""

DEFAULT_TIME_EMBED_DIM = 16
"""扩散网络时间编码维度。"""

DEFAULT_NUM_HIDDEN_LAYERS = 4
"""扩散网络默认隐藏层数量。"""

DEFAULT_LEARNING_RATE = 3e-4
"""Adam 优化器默认学习率。"""

DEFAULT_PERCENTILE = 99.0
"""阈值标定默认使用的分位数。"""

DEFAULT_NUM_WORKERS = 0
"""为避免 `pickle` 轨迹对象在多进程下重复拷贝，默认关闭 DataLoader worker。"""

DEFAULT_STD_FLOOR = 1e-6
"""D4RL 状态标准化时用于避免除零的最小标准差。"""

MODEL_KIND_BEHAVIOR = "behavior"
"""行为扩散模型训练类型，学习 `p(a | s)`。"""

MODEL_KIND_STATE = "state"
"""状态扩散模型训练类型，学习 `p(s_next | s, a)`。"""


@dataclass
class DatasetBundle:
    """封装统一的数据集视图与统计信息。

    Attributes:
        dataset (Dataset): 行为扩散模型使用的数据集，保留全部 transition。
        state_dataset (Dataset): 状态扩散模型使用的数据集，默认跳过 terminal transition。
        dataset_backend (str): 数据后端类型，取值为 `trajectory_pkl` 或 `d4rl`。
        dataset_path (Optional[str]): 原始数据路径；D4RL 后端为 `None`。
        state_dim (int): 状态向量维度。
        action_dim (int): 动作向量维度。
        num_users (Optional[int]): 轨迹用户数；D4RL 数据默认记为 `None`。
        num_transitions (int): 行为扩散模型实际可见的 transition 数。
        num_state_transitions (int): 状态扩散模型实际可见的 transition 数。
        normalization (Dict[str, Any]): 归一化策略说明。
    """

    dataset: Dataset
    state_dataset: Dataset
    dataset_backend: str
    dataset_path: Optional[str]
    state_dim: int
    action_dim: int
    num_users: Optional[int]
    num_transitions: int
    num_state_transitions: int
    normalization: Dict[str, Any]


@dataclass
class ArtifactPaths:
    """描述 diffusion 预训练产物的规范路径。"""

    artifact_dir: Path
    behavior_model_path: Path
    state_diffusion_path: Path
    config_path: Path
    metrics_path: Path


class ExperimentTracker:
    """管理可选的 SwanLab 实验日志后端。

    该类默认不开启在线日志；只有当用户显式传入 `--enable_swanlab`
    或兼容别名 `--enable_wandb` 时，才会尝试初始化 `swanlab`。如果
    运行环境未安装 `swanlab`，会自动降级为仅控制台日志。
    """

    def __init__(
        self,
        enabled: bool,
        project: str,
        run_name: str,
        config: Dict[str, Any],
        mode: Optional[str] = None,
    ) -> None:
        """初始化可选 SwanLab 日志跟踪器。

        Args:
            enabled (bool): 是否尝试启用 SwanLab 实验日志。
            project (str): SwanLab 项目名。
            run_name (str): 运行名。
            config (Dict[str, Any]): 配置快照。
            mode (Optional[str]): 可选 SwanLab 运行模式，写入 `SWANLAB_MODE`。

        Returns:
            None

        Raises:
            RuntimeError: 当前实现不会主动抛出该异常。
        """

        self.enabled = enabled
        self._run = None
        self._module = None

        if not enabled:
            LOGGER.info("SwanLab 实验日志已关闭，仅保留控制台日志。")
            return

        resolved_mode = (mode or os.environ.get("SWANLAB_MODE", "")).strip().lower()
        if resolved_mode:
            os.environ["SWANLAB_MODE"] = resolved_mode
        if resolved_mode == "disabled":
            LOGGER.info("检测到 SWANLAB_MODE=disabled，跳过 SwanLab 初始化。")
            self.enabled = False
            return

        try:
            import swanlab  # type: ignore
        except ImportError:
            LOGGER.warning("未安装 swanlab，跳过 SwanLab 实验日志。")
            self.enabled = False
            return

        self._module = swanlab
        self._run = swanlab.init(project=project, name=run_name, config=config)
        LOGGER.info("已启用 SwanLab 运行记录：project=%s, run=%s", project, run_name)

    def log(self, metrics: Dict[str, float]) -> None:
        """记录一组标量指标到 SwanLab。

        Args:
            metrics (Dict[str, float]): 需要记录的标量指标。

        Returns:
            None

        Raises:
            RuntimeError: 当前实现不会主动抛出该异常。
        """

        if self._run is not None and self._module is not None:
            self._module.log(metrics)

    def finish(self) -> None:
        """安全结束 SwanLab 日志会话。"""

        if self._run is not None and self._module is not None:
            self._module.finish()
            self._run = None


class ArrayTransitionDataset(Dataset):
    """基于二维数组的 transition 数据集。

    该类主要用于 D4RL 后端，其中状态、动作和下一状态已经天然以二维矩阵
    形式存在，不需要额外的轨迹索引映射。若提供 terminal 标记，可为状态
    扩散训练过滤掉没有有效下一状态语义的 transition。
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        terminals: Optional[np.ndarray] = None,
        transition_limit: Optional[int] = None,
        drop_terminal_transitions: bool = False,
    ) -> None:
        """构造数组数据集。

        Args:
            states (np.ndarray): 状态矩阵，形状为 `(N, state_dim)`。
            actions (np.ndarray): 动作矩阵，形状为 `(N, action_dim)`。
            next_states (np.ndarray): 下一状态矩阵，形状为 `(N, state_dim)`。
            terminals (Optional[np.ndarray]): 终止标记，形状为 `(N,)`。
            transition_limit (Optional[int]): 最多保留的 transition 数。
            drop_terminal_transitions (bool): 是否跳过 terminal transition。

        Returns:
            None

        Raises:
            ValueError: 当输入维度非法、样本数不一致或过滤后为空时抛出。
        """

        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)

        if states.ndim != 2:
            raise ValueError(f"`states` must be 2D, got shape={states.shape}.")
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        if actions.ndim != 2:
            raise ValueError(f"`actions` must be 1D or 2D, got shape={actions.shape}.")
        if next_states.ndim != 2:
            raise ValueError(f"`next_states` must be 2D, got shape={next_states.shape}.")
        if states.shape != next_states.shape:
            raise ValueError(
                "State/next-state shape mismatch: "
                f"states={states.shape}, next_states={next_states.shape}."
            )
        if len(states) != len(actions):
            raise ValueError(
                "State/action sample count mismatch: "
                f"len(states)={len(states)}, len(actions)={len(actions)}."
            )

        if terminals is not None:
            terminals = np.asarray(terminals, dtype=bool)
            if terminals.shape[0] != len(states):
                raise ValueError(
                    "Terminal sample count mismatch: "
                    f"len(states)={len(states)}, len(terminals)={len(terminals)}."
                )
            if drop_terminal_transitions:
                # 状态扩散模型学习有效的后继状态映射，terminal transition 默认不参与。
                valid_mask = ~terminals
                states = states[valid_mask]
                actions = actions[valid_mask]
                next_states = next_states[valid_mask]

        if transition_limit is not None:
            if transition_limit <= 0:
                raise ValueError("`max_transitions` must be positive when provided.")
            states = states[: int(transition_limit)]
            actions = actions[: int(transition_limit)]
            next_states = next_states[: int(transition_limit)]

        if len(states) == 0:
            raise ValueError("Dataset is empty after applying transition filters.")

        self.states = states
        self.actions = actions
        self.next_states = next_states

    def __len__(self) -> int:
        """返回样本总数。"""

        return len(self.states)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按索引返回单条 `(state, action, next_state)` 样本。"""

        return self.states[index], self.actions[index], self.next_states[index]




class TrajectoryTransitionDataset(Dataset):
    """基于 `list[dict]` 轨迹对象的 transition 数据集。

    该类保留原始用户轨迹列表，并通过前缀和把用户级轨迹映射成
    transition 级样本。行为扩散数据集保留所有样本，状态扩散数据集可
    跳过 terminal transition，避免无效 `next_observations` 进入条件状态建模。
    """

    def __init__(
        self,
        trajectories: Sequence[Dict[str, Any]],
        transition_limit: Optional[int] = None,
        drop_terminal_transitions: bool = False,
    ) -> None:
        """构造轨迹数据集。

        Args:
            trajectories (Sequence[Dict[str, Any]]): 顶层轨迹列表。
            transition_limit (Optional[int]): 仅保留前若干条 transition，
                主要用于 smoke test。
            drop_terminal_transitions (bool): 是否跳过 terminal transition。

        Returns:
            None

        Raises:
            ValueError: 当轨迹列表为空、字段缺失、长度不一致或维度不一致时抛出。
            KeyError: 当轨迹缺少必要键时抛出。
        """

        if len(trajectories) == 0:
            raise ValueError("Trajectory dataset is empty.")

        required_keys = {
            "actions",
            "terminals",
            "rewards",
            "observations",
            "next_observations",
            "user_id",
        }
        lengths: List[int] = []
        valid_lengths: List[int] = []
        valid_step_indices: List[np.ndarray] = []
        state_dim = None
        action_dim = None
        full_num_transitions = 0
        terminal_transitions = 0

        for trajectory_index, trajectory in enumerate(trajectories):
            missing_keys = sorted(required_keys.difference(trajectory.keys()))
            if missing_keys:
                raise KeyError(
                    f"Trajectory #{trajectory_index} is missing required keys: {missing_keys}."
                )

            observations = np.asarray(trajectory["observations"])
            actions = np.asarray(trajectory["actions"])
            next_observations = np.asarray(trajectory["next_observations"])
            rewards = np.asarray(trajectory["rewards"])
            terminals = np.asarray(trajectory["terminals"], dtype=bool)

            if observations.ndim != 2:
                raise ValueError(
                    f"Trajectory #{trajectory_index} observations must be 2D, "
                    f"got shape={observations.shape}."
                )
            if actions.ndim == 1:
                actions = actions.reshape(-1, 1)
                trajectory["actions"] = actions
            if actions.ndim != 2:
                raise ValueError(
                    f"Trajectory #{trajectory_index} actions must be 1D or 2D, "
                    f"got shape={actions.shape}."
                )
            if next_observations.ndim != 2:
                raise ValueError(
                    f"Trajectory #{trajectory_index} next_observations must be 2D, "
                    f"got shape={next_observations.shape}."
                )

            trajectory_length = observations.shape[0]
            if actions.shape[0] != trajectory_length:
                raise ValueError(
                    f"Trajectory #{trajectory_index} has inconsistent lengths: "
                    f"observations={trajectory_length}, actions={actions.shape[0]}."
                )
            if next_observations.shape[0] != trajectory_length:
                raise ValueError(
                    f"Trajectory #{trajectory_index} has inconsistent lengths: "
                    f"observations={trajectory_length}, next_observations={next_observations.shape[0]}."
                )
            if rewards.shape[0] != trajectory_length:
                raise ValueError(
                    f"Trajectory #{trajectory_index} has inconsistent lengths: "
                    f"observations={trajectory_length}, rewards={rewards.shape[0]}."
                )
            if terminals.shape[0] != trajectory_length:
                raise ValueError(
                    f"Trajectory #{trajectory_index} has inconsistent lengths: "
                    f"observations={trajectory_length}, terminals={terminals.shape[0]}."
                )

            if trajectory_length == 0:
                raise ValueError(f"Trajectory #{trajectory_index} is empty.")

            if state_dim is None:
                state_dim = int(observations.shape[1])
            elif state_dim != int(observations.shape[1]):
                raise ValueError(
                    f"Trajectory #{trajectory_index} has different state dim: "
                    f"expected {state_dim}, got {observations.shape[1]}."
                )
            if next_observations.shape[1] != state_dim:
                raise ValueError(
                    f"Trajectory #{trajectory_index} next_observations dim mismatch: "
                    f"expected {state_dim}, got {next_observations.shape[1]}."
                )

            if action_dim is None:
                action_dim = int(actions.shape[1])
            elif action_dim != int(actions.shape[1]):
                raise ValueError(
                    f"Trajectory #{trajectory_index} has different action dim: "
                    f"expected {action_dim}, got {actions.shape[1]}."
                )

            full_num_transitions += int(trajectory_length)
            current_terminal_count = int(terminals.sum())
            terminal_transitions += current_terminal_count
            lengths.append(int(trajectory_length))

            if drop_terminal_transitions:
                # 只有在确实存在 terminal 时才需要额外的 step 索引数组。
                valid_steps = np.flatnonzero(~terminals).astype(np.int64)
                valid_step_indices.append(valid_steps)
                valid_lengths.append(int(len(valid_steps)))

        use_filtered_indices = drop_terminal_transitions and terminal_transitions > 0
        effective_lengths = valid_lengths if use_filtered_indices else lengths
        offsets = np.cumsum(np.asarray([0] + effective_lengths, dtype=np.int64))
        num_transitions = int(offsets[-1])

        if transition_limit is not None:
            if transition_limit <= 0:
                raise ValueError("`max_transitions` must be positive when provided.")
            num_transitions = min(num_transitions, int(transition_limit))

        if num_transitions <= 0:
            raise ValueError("Dataset is empty after applying transition filters.")

        self.trajectories = list(trajectories)
        self.offsets = offsets
        self.valid_step_indices = valid_step_indices if use_filtered_indices else None
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.full_num_transitions = int(full_num_transitions)
        self.terminal_transitions = int(terminal_transitions)
        self.num_transitions = int(num_transitions)
        self.drop_terminal_transitions = drop_terminal_transitions

    def __len__(self) -> int:
        """返回实际可见的 transition 数。"""

        return self.num_transitions

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """根据全局 transition 索引返回 `(state, action, next_state)`。

        Args:
            index (int): 全局 transition 下标。

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 当前 transition 的状态、动作和下一状态。

        Raises:
            IndexError: 当下标超出范围时抛出。
        """

        if index < 0 or index >= self.num_transitions:
            raise IndexError(f"Index out of range: {index}.")

        trajectory_index = int(np.searchsorted(self.offsets, index, side="right") - 1)
        local_index = int(index - self.offsets[trajectory_index])
        if self.valid_step_indices is None:
            step_index = local_index
        else:
            step_index = int(self.valid_step_indices[trajectory_index][local_index])
        trajectory = self.trajectories[trajectory_index]

        state = np.asarray(trajectory["observations"][step_index], dtype=np.float32)
        action = np.asarray(trajectory["actions"][step_index], dtype=np.float32)
        next_state = np.asarray(trajectory["next_observations"][step_index], dtype=np.float32)
        return state, action, next_state




def configure_logging() -> None:
    """配置预训练脚本的日志格式。"""

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )


def set_seed(seed: int) -> None:
    """设置随机种子以尽量提高可复现性。

    Args:
        seed (int): 随机种子。

    Returns:
        None

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: Any) -> torch.device:
    """解析设备参数。

    Args:
        device_arg (Any): 用户传入的设备标识，支持 `cpu`、`0`、`cuda:0` 等格式。

    Returns:
        torch.device: 解析后的设备对象。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    if isinstance(device_arg, torch.device):
        return device_arg

    requested = str(device_arg).strip().lower()
    if requested == "cpu":
        return torch.device("cpu")

    if requested.startswith("cuda:"):
        requested = requested.split(":", 1)[1]

    if requested.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{int(requested)}")

    if torch.cuda.is_available():
        LOGGER.warning("无法识别设备 `%s`，自动回退到默认 CUDA 设备。", device_arg)
        return torch.device("cuda:0")

    LOGGER.warning("CUDA 不可用，自动回退到 CPU。")
    return torch.device("cpu")


def namespace_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """将 `argparse.Namespace` 转为 JSON 可序列化字典。"""

    serialized: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, torch.device):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def resolve_artifact_name(
    env_name: str,
    dataset_path: Optional[str],
    artifact_name: Optional[str],
) -> str:
    """解析保存目录中的 artifact 名称。

    Args:
        env_name (str): 环境名。
        dataset_path (Optional[str]): 数据集路径。
        artifact_name (Optional[str]): 用户显式指定的 artifact 名。

    Returns:
        str: 解析后的 artifact 名称。

    Raises:
        ValueError: 当最终名称为空时抛出。
    """

    if artifact_name is not None and artifact_name.strip():
        resolved = artifact_name.strip()
    elif dataset_path:
        resolved = Path(dataset_path).stem
    else:
        resolved = env_name

    if not resolved:
        raise ValueError("Resolved artifact_name is empty.")
    return resolved


def resolve_artifact_paths(
    save_root: str,
    env_name: str,
    artifact_name: Optional[str] = None,
) -> ArtifactPaths:
    """构建规范化的 diffusion 产物路径。

    Args:
        save_root (str): 模型保存根目录。
        env_name (str): 数据集或环境名称。
        artifact_name (Optional[str]): 可选子目录名；默认直接保存到环境目录。

    Returns:
        ArtifactPaths: 训练产物路径集合。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    artifact_dir = Path(save_root) / env_name
    if artifact_name is not None and artifact_name.strip():
        artifact_dir = artifact_dir / artifact_name.strip()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(
        artifact_dir=artifact_dir,
        behavior_model_path=artifact_dir / "behavior_diffusion.pt",
        state_diffusion_path=artifact_dir / "state_diffusion.pt",
        config_path=artifact_dir / "pretrain_config.json",
        metrics_path=artifact_dir / "training_metrics.json",
    )


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """为统一数据集构造 `DataLoader`。

    Args:
        dataset (Dataset): 数据集对象。
        batch_size (int): 小批量大小。
        shuffle (bool): 是否打乱样本顺序。

    Returns:
        DataLoader: 构造完成的数据加载器。

    Raises:
        ValueError: 当批大小非法时抛出。
    """

    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")

    effective_batch_size = min(batch_size, len(dataset))
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def normalize_d4rl_states(states: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """对 D4RL 状态做 z-score 标准化。

    Args:
        states (np.ndarray): 原始状态矩阵。

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            标准化后的状态矩阵，以及归一化策略说明。

    Raises:
        ValueError: 当状态矩阵维度非法时抛出。
    """

    if states.ndim != 2:
        raise ValueError(f"`states` must be 2D, got shape={states.shape}.")

    mean = states.mean(axis=0, keepdims=True)
    std = states.std(axis=0, keepdims=True)
    std = np.maximum(std, DEFAULT_STD_FLOOR)
    normalized_states = (states - mean) / std
    normalization = {
        "enabled": True,
        "type": "zscore",
        "std_floor": DEFAULT_STD_FLOOR,
    }
    return normalized_states.astype(np.float32), normalization



def normalize_d4rl_state_pairs(
    states: np.ndarray,
    next_states: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """使用同一组统计量标准化 D4RL 当前状态和下一状态。

    Args:
        states (np.ndarray): 当前状态矩阵，形状为 `(N, state_dim)`。
        next_states (np.ndarray): 下一状态矩阵，形状为 `(N, state_dim)`。

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 标准化后的当前状态、下一状态和归一化说明。

    Raises:
        ValueError: 当状态矩阵维度非法或两者形状不一致时抛出。
    """

    if states.ndim != 2:
        raise ValueError(f"`states` must be 2D, got shape={states.shape}.")
    if next_states.ndim != 2:
        raise ValueError(f"`next_states` must be 2D, got shape={next_states.shape}.")
    if states.shape != next_states.shape:
        raise ValueError(
            "State/next-state shape mismatch before normalization: "
            f"states={states.shape}, next_states={next_states.shape}."
        )

    mean = states.mean(axis=0, keepdims=True)
    std = np.maximum(states.std(axis=0, keepdims=True), DEFAULT_STD_FLOOR)
    normalization = {
        "enabled": True,
        "type": "zscore",
        "std_floor": DEFAULT_STD_FLOOR,
    }
    return (
        ((states - mean) / std).astype(np.float32),
        ((next_states - mean) / std).astype(np.float32),
        normalization,
    )

def load_trajectory_dataset_bundle(
    dataset_path: str,
    max_trajectories: Optional[int],
    max_transitions: Optional[int],
) -> DatasetBundle:
    """加载推荐系统轨迹 `pickle` 数据。

    Args:
        dataset_path (str): 轨迹文件路径。
        max_trajectories (Optional[int]): 仅保留前若干条用户轨迹。
        max_transitions (Optional[int]): 仅保留前若干条 transition。

    Returns:
        DatasetBundle: 统一包装后的数据集对象。

    Raises:
        FileNotFoundError: 当数据路径不存在时抛出。
        ValueError: 当顶层对象类型错误或数据非法时抛出。
    """

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory dataset does not exist: {path}")

    LOGGER.info("开始加载轨迹数据：%s", path)
    with path.open("rb") as file_obj:
        trajectories = pickle.load(file_obj)

    if not isinstance(trajectories, list):
        raise ValueError(
            f"Trajectory dataset must be `list[dict]`, got {type(trajectories).__name__}."
        )

    if max_trajectories is not None:
        if max_trajectories <= 0:
            raise ValueError("`max_trajectories` must be positive when provided.")
        trajectories = trajectories[: max_trajectories]

    behavior_dataset = TrajectoryTransitionDataset(
        trajectories=trajectories,
        transition_limit=max_transitions,
        drop_terminal_transitions=False,
    )
    state_dataset = TrajectoryTransitionDataset(
        trajectories=trajectories,
        transition_limit=max_transitions,
        drop_terminal_transitions=True,
    )
    LOGGER.info(
        "轨迹数据加载完成：users=%s, behavior_transitions=%s, "
        "state_transitions=%s, terminal_transitions=%s, state_dim=%s, action_dim=%s",
        len(trajectories),
        behavior_dataset.num_transitions,
        state_dataset.num_transitions,
        behavior_dataset.terminal_transitions,
        behavior_dataset.state_dim,
        behavior_dataset.action_dim,
    )

    normalization = {
        "enabled": False,
        "type": "none",
        "reason": "trajectory_pkl backend defaults to no normalization",
    }
    return DatasetBundle(
        dataset=behavior_dataset,
        state_dataset=state_dataset,
        dataset_backend="trajectory_pkl",
        dataset_path=str(path),
        state_dim=behavior_dataset.state_dim,
        action_dim=behavior_dataset.action_dim,
        num_users=len(trajectories),
        num_transitions=behavior_dataset.num_transitions,
        num_state_transitions=state_dataset.num_transitions,
        normalization=normalization,
    )


def load_d4rl_dataset_bundle(
    env_name: str,
    no_normalize: bool,
    max_transitions: Optional[int],
) -> DatasetBundle:
    """加载 D4RL 数据并构造统一数据视图。

    Args:
        env_name (str): D4RL 环境名。
        no_normalize (bool): 是否关闭状态标准化。
        max_transitions (Optional[int]): 仅保留前若干条 transition。

    Returns:
        DatasetBundle: 统一包装后的 D4RL 数据集对象。

    Raises:
        ImportError: 当缺少 `gym` 或 `d4rl` 依赖时抛出。
        ValueError: 当 transition 限制非法时抛出。
    """

    try:
        import d4rl  # type: ignore
        import gym  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "D4RL backend requires both `gym` and `d4rl` to be installed."
        ) from exc

    LOGGER.info("开始加载 D4RL 数据：env=%s", env_name)
    env = gym.make(env_name)
    raw_dataset = d4rl.qlearning_dataset(env)
    states = np.asarray(raw_dataset["observations"], dtype=np.float32)
    actions = np.asarray(raw_dataset["actions"], dtype=np.float32)
    next_states = np.asarray(raw_dataset["next_observations"], dtype=np.float32)
    terminals = np.asarray(raw_dataset.get("terminals", np.zeros(len(states))), dtype=bool)

    if no_normalize:
        normalization = {
            "enabled": False,
            "type": "none",
            "reason": "user disabled D4RL state normalization",
        }
    else:
        states, next_states, normalization = normalize_d4rl_state_pairs(states, next_states)

    behavior_dataset = ArrayTransitionDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        transition_limit=max_transitions,
        drop_terminal_transitions=False,
    )
    state_dataset = ArrayTransitionDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        transition_limit=max_transitions,
        drop_terminal_transitions=True,
    )
    LOGGER.info(
        "D4RL 数据加载完成：behavior_transitions=%s, state_transitions=%s, "
        "state_dim=%s, action_dim=%s, normalized=%s",
        len(behavior_dataset),
        len(state_dataset),
        states.shape[1],
        behavior_dataset.actions.shape[1],
        normalization["enabled"],
    )

    return DatasetBundle(
        dataset=behavior_dataset,
        state_dataset=state_dataset,
        dataset_backend="d4rl",
        dataset_path=None,
        state_dim=int(states.shape[1]),
        action_dim=int(behavior_dataset.actions.shape[1]),
        num_users=None,
        num_transitions=len(behavior_dataset),
        num_state_transitions=len(state_dataset),
        normalization=normalization,
    )


def build_dataset_bundle(args: argparse.Namespace) -> DatasetBundle:
    """根据命令行参数构造统一数据集对象。"""

    if args.dataset_backend == "trajectory_pkl":
        return load_trajectory_dataset_bundle(
            dataset_path=args.dataset_path,
            max_trajectories=args.max_trajectories,
            max_transitions=args.max_transitions,
        )

    return load_d4rl_dataset_bundle(
        env_name=args.env_name,
        no_normalize=args.no_normalize,
        max_transitions=args.max_transitions,
    )


def build_state_diffusion_model(
    state_dim: int,
    action_dim: int,
    device: torch.device,
) -> ScoreNetwork:
    """构造条件状态扩散模型。

    Args:
        state_dim (int): 状态向量维度，也是扩散目标 `next_state` 的维度。
        action_dim (int): 动作向量维度。
        device (torch.device): 模型所在设备。

    Returns:
        ScoreNetwork: 学习 `p(s_next | s, a)` 的条件扩散网络。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    return ScoreNetwork(
        x_dim=state_dim,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        time_embed_dim=DEFAULT_TIME_EMBED_DIM,
        cond_dim=state_dim + action_dim,
        cond_mask_prob=0.0,
        num_hidden_layers=DEFAULT_NUM_HIDDEN_LAYERS,
        output_dim=state_dim,
        device=str(device),
        cond_conditional=True,
    ).to(device)



def build_behavior_model(
    state_dim: int,
    action_dim: int,
    device: torch.device,
) -> ScoreNetwork:
    """构造条件行为扩散模型。"""

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


def cycle_dataloader(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, ...]]:
    """无限循环地从 `DataLoader` 中取出 batch。

    Args:
        loader (DataLoader): 数据加载器。

    Yields:
        Iterator[Tuple[torch.Tensor, ...]]: 一个可由训练类型解释的 batch。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    while True:
        for batch in loader:
            yield batch


def prepare_diffusion_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    model_kind: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """根据模型类型把三元 transition batch 转为扩散目标和条件。

    Args:
        batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
            `(state, action, next_state)` 批次。
        model_kind (str): 训练类型，支持 `behavior` 或 `state`。
        device (torch.device): 目标设备。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 扩散目标 `x` 与条件 `cond`。

    Raises:
        ValueError: 当模型类型未知时抛出。
    """

    batch_states, batch_actions, batch_next_states = batch
    batch_states = batch_states.to(device=device, dtype=torch.float32, non_blocking=True)
    batch_actions = batch_actions.to(device=device, dtype=torch.float32, non_blocking=True)
    batch_next_states = batch_next_states.to(device=device, dtype=torch.float32, non_blocking=True)

    if model_kind == MODEL_KIND_BEHAVIOR:
        return batch_actions, batch_states
    if model_kind == MODEL_KIND_STATE:
        # 状态扩散模型建模 p(s_next | s, a)，条件为当前状态与动作拼接。
        state_condition = torch.cat([batch_states, batch_actions], dim=-1)
        return batch_next_states, state_condition

    raise ValueError(f"Unknown diffusion model kind: {model_kind}.")


def train_score_model(
    model: ScoreNetwork,
    optimizer: torch.optim.Optimizer,
    diffusion_model: DiffusionModel,
    loader: DataLoader,
    num_steps: int,
    device: torch.device,
    metric_prefix: str,
    tracker: ExperimentTracker,
    model_kind: str,
) -> Dict[str, float]:
    """以小批量方式训练一个 score network。

    注意：为了保留原 DOSER 代码的超参数规模，这里的 `num_steps`
    表示优化步数而不是完整数据轮次。

    Args:
        model (ScoreNetwork): 目标网络。
        optimizer (torch.optim.Optimizer): 优化器。
        diffusion_model (DiffusionModel): 扩散训练器。
        loader (DataLoader): 训练数据加载器。
        num_steps (int): 优化步数。
        device (torch.device): 训练设备。
        metric_prefix (str): 日志前缀。
        tracker (ExperimentTracker): 可选实验记录器。
        model_kind (str): 模型类型，支持 `behavior` 或 `state`。

    Returns:
        Dict[str, float]: 训练摘要指标。

    Raises:
        ValueError: 当 `num_steps` 非正或模型类型未知时抛出。
    """

    if num_steps <= 0:
        raise ValueError("`pretrain_epochs` must be positive.")

    batch_iterator = cycle_dataloader(loader)
    loss_history = []
    progress = tqdm(range(num_steps), desc=f"train_{metric_prefix}")

    for step_index in progress:
        target_tensor, condition_tensor = prepare_diffusion_batch(
            batch=next(batch_iterator),
            model_kind=model_kind,
            device=device,
        )

        optimizer.zero_grad()
        loss = diffusion_model.diffusion_train_step(model, target_tensor, condition_tensor)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        loss_history.append(loss_value)
        tracker.log({f"{metric_prefix}/loss": loss_value, f"{metric_prefix}/step": float(step_index + 1)})
        progress.set_postfix(loss=f"{loss_value:.6f}")

    return {
        "final_loss": float(loss_history[-1]),
        "mean_loss": float(np.mean(loss_history)),
        "num_steps": float(num_steps),
    }


def load_or_train_model(
    model: ScoreNetwork,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    diffusion_model: DiffusionModel,
    loader: DataLoader,
    num_steps: int,
    device: torch.device,
    metric_prefix: str,
    tracker: ExperimentTracker,
    overwrite: bool,
    model_kind: str,
) -> Dict[str, Any]:
    """按需加载已有权重或重新训练模型。

    Args:
        model (ScoreNetwork): 目标模型。
        optimizer (torch.optim.Optimizer): 优化器。
        checkpoint_path (Path): 权重保存路径。
        diffusion_model (DiffusionModel): 扩散训练器。
        loader (DataLoader): 训练数据加载器。
        num_steps (int): 优化步数。
        device (torch.device): 训练设备。
        metric_prefix (str): 日志前缀。
        tracker (ExperimentTracker): 可选实验记录器。
        overwrite (bool): 是否强制覆盖已有模型。
        model_kind (str): 模型类型，支持 `behavior` 或 `state`。

    Returns:
        Dict[str, Any]: 训练或加载的结果摘要。

    Raises:
        ValueError: 当模型类型未知时抛出。
    """

    if model_kind not in {MODEL_KIND_BEHAVIOR, MODEL_KIND_STATE}:
        raise ValueError(f"Unknown diffusion model kind: {model_kind}.")

    if checkpoint_path.exists() and not overwrite:
        LOGGER.info("检测到已有模型，直接加载：%s", checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return {"status": "loaded", "checkpoint_path": str(checkpoint_path)}

    LOGGER.info("开始训练模型：%s", checkpoint_path)
    metrics = train_score_model(
        model=model,
        optimizer=optimizer,
        diffusion_model=diffusion_model,
        loader=loader,
        num_steps=num_steps,
        device=device,
        metric_prefix=metric_prefix,
        tracker=tracker,
        model_kind=model_kind,
    )
    torch.save(model.state_dict(), checkpoint_path)
    LOGGER.info("模型保存完成：%s", checkpoint_path)

    metrics.update({"status": "trained", "checkpoint_path": str(checkpoint_path)})
    return metrics




def compute_state_error(
    state_diffusion: ScoreNetwork,
    diffusion_model: DiffusionModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    state_levels: int,
    batch_size: int = 256,
) -> torch.Tensor:
    """计算条件状态扩散模型的下一状态重构误差。

    Args:
        state_diffusion (ScoreNetwork): 条件状态扩散模型。
        diffusion_model (DiffusionModel): 扩散工具对象。
        states (torch.Tensor): 当前状态张量，形状为 `(N, state_dim)`。
        actions (torch.Tensor): 动作张量，形状为 `(N, action_dim)`。
        next_states (torch.Tensor): 下一状态张量，形状为 `(N, state_dim)`。
        state_levels (int): 随机噪声级别采样次数。
        batch_size (int): 误差计算批大小。

    Returns:
        torch.Tensor: 每条 transition 的平均下一状态重构误差，形状为 `(N,)`。

    Raises:
        ValueError: 当输入参数非法时抛出。
    """

    if state_levels <= 0:
        raise ValueError("`state_levels` must be positive.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")
    if not (len(states) == len(actions) == len(next_states)):
        raise ValueError(
            "State/action/next-state batch length mismatch: "
            f"states={len(states)}, actions={len(actions)}, next_states={len(next_states)}."
        )

    recon_errors = []
    with torch.no_grad():
        for start_index in range(0, len(next_states), batch_size):
            batch_states = states[start_index : start_index + batch_size]
            batch_actions = actions[start_index : start_index + batch_size]
            batch_next_states = next_states[start_index : start_index + batch_size]
            batch_condition = torch.cat([batch_states, batch_actions], dim=-1)
            batch_errors = torch.zeros(len(batch_next_states), device=next_states.device)

            for _ in range(state_levels):
                noise_scale = diffusion_model.make_sample_density()(
                    shape=(len(batch_next_states),),
                    device=next_states.device,
                )
                noise = torch.randn_like(batch_next_states)
                noisy_next_states = batch_next_states + noise * append_dims(
                    noise_scale,
                    batch_next_states.ndim,
                )

                c_skip, c_out, c_in = [
                    append_dims(scale, batch_next_states.ndim)
                    for scale in diffusion_model.get_diffusion_scalings(noise_scale)
                ]
                model_output = state_diffusion(
                    noisy_next_states * c_in,
                    batch_condition,
                    torch.log(noise_scale) / 4,
                )
                denoised_next_states = c_skip * noisy_next_states + c_out * model_output
                batch_errors += torch.norm(denoised_next_states - batch_next_states, dim=1)

            batch_errors /= state_levels
            recon_errors.append(batch_errors)

    return torch.cat(recon_errors, dim=0)




def compute_action_error(
    behavior_model: ScoreNetwork,
    diffusion_model: DiffusionModel,
    actions: torch.Tensor,
    states: torch.Tensor,
    action_levels: int,
    batch_size: int = 256,
) -> torch.Tensor:
    """计算条件行为扩散模型的动作重构误差。"""

    if action_levels <= 0:
        raise ValueError("`action_levels` must be positive.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")

    recon_errors = []
    with torch.no_grad():
        for start_index in range(0, len(actions), batch_size):
            batch_actions = actions[start_index : start_index + batch_size]
            batch_states = states[start_index : start_index + batch_size]
            batch_errors = torch.zeros(len(batch_actions), device=actions.device)

            for _ in range(action_levels):
                noise_scale = diffusion_model.make_sample_density()(
                    shape=(len(batch_actions),),
                    device=actions.device,
                )
                noise = torch.randn_like(batch_actions)
                noisy_actions = batch_actions + noise * append_dims(noise_scale, batch_actions.ndim)

                c_skip, c_out, c_in = [
                    append_dims(scale, batch_actions.ndim)
                    for scale in diffusion_model.get_diffusion_scalings(noise_scale)
                ]
                model_output = behavior_model(
                    noisy_actions * c_in,
                    batch_states,
                    torch.log(noise_scale) / 4,
                )
                denoised_actions = c_skip * noisy_actions + c_out * model_output
                batch_errors += torch.norm(denoised_actions - batch_actions, dim=1)

            batch_errors /= action_levels
            recon_errors.append(batch_errors)

    return torch.cat(recon_errors, dim=0)


def get_state_threshold(
    state_diffusion: ScoreNetwork,
    diffusion_model: DiffusionModel,
    env_name: Optional[str] = None,
    no_normalize: bool = False,
    state_levels: int = 1,
    percentile: float = DEFAULT_PERCENTILE,
    batch_size: int = 256,
    device_override: Optional[Any] = None,
    dataset_backend: str = "d4rl",
    dataset_path: Optional[str] = None,
    max_trajectories: Optional[int] = None,
    max_transitions: Optional[int] = None,
    dataset_bundle: Optional[DatasetBundle] = None,
) -> float:
    """在指定数据集上计算条件下一状态 OOD 阈值。

    该函数使用训练集 transition 的 `next_observations` 重构误差分位数
    作为状态阈值，并复用 `(state, action)` 作为条件输入。

    Args:
        state_diffusion (ScoreNetwork): 条件状态扩散模型。
        diffusion_model (DiffusionModel): 扩散工具对象。
        env_name (Optional[str]): D4RL 环境名或推荐环境名。
        no_normalize (bool): D4RL 后端是否关闭归一化。
        state_levels (int): 随机噪声级别采样次数。
        percentile (float): 阈值分位数。
        batch_size (int): 计算批大小。
        device_override (Optional[Any]): 覆盖设备。
        dataset_backend (str): 数据后端类型。
        dataset_path (Optional[str]): 轨迹数据路径。
        max_trajectories (Optional[int]): 最大轨迹数。
        max_transitions (Optional[int]): 最大 transition 数。
        dataset_bundle (Optional[DatasetBundle]): 已加载的数据集对象。

    Returns:
        float: 状态扩散重构误差阈值。

    Raises:
        ValueError: 当数据为空或参数非法时抛出。
    """

    if dataset_bundle is None:
        args = argparse.Namespace(
            dataset_backend=dataset_backend,
            dataset_path=dataset_path,
            env_name=env_name,
            no_normalize=no_normalize,
            max_trajectories=max_trajectories,
            max_transitions=max_transitions,
        )
        dataset_bundle = build_dataset_bundle(args)

    target_device = resolve_device(device_override if device_override is not None else "cpu")
    threshold_loader = create_dataloader(
        dataset=dataset_bundle.state_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    state_diffusion.eval()
    all_errors = []
    for batch_states, batch_actions, batch_next_states in tqdm(
        threshold_loader,
        desc="state_threshold",
        leave=False,
    ):
        batch_states = batch_states.to(target_device, dtype=torch.float32, non_blocking=True)
        batch_actions = batch_actions.to(target_device, dtype=torch.float32, non_blocking=True)
        batch_next_states = batch_next_states.to(target_device, dtype=torch.float32, non_blocking=True)
        batch_errors = compute_state_error(
            state_diffusion=state_diffusion,
            diffusion_model=diffusion_model,
            states=batch_states,
            actions=batch_actions,
            next_states=batch_next_states,
            state_levels=state_levels,
            batch_size=batch_size,
        )
        all_errors.append(batch_errors.cpu().numpy())

    if not all_errors:
        raise ValueError("No state threshold errors were computed.")
    return float(np.percentile(np.concatenate(all_errors, axis=0), percentile))




def get_action_threshold(
    behavior_model: ScoreNetwork,
    diffusion_model: DiffusionModel,
    env_name: Optional[str] = None,
    no_normalize: bool = False,
    action_levels: int = 1,
    percentile: float = DEFAULT_PERCENTILE,
    batch_size: int = 256,
    device_override: Optional[Any] = None,
    dataset_backend: str = "d4rl",
    dataset_path: Optional[str] = None,
    max_trajectories: Optional[int] = None,
    max_transitions: Optional[int] = None,
    dataset_bundle: Optional[DatasetBundle] = None,
) -> float:
    """在指定数据集上计算动作 OOD 阈值。

    Args:
        behavior_model (ScoreNetwork): 条件行为扩散模型。
        diffusion_model (DiffusionModel): 扩散工具对象。
        env_name (Optional[str]): D4RL 环境名或推荐环境名。
        no_normalize (bool): D4RL 后端是否关闭归一化。
        action_levels (int): 随机噪声级别采样次数。
        percentile (float): 阈值分位数。
        batch_size (int): 计算批大小。
        device_override (Optional[Any]): 覆盖设备。
        dataset_backend (str): 数据后端类型。
        dataset_path (Optional[str]): 轨迹数据路径。
        max_trajectories (Optional[int]): 最大轨迹数。
        max_transitions (Optional[int]): 最大 transition 数。
        dataset_bundle (Optional[DatasetBundle]): 已加载的数据集对象。

    Returns:
        float: 行为扩散重构误差阈值。

    Raises:
        ValueError: 当数据为空或参数非法时抛出。
    """

    if dataset_bundle is None:
        args = argparse.Namespace(
            dataset_backend=dataset_backend,
            dataset_path=dataset_path,
            env_name=env_name,
            no_normalize=no_normalize,
            max_trajectories=max_trajectories,
            max_transitions=max_transitions,
        )
        dataset_bundle = build_dataset_bundle(args)

    target_device = resolve_device(device_override if device_override is not None else "cpu")
    threshold_loader = create_dataloader(
        dataset=dataset_bundle.dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    behavior_model.eval()
    all_errors = []
    for batch_states, batch_actions, _ in tqdm(threshold_loader, desc="action_threshold", leave=False):
        batch_states = batch_states.to(target_device, dtype=torch.float32, non_blocking=True)
        batch_actions = batch_actions.to(target_device, dtype=torch.float32, non_blocking=True)
        batch_errors = compute_action_error(
            behavior_model=behavior_model,
            diffusion_model=diffusion_model,
            actions=batch_actions,
            states=batch_states,
            action_levels=action_levels,
            batch_size=batch_size,
        )
        all_errors.append(batch_errors.cpu().numpy())

    if not all_errors:
        raise ValueError("No action threshold errors were computed.")
    return float(np.percentile(np.concatenate(all_errors, axis=0), percentile))




def save_pretrain_metadata(
    args: argparse.Namespace,
    dataset_bundle: DatasetBundle,
    artifact_paths: ArtifactPaths,
    state_threshold: Optional[float],
    action_threshold: Optional[float],
    state_result: Optional[Dict[str, Any]],
    behavior_result: Optional[Dict[str, Any]],
) -> None:
    """保存预训练配置快照和训练指标。

    Args:
        args (argparse.Namespace): 命令行参数。
        dataset_bundle (DatasetBundle): 已加载的数据集摘要。
        artifact_paths (ArtifactPaths): 产物路径集合。
        state_threshold (Optional[float]): 状态扩散阈值。
        action_threshold (Optional[float]): 行为扩散阈值。
        state_result (Optional[Dict[str, Any]]): 状态扩散训练或加载结果。
        behavior_result (Optional[Dict[str, Any]]): 行为扩散训练或加载结果。

    Returns:
        None

    Raises:
        OSError: 当文件写入失败时由底层文件系统抛出。
    """

    config = {
        "dataset_backend": dataset_bundle.dataset_backend,
        "dataset_path": dataset_bundle.dataset_path,
        "artifact_name": args.artifact_name,
        "artifact_dir": str(artifact_paths.artifact_dir),
        "state_dim": dataset_bundle.state_dim,
        "action_dim": dataset_bundle.action_dim,
        "state_condition_dim": dataset_bundle.state_dim + dataset_bundle.action_dim,
        "num_users": dataset_bundle.num_users,
        "num_behavior_transitions": dataset_bundle.num_transitions,
        "num_state_transitions": dataset_bundle.num_state_transitions,
        "normalization": dataset_bundle.normalization,
        "model_types": {
            "behavior_diffusion": "conditional_action_distribution_p_a_given_s",
            "state_diffusion": "conditional_next_state_distribution_p_s_next_given_s_a",
            "dynamic_model": "disabled_for_recommender_state_tracker_setting",
        },
        "training_args": namespace_to_dict(args),
        "checkpoints": {
            "behavior_diffusion": str(artifact_paths.behavior_model_path),
            "state_diffusion": str(artifact_paths.state_diffusion_path),
        },
    }
    metrics = {
        "state_threshold": None if state_threshold is None else float(state_threshold),
        "action_threshold": None if action_threshold is None else float(action_threshold),
        "percentile": float(args.percentile),
        "state_model_result": state_result,
        "behavior_model_result": behavior_result,
    }

    with artifact_paths.config_path.open("w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, ensure_ascii=False, indent=2)
    with artifact_paths.metrics_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2)
    LOGGER.info("预训练配置已保存：%s", artifact_paths.config_path)
    LOGGER.info("训练指标已保存：%s", artifact_paths.metrics_path)




def train(args: argparse.Namespace) -> Dict[str, Any]:
    """执行完整的 diffusion 预训练流程。

    Args:
        args (argparse.Namespace): 命令行参数对象。

    Returns:
        Dict[str, Any]: 关键训练结果摘要。

    Raises:
        FileNotFoundError: 当数据文件不存在时抛出。
        ValueError: 当参数非法或数据格式错误时抛出。
        ImportError: 当 D4RL 后端缺少依赖时抛出。
    """

    configure_logging()
    set_seed(args.seed)

    if not args.train_behavior and not args.train_state:
        raise ValueError("At least one of `--train_behavior` or `--train_state` must be enabled.")

    device = resolve_device(args.device)
    args.device = str(device)

    if args.dataset_backend == "trajectory_pkl":
        args.no_normalize = True
        LOGGER.info("trajectory_pkl 后端默认关闭状态归一化。")

    raw_artifact_name = args.artifact_name
    args.artifact_name = resolve_artifact_name(
        env_name=args.env_name,
        dataset_path=args.dataset_path,
        artifact_name=args.artifact_name,
    )

    artifact_paths = resolve_artifact_paths(
        save_root=args.save_root,
        env_name=args.env_name,
        artifact_name=raw_artifact_name,
    )
    dataset_bundle = build_dataset_bundle(args)

    LOGGER.info(
        "数据准备完成：backend=%s, path=%s, users=%s, behavior_transitions=%s, "
        "state_transitions=%s, state_dim=%s, action_dim=%s",
        dataset_bundle.dataset_backend,
        dataset_bundle.dataset_path,
        dataset_bundle.num_users,
        dataset_bundle.num_transitions,
        dataset_bundle.num_state_transitions,
        dataset_bundle.state_dim,
        dataset_bundle.action_dim,
    )

    behavior_loader = create_dataloader(
        dataset=dataset_bundle.dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    state_loader = create_dataloader(
        dataset=dataset_bundle.state_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    diffusion_model = DiffusionModel(
        sigma_data=args.sigma_data,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        device=str(device),
    )
    state_diffusion = build_state_diffusion_model(
        state_dim=dataset_bundle.state_dim,
        action_dim=dataset_bundle.action_dim,
        device=device,
    )
    behavior_model = build_behavior_model(dataset_bundle.state_dim, dataset_bundle.action_dim, device)

    state_optimizer = torch.optim.Adam(state_diffusion.parameters(), lr=DEFAULT_LEARNING_RATE)
    behavior_optimizer = torch.optim.Adam(behavior_model.parameters(), lr=DEFAULT_LEARNING_RATE)

    tracker = ExperimentTracker(
        enabled=args.enable_swanlab,
        project=args.swanlab_project,
        run_name=args.swanlab_run_name or f"{args.env_name}-{args.artifact_name}",
        config=namespace_to_dict(args),
        mode=args.swanlab_mode,
    )

    state_threshold = None
    action_threshold = None
    state_result = None
    behavior_result = None

    try:
        if args.train_state:
            state_result = load_or_train_model(
                model=state_diffusion,
                optimizer=state_optimizer,
                checkpoint_path=artifact_paths.state_diffusion_path,
                diffusion_model=diffusion_model,
                loader=state_loader,
                num_steps=args.pretrain_epochs,
                device=device,
                metric_prefix="state_diffusion",
                tracker=tracker,
                overwrite=args.overwrite,
                model_kind=MODEL_KIND_STATE,
            )
            state_diffusion.eval()
        else:
            LOGGER.info("跳过状态扩散模型训练：train_state=False。")

        if args.train_behavior:
            behavior_result = load_or_train_model(
                model=behavior_model,
                optimizer=behavior_optimizer,
                checkpoint_path=artifact_paths.behavior_model_path,
                diffusion_model=diffusion_model,
                loader=behavior_loader,
                num_steps=args.pretrain_epochs,
                device=device,
                metric_prefix="behavior_diffusion",
                tracker=tracker,
                overwrite=args.overwrite,
                model_kind=MODEL_KIND_BEHAVIOR,
            )
            behavior_model.eval()
        else:
            LOGGER.info("跳过行为扩散模型训练：train_behavior=False。")

        # 为了让“首次训练”和“仅加载已有权重”两种路径得到一致的阈值，
        # 在阈值计算前显式重置随机种子。
        set_seed(args.seed)
        if args.train_state:
            state_threshold = get_state_threshold(
                state_diffusion=state_diffusion,
                diffusion_model=diffusion_model,
                state_levels=args.state_levels,
                percentile=args.percentile,
                batch_size=args.batch_size,
                device_override=device,
                dataset_bundle=dataset_bundle,
            )
        if args.train_behavior:
            action_threshold = get_action_threshold(
                behavior_model=behavior_model,
                diffusion_model=diffusion_model,
                action_levels=args.action_levels,
                percentile=args.percentile,
                batch_size=args.batch_size,
                device_override=device,
                dataset_bundle=dataset_bundle,
            )
        LOGGER.info(
            "阈值计算完成：state_threshold=%s, action_threshold=%s",
            None if state_threshold is None else f"{state_threshold:.6f}",
            None if action_threshold is None else f"{action_threshold:.6f}",
        )

        save_pretrain_metadata(
            args=args,
            dataset_bundle=dataset_bundle,
            artifact_paths=artifact_paths,
            state_threshold=state_threshold,
            action_threshold=action_threshold,
            state_result=state_result,
            behavior_result=behavior_result,
        )
    finally:
        tracker.finish()

    return {
        "artifact_dir": str(artifact_paths.artifact_dir),
        "behavior_model_path": str(artifact_paths.behavior_model_path),
        "state_diffusion_path": str(artifact_paths.state_diffusion_path),
        "config_path": str(artifact_paths.config_path),
        "metrics_path": str(artifact_paths.metrics_path),
        "state_threshold": None if state_threshold is None else float(state_threshold),
        "action_threshold": None if action_threshold is None else float(action_threshold),
    }




def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default=DEFAULT_ENV_NAME)
    parser.add_argument("--dataset_backend", type=str, choices=["trajectory_pkl", "d4rl"], default="trajectory_pkl")
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_TRAJECTORY_DATASET_PATH))
    parser.add_argument("--artifact_name", type=str, default=None)
    parser.add_argument("--save_root", type=str, default=str(DEFAULT_SAVE_ROOT))
    parser.add_argument("--pretrain_epochs", type=int, default=100000)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--no_normalize", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--train_behavior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否训练行为扩散模型 p(a | s)，默认开启。",
    )
    parser.add_argument(
        "--train_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否训练条件状态扩散模型 p(s_next | s, a)，默认开启。",
    )
    parser.add_argument(
        "--enable_swanlab",
        action="store_true",
        default=False,
        help="是否启用 SwanLab 记录训练指标，默认关闭。",
    )
    parser.add_argument(
        "--enable_wandb",
        dest="enable_swanlab",
        action="store_true",
        help="兼容旧参数名；等价于 --enable_swanlab。",
    )
    parser.add_argument("--swanlab_project", type=str, default=DEFAULT_SWANLAB_PROJECT)
    parser.add_argument(
        "--wandb_project",
        dest="swanlab_project",
        type=str,
        help="兼容旧参数名；等价于 --swanlab_project。",
    )
    parser.add_argument("--swanlab_run_name", type=str, default=None)
    parser.add_argument(
        "--swanlab_mode",
        type=str,
        default=None,
        help="可选 SwanLab 运行模式，例如 disabled 或 offline。",
    )
    parser.add_argument("--max_trajectories", type=int, default=None)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--action_levels", type=int, default=1)
    parser.add_argument("--state_levels", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    TRAIN_ARGS = parse_args()
    TRAIN_RESULTS = train(TRAIN_ARGS)
    LOGGER.info("预训练完成，结果摘要：%s", json.dumps(TRAIN_RESULTS, ensure_ascii=False, indent=2))
