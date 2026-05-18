"""推荐系统版 DORL-DOSER 策略实现。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Discrete
from logzero import logger
from torch import nn
from torch.distributions import Categorical

from src.core.diffusion_doser.karras import DiffusionModel
from src.core.diffusion_doser.mlps import ScoreNetwork
from src.core.envs.Simulated_Env.penalty_ent_exp import (
    get_features_of_last_n_items_features,
)
from src.core.util.utils import clip0, compute_action_distance, compute_exposure
from src.core.util.wandb_utils import load_wandb
from src.tianshou.tianshou.data import Batch, ReplayBuffer
from src.tianshou.tianshou.policy import BasePolicy

wandb = load_wandb(repo_root=Path(__file__).resolve().parents[3])


MASKED_LOGIT_VALUE = -1e9
"""离散动作 mask 后使用的极小 logits。"""

DEFAULT_LOG_INTERVAL = 100
"""训练阶段向实验日志汇报指标的默认步数间隔。"""

DEFAULT_RECON_BATCH_SIZE = 256
"""扩散重建误差计算时的默认分块大小。"""

DEFAULT_DIFFUSION_HIDDEN_DIM = 256
"""DOSER 扩散网络的默认隐藏层维度。"""

DEFAULT_DIFFUSION_TIME_EMBED_DIM = 16
"""DOSER 扩散网络的默认时间嵌入维度。"""

DEFAULT_DIFFUSION_HIDDEN_LAYERS = 4
"""DOSER 扩散网络的默认隐藏层数。"""

DEFAULT_EPS = 1e-8
"""数值稳定性保护常量。"""


@dataclass
class DiffusionArtifact:
    """封装 DOSER 扩散预训练产物及其元信息。"""

    artifact_dir: Path
    meta: Dict[str, Any]
    diffusion_model: DiffusionModel
    behavior_model: ScoreNetwork
    state_distribution: ScoreNetwork
    state_dim: int
    action_dim: int
    state_threshold: float
    action_threshold: float


@dataclass
class RewardModelConfig:
    """描述推荐环境奖励重建所需配置。"""

    env_name: str
    predicted_mat: np.ndarray
    real_env: Any
    version: str
    tau: float
    use_exposure_intervention: bool
    gamma_exposure: float
    alpha_u: Optional[np.ndarray]
    beta_i: Optional[np.ndarray]
    entropy_dict: Dict[str, Any]
    entropy_window: Sequence[int]
    lambda_entropy: float
    step_n_actions: int
    entropy_min: float
    entropy_max: float
    feature_level: bool
    map_item_feat: Optional[Dict[int, Any]]
    is_sorted: bool


def build_mlp(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    activation_cls: type[nn.Module] = nn.GELU,
) -> nn.Sequential:
    """构建简单 MLP 网络。

    Args:
        input_dim (int): 输入维度。
        hidden_sizes (Sequence[int]): 隐藏层维度列表，至少包含一个元素。
        output_dim (int): 输出维度。
        activation_cls (type[nn.Module]): 激活函数类型。

    Returns:
        nn.Sequential: 构建完成的前馈网络。

    Raises:
        ValueError: 当 `hidden_sizes` 为空时抛出。
    """

    if not hidden_sizes:
        raise ValueError("hidden_sizes must contain at least one hidden layer.")

    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation_cls())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def resolve_diffusion_artifact_dir(
    save_root: Union[str, Path],
    env_name: str,
    artifact_name: str,
) -> Path:
    """解析 DOSER 扩散模型保存目录。

    Args:
        save_root (Union[str, Path]): 模型保存根目录。
        env_name (str): 环境名称。
        artifact_name (str): 数据产物名称。

    Returns:
        Path: 规范化后的扩散模型目录。
    """

    return Path(save_root) / env_name / "DOSER" / "diffusion" / artifact_name


def _load_json_file(json_path: Path) -> Dict[str, Any]:
    """读取 JSON 文件并返回字典对象。

    Args:
        json_path (Path): JSON 文件路径。

    Returns:
        Dict[str, Any]: 解析后的 JSON 内容。

    Raises:
        FileNotFoundError: 当文件不存在时抛出。
        ValueError: 当 JSON 解析失败时抛出。
    """

    if not json_path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {json_path}")
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Failed to parse JSON file: {json_path}") from error


def _resolve_checkpoint_path(
    artifact_dir: Path,
    meta: Dict[str, Any],
    checkpoint_key: str,
    default_name: str,
) -> Path:
    """根据元信息解析 checkpoint 路径。

    Args:
        artifact_dir (Path): 扩散产物目录。
        meta (Dict[str, Any]): `pretrain_meta.json` 内容。
        checkpoint_key (str): 在 `meta["checkpoints"]` 中的键名。
        default_name (str): 默认文件名。

    Returns:
        Path: checkpoint 文件路径。
    """

    checkpoint_info = meta.get("checkpoints", {})
    raw_path = checkpoint_info.get(checkpoint_key)
    if raw_path is None:
        return artifact_dir / default_name
    checkpoint_path = Path(raw_path)
    if checkpoint_path.is_absolute():
        return checkpoint_path
    if checkpoint_path.parts[: len(artifact_dir.parts)] == artifact_dir.parts:
        return checkpoint_path
    return Path(raw_path)


def load_diffusion_artifact(
    save_root: Union[str, Path],
    env_name: str,
    artifact_name: str,
    device: Union[str, torch.device],
) -> DiffusionArtifact:
    """加载 DOSER 扩散预训练模型及阈值元信息。

    Args:
        save_root (Union[str, Path]): 模型保存根目录。
        env_name (str): 环境名称。
        artifact_name (str): 扩散产物名。
        device (Union[str, torch.device]): 目标设备。

    Returns:
        DiffusionArtifact: 已加载的扩散模型及阈值。

    Raises:
        FileNotFoundError: 当模型文件或元信息不存在时抛出。
        ValueError: 当元信息字段缺失时抛出。
    """

    artifact_dir = resolve_diffusion_artifact_dir(save_root, env_name, artifact_name)
    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"Diffusion artifact directory does not exist: {artifact_dir}"
        )

    meta = _load_json_file(artifact_dir / "pretrain_meta.json")
    required_fields = [
        "state_dim",
        "action_dim",
        "state_threshold",
        "action_threshold",
    ]
    missing_fields = [field for field in required_fields if field not in meta]
    if missing_fields:
        raise ValueError(
            f"pretrain_meta.json is missing required fields: {missing_fields}"
        )

    training_args = meta.get("training_args", {})
    state_dim = int(meta["state_dim"])
    action_dim = int(meta["action_dim"])
    device_str = str(device)
    torch_device = torch.device(device_str)

    diffusion_model = DiffusionModel(
        sigma_data=float(training_args.get("sigma_data", 0.5)),
        sigma_min=float(training_args.get("sigma_min", 0.002)),
        sigma_max=float(training_args.get("sigma_max", 80.0)),
        device=device_str,
    )

    behavior_model = ScoreNetwork(
        x_dim=action_dim,
        hidden_dim=DEFAULT_DIFFUSION_HIDDEN_DIM,
        time_embed_dim=DEFAULT_DIFFUSION_TIME_EMBED_DIM,
        cond_dim=state_dim,
        cond_mask_prob=0.0,
        num_hidden_layers=DEFAULT_DIFFUSION_HIDDEN_LAYERS,
        output_dim=action_dim,
        device=device_str,
        cond_conditional=True,
    ).to(torch_device)
    state_distribution = ScoreNetwork(
        x_dim=state_dim,
        hidden_dim=DEFAULT_DIFFUSION_HIDDEN_DIM,
        time_embed_dim=DEFAULT_DIFFUSION_TIME_EMBED_DIM,
        cond_dim=0,
        cond_mask_prob=0.0,
        num_hidden_layers=DEFAULT_DIFFUSION_HIDDEN_LAYERS,
        output_dim=state_dim,
        device=device_str,
        cond_conditional=False,
    ).to(torch_device)

    behavior_model_path = _resolve_checkpoint_path(
        artifact_dir, meta, "behavior_model", "behavior_model.pth"
    )
    state_model_path = _resolve_checkpoint_path(
        artifact_dir, meta, "state_distribution", "state_distribution.pth"
    )
    if not behavior_model_path.exists():
        raise FileNotFoundError(
            f"Behavior diffusion checkpoint does not exist: {behavior_model_path}"
        )
    if not state_model_path.exists():
        raise FileNotFoundError(
            f"State diffusion checkpoint does not exist: {state_model_path}"
        )

    behavior_model.load_state_dict(
        torch.load(behavior_model_path, map_location=torch_device)
    )
    state_distribution.load_state_dict(
        torch.load(state_model_path, map_location=torch_device)
    )
    behavior_model.eval()
    state_distribution.eval()

    logger.info(
        "Loaded DOSER diffusion artifact from %s (state_dim=%s, action_dim=%s)",
        artifact_dir,
        state_dim,
        action_dim,
    )
    return DiffusionArtifact(
        artifact_dir=artifact_dir,
        meta=meta,
        diffusion_model=diffusion_model,
        behavior_model=behavior_model,
        state_distribution=state_distribution,
        state_dim=state_dim,
        action_dim=action_dim,
        state_threshold=float(meta["state_threshold"]),
        action_threshold=float(meta["action_threshold"]),
    )


def _safe_wandb_log(metrics: Dict[str, float], step: int) -> None:
    """在 wandb 可用且已初始化时写入日志。

    Args:
        metrics (Dict[str, float]): 需要记录的标量指标。
        step (int): 当前训练步。
    """

    if wandb is None:
        return
    if getattr(wandb, "run", None) is None:
        return
    wandb.log(metrics, step=step)


class DiscreteActorNetwork(nn.Module):
    """推荐系统离散动作 actor。

    该网络输入 state tracker 编码后的状态向量，输出全物品空间的 logits，
    供策略在离散 item id 空间中采样或贪心选择动作。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        """初始化离散 actor。

        Args:
            state_dim (int): 状态向量维度。
            action_dim (int): 离散动作数，即物品数量。
            hidden_sizes (Sequence[int]): 隐藏层配置。
        """

        super().__init__()
        self.network = build_mlp(
            input_dim=state_dim,
            hidden_sizes=hidden_sizes,
            output_dim=action_dim,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """根据状态输出离散动作 logits。

        Args:
            state (torch.Tensor): 形状为 `(batch_size, state_dim)` 的状态张量。

        Returns:
            torch.Tensor: 形状为 `(batch_size, num_items)` 的动作 logits。
        """

        return self.network(state)


class DORLCriticNetwork(nn.Module):
    """DOSER 风格的四头 Q 网络与双头 V 网络。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        """初始化 critic。

        Args:
            state_dim (int): 状态向量维度。
            action_dim (int): 动作 embedding 维度。
            hidden_sizes (Sequence[int]): 隐藏层配置。
        """

        super().__init__()
        q_input_dim = state_dim + action_dim
        self.q1 = build_mlp(q_input_dim, hidden_sizes, 1)
        self.q2 = build_mlp(q_input_dim, hidden_sizes, 1)
        self.q3 = build_mlp(q_input_dim, hidden_sizes, 1)
        self.q4 = build_mlp(q_input_dim, hidden_sizes, 1)
        self.v1 = build_mlp(state_dim, hidden_sizes, 1)
        self.v2 = build_mlp(state_dim, hidden_sizes, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """同时计算四个 Q 头的输出。

        Args:
            state (torch.Tensor): 状态张量。
            action (torch.Tensor): 动作 embedding 张量。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            四个 Q 头的输出。
        """

        critic_input = torch.cat([state, action], dim=-1)
        return (
            self.q1(critic_input),
            self.q2(critic_input),
            self.q3(critic_input),
            self.q4(critic_input),
        )

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """计算四个 Q 头中的最小值。"""

        q1, q2, q3, q4 = self.forward(state, action)
        return torch.min(torch.min(q1, q2), torch.min(q3, q4))

    def v(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算双 V 头输出。"""

        return self.v1(state), self.v2(state)

    def v_min(self, state: torch.Tensor) -> torch.Tensor:
        """计算双 V 头中的最小值。"""

        v1, v2 = self.v(state)
        return torch.min(v1, v2)


class CounterfactualRewardModel:
    """离线重建推荐环境即时奖励的辅助类。

    该类复用了 `PenaltyEntExpSimulatedEnv` 的奖励定义，包括预测奖励、
    entropy bonus 与 exposure intervention，确保 DOSER 的 counterfactual
    next-state 构造与当前训练环境保持一致。
    """

    def __init__(self, config: RewardModelConfig) -> None:
        """初始化奖励重建器。

        Args:
            config (RewardModelConfig): 奖励构造所需配置。
        """

        self.config = config
        self.min_reward = (
            float(np.min(self.config.predicted_mat))
            + self.config.lambda_entropy * self.config.entropy_min
        )

    def estimate_rewards(
        self,
        user_ids: np.ndarray,
        history_actions: Sequence[List[int]],
        action_ids: np.ndarray,
    ) -> np.ndarray:
        """批量估计 counterfactual reward。

        Args:
            user_ids (np.ndarray): 用户 id，形状为 `(batch_size,)`。
            history_actions (Sequence[List[int]]): 每个样本当前时刻之前的动作历史。
            action_ids (np.ndarray): 候选动作 id，形状为 `(batch_size,)`。

        Returns:
            np.ndarray: 估计得到的即时奖励，形状为 `(batch_size,)`。
        """

        rewards = np.zeros(len(user_ids), dtype=np.float32)
        for index, (user_id, history, action_id) in enumerate(
            zip(user_ids, history_actions, action_ids)
        ):
            rewards[index] = self._estimate_single_reward(
                int(user_id),
                history,
                int(action_id),
            )
        return rewards

    def _estimate_single_reward(
        self,
        user_id: int,
        history_actions: List[int],
        action_id: int,
    ) -> float:
        """估计单个样本在候选动作下的即时奖励。"""

        pred_reward = float(self.config.predicted_mat[user_id, action_id])
        entropy_bonus = self._compute_entropy_bonus(history_actions, action_id)
        penalized_reward = (
            pred_reward + self.config.lambda_entropy * entropy_bonus - self.min_reward
        )
        exposure_effect = self._compute_exposure_effect(
            user_id=user_id,
            history_actions=history_actions,
            action_id=action_id,
        )

        if self.config.version == "v1":
            final_reward = clip0(penalized_reward) / (1.0 + exposure_effect)
        else:
            final_reward = clip0(penalized_reward - exposure_effect)
        return float(max(0.0, final_reward))

    def _compute_entropy_bonus(
        self,
        history_actions: List[int],
        action_id: int,
    ) -> float:
        """按环境公式计算 entropy bonus。"""

        entropy_map = self.config.entropy_dict.get("map")
        entropy_set = set(self.config.entropy_window) - {0}
        if not entropy_set or entropy_map is None:
            return 0.0

        history_with_action = list(history_actions) + [int(action_id)]
        if self.config.step_n_actions > 0:
            action_window = history_with_action[-self.config.step_n_actions :]
        else:
            action_window = history_with_action

        if hasattr(self.config.real_env, "lbe_item") and self.config.real_env.lbe_item:
            action_trans = self.config.real_env.lbe_item.inverse_transform(action_window)
        else:
            action_trans = np.asarray(action_window)

        entropy_bonus = 0.0
        for entropy_width in entropy_set:
            if len(action_trans) < entropy_width:
                entropy_bonus += 1.0
                continue

            if self.config.feature_level:
                feature_histories = get_features_of_last_n_items_features(
                    entropy_width,
                    action_trans,
                    self.config.map_item_feat,
                    is_sort=self.config.is_sorted,
                )
                if len(feature_histories) == 0:
                    entropy_bonus += 1.0
                    continue
                feature_entropy = 0.0
                for feature_history in feature_histories:
                    feature_entropy += float(entropy_map.get(feature_history, 1.0))
                entropy_bonus += feature_entropy / len(feature_histories)
            else:
                history_tuple = tuple(
                    sorted(action_trans[-entropy_width:])
                    if self.config.is_sorted
                    else action_trans[-entropy_width:]
                )
                entropy_bonus += float(entropy_map.get(history_tuple, 1.0))
        return float(entropy_bonus)

    def _compute_exposure_effect(
        self,
        user_id: int,
        history_actions: List[int],
        action_id: int,
    ) -> float:
        """按环境公式计算 exposure penalty。"""

        if not self.config.use_exposure_intervention:
            return 0.0
        if len(history_actions) == 0:
            return 0.0

        history_array = np.asarray(history_actions, dtype=np.int64)
        distance = compute_action_distance(
            action_id,
            history_array,
            self.config.env_name,
            self.config.real_env,
        )
        time_delta = len(history_actions) - np.arange(len(history_actions))
        exposure_effect = compute_exposure(time_delta, distance, self.config.tau)

        if self.config.alpha_u is not None and self.config.beta_i is not None:
            transformed_user_id = user_id
            transformed_item_id = action_id
            if hasattr(self.config.real_env, "lbe_user") and self.config.real_env.lbe_user:
                transformed_user_id = self.config.real_env.lbe_user.inverse_transform(
                    [user_id]
                )[0]
            if hasattr(self.config.real_env, "lbe_item") and self.config.real_env.lbe_item:
                transformed_item_id = self.config.real_env.lbe_item.inverse_transform(
                    [action_id]
                )[0]
            exposure_effect = (
                exposure_effect
                * self.config.alpha_u[transformed_user_id]
                * self.config.beta_i[transformed_item_id]
            )
        return float(exposure_effect * self.config.gamma_exposure)


class DORLDOSEROODHelper:
    """封装 DOSER 扩散重建与 counterfactual 相关的共享工具。

    该辅助类不关心具体训练范式，只依赖宿主 policy 上已经初始化好的
    `state_tracker`、diffusion artifact、reward model 与 critic 接口，
    供 off-policy 与 on-policy 两套 DORL-DOSER 共享。
    """

    def __init__(self, owner: Any) -> None:
        """保存宿主 policy 的引用。

        Args:
            owner (Any): 宿主 policy，需要暴露本 helper 所依赖的属性。
        """

        self.owner = owner

    def get_all_item_embeddings(self) -> torch.Tensor:
        """读取当前 state tracker 中的全量物品 embedding。"""

        item_ids = np.arange(self.owner.num_items, dtype=np.int64).reshape(-1, 1)
        return self.owner.state_tracker.get_embedding(item_ids, "action")

    def lookup_action_embeddings(self, action_ids: torch.Tensor) -> torch.Tensor:
        """根据离散 item id 查询动作 embedding。"""

        action_array = action_ids.detach().cpu().numpy().reshape(-1, 1)
        return self.owner.state_tracker.get_embedding(action_array, "action")

    def extract_action_histories(self, indices: np.ndarray) -> List[List[int]]:
        """根据 replay buffer 还原每个样本的历史动作序列。"""

        if len(indices) == 0:
            return []

        histories: List[List[int]] = [[] for _ in range(len(indices))]
        cursor = np.array(indices, copy=True)
        live_mask = np.ones(len(indices), dtype=bool)
        while np.any(live_mask):
            current_obs = self.normalize_obs_array(self.owner._buffer.obs[cursor])
            current_actions = current_obs[:, 1]
            for row_index, is_live in enumerate(live_mask):
                if not is_live:
                    continue
                action_id = int(current_actions[row_index])
                if 0 <= action_id < self.owner.num_items:
                    histories[row_index].append(action_id)
            episode_start_mask = np.asarray(self.owner._buffer.is_start[cursor]).reshape(-1)
            live_mask[episode_start_mask] = False
            cursor = self.owner._buffer.prev(cursor)

        for history in histories:
            history.reverse()
        return histories

    def normalize_obs_array(self, obs: Any) -> np.ndarray:
        """把 replay buffer 中不同形状的观测统一整理为二维数组。"""

        obs_array = np.asarray(obs)
        if obs_array.ndim == 1:
            return obs_array.reshape(1, -1)
        if obs_array.ndim == 2:
            return obs_array
        return obs_array.reshape(-1, obs_array.shape[-1])

    def build_counterfactual_next_state(
        self,
        indices: np.ndarray,
        user_ids: np.ndarray,
        action_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """通过 synthetic batch 构造 counterfactual next-state。"""

        action_array = action_ids.detach().cpu().numpy().astype(np.int64)
        histories = self.extract_action_histories(indices)
        rewards = self.owner.reward_model.estimate_rewards(user_ids, histories, action_array)
        synthetic_obs = np.stack([user_ids, action_array], axis=1).astype(np.int64)
        synthetic_batch = Batch(obs=synthetic_obs, rew_prev=rewards)
        next_state = self.owner.state_tracker(
            buffer=self.owner._buffer,
            indices=indices,
            is_obs=True,
            batch=synthetic_batch,
            is_train=True,
            use_batch_in_statetracker=True,
        )
        return next_state, rewards

    def map_action_embeddings_to_item_ids(
        self, action_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """将连续动作 embedding 映射回最近的离散 item id。"""

        with torch.no_grad():
            item_embeddings = self.get_all_item_embeddings()
            normalized_item_embeddings = F.normalize(item_embeddings, dim=-1)
            normalized_action_embeddings = F.normalize(action_embeddings, dim=-1)
            similarity = torch.matmul(
                normalized_action_embeddings, normalized_item_embeddings.transpose(0, 1)
            )
            return similarity.argmax(dim=-1)

    def select_best_id_action(
        self,
        state_emb: torch.Tensor,
        action_mask: Optional[torch.Tensor],
        critic: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从行为扩散候选中筛选 critic 认为最优的 ID 动作。"""

        target_critic = self.owner.critic if critic is None else critic
        sampled_action_embeddings = self.owner.diffusion_model.sample(
            model=self.owner.behavior_model,
            cond=state_emb.detach(),
            action_samples=self.owner.action_samples,
            n_steps=self.owner.diffusion_sample_steps,
        )
        batch_size = sampled_action_embeddings.shape[0]
        flat_embeddings = sampled_action_embeddings.view(-1, self.owner.action_dim)
        candidate_action_ids = self.map_action_embeddings_to_item_ids(flat_embeddings).view(
            batch_size, self.owner.action_samples
        )
        candidate_action_embeddings = self.lookup_action_embeddings(
            candidate_action_ids.reshape(-1)
        ).view(batch_size, self.owner.action_samples, self.owner.action_dim)

        expanded_state = state_emb.unsqueeze(1).expand(-1, self.owner.action_samples, -1)
        flat_state = expanded_state.reshape(-1, self.owner.state_dim)
        flat_action = candidate_action_embeddings.reshape(-1, self.owner.action_dim)
        candidate_q = target_critic.q_min(flat_state, flat_action).view(
            batch_size, self.owner.action_samples
        )
        masked_candidate_q = candidate_q.clone()

        if action_mask is not None:
            safe_mask = action_mask.to(self.owner.device).bool()
            candidate_mask = torch.gather(
                safe_mask,
                dim=1,
                index=candidate_action_ids,
            )
            masked_candidate_q = masked_candidate_q.masked_fill(
                ~candidate_mask,
                float("-inf"),
            )
            invalid_rows = ~candidate_mask.any(dim=-1)
            if invalid_rows.any():
                masked_candidate_q[invalid_rows] = candidate_q[invalid_rows]

        best_indices = masked_candidate_q.argmax(dim=-1)
        batch_indices = torch.arange(batch_size, device=self.owner.device)
        best_action_ids = candidate_action_ids[batch_indices, best_indices]
        best_action_embeddings = candidate_action_embeddings[batch_indices, best_indices]
        best_q = candidate_q[batch_indices, best_indices].unsqueeze(-1)
        return best_action_ids, best_action_embeddings, best_q

    def compute_state_error(
        self,
        states: torch.Tensor,
        batch_size: int = DEFAULT_RECON_BATCH_SIZE,
    ) -> torch.Tensor:
        """基于状态扩散模型计算 state reconstruction error。"""

        reconstruction_errors: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(states), batch_size):
                batch_states = states[start : start + batch_size]
                sample_density = self.owner.diffusion_model.make_sample_density()
                sigma = sample_density(
                    shape=(len(batch_states),),
                    device=states.device,
                )
                noise = torch.randn_like(batch_states)
                sigma_expanded = sigma.view(-1, 1)
                noisy_states = batch_states + noise * sigma_expanded
                c_skip, c_out, c_in = [
                    value.view(-1, 1)
                    for value in self.owner.diffusion_model.get_diffusion_scalings(sigma)
                ]
                model_input = noisy_states * c_in
                model_output = self.owner.state_distribution(
                    model_input,
                    None,
                    torch.log(sigma) / 4,
                )
                denoised_states = c_skip * noisy_states + c_out * model_output
                reconstruction_errors.append(
                    torch.norm(denoised_states - batch_states, dim=-1)
                )
        return torch.cat(reconstruction_errors, dim=0)

    def compute_action_error(
        self,
        actions: torch.Tensor,
        states: torch.Tensor,
        batch_size: int = DEFAULT_RECON_BATCH_SIZE,
    ) -> torch.Tensor:
        """基于行为扩散模型计算 action reconstruction error。"""

        reconstruction_errors: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(actions), batch_size):
                batch_actions = actions[start : start + batch_size]
                batch_states = states[start : start + batch_size]
                sample_density = self.owner.diffusion_model.make_sample_density()
                sigma = sample_density(
                    shape=(len(batch_actions),),
                    device=actions.device,
                )
                noise = torch.randn_like(batch_actions)
                sigma_expanded = sigma.view(-1, 1)
                noisy_actions = batch_actions + noise * sigma_expanded
                c_skip, c_out, c_in = [
                    value.view(-1, 1)
                    for value in self.owner.diffusion_model.get_diffusion_scalings(sigma)
                ]
                model_input = noisy_actions * c_in
                model_output = self.owner.behavior_model(
                    model_input,
                    batch_states,
                    torch.log(sigma) / 4,
                )
                denoised_actions = c_skip * noisy_actions + c_out * model_output
                reconstruction_errors.append(
                    torch.norm(denoised_actions - batch_actions, dim=-1)
                )
        return torch.cat(reconstruction_errors, dim=0)


class DORLDOSERPolicy(BasePolicy):
    """离散推荐系统版 DORL-DOSER 策略。

    该策略复用了推荐系统训练栈中的 `RecPolicy + StateTracker + VectorReplayBuffer`
    链路，并在离散 item id 空间中实现 DOSER 风格的 actor-critic 与 OOD 识别。
    """

    def __init__(
        self,
        actor: DiscreteActorNetwork,
        critic: DORLCriticNetwork,
        optim: Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
        state_tracker: nn.Module,
        diffusion_artifact: DiffusionArtifact,
        reward_model: CounterfactualRewardModel,
        action_dim: int,
        discount_factor: float = 0.99,
        tau: float = 0.005,
        doser_beta: float = 0.001,
        doser_lam: float = 0.001,
        doser_eta: float = 0.9,
        doser_expectile: float = 0.9,
        doser_q_min: float = 0.0,
        doser_action_samples: int = 10,
        doser_policy_freq: int = 2,
        doser_target_update_freq: int = 2,
        diffusion_sample_steps: int = 20,
        exploration_eps: float = 0.0,
        log_interval: int = DEFAULT_LOG_INTERVAL,
        catalog_chunk_size: Optional[int] = 512,
        **kwargs: Any,
    ) -> None:
        """初始化推荐系统版 DORL-DOSER。

        Args:
            actor (DiscreteActorNetwork): 离散 actor。
            critic (DORLCriticNetwork): DOSER 风格 critic。
            optim (Tuple[torch.optim.Optimizer, torch.optim.Optimizer]): RL 与 state
                tracker 的优化器。
            state_tracker (nn.Module): 推荐系统状态编码器。
            diffusion_artifact (DiffusionArtifact): 扩散模型及阈值。
            reward_model (CounterfactualRewardModel): counterfactual reward 重建器。
            action_dim (int): 动作 embedding 维度。
            discount_factor (float): 折扣因子。
            tau (float): 目标网络软更新系数。
            doser_beta (float): negative OOD penalty 系数。
            doser_lam (float): positive OOD compensation 系数。
            doser_eta (float): compensation target 系数。
            doser_expectile (float): expectile value loss 系数。
            doser_q_min (float): Q 最小先验值。
            doser_action_samples (int): 行为扩散每个状态采样的动作数。
            doser_policy_freq (int): actor 更新频率。
            doser_target_update_freq (int): 目标网络更新频率。
            diffusion_sample_steps (int): 扩散采样步数。
            exploration_eps (float): epsilon-greedy 探索概率。
            log_interval (int): 日志打印间隔。
            catalog_chunk_size (Optional[int]): 全 catalog Q 评估的分块大小。
        """

        super().__init__(
            action_space=Discrete(state_tracker.num_item),
            action_scaling=False,
            action_bound_method="",
            **kwargs,
        )
        self.actor = actor
        self.actor_target = type(actor)(
            state_dim=diffusion_artifact.state_dim,
            action_dim=state_tracker.num_item,
            hidden_sizes=self._get_hidden_sizes(actor),
        ).to(next(actor.parameters()).device)
        self.actor_target.load_state_dict(actor.state_dict())

        self.critic = critic
        self.critic_target = type(critic)(
            state_dim=diffusion_artifact.state_dim,
            action_dim=action_dim,
            hidden_sizes=self._get_hidden_sizes(critic.q1),
        ).to(next(critic.parameters()).device)
        self.critic_target.load_state_dict(critic.state_dict())

        self.optim_RL, self.optim_state = optim
        self.state_tracker = state_tracker
        self.reward_model = reward_model

        self.diffusion_model = diffusion_artifact.diffusion_model
        self.behavior_model = diffusion_artifact.behavior_model
        self.state_distribution = diffusion_artifact.state_distribution
        self.state_threshold = diffusion_artifact.state_threshold
        self.action_threshold = diffusion_artifact.action_threshold
        self.state_dim = diffusion_artifact.state_dim
        self.action_dim = diffusion_artifact.action_dim

        self.device = next(actor.parameters()).device
        self._gamma = discount_factor
        self.tau = tau
        self.beta = doser_beta
        self.lam = doser_lam
        self.eta = doser_eta
        self.expectile = doser_expectile
        self.q_min = doser_q_min
        self.action_samples = doser_action_samples
        self.policy_freq = doser_policy_freq
        self.target_update_freq = doser_target_update_freq
        self.diffusion_sample_steps = diffusion_sample_steps
        self.exploration_eps = exploration_eps
        self.log_interval = log_interval
        self.catalog_chunk_size = catalog_chunk_size

        self.num_items = state_tracker.num_item
        self.target_entropy = float(np.log(max(self.num_items, 2)) * 0.98)
        self.log_alpha = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = self.log_alpha.exp().detach()

        self.total_it = 0
        self.last_actor_loss = 0.0
        self.last_entropy = 0.0
        self.ood_helper = DORLDOSEROODHelper(self)
        self._validate_dimensions()

    @staticmethod
    def _get_hidden_sizes(model: Union[nn.Module, nn.Sequential]) -> List[int]:
        """从已有网络中推断隐藏层配置。"""

        hidden_sizes: List[int] = []
        if isinstance(model, nn.Sequential):
            for layer in model:
                if isinstance(layer, nn.Linear):
                    hidden_sizes.append(layer.out_features)
            return hidden_sizes[:-1]
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                hidden_sizes.append(layer.out_features)
        return hidden_sizes[:-1]

    def _validate_dimensions(self) -> None:
        """校验训练状态空间与扩散产物元信息是否一致。"""

        if int(self.state_tracker.final_dim) != self.state_dim:
            raise ValueError(
                "State tracker final_dim does not match diffusion state_dim: "
                f"{self.state_tracker.final_dim} != {self.state_dim}"
            )
        if int(self.state_tracker.emb_dim) != self.action_dim:
            raise ValueError(
                "State tracker emb_dim does not match diffusion action_dim: "
                f"{self.state_tracker.emb_dim} != {self.action_dim}"
            )

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """离策略更新前不额外改写 batch。"""

        return batch

    def forward(
        self,
        batch: Batch,
        buffer: Optional[ReplayBuffer],
        indices: np.ndarray = None,
        is_obs: bool = None,
        is_train: bool = True,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        use_batch_in_statetracker: bool = False,
        **kwargs: Any,
    ) -> Batch:
        """根据当前状态输出离散推荐动作。

        Args:
            batch (Batch): tianshou batch。
            buffer (Optional[ReplayBuffer]): 回放缓存。
            indices (np.ndarray): 对应的 buffer 索引。
            is_obs (bool): 当前是否编码 `obs`。
            is_train (bool): 是否处于训练模式。
            state (Optional[Union[dict, Batch, np.ndarray]]): 兼容接口保留。
            use_batch_in_statetracker (bool): 是否把当前 batch 拼进 state tracker。

        Returns:
            Batch: 包含 `act`、`logits`、`dist` 的输出。
        """

        obs_emb = self.state_tracker(
            buffer=buffer,
            indices=indices,
            is_obs=is_obs,
            batch=batch,
            is_train=is_train,
            use_batch_in_statetracker=use_batch_in_statetracker,
        )
        raw_logits = self.actor(obs_emb)
        action_mask = batch.mask if is_obs else getattr(batch, "next_mask", None)
        masked_logits, safe_mask = self._mask_logits(raw_logits, action_mask)
        dist = Categorical(logits=masked_logits)

        if self.training and is_train:
            action = dist.sample()
        else:
            action = masked_logits.argmax(dim=-1)

        return Batch(
            logits=masked_logits,
            act=action,
            state=state,
            dist=dist,
            action_mask=safe_mask,
        )

    def exploration_noise(
        self,
        act: Union[np.ndarray, Batch],
        batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        """对离散动作执行 epsilon-greedy 探索。"""

        if not isinstance(act, np.ndarray) or np.isclose(self.exploration_eps, 0.0):
            return act

        batch_size = len(act)
        random_mask = np.random.rand(batch_size) < self.exploration_eps
        if not np.any(random_mask):
            return act

        if hasattr(batch, "mask"):
            valid_mask = batch.mask.detach().cpu().numpy().astype(bool)
        else:
            valid_mask = np.ones((batch_size, self.num_items), dtype=bool)

        for index in np.where(random_mask)[0]:
            valid_candidates = np.where(valid_mask[index])[0]
            if len(valid_candidates) == 0:
                continue
            act[index] = np.random.choice(valid_candidates)
        return act

    def learn(
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, float]:
        """执行 DORL-DOSER 的离策略更新。"""

        env_step = kwargs.get("env_step")
        aggregated_metrics: Dict[str, List[float]] = {}
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                self.total_it += 1
                alpha_metric = self._update_alpha(minibatch)
                if self.total_it % self.policy_freq == 0:
                    actor_metric = self._update_actor(minibatch)
                else:
                    actor_metric = {
                        "loss/actor": self.last_actor_loss,
                        "policy/entropy": self.last_entropy,
                    }
                critic_metric = self._update_critic(minibatch)

                if self.total_it % self.target_update_freq == 0:
                    self._sync_weight()

                step_metrics = {**alpha_metric, **actor_metric, **critic_metric}
                for metric_name, metric_value in step_metrics.items():
                    aggregated_metrics.setdefault(metric_name, []).append(metric_value)

                if self.total_it % self.log_interval == 0 and env_step is not None:
                    _safe_wandb_log(step_metrics, step=int(env_step))

        return {
            metric_name: float(np.mean(metric_values))
            for metric_name, metric_values in aggregated_metrics.items()
        }

    def _update_alpha(self, minibatch: Batch) -> Dict[str, float]:
        """更新温度系数 alpha。"""

        state_emb = self.state_tracker(self._buffer, minibatch.indices, is_obs=True)
        action_mask = getattr(minibatch, "mask", None)
        _, probs, log_probs, _ = self._policy_statistics(
            state_emb=state_emb,
            action_mask=action_mask,
            model_name="actor",
        )
        entropy = -(probs * log_probs).sum(dim=-1)
        alpha_loss = -(
            self.log_alpha * (entropy.detach() - self.target_entropy)
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()
        return {
            "loss/alpha": float(alpha_loss.item()),
            "alpha/value": float(self.alpha.item()),
        }

    def _update_actor(self, minibatch: Batch) -> Dict[str, float]:
        """更新离散 actor。"""

        self.optim_RL.zero_grad()
        self.optim_state.zero_grad()

        state_emb = self.state_tracker(self._buffer, minibatch.indices, is_obs=True)
        action_mask = getattr(minibatch, "mask", None)
        _, probs, log_probs, _ = self._policy_statistics(
            state_emb=state_emb,
            action_mask=action_mask,
            model_name="actor",
        )
        with torch.no_grad():
            catalog_q = self._evaluate_catalog_q(
                state_emb=state_emb,
                item_embeddings=self._get_all_item_embeddings(),
                critic=self.critic,
            )
        actor_loss = (
            probs * (self.alpha.detach() * log_probs - catalog_q)
        ).sum(dim=-1).mean()
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        actor_loss.backward()
        self.optim_RL.step()
        self.optim_state.step()

        self.last_actor_loss = float(actor_loss.item())
        self.last_entropy = float(entropy.item())
        return {
            "loss/actor": self.last_actor_loss,
            "policy/entropy": self.last_entropy,
        }

    def _update_critic(self, minibatch: Batch) -> Dict[str, float]:
        """更新 critic 与 OOD 识别相关项。"""

        self.optim_RL.zero_grad()
        self.optim_state.zero_grad()

        state_emb = self.state_tracker(self._buffer, minibatch.indices, is_obs=True)
        next_state_emb = self.state_tracker(self._buffer, minibatch.indices, is_obs=False)
        reward = torch.as_tensor(
            minibatch.rew,
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        done = torch.as_tensor(
            minibatch.done,
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        action_ids = self._to_action_id_tensor(minibatch.act)
        action_embeddings = self.ood_helper.lookup_action_embeddings(action_ids)

        critic_loss, metrics = self._compute_critic_loss(
            minibatch=minibatch,
            state_emb=state_emb,
            next_state_emb=next_state_emb,
            action_embeddings=action_embeddings,
            reward=reward,
            done=done,
        )
        critic_loss.backward()
        self.optim_RL.step()
        self.optim_state.step()
        metrics["loss/critic_total"] = float(critic_loss.item())
        return metrics

    def _compute_critic_loss(
        self,
        minibatch: Batch,
        state_emb: torch.Tensor,
        next_state_emb: torch.Tensor,
        action_embeddings: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算 DOSER 风格 critic 主损失与 OOD 正则。"""

        not_done = 1.0 - done
        current_q1, current_q2, current_q3, current_q4 = self.critic(
            state_emb, action_embeddings
        )
        current_q = torch.cat(
            [current_q1, current_q2, current_q3, current_q4],
            dim=-1,
        )
        current_v1, current_v2 = self.critic.v(state_emb)

        with torch.no_grad():
            _, next_probs, next_log_probs, _ = self._policy_statistics(
                state_emb=next_state_emb,
                action_mask=getattr(minibatch, "next_mask", None),
                model_name="actor_target",
            )
            target_catalog_q = self._evaluate_catalog_q(
                state_emb=next_state_emb,
                item_embeddings=self._get_all_item_embeddings(),
                critic=self.critic_target,
            )
            soft_next_q = (
                next_probs
                * (target_catalog_q - self.alpha.detach() * next_log_probs)
            ).sum(dim=-1, keepdim=True)
            target_q = reward + not_done * self._gamma * soft_next_q
            target_v = reward + not_done * self._gamma * self.critic_target.v_min(
                next_state_emb
            )

        value_loss = self._expectile_loss(target_v - current_v1).mean() + self._expectile_loss(
            target_v - current_v2
        ).mean()
        bellman_loss = (
            F.mse_loss(current_q1, target_q)
            + F.mse_loss(current_q2, target_q)
            + F.mse_loss(current_q3, target_q)
            + F.mse_loss(current_q4, target_q)
            + value_loss
        )

        with torch.no_grad():
            policy_action_ids = self._get_greedy_action_ids(
                state_emb=state_emb,
                action_mask=getattr(minibatch, "mask", None),
            )
            policy_action_emb = self.ood_helper.lookup_action_embeddings(policy_action_ids)
            action_error = self.ood_helper.compute_action_error(policy_action_emb, state_emb)

            normalized_obs = self.ood_helper.normalize_obs_array(minibatch.obs)
            user_ids = normalized_obs[:, 0].astype(np.int64)
            pred_next_state, pred_rewards = self.ood_helper.build_counterfactual_next_state(
                indices=minibatch.indices,
                user_ids=user_ids,
                action_ids=policy_action_ids,
            )
            best_id_action_ids, best_id_action_emb, best_id_q = self.ood_helper.select_best_id_action(
                state_emb=state_emb,
                action_mask=getattr(minibatch, "mask", None),
            )
            best_id_next_state, best_id_rewards = self.ood_helper.build_counterfactual_next_state(
                indices=minibatch.indices,
                user_ids=user_ids,
                action_ids=best_id_action_ids,
            )
            pred_state_error = self.ood_helper.compute_state_error(pred_next_state)
            value_s_pi = self.critic.v_min(pred_next_state)
            value_s_in = self.critic.v_min(best_id_next_state)

        ood_action_mask = action_error > self.action_threshold
        ood_next_state_mask = pred_state_error > self.state_threshold
        negative_value_mask = (value_s_pi < value_s_in).squeeze(-1)
        positive_value_mask = ~negative_value_mask

        negative_ood_mask = (
            ood_action_mask & (ood_next_state_mask | negative_value_mask)
        ).float().unsqueeze(-1)
        positive_ood_mask = (
            ood_action_mask & (~ood_next_state_mask) & positive_value_mask
        ).float().unsqueeze(-1)

        pi_q1, pi_q2, pi_q3, pi_q4 = self.critic(state_emb, policy_action_emb)
        policy_q = torch.cat([pi_q1, pi_q2, pi_q3, pi_q4], dim=-1)
        q_min_tensor = torch.full_like(policy_q, fill_value=self.q_min)
        reg_loss = self.beta * (((policy_q - q_min_tensor) ** 2) * negative_ood_mask).mean()

        value_diff = (value_s_pi - value_s_in).clamp(min=0.0)
        q_comp_target = self.eta * (best_id_q + value_diff).detach()
        vc_loss = self.lam * (((policy_q - q_comp_target) ** 2) * positive_ood_mask).mean()

        total_loss = bellman_loss + reg_loss + vc_loss
        metrics = self._build_ood_metrics(
            current_q=current_q,
            policy_q=policy_q,
            value_s_pi=value_s_pi,
            value_s_in=value_s_in,
            q_comp_target=q_comp_target,
            action_error=action_error,
            pred_state_error=pred_state_error,
            negative_ood_mask=negative_ood_mask,
            positive_ood_mask=positive_ood_mask,
            ood_action_mask=ood_action_mask,
            bellman_loss=bellman_loss,
            reg_loss=reg_loss,
            vc_loss=vc_loss,
            pred_rewards=pred_rewards,
            best_id_rewards=best_id_rewards,
        )
        return total_loss, metrics

    def _build_ood_metrics(
        self,
        current_q: torch.Tensor,
        policy_q: torch.Tensor,
        value_s_pi: torch.Tensor,
        value_s_in: torch.Tensor,
        q_comp_target: torch.Tensor,
        action_error: torch.Tensor,
        pred_state_error: torch.Tensor,
        negative_ood_mask: torch.Tensor,
        positive_ood_mask: torch.Tensor,
        ood_action_mask: torch.Tensor,
        bellman_loss: torch.Tensor,
        reg_loss: torch.Tensor,
        vc_loss: torch.Tensor,
        pred_rewards: np.ndarray,
        best_id_rewards: np.ndarray,
    ) -> Dict[str, float]:
        """整理 critic 更新阶段的监控指标。"""

        total_count = float(len(action_error))
        ood_count = float(ood_action_mask.float().sum().item())
        positive_count = float(positive_ood_mask.float().sum().item())
        negative_count = float(negative_ood_mask.float().sum().item())
        id_count = total_count - ood_count
        safe_total = max(total_count, 1.0)

        return {
            "loss/bellman": float(bellman_loss.item()),
            "loss/reg": float(reg_loss.item()),
            "loss/vc": float(vc_loss.item()),
            "ood/id_ratio": id_count / safe_total,
            "ood/ood_ratio": ood_count / safe_total,
            "ood/positive_ratio": positive_count / safe_total,
            "ood/negative_ratio": negative_count / safe_total,
            "ood/action_error_mean": float(action_error.mean().item()),
            "ood/state_error_mean": float(pred_state_error.mean().item()),
            "q/current_mean": float(current_q.mean().item()),
            "q/policy_mean": float(policy_q.mean().item()),
            "value/pi_mean": float(value_s_pi.mean().item()),
            "value/id_mean": float(value_s_in.mean().item()),
            "q/comp_target_mean": float(q_comp_target.mean().item()),
            "reward/policy_counterfactual_mean": float(np.mean(pred_rewards)),
            "reward/id_counterfactual_mean": float(np.mean(best_id_rewards)),
            "threshold/state": float(self.state_threshold),
            "threshold/action": float(self.action_threshold),
        }

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """计算 expectile loss。"""

        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return weight * diff.pow(2)

    def _policy_statistics(
        self,
        state_emb: torch.Tensor,
        action_mask: Optional[torch.Tensor],
        model_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Categorical]:
        """根据指定 actor 计算离散策略统计量。"""

        actor_model = getattr(self, model_name)
        raw_logits = actor_model(state_emb)
        masked_logits, _ = self._mask_logits(raw_logits, action_mask)
        probs = torch.softmax(masked_logits, dim=-1)
        log_probs = torch.log(probs.clamp_min(DEFAULT_EPS))
        dist = Categorical(probs=probs)
        return masked_logits, probs, log_probs, dist

    def _mask_logits(
        self,
        logits: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对 logits 应用推荐动作 mask。"""

        if action_mask is None:
            full_mask = torch.ones_like(logits, dtype=torch.bool)
            return logits, full_mask

        safe_mask = action_mask.to(logits.device).bool()
        empty_rows = ~safe_mask.any(dim=-1)
        if empty_rows.any():
            safe_mask[empty_rows] = True
        masked_logits = logits.masked_fill(~safe_mask, MASKED_LOGIT_VALUE)
        return masked_logits, safe_mask

    def _get_all_item_embeddings(self) -> torch.Tensor:
        """读取当前 state tracker 中的全量物品 embedding。"""

        item_ids = np.arange(self.num_items, dtype=np.int64).reshape(-1, 1)
        return self.state_tracker.get_embedding(item_ids, "action")

    def _evaluate_catalog_q(
        self,
        state_emb: torch.Tensor,
        item_embeddings: torch.Tensor,
        critic: DORLCriticNetwork,
    ) -> torch.Tensor:
        """按 chunk 评估全 catalog 的 Q 值。"""

        batch_size = state_emb.shape[0]
        num_items = item_embeddings.shape[0]
        chunk_size = self.catalog_chunk_size or num_items
        catalog_q_list: List[torch.Tensor] = []
        for start in range(0, num_items, chunk_size):
            end = min(start + chunk_size, num_items)
            action_chunk = item_embeddings[start:end]
            expanded_states = state_emb.unsqueeze(1).expand(-1, end - start, -1)
            expanded_actions = action_chunk.unsqueeze(0).expand(batch_size, -1, -1)
            flat_states = expanded_states.reshape(-1, state_emb.shape[-1])
            flat_actions = expanded_actions.reshape(-1, action_chunk.shape[-1])
            chunk_q = critic.q_min(flat_states, flat_actions).view(batch_size, -1)
            catalog_q_list.append(chunk_q)
        return torch.cat(catalog_q_list, dim=-1)

    def _to_action_id_tensor(
        self, action_ids: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """把 buffer 中的动作转换为 LongTensor。"""

        if isinstance(action_ids, torch.Tensor):
            return action_ids.to(self.device).long().view(-1)
        return torch.as_tensor(
            np.asarray(action_ids),
            dtype=torch.long,
            device=self.device,
        ).view(-1)

    def _lookup_action_embeddings(self, action_ids: torch.Tensor) -> torch.Tensor:
        """根据离散 item id 查询动作 embedding。"""

        action_array = action_ids.detach().cpu().numpy().reshape(-1, 1)
        return self.state_tracker.get_embedding(action_array, "action")

    def _get_greedy_action_ids(
        self,
        state_emb: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """获取当前 actor 的贪心动作。"""

        masked_logits, _, _, _ = self._policy_statistics(
            state_emb=state_emb,
            action_mask=action_mask,
            model_name="actor",
        )
        return masked_logits.argmax(dim=-1)

    def _extract_action_histories(
        self, indices: np.ndarray
    ) -> List[List[int]]:
        """根据 replay buffer 还原每个样本的历史动作序列。"""

        if len(indices) == 0:
            return []

        histories: List[List[int]] = [[] for _ in range(len(indices))]
        cursor = np.array(indices, copy=True)
        live_mask = np.ones(len(indices), dtype=bool)
        while np.any(live_mask):
            current_obs = self._normalize_obs_array(self._buffer.obs[cursor])
            current_actions = current_obs[:, 1]
            for row_index, is_live in enumerate(live_mask):
                if not is_live:
                    continue
                action_id = int(current_actions[row_index])
                if 0 <= action_id < self.num_items:
                    histories[row_index].append(action_id)
            episode_start_mask = np.asarray(self._buffer.is_start[cursor]).reshape(-1)
            live_mask[episode_start_mask] = False
            cursor = self._buffer.prev(cursor)

        for history in histories:
            history.reverse()
        return histories

    def _normalize_obs_array(self, obs: Any) -> np.ndarray:
        """把 replay buffer 中不同形状的观测统一整理为 `(batch, feature_dim)`。"""

        obs_array = np.asarray(obs)
        if obs_array.ndim == 1:
            return obs_array.reshape(1, -1)
        if obs_array.ndim == 2:
            return obs_array
        return obs_array.reshape(-1, obs_array.shape[-1])

    def _build_counterfactual_next_state(
        self,
        indices: np.ndarray,
        user_ids: np.ndarray,
        action_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """通过 synthetic batch 构造 counterfactual next-state。"""

        action_array = action_ids.detach().cpu().numpy().astype(np.int64)
        histories = self._extract_action_histories(indices)
        rewards = self.reward_model.estimate_rewards(user_ids, histories, action_array)
        synthetic_obs = np.stack([user_ids, action_array], axis=1).astype(np.int64)
        synthetic_batch = Batch(obs=synthetic_obs, rew_prev=rewards)
        next_state = self.state_tracker(
            buffer=self._buffer,
            indices=indices,
            is_obs=True,
            batch=synthetic_batch,
            is_train=True,
            use_batch_in_statetracker=True,
        )
        return next_state, rewards

    def _map_action_embeddings_to_item_ids(
        self, action_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """将连续动作 embedding 映射回最近的离散 item id。"""

        with torch.no_grad():
            item_embeddings = self._get_all_item_embeddings()
            normalized_item_embeddings = F.normalize(item_embeddings, dim=-1)
            normalized_action_embeddings = F.normalize(action_embeddings, dim=-1)
            similarity = torch.matmul(
                normalized_action_embeddings, normalized_item_embeddings.transpose(0, 1)
            )
            return similarity.argmax(dim=-1)

    def _select_best_id_action(
        self,
        state_emb: torch.Tensor,
        action_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从行为扩散候选中筛选 critic 认为最优的 ID 动作。"""

        sampled_action_embeddings = self.diffusion_model.sample(
            model=self.behavior_model,
            cond=state_emb.detach(),
            action_samples=self.action_samples,
            n_steps=self.diffusion_sample_steps,
        )
        batch_size = sampled_action_embeddings.shape[0]
        flat_embeddings = sampled_action_embeddings.view(-1, self.action_dim)
        candidate_action_ids = self._map_action_embeddings_to_item_ids(flat_embeddings).view(
            batch_size, self.action_samples
        )
        candidate_action_embeddings = self._lookup_action_embeddings(
            candidate_action_ids.reshape(-1)
        ).view(batch_size, self.action_samples, self.action_dim)

        expanded_state = state_emb.unsqueeze(1).expand(-1, self.action_samples, -1)
        flat_state = expanded_state.reshape(-1, self.state_dim)
        flat_action = candidate_action_embeddings.reshape(-1, self.action_dim)
        candidate_q = self.critic.q_min(flat_state, flat_action).view(
            batch_size, self.action_samples
        )
        masked_candidate_q = candidate_q.clone()

        if action_mask is not None:
            safe_mask = action_mask.to(self.device).bool()
            candidate_mask = torch.gather(
                safe_mask,
                dim=1,
                index=candidate_action_ids,
            )
            masked_candidate_q = masked_candidate_q.masked_fill(
                ~candidate_mask,
                float("-inf"),
            )
            invalid_rows = ~candidate_mask.any(dim=-1)
            if invalid_rows.any():
                masked_candidate_q[invalid_rows] = candidate_q[invalid_rows]

        best_indices = masked_candidate_q.argmax(dim=-1)
        batch_indices = torch.arange(batch_size, device=self.device)
        best_action_ids = candidate_action_ids[batch_indices, best_indices]
        best_action_embeddings = candidate_action_embeddings[batch_indices, best_indices]
        best_q = candidate_q[batch_indices, best_indices].unsqueeze(-1)
        return best_action_ids, best_action_embeddings, best_q

    def compute_state_error(
        self,
        states: torch.Tensor,
        batch_size: int = DEFAULT_RECON_BATCH_SIZE,
    ) -> torch.Tensor:
        """基于状态扩散模型计算 state reconstruction error。"""

        reconstruction_errors: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(states), batch_size):
                batch_states = states[start : start + batch_size]
                sample_density = self.diffusion_model.make_sample_density()
                sigma = sample_density(
                    shape=(len(batch_states),),
                    device=states.device,
                )
                noise = torch.randn_like(batch_states)
                sigma_expanded = sigma.view(-1, 1)
                noisy_states = batch_states + noise * sigma_expanded
                c_skip, c_out, c_in = [
                    value.view(-1, 1)
                    for value in self.diffusion_model.get_diffusion_scalings(sigma)
                ]
                model_input = noisy_states * c_in
                model_output = self.state_distribution(
                    model_input,
                    None,
                    torch.log(sigma) / 4,
                )
                denoised_states = c_skip * noisy_states + c_out * model_output
                reconstruction_errors.append(
                    torch.norm(denoised_states - batch_states, dim=-1)
                )
        return torch.cat(reconstruction_errors, dim=0)

    def compute_action_error(
        self,
        actions: torch.Tensor,
        states: torch.Tensor,
        batch_size: int = DEFAULT_RECON_BATCH_SIZE,
    ) -> torch.Tensor:
        """基于行为扩散模型计算 action reconstruction error。"""

        reconstruction_errors: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(actions), batch_size):
                batch_actions = actions[start : start + batch_size]
                batch_states = states[start : start + batch_size]
                sample_density = self.diffusion_model.make_sample_density()
                sigma = sample_density(
                    shape=(len(batch_actions),),
                    device=actions.device,
                )
                noise = torch.randn_like(batch_actions)
                sigma_expanded = sigma.view(-1, 1)
                noisy_actions = batch_actions + noise * sigma_expanded
                c_skip, c_out, c_in = [
                    value.view(-1, 1)
                    for value in self.diffusion_model.get_diffusion_scalings(sigma)
                ]
                model_input = noisy_actions * c_in
                model_output = self.behavior_model(
                    model_input,
                    batch_states,
                    torch.log(sigma) / 4,
                )
                denoised_actions = c_skip * noisy_actions + c_out * model_output
                reconstruction_errors.append(
                    torch.norm(denoised_actions - batch_actions, dim=-1)
                )
        return torch.cat(reconstruction_errors, dim=0)

    def _sync_weight(self) -> None:
        """软更新 actor_target 与 critic_target。"""

        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
