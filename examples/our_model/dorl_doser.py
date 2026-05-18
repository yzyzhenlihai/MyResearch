"""KuaiEnv-v0 上的推荐系统版 DORL-DOSER 训练入口。"""

import argparse
import os
import pickle
import random
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from logzero import logger
from tqdm import tqdm

sys.path.extend([".", "./examples/policy", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import (  # noqa: E402
    get_args_all,
    learn_policy,
    prepare_dir_log,
    prepare_test_envs,
    prepare_user_model,
    setup_state_tracker,
)

from src.core.collector.collector import Collector  # noqa: E402
from src.core.collector.collector_set import CollectorSet  # noqa: E402
from src.core.envs.Simulated_Env.penalty_ent_exp import (  # noqa: E402
    PenaltyEntExpSimulatedEnv,
    get_features_of_last_n_items_features,
)
from src.core.policy.RecPolicy import RecPolicy  # noqa: E402
from src.core.policy.doser import (  # noqa: E402
    CounterfactualRewardModel,
    DORLCriticNetwork,
    DORLDOSERPolicy,
    DiscreteActorNetwork,
    RewardModelConfig,
    load_diffusion_artifact,
)
from src.core.util.data import get_env_args, get_true_env  # noqa: E402
from src.core.util.wandb_utils import load_wandb  # noqa: E402
from src.tianshou.tianshou.data import VectorReplayBuffer  # noqa: E402
from src.tianshou.tianshou.env import DummyVectorEnv  # noqa: E402

wandb = load_wandb(repo_root=Path(__file__).resolve().parents[2])


DEFAULT_DOSER_SOFT_TAU = 0.005
"""DORL-DOSER 目标网络软更新系数。"""

DEFAULT_DIFFUSION_SAMPLE_STEPS = 20
"""行为扩散采样默认步数。"""

DEFAULT_DOSER_LOG_INTERVAL = 100
"""训练指标默认汇报间隔。"""

DEFAULT_WANDB_PROJECT = "DORL-DOSER"
"""DORL-DOSER 训练默认使用的 wandb project 名称。"""


WANDB_DISABLED_VALUES = {"1", "true", "yes", "on"}
"""将环境变量解析为布尔开关时认定为真值的集合。"""


def _get_wandb_mode() -> str:
    """读取 wandb 运行模式。

    优先使用标准的 `WANDB_MODE`，同时兼容历史 `SWANLAB_MODE` 配置，
    方便已有训练脚本平滑迁移。

    Returns:
        str: 当前 wandb 模式，小写字符串。
    """

    return os.environ.get("WANDB_MODE", os.environ.get("SWANLAB_MODE", "")).strip().lower()


def _is_wandb_disabled() -> bool:
    """判断当前进程是否显式关闭 wandb 日志。"""

    disabled_flag = os.environ.get("WANDB_DISABLED", "").strip().lower()
    return disabled_flag in WANDB_DISABLED_VALUES or _get_wandb_mode() == "disabled"


def _to_wandb_serializable(value: Any) -> Any:
    """把常见训练配置转换为 wandb 可接受的 JSON 风格数据。"""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_wandb_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wandb_serializable(item) for item in value]
    return str(value)


def _build_default_wandb_run_name(args: argparse.Namespace) -> str:
    """为当前训练构造默认的 wandb run 名称。"""

    return (
        f"{args.message}-{args.env}-seed:{args.seed}"
        f"-window:{args.window_size}"
        f"-beta:{args.doser_beta}"
        f"-lam:{args.doser_lam}"
        f"-eta:{args.doser_eta}"
    )


def set_wandb(
    args: argparse.Namespace,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """按需初始化 wandb 实验记录。

    Args:
        args (argparse.Namespace): 当前训练配置。
        run_metadata (Optional[Dict[str, Any]]): 训练运行阶段补充写入的元信息。
    """

    if wandb is None:
        logger.info("Skip wandb init because wandb is unavailable.")
        return

    if _is_wandb_disabled():
        logger.info(
            "Skip wandb init because wandb logging is disabled: WANDB_MODE=%s, WANDB_DISABLED=%s",
            _get_wandb_mode(),
            os.environ.get("WANDB_DISABLED", ""),
        )
        return

    config = {key: _to_wandb_serializable(value) for key, value in vars(args).items()}
    if run_metadata:
        config.update(
            {
                key: _to_wandb_serializable(value)
                for key, value in run_metadata.items()
            }
        )

    init_kwargs = dict(
        project=args.wandb_project,
        config=config,
        name=args.wandb_run_name or _build_default_wandb_run_name(args),
    )
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_group:
        init_kwargs["group"] = args.wandb_group
    if args.wandb_job_type:
        init_kwargs["job_type"] = args.wandb_job_type
    if args.wandb_tags:
        init_kwargs["tags"] = list(args.wandb_tags)
    if args.wandb_dir:
        init_kwargs["dir"] = args.wandb_dir

    if args.wandb_mode:
        init_kwargs["mode"] = args.wandb_mode
    wandb_mode = _get_wandb_mode()
    if wandb_mode and "mode" not in init_kwargs:
        init_kwargs["mode"] = wandb_mode
    wandb.init(**init_kwargs)


def finish_wandb() -> None:
    """安全结束 wandb 记录。"""

    if wandb is None:
        return
    if getattr(wandb, "run", None) is None:
        return
    wandb.finish()


def get_args_dorl_doser() -> argparse.Namespace:
    """解析 DORL-DOSER 额外超参数。

    Returns:
        argparse.Namespace: 仅包含本脚本特有配置的命名空间。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DORL_DOSER")
    parser.add_argument("--message", type=str, default="DORL_DOSER")

    parser.add_argument("--diffusion_save_root", type=str, default="saved_models")
    parser.add_argument("--diffusion_artifact_name", type=str, default=None)
    parser.add_argument(
        "--diffusion_sample_steps",
        type=int,
        default=DEFAULT_DIFFUSION_SAMPLE_STEPS,
    )

    parser.add_argument("--doser_beta", type=float, default=0.001) # 控制 negative OOD penalty 的强度。
    parser.add_argument("--doser_lam", type=float, default=0.001) # 控制 positive OOD compensation 的强度。
    parser.add_argument("--doser_eta", type=float, default=0.9) # 控制 positive OOD compensation 里目标值的缩放比例
    parser.add_argument("--doser_expectile", type=float, default=0.9) # 控制 value 分支里的 expectile loss
    parser.add_argument("--doser_action_samples", type=int, default=10) # 行为扩散模型每次为一个状态采样多少个候选动作 embedding，找最优ID动作用的
    parser.add_argument("--doser_policy_freq", type=int, default=2) # actor 的更新频率。
    parser.add_argument("--doser_target_update_freq", type=int, default=2) # target network 的更新频率
    parser.add_argument("--doser_q_min", type=float, default=0.0) # negative OOD penalty 里的 Q 下界先验
    parser.add_argument("--doser_log_interval", type=int, default=DEFAULT_DOSER_LOG_INTERVAL) # 日志打印/记录间隔。
    parser.add_argument("--doser_catalog_chunk_size", type=int, default=512) # 全物品集合计算 Q 时的分块大小

    parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_job_type", type=str, default="train")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_dir", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
 
    parser.add_argument("--is_exposure_intervention", dest="use_exposure_intervention", action="store_true")
    parser.add_argument("--no_exposure_intervention", dest="use_exposure_intervention", action="store_false")
    parser.set_defaults(use_exposure_intervention=False)

    parser.add_argument("--is_feature_level", dest="feature_level", action="store_true")
    parser.add_argument("--no_feature_level", dest="feature_level", action="store_false")
    parser.set_defaults(feature_level=True)

    parser.add_argument("--is_sorted", dest="is_sorted", action="store_true")
    parser.add_argument("--no_sorted", dest="is_sorted", action="store_false")
    parser.set_defaults(is_sorted=True)

    parser.add_argument("--entropy_window", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--gamma_exposure", type=float, default=10.0)
    parser.add_argument("--lambda_entropy", type=float, default=5.0)
    parser.add_argument("--read_message", type=str, default="UM")

    parser.set_defaults(exploration_noise=False)
    return parser.parse_known_args()[0]


def get_entropy(item_list: Sequence[int], need_count: bool = True) -> float:
    """计算物品序列熵。

    Args:
        item_list (Sequence[int]): 原始列表或计数字典。
        need_count (bool): 是否需要先统计频次。

    Returns:
        float: 归一化熵值。
    """

    if len(item_list) <= 1:
        return 1.0
    count_dict = Counter(item_list) if need_count else item_list
    probability = np.array(list(count_dict.values()), dtype=np.float64)
    probability = probability / np.sum(probability)
    return float(-np.sum(np.log2(probability) * probability) / np.log2(len(count_dict)))


def get_save_entropy_mat(
    dataset: Any,
    entropy_window: Sequence[int],
    feature_level: bool = True,
    is_sorted: bool = True,
) -> Tuple[Dict[Tuple[int, ...], float], Optional[Dict[int, Sequence[int]]]]:
    """根据训练数据统计 entropy map。

    Args:
        dataset (Any): 推荐数据集对象。
        entropy_window (Sequence[int]): 熵窗口大小集合。
        feature_level (bool): 是否在特征级统计熵。
        is_sorted (bool): 是否在窗口内排序。

    Returns:
        Tuple[Dict[Tuple[int, ...], float], Optional[Dict[int, Sequence[int]]]]:
        entropy 映射以及 item->feature 映射。
    """

    df_train, _, df_item, _ = dataset.get_train_data()
    map_item_feat = dict(zip(df_item.index, df_item["tags"])) if feature_level else None
    if "timestamp" not in df_train.columns:
        df_train = df_train.rename(columns={"time_ms": "timestamp"})

    df_uit = df_train[["user_id", "item_id", "timestamp"]].sort_values(
        ["user_id", "timestamp"]
    )
    map_hist_count = defaultdict(lambda: defaultdict(int))
    last_user_id = -1
    history_actions = []

    def update_map(
        target_map: Dict[Tuple[int, ...], Dict[int, int]],
        history: Sequence[int],
        item_id: int,
        require_len: int,
    ) -> None:
        """更新给定窗口长度下的频次表。"""

        if len(history) < require_len:
            return
        history_key = tuple(sorted(history[-require_len:]) if is_sorted else history[-require_len:])
        target_map[history_key][item_id] += 1

    for user_id, item_id, _ in tqdm(
        df_uit.to_numpy(),
        total=len(df_uit),
        desc="build entropy statistics",
    ):
        user_id = int(user_id)
        item_id = int(item_id)
        if user_id != last_user_id:
            history_actions = []
            last_user_id = user_id

        if feature_level:
            features = map_item_feat[item_id]
            for feature in features:
                for require_len in set(entropy_window) - {0}:
                    history_features = get_features_of_last_n_items_features(
                        require_len,
                        history_actions,
                        map_item_feat,
                        is_sort=is_sorted,
                    )
                    for feature_history in history_features:
                        update_map(map_hist_count, feature_history, feature, require_len)
        else:
            for require_len in set(entropy_window) - {0}:
                update_map(map_hist_count, history_actions, item_id, require_len)
        history_actions.append(item_id)

    map_entropy = {
        history_key: get_entropy(next_items, need_count=False)
        for history_key, next_items in tqdm(
            map_hist_count.items(),
            total=len(map_hist_count),
            desc="compute entropy map",
        )
    }
    return map_entropy, map_item_feat


def prepare_train_envs_and_reward_model(
    args: argparse.Namespace,
    ensemble_models: Any,
    env: Any,
    dataset: Any,
    kwargs_um: Dict[str, Any],
) -> Tuple[DummyVectorEnv, RewardModelConfig]:
    """构造训练环境与 counterfactual reward 配置。

    Args:
        args (argparse.Namespace): 训练配置。
        ensemble_models (Any): 用户模型集成。
        env (Any): 真实推荐环境。
        dataset (Any): 数据集对象。
        kwargs_um (Dict[str, Any]): 环境初始化参数。

    Returns:
        Tuple[DummyVectorEnv, RewardModelConfig]: 训练环境和奖励配置。
    """

    entropy_dict: Dict[str, Any] = {}
    map_item_feat = None
    if len(set(args.entropy_window) - {0}):
        map_entropy, map_item_feat = get_save_entropy_mat(
            dataset=dataset,
            entropy_window=args.entropy_window,
            feature_level=args.feature_level,
            is_sorted=args.is_sorted,
        )
        entropy_dict["map"] = map_entropy

    entropy_min = 0.0
    entropy_max = 0.0
    if entropy_dict.get("map"):
        for entropy_term in set(args.entropy_window):
            if entropy_term == 0:
                continue
            entropy_values = [
                value
                for history_key, value in entropy_dict["map"].items()
                if len(history_key) == entropy_term
            ]
            entropy_min += min(entropy_values + [1.0])
            entropy_max += max(entropy_values + [1.0])

    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    alpha_u, beta_i = None, None
    env_kwargs = {
        "ensemble_models": ensemble_models,
        "env_task_class": type(env),
        "task_env_param": kwargs_um,
        "task_name": args.env,
        "predicted_mat": predicted_mat,
        "version": args.version,
        "tau": args.tau,
        "use_exposure_intervention": args.use_exposure_intervention,
        "gamma_exposure": args.gamma_exposure,
        "alpha_u": alpha_u,
        "beta_i": beta_i,
        "entropy_dict": entropy_dict,
        "entropy_window": args.entropy_window,
        "lambda_entropy": args.lambda_entropy,
        "step_n_actions": max(args.entropy_window) if len(args.entropy_window) else 0,
        "entropy_min": entropy_min,
        "entropy_max": entropy_max,
        "feature_level": args.feature_level,
        "map_item_feat": map_item_feat,
        "is_sorted": args.is_sorted,
    }
    train_envs = DummyVectorEnv(
        [lambda: PenaltyEntExpSimulatedEnv(**env_kwargs) for _ in range(args.training_num)]
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)

    reward_model_config = RewardModelConfig(
        env_name=args.env,
        predicted_mat=predicted_mat,
        real_env=env,
        version=args.version,
        tau=args.tau,
        use_exposure_intervention=args.use_exposure_intervention,
        gamma_exposure=args.gamma_exposure,
        alpha_u=alpha_u,
        beta_i=beta_i,
        entropy_dict=entropy_dict,
        entropy_window=args.entropy_window,
        lambda_entropy=args.lambda_entropy,
        step_n_actions=max(args.entropy_window) if len(args.entropy_window) else 0,
        entropy_min=entropy_min,
        entropy_max=entropy_max,
        feature_level=args.feature_level,
        map_item_feat=map_item_feat,
        is_sorted=args.is_sorted,
    )
    return train_envs, reward_model_config


def setup_policy_model(
    args: argparse.Namespace,
    state_tracker: torch.nn.Module,
    train_envs: DummyVectorEnv,
    test_envs_dict: Dict[str, DummyVectorEnv],
    reward_model_config: RewardModelConfig,
    diffusion_artifact: Any,
) -> Tuple[RecPolicy, Collector, CollectorSet, Tuple[torch.optim.Optimizer, torch.optim.Optimizer]]:
    """初始化 DORL-DOSER policy、collector 与优化器。

    Args:
        args (argparse.Namespace): 训练配置。
        state_tracker (torch.nn.Module): 状态编码器。
        train_envs (DummyVectorEnv): 训练环境。
        test_envs_dict (Dict[str, DummyVectorEnv]): 测试环境字典。
        reward_model_config (RewardModelConfig): 奖励重建配置。
        diffusion_artifact (Any): 扩散模型产物。

    Returns:
        Tuple[RecPolicy, Collector, CollectorSet, Tuple[torch.optim.Optimizer, torch.optim.Optimizer]]:
        包装后的推荐策略、训练 collector、测试 collector 集以及优化器。
    """

    if args.cpu:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(
            f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        )

    actor = DiscreteActorNetwork(
        state_dim=args.state_dim,
        action_dim=args.action_shape,
        hidden_sizes=args.hidden_sizes,
    ).to(args.device)
    critic = DORLCriticNetwork(
        state_dim=args.state_dim,
        action_dim=state_tracker.emb_dim,
        hidden_sizes=args.hidden_sizes,
    ).to(args.device)
    optim_rl = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
    )
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)

    reward_model = CounterfactualRewardModel(reward_model_config)
    policy = DORLDOSERPolicy(
        actor=actor,
        critic=critic,
        optim=(optim_rl, optim_state),
        state_tracker=state_tracker,
        diffusion_artifact=diffusion_artifact,
        reward_model=reward_model,
        action_dim=state_tracker.emb_dim,
        discount_factor=args.gamma,
        tau=DEFAULT_DOSER_SOFT_TAU,
        doser_beta=args.doser_beta,
        doser_lam=args.doser_lam,
        doser_eta=args.doser_eta,
        doser_expectile=args.doser_expectile,
        doser_q_min=args.doser_q_min,
        doser_action_samples=args.doser_action_samples,
        doser_policy_freq=args.doser_policy_freq,
        doser_target_update_freq=args.doser_target_update_freq,
        diffusion_sample_steps=args.diffusion_sample_steps,
        exploration_eps=args.explore_eps if args.exploration_noise else 0.0,
        log_interval=args.doser_log_interval,
        catalog_chunk_size=args.doser_catalog_chunk_size,
    )
    rec_policy = RecPolicy(args, policy, state_tracker)

    train_collector = Collector(
        rec_policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=args.exploration_noise,
        remove_recommended_ids=args.remove_recommended_ids,
    )
    test_collector_set = CollectorSet(
        rec_policy,
        test_envs_dict,
        args.buffer_size,
        args.test_num,
        exploration_noise=args.exploration_noise,
        force_length=args.force_length,
    )
    return rec_policy, train_collector, test_collector_set, (optim_rl, optim_state)


def resolve_default_artifact_name(args: argparse.Namespace) -> str:
    """为不同环境推断默认扩散产物名。"""

    if args.diffusion_artifact_name:
        return args.diffusion_artifact_name
    if args.env == "KuaiEnv-v0":
        return "DM_KuaiEnv-v0_small_data"
    return args.env


def main(args: argparse.Namespace) -> None:
    """执行 DORL-DOSER 主训练流程。"""

    model_save_path, logger_path = prepare_dir_log(args)
    ensemble_models = prepare_user_model(args)
    env, dataset, kwargs_um = get_true_env(args)
    train_envs, reward_model_config = prepare_train_envs_and_reward_model(
        args=args,
        ensemble_models=ensemble_models,
        env=env,
        dataset=dataset,
        kwargs_um=kwargs_um,
    )
    test_envs_dict = prepare_test_envs(args, env, kwargs_um)
    state_tracker = setup_state_tracker(
        args,
        ensemble_models,
        env,
        train_envs,
        test_envs_dict,
    )

    args.diffusion_artifact_name = resolve_default_artifact_name(args)
    diffusion_artifact = load_diffusion_artifact(
        save_root=args.diffusion_save_root,
        env_name=args.env,
        artifact_name=args.diffusion_artifact_name,
        device=args.device,
    )
    wandb_metadata = {
        "diffusion_artifact_dir": diffusion_artifact.artifact_dir,
        "diffusion_state_dim": diffusion_artifact.state_dim,
        "diffusion_action_dim": diffusion_artifact.action_dim,
        "diffusion_state_threshold": diffusion_artifact.state_threshold,
        "diffusion_action_threshold": diffusion_artifact.action_threshold,
        "model_save_path": model_save_path,
        "logger_path": logger_path,
    }
    policy, train_collector, test_collector_set, optim = setup_policy_model(
        args=args,
        state_tracker=state_tracker,
        train_envs=train_envs,
        test_envs_dict=test_envs_dict,
        reward_model_config=reward_model_config,
        diffusion_artifact=diffusion_artifact,
    )

    set_wandb(args, run_metadata=wandb_metadata)
    try:
        learn_policy(
            args,
            env,
            dataset,
            policy,
            train_collector,
            test_collector_set,
            state_tracker,
            optim,
            model_save_path,
            logger_path,
            trainer="offpolicy",
        )
    finally:
        finish_wandb()


if __name__ == "__main__":
    trainer = "offpolicy"
    args_all = get_args_all(trainer)
    args_env = get_env_args(args_all)
    args_dorl_doser = get_args_dorl_doser()
    args_all.__dict__.update(args_env.__dict__)
    args_all.__dict__.update(args_dorl_doser.__dict__)
    try:
        main(args_all)
    except Exception:
        trace_info = traceback.format_exc()
        print(trace_info)
        logger.error(trace_info)
