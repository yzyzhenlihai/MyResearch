"""KuaiEnv-v0 等推荐环境上的 DARLR 训练入口。"""

import argparse
import os
import pickle
import random
import sys
import traceback
from collections import Counter, defaultdict

import logzero
import numpy as np
import torch
from gymnasium.spaces import Discrete
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
from src.core.darlr import (  # noqa: E402
    DynamicRewardStore,
    SelectorTrainingMetricsCallback,
)
from src.core.envs.Simulated_Env.darlr_dynamic_reward import DARLRDynamicRewardEnv  # noqa: E402
from src.core.envs.Simulated_Env.penalty_ent_exp import (  # noqa: E402
    get_features_of_last_n_items_features,
)
from src.core.policy.RecPolicy import RecPolicy  # noqa: E402
from src.core.policy.darlr import (  # noqa: E402
    DARLRPolicy,
    PreferenceEncoder,
    SelectorActor,
    SelectorCritic,
    SelectorStateEncoder,
)
from src.core.util.data import get_env_args, get_true_env  # noqa: E402
from src.tianshou.tianshou.data import VectorReplayBuffer  # noqa: E402
from src.tianshou.tianshou.env import DummyVectorEnv  # noqa: E402
from src.tianshou.tianshou.utils.net.common import Net  # noqa: E402
from src.tianshou.tianshou.utils.net.discrete import Actor, Critic  # noqa: E402

try:
    import swanlab as wandb
except ImportError:
    wandb = None


DEFAULT_SELECTOR_PREF_DIM = 64
"""selector 默认偏好投影维度。"""

DEFAULT_SELECTOR_CANDIDATE_SIZE = 512
"""selector 默认候选用户池大小。"""

DEFAULT_SELECTOR_K = 10
"""selector 默认参考用户数量。"""


def _positive_float(raw_value: str) -> float:
    """解析严格为正的命令行浮点数。

    Args:
        raw_value (str): argparse 接收的原始字符串。

    Returns:
        float: 解析后的正浮点数。

    Raises:
        argparse.ArgumentTypeError: 当输入不是浮点数或数值不大于零时抛出。
    """

    try:
        parsed_value = float(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a floating-point number") from error
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed_value


def _nonnegative_float(raw_value: str) -> float:
    """解析非负命令行浮点数。

    Args:
        raw_value (str): argparse 接收的原始字符串。

    Returns:
        float: 解析后的非负浮点数。

    Raises:
        argparse.ArgumentTypeError: 当输入不是浮点数或数值小于零时抛出。
    """

    try:
        parsed_value = float(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a floating-point number") from error
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed_value


_SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "").lower()
_SWANLAB_DISABLE_LOGIN = _SWANLAB_MODE in {"offline", "disabled"}

if wandb is not None and not _SWANLAB_DISABLE_LOGIN:
    wandb.login(api_key="ccVCViGdYDi4LOGAy6FBp", save=True)


def _activate_swanlab_metric_logger() -> None:
    """把 trainer 中的实验日志后端切换到 swanlab。

    当前仓库的 trainer 和 `policy_utils` 历史上通过名为 `wandb` 的
    模块变量写入指标。DARLR 入口恢复 swanlab 后，需要在初始化 run
    之后把这些模块变量指向 swanlab，否则 Coat/Yahoo 等数据集训练
    只有 run 初始化，没有 epoch/test 指标曲线。

    Returns:
        None: 仅执行模块级 logger 绑定。
    """

    if wandb is None:
        return

    try:
        import policy_utils as policy_utils_module

        policy_utils_module.wandb = wandb
    except ImportError:
        logzero.logger.warning("Skip binding policy_utils logger to swanlab.")

    for module_name in (
        "tianshou.trainer.base",
        "src.tianshou.tianshou.trainer.base",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            module.wandb = wandb


def set_wandb(args: argparse.Namespace) -> None:
    """按需初始化 swanlab 实验记录。

    Args:
        args (argparse.Namespace): 当前训练配置。

    Returns:
        None: 仅在 swanlab 可用且未禁用时初始化远程记录。
    """

    if wandb is None:
        logzero.logger.info("Skip swanlab because it is unavailable.")
        return
    if _SWANLAB_DISABLE_LOGIN:
        logzero.logger.info("Skip swanlab because SWANLAB_MODE=%s", _SWANLAB_MODE)
        return
    if hasattr(args, "device"):
        args.device = str(args.device)

    wandb.init(
        project="DORL",
        config=args,
        name=(
            f"{args.message}-{args.env}-seed:{args.seed}"
            f"-K:{args.selector_k}-cand:{args.selector_candidate_size}"
            f"-lambda_u:{args.lambda_uncertainty}-lambda_e:{args.lambda_entropy}"
        ),
    )
    _activate_swanlab_metric_logger()


def finish_wandb() -> None:
    """安全结束 swanlab 记录，不让观测后端故障改变训练结果。

    Returns:
        None: 当 swanlab 不可用、被禁用或收尾失败时直接返回。

    Notes:
        swanlab 的 ``finish`` 会访问远程 API。网络、证书或服务端故障属于
        非关键的观测链路异常，因此这里只记录警告，避免覆盖训练阶段的结果
        或原始异常。
    """

    if wandb is None or _SWANLAB_DISABLE_LOGIN:
        return
    try:
        wandb.finish()
    except Exception:
        logzero.logger.warning(
            "Failed to finish swanlab logging; preserving the training result.",
            exc_info=True,
        )


def get_args_DARLR() -> argparse.Namespace:
    """解析 DARLR 特有参数。

    Returns:
        argparse.Namespace: DARLR 训练参数命名空间。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DARLR")
    parser.add_argument("--message", type=str, default="DARLR")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--rew-norm", action="store_true", default=False)

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
    parser.add_argument("--lambda_entropy", type=float, default=0.05)
    parser.add_argument("--read_message", type=str, default="UM")

    parser.add_argument("--selector_k", type=int, default=DEFAULT_SELECTOR_K)
    parser.add_argument("--selector_candidate_size", type=int, default=DEFAULT_SELECTOR_CANDIDATE_SIZE)
    parser.add_argument(
        "--selector_candidate_mode",
        type=str,
        choices=["embedding_topk", "random"],
        default="embedding_topk",
    )
    parser.add_argument(
        "--selector_policy_mode",
        "--selector-policy-mode",
        type=str,
        choices=["learned", "random", "fixed"],
        default="learned",
        help=(
            "learned trains the selector; random samples uniformly without "
            "training; fixed samples from a frozen initialized selector."
        ),
    )
    parser.add_argument("--selector_lambda_s", type=float, default=1.0)
    parser.add_argument("--selector_lambda_d", type=float, default=0.05)
    parser.add_argument("--lambda_uncertainty", type=float, default=0.05)
    parser.add_argument("--darlr_eps", type=float, default=1.0e-8)
    parser.add_argument("--selector_pref_dim", type=int, default=DEFAULT_SELECTOR_PREF_DIM)
    parser.add_argument("--selector_num_heads", type=int, default=1)
    parser.add_argument("--selector_num_layers", type=int, default=1)
    parser.add_argument("--selector_dropout_rate", type=float, default=0.1)
    parser.add_argument(
        "--selector_reward_mode",
        type=str,
        choices=["full", "base", "sim", "div"],
        default="full",
    )
    parser.add_argument(
        "--dynamic_reward_mode",
        type=str,
        choices=["reference_mean", "static_dorl"],
        default="reference_mean",
    )
    parser.add_argument(
        "--dynamic_uncertainty_mode",
        type=str,
        choices=["dynamic", "static", "off"],
        default="dynamic",
    )
    # 新增: paper_core / stabilized 口径切换 (对应文档 §4.4)
    parser.add_argument(
        "--selector_gain_mode",
        type=str,
        choices=["paper_core", "stabilized"],
        default="paper_core",
    )
    # 新增: selector loss 权重、独立学习率
    parser.add_argument("--selector_loss_coef", type=float, default=1.0)
    parser.add_argument("--selector_lr", type=float, default=None,
                        help="If set, use a separate learning rate for selector optimizer.")
    parser.add_argument(
        "--selector_ent_coef",
        "--selector-ent-coef",
        type=_nonnegative_float,
        default=0.0,
        help="Entropy coefficient used only by the selector actor.",
    )
    parser.add_argument(
        "--selector_reward_normalization",
        "--selector-reward-normalization",
        dest="selector_reward_normalization",
        action="store_true",
    )
    parser.add_argument(
        "--no_selector_reward_normalization",
        "--no-selector-reward-normalization",
        dest="selector_reward_normalization",
        action="store_false",
    )
    parser.add_argument(
        "--selector_advantage_normalization",
        "--selector-advantage-normalization",
        dest="selector_advantage_normalization",
        action="store_true",
    )
    parser.add_argument(
        "--no_selector_advantage_normalization",
        "--no-selector-advantage-normalization",
        dest="selector_advantage_normalization",
        action="store_false",
    )
    parser.set_defaults(
        selector_reward_normalization=False,
        selector_advantage_normalization=False,
    )
    parser.add_argument(
        "--selector_normalization_eps",
        "--selector-normalization-eps",
        type=_positive_float,
        default=1.0e-8,
    )
    # 新增: DORL 静态方差惩罚 (供 static 消融与静态基线使用)
    parser.add_argument("--lambda_variance", type=float, default=0.0)

    return parser.parse_known_args()[0]


def validate_darlr_args(args: argparse.Namespace) -> None:
    """在加载数据前校验 DARLR 训练参数的数值与结构约束。

    Args:
        args (argparse.Namespace): 合并通用、环境和 DARLR 参数后的命名空间。

    Returns:
        None: 参数合法时不修改输入。

    Raises:
        ValueError: 当学习率、梯度阈值、selector 结构或损失权重非法时抛出。
    """

    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.selector_lr is not None and args.selector_lr <= 0:
        raise ValueError("--selector_lr must be positive when provided.")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive when provided.")
    if args.selector_k <= 0:
        raise ValueError("--selector_k must be positive.")
    if args.selector_candidate_size < args.selector_k:
        raise ValueError(
            "--selector_candidate_size must be greater than or equal to "
            "--selector_k."
        )
    if args.selector_pref_dim <= 0:
        raise ValueError("--selector_pref_dim must be positive.")
    if args.selector_num_heads <= 0:
        raise ValueError("--selector_num_heads must be positive.")
    if args.selector_pref_dim % args.selector_num_heads != 0:
        raise ValueError(
            "--selector_pref_dim must be divisible by --selector_num_heads."
        )
    if args.selector_num_layers <= 0:
        raise ValueError("--selector_num_layers must be positive.")
    if not 0 <= args.selector_dropout_rate < 1:
        raise ValueError("--selector_dropout_rate must be in [0, 1).")
    if args.selector_loss_coef < 0:
        raise ValueError("--selector_loss_coef must be non-negative.")
    if args.selector_ent_coef < 0:
        raise ValueError("--selector_ent_coef must be non-negative.")
    if args.selector_normalization_eps <= 0:
        raise ValueError("--selector_normalization_eps must be positive.")
    for argument_name in (
        "selector_lambda_s",
        "selector_lambda_d",
        "lambda_uncertainty",
        "lambda_entropy",
        "lambda_variance",
    ):
        if getattr(args, argument_name) < 0:
            raise ValueError(f"--{argument_name} must be non-negative.")


def get_entropy(mylist, need_count=True):
    """计算归一化熵。

    Args:
        mylist: 原始序列或频数字典。
        need_count (bool): 是否需要先统计频次。

    Returns:
        float: 归一化熵，长度不足时返回 1。
    """

    if len(mylist) <= 1:
        return 1
    cnt_dict = Counter(mylist) if need_count else mylist
    prob = np.array(list(cnt_dict.values())) / sum(cnt_dict.values())
    log_prob = np.log2(prob)
    entropy = -np.sum(log_prob * prob) / np.log2(len(cnt_dict))
    return entropy


def get_save_entropy_mat(dataset, entropy_window, feature_level=True, is_sorted=True):
    """根据训练集统计 DORL entropy map。

    Args:
        dataset: EasyRL4Rec 数据集对象。
        entropy_window: 历史窗口长度列表。
        feature_level (bool): 是否按 item feature 统计。
        is_sorted (bool): 历史窗口内是否排序。

    Returns:
        tuple: `(map_entropy, map_item_feat)`。
    """

    df_train, _, df_item, _ = dataset.get_train_data()
    map_item_feat = dict(zip(df_item.index, df_item["tags"])) if feature_level else None
    if "timestamp" not in df_train.columns:
        df_train = df_train.rename(columns={"time_ms": "timestamp"})

    map_hist_count = defaultdict(lambda: defaultdict(int))
    if len(set(entropy_window) - {0}):
        df_uit = df_train[["user_id", "item_id", "timestamp"]].sort_values(["user_id", "timestamp"])
        last_user = -1
        history_actions = []

        def update_map(target_map, history, item, require_len):
            """更新历史窗口频次表。"""

            if len(history) < require_len:
                return
            history_key = tuple(sorted(history[-require_len:]) if is_sorted else history[-require_len:])
            target_map[history_key][item] += 1

        for user_id, item_id, _ in tqdm(df_uit.to_numpy(), total=len(df_uit), desc="count frequency..."):
            user_id = int(user_id)
            item_id = int(item_id)
            if user_id != last_user:
                last_user = user_id
                history_actions = []

            if feature_level:
                for feature in map_item_feat[item_id]:
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
        key: get_entropy(value, need_count=False)
        for key, value in tqdm(map_hist_count.items(), total=len(map_hist_count), desc="compute entropy...")
    }
    return map_entropy, map_item_feat


def prepare_train_envs(args, ensemble_models, env, dataset, kwargs_um):
    """构造 DARLR 训练环境。

    Args:
        args (argparse.Namespace): 训练参数。
        ensemble_models: 已加载的 DeepFM ensemble。
        env: 真实测试环境。
        dataset: EasyRL4Rec 数据集。
        kwargs_um (dict): 真实环境初始化参数。

    Returns:
        tuple: `(train_envs, predicted_mat)`。
    """

    entropy_dict = {}
    map_item_feat = None
    if len(set(args.entropy_window) - {0}):
        map_entropy, map_item_feat = get_save_entropy_mat(
            dataset,
            args.entropy_window,
            args.feature_level,
            args.is_sorted,
        )
        entropy_dict["map"] = map_entropy

    entropy_min = 0.0
    entropy_max = 0.0
    if entropy_dict.get("map"):
        for entropy_term in set(args.entropy_window):
            if entropy_term == 0:
                continue
            values = [
                value for key, value in entropy_dict["map"].items()
                if len(key) == entropy_term
            ]
            entropy_min += min(values + [1.0])
            entropy_max += max(values + [1.0])

    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    # DARLR 需要静态 V0 用于静态 uncertainty 消融, 也用于 static_dorl 模式回退。
    maxvar_mat = None
    try:
        with open(ensemble_models.VAR_MAT_PATH, "rb") as file:
            maxvar_mat = pickle.load(file)
        assert maxvar_mat.shape == predicted_mat.shape, (
            "VAR_MAT_PATH shape must match PREDICTION_MAT_PATH shape, "
            f"got {maxvar_mat.shape} vs {predicted_mat.shape}."
        )
    except FileNotFoundError:
        logzero.logger.warning(
            "VAR_MAT_PATH not found; static uncertainty ablation & static_dorl "
            "mode will be unavailable for DARLR."
        )

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
        "alpha_u": None,
        "beta_i": None,
        "entropy_dict": entropy_dict,
        "entropy_window": args.entropy_window,
        "lambda_entropy": args.lambda_entropy,
        "step_n_actions": max(args.entropy_window) if len(args.entropy_window) else 0,
        "entropy_min": entropy_min,
        "entropy_max": entropy_max,
        "feature_level": args.feature_level,
        "map_item_feat": map_item_feat,
        "is_sorted": args.is_sorted,
        "lambda_uncertainty": args.lambda_uncertainty,
        "darlr_eps": args.darlr_eps,
        "dynamic_reward_mode": args.dynamic_reward_mode,
        "dynamic_uncertainty_mode": args.dynamic_uncertainty_mode,
        # DORL 静态 variance 惩罚 (static_dorl 模式与静态基线消融需要)
        "maxvar_mat": maxvar_mat,
        "lambda_variance": float(getattr(args, "lambda_variance", 0.0)),
    }

    train_envs = DummyVectorEnv(
        [lambda: DARLRDynamicRewardEnv(**env_kwargs) for _ in range(args.training_num)]
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    return train_envs, predicted_mat, maxvar_mat


def setup_policy_model(args, state_tracker, train_envs, test_envs_dict, predicted_mat, maxvar_mat=None):
    """初始化 DARLR policy、collector 和优化器。

    Args:
        args (argparse.Namespace): 训练参数。
        state_tracker: 推荐状态追踪器。
        train_envs: 训练向量环境。
        test_envs_dict (dict): 测试环境集合。
        predicted_mat (np.ndarray): world model 预测矩阵。

    Returns:
        tuple: `(rec_policy, train_collector, test_collector_set, optim)`。
    """

    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    net = Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)

    preference_encoder = PreferenceEncoder(
        num_items=predicted_mat.shape[1],
        pref_dim=args.selector_pref_dim,
    ).to(args.device)
    selector_state_encoder = SelectorStateEncoder(
        pref_dim=args.selector_pref_dim,
        max_len=args.selector_k,
        num_heads=args.selector_num_heads,
        num_layers=args.selector_num_layers,
        dropout_rate=args.selector_dropout_rate,
    ).to(args.device)
    selector_actor = SelectorActor(
        recommender_state_dim=args.state_dim,
        pref_dim=args.selector_pref_dim,
        hidden_sizes=args.hidden_sizes,
    ).to(args.device)
    selector_critic = SelectorCritic(
        recommender_state_dim=args.state_dim,
        pref_dim=args.selector_pref_dim,
        hidden_sizes=args.hidden_sizes,
    ).to(args.device)

    # P0#7: 拆分成两个独立 optimizer。recommender 只优化 actor/critic;
    # selector 单独优化 preference_encoder / selector_state_encoder /
    # selector_actor / selector_critic; state_tracker 依旧独立。
    optim_rec = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
    )
    selector_lr = args.selector_lr if args.selector_lr is not None else args.lr
    optim_selector = torch.optim.Adam(
        list(preference_encoder.parameters())
        + list(selector_state_encoder.parameters())
        + list(selector_actor.parameters())
        + list(selector_critic.parameters()),
        lr=selector_lr,
    )
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    # save_model_fn 只识别 optim[0]/optim[-1], 对接旧接口: [optim_rec, optim_state].
    # optim_selector 由 DARLRPolicy.darlr_extra_state() 单独保存。
    optim = [optim_rec, optim_state]

    # P0#4: 共享 previous-reward store, 全 run 唯一。
    dynamic_reward_store = DynamicRewardStore(predicted_mat)

    policy = DARLRPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        state_tracker=state_tracker,
        predicted_mat=predicted_mat,
        selector_actor=selector_actor,
        selector_critic=selector_critic,
        preference_encoder=preference_encoder,
        selector_state_encoder=selector_state_encoder,
        selector_k=args.selector_k,
        selector_candidate_size=args.selector_candidate_size,
        selector_candidate_mode=args.selector_candidate_mode,
        selector_lambda_s=args.selector_lambda_s,
        selector_lambda_d=args.selector_lambda_d,
        selector_reward_mode=args.selector_reward_mode,
        darlr_eps=args.darlr_eps,
        selector_discount_factor=args.gamma,
        selector_gain_mode=args.selector_gain_mode,
        selector_loss_coef=args.selector_loss_coef,
        selector_policy_mode=args.selector_policy_mode,
        selector_ent_coef=args.selector_ent_coef,
        selector_reward_normalization=args.selector_reward_normalization,
        selector_advantage_normalization=(
            args.selector_advantage_normalization
        ),
        selector_normalization_eps=args.selector_normalization_eps,
        dynamic_reward_store=dynamic_reward_store,
        maxvar_mat=maxvar_mat,
        optim_selector=optim_selector,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        reward_normalization=args.rew_norm,
        action_space=Discrete(args.action_shape),
        action_bound_method="",
        action_scaling=False,
    )
    rec_policy = RecPolicy(args, policy, state_tracker)
    # 通过通用 trainer 的 epoch callback 链路，将 selector 训练诊断量与
    # NX_0 等评估指标使用同一个 epoch 横坐标上传 SwanLab。
    rec_policy.training_callbacks = [SelectorTrainingMetricsCallback(policy)]

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
    return rec_policy, train_collector, test_collector_set, optim


def main(args: argparse.Namespace) -> None:
    """执行 DARLR 训练流程。

    Args:
        args (argparse.Namespace): 完整训练参数。

    Returns:
        None: 训练结果由 trainer 打印并写入日志。
    """

    validate_darlr_args(args)
    model_save_path, logger_path = prepare_dir_log(args)
    ensemble_models = prepare_user_model(args)
    env, dataset, kwargs_um = get_true_env(args)
    train_envs, predicted_mat, maxvar_mat = prepare_train_envs(args, ensemble_models, env, dataset, kwargs_um)
    test_envs_dict = prepare_test_envs(args, env, kwargs_um)
    state_tracker = setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict)
    policy, train_collector, test_collector_set, optim = setup_policy_model(
        args,
        state_tracker,
        train_envs,
        test_envs_dict,
        predicted_mat,
        maxvar_mat=maxvar_mat,
    )
    set_wandb(args)
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
            trainer="onpolicy",
        )
    finally:
        finish_wandb()


if __name__ == "__main__":
    trainer = "onpolicy"
    args_all = get_args_all(trainer)
    args_env = get_env_args(args_all)
    args_darlr = get_args_DARLR()
    args_all.__dict__.update(args_env.__dict__)
    args_all.__dict__.update(args_darlr.__dict__)
    try:
        main(args_all)
    except Exception:
        error = traceback.format_exc()
        print(error)
        logzero.logger.error(error)
        raise
