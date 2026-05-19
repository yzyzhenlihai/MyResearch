"""KuaiEnv-v0 上的 on-policy DORL-DOSER 训练入口。"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any, Dict, Tuple

import torch
from logzero import logger

sys.path.extend([".", "./examples/policy", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import (  # noqa: E402
    get_args_all,
    learn_policy,
    prepare_dir_log,
    prepare_test_envs,
    prepare_user_model,
    setup_state_tracker,
)

from dorl_doser import (  # noqa: E402
    DEFAULT_DIFFUSION_SAMPLE_STEPS,
    DEFAULT_DOSER_LOG_INTERVAL,
    DEFAULT_WANDB_PROJECT,
    finish_wandb,
    prepare_train_envs_and_reward_model,
    resolve_default_artifact_name,
    set_wandb,
)
from src.core.collector.collector import Collector  # noqa: E402
from src.core.collector.collector_set import CollectorSet  # noqa: E402
from src.core.policy.RecPolicy import RecPolicy  # noqa: E402
from src.core.policy.doser import (  # noqa: E402
    A2CDOSERAugmentedCritic,
    CounterfactualRewardModel,
    OnPolicyDORLDOSERPolicy,
    RewardModelConfig,
    load_diffusion_artifact,
)
from src.core.util.data import get_env_args, get_true_env  # noqa: E402
from src.tianshou.tianshou.data import VectorReplayBuffer  # noqa: E402
from src.tianshou.tianshou.env import DummyVectorEnv  # noqa: E402
from src.tianshou.tianshou.utils.net.common import ActorCritic, Net  # noqa: E402
from src.tianshou.tianshou.utils.net.discrete import Actor  # noqa: E402


def get_args_dorl_doser_onpolicy() -> argparse.Namespace:
    """解析 on-policy DORL-DOSER 额外超参数。

    Returns:
        argparse.Namespace: 仅包含 on-policy DORL-DOSER 入口新增参数的命名空间。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DORL_DOSER_ONPOLICY")
    parser.add_argument("--message", type=str, default="DORL_DOSER_ONPOLICY")

    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--rew-norm", action="store_true", default=False)

    parser.add_argument("--diffusion_save_root", type=str, default="saved_models")
    parser.add_argument("--diffusion_artifact_name", type=str, default=None)
    parser.add_argument(
        "--diffusion_sample_steps",
        type=int,
        default=DEFAULT_DIFFUSION_SAMPLE_STEPS,
    )

    parser.add_argument("--doser_beta", type=float, default=0.001)
    parser.add_argument("--doser_lam", type=float, default=0.001)
    parser.add_argument("--doser_eta", type=float, default=0.9)
    parser.add_argument("--doser_expectile", type=float, default=0.9)
    parser.add_argument("--doser_action_samples", type=int, default=10)
    parser.add_argument("--doser_q_min", type=float, default=0.0)
    parser.add_argument("--doser_aux_critic_coef", type=float, default=1.0)
    parser.add_argument(
        "--doser_detach_aux_state",
        dest="doser_detach_aux_state",
        action="store_true",
    )
    parser.add_argument(
        "--no_doser_detach_aux_state",
        dest="doser_detach_aux_state",
        action="store_false",
    )
    parser.set_defaults(doser_detach_aux_state=True)
    parser.add_argument("--doser_log_interval", type=int, default=DEFAULT_DOSER_LOG_INTERVAL)

    parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_job_type", type=str, default="train")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_dir", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)

    parser.add_argument(
        "--is_exposure_intervention",
        dest="use_exposure_intervention",
        action="store_true",
    )
    parser.add_argument(
        "--no_exposure_intervention",
        dest="use_exposure_intervention",
        action="store_false",
    )
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


def setup_onpolicy_policy_model(
    args: argparse.Namespace,
    state_tracker: torch.nn.Module,
    train_envs: DummyVectorEnv,
    test_envs_dict: Dict[str, DummyVectorEnv],
    reward_model_config: RewardModelConfig,
    diffusion_artifact: Any,
) -> Tuple[
    RecPolicy,
    Collector,
    CollectorSet,
    Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
]:
    """初始化 on-policy DORL-DOSER 的策略、collector 与优化器。

    Args:
        args (argparse.Namespace): 当前训练配置。
        state_tracker (torch.nn.Module): 推荐系统状态编码器。
        train_envs (DummyVectorEnv): 训练环境集合。
        test_envs_dict (Dict[str, DummyVectorEnv]): 测试环境集合。
        reward_model_config (RewardModelConfig): counterfactual reward 配置。
        diffusion_artifact (Any): 预训练 diffusion artifact。

    Returns:
        Tuple[RecPolicy, Collector, CollectorSet, Tuple[torch.optim.Optimizer, torch.optim.Optimizer]]:
        包装后的推荐策略、训练 collector、测试 collector 集合，以及 RL/state tracker
        优化器。
    """

    if args.cpu:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(
            f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        )

    # 共享 backbone，保持 actor 仍是原始 DORL/A2C 的离散策略结构。
    net = Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = A2CDOSERAugmentedCritic(
        preprocess_net=net,
        action_dim=state_tracker.emb_dim,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    ).to(args.device)
    optim_rl = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)

    reward_model = CounterfactualRewardModel(reward_model_config)
    policy = OnPolicyDORLDOSERPolicy(
        actor=actor,
        critic=critic,
        optim=(optim_rl, optim_state),
        dist_fn=torch.distributions.Categorical,
        state_tracker=state_tracker,
        diffusion_artifact=diffusion_artifact,
        reward_model=reward_model,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        reward_normalization=args.rew_norm,
        doser_beta=args.doser_beta,
        doser_lam=args.doser_lam,
        doser_eta=args.doser_eta,
        doser_expectile=args.doser_expectile,
        doser_q_min=args.doser_q_min,
        doser_aux_critic_coef=args.doser_aux_critic_coef,
        doser_detach_aux_state=args.doser_detach_aux_state,
        doser_action_samples=args.doser_action_samples,
        diffusion_sample_steps=args.diffusion_sample_steps,
        log_interval=args.doser_log_interval,
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


def main(args: argparse.Namespace) -> None:
    """执行 on-policy DORL-DOSER 主训练流程。

    Args:
        args (argparse.Namespace): 合并后的完整命令行参数。
    """

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
    policy, train_collector, test_collector_set, optim = setup_onpolicy_policy_model(
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
            trainer="onpolicy",
        )
    finally:
        finish_wandb()


if __name__ == "__main__":
    trainer = "onpolicy"
    args_all = get_args_all(trainer)
    args_env = get_env_args(args_all)
    args_dorl_doser = get_args_dorl_doser_onpolicy()
    args_all.__dict__.update(args_env.__dict__)
    args_all.__dict__.update(args_dorl_doser.__dict__)
    try:
        main(args_all)
    except Exception:
        trace_info = traceback.format_exc()
        print(trace_info)
        logger.error(trace_info)
