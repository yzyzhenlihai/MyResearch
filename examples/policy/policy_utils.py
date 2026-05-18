import datetime
import functools
import json
import os
import pickle
import pprint
import random
import socket
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from src.core.state_tracker.Caser import StateTracker_Caser
from src.core.state_tracker.Average import StateTrackerAvg
from src.core.state_tracker.GRU import StateTracker_GRU
from src.core.state_tracker.NextItNet import StateTracker_NextItNet
from src.core.state_tracker.SASRec import StateTracker_SASRec
from src.core.util.inputs import get_dataset_columns
from src.core.userModel.user_model_ensemble import EnsembleModel
from src.core.envs.Simulated_Env.base import BaseSimulatedEnv

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.core.util.data import get_true_env
from src.core.evaluation.evaluator import Evaluator_Feat, Evaluator_Coverage_Count, Evaluator_User_Experience, save_model_fn
from src.core.evaluation.loggers import LoggerEval_Policy

# from src.tianshou.tianshou.data.buffer.base import 
from src.tianshou.tianshou.data import ReplayBuffer, VectorReplayBuffer, Batch
from src.tianshou.tianshou.env import DummyVectorEnv
from src.tianshou.tianshou.trainer import onpolicy_trainer, offpolicy_trainer
from src.tianshou.tianshou.trainer.offline import offline_trainer

from src.core.util.utils import create_dir
from src.core.util.wandb_utils import load_wandb
import logzero
from logzero import logger
from omegaconf import open_dict

wandb = load_wandb(repo_root=Path(__file__).resolve().parents[2])

OFFPOLICY_COLLECT_MODE_STEP = "step"
"""离策略训练按 transition 数收集样本的模式名。"""

OFFPOLICY_COLLECT_MODE_EPISODE = "episode"
"""离策略训练按完整轨迹数收集样本的模式名。"""

OFFPOLICY_COLLECT_MODES = (
    OFFPOLICY_COLLECT_MODE_STEP,
    OFFPOLICY_COLLECT_MODE_EPISODE,
)
"""离策略训练允许的 collect 模式集合。"""


def _get_active_wandb_run():
    """返回当前激活的 wandb run；未初始化时返回空。"""

    if wandb is None:
        return None
    return getattr(wandb, "run", None)


def _to_wandb_scalar(value):
    """把 trainer 结果字典里的值转换成适合 wandb 的标量。"""

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return None
    if isinstance(value, (float, int, bool)):
        return value
    return None


def _log_final_result_to_wandb(result):
    """把 trainer 最终结果汇总写入 wandb。"""

    active_run = _get_active_wandb_run()
    if active_run is None:
        return

    final_metrics = {}
    for key, value in result.items():
        scalar_value = _to_wandb_scalar(value)
        if scalar_value is None:
            continue
        final_metrics[f"final/{key}"] = scalar_value

    if not final_metrics:
        return

    wandb.log(final_metrics)
    for key, value in final_metrics.items():
        active_run.summary[key] = value


def _build_offpolicy_collect_kwargs(args):
    """构造离策略训练的 collect 参数。

    Args:
        args (argparse.Namespace): 训练配置，需包含
            `offpolicy_collect_mode`、`step_per_collect` 与
            `episode_per_collect` 字段。

    Returns:
        dict: 传递给 `offpolicy_trainer` 的 collect 相关关键字参数。

    Raises:
        ValueError: 当 collect 模式非法，或对应模式下的收集数量不是正整数时抛出。
    """

    collect_mode = getattr(args, "offpolicy_collect_mode", OFFPOLICY_COLLECT_MODE_STEP)
    if collect_mode not in OFFPOLICY_COLLECT_MODES:
        raise ValueError(
            f"Unsupported offpolicy collect mode: {collect_mode}. "
            f"Expected one of {OFFPOLICY_COLLECT_MODES}."
        )

    if collect_mode == OFFPOLICY_COLLECT_MODE_STEP:
        step_per_collect = int(getattr(args, "step_per_collect", 0))
        if step_per_collect <= 0:
            raise ValueError(
                "step_per_collect must be a positive integer when "
                "offpolicy_collect_mode='step'."
            )
        return {
            "step_per_collect": step_per_collect,
            "episode_per_collect": None,
        }

    episode_per_collect = int(getattr(args, "episode_per_collect", 0))
    if episode_per_collect <= 0:
        raise ValueError(
            "episode_per_collect must be a positive integer when "
            "offpolicy_collect_mode='episode'."
        )
    return {
        "step_per_collect": None,
        "episode_per_collect": episode_per_collect,
    }

def get_args_all(trainer="onpolicy"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--user_model_name", type=str, default="DeepFM")
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--cuda', default=0, type=int)

    # training
    parser.add_argument('--remove_recommended_ids', action="store_true", default=False)

    parser.add_argument('--is_draw_bar', dest='draw_bar', action='store_true')
    parser.add_argument('--no_draw_bar', dest='draw_bar', action='store_false')
    parser.set_defaults(draw_bar=False)

    parser.add_argument('--is_userinfo', dest='is_userinfo', action='store_true')
    parser.add_argument('--no_userinfo', dest='is_userinfo', action='store_false')
    parser.set_defaults(is_userinfo=False)

    parser.add_argument('--is_all_item_ranking', dest='is_all_item_ranking', action='store_true')
    parser.add_argument('--no_all_item_ranking', dest='is_all_item_ranking', action='store_false')
    parser.set_defaults(all_item_ranking=False)

    parser.add_argument('--cpu', dest='cpu', action='store_true')
    parser.set_defaults(cpu=False)

    parser.add_argument('--is_save', dest='is_save', action='store_true')
    parser.add_argument('--no_save', dest='is_save', action='store_false')
    parser.set_defaults(is_save=False)

    parser.add_argument('--is_use_userEmbedding', dest='use_userEmbedding', action='store_true')
    parser.add_argument('--no_use_userEmbedding', dest='use_userEmbedding', action='store_false')
    parser.set_defaults(use_userEmbedding=False)

    parser.add_argument('--is_use_pretrained_embedding', dest='use_pretrained_embedding', action='store_true')
    parser.add_argument('--no_use_pretrained_embedding', dest='use_pretrained_embedding', action='store_false')
    parser.set_defaults(use_pretrained_embedding=True)

    parser.add_argument('--is_exploration_noise', dest='exploration_noise', action='store_true')
    parser.add_argument('--no_exploration_noise', dest='exploration_noise', action='store_false')
    parser.set_defaults(exploration_noise=(False if trainer == "onpolicy" else True))
    parser.add_argument('--explore_eps', default=0.01, type=float)

    parser.add_argument('--is_need_state_norm', dest='need_state_norm', action='store_true')
    parser.add_argument('--no_need_state_norm', dest='need_state_norm', action='store_false')
    parser.set_defaults(need_state_norm=False)

    parser.add_argument('--is_freeze_emb', dest='freeze_emb', action='store_true')
    parser.add_argument('--no_freeze_emb', dest='freeze_emb', action='store_false')
    parser.set_defaults(freeze_emb=False)

    # state tracker
    parser.add_argument('--reward_handle', type=str, default='cat')  # in {"no", "cat", "cat2", "mul"}
    parser.add_argument("--which_tracker", type=str, default="avg")  # in {"avg", "caser", "sasrec", "gru", "nextitnet"}

    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument('--window_size', default=3, type=int)

    parser.add_argument('--is_random_init', dest='random_init', action='store_true')
    parser.add_argument('--no_random_init', dest='random_init', action='store_false')
    parser.set_defaults(random_init=True)

    # State_tracker Caser
    parser.add_argument('--filter_sizes', type=int, nargs='*', default=[2, 3, 4])
    parser.add_argument("--num_filters", type=int, default=16)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    # State_tracker SASRec
    # parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=1)
    # State_tracker nextitnet
    parser.add_argument("--dilations", type=str, default='[1, 2, 1, 2, 1, 2]')

    # tianshou
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=(1024 if trainer == "onpolicy" else 64))
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])

    # For off-policy methods: DDPG, DQN, C51
    parser.add_argument('--episode-per-collect', type=int, default=100)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--update-per-step', type=float, default=0.125)

    parser.add_argument('--training-num', type=int, default=100)
    parser.add_argument('--test-num', type=int, default=100)

    parser.add_argument('--render', type=float, default=0)
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--step-per-epoch', type=int, default=(100000 if trainer == "onpolicy" else 10000))
    parser.add_argument('--step-per-collect', type=int, default=100)
    parser.add_argument(
        '--offpolicy_collect_mode',
        type=str,
        choices=list(OFFPOLICY_COLLECT_MODES),
        default=OFFPOLICY_COLLECT_MODE_STEP,
    )
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')

    parser.add_argument("--read_message", type=str, default="UM")

    args = parser.parse_known_args()[0]
    return args


def prepare_dir_log(args):
    # %% 1. Create dirs
    MODEL_SAVE_PATH = os.path.join("", "saved_models", args.env, args.model_name)
    create_dirs = [os.path.join("", "saved_models"),
                   os.path.join("", "saved_models", args.env),
                   MODEL_SAVE_PATH,
                   os.path.join(MODEL_SAVE_PATH, "logs")]
    create_dir(create_dirs)

    nowtime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d-%H_%M_%S")
    logger_path = os.path.join(MODEL_SAVE_PATH, "logs", "[{}]_{}.log".format(args.message, nowtime))
    logzero.logfile(logger_path)
    hostname = socket.gethostname()
    args.hostname = hostname
    logger.info(json.dumps(vars(args), indent=2))

    return MODEL_SAVE_PATH, logger_path


def prepare_user_model(args):
    
    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)

    UM_SAVE_PATH = os.path.join("", "saved_models", args.env, args.user_model_name)
    # MODEL_MAT_PATH = os.path.join(UM_SAVE_PATH, "mats", f"[{args.read_message}]_mat.pickle")
    MODEL_PARAMS_PATH = os.path.join(UM_SAVE_PATH, "params", f"[{args.read_message}]_params.pickle")

    with open(MODEL_PARAMS_PATH, "rb") as file:
        model_params = pickle.load(file)

    n_models = model_params["n_models"]
    model_params.pop('n_models')

    ensemble_models = EnsembleModel(n_models, args.read_message, UM_SAVE_PATH, **model_params)
    ensemble_models.load_all_models()

    return ensemble_models


def prepare_train_envs(args, ensemble_models, env, kwargs_um):
    with open(ensemble_models.PREDICTION_MAT_PATH, "rb") as file:
        predicted_mat = pickle.load(file)

    kwargs = {
        "ensemble_models": ensemble_models,
        "env_task_class": type(env),
        "task_env_param": kwargs_um,
        "task_name": args.env,
        "predicted_mat": predicted_mat,
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_envs = DummyVectorEnv(
        [lambda: BaseSimulatedEnv(**kwargs) for _ in range(args.training_num)])
    train_envs.seed(args.seed)

    return train_envs

"""
构造测试环境
这里测试环境就是KuaiEnv/KuaiRand定义的环境，里面的mat是user-item的rewards
"""
def prepare_test_envs(args, env, kwargs_um):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_task_class = type(env)
    test_envs = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])
    test_envs_NX_0 = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])
    test_envs_NX_x = DummyVectorEnv(
        [lambda: env_task_class(**kwargs_um) for _ in range(args.test_num)])
    test_envs.seed(args.seed)
    test_envs_NX_0.seed(args.seed)
    test_envs_NX_x.seed(args.seed)

    test_envs_dict = {"FB": test_envs, "NX_0": test_envs_NX_0, f"NX_{args.force_length}": test_envs_NX_x}

    # seed


    return test_envs_dict


def prepare_train_test_envs(args, ensemble_models):
    env, dataset, kwargs_um = get_true_env(args)
    train_envs = prepare_train_envs(args, ensemble_models, env, kwargs_um)
    test_envs_dict = prepare_test_envs(args, env, kwargs_um)
    return env, dataset, train_envs, test_envs_dict


def setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict, use_buffer_in_train=False):
    if use_buffer_in_train:
        buffer = train_envs
        train_envs = None
    # feat_user(3327,41) feat_item(1411, 8) 
    saved_embedding = ensemble_models.load_val_user_item_embedding(freeze_emb=args.freeze_emb) # 这边加载的是验证集的物品嵌入
    if args.use_pretrained_embedding:
        # if args.which_tracker.lower() == "avg":
        user_columns, action_columns, feedback_columns, have_user_embedding, have_action_embedding, have_feedback_embedding = \
            get_dataset_columns(saved_embedding["feat_user"].weight.shape[1], # 8
                                saved_embedding["feat_item"].weight.shape[1], # 41
                                env.mat.shape[0], env.mat.shape[1], envname=args.env)
    else:
        user_columns, action_columns, feedback_columns, have_user_embedding, have_action_embedding, have_feedback_embedding = \
            get_dataset_columns(args.embedding_dim, args.embedding_dim, env.mat.shape[0], env.mat.shape[1],
                                envname=args.env)

    args.action_shape = action_columns[0].vocabulary_size # 等于item的数量 KuaiEnv 3327
    args.state_dim = action_columns[0].embedding_dim # KuaiEnv 41

    if args.use_userEmbedding:
        args.state_dim = action_columns[0].embedding_dim + saved_embedding.feat_user.weight.shape[1]
    train_max = None
    train_min = None
    test_max = None
    test_min = None
    if train_envs is not None:
        if use_buffer_in_train:
            train_max = buffer.rew.max()
            train_min = buffer.rew.min()
        else:
            train_max = train_envs.get_env_attr("MAX_R")[0] - train_envs.get_env_attr("MIN_R")[0]
            train_min = 0

    if test_envs_dict is not None:
        test_max = test_envs_dict['FB'].get_env_attr("mat")[0].max()
        test_min = test_envs_dict['FB'].get_env_attr("mat")[0].min()

    if args.which_tracker.lower() == "caser":
        assert args.window_size >= max(args.filter_sizes)
        state_tracker = StateTracker_Caser(user_columns, action_columns, feedback_columns, args.state_dim,
                                           train_max, train_min, test_max, test_min, reward_handle=args.reward_handle,
                                           saved_embedding=saved_embedding,
                                           device=args.device,
                                           window_size=args.window_size,
                                           filter_sizes=args.filter_sizes, num_filters=args.num_filters,
                                           dropout_rate=args.dropout_rate).to(args.device)
    elif args.which_tracker.lower() == "gru":
        state_tracker = StateTracker_GRU(user_columns, action_columns, feedback_columns, args.state_dim,
                                         train_max, train_min, test_max, test_min, reward_handle=args.reward_handle,
                                         saved_embedding=saved_embedding,
                                         device=args.device,
                                         window_size=args.window_size).to(args.device)
    elif args.which_tracker.lower() == "sasrec":
        state_tracker = StateTracker_SASRec(user_columns, action_columns, feedback_columns, args.state_dim,
                                            train_max, train_min, test_max, test_min, reward_handle=args.reward_handle,
                                            saved_embedding=saved_embedding,
                                            device=args.device, window_size=args.window_size,
                                            dropout_rate=args.dropout_rate, num_heads=args.num_heads).to(args.device)
    elif args.which_tracker.lower() == "nextitnet":
        state_tracker = StateTracker_NextItNet(user_columns, action_columns, feedback_columns, args.state_dim,
                                               train_max, train_min, test_max, test_min,
                                               reward_handle=args.reward_handle, saved_embedding=saved_embedding,
                                               device=args.device, window_size=args.window_size,
                                               dilations=args.dilations).to(args.device)
    elif args.which_tracker.lower() == "avg":
        assert args.use_pretrained_embedding
        state_tracker = StateTrackerAvg(user_columns, action_columns, feedback_columns, args.state_dim,
                                        train_max, train_min, test_max, test_min, reward_handle=args.reward_handle,
                                        saved_embedding=saved_embedding,
                                        device=args.device, window_size=args.window_size,
                                        use_userEmbedding=args.use_userEmbedding).to(args.device)
    else:
        return None

    state_tracker.set_need_normalization(args.need_state_norm)
    args.state_dim = state_tracker.final_dim

    return state_tracker



def setup_state_tracker_bigmatrix(args, ensemble_models, env=None, train_envs=None, test_envs_dict=None,
                                  use_buffer_in_train=False, use_training_embedding=True):
    """
    Variant of `setup_state_tracker` that loads embeddings aligned with the (big) training matrix.

    - If `use_training_embedding` is True, this loads embeddings saved for the training set
      via `EnsembleModel.load_user_item_embedding()`; otherwise it falls back to val embeddings.
    - If `env` is None, the function infers user/item counts from the saved embedding sizes.
    This function does not modify the original `setup_state_tracker` so other code paths remain unchanged.
    """
    if use_buffer_in_train:
        buffer = train_envs
        train_envs = None

    # load embeddings for big matrix (training) or val depending on flag
    if use_training_embedding:
        saved_embedding = ensemble_models.load_user_item_embedding(freeze_emb=args.freeze_emb)
    else:
        saved_embedding = ensemble_models.load_val_user_item_embedding(freeze_emb=args.freeze_emb)

    # determine user/item counts and embedding dims
    item_emb_dim = saved_embedding["feat_item"].weight.shape[1]
    user_emb_dim = saved_embedding["feat_user"].weight.shape[1]

    if env is not None:
        num_users, num_items = env.mat.shape[0], env.mat.shape[1]
    else:
        # infer counts from the pretrained embedding weight sizes
        num_users = saved_embedding["feat_user"].weight.shape[0]
        num_items = saved_embedding["feat_item"].weight.shape[0]

    if args.use_pretrained_embedding:
        user_columns, action_columns, feedback_columns, have_user_embedding, have_action_embedding, have_feedback_embedding = \
            get_dataset_columns(user_emb_dim, item_emb_dim, num_users, num_items, envname=args.env)
    else:
        user_columns, action_columns, feedback_columns, have_user_embedding, have_action_embedding, have_feedback_embedding = \
            get_dataset_columns(args.embedding_dim, args.embedding_dim, num_users, num_items, envname=args.env)

    args.action_shape = action_columns[0].vocabulary_size
    args.state_dim = action_columns[0].embedding_dim

    if args.use_userEmbedding:
        args.state_dim = action_columns[0].embedding_dim + saved_embedding.feat_user.weight.shape[1]

    train_max = None
    train_min = None
    test_max = None
    test_min = None
    if train_envs is not None:
        if use_buffer_in_train:
            train_max = buffer.rew.max()
            train_min = buffer.rew.min()
        else:
            train_max = train_envs.get_env_attr("MAX_R")[0] - train_envs.get_env_attr("MIN_R")[0]
            train_min = 0

    if test_envs_dict is not None:
        test_max = test_envs_dict['FB'].get_env_attr("mat")[0].max()
        test_min = test_envs_dict['FB'].get_env_attr("mat")[0].min()

    # build the same set of trackers as setup_state_tracker
    if args.which_tracker.lower() == "caser":
        assert args.window_size >= max(args.filter_sizes)
        state_tracker = StateTracker_Caser(user_columns, action_columns, feedback_columns, args.state_dim,
                                           train_max, train_min, test_max, test_min, reward_handle=args.reward_handle,
                                           saved_embedding=saved_embedding,
                                           device=args.device,
                                           window_size=args.window_size,
                                           filter_sizes=args.filter_sizes, num_filters=args.num_filters,
                                           dropout_rate=args.dropout_rate).to(args.device)
    elif args.which_tracker.lower() == "gru":
        state_tracker = StateTracker_GRU(user_columns, action_columns, feedback_columns, args.state_dim,
                                         train_max, train_min, test_max, test_min, reward_handle=args.reward_handle,
                                         saved_embedding=saved_embedding,
                                         device=args.device,
                                         window_size=args.window_size).to(args.device)
    elif args.which_tracker.lower() == "sasrec":
        state_tracker = StateTracker_SASRec(user_columns, action_columns, feedback_columns, args.state_dim,
                                            train_max, train_min, test_max, test_min, reward_handle=args.reward_handle,
                                            saved_embedding=saved_embedding,
                                            device=args.device, window_size=args.window_size,
                                            dropout_rate=args.dropout_rate, num_heads=args.num_heads).to(args.device)
    elif args.which_tracker.lower() == "nextitnet":
        state_tracker = StateTracker_NextItNet(user_columns, action_columns, feedback_columns, args.state_dim,
                                               train_max, train_min, test_max, test_min,
                                               reward_handle=args.reward_handle, saved_embedding=saved_embedding,
                                               device=args.device, window_size=args.window_size,
                                               dilations=args.dilations).to(args.device)
    elif args.which_tracker.lower() == "avg":
        assert args.use_pretrained_embedding
        state_tracker = StateTrackerAvg(user_columns, action_columns, feedback_columns, args.state_dim,
                                        train_max, train_min, test_max, test_min, reward_handle=args.reward_handle,
                                        saved_embedding=saved_embedding,
                                        device=args.device, window_size=args.window_size,
                                        use_userEmbedding=args.use_userEmbedding).to(args.device)
    else:
        return None

    state_tracker.set_need_normalization(args.need_state_norm)
    args.state_dim = state_tracker.final_dim

    return state_tracker


def learn_policy(args, env, dataset, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path, trainer="onpolicy"):
    # Evaluation
    df_val, df_user_val, df_item_val, list_feat = dataset.get_val_data()
    item_feat_domination = dataset.get_domination()
    item_similarity = dataset.get_item_similarity()
    item_popularity = dataset.get_item_popularity()

    if args.need_transform:
        assert len(item_similarity) > max(env.lbe_item.classes_)  # computing similarity requires
    item_popularity[item_popularity == 0] = min(item_popularity[item_popularity > 0])  # addressing novelty==inf problem

    metrics = ['len_tra', 'R_tra', 'ctr', 'CV', 'CV_turn', 'ifeat_', 'Diversity', 'Novelty']

    policy.callbacks = [
        Evaluator_Feat(test_collector_set, df_item_val, args.need_transform, item_feat_domination,
                       lbe_item=env.lbe_item if args.need_transform else None, top_rate=args.top_rate,
                       draw_bar=args.draw_bar),
        Evaluator_Coverage_Count(test_collector_set, df_item_val, args.need_transform),
        Evaluator_User_Experience(test_collector_set, df_item_val, item_similarity, item_popularity,
                                  args.need_transform, lbe_item=env.lbe_item if args.need_transform else None),
        LoggerEval_Policy(args.force_length, metrics)]
    model_save_path = os.path.join(MODEL_SAVE_PATH, "{}_{}.pt".format(args.model_name, args.message))

    if trainer == "offline":
        buffer = train_collector
        train_collector = None
        assert isinstance(buffer, ReplayBuffer)

        result = offline_trainer(
            policy,
            buffer,
            test_collector_set,
            args.epoch,
            args.step_per_epoch,
            args.test_num,
            args.batch_size,
            # save_best_fn=save_best_fn,
            # stop_fn=stop_fn,
            # logger=logger1,
            save_model_fn=functools.partial(save_model_fn,
                                            model_save_path=model_save_path,
                                            state_tracker=state_tracker,
                                            optim=optim,
                                            is_save=args.is_save)
        )

    elif trainer == "onpolicy":
        result = onpolicy_trainer(
            policy,
            train_collector,
            test_collector_set,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            episode_per_collect=args.episode_per_collect,
            # stop_fn=stop_fn,
            # save_best_fn=save_best_fn,
            # logger=logger1,
            save_model_fn=functools.partial(save_model_fn,
                                            model_save_path=model_save_path,
                                            state_tracker=state_tracker,
                                            optim=optim,
                                            is_save=args.is_save)
        )

    elif trainer == "offpolicy":
        collect_kwargs = _build_offpolicy_collect_kwargs(args)
        logger.info(
            "Use offpolicy collect mode=%s, step_per_collect=%s, episode_per_collect=%s",
            args.offpolicy_collect_mode,
            collect_kwargs["step_per_collect"],
            collect_kwargs["episode_per_collect"],
        )
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector_set,
            args.epoch,
            args.step_per_epoch,
            collect_kwargs["step_per_collect"],
            args.test_num,
            args.batch_size,
            update_per_step=args.update_per_step,
            episode_per_collect=collect_kwargs["episode_per_collect"],
            # stop_fn=stop_fn,	
            # save_best_fn=save_best_fn,	
            # logger=logger1,	
            save_model_fn=functools.partial(save_model_fn,
                                            model_save_path=model_save_path,
                                            state_tracker=state_tracker,
                                            optim=optim,
                                            is_save=args.is_save)
        )

    else:
        raise ValueError("Unexpected Trainer")

    print(__file__)
    pprint.pprint(result)
    logger.info(result)
    _log_final_result_to_wandb(result)
