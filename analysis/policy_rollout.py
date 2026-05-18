import glob
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from gymnasium.spaces import Discrete

from analysis.common import CoreAssets
from examples.policy.policy_utils import prepare_test_envs, setup_state_tracker
from src.core.collector.collector import Collector
from src.core.policy.RecPolicy import RecPolicy
from src.core.util.data import get_true_env
from src.tianshou.tianshou.data import Batch, VectorReplayBuffer, to_numpy
from src.tianshou.tianshou.policy import A2CPolicy
from src.tianshou.tianshou.utils.net.common import ActorCritic, Net
from src.tianshou.tianshou.utils.net.discrete import Actor, Critic


def build_dorl_policy_args(config: Dict[str, Any], assets: CoreAssets) -> Namespace:
    args = Namespace(**vars(assets.args))
    defaults = {
        "model_name": "DORL",
        "vf_coef": 0.5,
        "ent_coef": 0.0,
        "max_grad_norm": None,
        "gae_lambda": 1.0,
        "rew_norm": False,
        "reward_handle": "cat",
        "which_tracker": "avg",
        "embedding_dim": 32,
        "window_size": 3,
        "filter_sizes": [2, 3, 4],
        "num_filters": 16,
        "dropout_rate": 0.1,
        "num_heads": 1,
        "dilations": "[1, 2, 1, 2, 1, 2]",
        "lr": 1e-3,
        "hidden_sizes": [64, 64],
        "gamma": 0.9,
        "remove_recommended_ids": False,
        "use_userEmbedding": False,
        "use_pretrained_embedding": True,
        "need_state_norm": False,
        "freeze_emb": False,
        "force_length": 10,
        "exploration_noise": False,
        "test_num": int(config.get("policy_rollout_env_num", 32)),
        "cpu": bool(config.get("cpu", True)),
        "cuda": int(config.get("cuda", 0)),
        "random_init": bool(config.get("random_init", True)),
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    for key in list(defaults.keys()):
        policy_key = f"policy_{key}"
        if policy_key in config:
            setattr(args, key, config[policy_key])

    if not getattr(args, "cpu", False):
        args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    else:
        args.device = "cpu"
    args.test_num = int(config.get("policy_rollout_env_num", getattr(args, "test_num", 32)))
    args.remove_recommended_ids = bool(config.get("policy_remove_recommended_ids", getattr(args, "remove_recommended_ids", False)))
    return args


def resolve_policy_checkpoint(config: Dict[str, Any]) -> Optional[str]:
    explicit_path = str(config.get("policy_checkpoint_path", "")).strip()
    if explicit_path:
        return explicit_path

    search_dir = str(config.get("policy_checkpoint_dir", os.path.join("saved_models", str(config["env"]), "DORL")))
    if not os.path.isdir(search_dir):
        return None
    preferred_epoch = config.get("training_best_epoch")
    if preferred_epoch is not None:
        preferred = sorted(
            glob.glob(os.path.join(search_dir, f"*-e{int(preferred_epoch)}.pt"))
        ) + sorted(
            glob.glob(os.path.join(search_dir, f"*-e{int(preferred_epoch)}.pth"))
        )
        if preferred:
            return preferred[-1]
    candidates = sorted(glob.glob(os.path.join(search_dir, "*.pt"))) + sorted(glob.glob(os.path.join(search_dir, "*.pth")))
    return candidates[-1] if candidates else None


def build_dorl_policy_for_rollout(
    assets: CoreAssets,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    args = build_dorl_policy_args(config, assets)
    env_template, _, kwargs_um = get_true_env(args)
    test_envs_dict = prepare_test_envs(args, env_template, kwargs_um)
    rollout_envs = test_envs_dict["FB"]

    state_tracker = setup_state_tracker(args, assets.ensemble, env_template, train_envs=None, test_envs_dict=test_envs_dict)
    net = Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)
    optim_rl = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    base_policy = A2CPolicy(
        actor,
        critic,
        [optim_rl, optim_state],
        torch.distributions.Categorical,
        state_tracker=state_tracker,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        reward_normalization=args.rew_norm,
        action_space=Discrete(args.action_shape),
        action_bound_method="",
        action_scaling=False,
        deterministic_eval=bool(config.get("policy_deterministic_eval", False)),
    )
    rec_policy = RecPolicy(args, base_policy, state_tracker)

    checkpoint_path = resolve_policy_checkpoint(config)
    checkpoint_loaded = False
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        if "policy" not in checkpoint:
            raise KeyError(f"Checkpoint at {checkpoint_path} does not contain 'policy'")
        rec_policy.policy.load_state_dict(checkpoint["policy"])
        if "state_tracker" in checkpoint:
            state_tracker.load_state_dict(checkpoint["state_tracker"])
        checkpoint_loaded = True
    elif not bool(config.get("policy_allow_random_init_for_rollout", False)):
        raise FileNotFoundError(
            "No DORL policy checkpoint found. Please set policy_checkpoint_path, "
            "or set policy_allow_random_init_for_rollout=true for a smoke run."
        )

    rec_policy.policy.eval()
    state_tracker.eval()
    return {
        "args": args,
        "policy": rec_policy,
        "state_tracker": state_tracker,
        "rollout_envs": rollout_envs,
        "checkpoint_path": checkpoint_path,
        "checkpoint_loaded": checkpoint_loaded,
    }


def collect_policy_rollout_payload(
    assets: CoreAssets,
    config: Dict[str, Any],
    oracle_norm_mat: np.ndarray,
    oracle_raw_mat: np.ndarray,
    oracle_cost_mat: np.ndarray,
) -> Dict[str, Any]:
    build_payload = build_dorl_policy_for_rollout(assets, config)
    rec_policy = build_payload["policy"]
    rollout_envs = build_payload["rollout_envs"]
    env_num = len(rollout_envs)
    max_states = int(config.get("policy_rollout_n_steps", 1024))
    buffer_size = max(int(config.get("policy_rollout_buffer_size", max_states + env_num + 1)), env_num * 4)

    collector = Collector(
        rec_policy,
        rollout_envs,
        VectorReplayBuffer(buffer_size, env_num),
        exploration_noise=False,
        remove_recommended_ids=bool(config.get("policy_remove_recommended_ids", rec_policy.remove_recommended_ids)),
        force_length=0,
    )

    ready_env_ids = np.arange(env_num)
    turn_by_local = np.zeros(env_num, dtype=np.int32)
    episode_id_by_local = np.arange(env_num, dtype=np.int64)
    next_episode_id = int(env_num)
    step_count = 0

    trace_rows: List[Dict[str, Any]] = []
    prob_rows: List[np.ndarray] = []
    mask_rows: List[np.ndarray] = []

    while step_count < max_states:
        last_state = collector.data.policy.pop("hidden_state", None)
        indices = collector.buffer.last_index if len(collector.buffer) > 0 else None

        with torch.no_grad():
            result = rec_policy(
                collector.data,
                collector.buffer,
                indices=indices,
                is_obs=True,
                state=last_state,
                remove_recommended_ids=collector.remove_recommended_ids,
                is_train=False,
                use_batch_in_statetracker=True,
            )

        policy = result.get("policy", Batch())
        state = result.get("state", None)
        if state is not None:
            policy.hidden_state = state
        act = to_numpy(result.act).astype(np.int64)
        collector.data.update(policy=policy, act=act)

        current_obs = to_numpy_array(collector.data.obs).copy()
        current_mask = to_numpy_array(
            getattr(collector.data, "mask", np.ones((len(current_obs), len(assets.item_raw_ids)), dtype=bool))
        ).astype(bool)
        probs = result.dist.probs.detach().cpu().numpy().astype(np.float32)
        probs = normalize_row_probs(probs, current_mask)
        action_remap = np.asarray(rec_policy.map_action(collector.data), dtype=np.int64)

        user_indices = current_obs[:, 0].astype(np.int64)
        prev_action_indices = current_obs[:, 1].astype(np.int64)
        oracle_norm_rows = oracle_norm_mat[user_indices]
        oracle_raw_rows = oracle_raw_mat[user_indices]
        oracle_cost_rows = oracle_cost_mat[user_indices]

        sampled_prob = probs[np.arange(len(action_remap)), action_remap]
        sampled_norm_reward = oracle_norm_rows[np.arange(len(action_remap)), action_remap]
        sampled_raw_reward = oracle_raw_rows[np.arange(len(action_remap)), action_remap]
        sampled_cost_reward = oracle_cost_rows[np.arange(len(action_remap)), action_remap]
        best_norm_action, best_norm_reward = masked_row_argmax(oracle_norm_rows, current_mask)
        best_raw_action, best_raw_reward = masked_row_argmax(oracle_raw_rows, current_mask)
        best_cost_action, best_cost_reward = masked_row_argmax(oracle_cost_rows, current_mask)
        expected_norm_reward = np.sum(probs * oracle_norm_rows, axis=1)
        expected_raw_reward = np.sum(probs * oracle_raw_rows, axis=1)
        expected_cost_reward = np.sum(probs * oracle_cost_rows, axis=1)
        entropy = -np.sum(np.where(probs > 0, probs * np.log(np.clip(probs, 1e-12, 1.0)), 0.0), axis=1)
        policy_top1 = probs.argmax(axis=1)
        policy_top1_prob = probs[np.arange(len(policy_top1)), policy_top1]

        for local_idx in range(len(action_remap)):
            trace_rows.append(
                {
                    "state_id": int(step_count + local_idx),
                    "episode_id": int(episode_id_by_local[local_idx]),
                    "turn": int(turn_by_local[local_idx]),
                    "user_index": int(user_indices[local_idx]),
                    "user_id": int(assets.user_raw_ids[user_indices[local_idx]]),
                    "prev_action_index": int(prev_action_indices[local_idx]),
                    "prev_action_id": int(assets.item_raw_ids[prev_action_indices[local_idx]]) if 0 <= prev_action_indices[local_idx] < len(assets.item_raw_ids) else -1,
                    "sampled_action_index": int(action_remap[local_idx]),
                    "sampled_action_id": int(assets.item_raw_ids[action_remap[local_idx]]),
                    "sampled_action_prob": float(sampled_prob[local_idx]),
                    "sampled_action_oracle_norm": float(sampled_norm_reward[local_idx]),
                    "sampled_action_oracle_raw": float(sampled_raw_reward[local_idx]),
                    "sampled_action_oracle_cost": float(sampled_cost_reward[local_idx]),
                    "policy_top1_action_index": int(policy_top1[local_idx]),
                    "policy_top1_action_id": int(assets.item_raw_ids[policy_top1[local_idx]]),
                    "policy_top1_prob": float(policy_top1_prob[local_idx]),
                    "best_available_action_norm": int(best_norm_action[local_idx]),
                    "best_available_action_raw": int(best_raw_action[local_idx]),
                    "best_available_action_cost": int(best_cost_action[local_idx]),
                    "best_available_oracle_norm": float(best_norm_reward[local_idx]),
                    "best_available_oracle_raw": float(best_raw_reward[local_idx]),
                    "best_available_oracle_cost": float(best_cost_reward[local_idx]),
                    "expected_oracle_norm": float(expected_norm_reward[local_idx]),
                    "expected_oracle_raw": float(expected_raw_reward[local_idx]),
                    "expected_oracle_cost": float(expected_cost_reward[local_idx]),
                    "action_regret_norm": float(best_norm_reward[local_idx] - sampled_norm_reward[local_idx]),
                    "action_regret_raw": float(best_raw_reward[local_idx] - sampled_raw_reward[local_idx]),
                    "action_regret_cost": float(best_cost_reward[local_idx] - sampled_cost_reward[local_idx]),
                    "policy_entropy": float(entropy[local_idx]),
                    "available_action_count": int(current_mask[local_idx].sum()),
                    "sampled_equals_top1": bool(action_remap[local_idx] == policy_top1[local_idx]),
                }
            )

        prob_rows.append(probs)
        mask_rows.append(current_mask.astype(np.uint8))

        obs_next, rew, terminated, truncated, info = rollout_envs.step(action_remap, ready_env_ids)
        done = np.logical_or(terminated, truncated)
        collector.data.update(
            obs_next=obs_next,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
            done=done,
            info=info,
        )
        ptr, ep_rew, ep_len, ep_idx = collector.buffer.add(collector.data, buffer_ids=ready_env_ids)
        collector.data.is_start = np.zeros(len(collector.data), dtype=bool)
        turn_by_local += 1

        if np.any(done):
            env_ind_local = np.where(done)[0]
            env_ind_global = ready_env_ids[env_ind_local]
            collector._reset_env_with_ids(env_ind_local, env_ind_global, None)
            collector.data.obs_next[env_ind_local] = collector.data.obs[env_ind_local]
            for i in env_ind_local:
                collector._reset_state(i)
                turn_by_local[i] = 0
                episode_id_by_local[i] = next_episode_id
                next_episode_id += 1

        collector.data.obs = collector.data.obs_next
        step_count += len(action_remap)

    trace_df = pd.DataFrame(trace_rows).iloc[:max_states].copy()
    policy_prob_mat = np.concatenate(prob_rows, axis=0)[:max_states]
    action_mask_mat = np.concatenate(mask_rows, axis=0)[:max_states].astype(bool)
    rollout_user_indices = trace_df["user_index"].to_numpy(dtype=np.int64)
    unique_user_indices, inverse = np.unique(rollout_user_indices, return_inverse=True)
    unique_user_raw_ids = assets.user_raw_ids[unique_user_indices]

    return {
        "trace_df": trace_df,
        "policy_prob_mat": policy_prob_mat.astype(np.float32),
        "action_mask_mat": action_mask_mat,
        "rollout_user_indices": rollout_user_indices,
        "rollout_user_raw_ids": trace_df["user_id"].to_numpy(dtype=np.int64),
        "unique_user_indices": unique_user_indices.astype(np.int64),
        "unique_user_raw_ids": np.asarray(unique_user_raw_ids, dtype=np.int64),
        "user_inverse": inverse.astype(np.int64),
        "checkpoint_path": build_payload["checkpoint_path"],
        "checkpoint_loaded": build_payload["checkpoint_loaded"],
        "env_num": int(env_num),
        "n_states": int(len(trace_df)),
    }


def masked_row_argmax(values: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    masked_values = np.where(mask, values, -np.inf)
    argmax = np.argmax(masked_values, axis=1)
    best = masked_values[np.arange(len(argmax)), argmax]
    return argmax.astype(np.int64), best.astype(np.float32)


def normalize_row_probs(probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    work = np.asarray(probs, dtype=np.float32) * mask.astype(np.float32)
    denom = work.sum(axis=1, keepdims=True)
    valid = denom.squeeze(-1) > 0
    if np.any(valid):
        work[valid] = work[valid] / denom[valid]
    if np.any(~valid):
        invalid_idx = np.where(~valid)[0]
        fallback = mask[invalid_idx].astype(np.float32)
        fallback_denom = fallback.sum(axis=1, keepdims=True)
        fallback_valid = fallback_denom.squeeze(-1) > 0
        if np.any(fallback_valid):
            fallback[fallback_valid] = fallback[fallback_valid] / fallback_denom[fallback_valid]
        work[invalid_idx] = fallback
    return work.astype(np.float32)


def to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def summarize_policy_trace(trace_df: pd.DataFrame) -> Dict[str, Any]:
    if trace_df.empty:
        return {"n_states": 0}
    return {
        "n_states": int(len(trace_df)),
        "n_episodes_observed": int(trace_df["episode_id"].nunique()),
        "mean_action_regret_cost": float(trace_df["action_regret_cost"].mean()),
        "mean_action_regret_norm": float(trace_df["action_regret_norm"].mean()),
        "mean_action_regret_raw": float(trace_df["action_regret_raw"].mean()),
        "mean_sampled_action_prob": float(trace_df["sampled_action_prob"].mean()),
        "mean_policy_entropy": float(trace_df["policy_entropy"].mean()),
        "sampled_equals_top1_rate": float(trace_df["sampled_equals_top1"].mean()),
    }
