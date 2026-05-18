import argparse
import os
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

sys.path.extend([".", "./src", "./examples/policy", "./src/DeepCTR-Torch", "./src/tianshou"])

import numpy as np
import pandas as pd

from analysis.common import (
    RUNTIME_PREFIX,
    build_manifest,
    bucketize_by_edges,
    choose_matrix_by_name,
    load_config,
    load_core_assets,
    matrix_slice,
    safe_average_precision,
    safe_roc_auc,
    safe_spearman,
    sanitize_column_token,
    save_json,
)
from analysis.compute_flip_risk import (
    activate_region_columns,
    build_primary_slice_mask,
    build_sample_metrics_frame,
    compute_bucket_statistics,
    compute_correlation_summary,
    compute_margin_region_payload,
    compute_margin_summary,
    compute_ranking_metric_matrices,
    compute_support_stratified_summary,
    filter_sample_df,
)
from analysis.compute_support_score import compute_support_matrices
from analysis.compute_uncertainty import compute_uncertainty_payload
from analysis.generate_motivation_report import generate_implementation_log, generate_report
from analysis.oracle_cost_analysis import compute_cost_analysis
from analysis.policy_rollout import collect_policy_rollout_payload, summarize_policy_trace
from analysis.plot_motivation_results import generate_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run KuaiRec motivation experiments")
    parser.add_argument("--config", type=str, default="configs/motivation_experiments.yaml")
    parser.add_argument("--override", action="append", default=[], help="Override config entries with key=value")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    config = load_config(cli_args.config, cli_args.override)
    config = apply_smoke_defaults(config)
    run_commands = build_run_commands(
        config=config,
        config_path=cli_args.config,
        overrides=cli_args.override,
    )

    ablation_df = pd.DataFrame()
    ablation_run_dir: Optional[str] = None
    if bool(config.get("run_subset_ablation", False)) and not bool(config.get("smoke_test", False)):
        print("[Motivation] Running subset ablation...")
        ablation_bundle = run_subset_ablation(config)
        ablation_df = ablation_bundle["ablation_df"]
        ablation_run_dir = ablation_bundle["output_dir"]

    print("[Motivation] Running main experiment...")
    bundle = execute_experiment(config)
    bundle["run_commands"] = run_commands
    baseline_df = build_baseline_comparison(
        config=config,
        current_correlation_df=bundle["correlation_df"],
        current_support_df=bundle["support_df"],
        current_cost_df=bundle["cost_df"],
    )
    save_experiment_outputs(
        bundle=bundle,
        config=config,
        ablation_df=ablation_df,
        baseline_df=baseline_df,
        ablation_run_dir=ablation_run_dir,
    )
    print(f"[Motivation] Finished. Results saved to {bundle['output_dir']}")


def execute_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    start_time = time.time()
    data_source = str(config.get("analysis_data_source", "grid"))
    assets_config = dict(config)
    if data_source == "policy_rollout":
        assets_config["max_users"] = 0
        assets_config["subset_items"] = 0
    assets = load_core_assets(assets_config)

    policy_trace_df = pd.DataFrame()
    policy_trace_summary: Dict[str, Any] = {}
    policy_rollout_bundle: Optional[Dict[str, Any]] = None
    if data_source == "policy_rollout":
        policy_rollout_bundle = collect_policy_rollout_payload(
            assets=assets,
            config=config,
            oracle_norm_mat=assets.oracle_norm_mat,
            oracle_raw_mat=assets.oracle_raw_mat,
            oracle_cost_mat=choose_matrix_by_name(
                str(config.get("decision_cost_reward", "watch_ratio")),
                assets.oracle_norm_mat,
                assets.oracle_raw_mat,
            ),
        )
        policy_trace_df = policy_rollout_bundle["trace_df"]
        policy_trace_summary = summarize_policy_trace(policy_trace_df)
        policy_trace_summary["checkpoint_path"] = policy_rollout_bundle["checkpoint_path"]
        policy_trace_summary["checkpoint_loaded"] = bool(policy_rollout_bundle["checkpoint_loaded"])

        pred_score_mat = policy_rollout_bundle["policy_prob_mat"]
        oracle_norm_mat = np.asarray(assets.oracle_norm_mat[policy_rollout_bundle["rollout_user_indices"]], dtype=np.float32)
        oracle_raw_mat = np.asarray(assets.oracle_raw_mat[policy_rollout_bundle["rollout_user_indices"]], dtype=np.float32)
        item_raw_ids = assets.item_raw_ids
        row_user_raw_ids = policy_rollout_bundle["rollout_user_raw_ids"]

        uncertainty_payload_unique = compute_uncertainty_payload(
            assets,
            config,
            user_raw_ids=policy_rollout_bundle["unique_user_raw_ids"],
            item_raw_ids=item_raw_ids,
        )
        uncertainty_payload = {
            "primary_source": uncertainty_payload_unique["primary_source"],
            "meta": uncertainty_payload_unique["meta"],
            "matrices": {
                source: np.asarray(mat[policy_rollout_bundle["user_inverse"]], dtype=np.float32)
                for source, mat in uncertainty_payload_unique["matrices"].items()
            },
        }
        uncertainty_payload["primary_matrix"] = uncertainty_payload["matrices"][uncertainty_payload["primary_source"]]

        support_payload_unique = compute_support_matrices(
            assets,
            config,
            user_raw_ids=policy_rollout_bundle["unique_user_raw_ids"],
            item_raw_ids=item_raw_ids,
        )
        support_payload = dict(support_payload_unique)
        support_payload["support_score_mat"] = np.asarray(
            support_payload_unique["support_score_mat"][policy_rollout_bundle["user_inverse"]],
            dtype=np.float32,
        )
        support_payload["support_region_codes"] = np.asarray(
            support_payload_unique["support_region_codes"][policy_rollout_bundle["user_inverse"]],
            dtype=np.int8,
        )
        support_payload["support_region_codes_by_mode"] = {
            mode: np.asarray(codes[policy_rollout_bundle["user_inverse"]], dtype=np.int8)
            for mode, codes in support_payload_unique["support_region_codes_by_mode"].items()
        }
        row_metadata = {
            "state_id": policy_trace_df["state_id"].to_numpy(dtype=np.int64),
            "episode_id": policy_trace_df["episode_id"].to_numpy(dtype=np.int64),
            "turn": policy_trace_df["turn"].to_numpy(dtype=np.int32),
            "sampled_action_index": policy_trace_df["sampled_action_index"].to_numpy(dtype=np.int64),
            "sampled_action_id": policy_trace_df["sampled_action_id"].to_numpy(dtype=np.int64),
            "sampled_action_prob": policy_trace_df["sampled_action_prob"].to_numpy(dtype=np.float32),
            "action_regret_cost": policy_trace_df["action_regret_cost"].to_numpy(dtype=np.float32),
            "action_regret_norm": policy_trace_df["action_regret_norm"].to_numpy(dtype=np.float32),
            "action_regret_raw": policy_trace_df["action_regret_raw"].to_numpy(dtype=np.float32),
            "policy_entropy": policy_trace_df["policy_entropy"].to_numpy(dtype=np.float32),
            "sampled_equals_top1": policy_trace_df["sampled_equals_top1"].astype(np.uint8).to_numpy(),
        }
    else:
        pred_score_mat = matrix_slice(assets.pred_mat, assets)
        oracle_norm_mat = matrix_slice(assets.oracle_norm_mat, assets)
        oracle_raw_mat = matrix_slice(assets.oracle_raw_mat, assets)
        item_raw_ids = assets.item_raw_ids
        row_user_raw_ids = assets.user_raw_ids
        uncertainty_payload = compute_uncertainty_payload(assets, config)
        support_payload = compute_support_matrices(assets, config)
        row_metadata = None

    error_oracle_mat = choose_matrix_by_name(
        str(config.get("oracle_reward_for_error", "watch_ratio_normed")),
        oracle_norm_mat,
        oracle_raw_mat,
    )
    ranking_oracle_mat = choose_matrix_by_name(
        str(config.get("oracle_reward_for_ranking", "watch_ratio_normed")),
        oracle_norm_mat,
        oracle_raw_mat,
    )
    decision_cost_reward_mat = choose_matrix_by_name(
        str(config.get("decision_cost_reward", "watch_ratio")),
        oracle_norm_mat,
        oracle_raw_mat,
    )

    ranking_metrics = compute_ranking_metric_matrices(
        pred_score_mat=pred_score_mat,
        oracle_rank_score_mat=ranking_oracle_mat,
        item_raw_ids=item_raw_ids,
        topk=int(config.get("topk", 20)),
        shortlist_topl=int(config.get("shortlist_topl", 100)),
    )
    margin_payload = compute_margin_region_payload(ranking_metrics["margin_mat"], config)
    evaluation_slice_name, primary_slice_mask = build_primary_slice_mask(ranking_metrics, config)

    sample_df = build_sample_metrics_frame(
        user_raw_ids=row_user_raw_ids,
        item_raw_ids=item_raw_ids,
        pred_score_mat=pred_score_mat,
        uncertainty_mats=uncertainty_payload["matrices"],
        primary_uncertainty_source=uncertainty_payload["primary_source"],
        oracle_norm_mat=oracle_norm_mat,
        oracle_raw_mat=oracle_raw_mat,
        error_oracle_mat=error_oracle_mat,
        support_score_mat=support_payload["support_score_mat"],
        support_region_codes_by_mode=support_payload["support_region_codes_by_mode"],
        margin_region_codes_by_mode=margin_payload["codes_by_mode"],
        primary_slice_mask=primary_slice_mask,
        ranking_metrics=ranking_metrics,
        row_metadata=row_metadata,
    )
    if data_source == "policy_rollout":
        sampled_action_id = sample_df["sampled_action_id"].to_numpy(dtype=np.int64)
        item_id = sample_df["item_id"].to_numpy(dtype=np.int64)
        sample_df["is_sampled_action"] = (sampled_action_id == item_id).astype(np.uint8)

    sample_df = activate_region_columns(
        sample_df,
        support_mode=str(config.get("support_quantile_mode", "global")),
        margin_mode=str(config.get("margin_quantile_mode", "global")),
    )
    primary_df = filter_sample_df(sample_df)
    analysis_slices = build_analysis_slices(primary_df, str(config.get("conditional_support_slice", "small_margin")))

    correlation_df, threshold_stats = compute_correlation_summary(primary_df, config)
    bucket_df, uncertainty_edges = compute_bucket_statistics(primary_df, config)
    support_df = compute_support_stratified_summary(primary_df, config, analysis_slices)
    margin_df = compute_margin_summary(primary_df, config, uncertainty_edges)
    uncertainty_compare_df = compute_uncertainty_source_comparison(primary_df, config)
    chosen_action_summary_df = pd.DataFrame()
    chosen_action_corr_df = pd.DataFrame()
    chosen_action_support_df = pd.DataFrame()
    chosen_action_margin_df = pd.DataFrame()
    chosen_action_overview: Dict[str, Any] = {}
    if data_source == "policy_rollout":
        (
            chosen_action_summary_df,
            chosen_action_corr_df,
            chosen_action_support_df,
            chosen_action_margin_df,
            chosen_action_overview,
        ) = compute_chosen_action_outputs(
            sample_df=sample_df,
            uncertainty_edges=uncertainty_edges,
        )

    rollout_return_payload = None
    if bool(config.get("run_rollout_return", False)):
        rollout_return_payload = {
            "enabled": True,
            "oracle_reward_mat": decision_cost_reward_mat,
            "list_feat_small": [assets.env.list_feat_small[int(i)] for i in assets.item_indices],
            "leave_threshold": int(assets.args.leave_threshold),
            "num_leave_compute": int(assets.args.num_leave_compute),
            "max_turn": int(assets.args.max_turn),
        }

    cost_primary_risk = str(config.get("cost_primary_risk", "local_pairwise_flip"))
    oracle_local_flip_risk = choose_risk_matrix(cost_primary_risk, ranking_metrics)
    cost_df = compute_cost_analysis(
        pred_score_mat=pred_score_mat,
        decision_cost_reward_mat=decision_cost_reward_mat,
        risk_signal_mats={
            "raw_uncertainty": uncertainty_payload["primary_matrix"],
            "oracle_pointwise_error": np.abs(pred_score_mat - error_oracle_mat).astype(np.float32),
            "oracle_local_flip_risk": oracle_local_flip_risk.astype(np.float32),
            "oracle_topk_flip": ranking_metrics["topk_flip_mat"].astype(np.float32),
        },
        item_raw_ids=item_raw_ids,
        config=config,
        rollout_payload=rollout_return_payload,
    )

    return {
        "assets": assets,
        "output_dir": assets.output_paths.root,
        "pred_score_mat": pred_score_mat,
        "oracle_norm_mat": oracle_norm_mat,
        "oracle_raw_mat": oracle_raw_mat,
        "error_oracle_mat": error_oracle_mat,
        "decision_cost_reward_mat": decision_cost_reward_mat,
        "uncertainty_payload": uncertainty_payload,
        "support_payload": support_payload,
        "margin_payload": margin_payload,
        "ranking_metrics": ranking_metrics,
        "sample_df": sample_df,
        "primary_df": primary_df,
        "correlation_df": correlation_df,
        "bucket_df": bucket_df,
        "support_df": support_df,
        "margin_df": margin_df,
        "cost_df": cost_df,
        "uncertainty_compare_df": uncertainty_compare_df,
        "chosen_action_summary_df": chosen_action_summary_df,
        "chosen_action_corr_df": chosen_action_corr_df,
        "chosen_action_support_df": chosen_action_support_df,
        "chosen_action_margin_df": chosen_action_margin_df,
        "chosen_action_overview": chosen_action_overview,
        "threshold_stats": threshold_stats,
        "uncertainty_edges": uncertainty_edges,
        "evaluation_slice_name": evaluation_slice_name,
        "cost_primary_risk": cost_primary_risk,
        "data_source": data_source,
        "policy_trace_df": policy_trace_df,
        "policy_trace_summary": policy_trace_summary,
        "policy_rollout_bundle": policy_rollout_bundle,
        "elapsed_seconds": float(time.time() - start_time),
    }


def save_experiment_outputs(
    bundle: Dict[str, Any],
    config: Dict[str, Any],
    ablation_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    ablation_run_dir: Optional[str],
) -> None:
    assets = bundle["assets"]
    output_dir = assets.output_paths.root

    sample_metrics_path = os.path.join(output_dir, "sample_metrics.csv.gz")
    correlation_path = os.path.join(output_dir, "correlation_summary.csv")
    support_path = os.path.join(output_dir, "support_stratified_summary.csv")
    margin_path = os.path.join(output_dir, "margin_summary.csv")
    cost_path = os.path.join(output_dir, "cost_analysis_summary.csv")
    bucket_path = os.path.join(output_dir, "bucket_summary.csv")
    uncertainty_compare_path = os.path.join(output_dir, "uncertainty_source_comparison.csv")
    baseline_compare_path = os.path.join(output_dir, "baseline_comparison.csv")
    ablation_path = os.path.join(output_dir, "subset_ablation_summary.csv")
    trace_path = os.path.join(output_dir, "policy_rollout_trace.csv.gz")
    trace_summary_path = os.path.join(output_dir, "policy_rollout_summary.json")
    chosen_action_summary_path = os.path.join(output_dir, "chosen_action_summary.csv")
    chosen_action_corr_path = os.path.join(output_dir, "chosen_action_correlation_summary.csv")
    chosen_action_support_path = os.path.join(output_dir, "chosen_action_support_summary.csv")
    chosen_action_margin_path = os.path.join(output_dir, "chosen_action_margin_summary.csv")
    run_commands_path = os.path.join(output_dir, "run_commands.json")

    bundle["sample_df"].to_csv(sample_metrics_path, index=False, compression="gzip")
    bundle["correlation_df"].to_csv(correlation_path, index=False)
    bundle["support_df"].to_csv(support_path, index=False)
    bundle["margin_df"].to_csv(margin_path, index=False)
    bundle["cost_df"].to_csv(cost_path, index=False)
    bundle["bucket_df"].to_csv(bucket_path, index=False)
    bundle["uncertainty_compare_df"].to_csv(uncertainty_compare_path, index=False)
    if not bundle["chosen_action_summary_df"].empty:
        bundle["chosen_action_summary_df"].to_csv(chosen_action_summary_path, index=False)
    if not bundle["chosen_action_corr_df"].empty:
        bundle["chosen_action_corr_df"].to_csv(chosen_action_corr_path, index=False)
    if not bundle["chosen_action_support_df"].empty:
        bundle["chosen_action_support_df"].to_csv(chosen_action_support_path, index=False)
    if not bundle["chosen_action_margin_df"].empty:
        bundle["chosen_action_margin_df"].to_csv(chosen_action_margin_path, index=False)
    if not bundle["policy_trace_df"].empty:
        bundle["policy_trace_df"].to_csv(trace_path, index=False, compression="gzip")
        save_json(bundle["policy_trace_summary"], trace_summary_path)
    save_json(bundle["run_commands"], run_commands_path)
    if not baseline_df.empty:
        baseline_df.to_csv(baseline_compare_path, index=False)
    if not ablation_df.empty:
        ablation_df.to_csv(ablation_path, index=False)

    artifact_npz = os.path.join(assets.output_paths.artifacts, "motivation_artifacts.npz")
    artifact_payload = {
        "pred_score_mat": bundle["pred_score_mat"].astype(np.float32),
        "oracle_norm_mat": bundle["oracle_norm_mat"].astype(np.float32),
        "oracle_raw_mat": bundle["oracle_raw_mat"].astype(np.float32),
        "pointwise_error_mat": np.abs(bundle["pred_score_mat"] - bundle["error_oracle_mat"]).astype(np.float32),
        "support_score_mat": bundle["support_payload"]["support_score_mat"].astype(np.float32),
        "support_region_codes_active": bundle["support_payload"]["support_region_codes"].astype(np.int8),
        "margin_region_codes_active": bundle["margin_payload"]["active_codes"].astype(np.int8),
        "pred_rank_mat": bundle["ranking_metrics"]["pred_rank_mat"].astype(np.int32),
        "oracle_rank_mat": bundle["ranking_metrics"]["oracle_rank_mat"].astype(np.int32),
        "margin_mat": bundle["ranking_metrics"]["margin_mat"].astype(np.float32),
        "topk_flip_mat": bundle["ranking_metrics"]["topk_flip_mat"].astype(np.uint8),
        "global_pairwise_flip_mat": bundle["ranking_metrics"]["global_pairwise_flip_mat"].astype(np.float32),
        "local_pairwise_flip_mat": bundle["ranking_metrics"]["local_pairwise_flip_mat"].astype(np.float32),
        "boundary_crossing_score_mat": bundle["ranking_metrics"]["boundary_crossing_score_mat"].astype(np.float32),
        "shortlist_mask_mat": bundle["ranking_metrics"]["shortlist_mask_mat"].astype(np.uint8),
    }
    if bundle["policy_rollout_bundle"] is not None:
        artifact_payload["policy_action_mask_mat"] = bundle["policy_rollout_bundle"]["action_mask_mat"].astype(np.uint8)
    for source, mat in bundle["uncertainty_payload"]["matrices"].items():
        artifact_payload[f"uncertainty_{sanitize_column_token(source)}"] = np.asarray(mat, dtype=np.float32)
    for mode, codes in bundle["support_payload"]["support_region_codes_by_mode"].items():
        artifact_payload[f"support_region_codes_{mode}"] = codes.astype(np.int8)
    for mode, codes in bundle["margin_payload"]["codes_by_mode"].items():
        artifact_payload[f"margin_region_codes_{mode}"] = codes.astype(np.int8)
    np.savez_compressed(artifact_npz, **artifact_payload)

    bucket_stats = {
        "thresholds": bundle["threshold_stats"],
        "uncertainty_buckets": bundle["bucket_df"].to_dict(orient="records"),
        "support_stats": bundle["support_payload"]["stats"],
        "margin_stats": bundle["margin_payload"]["stats_by_mode"],
        "evaluation_slice_name": bundle["evaluation_slice_name"],
        "primary_uncertainty_source": bundle["uncertainty_payload"]["primary_source"],
        "cost_primary_risk": bundle["cost_primary_risk"],
        "analysis_data_source": bundle["data_source"],
        "policy_trace_summary": bundle["policy_trace_summary"],
        "chosen_action_overview": bundle["chosen_action_overview"],
    }
    bucket_stats_path = os.path.join(output_dir, "bucket_stats.json")
    save_json(bucket_stats, bucket_stats_path)

    figure_paths = generate_all_plots(
        bucket_df=bundle["bucket_df"],
        support_df=bundle["support_df"],
        margin_df=bundle["margin_df"],
        cost_df=bundle["cost_df"],
        chosen_action_df=bundle["chosen_action_summary_df"],
        figure_dir=assets.output_paths.figures,
    )

    summary_paths = {
        "sample_metrics": sample_metrics_path,
        "correlation_summary": correlation_path,
        "bucket_summary": bucket_path,
        "bucket_stats_json": bucket_stats_path,
        "support_stratified_summary": support_path,
        "margin_summary": margin_path,
        "cost_analysis_summary": cost_path,
        "uncertainty_source_comparison": uncertainty_compare_path,
        "artifact_npz": artifact_npz,
        "run_commands": run_commands_path,
    }
    if not bundle["chosen_action_summary_df"].empty:
        summary_paths["chosen_action_summary"] = chosen_action_summary_path
    if not bundle["chosen_action_corr_df"].empty:
        summary_paths["chosen_action_correlation_summary"] = chosen_action_corr_path
    if not bundle["chosen_action_support_df"].empty:
        summary_paths["chosen_action_support_summary"] = chosen_action_support_path
    if not bundle["chosen_action_margin_df"].empty:
        summary_paths["chosen_action_margin_summary"] = chosen_action_margin_path
    if not bundle["policy_trace_df"].empty:
        summary_paths["policy_rollout_trace"] = trace_path
        summary_paths["policy_rollout_summary"] = trace_summary_path
    if not baseline_df.empty:
        summary_paths["baseline_comparison"] = baseline_compare_path
    if not ablation_df.empty:
        summary_paths["subset_ablation_summary"] = ablation_path

    report_text = generate_report(
        config=config,
        correlation_df=bundle["correlation_df"],
        bucket_df=bundle["bucket_df"],
        support_df=bundle["support_df"],
        margin_df=bundle["margin_df"],
        cost_df=bundle["cost_df"],
        figure_paths=figure_paths,
        output_dir=output_dir,
        uncertainty_compare_df=bundle["uncertainty_compare_df"],
        baseline_df=baseline_df,
        ablation_df=ablation_df,
        policy_trace_summary=bundle["policy_trace_summary"],
        chosen_action_summary_df=bundle["chosen_action_summary_df"],
        chosen_action_corr_df=bundle["chosen_action_corr_df"],
        chosen_action_support_df=bundle["chosen_action_support_df"],
        chosen_action_margin_df=bundle["chosen_action_margin_df"],
    )
    report_path = os.path.join(output_dir, "motivation_experiment_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    summary_paths["report"] = report_path

    impl_log = generate_implementation_log(
        config=config,
        output_dir=output_dir,
        figure_paths=figure_paths,
        summary_paths=summary_paths,
        primary_uncertainty_source=bundle["uncertainty_payload"]["primary_source"],
        evaluation_slice_name=bundle["evaluation_slice_name"],
        cost_primary_risk=bundle["cost_primary_risk"],
        ablation_run_dir=ablation_run_dir,
        run_commands=bundle["run_commands"],
        policy_trace_summary=bundle["policy_trace_summary"],
    )
    impl_log_path = os.path.join(output_dir, "implementation_log.md")
    with open(impl_log_path, "w", encoding="utf-8") as f:
        f.write(impl_log)

    manifest_path = os.path.join(output_dir, "run_manifest.json")
    save_json(
        build_manifest(
            config=config,
            extra={
                "uncertainty_meta": bundle["uncertainty_payload"]["meta"],
                "support_stats": bundle["support_payload"]["stats"],
                "margin_stats": bundle["margin_payload"]["stats_by_mode"],
                "elapsed_seconds": bundle["elapsed_seconds"],
                "summary_paths": summary_paths,
                "figure_paths": figure_paths,
                "evaluation_slice_name": bundle["evaluation_slice_name"],
                "cost_primary_risk": bundle["cost_primary_risk"],
                "ablation_run_dir": ablation_run_dir,
                "run_commands": bundle["run_commands"],
                "chosen_action_overview": bundle["chosen_action_overview"],
            },
        ),
        manifest_path,
    )


def build_analysis_slices(primary_df: pd.DataFrame, conditional_slice: str) -> Dict[str, pd.DataFrame]:
    slices = {"all": primary_df}
    if conditional_slice == "small_margin":
        slices["small_margin"] = primary_df[primary_df["margin_region"] == "small"].copy()
    return slices


def choose_risk_matrix(risk_name: str, ranking_metrics: Mapping[str, np.ndarray]) -> np.ndarray:
    mapping = {
        "local_pairwise_flip": ranking_metrics["local_pairwise_flip_mat"],
        "boundary_crossing_score": ranking_metrics["boundary_crossing_score_mat"],
        "topk_flip": ranking_metrics["topk_flip_mat"].astype(np.float32),
        "global_pairwise_flip": ranking_metrics["global_pairwise_flip_mat"],
    }
    if risk_name not in mapping:
        raise ValueError(f"Unsupported cost_primary_risk: {risk_name}")
    return np.asarray(mapping[risk_name], dtype=np.float32)


def build_run_commands(
    config: Dict[str, Any],
    config_path: str,
    overrides: List[str],
) -> Dict[str, Any]:
    runtime_prefix = RUNTIME_PREFIX
    cuda_visible_devices = str(config.get("cuda_visible_devices", "")).strip()
    if cuda_visible_devices:
        runtime_prefix = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} {runtime_prefix}"

    base_command = f"{runtime_prefix} python scripts/run_motivation_experiments.py --config {config_path}"
    override_suffix = ""
    if overrides:
        override_suffix = " " + " ".join([f"--override {item}" for item in overrides])

    resolved_override_items = [
        f"policy_checkpoint_path={config.get('policy_checkpoint_path', '')}",
        f"output_dir={config.get('output_dir', '')}",
        f"cpu={config.get('cpu', True)}",
        f"cuda={config.get('cuda', 0)}",
        f"policy_deterministic_eval={config.get('policy_deterministic_eval', False)}",
    ]
    resolved_command = base_command + " " + " ".join([f"--override {item}" for item in resolved_override_items])
    return {
        "runtime_prefix": runtime_prefix,
        "config_path": config_path,
        "overrides": overrides,
        "v3_command": base_command + override_suffix,
        "v3_resolved_command": resolved_command,
        "training_command": str(config.get("training_command", "")).strip(),
        "checkpoint_path": str(config.get("policy_checkpoint_path", "")).strip(),
        "training_log_path": str(config.get("training_log_path", "")).strip(),
        "output_dir": str(config.get("output_dir", "")).strip(),
    }


def compute_chosen_action_outputs(
    sample_df: pd.DataFrame,
    uncertainty_edges: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if "is_sampled_action" not in sample_df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    chosen_df = sample_df[sample_df["is_sampled_action"].to_numpy().astype(bool)].copy()
    if chosen_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    chosen_df["chosen_topk_flip"] = chosen_df["topk_flip"].astype(np.float32)
    chosen_df["chosen_local_pairwise_flip"] = chosen_df["local_pairwise_flip"].astype(np.float32)
    chosen_df["chosen_boundary_crossing_score"] = chosen_df["boundary_crossing_score"].astype(np.float32)
    chosen_df["chosen_is_oracle_topk"] = chosen_df["is_oracle_topk"].astype(np.uint8)
    chosen_df["chosen_not_oracle_topk"] = (~chosen_df["is_oracle_topk"]).astype(np.uint8)
    chosen_df["uncertainty_bucket"] = bucketize_by_edges(
        chosen_df["uncertainty"].to_numpy(dtype=np.float32),
        np.asarray(uncertainty_edges, dtype=np.float64),
    )
    chosen_df["bucket_label"] = chosen_df["uncertainty_bucket"].map(lambda x: f"B{int(x) + 1}")

    bucket_summary = (
        chosen_df.groupby(["uncertainty_bucket", "bucket_label"], observed=False)
        .agg(
            count=("state_id", "count"),
            mean_uncertainty=("uncertainty", "mean"),
            mean_action_regret_norm=("action_regret_norm", "mean"),
            mean_action_regret_raw=("action_regret_raw", "mean"),
            mean_action_regret_cost=("action_regret_cost", "mean"),
            mean_chosen_topk_flip=("chosen_topk_flip", "mean"),
            mean_chosen_local_pairwise_flip=("chosen_local_pairwise_flip", "mean"),
            mean_chosen_boundary_crossing_score=("chosen_boundary_crossing_score", "mean"),
            mean_sampled_action_prob=("sampled_action_prob", "mean"),
            mean_policy_entropy=("policy_entropy", "mean"),
            oracle_topk_rate=("chosen_is_oracle_topk", "mean"),
            sampled_equals_top1_rate=("sampled_equals_top1", "mean"),
        )
        .reset_index()
        .sort_values("uncertainty_bucket")
    )

    corr_rows: List[Dict[str, Any]] = []
    uncertainty = chosen_df["uncertainty"].to_numpy(dtype=np.float32)
    for target in [
        "action_regret_norm",
        "action_regret_raw",
        "action_regret_cost",
        "chosen_topk_flip",
        "chosen_local_pairwise_flip",
        "chosen_boundary_crossing_score",
        "chosen_not_oracle_topk",
    ]:
        target_values = chosen_df[target].to_numpy(dtype=np.float32)
        corr_rows.append(
            {
                "target": target,
                "spearman_rho": safe_spearman(uncertainty, target_values),
                "mean_target": float(np.mean(target_values)),
                "std_target": float(np.std(target_values)),
                "n_states": int(len(chosen_df)),
            }
        )
    corr_df = pd.DataFrame(corr_rows)

    support_summary = (
        chosen_df.groupby("support_region", observed=False)
        .agg(
            count=("state_id", "count"),
            mean_uncertainty=("uncertainty", "mean"),
            mean_action_regret_norm=("action_regret_norm", "mean"),
            mean_action_regret_raw=("action_regret_raw", "mean"),
            mean_action_regret_cost=("action_regret_cost", "mean"),
            mean_chosen_topk_flip=("chosen_topk_flip", "mean"),
            mean_chosen_local_pairwise_flip=("chosen_local_pairwise_flip", "mean"),
            mean_chosen_boundary_crossing_score=("chosen_boundary_crossing_score", "mean"),
            oracle_topk_rate=("chosen_is_oracle_topk", "mean"),
        )
        .reset_index()
    )

    margin_summary = (
        chosen_df.groupby(["margin_region", "uncertainty_bucket"], observed=False)
        .agg(
            count=("state_id", "count"),
            mean_uncertainty=("uncertainty", "mean"),
            mean_action_regret_norm=("action_regret_norm", "mean"),
            mean_action_regret_raw=("action_regret_raw", "mean"),
            mean_action_regret_cost=("action_regret_cost", "mean"),
            mean_chosen_topk_flip=("chosen_topk_flip", "mean"),
            mean_chosen_local_pairwise_flip=("chosen_local_pairwise_flip", "mean"),
            mean_chosen_boundary_crossing_score=("chosen_boundary_crossing_score", "mean"),
            oracle_topk_rate=("chosen_is_oracle_topk", "mean"),
        )
        .reset_index()
        .sort_values(["margin_region", "uncertainty_bucket"])
    )

    overview = {
        "n_states": int(len(chosen_df)),
        "mean_uncertainty": float(chosen_df["uncertainty"].mean()),
        "mean_action_regret_norm": float(chosen_df["action_regret_norm"].mean()),
        "mean_action_regret_raw": float(chosen_df["action_regret_raw"].mean()),
        "mean_action_regret_cost": float(chosen_df["action_regret_cost"].mean()),
        "mean_sampled_action_prob": float(chosen_df["sampled_action_prob"].mean()),
        "mean_policy_entropy": float(chosen_df["policy_entropy"].mean()),
        "oracle_topk_rate": float(chosen_df["chosen_is_oracle_topk"].mean()),
        "sampled_equals_top1_rate": float(chosen_df["sampled_equals_top1"].mean()),
    }
    return bucket_summary, corr_df, support_summary, margin_summary, overview


def compute_uncertainty_source_comparison(sample_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    uncertainty_cols = [col for col in sample_df.columns if col.startswith("uncertainty_")]
    rows: List[Dict[str, Any]] = []
    high_error_q = float(config.get("high_error_quantile", 0.8))
    high_flip_q = float(config.get("high_flip_quantile", 0.8))
    error_threshold = float(np.quantile(sample_df["pointwise_error"], high_error_q))
    local_flip_threshold = float(np.quantile(sample_df["local_pairwise_flip"], high_flip_q))

    for col in uncertainty_cols:
        score = sample_df[col].to_numpy()
        source_name = col.replace("uncertainty_", "", 1)
        for target_col, positive_mask, threshold, positive_rule in [
            ("pointwise_error", (sample_df["pointwise_error"].to_numpy() >= error_threshold).astype(np.uint8), error_threshold, f">={high_error_q:.2f} quantile"),
            ("topk_flip", sample_df["topk_flip"].to_numpy().astype(np.uint8), 1.0, "binary topk flip"),
            ("local_pairwise_flip", (sample_df["local_pairwise_flip"].to_numpy() >= local_flip_threshold).astype(np.uint8), local_flip_threshold, f">={high_flip_q:.2f} quantile"),
        ]:
            target = sample_df[target_col].to_numpy()
            rows.append(
                {
                    "uncertainty_source": source_name,
                    "target": target_col,
                    "spearman_rho": float(pd.Series(score).corr(pd.Series(target), method="spearman")),
                    "auc": safe_roc_auc(positive_mask, score),
                    "average_precision": safe_average_precision(positive_mask, score),
                    "positive_threshold": float(threshold),
                    "positive_rule": positive_rule,
                }
            )
    return pd.DataFrame(rows)


def build_baseline_comparison(
    config: Dict[str, Any],
    current_correlation_df: pd.DataFrame,
    current_support_df: pd.DataFrame,
    current_cost_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline_dir = str(config.get("baseline_results_dir", "")).strip()
    data_source = str(config.get("analysis_data_source", "grid"))
    current_run_name = "v3_policy_rollout" if data_source == "policy_rollout" else "v2_primary"
    rows = [
        summarize_run_for_baseline(
            run_name=current_run_name,
            correlation_df=current_correlation_df,
            support_df=current_support_df,
            cost_df=current_cost_df,
        )
    ]
    if baseline_dir and os.path.isdir(baseline_dir):
        correlation_path = os.path.join(baseline_dir, "correlation_summary.csv")
        support_path = os.path.join(baseline_dir, "support_stratified_summary.csv")
        cost_path = os.path.join(baseline_dir, "cost_analysis_summary.csv")
        if os.path.exists(correlation_path) and os.path.exists(support_path) and os.path.exists(cost_path):
            rows.insert(
                0,
                summarize_run_for_baseline(
                    run_name="v1_baseline",
                    correlation_df=pd.read_csv(correlation_path),
                    support_df=pd.read_csv(support_path),
                    cost_df=pd.read_csv(cost_path),
                ),
            )
    return pd.DataFrame(rows)


def summarize_run_for_baseline(
    run_name: str,
    correlation_df: pd.DataFrame,
    support_df: pd.DataFrame,
    cost_df: pd.DataFrame,
) -> Dict[str, Any]:
    corr_map = correlation_df.set_index("target") if not correlation_df.empty else pd.DataFrame()
    support_region = None
    alignment_gap = float("nan")
    if not support_df.empty:
        if "analysis_slice" in support_df.columns:
            support_view = support_df[support_df["analysis_slice"] == "small_margin"].copy()
            if support_view.empty:
                support_view = support_df.copy()
        else:
            support_view = support_df.copy()
        gap_col = "alignment_gap_local_pairwise" if "alignment_gap_local_pairwise" in support_view.columns else "alignment_gap_pairwise"
        if gap_col in support_view.columns:
            top_row = support_view.sort_values(gap_col, ascending=False).iloc[0]
            support_region = top_row["support_region"]
            alignment_gap = float(top_row[gap_col])

    best_signal = None
    best_loss = float("nan")
    if not cost_df.empty:
        best_row = cost_df.sort_values("topk_oracle_reward_loss_mean").iloc[0]
        best_signal = best_row["signal_name"]
        best_loss = float(best_row["topk_oracle_reward_loss_mean"])

    return {
        "run_name": run_name,
        "rho_error": float(corr_map.loc["pointwise_error", "spearman_rho"]) if "pointwise_error" in corr_map.index else float("nan"),
        "rho_topk_flip": float(corr_map.loc["topk_flip", "spearman_rho"]) if "topk_flip" in corr_map.index else float("nan"),
        "rho_local_pairwise_flip": float(corr_map.loc["local_pairwise_flip", "spearman_rho"]) if "local_pairwise_flip" in corr_map.index else float(corr_map.loc["pairwise_flip", "spearman_rho"]) if "pairwise_flip" in corr_map.index else float("nan"),
        "best_support_region": support_region,
        "best_support_alignment_gap": alignment_gap,
        "best_cost_signal": best_signal,
        "best_cost_loss": best_loss,
    }


def run_subset_ablation(config: Dict[str, Any]) -> Dict[str, Any]:
    subset_config = dict(config)
    subset_config["max_users"] = int(config.get("ablation_max_users", 64))
    subset_config["subset_items"] = int(config.get("ablation_subset_items", 512))
    subset_config["output_dir"] = str(config.get("subset_ablation_output_dir", "results/motivation_kuairec_dorl_v2_ablation"))
    subset_bundle = execute_experiment(subset_config)
    ablation_df = compute_subset_ablation_summary(subset_bundle["sample_df"], subset_config)

    baseline_df = build_baseline_comparison(
        config=subset_config,
        current_correlation_df=subset_bundle["correlation_df"],
        current_support_df=subset_bundle["support_df"],
        current_cost_df=subset_bundle["cost_df"],
    )
    save_experiment_outputs(
        bundle=subset_bundle,
        config=subset_config,
        ablation_df=ablation_df,
        baseline_df=baseline_df,
        ablation_run_dir=None,
    )
    return {
        "ablation_df": ablation_df,
        "output_dir": subset_bundle["output_dir"],
    }


def compute_subset_ablation_summary(sample_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    variants = [
        {
            "variant_name": "base_primary",
            "description": "ensemble_disagreement + shortlist_union + local_pairwise + per_user quantile",
            "uncertainty_col": "uncertainty",
            "support_mode": "per_user",
            "margin_mode": "per_user",
            "slice_mode": "shortlist_union",
            "pairwise_target": "local_pairwise_flip",
        },
        {
            "variant_name": "uncertainty_variance_head",
            "description": "replace primary uncertainty with variance_head",
            "uncertainty_col": "uncertainty_variance_head",
            "support_mode": "per_user",
            "margin_mode": "per_user",
            "slice_mode": "shortlist_union",
            "pairwise_target": "local_pairwise_flip",
        },
        {
            "variant_name": "slice_all_items",
            "description": "replace shortlist_union with all_items",
            "uncertainty_col": "uncertainty",
            "support_mode": "per_user",
            "margin_mode": "per_user",
            "slice_mode": "all_items",
            "pairwise_target": "local_pairwise_flip",
        },
        {
            "variant_name": "pairwise_global",
            "description": "replace local_pairwise_flip with global_pairwise_flip",
            "uncertainty_col": "uncertainty",
            "support_mode": "per_user",
            "margin_mode": "per_user",
            "slice_mode": "shortlist_union",
            "pairwise_target": "global_pairwise_flip",
        },
        {
            "variant_name": "quantile_global",
            "description": "replace per_user support/margin quantiles with global quantiles",
            "uncertainty_col": "uncertainty",
            "support_mode": "global",
            "margin_mode": "global",
            "slice_mode": "shortlist_union",
            "pairwise_target": "local_pairwise_flip",
        },
    ]

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        variant_df = sample_df.copy()
        if variant["uncertainty_col"] in variant_df.columns:
            variant_df["uncertainty"] = variant_df[variant["uncertainty_col"]]
        support_col = f"support_region_{variant['support_mode']}"
        margin_col = f"margin_region_{variant['margin_mode']}"
        if support_col in variant_df.columns:
            variant_df["support_region"] = variant_df[support_col]
        if margin_col in variant_df.columns:
            variant_df["margin_region"] = variant_df[margin_col]
        if variant["slice_mode"] == "all_items":
            variant_df["in_primary_slice"] = True
        else:
            variant_df["in_primary_slice"] = variant_df["in_shortlist"]
        if variant["pairwise_target"] in variant_df.columns:
            variant_df["local_pairwise_flip"] = variant_df[variant["pairwise_target"]]

        primary_df = filter_sample_df(variant_df)
        correlation_df, _ = compute_correlation_summary(primary_df, config)
        support_df = compute_support_stratified_summary(
            primary_df,
            config,
            build_analysis_slices(primary_df, "small_margin"),
        )
        corr_map = correlation_df.set_index("target")
        support_view = support_df[support_df["analysis_slice"] == "small_margin"].copy()
        if support_view.empty:
            support_view = support_df.copy()
        best_region = None
        best_gap = float("nan")
        if not support_view.empty:
            top_row = support_view.sort_values("alignment_gap_local_pairwise", ascending=False).iloc[0]
            best_region = top_row["support_region"]
            best_gap = float(top_row["alignment_gap_local_pairwise"])
        rows.append(
            {
                "variant_name": variant["variant_name"],
                "description": variant["description"],
                "rho_error": float(corr_map.loc["pointwise_error", "spearman_rho"]),
                "rho_topk_flip": float(corr_map.loc["topk_flip", "spearman_rho"]),
                "rho_pairwise_target": float(corr_map.loc["local_pairwise_flip", "spearman_rho"]),
                "best_support_region_small_margin": best_region,
                "best_alignment_gap_small_margin": best_gap,
                "n_samples": int(len(primary_df)),
            }
        )
    return pd.DataFrame(rows)


def apply_smoke_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(config)
    if bool(config.get("smoke_test", False)):
        if int(config.get("max_users", 0)) <= 0:
            config["max_users"] = 32
        if int(config.get("subset_items", 0)) <= 0:
            config["subset_items"] = 256
        if config.get("output_dir") == "results/motivation_kuairec_dorl_v2_full":
            config["output_dir"] = "results/motivation_kuairec_dorl_v2_smoke"
        config["run_subset_ablation"] = False
    return config


if __name__ == "__main__":
    main()
