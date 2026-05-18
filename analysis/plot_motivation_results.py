import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_all_plots(
    bucket_df: pd.DataFrame,
    support_df: pd.DataFrame,
    margin_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    chosen_action_df: pd.DataFrame,
    figure_dir: str,
) -> Dict[str, str]:
    sns.set_theme(style="whitegrid")
    paths = {
        "uncertainty_vs_error": os.path.join(figure_dir, "uncertainty_vs_error.png"),
        "uncertainty_vs_flip": os.path.join(figure_dir, "uncertainty_vs_flip.png"),
        "support_stratified_alignment_gap": os.path.join(figure_dir, "support_stratified_alignment_gap.png"),
        "margin_stratified_flip": os.path.join(figure_dir, "margin_stratified_flip.png"),
        "oracle_risk_cost_comparison": os.path.join(figure_dir, "oracle_risk_cost_comparison.png"),
        "chosen_action_uncertainty_vs_regret": os.path.join(figure_dir, "chosen_action_uncertainty_vs_regret.png"),
    }
    plot_uncertainty_vs_error(bucket_df, paths["uncertainty_vs_error"])
    plot_uncertainty_vs_flip(bucket_df, paths["uncertainty_vs_flip"])
    plot_support_alignment(support_df, paths["support_stratified_alignment_gap"])
    plot_margin_flip(margin_df, paths["margin_stratified_flip"])
    plot_cost_comparison(cost_df, paths["oracle_risk_cost_comparison"])
    plot_chosen_action_uncertainty_vs_regret(chosen_action_df, paths["chosen_action_uncertainty_vs_regret"])
    return paths


def plot_uncertainty_vs_error(bucket_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(7, 4.5))
    if bucket_df.empty:
        _save_empty_figure(output_path, "No bucket statistics available")
        return
    ax = sns.lineplot(
        data=bucket_df,
        x="uncertainty_bucket",
        y="mean_pointwise_error",
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    ax.set_xlabel("Uncertainty Bucket")
    ax.set_ylabel("Average Pointwise Error")
    ax.set_title("Raw Uncertainty vs Pointwise Error")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_uncertainty_vs_flip(bucket_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(7, 4.5))
    if bucket_df.empty:
        _save_empty_figure(output_path, "No bucket statistics available")
        return
    plot_df = bucket_df.melt(
        id_vars=["uncertainty_bucket"],
        value_vars=["mean_topk_flip", "mean_local_pairwise_flip", "mean_boundary_crossing_score"],
        var_name="target",
        value_name="value",
    )
    plot_df["target"] = plot_df["target"].map(
        {
            "mean_topk_flip": "Top-K Membership Flip",
            "mean_local_pairwise_flip": "Local Pairwise Flip@L",
            "mean_boundary_crossing_score": "Boundary Crossing Score",
        }
    )
    ax = sns.lineplot(
        data=plot_df,
        x="uncertainty_bucket",
        y="value",
        hue="target",
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("Uncertainty Bucket")
    ax.set_ylabel("Average Flip Risk")
    ax.set_title("Raw Uncertainty vs Ranking Flip Risk")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_support_alignment(support_df: pd.DataFrame, output_path: str) -> None:
    if support_df.empty:
        _save_empty_figure(output_path, "No support summary available")
        return

    plot_df = support_df.copy()
    if "analysis_slice" in plot_df.columns:
        small_margin_df = plot_df[plot_df["analysis_slice"] == "small_margin"].copy()
        if not small_margin_df.empty:
            plot_df = small_margin_df

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    corr_df = plot_df.melt(
        id_vars=["support_region"],
        value_vars=["rho_uncertainty_error", "rho_uncertainty_topk_flip", "rho_uncertainty_local_pairwise_flip"],
        var_name="metric",
        value_name="rho",
    )
    corr_df["metric"] = corr_df["metric"].map(
        {
            "rho_uncertainty_error": "rho(u,error)",
            "rho_uncertainty_topk_flip": "rho(u,topK flip)",
            "rho_uncertainty_local_pairwise_flip": "rho(u,local pairwise)",
        }
    )
    sns.barplot(data=corr_df, x="support_region", y="rho", hue="metric", ax=axes[0])
    axes[0].set_xlabel("Support Region")
    axes[0].set_ylabel("Spearman Correlation")
    axes[0].set_title("Support-Stratified Correlation")

    gap_df = plot_df.melt(
        id_vars=["support_region"],
        value_vars=["alignment_gap_topk", "alignment_gap_local_pairwise"],
        var_name="gap_type",
        value_name="gap",
    )
    gap_df["gap_type"] = gap_df["gap_type"].map(
        {
            "alignment_gap_topk": "Gap vs Top-K Flip",
            "alignment_gap_local_pairwise": "Gap vs Local Pairwise Flip",
        }
    )
    sns.barplot(data=gap_df, x="support_region", y="gap", hue="gap_type", ax=axes[1])
    axes[1].set_xlabel("Support Region")
    axes[1].set_ylabel("Alignment Gap")
    axes[1].set_title("Support-Stratified Alignment Gap")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_margin_flip(margin_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(7.5, 4.8))
    if margin_df.empty:
        _save_empty_figure(output_path, "No margin summary available")
        return
    ax = sns.lineplot(
        data=margin_df,
        x="uncertainty_bucket",
        y="mean_topk_flip",
        hue="margin_region",
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("Uncertainty Bucket")
    ax.set_ylabel("Average Top-K Flip Rate")
    ax.set_title("Margin-Stratified Top-K Flip")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_cost_comparison(cost_df: pd.DataFrame, output_path: str) -> None:
    if cost_df.empty:
        _save_empty_figure(output_path, "No cost summary available")
        return

    best_df = cost_df[cost_df["is_best_for_source"]].copy()
    best_df["signal_name"] = best_df["signal_name"].map(
        {
            "baseline": "Baseline",
            "raw_uncertainty": "Raw Uncertainty",
            "oracle_pointwise_error": "Oracle Pointwise Error",
            "oracle_local_flip_risk": "Oracle Local Flip Risk",
            "oracle_topk_flip": "Oracle Top-K Flip",
        }
    ).fillna(best_df["signal_name"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(data=best_df, x="signal_name", y="topk_oracle_reward_loss_mean", ax=axes[0], color="#2ca02c")
    axes[0].set_xlabel("Risk Signal")
    axes[0].set_ylabel("Top-K Oracle Reward Loss")
    axes[0].set_title("Best Penalty by Risk Source")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=best_df, x="signal_name", y="exposure_weighted_cost_mean", ax=axes[1], color="#d62728")
    axes[1].set_xlabel("Risk Signal")
    axes[1].set_ylabel("Exposure-Weighted Cost")
    axes[1].set_title("Best Exposure Cost by Risk Source")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_chosen_action_uncertainty_vs_regret(chosen_action_df: pd.DataFrame, output_path: str) -> None:
    if chosen_action_df.empty:
        _save_empty_figure(output_path, "No chosen-action summary available")
        return

    if "uncertainty_bucket" not in chosen_action_df.columns:
        _save_empty_figure(output_path, "Chosen-action summary misses uncertainty buckets")
        return

    plot_df = chosen_action_df.melt(
        id_vars=["uncertainty_bucket"],
        value_vars=[
            "mean_action_regret_norm",
            "mean_chosen_local_pairwise_flip",
            "mean_chosen_boundary_crossing_score",
        ],
        var_name="metric",
        value_name="value",
    )
    plot_df["metric"] = plot_df["metric"].map(
        {
            "mean_action_regret_norm": "Mean Action Regret (Norm)",
            "mean_chosen_local_pairwise_flip": "Mean Chosen Local Flip",
            "mean_chosen_boundary_crossing_score": "Mean Chosen Boundary Risk",
        }
    )
    plt.figure(figsize=(7.5, 4.8))
    ax = sns.lineplot(
        data=plot_df,
        x="uncertainty_bucket",
        y="value",
        hue="metric",
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("Chosen-Action Uncertainty Bucket")
    ax.set_ylabel("Average Value")
    ax.set_title("Chosen Action Uncertainty vs Regret/Risk")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def _save_empty_figure(output_path: str, text: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.5, text, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
