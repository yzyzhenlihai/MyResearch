import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from analysis.common import dataframe_to_markdown


def generate_report(
    config: Dict,
    correlation_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    support_df: pd.DataFrame,
    margin_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    figure_paths: Dict[str, str],
    output_dir: str,
    uncertainty_compare_df: Optional[pd.DataFrame] = None,
    baseline_df: Optional[pd.DataFrame] = None,
    ablation_df: Optional[pd.DataFrame] = None,
    policy_trace_summary: Optional[Dict[str, float]] = None,
    chosen_action_summary_df: Optional[pd.DataFrame] = None,
    chosen_action_corr_df: Optional[pd.DataFrame] = None,
    chosen_action_support_df: Optional[pd.DataFrame] = None,
    chosen_action_margin_df: Optional[pd.DataFrame] = None,
) -> str:
    data_source = str(config.get("analysis_data_source", "grid"))
    has_chosen_action = chosen_action_summary_df is not None and not chosen_action_summary_df.empty
    version_name = "v3 policy-faithful 单动作版" if data_source == "policy_rollout" else "v2 口径修正版"
    sec_uncertainty_compare = 6 if has_chosen_action else 5
    sec_bucket = 7 if has_chosen_action else 6
    sec_support = 8 if has_chosen_action else 7
    sec_margin = 9 if has_chosen_action else 8
    sec_cost = 10 if has_chosen_action else 9
    sec_ablation = 11 if has_chosen_action else 10
    sec_baseline = 12 if has_chosen_action else 11
    sec_limit = 13 if has_chosen_action else 12
    sec_method = 14 if has_chosen_action else 13
    report_lines = [
        f"# 动机实验报告（{version_name}）",
        "",
        "## 1. 实验目的",
        "",
        "本报告用于诊断：在 KuaiRec 场景下，raw predictive uncertainty 更接近 pointwise model error，还是更接近排序决策侧的 ranking risk。",
        "",
        "当前版本的核心变化是：",
        "- 主 uncertainty 改为 `ensemble_disagreement`，`variance_head` 仅作为 appendix baseline。",
        "- 主评估切片改为 `pred_topL ∪ oracle_topL` 的 policy-relevant shortlist。",
        "- 主 pairwise risk 改为 shortlist 内、并按 boundary proximity 加权的 `local_pairwise_flip@L`。",
        "- support / margin 的主分层改为 per-user quantile，并额外查看 `small-margin` 条件化结果。",
        "",
        "## 2. 数据与设置",
        "",
        f"- 环境：`{config['env']}`",
        f"- reward model：`{config['user_model_name']}`，checkpoint 标记为 `{config['read_message']}`",
        f"- 分析数据源：`{data_source}`",
        f"- 主 uncertainty：`{config.get('primary_uncertainty_source', config.get('uncertainty_source'))}`",
        f"- appendix uncertainty：`{config.get('appendix_uncertainty_sources', [])}`",
        f"- 主评估切片：`{config.get('evaluation_slice', 'all_items')}`，`shortlist_topl={config.get('shortlist_topl', 100)}`",
        f"- Top-K：`{config['topk']}`",
        f"- support 分层模式：`{config.get('support_quantile_mode', 'global')}`",
        f"- margin 分层模式：`{config.get('margin_quantile_mode', 'global')}`",
        f"- 主 cost risk：`{config.get('cost_primary_risk', 'local_pairwise_flip')}`",
        f"- 决策成本 oracle：`{config['decision_cost_reward']}`",
        f"- policy rollout 步数：`{config.get('policy_rollout_n_steps', 'N/A')}`",
        f"- policy rollout 是否确定性评估：`{config.get('policy_deterministic_eval', False)}`",
        "",
        "## 3. 指标定义",
        "",
        "- Pointwise error：`|r_hat(s,a) - r_oracle(s,a)|`。",
        "- Top-K membership flip：预测 top-K 与 oracle top-K 成员关系的异或。",
        "- Local pairwise flip@L：在 `pred_topL ∪ oracle_topL` 内统计的局部 pairwise 乱序度，并按离 top-K cutoff 的 boundary proximity 加权。",
        "- Boundary crossing score：`topk_flip × boundary_weight`，用于强调“接近 cutoff 且发生 membership crossing”的动作。",
        "- Support score：由 item embedding 上的 kNN support proxy 经指数核变换得到；主分层基于 per-user quantile。",
        "- Margin：动作分数到预测 top-K cutoff 的绝对距离；主分层基于 per-user quantile。",
        "",
        "## 4. 主结果：相关性与检测能力",
        "",
        dataframe_to_markdown(correlation_df, float_digits=4),
        "",
        hypothesis_text(correlation_df, support_df, margin_df, cost_df),
        "",
    ]

    if has_chosen_action:
        report_lines.extend(
            [
                "## 5. 真实 Policy Rollout 与 Chosen-Action 结果",
                "",
                policy_rollout_text(config, policy_trace_summary),
                "",
                "### 5.1 Chosen-Action Uncertainty 分桶摘要",
                "",
                dataframe_to_markdown(chosen_action_summary_df, float_digits=4),
                "",
                figure_block(
                    "图 2A. 真实 sampled action 上，uncertainty 与 regret/risk 的关系。",
                    figure_paths["chosen_action_uncertainty_vs_regret"],
                    output_dir,
                ),
                "",
                "### 5.2 Chosen-Action 相关性",
                "",
                dataframe_to_markdown(chosen_action_corr_df, float_digits=4),
                "",
                chosen_action_text(chosen_action_corr_df),
                "",
                "### 5.3 Chosen-Action 的 Support / Margin 分层",
                "",
                "**Support 分层**",
                "",
                dataframe_to_markdown(chosen_action_support_df, float_digits=4),
                "",
                "**Margin 分层（前 15 行）**",
                "",
                dataframe_to_markdown(chosen_action_margin_df.head(15), float_digits=4),
                "",
            ]
        )

    if uncertainty_compare_df is not None and not uncertainty_compare_df.empty:
        report_lines.extend(
            [
                f"## {sec_uncertainty_compare}. Uncertainty Source 对照",
                "",
                dataframe_to_markdown(uncertainty_compare_df, float_digits=4),
                "",
                uncertainty_source_text(uncertainty_compare_df),
                "",
            ]
        )

    report_lines.extend(
        [
            f"## {sec_bucket}. Uncertainty 分桶结果",
            "",
            dataframe_to_markdown(bucket_df, float_digits=4),
            "",
            figure_block("图 1. Raw uncertainty 与 pointwise error 的关系。", figure_paths["uncertainty_vs_error"], output_dir),
            "",
            figure_block("图 2. Raw uncertainty 与 ranking flip risk 的关系。", figure_paths["uncertainty_vs_flip"], output_dir),
            "",
            "解读：如果 error 曲线随 uncertainty 上升而更稳定，而 local flip / boundary crossing 曲线更弱，就支持“预测侧 proxy 与决策侧风险信号错位”的论点。",
            "",
            f"## {sec_support}. Support 分层分析",
            "",
            dataframe_to_markdown(support_df, float_digits=4),
            "",
            figure_block("图 3. Support 分层下的 alignment gap。", figure_paths["support_stratified_alignment_gap"], output_dir),
            "",
            support_analysis_text(support_df),
            "",
            f"## {sec_margin}. Margin 分层分析",
            "",
            dataframe_to_markdown(margin_df.head(15), float_digits=4),
            "",
            figure_block("图 4. Margin 分层下的 ranking flip。", figure_paths["margin_stratified_flip"], output_dir),
            "",
            margin_analysis_text(margin_df),
            "",
            f"## {sec_cost}. Oracle Risk 决策成本分析",
            "",
            dataframe_to_markdown(cost_df, float_digits=4),
            "",
            figure_block("图 5. 不同 oracle risk 信号的决策成本比较。", figure_paths["oracle_risk_cost_comparison"], output_dir),
            "",
            cost_analysis_text(cost_df),
            "",
        ]
    )

    if ablation_df is not None and not ablation_df.empty:
        report_lines.extend(
            [
                f"## {sec_ablation}. Subset Ablation",
                "",
                dataframe_to_markdown(ablation_df, float_digits=4),
                "",
                ablation_text(ablation_df),
                "",
            ]
        )

    if baseline_df is not None and not baseline_df.empty:
        report_lines.extend(
            [
                f"## {sec_baseline}. 与 v2 Baseline 对照" if data_source == "policy_rollout" else f"## {sec_baseline}. 与 v1 Baseline 对照",
                "",
                dataframe_to_markdown(baseline_df, float_digits=4),
                "",
                baseline_text(baseline_df),
                "",
            ]
        )

    report_lines.extend(
        [
            f"## {sec_limit}. 局限性与后续方向",
            "",
            "- 当前实现仍然没有把 uncertainty 本身改成 sequence-conditioned policy uncertainty；即便使用 policy rollout，uncertainty 侧仍主要来自静态 user-item 资产。",
            "- support 仍是 embedding-based proxy，不应被解释为精确的 `mu_hat(a|s)`。",
            "- local pairwise flip@L 是更 policy-relevant 的代理，但仍不是最终的真实 deployment risk；后续可继续做 simulator rollout 或 sequence-conditioned risk。",
            "- 如果 H3/H5 仍然只得到部分支持，下一步应重点检查：support boundary 的定义、policy-conditioned weighting，以及是否需要显式学习 calibrated flip-risk head。",
            "",
            f"## {sec_method}. 方法启发",
            "",
            "如果当前设置下 H1/H2 比 baseline 更稳定，而 H3/H5 只在 small-margin / shortlist 条件下更明显，那么更合理的结论是：raw uncertainty 本身不是错误的，但它需要被放在更接近决策边界的分析口径里，或者被进一步校准成 ranking flip risk。",
            "",
        ]
    )
    return "\n".join(report_lines)


def generate_implementation_log(
    config: Dict,
    output_dir: str,
    figure_paths: Dict[str, str],
    summary_paths: Dict[str, str],
    primary_uncertainty_source: str,
    evaluation_slice_name: str,
    cost_primary_risk: str,
    ablation_run_dir: Optional[str] = None,
    run_commands: Optional[Dict[str, str]] = None,
    policy_trace_summary: Optional[Dict[str, float]] = None,
) -> str:
    data_source = str(config.get("analysis_data_source", "grid"))
    lines = [
        "# 动机实验实现日志",
        "",
        "## 实现范围",
        "",
        "实现了 KuaiRec 动机实验管线，并在不改动训练主流程的前提下，支持静态 grid 分析与基于 DORL policy rollout 的 policy-faithful 单动作分析。",
        "",
        "## 关键设计选择",
        "",
        f"- 分析数据源：`{data_source}`。",
        f"- 主 uncertainty：`{primary_uncertainty_source}`。",
        "- appendix uncertainty：保留 `variance_head` 用于对照。",
        f"- 主评估切片：`{evaluation_slice_name}`。",
        "- 主 pairwise risk：`local_pairwise_flip@L`，在 shortlist 内计算并按 boundary proximity 加权。",
        f"- 主 cost risk：`{cost_primary_risk}`。",
        "- Support / Margin：同时产出 `global` 与 `per_user` 两套 region code，主分析默认使用 `per_user`。",
        "- 决策成本：统一采用 matched-budget，比较 `soft_penalty` 与 `hard_mask`。",
        "",
        "## 主入口",
        "",
        "`scripts/run_motivation_experiments.py --config configs/motivation_experiments_policy_v3.yaml`" if data_source == "policy_rollout" else "`scripts/run_motivation_experiments.py --config configs/motivation_experiments.yaml`",
        "",
    ]
    if run_commands:
        lines.extend(
            [
                "## 详细运行命令",
                "",
                f"- 训练命令：`{run_commands.get('training_command', '')}`",
                f"- v3 分析命令：`{run_commands.get('v3_command', '')}`",
                f"- v3 解析后命令：`{run_commands.get('v3_resolved_command', '')}`",
                f"- 配置文件：`{run_commands.get('config_path', '')}`",
                f"- checkpoint：`{run_commands.get('checkpoint_path', '')}`",
                f"- 训练日志：`{run_commands.get('training_log_path', '')}`",
                f"- 输出目录：`{run_commands.get('output_dir', '')}`",
                "",
            ]
        )
    lines.extend(
        [
        "## 解析后的配置",
        "",
        "```yaml",
        _yaml_dump(config),
        "```",
        "",
        ]
    )
    if policy_trace_summary:
        lines.extend(
            [
                "## Policy Rollout 摘要",
                "",
                f"- `n_states`: `{policy_trace_summary.get('n_states', 'N/A')}`",
                f"- `n_episodes_observed`: `{policy_trace_summary.get('n_episodes_observed', 'N/A')}`",
                f"- `mean_action_regret_norm`: `{policy_trace_summary.get('mean_action_regret_norm', float('nan')):.4f}`" if "mean_action_regret_norm" in policy_trace_summary else "- `mean_action_regret_norm`: `N/A`",
                f"- `mean_action_regret_cost`: `{policy_trace_summary.get('mean_action_regret_cost', float('nan')):.4f}`" if "mean_action_regret_cost" in policy_trace_summary else "- `mean_action_regret_cost`: `N/A`",
                f"- `mean_sampled_action_prob`: `{policy_trace_summary.get('mean_sampled_action_prob', float('nan')):.4f}`" if "mean_sampled_action_prob" in policy_trace_summary else "- `mean_sampled_action_prob`: `N/A`",
                f"- `sampled_equals_top1_rate`: `{policy_trace_summary.get('sampled_equals_top1_rate', float('nan')):.4f}`" if "sampled_equals_top1_rate" in policy_trace_summary else "- `sampled_equals_top1_rate`: `N/A`",
                "",
            ]
        )
    lines.extend(
        [
        "## 已生成的汇总文件",
        "",
        ]
    )
    for name, path in summary_paths.items():
        rel_path = os.path.relpath(path, output_dir)
        lines.append(f"- `{name}`: `{rel_path}`")
    lines.extend(["", "## 已生成的图表", ""])
    for name, path in figure_paths.items():
        rel_path = os.path.relpath(path, output_dir)
        lines.append(f"- `{name}`: `{rel_path}`")
    if ablation_run_dir:
        lines.extend(["", "## 关联运行", "", f"- subset ablation 输出目录：`{os.path.relpath(ablation_run_dir, output_dir)}`"])
    lines.extend(
        [
            "",
            "## 已知局限性",
            "",
            "- v3 已切到真实 rollout 状态和真实 sampled action，但 uncertainty 侧仍来自静态 user-item 资产，而不是 sequence-conditioned policy uncertainty。",
            "- shortlist 与 local flip 更贴近决策边界，但 matrix-level 结果仍不等同于真实在线 rollout 风险，必须结合 chosen-action 结果一起解释。",
            "- support 估计仍是 embedding-generalized proxy，而不是显式训练的行为策略模型。",
            "",
        ]
    )
    return "\n".join(lines)


def hypothesis_text(
    correlation_df: pd.DataFrame,
    support_df: pd.DataFrame,
    margin_df: pd.DataFrame,
    cost_df: pd.DataFrame,
) -> str:
    if correlation_df.empty:
        return "当前没有可用的相关性汇总结果。"
    corr_map = correlation_df.set_index("target")
    rho_error = corr_map.loc["pointwise_error", "spearman_rho"] if "pointwise_error" in corr_map.index else np.nan
    rho_topk = corr_map.loc["topk_flip", "spearman_rho"] if "topk_flip" in corr_map.index else np.nan
    rho_local = corr_map.loc["local_pairwise_flip", "spearman_rho"] if "local_pairwise_flip" in corr_map.index else np.nan

    parts = []
    if np.isfinite(rho_error) and rho_error > 0:
        parts.append(f"H1 {'得到支持' if rho_error > 0.05 else '得到较弱支持'}，因为 `rho(u,error)={rho_error:.4f}` 为正。")
    else:
        parts.append("H1 仍未被清晰支持，说明 uncertainty proxy 或 error 口径还可能存在残余错位。")

    if np.isfinite(rho_error) and np.isfinite(rho_local):
        if rho_error > rho_local:
            parts.append(f"H2 得到支持，因为 `rho(u,error)={rho_error:.4f}` 强于 `rho(u,local_pairwise)={rho_local:.4f}`。")
        else:
            parts.append("H2 仅得到部分支持或未被支持：当前 uncertainty 与 local flip 的对齐并没有明显弱于与 error 的对齐。")

    if not support_df.empty:
        support_view = support_df[support_df["analysis_slice"] == "small_margin"].copy() if "analysis_slice" in support_df.columns else support_df.copy()
        if support_view.empty:
            support_view = support_df.copy()
        best_row = support_view.sort_values("alignment_gap_local_pairwise", ascending=False).iloc[0]
        parts.append(f"H3 的主判据取 `small-margin` 条件化版本，当前最大 alignment gap 出现在 `{best_row['support_region']}`。")
    if not margin_df.empty:
        agg = margin_df.groupby("margin_region", observed=False)["mean_topk_flip"].mean()
        if "small" in agg.index and "large" in agg.index and agg["small"] > agg["large"]:
            parts.append("H4 得到支持：small-margin 区域的 flip risk 明显高于 large-margin 区域。")
        else:
            parts.append("H4 没有得到清晰支持。")
    if not cost_df.empty:
        best_rows = cost_df[cost_df["is_best_for_source"]]
        local_rows = best_rows[best_rows["signal_name"] == "oracle_local_flip_risk"]
        raw_rows = best_rows[best_rows["signal_name"] == "raw_uncertainty"]
        if not local_rows.empty and not raw_rows.empty:
            best_local = float(local_rows["topk_oracle_reward_loss_mean"].iloc[0])
            best_raw = float(raw_rows["topk_oracle_reward_loss_mean"].iloc[0])
            if best_local <= best_raw:
                parts.append(f"H5 得到支持：oracle local flip risk 的最佳 loss（`{best_local:.4f}`）不劣于 raw uncertainty（`{best_raw:.4f}`）。")
            else:
                parts.append("H5 当前只得到部分支持：matched-budget 下 raw uncertainty 仍可能更保守。")
    return " ".join(parts)


def uncertainty_source_text(uncertainty_compare_df: pd.DataFrame) -> str:
    if uncertainty_compare_df.empty:
        return "当前没有 uncertainty source 对照结果。"
    error_df = uncertainty_compare_df[uncertainty_compare_df["target"] == "pointwise_error"].copy()
    if error_df.empty:
        return "当前没有 uncertainty source 与 pointwise error 的对照结果。"
    best_row = error_df.sort_values("spearman_rho", ascending=False).iloc[0]
    return f"在 uncertainty source 对照中，`{best_row['uncertainty_source']}` 对 pointwise error 的相关性最强（`rho={best_row['spearman_rho']:.4f}`），可作为 v2 主 uncertainty 的经验依据。"


def policy_rollout_text(config: Dict, policy_trace_summary: Optional[Dict[str, float]]) -> str:
    if not policy_trace_summary:
        return "当前没有可用的 policy rollout 摘要。"
    checkpoint = str(config.get("policy_checkpoint_path", ""))
    best_epoch = config.get("training_best_epoch", "N/A")
    best_reward = config.get("training_best_reward", "N/A")
    return (
        f"v3 使用真实 DORL policy rollout 作为分析数据源，"
        f"主 checkpoint 固定为 `{checkpoint}`，训练日志给出的最佳 epoch 为 `{best_epoch}`，"
        f"对应 `best_reward={best_reward}`。"
        f"`checkpoint_loaded={policy_trace_summary.get('checkpoint_loaded', 'N/A')}`。"
        f"本次 rollout 共观测 `{policy_trace_summary.get('n_states', 'N/A')}` 个状态、"
        f"`{policy_trace_summary.get('n_episodes_observed', 'N/A')}` 条 episode，"
        f"并保持 `policy_deterministic_eval={config.get('policy_deterministic_eval', False)}`，"
        "因此主结果对应真实单动作随机采样决策，而不是静态 top-K 排序近似。"
    )


def chosen_action_text(chosen_action_corr_df: pd.DataFrame) -> str:
    if chosen_action_corr_df is None or chosen_action_corr_df.empty:
        return "当前没有可用的 chosen-action 相关性结果。"
    corr_map = chosen_action_corr_df.set_index("target")
    rho_regret = corr_map.loc["action_regret_norm", "spearman_rho"] if "action_regret_norm" in corr_map.index else np.nan
    rho_local = corr_map.loc["chosen_local_pairwise_flip", "spearman_rho"] if "chosen_local_pairwise_flip" in corr_map.index else np.nan
    rho_topk = corr_map.loc["chosen_topk_flip", "spearman_rho"] if "chosen_topk_flip" in corr_map.index else np.nan
    parts = []
    if np.isfinite(rho_regret):
        if rho_regret < 0:
            parts.append(
                f"在真实 sampled action 层面，uncertainty 与 `action_regret_norm` 的相关性为 `{rho_regret:.4f}`，"
                "说明它仍不是稳定的逐动作 regret/error proxy。"
            )
        else:
            parts.append(
                f"在真实 sampled action 层面，uncertainty 与 `action_regret_norm` 的相关性为 `{rho_regret:.4f}`，"
                "说明它已经体现出一定的决策损失指示能力。"
            )
    if np.isfinite(rho_local):
        parts.append(
            f"同时，uncertainty 与 chosen action 的 `local_pairwise_flip` 相关性为 `{rho_local:.4f}`，"
            "这可以直接回答“局部排序不稳定性是否会延伸到真实单动作采样决策”。"
        )
    if np.isfinite(rho_topk):
        parts.append(f"对 chosen action 的 `topk_flip`，相关性为 `{rho_topk:.4f}`。")
    return " ".join(parts)


def support_analysis_text(support_df: pd.DataFrame) -> str:
    if support_df.empty:
        return "当前没有可用的 support 分层结果。"
    plot_df = support_df.copy()
    if "analysis_slice" in plot_df.columns:
        small_margin_df = plot_df[plot_df["analysis_slice"] == "small_margin"].copy()
        if not small_margin_df.empty:
            plot_df = small_margin_df
    best_row = plot_df.sort_values("alignment_gap_local_pairwise", ascending=False).iloc[0]
    return (
        f"在主判据（`small-margin` 条件化 + local pairwise gap）下，"
        f"最大的错位出现在 `{best_row['support_region']}`，"
        f"`alignment_gap_local_pairwise={best_row['alignment_gap_local_pairwise']:.4f}`。"
    )


def margin_analysis_text(margin_df: pd.DataFrame) -> str:
    if margin_df.empty:
        return "当前没有可用的 margin 汇总结果。"
    agg = margin_df.groupby("margin_region", observed=False)[["mean_topk_flip", "mean_local_pairwise_flip"]].mean()
    return (
        "在 uncertainty bucket 上做平均后，Top-K flip 的 margin 汇总为："
        f"`small={agg.loc['small', 'mean_topk_flip']:.4f}`，"
        f"`medium={agg.loc['medium', 'mean_topk_flip']:.4f}`，"
        f"`large={agg.loc['large', 'mean_topk_flip']:.4f}`。"
        "如果 small-margin 的 flip risk 最高，就说明真正的 ranking sensitivity 不能被 raw uncertainty 单独编码。"
    )


def cost_analysis_text(cost_df: pd.DataFrame) -> str:
    if cost_df.empty:
        return "当前没有可用的决策成本汇总结果。"
    best_df = cost_df[cost_df["is_best_for_source"]].copy()
    best_df = best_df.sort_values("topk_oracle_reward_loss_mean")
    best_row = best_df.iloc[0]
    return (
        f"当前观察到的最佳配置是 `{best_row['signal_name']}`，"
        f"`cost_mode={best_row['cost_mode']}`，"
        f"`budget={best_row['budget_label']}`，"
        f"`lambda={best_row['penalty_lambda']:.4f}`，"
        f"`topk_oracle_reward_loss_mean={best_row['topk_oracle_reward_loss_mean']:.4f}`。"
    )


def ablation_text(ablation_df: pd.DataFrame) -> str:
    if ablation_df.empty:
        return "当前没有 subset ablation 结果。"
    base_df = ablation_df[ablation_df["variant_name"] == "base_primary"]
    if base_df.empty:
        return "subset ablation 已生成，但缺少 base variant。"
    base_row = base_df.iloc[0]
    return (
        f"subset ablation 中，base variant 的 `rho_error={base_row['rho_error']:.4f}`、"
        f"`rho_pairwise_target={base_row['rho_pairwise_target']:.4f}`。"
        "后续应重点观察：切回 `variance_head`、`global_pairwise` 或 `all_items` 时，这两个指标是否再次退化。"
    )


def baseline_text(baseline_df: pd.DataFrame) -> str:
    if baseline_df.empty or len(baseline_df) < 2:
        return "当前没有可用的 baseline 对照。"
    base = baseline_df[baseline_df["run_name"] == "v1_baseline"]
    cur = baseline_df[baseline_df["run_name"] != "v1_baseline"]
    if base.empty or cur.empty:
        return "baseline 对照信息不完整。"
    base_row = base.iloc[0]
    cur_row = cur.iloc[0]
    return (
        f"与 v1 相比，v2 的 `rho_error` 从 `{base_row['rho_error']:.4f}` 变为 `{cur_row['rho_error']:.4f}`，"
        f"`rho_local_pairwise_flip` 从 `{base_row['rho_local_pairwise_flip']:.4f}` 变为 `{cur_row['rho_local_pairwise_flip']:.4f}`。"
    )


def figure_block(caption: str, absolute_path: str, output_dir: str) -> str:
    rel_path = os.path.relpath(absolute_path, output_dir)
    return f"**{caption}**\n\n![{caption}]({rel_path})"


def _yaml_dump(config: Dict) -> str:
    lines = []
    for key, value in config.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)
