"""生成或执行 on-policy DORL-DOSER 单步质量保护实验矩阵。"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


DEFAULT_ENV = "KuaiEnv-v0"
"""实验默认环境名称。"""

DEFAULT_READ_MESSAGE = "pointneg"
"""复现实验使用的用户模型读取标识。"""

DEFAULT_LAMBDA_ENTROPY = 0.05
"""与当前原始 DORL 对齐的熵奖励权重。"""

DEFAULT_LAMBDA_VARIANCE = 0.0
"""原始 DORL without variance 参考线使用的不确定性惩罚权重。"""

DEFAULT_CUDA = 7
"""默认使用的 CUDA 设备编号。"""

DEFAULT_EPOCH = 100
"""完整实验训练轮数。"""

DEFAULT_NUM_LEAVE_COMPUTE = 1
"""KuaiEnv 离开判断使用的最近反馈窗口大小。"""

DEFAULT_LEAVE_THRESHOLD = 0
"""KuaiEnv 离开判断阈值。"""

DEFAULT_TRACKER = "avg"
"""默认 state tracker 类型。"""

DEFAULT_REWARD_HANDLE = "cat"
"""默认 state tracker reward 拼接方式。"""

DEFAULT_WINDOW_SIZE = 3
"""默认 state tracker 历史窗口大小。"""

DEFAULT_TRAINING_NUM = 100
"""默认训练环境并行数。"""

DEFAULT_EPISODE_PER_COLLECT = 100
"""on-policy 每次采样 episode 数。"""

DEFAULT_REPEAT_PER_COLLECT = 1
"""每次采样后的更新轮数。"""

DEFAULT_DOSER_ETA = 1.5
"""与用户当前 eta=1.5 运行命令对齐的默认 DOSER eta。"""

DEFAULT_DIFFUSION_ARTIFACT_NAME = "DM_KuaiEnv-v0_small_data"
"""默认加载的小数据 diffusion artifact 名称。"""

DEFAULT_DORL_MODEL_NAME = "DORL"
"""原始 DORL 入口使用的模型名。"""

DEFAULT_DOSER_MODEL_NAME = "DORL_DOSER_ONPOLICY"
"""on-policy DORL-DOSER 入口使用的模型名。"""

DEFAULT_LOG_DIR = Path("./logs")
"""默认训练日志目录。"""

DEFAULT_STAGE1_SEEDS = (2023,)
"""Stage 1 单 seed 诊断使用的随机种子。"""

DEFAULT_STAGE2_SEEDS = (2023, 2024, 2025)
"""Stage 2 多 seed 确认使用的随机种子。"""

DORL_SCRIPT = "examples/advance/run_DORL.py"
"""原始 DORL 训练入口。"""

DOSER_ONPOLICY_SCRIPT = "examples/our_model/dorl_doser_onpolicy.py"
"""on-policy DORL-DOSER 训练入口。"""

PRIMARY_CTR_COLUMN = "NX_0_ctr"
"""Stage 2 选择配置时优先使用的单步质量指标列。"""

PRIMARY_RETURN_COLUMN = "NX_0_R_tra"
"""Stage 2 兜底选择配置时使用的轨迹奖励指标列。"""

DEFAULT_OUTPUT_DIR = Path("results/dorl_doser_onpolicy_quality_ablation")
"""默认命令清单输出目录。"""


@dataclass(frozen=True)
class ExperimentSpec:
    """描述一个 DORL 或 on-policy DORL-DOSER 实验配置。

    Attributes:
        experiment_id (str): 实验编号，例如 `C0` 或 `A7`。
        description (str): 实验目的说明。
        is_dorl (bool): 是否使用原始 DORL 入口。
        doser_beta (float): negative OOD penalty 系数。
        doser_lam (float): positive OOD compensation 系数。
        doser_eta (float): DOSER compensation target 系数。
        doser_aux_critic_coef (float): 辅助 Q/V 主损失权重。
        doser_detach_aux_state (bool): 是否阻断辅助 loss 对共享特征主干的梯度。
    """

    experiment_id: str
    description: str
    is_dorl: bool = False
    doser_beta: float = 0.001
    doser_lam: float = 0.001
    doser_eta: float = DEFAULT_DOSER_ETA
    doser_aux_critic_coef: float = 1.0
    doser_detach_aux_state: bool = False


def build_stage1_specs() -> List[ExperimentSpec]:
    """构建 Stage 1 单 seed 诊断实验配置。

    Args:
        无: Stage 1 配置固定来自实验计划。

    Returns:
        List[ExperimentSpec]: 按计划顺序排列的 Stage 1 实验配置。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    return [
        ExperimentSpec("D0", "原始 DORL without variance 同 seed 参考线", is_dorl=True),
        ExperimentSpec("C0", "当前 on-policy DOSER 对照配置"),
        ExperimentSpec(
            "A1",
            "只保留辅助 Q/V，关闭 OOD penalty 与 compensation",
            doser_beta=0.0,
            doser_lam=0.0,
        ),
        ExperimentSpec("A2", "关闭 positive compensation，检查 eta/vc 影响", doser_lam=0.0),
        ExperimentSpec("A3", "关闭 negative OOD penalty，检查 beta/reg 影响", doser_beta=0.0),
        ExperimentSpec("A4", "降低辅助 Q/V 权重到 0.1", doser_aux_critic_coef=0.1),
        ExperimentSpec("A5", "降低辅助 Q/V 权重到 0.05", doser_aux_critic_coef=0.05),
        ExperimentSpec("A6", "阻断辅助 loss 对共享特征主干的梯度", doser_detach_aux_state=True),
        ExperimentSpec(
            "A7",
            "辅助 Q/V 权重 0.1 且阻断辅助 loss 对共享特征主干的梯度",
            doser_aux_critic_coef=0.1,
            doser_detach_aux_state=True,
        ),
    ]


def build_spec_map() -> Dict[str, ExperimentSpec]:
    """构建实验编号到配置对象的映射。

    Args:
        无: 映射来自 Stage 1 完整配置集合。

    Returns:
        Dict[str, ExperimentSpec]: 以实验编号为键的配置映射。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    return {spec.experiment_id: spec for spec in build_stage1_specs()}


def load_stage1_summary(summary_path: Path) -> Dict[str, Mapping[str, float]]:
    """读取 Stage 1 指标汇总 CSV。

    Args:
        summary_path (Path): CSV 文件路径。文件必须包含 `experiment_id`、
            `NX_0_ctr` 与 `NX_0_R_tra` 等列。

    Returns:
        Dict[str, Mapping[str, float]]: 以实验编号为键、指标名到数值为值的映射。

    Raises:
        FileNotFoundError: 当 `summary_path` 不存在时抛出。
        ValueError: 当 CSV 缺少必要列或指标无法转为浮点数时抛出。
    """

    if not summary_path.exists():
        raise FileNotFoundError(f"Stage 1 summary does not exist: {summary_path}")

    summary: Dict[str, Mapping[str, float]] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        required_columns = {"experiment_id", PRIMARY_CTR_COLUMN, PRIMARY_RETURN_COLUMN}
        missing_columns = required_columns - fieldnames
        if missing_columns:
            raise ValueError(
                "Stage 1 summary is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            experiment_id = str(row["experiment_id"]).strip()
            if not experiment_id:
                continue
            metric_payload: Dict[str, float] = {}
            for metric_name, metric_value in row.items():
                if metric_name == "experiment_id" or metric_value in (None, ""):
                    continue
                try:
                    metric_payload[metric_name] = float(metric_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Metric {metric_name} for {experiment_id} is not a float: "
                        f"{metric_value}"
                    ) from exc
            summary[experiment_id] = metric_payload

    return summary


def select_best_experiment(
    summary: Mapping[str, Mapping[str, float]],
    candidate_ids: Sequence[str],
    metric_name: str,
) -> str:
    """按指定指标选择最优实验编号。

    Args:
        summary (Mapping[str, Mapping[str, float]]): Stage 1 指标汇总。
        candidate_ids (Sequence[str]): 候选实验编号列表。
        metric_name (str): 用于比较的指标名，数值越大越好。

    Returns:
        str: 指标最高的实验编号。

    Raises:
        ValueError: 当候选集合为空或缺少目标指标时抛出。
    """

    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty.")

    best_id: Optional[str] = None
    best_value = float("-inf")
    for experiment_id in candidate_ids:
        metric_payload = summary.get(experiment_id)
        if metric_payload is None or metric_name not in metric_payload:
            raise ValueError(f"Missing metric {metric_name} for experiment {experiment_id}.")
        metric_value = metric_payload[metric_name]
        if metric_value > best_value:
            best_id = experiment_id
            best_value = metric_value

    if best_id is None:
        raise ValueError(f"No valid candidate found for metric {metric_name}.")
    return best_id


def build_stage2_specs(stage1_summary: Mapping[str, Mapping[str, float]]) -> List[ExperimentSpec]:
    """根据 Stage 1 指标选择 Stage 2 多 seed 确认配置。

    Args:
        stage1_summary (Mapping[str, Mapping[str, float]]): Stage 1 指标汇总，
            至少包含 D0、C0、A1-A7 的主指标。

    Returns:
        List[ExperimentSpec]: Stage 2 需要多 seed 运行的配置。

    Raises:
        ValueError: 当 Stage 1 汇总缺少必要实验或指标时抛出。
    """

    spec_map = build_spec_map()
    selected_ids = ["D0", "C0", "A7"]

    coef_winner = select_best_experiment(
        stage1_summary,
        candidate_ids=["A4", "A5"],
        metric_name=PRIMARY_CTR_COLUMN,
    )
    component_winner = select_best_experiment(
        stage1_summary,
        candidate_ids=["A1", "A2", "A3"],
        metric_name=PRIMARY_CTR_COLUMN,
    )
    selected_ids.extend([coef_winner, component_winner])

    a6_ctr = stage1_summary.get("A6", {}).get(PRIMARY_CTR_COLUMN)
    c0_ctr = stage1_summary.get("C0", {}).get(PRIMARY_CTR_COLUMN)
    if a6_ctr is None or c0_ctr is None:
        raise ValueError("Stage 1 summary must include NX_0_ctr for A6 and C0.")

    if a6_ctr > c0_ctr:
        third_winner = "A6"
    else:
        remaining_candidates = [
            experiment_id
            for experiment_id in ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
            if experiment_id not in selected_ids
        ]
        third_winner = select_best_experiment(
            stage1_summary,
            candidate_ids=remaining_candidates,
            metric_name=PRIMARY_RETURN_COLUMN,
        )
    selected_ids.append(third_winner)

    # 去重保持顺序，避免动态选择与必跑项重复时生成冗余命令。
    deduplicated_ids = list(dict.fromkeys(selected_ids))
    return [spec_map[experiment_id] for experiment_id in deduplicated_ids]


def build_common_args(seed: int) -> List[str]:
    """构建 DORL 与 DOSER 共享的训练参数。

    Args:
        seed (int): 随机种子。

    Returns:
        List[str]: 可直接传给 Python 训练脚本的参数列表。

    Raises:
        ValueError: 当 `seed` 为负数时抛出。
    """

    if seed < 0:
        raise ValueError(f"seed must be non-negative: {seed}")

    return [
        "--env",
        DEFAULT_ENV,
        "--seed",
        str(seed),
        "--cuda",
        str(DEFAULT_CUDA),
        "--epoch",
        str(DEFAULT_EPOCH),
        "--num_leave_compute",
        str(DEFAULT_NUM_LEAVE_COMPUTE),
        "--leave_threshold",
        str(DEFAULT_LEAVE_THRESHOLD),
        "--which_tracker",
        DEFAULT_TRACKER,
        "--reward_handle",
        DEFAULT_REWARD_HANDLE,
        "--lambda_variance",
        str(DEFAULT_LAMBDA_VARIANCE),
        "--lambda_entropy",
        str(DEFAULT_LAMBDA_ENTROPY),
        "--window_size",
        str(DEFAULT_WINDOW_SIZE),
        "--read_message",
        DEFAULT_READ_MESSAGE,
        "--training-num",
        str(DEFAULT_TRAINING_NUM),
        "--episode-per-collect",
        str(DEFAULT_EPISODE_PER_COLLECT),
        "--repeat-per-collect",
        str(DEFAULT_REPEAT_PER_COLLECT),
    ]


def build_message(spec: ExperimentSpec, seed: int) -> str:
    """构建训练入口使用的 message 标识。

    Args:
        spec (ExperimentSpec): 单个实验配置。
        seed (int): 随机种子。

    Returns:
        str: 传给训练脚本 `--message` 的唯一标识。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    prefix = DEFAULT_DORL_MODEL_NAME if spec.is_dorl else DEFAULT_DOSER_MODEL_NAME
    return f"{prefix}_{spec.experiment_id}_seed{seed}"


def format_float_token(value: float) -> str:
    """把浮点数转换为适合日志文件名的短字符串。

    Args:
        value (float): 需要转换的浮点数。

    Returns:
        str: 去掉冗余尾零后的浮点数字符串。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    return f"{value:g}"


def build_log_path(spec: ExperimentSpec, seed: int) -> Path:
    """构建单个实验的 nohup 日志路径。

    Args:
        spec (ExperimentSpec): 单个实验配置。
        seed (int): 随机种子。

    Returns:
        Path: 训练 stdout/stderr 重定向日志路径。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    if spec.is_dorl:
        log_name = f"train_DORL_{DEFAULT_ENV}_{spec.experiment_id}_seed{seed}.log"
        return DEFAULT_LOG_DIR / log_name

    detach_token = "_detach" if spec.doser_detach_aux_state else ""
    log_name = (
        f"train_dorl_doser_onpolicy_{DEFAULT_ENV}_{spec.experiment_id}"
        f"_seed{seed}_eta{format_float_token(spec.doser_eta)}"
        f"_beta{format_float_token(spec.doser_beta)}"
        f"_lam{format_float_token(spec.doser_lam)}"
        f"_aux{format_float_token(spec.doser_aux_critic_coef)}"
        f"{detach_token}.log"
    )
    return DEFAULT_LOG_DIR / log_name


def build_command(
    spec: ExperimentSpec,
    seed: int,
    python_command: str,
    extra_args: Sequence[str],
) -> List[str]:
    """为单个实验配置构建命令行。

    Args:
        spec (ExperimentSpec): 实验配置。
        seed (int): 随机种子。
        python_command (str): Python 可执行命令，例如 `python`。
        extra_args (Sequence[str]): 追加到每条命令末尾的额外参数。

    Returns:
        List[str]: subprocess 可直接执行的命令参数列表。

    Raises:
        ValueError: 当 `python_command` 为空时抛出。
    """

    if not python_command:
        raise ValueError("python_command must not be empty.")

    script_path = DORL_SCRIPT if spec.is_dorl else DOSER_ONPOLICY_SCRIPT
    command = [python_command, script_path]
    command.extend(build_common_args(seed=seed))
    command.extend(
        [
            "--model_name",
            DEFAULT_DORL_MODEL_NAME if spec.is_dorl else DEFAULT_DOSER_MODEL_NAME,
            "--message",
            build_message(spec=spec, seed=seed),
        ]
    )

    if spec.is_dorl:
        return command + list(extra_args)
    else:
        command.extend(
            [
                "--diffusion_artifact_name",
                DEFAULT_DIFFUSION_ARTIFACT_NAME,
                "--doser_eta",
                str(spec.doser_eta),
                "--doser_beta",
                str(spec.doser_beta),
                "--doser_lam",
                str(spec.doser_lam),
                "--doser_aux_critic_coef",
                str(spec.doser_aux_critic_coef),
            ]
        )
        if spec.doser_detach_aux_state:
            command.append("--doser_detach_aux_state")

    command.extend(extra_args)
    return command


def format_shell_command(command: Sequence[str], log_path: Optional[Path] = None) -> str:
    """将命令参数列表格式化为可复制的 shell 命令。

    Args:
        command (Sequence[str]): subprocess 风格的命令参数。
        log_path (Optional[Path]): 若提供，则生成 `nohup ... > log 2>&1 &`
            形式的后台命令。

    Returns:
        str: 使用 shell quote 后的命令字符串。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    command_text = " ".join(shlex.quote(part) for part in command)
    if log_path is None:
        return command_text
    return f"nohup {command_text} > {shlex.quote(str(log_path))} 2>&1 &"


def build_run_items(
    specs: Sequence[ExperimentSpec],
    seeds: Sequence[int],
    python_command: str,
    extra_args: Sequence[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """构建命令清单条目。

    Args:
        specs (Sequence[ExperimentSpec]): 需要运行的实验配置。
        seeds (Sequence[int]): 每个配置需要使用的随机种子。
        python_command (str): Python 可执行命令。
        extra_args (Sequence[str]): 追加到所有命令末尾的额外参数。
        limit (int): 最多保留多少条命令，0 表示不限制。

    Returns:
        List[Dict[str, Any]]: 包含实验元信息与命令的清单条目。

    Raises:
        ValueError: 当 `limit` 为负数时抛出。
    """

    if limit < 0:
        raise ValueError(f"limit must be non-negative: {limit}")

    run_items: List[Dict[str, Any]] = []
    for spec in specs:
        for seed in seeds:
            command = build_command(
                spec=spec,
                seed=seed,
                python_command=python_command,
                extra_args=extra_args,
            )
            log_path = build_log_path(spec=spec, seed=seed)
            run_items.append(
                {
                    "experiment_id": spec.experiment_id,
                    "seed": seed,
                    "description": spec.description,
                    "log_path": str(log_path),
                    "command": command,
                    "shell_command": format_shell_command(command, log_path=log_path),
                }
            )

    if limit:
        return run_items[:limit]
    return run_items


def save_manifest(run_items: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """保存实验命令清单。

    Args:
        run_items (Sequence[Mapping[str, Any]]): 命令清单条目。
        output_path (Path): JSON 输出路径。

    Returns:
        None: 函数将清单写入 `output_path`。

    Raises:
        OSError: 当目录创建或文件写入失败时抛出。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(list(run_items), output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def execute_run_items(run_items: Sequence[Mapping[str, Any]]) -> None:
    """按顺序执行命令清单中的训练命令。

    Args:
        run_items (Sequence[Mapping[str, Any]]): 命令清单条目，每个条目必须包含
            `command` 字段。

    Returns:
        None: 子命令输出直接写到当前终端。

    Raises:
        subprocess.CalledProcessError: 当任一训练命令返回非零状态码时抛出。
        ValueError: 当清单条目缺少合法命令时抛出。
    """

    for run_item in run_items:
        command = run_item.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError(f"Invalid command in run item: {run_item}")
        log_path_value = run_item.get("log_path")
        if not isinstance(log_path_value, str) or not log_path_value:
            raise ValueError(f"Missing log_path in run item: {run_item}")
        log_path = Path(log_path_value)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Run] {run_item['experiment_id']} seed={run_item['seed']}")
        print(format_shell_command(command, log_path=log_path))
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=True)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Args:
        无: 参数直接来自命令行。

    Returns:
        argparse.Namespace: 解析后的命令行对象。

    Raises:
        RuntimeError: 当前实现不会主动抛出该异常。
    """

    parser = argparse.ArgumentParser(
        "Build or execute on-policy DORL-DOSER quality ablation commands"
    )
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2-core", "stage2"],
        default="stage1",
        help="stage2 需要 --stage1-summary；stage2-core 只生成 D0/C0/A7 多 seed 命令。",
    )
    parser.add_argument("--stage1-summary", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--python-command", type=str, default="python")
    parser.add_argument("--output-manifest", type=str, default=None)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="追加到每条训练命令末尾的参数，可重复使用，例如 --extra-arg=--cpu。",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def resolve_specs_and_seeds(
    args: argparse.Namespace,
) -> tuple[List[ExperimentSpec], Sequence[int]]:
    """根据 CLI 参数解析实验配置与随机种子。

    Args:
        args (argparse.Namespace): `parse_args` 返回的命名空间。

    Returns:
        tuple[List[ExperimentSpec], Sequence[int]]: 实验配置列表与随机种子列表。

    Raises:
        ValueError: 当 stage2 缺少 Stage 1 汇总文件时抛出。
    """

    if args.stage == "stage1":
        specs = build_stage1_specs()
        seeds = args.seeds or DEFAULT_STAGE1_SEEDS
        return specs, seeds

    if args.stage == "stage2-core":
        spec_map = build_spec_map()
        specs = [spec_map[experiment_id] for experiment_id in ["D0", "C0", "A7"]]
        seeds = args.seeds or DEFAULT_STAGE2_SEEDS
        return specs, seeds

    if args.stage1_summary is None:
        raise ValueError("--stage stage2 requires --stage1-summary.")
    summary = load_stage1_summary(Path(args.stage1_summary))
    specs = build_stage2_specs(summary)
    seeds = args.seeds or DEFAULT_STAGE2_SEEDS
    return specs, seeds


def main() -> None:
    """执行命令清单生成或训练启动流程。

    Args:
        无: 参数直接来自命令行。

    Returns:
        None: 默认只打印并保存命令清单；传入 `--execute` 时顺序执行训练。

    Raises:
        ValueError: 当 CLI 参数组合非法时抛出。
        OSError: 当命令清单写入失败时抛出。
        subprocess.CalledProcessError: 当 `--execute` 下训练命令失败时抛出。
    """

    args = parse_args()
    specs, seeds = resolve_specs_and_seeds(args)
    output_manifest = (
        Path(args.output_manifest)
        if args.output_manifest
        else DEFAULT_OUTPUT_DIR / f"{args.stage}_manifest.json"
    )
    run_items = build_run_items(
        specs=specs,
        seeds=seeds,
        python_command=args.python_command,
        extra_args=args.extra_arg,
        limit=args.limit,
    )
    save_manifest(run_items, output_manifest)

    print(f"[Manifest] saved to {output_manifest}")
    for run_item in run_items:
        print(run_item["shell_command"])

    if args.execute:
        execute_run_items(run_items)


if __name__ == "__main__":
    main()
