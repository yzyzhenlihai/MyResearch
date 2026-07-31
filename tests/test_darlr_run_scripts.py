"""DARLR 分阶段运行、调参与 dry-run shell 契约测试。"""

import os
import re
import subprocess
import unittest
from pathlib import Path
from typing import Mapping, Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""仓库根目录。"""

SCRIPT_DIRECTORY = REPOSITORY_ROOT / "script"
"""DARLR shell launcher 所在目录。"""

PIPELINE_SCRIPT = SCRIPT_DIRECTORY / "run_DARLR_KuaiRec_pipeline.sh"
"""按诊断顺序运行消融实验的脚本。"""

TUNE_SCRIPT = SCRIPT_DIRECTORY / "run_DARLR_KuaiRec_tune.sh"
"""逐组搜索论文超参数的脚本。"""

SUBPROCESS_TIMEOUT_SECONDS = 20
"""dry-run 子进程允许的最大执行时间。"""

DRY_RUN_BASE_ENVIRONMENT = {
    "DRY_RUN": "1",
    "SWANLAB_MODE": "disabled",
    "PYTHON_BIN": "/bin/false",
    "CUDA": "0",
    "SEEDS": "2023",
    "EPOCH": "1",
    "STEP_PER_EPOCH": "1",
    "EPISODE_PER_COLLECT": "1",
    "TRAINING_NUM": "1",
    "TEST_NUM": "2",
}
"""阻止真实训练并缩小命令规模的通用环境变量。"""


def run_shell_script(
    script_path: Path,
    extra_environment: Optional[Mapping[str, str]] = None,
    extra_arguments: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    """以 dry-run 方式执行一个 DARLR shell 脚本。

    Args:
        script_path (Path): 待执行脚本的绝对路径。
        extra_environment (Mapping[str, str], optional): 覆盖或补充的环境变量。
        extra_arguments (tuple[str, ...]): 追加到脚本后的命令行参数。

    Returns:
        subprocess.CompletedProcess: 捕获 stdout/stderr 的执行结果。

    Raises:
        subprocess.TimeoutExpired: 脚本未在限定时间内结束时抛出。
    """

    environment = os.environ.copy()
    environment.update(DRY_RUN_BASE_ENVIRONMENT)
    if extra_environment:
        environment.update(extra_environment)
    return subprocess.run(
        ["bash", str(script_path), *extra_arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )


def combined_output(result: subprocess.CompletedProcess) -> str:
    """合并 shell dry-run 的标准输出和标准错误。

    Args:
        result (subprocess.CompletedProcess): shell 子进程结果。

    Returns:
        str: 用换行连接的完整可诊断输出。
    """

    return "\n".join(part for part in (result.stdout, result.stderr) if part)


class DarlrShellSyntaxTest(unittest.TestCase):
    """验证所有 DARLR launcher 的 shell 语法与 dry-run 基础能力。"""

    def test_all_darlr_shell_scripts_have_valid_syntax(self) -> None:
        """验证 `script/run_DARLR_*.sh` 均可通过 `bash -n`。"""

        script_paths = sorted(SCRIPT_DIRECTORY.glob("run_DARLR_*.sh"))
        self.assertTrue(script_paths, "未发现 DARLR shell launcher。")

        result = subprocess.run(
            ["bash", "-n", *(str(path) for path in script_paths)],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )

        self.assertEqual(result.returncode, 0, combined_output(result))

    def test_all_darlr_launchers_support_dry_run(self) -> None:
        """验证每个 launcher 在 `/bin/false` 作为 Python 时仍只打印命令。"""

        script_paths = sorted(SCRIPT_DIRECTORY.glob("run_DARLR_*.sh"))
        self.assertTrue(script_paths, "未发现 DARLR shell launcher。")
        for script_path in script_paths:
            extra_environment = {}
            if script_path == PIPELINE_SCRIPT:
                extra_environment["STAGE"] = "baseline"
            if script_path == TUNE_SCRIPT:
                extra_environment["TUNE_TARGET"] = "k"
            with self.subTest(script=script_path.name):
                result = run_shell_script(
                    script_path,
                    extra_environment=extra_environment,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    combined_output(result),
                )
                self.assertTrue(
                    combined_output(result).strip(),
                    f"{script_path.name} dry-run 未输出计划命令。",
                )

    def test_reproduce_launcher_forwards_extra_arguments(self) -> None:
        """验证数据集 launcher 传入的覆盖参数不会在 reproduce 层丢失。"""

        sentinel_message = "DARLR_forwarded_argument_sentinel"
        result = run_shell_script(
            SCRIPT_DIRECTORY / "run_DARLR_reproduce.sh",
            extra_arguments=("--message", sentinel_message),
        )
        output = combined_output(result)

        self.assertEqual(result.returncode, 0, output)
        self.assertIn(sentinel_message, output)


class DarlrPipelineDryRunTest(unittest.TestCase):
    """验证 pipeline 的阶段枚举和关键实验语义。"""

    def test_each_pipeline_stage_dry_runs_without_data(self) -> None:
        """验证五个约定阶段均能在无模型加载的情况下生成命令。"""

        stage_markers = {
            "baseline": (r"run_DORL\.py",),
            "causal": (r"static_dorl", r"reference_mean"),
            "stability": (
                r"selector[-_]ent[-_]coef",
                r"selector[-_]reward[-_]normalization",
                r"selector[-_]advantage[-_]normalization",
                r"max[-_]grad[-_]norm",
            ),
            "paper": (
                r"selector[-_]k[ =]+40",
                r"selector[-_]lambda[-_]s[ =]+2(?:\.0)?",
                r"selector[-_]lambda[-_]d[ =]+0\.1",
                r"lambda[-_]uncertainty[ =]+0\.1",
                r"lambda[-_]entropy[ =]+0\.1",
            ),
            "all": (r"run_DORL\.py", r"static_dorl"),
        }

        for stage, expected_patterns in stage_markers.items():
            with self.subTest(stage=stage):
                result = run_shell_script(
                    PIPELINE_SCRIPT,
                    extra_environment={"STAGE": stage},
                )
                output = combined_output(result)
                self.assertEqual(result.returncode, 0, output)
                for expected_pattern in expected_patterns:
                    self.assertRegex(output, expected_pattern)
                self.assertRegex(output, r"best[-_]metric[ =]+NX_0(?:\s|$)")
                self.assertRegex(output, r"(?:--is_save|--is-save)")

    def test_invalid_pipeline_stage_fails_before_training(self) -> None:
        """验证未知阶段会返回非零状态而不是退化为全量训练。"""

        result = run_shell_script(
            PIPELINE_SCRIPT,
            extra_environment={"STAGE": "unknown-stage"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(
            combined_output(result).lower(),
            r"(unsupported|invalid|unknown|stage)",
        )

    def test_parallel_pipeline_assigns_distinct_cuda_devices(self) -> None:
        """验证并行 baseline 会把两个训练任务分配到不同 GPU。

        该测试仅执行 dry-run，不会启动 Python 或访问真实 CUDA 设备。
        """

        result = run_shell_script(
            PIPELINE_SCRIPT,
            extra_environment={
                "STAGE": "baseline",
                "PARALLEL_JOBS": "2",
                "CUDA_DEVICES": "2 3",
            },
        )
        output = combined_output(result)

        self.assertEqual(result.returncode, 0, output)
        self.assertRegex(output, r"LAUNCH: baseline/dorl_control on CUDA=2")
        self.assertRegex(
            output,
            r"LAUNCH: baseline/darlr_static_dorl on CUDA=3",
        )
        self.assertRegex(output, r"--cuda[ =]+2(?:\s|$)")
        self.assertRegex(output, r"--cuda[ =]+3(?:\s|$)")
        self.assertEqual(
            len(re.findall(r"--num_leave_compute[ =]+3(?:\s|$)", output)),
            2,
        )

    def test_parallel_jobs_cannot_exceed_cuda_device_count(self) -> None:
        """验证并发数大于 GPU 池时会在启动训练前明确失败。"""

        result = run_shell_script(
            PIPELINE_SCRIPT,
            extra_environment={
                "STAGE": "baseline",
                "PARALLEL_JOBS": "2",
                "CUDA_DEVICES": "0",
            },
        )
        output = combined_output(result)

        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(
            output,
            r"PARALLEL_JOBS=2 exceeds CUDA device count=1",
        )

    def test_parallel_child_failure_stops_pipeline(self) -> None:
        """验证并行子任务失败会向上传播非零退出码。"""

        result = run_shell_script(
            PIPELINE_SCRIPT,
            extra_environment={
                "STAGE": "baseline",
                "DRY_RUN": "0",
                "PARALLEL_JOBS": "2",
                "CUDA_DEVICES": "0 1",
                "PYTHON_BIN": "/bin/false",
                "SAVE_MODEL": "0",
            },
        )
        output = combined_output(result)

        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(output, r"FAILED\(1\): baseline/")


class DarlrTuneDryRunTest(unittest.TestCase):
    """验证单参数调优脚本的目标枚举与 seed 展开。"""

    def test_each_tune_target_generates_matching_commands(self) -> None:
        """验证八个调参目标均映射到对应训练参数。"""

        target_patterns = {
            "k": r"selector[-_]k",
            "lambda_s": r"selector[-_]lambda[-_]s",
            "lambda_d": r"selector[-_]lambda[-_]d",
            "lambda_u": r"lambda[-_]uncertainty",
            "lambda_e": r"lambda[-_]entropy",
            "layers": r"selector[-_]num[-_]layers",
            "heads": r"selector[-_]num[-_]heads",
            "window": r"window[-_]size",
        }

        for tune_target, expected_pattern in target_patterns.items():
            with self.subTest(tune_target=tune_target):
                result = run_shell_script(
                    TUNE_SCRIPT,
                    extra_environment={"TUNE_TARGET": tune_target},
                )
                output = combined_output(result)
                self.assertEqual(result.returncode, 0, output)
                self.assertRegex(output, expected_pattern)
                self.assertRegex(output, r"best[-_]metric[ =]+NX_0(?:\s|$)")
                self.assertRegex(output, r"seed[ =]+2023")

    def test_invalid_tune_target_fails_before_training(self) -> None:
        """验证未知调参目标在命令展开前明确失败。"""

        result = run_shell_script(
            TUNE_SCRIPT,
            extra_environment={"TUNE_TARGET": "unknown-target"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(
            combined_output(result).lower(),
            r"(unsupported|invalid|unknown|target)",
        )


if __name__ == "__main__":
    unittest.main()
