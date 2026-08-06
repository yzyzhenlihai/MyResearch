"""DARLR 分阶段运行、调参与 dry-run shell 契约测试。"""

import fcntl
import os
import re
import signal
import subprocess
import tempfile
import time
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

MOCK_TRAINING_DURATION_SECONDS = 30
"""信号清理测试中 mock Python 的最大自然运行时间。"""

SIGNAL_CHILD_START_TIMEOUT_SECONDS = 5
"""等待 mock Python 启动的最长时间。"""

SIGNAL_PIPELINE_EXIT_TIMEOUT_SECONDS = 10
"""pipeline 收到 TERM 后允许的最长退出时间。"""

SIGNAL_PROCESS_EXIT_TIMEOUT_SECONDS = 2
"""确认 mock Python PID 消失的最长等待时间。"""

SIGNAL_FORCE_CLEANUP_TIMEOUT_SECONDS = 5
"""信号测试异常分支强制清理的最长等待时间。"""

SIGNAL_POLL_INTERVAL_SECONDS = 0.05
"""信号清理测试的轮询间隔。"""

DRY_RUN_BASE_ENVIRONMENT = {
    "DRY_RUN": "1",
    "SWANLAB_MODE": "disabled",
    "PYTHON_BIN": "/bin/false",
    "CUDA": "0",
    "CUDA_DEVICES": "0",
    "PARALLEL_JOBS": "1",
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
    extra_environment: Optional[Mapping[str, Optional[str]]] = None,
    extra_arguments: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    """以 dry-run 方式执行一个 DARLR shell 脚本。

    Args:
        script_path (Path): 待执行脚本的绝对路径。
        extra_environment (Mapping[str, Optional[str]], optional): 覆盖或删除的环境变量。
        extra_arguments (tuple[str, ...]): 追加到脚本后的命令行参数。

    Returns:
        subprocess.CompletedProcess: 捕获 stdout/stderr 的执行结果。

    Raises:
        subprocess.TimeoutExpired: 脚本未在限定时间内结束时抛出。
    """

    environment = os.environ.copy()
    environment.update(DRY_RUN_BASE_ENVIRONMENT)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if extra_environment:
        for variable_name, variable_value in extra_environment.items():
            if variable_value is None:
                environment.pop(variable_name, None)
            else:
                environment[variable_name] = variable_value
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


def create_mock_nvidia_smi(directory: Path) -> Path:
    """创建可由环境变量控制输出与退出码的 nvidia-smi 替身。

    Args:
        directory (Path): 用于放置可执行替身脚本的临时目录。

    Returns:
        Path: 已设置执行权限的替身脚本路径。

    Raises:
        OSError: 替身脚本无法写入或修改权限时抛出。
    """

    mock_path = directory / "nvidia-smi"
    mock_path.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$#\" -ne 2 || \"${1:-}\" != "
        "\"--query-gpu=index,memory.total,memory.free,utilization.gpu\" "
        "|| \"${2:-}\" != \"--format=csv,noheader,nounits\" ]]; then\n"
        "  printf 'unexpected nvidia-smi arguments: %s\\n' \"$*\" >&2\n"
        "  exit 64\n"
        "fi\n"
        "if [[ \"${MOCK_NVIDIA_SMI_STATUS:-0}\" -ne 0 ]]; then\n"
        "  exit \"${MOCK_NVIDIA_SMI_STATUS}\"\n"
        "fi\n"
        "printf '%s\\n' \"${MOCK_NVIDIA_SMI_OUTPUT:-}\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    mock_path.chmod(0o755)
    return mock_path


def gpu_lock_is_available(lock_path: Path) -> bool:
    """检查指定 GPU 文件锁能否被当前进程非阻塞获取。

    Args:
        lock_path (Path): pipeline 使用的 GPU lock 文件路径。

    Returns:
        bool: 可立即获取时为 True，仍被其他进程持有时为 False。

    Raises:
        OSError: 锁文件无法创建或访问时抛出。
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            return False
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return True


def create_sleeping_python(directory: Path) -> Path:
    """创建记录 PID 后长时间运行的 Python 命令替身。

    Args:
        directory (Path): 用于放置可执行替身脚本的临时目录。

    Returns:
        Path: 已设置执行权限的替身脚本路径。

    Raises:
        OSError: 替身脚本无法写入或修改权限时抛出。
    """

    mock_path = directory / "sleeping-python"
    mock_path.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$$\" > \"${MOCK_PYTHON_PID_FILE}\"\n"
        f"exec sleep {MOCK_TRAINING_DURATION_SECONDS}\n",
        encoding="utf-8",
    )
    mock_path.chmod(0o755)
    return mock_path


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

    def test_auto_parallelism_selects_only_memory_eligible_devices(self) -> None:
        """验证自动模式按空闲显存过滤 GPU 并并行分配。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            mock_nvidia_smi = create_mock_nvidia_smi(temporary_path)
            result = run_shell_script(
                PIPELINE_SCRIPT,
                extra_environment={
                    "STAGE": "baseline",
                    "PARALLEL_JOBS": "auto",
                    "CUDA_DEVICES": "0 1 2",
                    "GPU_MIN_FREE_MEMORY_MB": "12000",
                    "NVIDIA_SMI_BIN": str(mock_nvidia_smi),
                    "GPU_LOCK_DIR": str(temporary_path / "locks"),
                    "MOCK_NVIDIA_SMI_OUTPUT": (
                        "0, 16376, 15000, 10\n"
                        "1, 16376, 4096, 5\n"
                        "2, 16376, 12000, 20"
                    ),
                },
            )
        output = combined_output(result)

        self.assertEqual(result.returncode, 0, output)
        self.assertRegex(output, r"initially_eligible=2, PARALLEL_JOBS=2")
        self.assertRegex(output, r"AUTO_GPU_ASSIGN: GPU0 free=15000 MiB")
        self.assertRegex(output, r"AUTO_GPU_ASSIGN: GPU2 free=12000 MiB")
        self.assertRegex(output, r"LAUNCH: baseline/dorl_control on CUDA=0")
        self.assertRegex(
            output,
            r"LAUNCH: baseline/darlr_static_dorl on CUDA=2",
        )
        self.assertNotRegex(output, r"LAUNCH: .* on CUDA=1(?:\s|$)")

    def test_auto_parallelism_honors_maximum_job_cap(self) -> None:
        """验证自动模式可用上限进一步降低并发数。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            mock_nvidia_smi = create_mock_nvidia_smi(temporary_path)
            result = run_shell_script(
                PIPELINE_SCRIPT,
                extra_environment={
                    "STAGE": "baseline",
                    "PARALLEL_JOBS": "auto",
                    "CUDA_DEVICES": "4 5",
                    "GPU_MIN_FREE_MEMORY_MB": "12000",
                    "AUTO_PARALLEL_MAX_JOBS": "1",
                    "NVIDIA_SMI_BIN": str(mock_nvidia_smi),
                    "GPU_LOCK_DIR": str(temporary_path / "locks"),
                    "MOCK_NVIDIA_SMI_OUTPUT": (
                        "4, 16376, 14000, 20\n"
                        "5, 16376, 15000, 30"
                    ),
                },
            )
            lock_released = gpu_lock_is_available(
                temporary_path / "locks" / "gpu-5.lock",
            )
        output = combined_output(result)

        self.assertEqual(result.returncode, 0, output)
        self.assertTrue(lock_released, "pipeline 正常退出后 GPU5 锁未释放。")
        self.assertRegex(output, r"initially_eligible=2, PARALLEL_JOBS=1")
        self.assertRegex(output, r"LAUNCH: baseline/dorl_control on CUDA=5")
        self.assertRegex(
            output,
            r"LAUNCH: baseline/darlr_static_dorl on CUDA=5",
        )

    def test_auto_parallelism_fails_when_no_device_has_enough_memory(self) -> None:
        """验证没有 GPU 满足显存门槛时不会启动训练。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            mock_nvidia_smi = create_mock_nvidia_smi(temporary_path)
            result = run_shell_script(
                PIPELINE_SCRIPT,
                extra_environment={
                    "STAGE": "baseline",
                    "PARALLEL_JOBS": "auto",
                    "CUDA_DEVICES": "0 1",
                    "GPU_MIN_FREE_MEMORY_MB": "15000",
                    "NVIDIA_SMI_BIN": str(mock_nvidia_smi),
                    "GPU_LOCK_DIR": str(temporary_path / "locks"),
                    "MOCK_NVIDIA_SMI_OUTPUT": (
                        "0, 16376, 14000, 0\n"
                        "1, 16376, 13000, 0"
                    ),
                },
            )
        output = combined_output(result)

        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(output, r"No GPU .* free memory >= 15000 MiB")
        self.assertNotIn("LAUNCH:", output)

    def test_auto_parallelism_discovers_default_pool_from_snapshot(self) -> None:
        """验证未显式指定设备池时不依赖硬编码的八卡索引。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            mock_nvidia_smi = create_mock_nvidia_smi(temporary_path)
            result = run_shell_script(
                PIPELINE_SCRIPT,
                extra_environment={
                    "STAGE": "baseline",
                    "PARALLEL_JOBS": "auto",
                    "CUDA": None,
                    "CUDA_DEVICES": None,
                    "GPU_MIN_FREE_MEMORY_MB": "12000",
                    "NVIDIA_SMI_BIN": str(mock_nvidia_smi),
                    "GPU_LOCK_DIR": str(temporary_path / "locks"),
                    "MOCK_NVIDIA_SMI_OUTPUT": (
                        "0, 16376, 15000, 10\n"
                        "1, 16376, 14000, 20"
                    ),
                },
            )
        output = combined_output(result)

        self.assertEqual(result.returncode, 0, output)
        self.assertRegex(output, r"candidate CUDA devices \[0,1\]")
        self.assertRegex(output, r"LAUNCH: baseline/dorl_control on CUDA=0")
        self.assertRegex(
            output,
            r"LAUNCH: baseline/darlr_static_dorl on CUDA=1",
        )

    def test_auto_parallelism_limits_single_paper_target_to_one_gpu(self) -> None:
        """验证单个 paper target 不会预占多余 GPU。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            mock_nvidia_smi = create_mock_nvidia_smi(temporary_path)
            result = run_shell_script(
                PIPELINE_SCRIPT,
                extra_environment={
                    "STAGE": "paper",
                    "PAPER_TARGETS": "k",
                    "PARALLEL_JOBS": "auto",
                    "CUDA_DEVICES": "0 1 2 3",
                    "GPU_MIN_FREE_MEMORY_MB": "12000",
                    "NVIDIA_SMI_BIN": str(mock_nvidia_smi),
                    "GPU_LOCK_DIR": str(temporary_path / "locks"),
                    "MOCK_NVIDIA_SMI_OUTPUT": (
                        "0, 16376, 15000, 0\n"
                        "1, 16376, 15000, 0\n"
                        "2, 16376, 15000, 0\n"
                        "3, 16376, 15000, 0"
                    ),
                },
            )
        output = combined_output(result)

        self.assertEqual(result.returncode, 0, output)
        self.assertRegex(output, r"initially_eligible=4, PARALLEL_JOBS=1")
        self.assertEqual(output.count("AUTO_GPU_ASSIGN:"), 1)

    def test_auto_parallelism_skips_a_gpu_locked_by_another_pipeline(self) -> None:
        """验证首选 GPU 被锁定时自动改用下一张卡。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            lock_directory = temporary_path / "locks"
            lock_directory.mkdir()
            locked_file_path = lock_directory / "gpu-0.lock"
            with locked_file_path.open("a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                mock_nvidia_smi = create_mock_nvidia_smi(temporary_path)
                result = run_shell_script(
                    PIPELINE_SCRIPT,
                    extra_environment={
                        "STAGE": "baseline",
                        "PARALLEL_JOBS": "auto",
                        "CUDA_DEVICES": "0 1",
                        "GPU_MIN_FREE_MEMORY_MB": "12000",
                        "NVIDIA_SMI_BIN": str(mock_nvidia_smi),
                        "GPU_LOCK_DIR": str(lock_directory),
                        "MOCK_NVIDIA_SMI_OUTPUT": (
                            "0, 16376, 15000, 0\n"
                            "1, 16376, 14000, 0"
                        ),
                    },
                )
        output = combined_output(result)

        self.assertEqual(result.returncode, 0, output)
        self.assertIn("GPU0 is locked by another DARLR pipeline", output)
        self.assertNotRegex(output, r"LAUNCH: .* on CUDA=0(?:\s|$)")
        self.assertRegex(output, r"LAUNCH: baseline/dorl_control on CUDA=1")

    def test_auto_parallelism_fails_when_all_eligible_gpus_are_locked(self) -> None:
        """验证所有合格 GPU 均被其他 pipeline 锁定时不启动训练。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            lock_directory = temporary_path / "locks"
            lock_directory.mkdir()
            locked_file_path = lock_directory / "gpu-0.lock"
            with locked_file_path.open("a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                mock_nvidia_smi = create_mock_nvidia_smi(temporary_path)
                result = run_shell_script(
                    PIPELINE_SCRIPT,
                    extra_environment={
                        "STAGE": "baseline",
                        "PARALLEL_JOBS": "auto",
                        "CUDA_DEVICES": "0",
                        "GPU_MIN_FREE_MEMORY_MB": "12000",
                        "NVIDIA_SMI_BIN": str(mock_nvidia_smi),
                        "GPU_LOCK_DIR": str(lock_directory),
                        "MOCK_NVIDIA_SMI_OUTPUT": (
                            "0, 16376, 15000, 0"
                        ),
                    },
                )
        output = combined_output(result)

        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("LAUNCH:", output)

    def test_auto_parallelism_rejects_ambiguous_visible_device_mapping(self) -> None:
        """验证 CUDA_VISIBLE_DEVICES 为空或重映射时均安全失败。"""

        for visible_devices in ("", "4"):
            with self.subTest(cuda_visible_devices=visible_devices):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    temporary_path = Path(temporary_directory)
                    mock_nvidia_smi = create_mock_nvidia_smi(temporary_path)
                    result = run_shell_script(
                        PIPELINE_SCRIPT,
                        extra_environment={
                            "STAGE": "baseline",
                            "PARALLEL_JOBS": "auto",
                            "CUDA_DEVICES": "0",
                            "CUDA_VISIBLE_DEVICES": visible_devices,
                            "NVIDIA_SMI_BIN": str(mock_nvidia_smi),
                            "GPU_LOCK_DIR": str(temporary_path / "locks"),
                            "MOCK_NVIDIA_SMI_OUTPUT": (
                                "0, 16376, 15000, 0"
                            ),
                        },
                    )
                output = combined_output(result)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("cannot safely map physical GPU indexes", output)
                self.assertNotIn("LAUNCH:", output)

    def test_auto_parallelism_rejects_invalid_gpu_snapshots(self) -> None:
        """验证空、畸形、重复和缺卡快照均不会启动训练。"""

        invalid_snapshots = {
            "empty": ("", "0"),
            "malformed": ("0, 16376, N/A, 0", "0"),
            "duplicate": (
                "0, 16376, 15000, 0\n0, 16376, 15000, 0",
                "0",
            ),
            "missing": ("0, 16376, 15000, 0", "1"),
        }
        for case_name, (snapshot, cuda_devices) in invalid_snapshots.items():
            with self.subTest(case=case_name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    temporary_path = Path(temporary_directory)
                    mock_nvidia_smi = create_mock_nvidia_smi(temporary_path)
                    result = run_shell_script(
                        PIPELINE_SCRIPT,
                        extra_environment={
                            "STAGE": "baseline",
                            "PARALLEL_JOBS": "auto",
                            "CUDA_DEVICES": cuda_devices,
                            "GPU_MIN_FREE_MEMORY_MB": "12000",
                            "NVIDIA_SMI_BIN": str(mock_nvidia_smi),
                            "GPU_LOCK_DIR": str(temporary_path / "locks"),
                            "MOCK_NVIDIA_SMI_OUTPUT": snapshot,
                        },
                    )
                output = combined_output(result)

                self.assertNotEqual(result.returncode, 0, output)
                self.assertNotIn("LAUNCH:", output)

    def test_auto_parallelism_propagates_gpu_probe_failure(self) -> None:
        """验证 nvidia-smi 失败时给出明确错误且不启动训练。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            mock_nvidia_smi = create_mock_nvidia_smi(temporary_path)
            result = run_shell_script(
                PIPELINE_SCRIPT,
                extra_environment={
                    "STAGE": "baseline",
                    "PARALLEL_JOBS": "auto",
                    "CUDA_DEVICES": "0",
                    "NVIDIA_SMI_BIN": str(mock_nvidia_smi),
                    "GPU_LOCK_DIR": str(temporary_path / "locks"),
                    "MOCK_NVIDIA_SMI_STATUS": "42",
                },
            )
        output = combined_output(result)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to query GPU memory", output)
        self.assertNotIn("LAUNCH:", output)

    def test_manual_parallelism_bypasses_gpu_probe(self) -> None:
        """验证显式整数并发模式不依赖 nvidia-smi。"""

        result = run_shell_script(
            PIPELINE_SCRIPT,
            extra_environment={
                "STAGE": "baseline",
                "PARALLEL_JOBS": "1",
                "CUDA_DEVICES": "7",
                "NVIDIA_SMI_BIN": "/bin/false",
            },
        )
        output = combined_output(result)

        self.assertEqual(result.returncode, 0, output)
        self.assertNotIn("AUTO_GPU:", output)
        self.assertRegex(output, r"LAUNCH: baseline/dorl_control on CUDA=7")

    def test_duplicate_cuda_devices_fail_before_training(self) -> None:
        """验证重复 GPU 索引不会被误当作两个并发设备。"""

        result = run_shell_script(
            PIPELINE_SCRIPT,
            extra_environment={
                "STAGE": "baseline",
                "PARALLEL_JOBS": "2",
                "CUDA_DEVICES": "0 0",
            },
        )
        output = combined_output(result)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate GPU index '0'", output)
        self.assertNotIn("LAUNCH:", output)

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

    def test_auto_child_failure_releases_gpu_lock(self) -> None:
        """验证自动模式子任务失败后仍释放 GPU 锁。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            mock_nvidia_smi = create_mock_nvidia_smi(temporary_path)
            result = run_shell_script(
                PIPELINE_SCRIPT,
                extra_environment={
                    "STAGE": "baseline",
                    "DRY_RUN": "0",
                    "PARALLEL_JOBS": "auto",
                    "CUDA_DEVICES": "0",
                    "GPU_MIN_FREE_MEMORY_MB": "12000",
                    "NVIDIA_SMI_BIN": str(mock_nvidia_smi),
                    "GPU_LOCK_DIR": str(temporary_path / "locks"),
                    "MOCK_NVIDIA_SMI_OUTPUT": "0, 16376, 15000, 0",
                    "PYTHON_BIN": "/bin/false",
                    "SAVE_MODEL": "0",
                },
            )
            lock_released = gpu_lock_is_available(
                temporary_path / "locks" / "gpu-0.lock",
            )
        output = combined_output(result)

        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(output, r"FAILED\(1\): baseline/")
        self.assertTrue(lock_released, "子任务失败后 GPU0 锁未释放。")

    def test_sigterm_stops_training_process_group_and_releases_lock(self) -> None:
        """验证 TERM 会停止 paper 训练进程树并释放 GPU 锁。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            mock_nvidia_smi = create_mock_nvidia_smi(temporary_path)
            sleeping_python = create_sleeping_python(temporary_path)
            python_pid_path = temporary_path / "python.pid"
            environment = os.environ.copy()
            environment.update(DRY_RUN_BASE_ENVIRONMENT)
            environment.pop("CUDA_VISIBLE_DEVICES", None)
            environment.update(
                {
                    "STAGE": "paper",
                    "PAPER_TARGETS": "k",
                    "K_VALUES": "10",
                    "DRY_RUN": "0",
                    "PARALLEL_JOBS": "auto",
                    "CUDA_DEVICES": "0",
                    "GPU_MIN_FREE_MEMORY_MB": "12000",
                    "NVIDIA_SMI_BIN": str(mock_nvidia_smi),
                    "GPU_LOCK_DIR": str(temporary_path / "locks"),
                    "MOCK_NVIDIA_SMI_OUTPUT": "0, 16376, 15000, 0",
                    "MOCK_PYTHON_PID_FILE": str(python_pid_path),
                    "PYTHON_BIN": str(sleeping_python),
                    "SAVE_MODEL": "0",
                }
            )
            pipeline_process = subprocess.Popen(
                ["bash", str(PIPELINE_SCRIPT)],
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            python_pid = None
            python_process_group = None
            stdout = ""
            stderr = ""
            process_gone = False
            lock_released = False
            try:
                deadline = (
                    time.monotonic() + SIGNAL_CHILD_START_TIMEOUT_SECONDS
                )
                while (
                    not python_pid_path.exists()
                    and pipeline_process.poll() is None
                    and time.monotonic() < deadline
                ):
                    time.sleep(SIGNAL_POLL_INTERVAL_SECONDS)

                self.assertTrue(
                    python_pid_path.exists(),
                    "未在时限内启动测试用 Python 子进程。",
                )
                python_pid = int(
                    python_pid_path.read_text(encoding="utf-8").strip()
                )
                python_process_group = os.getpgid(python_pid)
                pipeline_process.terminate()
                try:
                    stdout, stderr = pipeline_process.communicate(
                        timeout=SIGNAL_PIPELINE_EXIT_TIMEOUT_SECONDS,
                    )
                except subprocess.TimeoutExpired:
                    if python_process_group != os.getpgrp():
                        try:
                            os.killpg(python_process_group, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    pipeline_process.kill()
                    stdout, stderr = pipeline_process.communicate(
                        timeout=SIGNAL_FORCE_CLEANUP_TIMEOUT_SECONDS,
                    )
                    self.fail(
                        "pipeline 收到 TERM 后未在 "
                        f"{SIGNAL_PIPELINE_EXIT_TIMEOUT_SECONDS} 秒内退出。\n"
                        f"{stdout}\n{stderr}"
                    )

                deadline = (
                    time.monotonic() + SIGNAL_PROCESS_EXIT_TIMEOUT_SECONDS
                )
                while time.monotonic() < deadline:
                    try:
                        os.kill(python_pid, 0)
                    except ProcessLookupError:
                        process_gone = True
                        break
                    time.sleep(SIGNAL_POLL_INTERVAL_SECONDS)
                lock_released = gpu_lock_is_available(
                    temporary_path / "locks" / "gpu-0.lock",
                )
            finally:
                if pipeline_process.poll() is None:
                    pipeline_process.terminate()
                    try:
                        pipeline_process.communicate(
                            timeout=SIGNAL_FORCE_CLEANUP_TIMEOUT_SECONDS,
                        )
                    except subprocess.TimeoutExpired:
                        pipeline_process.kill()
                        pipeline_process.communicate(
                            timeout=SIGNAL_FORCE_CLEANUP_TIMEOUT_SECONDS,
                        )
                if python_pid is not None:
                    try:
                        os.kill(python_pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        if (
                            python_process_group is not None
                            and python_process_group != os.getpgrp()
                        ):
                            try:
                                os.killpg(
                                    python_process_group,
                                    signal.SIGKILL,
                                )
                            except ProcessLookupError:
                                pass
                        else:
                            try:
                                os.kill(python_pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass

        output = "\n".join((stdout, stderr))
        self.assertEqual(pipeline_process.returncode, 143, output)
        self.assertTrue(process_gone, "TERM 后仍存在训练子进程。")
        self.assertTrue(lock_released, "TERM 退出后 GPU0 锁未释放。")


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
