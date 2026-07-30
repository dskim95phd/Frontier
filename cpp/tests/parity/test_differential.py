"""Python/C++ differential gates for the current C++ contract."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Iterable

import pytest

from python_oracle import (
    OracleError,
    parse_normalized_workload,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "cpp" / "tests" / "fixtures"
CURRENT_CONFIG = (
    FIXTURE_ROOT / "config" / "fixed_parallel_colocation.json"
)
SESSION_WORKLOAD = FIXTURE_ROOT / "workloads" / "session_prefix.csv"
ANALYTICAL_FIXTURE = (
    FIXTURE_ROOT / "analytical" / "analytical_v1.json"
)
ANALYTICAL_GENERATOR = (
    REPO_ROOT
    / "cpp"
    / "tests"
    / "golden"
    / "generate_analytical_golden.py"
)
ANALYTICAL_BATCH_FIXTURE = (
    FIXTURE_ROOT / "analytical" / "analytical_batch_v1.json"
)
ANALYTICAL_BATCH_GENERATOR = (
    REPO_ROOT
    / "cpp"
    / "tests"
    / "golden"
    / "generate_analytical_batch_golden.py"
)
STEP25_SIMULATOR_ORACLE = (
    REPO_ROOT
    / "cpp"
    / "tests"
    / "parity"
    / "step25_simulator_oracle.py"
)
STEP25_FIXED_CONFIG = (
    FIXTURE_ROOT / "config" / "fixed_parallel_colocation.json"
)
STEP25_PP4_CONFIG = (
    FIXTURE_ROOT / "config" / "fixed_pp4_colocation.json"
)
STEP25_ANALYTICAL_CONFIG = (
    FIXTURE_ROOT
    / "config"
    / "analytical_parallel_colocation.json"
)
STEP25_WORKLOAD = (
    FIXTURE_ROOT / "workloads" / "step25_parallel.csv"
)
STEP25_PRESSURE_CONFIG = (
    FIXTURE_ROOT / "config" / "fixed_dp2_pressure_colocation.json"
)
STEP25_PRESSURE_WORKLOAD = (
    FIXTURE_ROOT / "workloads" / "step25_parallel_pressure.csv"
)
STEP25_MATRIX_WORKLOAD = (
    FIXTURE_ROOT / "workloads" / "step25_matrix.csv"
)
STEP3_SIMULATOR_ORACLE = (
    REPO_ROOT
    / "cpp"
    / "tests"
    / "parity"
    / "step3_simulator_oracle.py"
)
STEP3_FIXED_CONFIG = (
    FIXTURE_ROOT / "config" / "fixed_sequential_pdd.json"
)
STEP3_SMALL_WORKLOAD = (
    FIXTURE_ROOT / "workloads" / "step3_pdd_small.csv"
)
STEP3_ANALYTICAL_PP_CONTEXT_WORKLOAD = (
    FIXTURE_ROOT
    / "workloads"
    / "step3_analytical_pp_context.csv"
)
STEP3_DECODE_PREEMPTION_ORDER_WORKLOAD = (
    FIXTURE_ROOT
    / "workloads"
    / "step3_decode_preemption_order.csv"
)
STEP3_EQUAL_TIME_DECODE_ROUTING_WORKLOAD = (
    FIXTURE_ROOT
    / "workloads"
    / "step3_equal_time_decode_routing.csv"
)

STEP3_MATRIX_CASES: list[dict[str, Any]] = [
    {
        "name": "pdd01_online_minimal",
        "mode": "online",
        "prefill": (1, 1, 1, 1),
        "decode": (1, 1, 1, 1),
        "workload": STEP3_SMALL_WORKLOAD,
    },
    {
        "name": "pdd02_offline_barrier",
        "mode": "offline",
        "prefill": (1, 1, 1, 1),
        "decode": (1, 1, 1, 1),
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "pdd03_zero_base_latency",
        "mode": "online",
        "prefill": (1, 1, 1, 1),
        "decode": (1, 1, 1, 1),
        "transfer": {"network_latency_ms": 0.0},
        "workload": STEP3_SMALL_WORKLOAD,
    },
    {
        "name": "pdd04_asymmetric_tp_pp",
        "mode": "online",
        "prefill": (1, 2, 2, 1),
        "decode": (1, 4, 4, 1),
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "pdd05_independent_routing",
        "mode": "online",
        "prefill": (2, 2, 2, 2),
        "decode": (1, 4, 2, 2),
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "pdd06_offline_target_remap",
        "mode": "offline",
        "prefill": (1, 2, 4, 2),
        "decode": (2, 2, 2, 1),
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "pdd07_source_retention_pressure",
        "mode": "online",
        "prefill": (1, 1, 1, 1),
        "decode": (1, 1, 1, 1),
        "prefill_scheduler": {
            "batch_size_cap": 1,
            "max_tokens_in_batch": 4,
            "num_blocks": 2,
        },
        "transfer": {
            "network_bandwidth_gbps": 1.0,
            "network_latency_ms": 5.0,
        },
        "workload": STEP25_PRESSURE_WORKLOAD,
    },
    {
        "name": "pdd08_decode_preemption_pressure",
        "mode": "online",
        "prefill": (1, 1, 1, 1),
        "decode": (1, 1, 1, 1),
        "decode_scheduler": {"num_blocks": 3},
        "workload": STEP25_PRESSURE_WORKLOAD,
    },
    {
        "name": "pdd09_chunked_prefill",
        "mode": "online",
        "prefill": (1, 2, 2, 1),
        "decode": (1, 2, 2, 1),
        "prefill_scheduler": {
            "max_tokens_in_batch": 3,
            "long_prefill_token_threshold": 2,
        },
        "workload": STEP3_SMALL_WORKLOAD,
    },
    {
        "name": "pdd10_multi_request_transfer_batch",
        "mode": "online",
        "prefill": (1, 1, 1, 1),
        "decode": (1, 2, 2, 1),
        "prefill_scheduler": {
            "batch_size_cap": 8,
            "max_tokens_in_batch": 32,
        },
        "workload": STEP25_PRESSURE_WORKLOAD,
    },
    {
        "name": "pdd11_pp4_backpressure",
        "mode": "online",
        "prefill": (1, 2, 4, 1),
        "decode": (1, 4, 4, 1),
        "prefill_latencies_ms": (0.2, 1.1, 0.3, 0.8),
        "decode_latencies_ms": (0.7, 0.2, 1.0, 0.4),
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "pdd12_offline_pp8_terminal_drain",
        "mode": "offline",
        "prefill": (1, 1, 8, 1),
        "decode": (1, 2, 4, 2),
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "pdd13_analytical_tp1_to_tp8",
        "mode": "online",
        "execution_type": "analytical",
        "prefill": (1, 1, 1, 1),
        "decode": (1, 8, 2, 1),
        "workload": STEP3_SMALL_WORKLOAD,
    },
    {
        "name": "pdd14_analytical_tp8_to_tp1",
        "mode": "online",
        "execution_type": "analytical",
        "prefill": (1, 8, 2, 1),
        "decode": (1, 1, 1, 1),
        "workload": STEP3_SMALL_WORKLOAD,
    },
    {
        "name": "pdd15_analytical_pp_runtime_context",
        "mode": "offline",
        "execution_type": "analytical",
        "prefill": (1, 8, 4, 1),
        "decode": (1, 1, 1, 1),
        "prefill_scheduler": {
            "batch_size_cap": 16,
            "max_tokens_in_batch": 32,
            "block_size": 4,
            "num_blocks": 512,
        },
        "decode_scheduler": {
            "batch_size_cap": 16,
            "max_tokens_in_batch": 32,
            "block_size": 4,
            "num_blocks": 512,
        },
        "workload": STEP3_ANALYTICAL_PP_CONTEXT_WORKLOAD,
    },
    {
        "name": "pdd16_decode_preemption_waiting_order",
        "mode": "online",
        "prefill": (1, 2, 2, 1),
        "decode": (1, 1, 1, 1),
        "prefill_scheduler": {
            "batch_size_cap": 16,
            "max_tokens_in_batch": 32,
            "block_size": 4,
            "num_blocks": 512,
        },
        "decode_scheduler": {
            "batch_size_cap": 8,
            "max_tokens_in_batch": 8,
            "block_size": 4,
            "num_blocks": 20,
        },
        "workload": STEP3_DECODE_PREEMPTION_ORDER_WORKLOAD,
    },
    {
        "name": "pdd17_equal_time_decode_routing",
        "mode": "online",
        "prefill": (2, 2, 2, 2),
        "decode": (2, 4, 4, 2),
        "prefill_scheduler": {
            "batch_size_cap": 16,
            "max_tokens_in_batch": 32,
            "block_size": 4,
            "num_blocks": 512,
        },
        "decode_scheduler": {
            "batch_size_cap": 16,
            "max_tokens_in_batch": 32,
            "block_size": 4,
            "num_blocks": 512,
        },
        "workload": STEP3_EQUAL_TIME_DECODE_ROUTING_WORKLOAD,
    },
]

STEP25_MATRIX_CASES: list[dict[str, Any]] = [
    {
        "name": "fixed_offline_tp1_pp1_dp1_r1_small_budget",
        "simulation_mode": "offline",
        "topology": (1, 1, 1, 1),
        "execution_type": "fixed",
        "stage_latencies_ms": (0.7,),
        "scheduler": {
            "batch_size_cap": 3,
            "max_tokens_in_batch": 5,
            "block_size": 4,
            "num_blocks": 128,
        },
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "fixed_online_tp8_pp1_dp4_r2_wide_routing",
        "simulation_mode": "online",
        "topology": (2, 8, 1, 4),
        "execution_type": "fixed",
        "stage_latencies_ms": (1.1,),
        "scheduler": {
            "batch_size_cap": 4,
            "max_tokens_in_batch": 9,
            "block_size": 4,
            "num_blocks": 64,
        },
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "fixed_offline_tp2_pp2_dp1_r3_uneven_replicas",
        "simulation_mode": "offline",
        "topology": (3, 2, 2, 1),
        "execution_type": "fixed",
        "stage_latencies_ms": (0.3, 1.7),
        "scheduler": {
            "batch_size_cap": 2,
            "max_tokens_in_batch": 7,
            "block_size": 2,
            "num_blocks": 128,
        },
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "fixed_online_tp4_pp4_dp2_r2_asymmetric_stages",
        "simulation_mode": "online",
        "topology": (2, 4, 4, 2),
        "execution_type": "fixed",
        "stage_latencies_ms": (0.25, 0.5, 1.0, 2.0),
        "scheduler": {
            "batch_size_cap": 5,
            "max_tokens_in_batch": 8,
            "block_size": 4,
            "num_blocks": 96,
        },
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "fixed_offline_tp8_pp8_dp1_r1_deep_pipeline",
        "simulation_mode": "offline",
        "topology": (1, 8, 8, 1),
        "execution_type": "fixed",
        "stage_latencies_ms": (
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ),
        "scheduler": {
            "batch_size_cap": 6,
            "max_tokens_in_batch": 12,
            "block_size": 4,
            "num_blocks": 128,
        },
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "fixed_online_tp1_pp2_dp3_r1_uneven_dp",
        "simulation_mode": "online",
        "topology": (1, 1, 2, 3),
        "execution_type": "fixed",
        "stage_latencies_ms": (2.0, 0.5),
        "scheduler": {
            "batch_size_cap": 3,
            "max_tokens_in_batch": 6,
            "block_size": 2,
            "num_blocks": 96,
        },
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "fixed_offline_no_chunked_no_preemption",
        "simulation_mode": "offline",
        "topology": (1, 2, 1, 2),
        "execution_type": "fixed",
        "stage_latencies_ms": (0.9,),
        "scheduler": {
            "batch_size_cap": 4,
            "max_tokens_in_batch": 8,
            "enable_preemption": False,
            "enable_chunked_prefill": False,
            "block_size": 4,
            "num_blocks": 32,
        },
        "workload": STEP25_PRESSURE_WORKLOAD,
    },
    {
        "name": "fixed_offline_long_prefill_threshold",
        "simulation_mode": "offline",
        "topology": (2, 4, 2, 2),
        "execution_type": "fixed",
        "stage_latencies_ms": (0.4, 1.3),
        "scheduler": {
            "batch_size_cap": 4,
            "max_tokens_in_batch": 10,
            "long_prefill_token_threshold": 3,
            "block_size": 2,
            "num_blocks": 128,
        },
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "fixed_offline_nonzero_watermark",
        "simulation_mode": "offline",
        "topology": (1, 2, 2, 2),
        "execution_type": "fixed",
        "stage_latencies_ms": (0.6, 1.2),
        "scheduler": {
            "batch_size_cap": 2,
            "max_tokens_in_batch": 8,
            "block_size": 2,
            "num_blocks": 8,
            "watermark_blocks_fraction": 0.25,
        },
        "workload": STEP25_PRESSURE_WORKLOAD,
    },
    {
        "name": "fixed_online_preallocation",
        "simulation_mode": "online",
        "topology": (1, 2, 2, 2),
        "execution_type": "fixed",
        "stage_latencies_ms": (0.8, 1.4),
        "scheduler": {
            "batch_size_cap": 4,
            "max_tokens_in_batch": 8,
            "block_size": 4,
            "num_blocks": 128,
            "num_preallocate_tokens": 4,
        },
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "analytical_offline_tp1_pp1_dp1_r1",
        "simulation_mode": "offline",
        "topology": (1, 1, 1, 1),
        "execution_type": "analytical",
        "scheduler": {
            "batch_size_cap": 4,
            "max_tokens_in_batch": 12,
            "block_size": 8,
            "num_blocks": 128,
        },
        "workload": STEP25_WORKLOAD,
    },
    {
        "name": "analytical_online_tp2_pp2_dp2_r2",
        "simulation_mode": "online",
        "topology": (2, 2, 2, 2),
        "execution_type": "analytical",
        "scheduler": {
            "batch_size_cap": 4,
            "max_tokens_in_batch": 8,
            "block_size": 4,
            "num_blocks": 96,
        },
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "analytical_offline_tp8_pp4_dp2_r1",
        "simulation_mode": "offline",
        "topology": (1, 8, 4, 2),
        "execution_type": "analytical",
        "scheduler": {
            "batch_size_cap": 5,
            "max_tokens_in_batch": 16,
            "block_size": 8,
            "num_blocks": 128,
        },
        "workload": STEP25_MATRIX_WORKLOAD,
    },
    {
        "name": "analytical_offline_tp4_pp8_dp1_r1",
        "simulation_mode": "offline",
        "topology": (1, 4, 8, 1),
        "execution_type": "analytical",
        "scheduler": {
            "batch_size_cap": 6,
            "max_tokens_in_batch": 16,
            "block_size": 8,
            "num_blocks": 128,
        },
        "workload": STEP25_WORKLOAD,
    },
]


@dataclass(frozen=True)
class CppRunner:
    prefix: tuple[str, ...]
    binary: str
    wsl_paths: bool

    def path_argument(self, path: Path) -> str:
        resolved = path.resolve()
        if not self.wsl_paths:
            return str(resolved)
        drive = resolved.drive.rstrip(":").lower()
        if not drive:
            raise RuntimeError(
                f"cannot translate non-drive path to WSL: {resolved}"
            )
        relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
        return f"/mnt/{drive}/{relative}"

    def sibling_binary(self, name: str) -> str:
        if self.wsl_paths:
            return str(PurePosixPath(self.binary).with_name(name))
        binary_path = Path(self.binary)
        sibling_name = name + binary_path.suffix
        return str(binary_path.with_name(sibling_name))

    def run(
        self,
        arguments: Iterable[str],
        *,
        binary: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            *self.prefix,
            binary or self.binary,
            *arguments,
        ]
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )


@pytest.fixture(scope="session")
def cpp_runner() -> CppRunner:
    binary = os.environ.get("FRONTIER_CPP_BINARY")
    if not binary:
        pytest.skip(
            "set FRONTIER_CPP_BINARY to the built frontier_sim executable "
            "to run C++ parity gates"
        )

    raw_prefix = os.environ.get("FRONTIER_CPP_RUNNER", "[]")
    try:
        parsed_prefix = json.loads(raw_prefix)
    except json.JSONDecodeError as error:
        pytest.fail(f"FRONTIER_CPP_RUNNER must be a JSON array: {error}")
    if not isinstance(parsed_prefix, list) or not all(
        isinstance(item, str) for item in parsed_prefix
    ):
        pytest.fail("FRONTIER_CPP_RUNNER must be a JSON array of strings")

    path_style = os.environ.get("FRONTIER_CPP_PATH_STYLE", "auto")
    if path_style not in {"auto", "native", "wsl"}:
        pytest.fail(
            "FRONTIER_CPP_PATH_STYLE must be auto, native, or wsl"
        )
    inferred_wsl = bool(parsed_prefix) and (
        Path(parsed_prefix[0]).name.lower() in {"wsl", "wsl.exe"}
    )
    return CppRunner(
        prefix=tuple(parsed_prefix),
        binary=binary,
        wsl_paths=path_style == "wsl"
        or (path_style == "auto" and inferred_wsl),
    )


class ParityMismatch(AssertionError):
    pass


def _compare_values(
    expected: Any,
    actual: Any,
    *,
    path: str = "$",
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if type(expected) is not type(actual) or expected != actual:
            raise ParityMismatch(
                f"{path}: expected {expected!r}, got {actual!r}"
            )
        return
    if isinstance(expected, float) or isinstance(actual, float):
        if type(expected) is not float or type(actual) is not float:
            raise ParityMismatch(
                f"{path}: numeric types differ; expected "
                f"{type(expected).__name__} {expected!r}, got "
                f"{type(actual).__name__} {actual!r}"
            )
        if not math.isclose(
            expected,
            actual,
            abs_tol=absolute_tolerance,
            rel_tol=relative_tolerance,
        ):
            raise ParityMismatch(
                f"{path}: expected {expected!r}, got {actual!r}; "
                f"abs_tol={absolute_tolerance}, "
                f"rel_tol={relative_tolerance}"
            )
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ParityMismatch(
                f"{path}: expected object, got {type(actual).__name__}"
            )
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            raise ParityMismatch(
                f"{path}: object keys differ; "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )
        for key in expected:
            _compare_values(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise ParityMismatch(
                f"{path}: expected array, got {type(actual).__name__}"
            )
        if len(expected) != len(actual):
            raise ParityMismatch(
                f"{path}: expected {len(expected)} items, "
                f"got {len(actual)}"
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual)
        ):
            _compare_values(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        return
    if type(expected) is not type(actual) or expected != actual:
        raise ParityMismatch(
            f"{path}: expected {expected!r}, got {actual!r}"
        )


def test_compare_values_requires_exact_integer_types_and_values() -> None:
    _compare_values(7, 7)
    with pytest.raises(ParityMismatch, match="numeric types differ"):
        _compare_values(7, 7.0)
    with pytest.raises(ParityMismatch, match="numeric types differ"):
        _compare_values(7.0, 7)
    with pytest.raises(ParityMismatch):
        _compare_values(7, 8)


def test_compare_values_uses_tolerance_only_for_floats() -> None:
    _compare_values(1.0, 1.0 + 1e-13)
    with pytest.raises(ParityMismatch):
        _compare_values(1.0, 1.0 + 1e-6)


def _artifact_directory(tmp_path: Path, case_name: str) -> Path:
    configured = os.environ.get("FRONTIER_PARITY_ARTIFACT_DIR")
    root = Path(configured) if configured else tmp_path / "parity-artifacts"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_name)
    directory = root / safe_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _raise_with_artifacts(
    *,
    tmp_path: Path,
    case_name: str,
    message: str,
    python_output: str,
    cpp_result: subprocess.CompletedProcess[str],
) -> None:
    directory = _artifact_directory(tmp_path, case_name)
    (directory / "python.txt").write_text(
        python_output,
        encoding="utf-8",
    )
    (directory / "cpp.stdout.txt").write_text(
        cpp_result.stdout,
        encoding="utf-8",
    )
    (directory / "cpp.stderr.txt").write_text(
        cpp_result.stderr,
        encoding="utf-8",
    )
    (directory / "difference.txt").write_text(
        message + "\n",
        encoding="utf-8",
    )
    raise AssertionError(f"{message}; raw artifacts: {directory}")


def _compare_or_record(
    *,
    tmp_path: Path,
    case_name: str,
    expected: Any,
    actual: Any,
    cpp_result: subprocess.CompletedProcess[str],
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
) -> None:
    try:
        _compare_values(
            expected,
            actual,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    except ParityMismatch as error:
        _raise_with_artifacts(
            tmp_path=tmp_path,
            case_name=case_name,
            message=str(error),
            python_output=json.dumps(
                expected,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            cpp_result=cpp_result,
        )


def _require_cpp_success(
    result: subprocess.CompletedProcess[str],
    *,
    tmp_path: Path,
    case_name: str,
    python_output: str,
) -> None:
    if result.returncode != 0:
        _raise_with_artifacts(
            tmp_path=tmp_path,
            case_name=case_name,
            message=f"C++ command exited with {result.returncode}",
            python_output=python_output,
            cpp_result=result,
        )


def test_config_acceptance_and_normalization_match(
    cpp_runner: CppRunner,
    tmp_path: Path,
) -> None:
    text = CURRENT_CONFIG.read_text(encoding="utf-8")
    expected = json.loads(text)
    result = cpp_runner.run(
        [
            "--normalize-config",
            cpp_runner.path_argument(CURRENT_CONFIG),
        ]
    )
    _require_cpp_success(
        result,
        tmp_path=tmp_path,
        case_name="config_acceptance",
        python_output=json.dumps(expected, indent=2),
    )
    actual = json.loads(result.stdout)
    _compare_or_record(
        tmp_path=tmp_path,
        case_name="config_acceptance",
        expected=expected,
        actual=actual,
        cpp_result=result,
    )


@pytest.mark.parametrize(
    ("case_name", "mutation"),
    [
        ("config_unknown_field", {"unexpected": 1}),
        (
            "config_parallel_clusters",
            {"enable_parallel_clusters": True},
        ),
        (
            "config_block_hash",
            {"prefix_cache": {"enabled": True, "key_mode": "block_hash"}},
        ),
    ],
)
def test_config_rejection_matches(
    cpp_runner: CppRunner,
    tmp_path: Path,
    case_name: str,
    mutation: dict[str, Any],
) -> None:
    raw = json.loads(CURRENT_CONFIG.read_text(encoding="utf-8"))
    raw.update(mutation)
    text = json.dumps(raw)
    config_path = tmp_path / f"{case_name}.json"
    config_path.write_text(text, encoding="utf-8")
    result = cpp_runner.run(
        ["--normalize-config", cpp_runner.path_argument(config_path)]
    )
    if result.returncode == 0:
        _raise_with_artifacts(
            tmp_path=tmp_path,
            case_name=case_name,
            message="C++ accepted an invalid current config",
            python_output=text,
            cpp_result=result,
        )


def test_workload_acceptance_and_normalization_match(
    cpp_runner: CppRunner,
    tmp_path: Path,
) -> None:
    text = SESSION_WORKLOAD.read_text(encoding="utf-8")
    expected = parse_normalized_workload(text)
    result = cpp_runner.run(
        [
            "--normalize-workload",
            cpp_runner.path_argument(SESSION_WORKLOAD),
        ]
    )
    _require_cpp_success(
        result,
        tmp_path=tmp_path,
        case_name="workload_acceptance",
        python_output=json.dumps(expected, indent=2),
    )
    actual = parse_normalized_workload(result.stdout)
    _compare_or_record(
        tmp_path=tmp_path,
        case_name="workload_acceptance",
        expected=expected,
        actual=actual,
        cpp_result=result,
    )


@pytest.mark.parametrize(
    ("case_name", "text"),
    [
        (
            "workload_block_hash",
            "arrived_at,num_prefill_tokens,num_decode_tokens,"
            "block_hash_ids\n0,1,1,9\n",
        ),
        (
            "workload_nonfinite_arrival",
            "arrived_at,num_prefill_tokens,num_decode_tokens\n"
            "nan,1,1\n",
        ),
        (
            "workload_noninteger_tokens",
            "arrived_at,num_prefill_tokens,num_decode_tokens\n"
            "0,1.5,1\n",
        ),
    ],
)
def test_workload_rejection_matches(
    cpp_runner: CppRunner,
    tmp_path: Path,
    case_name: str,
    text: str,
) -> None:
    with pytest.raises(OracleError) as python_error:
        parse_normalized_workload(text)

    workload_path = tmp_path / f"{case_name}.csv"
    workload_path.write_text(text, encoding="utf-8")
    result = cpp_runner.run(
        ["--normalize-workload", cpp_runner.path_argument(workload_path)]
    )
    if result.returncode == 0:
        _raise_with_artifacts(
            tmp_path=tmp_path,
            case_name=case_name,
            message="Python rejected workload but C++ accepted it",
            python_output=str(python_error.value),
            cpp_result=result,
        )


def test_analytical_golden_matches_both_implementations(
    cpp_runner: CppRunner,
    tmp_path: Path,
) -> None:
    python_result = subprocess.run(
        [sys.executable, str(ANALYTICAL_GENERATOR)],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if python_result.returncode != 0:
        pytest.fail(
            "Python analytical oracle failed:\n"
            f"{python_result.stdout}\n{python_result.stderr}"
        )
    generated = json.loads(python_result.stdout)
    checked_in = json.loads(
        ANALYTICAL_FIXTURE.read_text(encoding="utf-8")
    )

    analytical_binary = cpp_runner.sibling_binary(
        "frontier_analytical_model_test"
    )
    cpp_result = cpp_runner.run([], binary=analytical_binary)
    _require_cpp_success(
        cpp_result,
        tmp_path=tmp_path,
        case_name="analytical_golden",
        python_output=python_result.stdout,
    )
    _compare_or_record(
        tmp_path=tmp_path,
        case_name="analytical_golden",
        expected=generated,
        actual=checked_in,
        cpp_result=cpp_result,
    )


def test_analytical_batch_golden_is_current() -> None:
    python_result = subprocess.run(
        [sys.executable, str(ANALYTICAL_BATCH_GENERATOR)],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if python_result.returncode != 0:
        pytest.fail(
            "Analytical batch oracle failed:\n"
            f"{python_result.stdout}\n{python_result.stderr}"
        )
    generated = json.loads(python_result.stdout)
    checked_in = json.loads(
        ANALYTICAL_BATCH_FIXTURE.read_text(encoding="utf-8")
    )
    _compare_values(generated, checked_in)


def _normalize_step25_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for sequence, raw in enumerate(events, start=1):
        event_type = raw.get("type")
        event = {
            "sequence": sequence,
            "time_s": float(raw["time_s"]),
            "type": event_type,
        }
        if event_type == "request_arrival":
            event["request_id"] = int(raw["request_id"])
        if event_type in {
            "replica_schedule",
            "batch_stage_arrival",
            "replica_stage_schedule",
            "batch_stage_end",
        }:
            event["replica_id"] = int(raw["replica_id"])
            event["dp_id"] = int(raw["dp_id"])
        if event_type in {
            "batch_stage_arrival",
            "batch_stage_end",
        }:
            event["batch_id"] = int(raw["batch_id"])
        if event_type in {
            "batch_stage_arrival",
            "replica_stage_schedule",
            "batch_stage_end",
        }:
            event["stage_id"] = int(raw["stage_id"])
        result.append(event)
    return result


def _normalize_step25_cpp(
    output: dict[str, Any],
    *,
    include_components: bool,
) -> dict[str, Any]:
    batches = {
        int(batch["batch_id"]): batch
        for batch in output["batches"]
    }
    stages: list[dict[str, Any]] = []
    for raw in output["batch_stages"]:
        batch = batches[int(raw["batch_id"])]
        stage = {
            "batch_id": int(raw["batch_id"]),
            "replica_id": int(raw["replica_id"]),
            "dp_id": int(raw["dp_id"]),
            "stage_id": int(raw["stage_id"]),
            "request_ids": [
                int(value) for value in batch["request_ids"]
            ],
            "request_num_tokens": [
                int(value) for value in batch["scheduled_tokens"]
            ],
            "started_at_s": float(raw["started_at_s"]),
            "completed_at_s": float(raw["completed_at_s"]),
            "duration_ms": float(raw["duration_ms"]),
        }
        if include_components:
            stage.update(
                {
                    "dense_compute_ms": float(
                        raw["dense_compute_ms"]
                    ),
                    "tp_communication_ms": float(
                        raw["tp_communication_ms"]
                    ),
                    "pp_communication_ms": float(
                        raw["pp_communication_ms"]
                    ),
                }
            )
        stages.append(stage)

    requests = sorted(
        (
            {
                "request_id": int(raw["request_id"]),
                "replica_id": int(raw["replica_id"]),
                "dp_id": int(raw["dp_id"]),
                "arrived_at_s": float(raw["arrived_at_s"]),
                "first_scheduled_at_s": float(
                    raw["first_scheduled_at_s"]
                ),
                "prefill_completed_at_s": float(
                    raw["prefill_completed_at_s"]
                ),
                "completed_at_s": float(raw["completed_at_s"]),
                "num_processed_tokens": int(
                    raw["num_processed_tokens"]
                ),
                "preemption_count": int(raw["preemption_count"]),
            }
            for raw in output["requests"]
        ),
        key=lambda row: row["request_id"],
    )
    scheduler_trace = [
        {
            "iteration_id": int(raw["iteration_id"]),
            "simulation_time_s": float(raw["simulation_time_s"]),
            "decisions": [
                {
                    "decision_result": str(decision["decision_result"]),
                    "request_id": int(decision["request_id"]),
                    "num_tokens": int(decision["num_tokens"]),
                    "token_budget_after": int(
                        decision["token_budget_after"]
                    ),
                    "available_blocks_after": int(
                        decision["available_blocks_after"]
                    ),
                }
                for decision in raw["decisions"]
            ],
            "token_budget_before": int(raw["token_budget_before"]),
            "token_budget_after": int(raw["token_budget_after"]),
            "available_blocks_before": int(
                raw["available_blocks_before"]
            ),
            "available_blocks_after": int(
                raw["available_blocks_after"]
            ),
            "waiting_count_before": int(raw["waiting_count_before"]),
            "waiting_count_after": int(raw["waiting_count_after"]),
            "running_count_before": int(raw["running_count_before"]),
            "running_count_after": int(raw["running_count_after"]),
            "preempted_count": int(raw["preempted_count"]),
            "batch_request_ids": [
                int(value) for value in raw["batch_request_ids"]
            ],
            "request_num_tokens": [
                int(value) for value in raw["request_num_tokens"]
            ],
            "replica_id": int(raw["replica_id"]),
            "dp_id": int(raw["dp_id"]),
        }
        for raw in output["scheduler_trace"]
    ]
    return {
        "request_owners": [
            {
                "request_id": row["request_id"],
                "replica_id": row["replica_id"],
                "dp_id": row["dp_id"],
            }
            for row in requests
        ],
        "requests": requests,
        "event_trace": _normalize_step25_events(
            output["event_trace"]
        ),
        "scheduler_trace": scheduler_trace,
        "batch_stages": stages,
        "simulation_completed_at_s": max(
            row["completed_at_s"] for row in requests
        ),
    }


def _assert_step25_simulator_parity(
    *,
    cpp_runner: CppRunner,
    tmp_path: Path,
    case_name: str,
    config_path: Path,
    workload_path: Path,
    include_components: bool,
) -> None:
    python_result = subprocess.run(
        [
            sys.executable,
            str(STEP25_SIMULATOR_ORACLE),
            "--config",
            str(config_path),
            "--workload",
            str(workload_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if python_result.returncode != 0:
        pytest.fail(
            f"{case_name} production Simulator oracle failed:\n"
            f"{python_result.stdout}\n{python_result.stderr}"
        )
    oracle = json.loads(python_result.stdout)
    assert oracle.pop("oracle") == "frontier.simulator.Simulator"

    cpp_result = cpp_runner.run(
        [
            "--config",
            cpp_runner.path_argument(config_path),
            "--workload",
            cpp_runner.path_argument(workload_path),
        ]
    )
    _require_cpp_success(
        cpp_result,
        tmp_path=tmp_path,
        case_name=case_name,
        python_output=python_result.stdout,
    )
    actual = _normalize_step25_cpp(
        json.loads(cpp_result.stdout),
        include_components=include_components,
    )
    _compare_or_record(
        tmp_path=tmp_path,
        case_name=case_name,
        expected=oracle,
        actual=actual,
        cpp_result=cpp_result,
    )


def _write_step25_matrix_config(
    tmp_path: Path,
    case: dict[str, Any],
) -> Path:
    execution_type = str(case["execution_type"])
    base_path = (
        STEP25_ANALYTICAL_CONFIG
        if execution_type == "analytical"
        else STEP25_FIXED_CONFIG
    )
    config = json.loads(base_path.read_text(encoding="utf-8"))
    config["run_id"] = f"step25-matrix-{case['name']}"
    config["simulation_mode"] = str(case["simulation_mode"])
    replicas, tp, pp, dp = case["topology"]
    cluster = config["clusters"]["monolithic"]
    cluster["parallelism"] = {
        "num_replicas": int(replicas),
        "tensor_parallel_size": int(tp),
        "pipeline_parallel_size": int(pp),
        "data_parallel_size": int(dp),
    }
    cluster["scheduler"].update(case["scheduler"])
    if execution_type == "fixed":
        cluster["execution_model"] = {
            "type": "fixed",
            "stage_latencies_ms": [
                float(value)
                for value in case["stage_latencies_ms"]
            ],
        }
    config_path = tmp_path / f"{case['name']}.json"
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


@pytest.mark.parametrize(
    (
        "case_name",
        "config_path",
        "workload_path",
        "include_components",
    ),
    [
        (
            "step25_online_parallel_fixed",
            STEP25_FIXED_CONFIG,
            STEP25_WORKLOAD,
            False,
        ),
        (
            "step25_offline_pp4_fixed",
            STEP25_PP4_CONFIG,
            STEP25_WORKLOAD,
            False,
        ),
        (
            "step25_offline_parallel_analytical",
            STEP25_ANALYTICAL_CONFIG,
            STEP25_WORKLOAD,
            True,
        ),
        (
            "step25_dp2_target_local_pressure",
            STEP25_PRESSURE_CONFIG,
            STEP25_PRESSURE_WORKLOAD,
            False,
        ),
    ],
)
def test_step25_parallel_simulator_matches_production_python(
    cpp_runner: CppRunner,
    tmp_path: Path,
    case_name: str,
    config_path: Path,
    workload_path: Path,
    include_components: bool,
) -> None:
    _assert_step25_simulator_parity(
        cpp_runner=cpp_runner,
        tmp_path=tmp_path,
        case_name=case_name,
        config_path=config_path,
        workload_path=workload_path,
        include_components=include_components,
    )


@pytest.mark.parametrize(
    "case",
    STEP25_MATRIX_CASES,
    ids=lambda case: str(case["name"]),
)
def test_step25_configuration_matrix_matches_production_python(
    cpp_runner: CppRunner,
    tmp_path: Path,
    case: dict[str, Any],
) -> None:
    config_path = _write_step25_matrix_config(tmp_path, case)
    _assert_step25_simulator_parity(
        cpp_runner=cpp_runner,
        tmp_path=tmp_path,
        case_name=str(case["name"]),
        config_path=config_path,
        workload_path=case["workload"],
        include_components=(
            str(case["execution_type"]) == "analytical"
        ),
    )


def _write_step3_matrix_config(
    tmp_path: Path,
    case: dict[str, Any],
) -> Path:
    config = json.loads(
        STEP3_FIXED_CONFIG.read_text(encoding="utf-8")
    )
    config["run_id"] = str(case["name"])
    config["simulation_mode"] = str(case["mode"])
    execution_type = str(
        case.get("execution_type", "fixed")
    )
    for cluster_name in ("prefill", "decode"):
        replicas, tp, pp, dp = case[cluster_name]
        cluster = config["clusters"][cluster_name]
        cluster["parallelism"] = {
            "num_replicas": int(replicas),
            "tensor_parallel_size": int(tp),
            "pipeline_parallel_size": int(pp),
            "data_parallel_size": int(dp),
        }
        cluster["scheduler"].update(
            case.get(f"{cluster_name}_scheduler", {})
        )
        if execution_type == "analytical":
            cluster["execution_model"] = {
                "type": "analytical",
                "device": "rubin",
                "model": "llama2-7b",
                "precision": "fp16",
                "num_layers": 32,
                "network_bandwidth_gbps": 400.0,
                "network_latency_us": 1.0,
                "intra_node_bandwidth_gbps": 14400.0,
            }
        else:
            latencies = case.get(
                f"{cluster_name}_latencies_ms",
                tuple(0.2 * (index + 1) for index in range(pp)),
            )
            cluster["execution_model"] = {
                "type": "fixed",
                "stage_latencies_ms": [
                    float(value) for value in latencies
                ],
            }
    config["kv_cache_transfer"].update(
        case.get("transfer", {})
    )
    path = tmp_path / f"{case['name']}.json"
    path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _normalize_step3_cpp(
    output: dict[str, Any],
    *,
    include_components: bool,
) -> dict[str, Any]:
    requests = sorted(
        (
            {
                "request_id": int(row["request_id"]),
                "arrived_at_s": float(row["arrived_at_s"]),
                "first_scheduled_at_s": float(
                    row["first_scheduled_at_s"]
                ),
                "prefill_completed_at_s": float(
                    row["prefill_completed_at_s"]
                ),
                "first_token_completed_at_s": float(
                    row["first_token_completed_at_s"]
                ),
                "completed_at_s": float(row["completed_at_s"]),
                "num_processed_tokens": int(
                    row["num_processed_tokens"]
                ),
                "preemption_count": int(
                    row["preemption_count"]
                ),
                "prefill_replica_id": int(
                    row["prefill_replica_id"]
                ),
                "prefill_dp_id": int(row["prefill_dp_id"]),
                "decode_replica_id": int(
                    row["decode_replica_id"]
                ),
                "decode_dp_id": int(row["decode_dp_id"]),
                "kv_cache_transfer_start_time_s": float(
                    row["kv_cache_transfer_start_time_s"]
                ),
                "kv_cache_transfer_end_time_s": float(
                    row["kv_cache_transfer_end_time_s"]
                ),
            }
            for row in output["requests"]
        ),
        key=lambda row: row["request_id"],
    )
    owners = {
        cluster: [
            {
                "request_id": row["request_id"],
                "replica_id": row[
                    f"{cluster.lower()}_replica_id"
                ],
                "dp_id": row[f"{cluster.lower()}_dp_id"],
            }
            for row in requests
        ]
        for cluster in ("PREFILL", "DECODE")
    }
    transfers = sorted(
        (
            {
                "request_id": int(row["request_id"]),
                "source_cluster_type": str(
                    row["source_cluster_type"]
                ),
                "target_cluster_type": str(
                    row["target_cluster_type"]
                ),
                "source_replica_id": int(
                    row["source_replica_id"]
                ),
                "source_dp_id": int(row["source_dp_id"]),
                "size_bytes": int(row["size_bytes"]),
                "predicted_time_ms": float(
                    row["predicted_time_ms"]
                ),
                "started_at_s": float(row["started_at_s"]),
                "completed_at_s": float(
                    row["completed_at_s"]
                ),
            }
            for row in output["kv_cache_transfers"]
        ),
        key=lambda row: row["request_id"],
    )
    events: list[dict[str, Any]] = []
    target_event_types = {
        "replica_schedule",
        "batch_stage_arrival",
        "replica_stage_schedule",
        "batch_stage_end",
    }
    stage_event_types = {
        "batch_stage_arrival",
        "replica_stage_schedule",
        "batch_stage_end",
    }
    cluster_event_types = {
        "request_arrival",
        "cluster_schedule",
        *target_event_types,
        "kv_cache_transfer_start",
        "kv_cache_transfer_end",
    }
    for sequence, raw in enumerate(output["event_trace"], start=1):
        event_type = str(raw["type"])
        event: dict[str, Any] = {
            "sequence": sequence,
            "time_s": float(raw["time_s"]),
            "type": event_type,
        }
        if event_type in cluster_event_types:
            event["cluster_type"] = str(raw["cluster_type"])
        if event_type == "request_arrival":
            event["request_id"] = int(raw["request_id"])
        if event_type in target_event_types:
            event["replica_id"] = int(raw["replica_id"])
            event["dp_id"] = int(raw["dp_id"])
        if event_type in stage_event_types:
            event["stage_id"] = int(raw["stage_id"])
        events.append(event)

    scheduler_trace = [
        {
            "iteration_id": int(row["iteration_id"]),
            "simulation_time_s": float(
                row["simulation_time_s"]
            ),
            "decisions": [
                {
                    "decision_result": str(
                        decision["decision_result"]
                    ),
                    "request_id": int(decision["request_id"]),
                    "num_tokens": int(decision["num_tokens"]),
                    "token_budget_after": int(
                        decision["token_budget_after"]
                    ),
                    "available_blocks_after": int(
                        decision["available_blocks_after"]
                    ),
                }
                for decision in row["decisions"]
            ],
            "token_budget_before": int(
                row["token_budget_before"]
            ),
            "token_budget_after": int(row["token_budget_after"]),
            "available_blocks_before": int(
                row["available_blocks_before"]
            ),
            "available_blocks_after": int(
                row["available_blocks_after"]
            ),
            "waiting_count_before": int(
                row["waiting_count_before"]
            ),
            "waiting_count_after": int(
                row["waiting_count_after"]
            ),
            "running_count_before": int(
                row["running_count_before"]
            ),
            "running_count_after": int(
                row["running_count_after"]
            ),
            "preempted_count": int(row["preempted_count"]),
            "batch_request_ids": [
                int(value) for value in row["batch_request_ids"]
            ],
            "request_num_tokens": [
                int(value) for value in row["request_num_tokens"]
            ],
            "replica_id": int(row["replica_id"]),
            "dp_id": int(row["dp_id"]),
            "cluster_type": str(row["cluster_type"]),
        }
        for row in output["scheduler_trace"]
    ]
    batches = {
        int(row["batch_id"]): row for row in output["batches"]
    }
    stages: list[dict[str, Any]] = []
    for row in output["batch_stages"]:
        batch = batches[int(row["batch_id"])]
        stage: dict[str, Any] = {
            "cluster_type": str(row["cluster_type"]),
            "replica_id": int(row["replica_id"]),
            "dp_id": int(row["dp_id"]),
            "stage_id": int(row["stage_id"]),
            "request_ids": [
                int(value) for value in batch["request_ids"]
            ],
            "request_num_tokens": [
                int(value)
                for value in batch["scheduled_tokens"]
            ],
            "started_at_s": float(row["started_at_s"]),
            "completed_at_s": float(row["completed_at_s"]),
            "duration_ms": float(row["duration_ms"]),
        }
        if include_components:
            stage.update(
                {
                    "dense_compute_ms": float(
                        row["dense_compute_ms"]
                    ),
                    "tp_communication_ms": float(
                        row["tp_communication_ms"]
                    ),
                    "pp_communication_ms": float(
                        row["pp_communication_ms"]
                    ),
                }
            )
        stages.append(stage)
    return {
        "request_owners": owners,
        "requests": requests,
        "kv_cache_transfers": transfers,
        "event_trace": events,
        "scheduler_trace": scheduler_trace,
        "batch_stages": stages,
        "simulation_completed_at_s": max(
            row["completed_at_s"] for row in requests
        ),
    }


@pytest.mark.parametrize(
    "case",
    STEP3_MATRIX_CASES,
    ids=lambda case: str(case["name"]),
)
def test_step3_sequential_pdd_matches_production_python(
    cpp_runner: CppRunner,
    tmp_path: Path,
    case: dict[str, Any],
) -> None:
    config_path = _write_step3_matrix_config(tmp_path, case)
    workload_path = Path(case["workload"])
    python_result = subprocess.run(
        [
            sys.executable,
            str(STEP3_SIMULATOR_ORACLE),
            "--config",
            str(config_path),
            "--workload",
            str(workload_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    case_name = str(case["name"])
    if python_result.returncode != 0:
        pytest.fail(
            f"{case_name} production Simulator oracle failed:\n"
            f"{python_result.stdout}\n{python_result.stderr}"
        )
    oracle = json.loads(python_result.stdout)
    assert oracle.pop("oracle") == "frontier.simulator.Simulator"
    cpp_result = cpp_runner.run(
        [
            "--config",
            cpp_runner.path_argument(config_path),
            "--workload",
            cpp_runner.path_argument(workload_path),
        ]
    )
    _require_cpp_success(
        cpp_result,
        tmp_path=tmp_path,
        case_name=case_name,
        python_output=python_result.stdout,
    )
    include_components = (
        str(case.get("execution_type", "fixed"))
        == "analytical"
    )
    actual = _normalize_step3_cpp(
        json.loads(cpp_result.stdout),
        include_components=include_components,
    )
    _compare_or_record(
        tmp_path=tmp_path,
        case_name=case_name,
        expected=oracle,
        actual=actual,
        cpp_result=cpp_result,
        absolute_tolerance=(
            1e-8 if include_components else 1e-12
        ),
        relative_tolerance=(
            1e-8 if include_components else 1e-12
        ),
    )
