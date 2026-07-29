"""Step 1 Python/C++ differential gates."""

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
    parse_normalized_config,
    parse_normalized_workload,
    run_foundation_oracle,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "cpp" / "tests" / "fixtures"
FOUNDATION_CONFIG = (
    FIXTURE_ROOT / "config" / "minimal_foundation_colocation.json"
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
        if not isinstance(expected, (int, float)) or not isinstance(
            actual, (int, float)
        ):
            raise ParityMismatch(
                f"{path}: expected numeric {expected!r}, got {actual!r}"
            )
        if not math.isclose(
            float(expected),
            float(actual),
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
) -> None:
    try:
        _compare_values(expected, actual)
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
    text = FOUNDATION_CONFIG.read_text(encoding="utf-8")
    expected = parse_normalized_config(text)
    result = cpp_runner.run(
        [
            "--normalize-config",
            cpp_runner.path_argument(FOUNDATION_CONFIG),
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
    raw = json.loads(FOUNDATION_CONFIG.read_text(encoding="utf-8"))
    raw.update(mutation)
    text = json.dumps(raw)
    with pytest.raises(OracleError) as python_error:
        parse_normalized_config(text)

    config_path = tmp_path / f"{case_name}.json"
    config_path.write_text(text, encoding="utf-8")
    result = cpp_runner.run(
        ["--normalize-config", cpp_runner.path_argument(config_path)]
    )
    if result.returncode == 0:
        _raise_with_artifacts(
            tmp_path=tmp_path,
            case_name=case_name,
            message="Python rejected config but C++ accepted it",
            python_output=str(python_error.value),
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


def test_foundation_lifecycle_matches(
    cpp_runner: CppRunner,
    tmp_path: Path,
) -> None:
    config = parse_normalized_config(
        FOUNDATION_CONFIG.read_text(encoding="utf-8")
    )
    workload = parse_normalized_workload(
        SESSION_WORKLOAD.read_text(encoding="utf-8")
    )
    expected = run_foundation_oracle(config, workload)
    result = cpp_runner.run(
        [
            "--config",
            cpp_runner.path_argument(FOUNDATION_CONFIG),
            "--workload",
            cpp_runner.path_argument(SESSION_WORKLOAD),
        ]
    )
    _require_cpp_success(
        result,
        tmp_path=tmp_path,
        case_name="foundation_lifecycle",
        python_output=json.dumps(expected, indent=2),
    )
    actual = json.loads(result.stdout)
    _compare_or_record(
        tmp_path=tmp_path,
        case_name="foundation_lifecycle",
        expected=expected,
        actual=actual,
        cpp_result=result,
    )


def test_equal_time_event_order_matches(
    cpp_runner: CppRunner,
    tmp_path: Path,
) -> None:
    workload_text = (
        "arrived_at,num_prefill_tokens,num_decode_tokens\n"
        "0,32,8\n"
        "0,16,4\n"
        "0.001,8,2\n"
    )
    workload_path = tmp_path / "equal_time.csv"
    workload_path.write_text(workload_text, encoding="utf-8")
    config = parse_normalized_config(
        FOUNDATION_CONFIG.read_text(encoding="utf-8")
    )
    expected = run_foundation_oracle(
        config,
        parse_normalized_workload(workload_text),
    )
    result = cpp_runner.run(
        [
            "--config",
            cpp_runner.path_argument(FOUNDATION_CONFIG),
            "--workload",
            cpp_runner.path_argument(workload_path),
        ]
    )
    _require_cpp_success(
        result,
        tmp_path=tmp_path,
        case_name="equal_time_order",
        python_output=json.dumps(expected, indent=2),
    )
    actual = json.loads(result.stdout)
    _compare_or_record(
        tmp_path=tmp_path,
        case_name="equal_time_order",
        expected=expected,
        actual=actual,
        cpp_result=result,
    )
    assert [
        event["sequence"] for event in actual["event_trace"]
    ] == [1, 2, 3, 4, 5, 6]


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
