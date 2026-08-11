#!/usr/bin/env python3
"""Run a dense CPU-capacity/session-rate matrix for the Kimi K2 P8/D16 study.

Edit ``CPU_CAPACITY_RATE_RANGES`` and ``MAX_CONCURRENT_SIMULATIONS`` below for
the server.  Every inclusive rate range is expanded at ``RATE_STEP``.  The
runner generates one deterministic TraceLab workload per distinct rate, runs
all requested capacity/rate pairs, supports ``--resume``, and generates one
HTML capacity report per completed session-injection rate.

Capacity labels are physical decimal GB for one CPU pool shared by two PREFILL
GPUs.  The simulator's static CPU KV slice is therefore capacity / 2 per
PREFILL GPU.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

try:
    from .convert_tracelab_workload import (
        CONVERTER_VERSION,
        count_eligible_source_sessions,
    )
except ImportError:  # Direct script execution.
    from convert_tracelab_workload import (
        CONVERTER_VERSION,
        count_eligible_source_sessions,
    )


# ---------------------------------------------------------------------------
# Server experiment settings: edit this block before launching the sweep.
# Values are inclusive and expressed in source sessions/second.
# ---------------------------------------------------------------------------
CPU_CAPACITY_RATE_RANGES: dict[int, tuple[str, str]] = {
    500: ("0.28", "0.34"),
    750: ("0.30", "0.39"),
    1000: ("0.32", "0.40"),
    1250: ("0.34", "0.42"),
    1500: ("0.37", "0.44"),
    1750: ("0.40", "0.46"),
    2000: ("0.42", "0.47"),
    2500: ("0.43", "0.48"),
    3000: ("0.44", "0.49"),
    3500: ("0.44", "0.49"),
    4000: ("0.44", "0.49"),
}
RATE_STEP = "0.01"
MAX_CONCURRENT_SIMULATIONS = 4
SIMULATION_END_TIME_S = 10 * 60 * 60
PROGRESS_INTERVAL_S = 60 * 60
# None means use every eligible source session.  SESSION_REPETITIONS=None
# selects the minimum number of full epochs that keeps injection active for
# the entire simulation at the highest configured rate.
SAMPLE_SESSIONS: int | None = None
SESSION_REPETITIONS: int | None = None
SEED = 20260803


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CONFIG = HERE / "configs" / "tracelab_vera_rubin_p8_d16_cache_aware_cpu4tb.json"
DEFAULT_TRACELAB_DB = (
    REPO_ROOT / "outputs" / "datasets" / "tracelab" / "v0.0.2" / "syfi_coding_trace.duckdb"
)
DEFAULT_WORKLOAD_ROOT = (
    REPO_ROOT
    / "outputs"
    / "datasets"
    / "tracelab"
    / "v0.0.2"
    / "frontier"
    / "all_sessions_continuous_10h"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs" / "tracelab_vera_rubin_p8_d16_cpu_capacity_rate_dense_10h"
)
DECIMAL_GB = 1_000_000_000


@dataclass(frozen=True)
class MatrixCase:
    capacity_gb: int
    rate: Decimal
    rate_label: str
    capacity_label: str
    workload: Path
    manifest: Path
    metadata: Path
    output_dir: Path
    config_path: Path

    @property
    def label(self) -> str:
        return f"{self.rate_label}/{self.capacity_label}"


def _default_binary() -> Path:
    candidates = (
        REPO_ROOT / "cpp" / "build" / "Release" / "frontier_sim.exe",
        REPO_ROOT / "cpp" / "build" / "frontier_sim",
        REPO_ROOT / "build" / "cpp" / "frontier_sim",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _decimal(value: str | float | Decimal, *, name: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"invalid {name}: {value!r}") from error
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be finite and positive: {value!r}")
    return result


def decimal_range(minimum: str | float | Decimal, maximum: str | float | Decimal,
                  step: str | float | Decimal = RATE_STEP) -> list[Decimal]:
    low = _decimal(minimum, name="minimum rate")
    high = _decimal(maximum, name="maximum rate")
    increment = _decimal(step, name="rate step")
    if low > high:
        raise ValueError(f"minimum rate {low} exceeds maximum rate {high}")
    span = high - low
    if span % increment != 0:
        raise ValueError(
            f"rate range [{low}, {high}] is not aligned to step {increment}"
        )
    return [low + index * increment for index in range(int(span / increment) + 1)]


def rate_label(rate: Decimal | float | str) -> str:
    # Preserve the grid precision so 0.30 maps to the existing r0p30 workload
    # instead of creating a duplicate r0p3 directory.
    value = _decimal(rate, name="rate")
    return "r" + format(value, "f").replace("-", "m").replace(".", "p")


def capacity_label(capacity_gb: int) -> str:
    if capacity_gb < 0:
        raise ValueError("capacity must be nonnegative")
    return f"cpu{capacity_gb:04d}gb"


def workload_paths(root: Path, rate: Decimal, seed: int) -> tuple[Path, Path, Path]:
    directory = root / rate_label(rate)
    stem = f"seed_{seed}"
    return (
        directory / f"{stem}.csv",
        directory / f"{stem}_manifest.csv",
        directory / f"{stem}_metadata.json",
    )


def build_matrix(
    ranges: Mapping[int, tuple[str | float | Decimal, str | float | Decimal]],
    *,
    step: str | float | Decimal,
    workload_root: Path,
    output_root: Path,
    seed: int,
) -> list[MatrixCase]:
    cases: list[MatrixCase] = []
    for capacity_gb, (minimum, maximum) in sorted(ranges.items()):
        if not isinstance(capacity_gb, int) or capacity_gb < 0:
            raise ValueError(f"CPU capacity must be a nonnegative integer GB value: {capacity_gb!r}")
        for rate in decimal_range(minimum, maximum, step):
            tag = rate_label(rate)
            cap = capacity_label(capacity_gb)
            workload, manifest, metadata = workload_paths(workload_root, rate, seed)
            case_dir = output_root / tag / cap / "r1"
            cases.append(
                MatrixCase(
                    capacity_gb=capacity_gb,
                    rate=rate,
                    rate_label=tag,
                    capacity_label=cap,
                    workload=workload,
                    manifest=manifest,
                    metadata=metadata,
                    output_dir=case_dir,
                    config_path=case_dir / "config.input.json",
                )
            )
    return sorted(cases, key=lambda case: (case.rate, case.capacity_gb))


def _json_read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_session_repetitions(
    cases: Sequence[MatrixCase],
    *,
    source_sessions_per_epoch: int,
    configured_repetitions: int | None = SESSION_REPETITIONS,
) -> int:
    if not cases:
        raise ValueError("CPU_CAPACITY_RATE_RANGES produced no cases")
    if source_sessions_per_epoch <= 0:
        raise ValueError("source_sessions_per_epoch must be positive")
    highest_rate = max(case.rate for case in cases)
    required = math.ceil(
        Decimal(str(SIMULATION_END_TIME_S)) * highest_rate
        / Decimal(source_sessions_per_epoch)
    )
    repetitions = required if configured_repetitions is None else configured_repetitions
    if repetitions <= 0:
        raise ValueError("SESSION_REPETITIONS must be positive when explicitly configured")
    injection_duration = Decimal(source_sessions_per_epoch * repetitions) / highest_rate
    if injection_duration < Decimal(SIMULATION_END_TIME_S):
        raise ValueError(
            f"SESSION_REPETITIONS={repetitions} injects only through "
            f"{float(injection_duration):.1f}s at {highest_rate}/s; use at least {required} "
            f"repetitions for a {SIMULATION_END_TIME_S:g}s continuous run"
        )
    return repetitions


def _converter_command(args: argparse.Namespace, rate: Decimal, workload: Path,
                       manifest: Path, metadata: Path,
                       session_repetitions: int) -> list[str]:
    command = [
        str(args.python),
        str(HERE / "convert_tracelab_workload.py"),
        "--db",
        str(args.tracelab_db),
        "--output",
        str(workload),
        "--manifest-output",
        str(manifest),
        "--metadata-output",
        str(metadata),
        "--seed",
        str(SEED),
        "--session-arrival-rate",
        format(rate, "f"),
        "--session-repetitions",
        str(session_repetitions),
    ]
    if SAMPLE_SESSIONS is not None:
        command.extend(("--sample-sessions", str(SAMPLE_SESSIONS)))
    return command


def _workload_is_compatible(
    metadata: Path, rate: Decimal, session_repetitions: int
) -> bool:
    if not metadata.is_file():
        return False
    try:
        payload = _json_read(metadata)
        sampling = payload.get("sampling", {})
        return bool(
            payload.get("converter_version") == CONVERTER_VERSION
            and isinstance(sampling, Mapping)
            and sampling.get("sample_sessions") == SAMPLE_SESSIONS
            and sampling.get("seed") == SEED
            and sampling.get("session_repetitions") == session_repetitions
            and Decimal(str(sampling.get("session_arrival_rate_per_second"))) == rate
        )
    except (OSError, ValueError, json.JSONDecodeError, InvalidOperation):
        return False


def ensure_workloads(
    args: argparse.Namespace,
    cases: Sequence[MatrixCase],
    *,
    session_repetitions: int,
) -> None:
    unique: dict[Decimal, MatrixCase] = {}
    for case in cases:
        unique.setdefault(case.rate, case)
    for rate, case in sorted(unique.items()):
        complete = case.workload.is_file() and case.manifest.is_file() and _workload_is_compatible(
            case.metadata, rate, session_repetitions
        )
        if complete and not args.regenerate_workloads:
            print(f"[workload {case.rate_label}] reused", flush=True)
            continue
        command = _converter_command(
            args,
            rate,
            case.workload,
            case.manifest,
            case.metadata,
            session_repetitions,
        )
        if args.dry_run:
            print("[workload] " + subprocess.list2cmdline(command))
            continue
        if args.no_generate_workloads:
            raise FileNotFoundError(f"missing or incompatible workload for {case.rate_label}")
        case.workload.parent.mkdir(parents=True, exist_ok=True)
        print(f"[workload {case.rate_label}] generating", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        if not (
            case.workload.is_file()
            and case.manifest.is_file()
            and _workload_is_compatible(case.metadata, rate, session_repetitions)
        ):
            raise RuntimeError(f"converter did not produce a compatible workload for {case.rate_label}")


def build_config(template: Mapping[str, Any], case: MatrixCase) -> dict[str, Any]:
    config = copy.deepcopy(dict(template))
    cpu = config.setdefault("cpu_kv_cache", {})
    if not isinstance(cpu, dict):
        raise ValueError("cpu_kv_cache must be a JSON object")
    cpu["enabled"] = case.capacity_gb > 0
    cpu["capacity_bytes"] = case.capacity_gb * DECIMAL_GB
    cpu["capacity_bytes_per_gpu"] = max(
        DECIMAL_GB, case.capacity_gb * DECIMAL_GB // 2
    )
    config["run_id"] = (
        f"kimi-k2-tracelab-vera-rubin-p8-d16-{case.rate_label}-{case.capacity_label}"
    )
    return config


def _run_is_complete(case: MatrixCase) -> bool:
    required = (
        case.output_dir / "summary.json",
        case.output_dir / "requests.csv",
        case.output_dir / "config.normalized.json",
        case.output_dir / "run.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        return _json_read(case.output_dir / "run.json").get("status") == "completed"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _simulator_command(args: argparse.Namespace, case: MatrixCase) -> list[str]:
    return [
        str(args.binary),
        "--config",
        str(case.config_path),
        "--workload",
        str(case.workload),
        "--output-dir",
        str(case.output_dir),
        "--output-mode",
        "requests",
        "--runtime-validation",
        "true" if args.runtime_validation else "false",
        "--gpu-kv-occupancy",
        "true" if args.gpu_kv_occupancy else "false",
        "--simulation-end-time-s",
        format(SIMULATION_END_TIME_S, "g"),
        "--progress-interval-s",
        format(PROGRESS_INTERVAL_S, "g"),
    ]


def _case_start_message(
    case: MatrixCase,
    *,
    source_sessions_per_epoch: int,
    session_repetitions: int,
) -> str:
    return (
        f"[running] {case.label} rate={format(case.rate, 'f')}/s "
        f"physical_cpu={case.capacity_gb}GB "
        f"per_prefill_gpu={case.capacity_gb / 2.0:g}GB "
        f"source_sessions={source_sessions_per_epoch} "
        f"epochs={session_repetitions} "
        f"horizon={SIMULATION_END_TIME_S / 3600.0:g}h"
    )


def _case_progress_message(
    case: MatrixCase,
    *,
    simulation_time_s: float,
    wall_seconds: float,
) -> str:
    simulated_hours = simulation_time_s / 3600.0
    horizon_hours = SIMULATION_END_TIME_S / 3600.0
    percent = 100.0 * simulation_time_s / SIMULATION_END_TIME_S
    return (
        f"[progress] {case.label} simulated={simulated_hours:g}/{horizon_hours:g}h "
        f"({percent:.0f}%) wall={wall_seconds:.1f}s"
    )


def run_case(
    args: argparse.Namespace,
    template: Mapping[str, Any],
    case: MatrixCase,
    *,
    source_sessions_per_epoch: int,
    session_repetitions: int,
) -> tuple[str, bool]:
    if args.resume and _run_is_complete(case):
        print(f"[reused] {case.label}", flush=True)
        return case.label, True
    command = _simulator_command(args, case)
    if args.dry_run:
        print(f"[case {case.label}] " + subprocess.list2cmdline(command))
        return case.label, True
    case.output_dir.mkdir(parents=True, exist_ok=True)
    _json_write(case.config_path, build_config(template, case))
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "capacity_gb": case.capacity_gb,
        "cpu_dram_gb_per_prefill_gpu": case.capacity_gb / 2.0,
        "session_arrival_rate_per_second": float(case.rate),
        "rate_label": case.rate_label,
        "simulation_end_time_s": SIMULATION_END_TIME_S,
        "sample_sessions": SAMPLE_SESSIONS,
        "source_sessions_per_epoch": source_sessions_per_epoch,
        "session_repetitions": session_repetitions,
        "workload_csv": str(case.workload.resolve()),
        "workload_metadata": str(case.metadata.resolve()),
        "config_input": str(case.config_path.resolve()),
        "binary": str(args.binary.resolve()),
        "command": command,
        "started_at_unix_s": time.time(),
    }
    _json_write(case.output_dir / "run.json", record)
    started = time.perf_counter()
    print(
        _case_start_message(
            case,
            source_sessions_per_epoch=source_sessions_per_epoch,
            session_repetitions=session_repetitions,
        ),
        flush=True,
    )
    with (case.output_dir / "simulator.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("simulator stdout pipe was not created")
        for line in process.stdout:
            log.write(line)
            if not line.startswith("simulation_progress_s="):
                continue
            log.flush()
            try:
                simulation_time_s = float(line.split("=", 1)[1].strip())
            except ValueError:
                continue
            wall_seconds = time.perf_counter() - started
            record["last_progress_simulation_time_s"] = simulation_time_s
            record["last_progress_wall_clock_seconds"] = wall_seconds
            _json_write(case.output_dir / "run.json", record)
            print(
                _case_progress_message(
                    case,
                    simulation_time_s=simulation_time_s,
                    wall_seconds=wall_seconds,
                ),
                flush=True,
            )
        returncode = process.wait()
    record.update(
        {
            "status": "completed" if returncode == 0 else "failed",
            "exit_code": returncode,
            "finished_at_unix_s": time.time(),
            "process_wall_clock_seconds": time.perf_counter() - started,
        }
    )
    _json_write(case.output_dir / "run.json", record)
    print(
        f"[{record['status']}] {case.label} wall={record['process_wall_clock_seconds']:.1f}s",
        flush=True,
    )
    return case.label, returncode == 0


def _parse_capacity_filter(text: str) -> set[int]:
    try:
        values = {int(item.strip()) for item in text.split(",") if item.strip()}
    except ValueError as error:
        raise argparse.ArgumentTypeError("capacities must be comma-separated integer GB values") from error
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("capacities must be nonnegative and non-empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=_default_binary())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tracelab-db", type=Path, default=DEFAULT_TRACELAB_DB)
    parser.add_argument("--workload-root", type=Path, default=DEFAULT_WORKLOAD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--jobs",
        type=int,
        default=MAX_CONCURRENT_SIMULATIONS,
        help="maximum concurrent simulator processes",
    )
    parser.add_argument(
        "--capacities-gb",
        type=_parse_capacity_filter,
        help="optional subset of the capacities declared at the top of this file",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-generate-workloads", action="store_true")
    parser.add_argument("--regenerate-workloads", action="store_true")
    parser.add_argument("--generate-reports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-validation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu-kv-occupancy", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    full_matrix = build_matrix(
        CPU_CAPACITY_RATE_RANGES,
        step=RATE_STEP,
        workload_root=args.workload_root.resolve(),
        output_root=args.output_root.resolve(),
        seed=SEED,
    )
    ranges = CPU_CAPACITY_RATE_RANGES
    if args.capacities_gb is not None:
        unknown = sorted(args.capacities_gb - set(ranges))
        if unknown:
            raise SystemExit(f"capacities not declared in CPU_CAPACITY_RATE_RANGES: {unknown}")
        ranges = {capacity: bounds for capacity, bounds in ranges.items() if capacity in args.capacities_gb}
    cases = build_matrix(
        ranges,
        step=RATE_STEP,
        workload_root=args.workload_root.resolve(),
        output_root=args.output_root.resolve(),
        seed=SEED,
    )
    for path, label in ((args.config, "config"), (args.python, "Python")):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    if not args.dry_run and not args.binary.is_file():
        raise SystemExit(f"missing simulator binary: {args.binary}")
    if not args.tracelab_db.is_file():
        raise SystemExit(f"missing TraceLab database: {args.tracelab_db}")

    eligible_source_sessions = count_eligible_source_sessions(args.tracelab_db.resolve())
    source_sessions_per_epoch = (
        eligible_source_sessions
        if SAMPLE_SESSIONS is None
        else min(SAMPLE_SESSIONS, eligible_source_sessions)
    )
    session_repetitions = resolve_session_repetitions(
        full_matrix,
        source_sessions_per_epoch=source_sessions_per_epoch,
    )
    # Size every workload against the full declared grid, even when one server
    # invocation uses --capacities-gb.  This keeps shared rate CSVs identical
    # across split/resumed batches.
    highest_rate = max(case.rate for case in full_matrix)
    injection_duration_s = (
        Decimal(source_sessions_per_epoch * session_repetitions) / highest_rate
    )
    session_scope = (
        f"all {source_sessions_per_epoch} eligible source sessions"
        if SAMPLE_SESSIONS is None
        else f"sampled {source_sessions_per_epoch} of {eligible_source_sessions} eligible source sessions"
    )
    print(
        f"[workload plan] {session_scope}; "
        f"{session_repetitions} epoch(s); injection through "
        f"{float(injection_duration_s):.1f}s at max rate {highest_rate}/s",
        flush=True,
    )
    ensure_workloads(
        args,
        cases,
        session_repetitions=session_repetitions,
    )
    template = _json_read(args.config)
    if not args.dry_run:
        workload_hashes: dict[str, str] = {}
        for case in cases:
            if case.rate_label not in workload_hashes:
                workload_hashes[case.rate_label] = _sha256(case.workload.resolve())
        plan = {
            "schema_version": 1,
            "study": "kimi_k2_p8_d16_dense_cpu_capacity_session_rate_10h",
            "git_revision": _git_revision(),
            "rate_step_per_second": RATE_STEP,
            "capacity_rate_ranges": {str(key): list(value) for key, value in ranges.items()},
            "capacity_mapping": "one physical CPU DRAM pool per two PREFILL GPUs; configured static slice = physical capacity / 2 per PREFILL GPU",
            "prefill_gpus": 8,
            "decode_gpus": 16,
            "simulation_end_time_s": SIMULATION_END_TIME_S,
            "sample_sessions": SAMPLE_SESSIONS,
            "eligible_source_sessions": eligible_source_sessions,
            "source_sessions_per_epoch": source_sessions_per_epoch,
            "session_repetitions": session_repetitions,
            "session_repetitions_configured": SESSION_REPETITIONS,
            "injection_duration_at_max_rate_s": float(injection_duration_s),
            "seed": SEED,
            "jobs": args.jobs,
            "config": str(args.config.resolve()),
            "config_sha256": _sha256(args.config.resolve()),
            "binary": str(args.binary.resolve()),
            "binary_sha256": _sha256(args.binary.resolve()),
            "tracelab_db": str(args.tracelab_db.resolve()),
            "tracelab_db_sha256": _sha256(args.tracelab_db.resolve()),
            "cases": [
                {
                    "rate": float(case.rate),
                    "rate_decimal": format(case.rate, "f"),
                    "rate_label": case.rate_label,
                    "capacity_gb": case.capacity_gb,
                    "cpu_dram_gb_per_prefill_gpu": case.capacity_gb / 2.0,
                    "output_dir": str(case.output_dir.resolve()),
                    "workload_csv": str(case.workload.resolve()),
                    "workload_metadata": str(case.metadata.resolve()),
                    "workload_sha256": workload_hashes[case.rate_label],
                }
                for case in cases
            ],
        }
        _json_write(args.output_root.resolve() / "sweep_plan.json", plan)

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_case,
                args,
                template,
                case,
                source_sessions_per_epoch=source_sessions_per_epoch,
                session_repetitions=session_repetitions,
            ): case
            for case in cases
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                _, success = future.result()
                if not success:
                    failures.append(case.label)
            except Exception as error:  # noqa: BLE001 - retain every failed case
                failures.append(case.label)
                print(f"[failed] {case.label}: {error}", file=sys.stderr, flush=True)

    if args.generate_reports and not args.dry_run:
        try:
            from generate_dense_cpu_capacity_rate_reports import generate_reports

            result = generate_reports(args.output_root.resolve())
            print(json.dumps(result, indent=2), flush=True)
        except Exception as error:  # noqa: BLE001 - simulation results remain resumable
            failures.append("report-generation")
            print(f"[failed] report generation: {error}", file=sys.stderr, flush=True)
    if failures:
        print("failed cases: " + ", ".join(sorted(failures)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
