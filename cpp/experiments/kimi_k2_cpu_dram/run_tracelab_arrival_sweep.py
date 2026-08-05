#!/usr/bin/env python3
"""Run the TraceLab cache-aware session-arrival sweep on the C++ simulator.

The default matrix replays the same deterministic 3,000-source-session sample
twice at 0.10, 0.09, ..., 0.05 new source sessions/second.  Missing workload
rates are generated with ``convert_tracelab_workload.py``; existing workload
metadata is validated before reuse.  Each rate gets an immutable input config,
run record, simulator log, compact request metrics, and GPU KV occupancy data.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CONFIG = HERE / "configs" / "tracelab_p8_d24_cache_aware.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "tracelab_cache_aware_arrival_sweep"
DEFAULT_WORKLOAD_ROOT = (
    REPO_ROOT / "outputs" / "datasets" / "tracelab" / "v0.0.2" / "frontier" / "epoch3000"
)
DEFAULT_TRACELAB_DB = (
    REPO_ROOT / "outputs" / "datasets" / "tracelab" / "v0.0.2" / "syfi_coding_trace.duckdb"
)
DEFAULT_RATES = "0.10,0.09,0.08,0.07,0.06,0.05"


@dataclass(frozen=True)
class SweepCase:
    rate: float
    label: str
    workload: Path
    manifest: Path
    metadata: Path
    output_dir: Path
    config_path: Path
    simulation_end_time_s: float


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return value


def parse_rates(text: str) -> list[float]:
    result: list[float] = []
    seen: set[Decimal] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            decimal = Decimal(item)
        except InvalidOperation as error:
            raise argparse.ArgumentTypeError(f"invalid rate: {item}") from error
        if not decimal.is_finite() or decimal <= 0:
            raise argparse.ArgumentTypeError(f"rate must be positive: {item}")
        if decimal in seen:
            raise argparse.ArgumentTypeError(f"duplicate rate: {item}")
        seen.add(decimal)
        result.append(float(decimal))
    if not result:
        raise argparse.ArgumentTypeError("rates must be a non-empty comma-separated list")
    return result


def rate_label(rate: float) -> str:
    decimal = Decimal(str(rate)).normalize()
    if decimal <= 0:
        raise ValueError("rate must be positive")
    return "r" + format(decimal, "f").replace("-", "m").replace(".", "p")


def simulation_end_time_s(
    *,
    sample_sessions: int,
    rate: float,
    repetitions: int,
    drain_fraction: float,
    minimum_drain_seconds: float,
) -> float:
    """Return two injection epochs plus a rate-scaled settling window.

    With the defaults this reproduces the previously used horizons at the
    anchor rates: 144,000 s for 0.05, 90,000 s for 0.08, and 75,000 s for 0.10.
    """

    if sample_sessions <= 0 or repetitions <= 0:
        raise ValueError("sample_sessions and repetitions must be positive")
    if rate <= 0.0 or drain_fraction < 0.0 or minimum_drain_seconds < 0.0:
        raise ValueError("rate must be positive and drain settings nonnegative")
    epoch = sample_sessions / rate
    drain = max(minimum_drain_seconds, drain_fraction * epoch)
    return repetitions * epoch + drain


def _json_read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def workload_paths(root: Path, rate: float, seed: int) -> tuple[Path, Path, Path]:
    directory = root / rate_label(rate)
    stem = f"seed_{seed}"
    return (
        directory / f"{stem}.csv",
        directory / f"{stem}_manifest.csv",
        directory / f"{stem}_metadata.json",
    )


def validate_workload_metadata(
    path: Path,
    *,
    rate: float,
    seed: int,
    sample_sessions: int,
    repetitions: int,
) -> None:
    metadata = _json_read(path)
    sampling = metadata.get("sampling", {})
    if not isinstance(sampling, dict):
        raise ValueError(f"invalid sampling metadata: {path}")
    expected = {
        "sample_sessions": sample_sessions,
        "seed": seed,
        "session_repetitions": repetitions,
    }
    mismatches = [
        f"{key}={sampling.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if sampling.get(key) != value
    ]
    actual_rate = sampling.get("session_arrival_rate_per_second")
    if actual_rate is None or not math.isclose(float(actual_rate), rate, rel_tol=0.0, abs_tol=1e-12):
        mismatches.append(f"session_arrival_rate_per_second={actual_rate!r} (expected {rate!r})")
    if mismatches:
        raise ValueError(f"workload metadata mismatch in {path}: " + "; ".join(mismatches))


def converter_command(
    *,
    python: Path,
    database: Path,
    workload: Path,
    manifest: Path,
    metadata: Path,
    rate: float,
    seed: int,
    sample_sessions: int,
    repetitions: int,
) -> list[str]:
    return [
        str(python),
        str(HERE / "convert_tracelab_workload.py"),
        "--db",
        str(database),
        "--output",
        str(workload),
        "--manifest-output",
        str(manifest),
        "--metadata-output",
        str(metadata),
        "--sample-sessions",
        str(sample_sessions),
        "--seed",
        str(seed),
        "--session-arrival-rate",
        format(rate, ".12g"),
        "--session-repetitions",
        str(repetitions),
    ]


def ensure_workload(args: argparse.Namespace, rate: float) -> tuple[Path, Path, Path]:
    workload, manifest, metadata = workload_paths(args.workload_root, rate, args.seed)
    complete = workload.is_file() and manifest.is_file() and metadata.is_file()
    if complete and not args.regenerate_workloads:
        validate_workload_metadata(
            metadata,
            rate=rate,
            seed=args.seed,
            sample_sessions=args.sample_sessions,
            repetitions=args.session_repetitions,
        )
        return workload, manifest, metadata
    command = converter_command(
        python=args.converter_python,
        database=args.tracelab_db,
        workload=workload,
        manifest=manifest,
        metadata=metadata,
        rate=rate,
        seed=args.seed,
        sample_sessions=args.sample_sessions,
        repetitions=args.session_repetitions,
    )
    if args.dry_run:
        print("[workload] " + subprocess.list2cmdline(command))
        return workload, manifest, metadata
    if args.no_generate_workloads:
        missing = [str(path) for path in (workload, manifest, metadata) if not path.is_file()]
        raise FileNotFoundError("missing workload artifacts: " + ", ".join(missing))
    if not args.tracelab_db.is_file():
        raise FileNotFoundError(f"TraceLab database not found: {args.tracelab_db}")
    workload.parent.mkdir(parents=True, exist_ok=True)
    print(f"[workload {rate_label(rate)}] generating", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    validate_workload_metadata(
        metadata,
        rate=rate,
        seed=args.seed,
        sample_sessions=args.sample_sessions,
        repetitions=args.session_repetitions,
    )
    return workload, manifest, metadata


def build_config(template: dict[str, Any], args: argparse.Namespace, rate: float) -> dict[str, Any]:
    config = json.loads(json.dumps(template))
    scheduler = config.setdefault("cluster_scheduler", {})
    scheduler.update(
        {
            "type": "sticky_round_robin",
            "prefill_type": "cache_aware",
            "decode_type": "vllm_queue_aware",
            "cache_threshold": args.cache_threshold,
            "balance_abs_threshold": args.balance_abs_threshold,
            "balance_rel_threshold": args.balance_rel_threshold,
        }
    )
    config.setdefault("cpu_kv_cache", {})["enabled"] = False
    config["run_id"] = (
        f"kimi-k2-tracelab-{rate_label(rate)}-p8-d24-cache-aware-"
        f"c{args.cache_threshold:g}-a{args.balance_abs_threshold}-r{args.balance_rel_threshold:g}"
    )
    return config


def simulator_command(args: argparse.Namespace, case: SweepCase) -> list[str]:
    return [
        str(args.binary),
        "--config",
        str(case.config_path),
        "--workload",
        str(case.workload),
        "--output-dir",
        str(case.output_dir),
        "--output-mode",
        args.output_mode,
        "--runtime-validation",
        "true" if args.runtime_validation else "false",
        "--gpu-kv-occupancy",
        "true" if args.gpu_kv_occupancy else "false",
        "--simulation-end-time-s",
        format(case.simulation_end_time_s, ".12g"),
    ]


def run_is_complete(case: SweepCase, output_mode: str) -> bool:
    required = [case.output_dir / "summary.json", case.output_dir / "config.normalized.json"]
    if output_mode in {"requests", "full"}:
        required.append(case.output_dir / "requests.csv")
    if output_mode == "full":
        required.append(case.output_dir / "trace.json")
    run_path = case.output_dir / "run.json"
    if not run_path.is_file() or not all(path.is_file() for path in required):
        return False
    try:
        return _json_read(run_path).get("status") == "completed"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def run_case(args: argparse.Namespace, case: SweepCase, config: dict[str, Any]) -> tuple[str, bool]:
    if args.resume and run_is_complete(case, args.output_mode):
        print(f"[{case.label}] already complete; skipped", flush=True)
        return case.label, True
    command = simulator_command(args, case)
    if args.dry_run:
        print(f"[{case.label}] end={case.simulation_end_time_s:.3f}s " + subprocess.list2cmdline(command))
        return case.label, True
    case.output_dir.mkdir(parents=True, exist_ok=True)
    _json_write(case.config_path, config)
    run_path = case.output_dir / "run.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "rate_label": case.label,
        "session_arrival_rate_per_second": case.rate,
        "sample_sessions": args.sample_sessions,
        "session_repetitions": args.session_repetitions,
        "simulation_end_time_s": case.simulation_end_time_s,
        "drain_fraction": args.drain_fraction,
        "minimum_drain_seconds": args.minimum_drain_seconds,
        "cache_threshold": args.cache_threshold,
        "balance_abs_threshold": args.balance_abs_threshold,
        "balance_rel_threshold": args.balance_rel_threshold,
        "workload_csv": str(case.workload.resolve()),
        "workload_manifest": str(case.manifest.resolve()),
        "workload_metadata": str(case.metadata.resolve()),
        "config_input": str(case.config_path.resolve()),
        "binary": str(args.binary.resolve()),
        "command": command,
        "started_at_unix_s": time.time(),
    }
    _json_write(run_path, record)
    started = time.perf_counter()
    log_path = case.output_dir / "simulator.log"
    print(f"[{case.label}] running (end={case.simulation_end_time_s:.3f}s)", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    record["process_wall_clock_seconds"] = time.perf_counter() - started
    record["finished_at_unix_s"] = time.time()
    record["exit_code"] = result.returncode
    record["status"] = "completed" if result.returncode == 0 else "failed"
    _json_write(run_path, record)
    print(
        f"[{case.label}] {record['status']} in {record['process_wall_clock_seconds']:.1f}s",
        flush=True,
    )
    return case.label, result.returncode == 0


def _default_binary() -> Path:
    names = (
        REPO_ROOT / "cpp" / "build" / "Release" / "frontier_sim.exe",
        REPO_ROOT / "cpp" / "build" / "frontier_sim",
        REPO_ROOT / "build" / "cpp" / "frontier_sim",
    )
    return next((path for path in names if path.is_file()), names[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=_default_binary())
    parser.add_argument("--config-template", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workload-root", type=Path, default=DEFAULT_WORKLOAD_ROOT)
    parser.add_argument("--tracelab-db", type=Path, default=DEFAULT_TRACELAB_DB)
    parser.add_argument("--converter-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--rates", type=parse_rates, default=parse_rates(DEFAULT_RATES))
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--sample-sessions", type=int, default=3000)
    parser.add_argument("--session-repetitions", type=int, default=2)
    parser.add_argument("--cache-threshold", type=float, default=0.5)
    parser.add_argument("--balance-abs-threshold", type=int, default=8)
    parser.add_argument("--balance-rel-threshold", type=float, default=1.5)
    parser.add_argument("--drain-fraction", type=_nonnegative_float, default=0.4)
    parser.add_argument("--minimum-drain-seconds", type=_nonnegative_float, default=15000.0)
    parser.add_argument(
        "--simulation-end-time-s",
        type=_positive_float,
        help="Override every computed horizon for a diagnostic smoke run.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent simulations (default: 1).")
    parser.add_argument("--output-mode", choices=("summary", "requests", "full"), default="requests")
    parser.add_argument("--runtime-validation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu-kv-occupancy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-generate-workloads", action="store_true")
    parser.add_argument("--regenerate-workloads", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_sessions <= 0 or args.session_repetitions <= 0 or args.jobs <= 0:
        raise SystemExit("sample-sessions, session-repetitions, and jobs must be positive")
    if not 0.0 <= args.cache_threshold <= 1.0:
        raise SystemExit("cache-threshold must be in [0, 1]")
    if args.balance_abs_threshold < 0 or args.balance_rel_threshold < 1.0:
        raise SystemExit("balance-abs-threshold must be nonnegative and balance-rel-threshold >= 1")
    if not args.config_template.is_file():
        raise SystemExit(f"config template not found: {args.config_template}")
    if not args.dry_run and not args.binary.is_file():
        raise SystemExit(f"simulator binary not found: {args.binary}")
    template = _json_read(args.config_template)
    prepared: list[tuple[SweepCase, dict[str, Any]]] = []
    for rate in args.rates:
        workload, manifest, metadata = ensure_workload(args, rate)
        label = rate_label(rate)
        output_dir = args.output_root / label
        case = SweepCase(
            rate=rate,
            label=label,
            workload=workload,
            manifest=manifest,
            metadata=metadata,
            output_dir=output_dir,
            config_path=output_dir / "config.input.json",
            simulation_end_time_s=(
                args.simulation_end_time_s
                if args.simulation_end_time_s is not None
                else simulation_end_time_s(
                    sample_sessions=args.sample_sessions,
                    rate=rate,
                    repetitions=args.session_repetitions,
                    drain_fraction=args.drain_fraction,
                    minimum_drain_seconds=args.minimum_drain_seconds,
                )
            ),
        )
        prepared.append((case, build_config(template, args, rate)))
    if not args.dry_run:
        _json_write(
            args.output_root / "sweep_plan.json",
            {
                "schema_version": 1,
                "config_template": str(args.config_template.resolve()),
                "binary": str(args.binary.resolve()),
                "seed": args.seed,
                "sample_sessions": args.sample_sessions,
                "session_repetitions": args.session_repetitions,
                "rates": args.rates,
                "cache_threshold": args.cache_threshold,
                "balance_abs_threshold": args.balance_abs_threshold,
                "balance_rel_threshold": args.balance_rel_threshold,
                "drain_fraction": args.drain_fraction,
                "minimum_drain_seconds": args.minimum_drain_seconds,
                "diagnostic_simulation_end_time_override_s": args.simulation_end_time_s,
                "output_mode": args.output_mode,
                "runtime_validation": args.runtime_validation,
                "gpu_kv_occupancy": args.gpu_kv_occupancy,
                "jobs": args.jobs,
                "cases": [
                    {
                        "rate_label": case.label,
                        "session_arrival_rate_per_second": case.rate,
                        "simulation_end_time_s": case.simulation_end_time_s,
                        "workload_csv": str(case.workload.resolve()),
                        "workload_manifest": str(case.manifest.resolve()),
                        "workload_metadata": str(case.metadata.resolve()),
                        "output_dir": str(case.output_dir.resolve()),
                    }
                    for case, _ in prepared
                ],
            },
        )
    failures: list[str] = []
    if args.jobs == 1:
        for case, config in prepared:
            _, success = run_case(args, case, config)
            if not success:
                failures.append(case.label)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(run_case, args, case, config): case.label
                for case, config in prepared
            }
            for future in as_completed(futures):
                label, success = future.result()
                if not success:
                    failures.append(label)
    if failures:
        print("failed rates: " + ", ".join(sorted(failures)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
