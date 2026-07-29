#!/usr/bin/env python3
"""Run the server-oriented Kimi K2 NVL12 128K CPU-DRAM sweep.

Defaults:
  * CPU DRAM per Vera CPU: 0 through 1000 decimal GB, inclusive, in 200 GB steps
  * NVL12 mapping: 6 Vera CPUs, 2 Rubin GPUs per CPU
  * Workload: pressure128k, 144 closed-loop sessions, 2 sessions/s
  * Parallelism: up to 4 independent simulator worker processes
  * Outputs: CSV, JSON, per-run logs/metrics, and sweep_dashboard.html

Example:
    python scripts/run_kimi_k2_nvl12_pressure128k_server_sweep.py \
        --max-workers 4 \
        --output-dir outputs/kimi_k2_nvl12_pressure128k_server_sweep
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path
from typing import Sequence

try:
    from scripts.run_kimi_k2_nvl12_dram_sweep import main as run_sweep
except ModuleNotFoundError:
    from run_kimi_k2_nvl12_dram_sweep import main as run_sweep


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "kimi_k2_nvl12_pressure128k_server_sweep"
)


def build_capacity_range(
    start_gb: float,
    stop_gb: float,
    step_gb: float,
) -> list[float]:
    start = Decimal(str(start_gb))
    stop = Decimal(str(stop_gb))
    step = Decimal(str(step_gb))
    if start < 0:
        raise ValueError("capacity start must be non-negative")
    if stop < start:
        raise ValueError("capacity stop must be >= start")
    if step <= 0:
        raise ValueError("capacity step must be positive")

    capacities: list[float] = []
    current = start
    while current <= stop:
        capacities.append(float(current))
        current += step
    if Decimal(str(capacities[-1])) != stop:
        raise ValueError(
            "capacity range must land exactly on stop; "
            f"start={start}, stop={stop}, step={step}"
        )
    return capacities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capacity-start-gb",
        type=float,
        default=0.0,
        help="CPU DRAM capacity per Vera CPU in decimal GB.",
    )
    parser.add_argument(
        "--capacity-stop-gb",
        type=float,
        default=1000.0,
        help="Inclusive CPU DRAM capacity per Vera CPU in decimal GB.",
    )
    parser.add_argument(
        "--capacity-step-gb",
        type=float,
        default=200.0,
        help="CPU DRAM capacity step per Vera CPU in decimal GB.",
    )
    parser.add_argument("--sessions", type=int, default=144)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--session-arrival-rate", type=float, default=2.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one session at 0 and 200 GB per CPU with at most two workers. "
            "This validates multiprocessing, metrics collection, and graphs."
        ),
    )
    return parser


def _format_capacity(capacity: float) -> str:
    return f"{capacity:g}"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sessions <= 0:
        raise SystemExit("--sessions must be positive")
    if args.session_arrival_rate <= 0:
        raise SystemExit("--session-arrival-rate must be positive")
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")

    if args.smoke_test:
        capacities = [0.0, 200.0]
        sessions = 1
        max_workers = min(args.max_workers, 2)
    else:
        try:
            capacities = build_capacity_range(
                args.capacity_start_gb,
                args.capacity_stop_gb,
                args.capacity_step_gb,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        sessions = args.sessions
        max_workers = args.max_workers

    forwarded = [
        "--capacities-gb",
        ",".join(_format_capacity(item) for item in capacities),
        "--sessions",
        str(sessions),
        "--seed",
        str(args.seed),
        "--session-arrival-rate",
        str(args.session_arrival_rate),
        "--workload-profile",
        "pressure128k",
        "--max-workers",
        str(max_workers),
        "--output-dir",
        str(args.output_dir),
        "--python",
        str(args.python),
    ]
    if args.resume:
        forwarded.append("--resume")
    if args.dry_run:
        forwarded.append("--dry-run")
    if args.no_dashboard:
        forwarded.append("--no-dashboard")
    return run_sweep(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
