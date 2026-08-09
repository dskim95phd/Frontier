#!/usr/bin/env python3
"""Sweep physical CPU DRAM capacity for the TraceLab P8/D16 experiment.

The physical capacity label follows the Vera/Rubin topology used by the
cpu4tb configuration: one CPU DRAM pool is shared by two PREFILL GPUs.  The
simulator models one static slice per PREFILL GPU, so a physical capacity of
X TB is encoded as X/2 TB per GPU (and is replicated over the eight PREFILL
DP lanes).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CONFIG = HERE / "configs" / "tracelab_vera_rubin_p8_d16_cache_aware_cpu4tb.json"
DEFAULT_WORKLOAD = (
    REPO_ROOT
    / "outputs"
    / "datasets"
    / "tracelab"
    / "v0.0.2"
    / "frontier"
    / "epoch1000_rep4"
    / "r1"
    / "seed_20260803.csv"
)
DEFAULT_BINARY = REPO_ROOT / "cpp" / "build" / "Release" / "frontier_sim.exe"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu_capacity_sweep_tiered_discard_20260808"
)
DEFAULT_CAPACITIES_GB = tuple(range(0, 4001, 500))
DECIMAL_GB = 1_000_000_000


def capacity_label(capacity_gb: int) -> str:
    return f"cpu{capacity_gb:04d}gb"


def build_config(template: dict[str, object], capacity_gb: int) -> dict[str, object]:
    config = copy.deepcopy(template)
    label = capacity_label(capacity_gb)
    config["run_id"] = f"kimi-k2-tracelab-vera-rubin-p8-d16-{label}"
    cpu = config["cpu_kv_cache"]
    assert isinstance(cpu, dict)
    cpu["enabled"] = capacity_gb > 0
    cpu["capacity_bytes"] = capacity_gb * DECIMAL_GB
    # The parser requires a positive slice even while the cache is disabled.
    cpu["capacity_bytes_per_gpu"] = max(DECIMAL_GB, capacity_gb * DECIMAL_GB // 2)
    return config


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_case(
    *,
    capacity_gb: int,
    binary: Path,
    workload: Path,
    output_root: Path,
    simulation_end_time_s: float,
    resume: bool,
) -> dict[str, object]:
    label = capacity_label(capacity_gb)
    case_dir = output_root / label / "r1"
    summary_path = case_dir / "summary.json"
    requests_path = case_dir / "requests.csv"
    if resume and summary_path.is_file() and requests_path.is_file():
        return {"capacity_gb": capacity_gb, "label": label, "status": "reused", "wall_seconds": 0.0}

    command = [
        str(binary),
        "--config",
        str(output_root / "configs" / f"{label}.json"),
        "--workload",
        str(workload),
        "--output-dir",
        str(case_dir),
        "--output-mode",
        "requests",
        "--runtime-validation",
        "false",
        "--gpu-kv-occupancy",
        "false",
        "--simulation-end-time-s",
        f"{simulation_end_time_s:g}",
    ]
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    record: dict[str, object] = {
        "capacity_gb": capacity_gb,
        "label": label,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "wall_seconds": wall_seconds,
        "command": command,
    }
    write_json(case_dir / "run.json", record)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}: {completed.stderr[-2000:]}")
    return record


def parse_capacities(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("capacities must be unique nonnegative GB values")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--capacities-gb",
        type=parse_capacities,
        default=list(DEFAULT_CAPACITIES_GB),
    )
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--simulation-end-time-s", type=float, default=19_000.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    workload = args.workload.resolve()
    binary = args.binary.resolve()
    output_root = args.output_root.resolve()
    for path, description in ((config_path, "config"), (workload, "workload"), (binary, "binary")):
        if not path.is_file():
            raise SystemExit(f"missing {description}: {path}")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")

    template = json.loads(config_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    capacities = sorted(args.capacities_gb)
    for capacity_gb in capacities:
        write_json(
            output_root / "configs" / f"{capacity_label(capacity_gb)}.json",
            build_config(template, capacity_gb),
        )
    write_json(
        output_root / "sweep.json",
        {
            "study": "p8_cpu_dram_capacity",
            "physical_capacity_gb": capacities,
            "capacity_mapping": "one physical CPU DRAM pool per two PREFILL GPUs; static simulator slice = physical capacity / 2 per GPU",
            "prefill_gpus": 8,
            "decode_gpus": 16,
            "workload": str(workload),
            "base_config": str(config_path),
            "simulation_end_time_s": args.simulation_end_time_s,
            "jobs": args.jobs,
        },
    )

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_case,
                capacity_gb=capacity_gb,
                binary=binary,
                workload=workload,
                output_root=output_root,
                simulation_end_time_s=args.simulation_end_time_s,
                resume=args.resume,
            ): capacity_gb
            for capacity_gb in capacities
        }
        for future in as_completed(futures):
            capacity_gb = futures[future]
            try:
                result = future.result()
                print(
                    f"[{result['status']}] {capacity_label(capacity_gb)} "
                    f"wall={float(result['wall_seconds']):.1f}s",
                    flush=True,
                )
            except Exception as error:  # noqa: BLE001 - preserve all case failures
                failures.append(f"{capacity_label(capacity_gb)}: {error}")
                print(f"[failed] {failures[-1]}", flush=True)
    if failures:
        raise SystemExit("\n".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
