#!/usr/bin/env python3
"""Reconstruct request arrivals and exogenous session starts over time.

Request metrics contain completed requests.  Because a Frontier session is
closed-loop, at most one additional request per session can have arrived but
remain incomplete at the simulation horizon.  This script reconstructs that
request from the workload think time, so the final arrival buckets are not
censored by completion-only metrics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_WORKLOAD = (
    REPO_ROOT / "outputs" / "datasets" / "tracelab" / "v0.0.2" / "frontier"
    / "epoch1000_rep4" / "r1" / "seed_20260803.csv"
)
DEFAULT_METADATA = DEFAULT_WORKLOAD.with_name("seed_20260803_metadata.json")
DEFAULT_RUN = (
    REPO_ROOT / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu_capacity_sweep_tiered_discard_20260808"
    / "cpu4000gb" / "r1"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--bucket-seconds", type=float, default=300.0)
    parser.add_argument("--simulation-end-time-s", type=float, default=19_000.0)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    think_times: dict[int, list[float]] = defaultdict(list)
    session_starts: dict[int, float] = {}
    with args.workload.resolve().open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            session_id = int(row["session_id"])
            think_times[session_id].append(float(row["think_time"]))
            if row["session_start_at"]:
                session_starts[session_id] = float(row["session_start_at"])

    completed: dict[int, list[tuple[float, float]]] = defaultdict(list)
    completed_arrivals: list[float] = []
    with (args.run_dir.resolve() / "requests.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            session_id = int(row["session_id"])
            arrived_at = float(row["arrived_at_s"])
            completed_at = float(row["completed_at_s"])
            completed[session_id].append((arrived_at, completed_at))
            completed_arrivals.append(arrived_at)

    incomplete_arrivals: list[float] = []
    for session_id, turns in think_times.items():
        records = sorted(completed.get(session_id, []))
        completed_count = len(records)
        if completed_count == 0:
            next_arrival = session_starts[session_id]
        elif completed_count < len(turns):
            next_arrival = records[-1][1] + turns[completed_count]
        else:
            continue
        if next_arrival < args.simulation_end_time_s:
            incomplete_arrivals.append(next_arrival)

    metadata = json.loads(args.metadata.resolve().read_text(encoding="utf-8"))
    root_starts: list[float] = []
    split_starts: list[float] = []
    for session in metadata["session_map"]:
        start = float(session["segment_root_arrival_seconds"])
        if start >= args.simulation_end_time_s:
            continue
        if int(session["segment_index"]) == 0:
            root_starts.append(start)
        else:
            split_starts.append(start)

    bucket_count = int((args.simulation_end_time_s + args.bucket_seconds - 1) // args.bucket_seconds)
    rows: list[dict[str, Any]] = []
    all_arrivals = completed_arrivals + incomplete_arrivals
    for index in range(bucket_count):
        start = index * args.bucket_seconds
        end = min(args.simulation_end_time_s, start + args.bucket_seconds)
        duration = end - start
        def count(values: list[float]) -> int:
            return sum(start <= value < end for value in values)
        total = count(all_arrivals)
        roots = count(root_starts)
        splits = count(split_starts)
        rows.append(
            {
                "start_time_s": start,
                "end_time_s": end,
                "total_request_arrivals": total,
                "total_request_arrivals_per_s": total / duration,
                "source_root_session_starts": roots,
                "source_root_session_starts_per_s": roots / duration,
                "split_session_starts": splits,
                "split_session_starts_per_s": splits / duration,
                "all_simulator_session_starts_per_s": (roots + splits) / duration,
            }
        )

    output = args.output_csv or args.run_dir.resolve() / "request_arrivals_5min.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "total_arrivals_within_horizon": len(all_arrivals),
                "incomplete_arrivals_reconstructed": len(incomplete_arrivals),
                "source_root_injections": len(root_starts),
                "last_source_root_injection_s": max(root_starts),
                "split_session_starts_within_horizon": len(split_starts),
                "last_split_session_start_within_horizon_s": max(split_starts),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
