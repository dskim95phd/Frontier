#!/usr/bin/env python3
"""Derive time-bucketed PREFILL GPU pool utilization from sweep summaries.

Each PREFILL batch occupies one DP target because the P8 configuration uses
TP=PP=1 and DP=8.  Summed predicted batch execution time divided by the eight
GPU-seconds available in a time bucket is therefore the pool's busy-equivalent
utilization.  This is a simulator service-time utilization metric, not hardware
MFU.  Five-minute buckets reduce the small boundary error caused by assigning a
whole batch to the bucket in which it was scheduled.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "tracelab_vera_rubin_p8_d16_cpu_capacity_sweep_tiered_discard_20260808"
)
PREFILL_GPUS = 8
OUTPUT_BUCKET_SECONDS = 300
COMPUTE_COMPONENTS = (
    "dense_compute_ms",
    "lm_head_ms",
    "moe_gating_linear_ms",
    "moe_gating_routing_topk_ms",
    "moe_grouped_gemm_ms",
    "moe_post_attention_norm_ms",
)


def aggregate_case(case_dir: Path, capacity_gb: int) -> list[dict[str, Any]]:
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    simulation_end = float(summary["simulation_window_seconds"])
    buckets: dict[int, dict[str, float]] = {}
    for source in summary["batch_summary_by_cluster_time_bucket"]["PREFILL"]:
        bucket = int(float(source["start_time_s"]) // OUTPUT_BUCKET_SECONDS)
        target = buckets.setdefault(
            bucket,
            {"busy_ms": 0.0, "compute_ms": 0.0, "attention_pairs": 0.0},
        )
        target["busy_ms"] += float(source["predicted_execution_ms"])
        components = source["execution_time_components_ms"]
        target["compute_ms"] += sum(float(components.get(key, 0.0)) for key in COMPUTE_COMPONENTS)
        target["attention_pairs"] += float(source["prefill_attention_token_pairs"])

    rows: list[dict[str, Any]] = []
    for bucket, values in sorted(buckets.items()):
        start = bucket * OUTPUT_BUCKET_SECONDS
        duration = min(OUTPUT_BUCKET_SECONDS, simulation_end - start)
        if duration <= 0.0:
            continue
        available_gpu_ms = duration * 1000.0 * PREFILL_GPUS
        busy_gpus = values["busy_ms"] / (duration * 1000.0)
        rows.append(
            {
                "physical_cpu_dram_gb": capacity_gb,
                "start_time_s": start,
                "end_time_s": start + duration,
                "busy_equivalent_gpus": busy_gpus,
                "pool_busy_pct": 100.0 * values["busy_ms"] / available_gpu_ms,
                "compute_active_pct": 100.0 * values["compute_ms"] / available_gpu_ms,
                "prefill_attention_pairs_trillion": values["attention_pairs"] / 1e12,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.output_root.resolve()
    rows: list[dict[str, Any]] = []
    for case in sorted(root.glob("cpu*gb/r1")):
        summary = case / "summary.json"
        if not summary.is_file():
            continue
        capacity_gb = int(case.parent.name.removeprefix("cpu").removesuffix("gb"))
        rows.extend(aggregate_case(case, capacity_gb))
    if not rows:
        raise SystemExit(f"no completed cases under {root}")

    csv_path = root / "prefill_gpu_utilization_5min.csv"
    json_path = root / "prefill_gpu_utilization_5min.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
