"""Focused checks for the detailed TraceLab sweep collector."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import collect_tracelab_arrival_sweep as collector  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_collects_cache_migration_idle_lanes_and_occupancy(tmp_path: Path) -> None:
    run_dir = tmp_path / "r0p05"
    run_dir.mkdir()
    manifest = run_dir / "manifest.csv"
    _write_csv(
        manifest,
        [
            {
                "request_id": 0,
                "session_id": 7,
                "theoretical_min_prefill_tokens": 10,
                "implementation_min_prefill_tokens": 10,
                "materialized_prefill_tokens": 10,
            },
            {
                "request_id": 1,
                "session_id": 7,
                "theoretical_min_prefill_tokens": 10,
                "implementation_min_prefill_tokens": 12,
                "materialized_prefill_tokens": 100,
            },
        ],
    )
    _write_json(
        run_dir / "run.json",
        {
            "status": "completed",
            "session_arrival_rate_per_second": 0.05,
            "process_wall_clock_seconds": 3.0,
            "workload_manifest": str(manifest),
        },
    )
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": 1,
            "simulation_window_seconds": 10.0,
            "wall_clock_seconds": 1.0,
            "counts": {"requests": 2},
            "throughput": {"requests_per_second": 0.2},
            "latency_ms": {},
            "batch_summary_by_cluster": {},
            "prefix_cache": {},
            "cpu_kv_cache": {},
        },
    )
    _write_json(
        run_dir / "config.normalized.json",
        {
            "simulation_mode": "online",
            "system_architecture": "pd-disaggregation",
            "clusters": {
                "prefill": {"parallelism": {"num_replicas": 1, "data_parallel_size": 8}},
                "decode": {"parallelism": {"num_replicas": 1, "data_parallel_size": 3}},
            },
        },
    )
    _write_csv(
        run_dir / "requests.csv",
        [
            {
                "request_id": 0,
                "session_id": 7,
                "num_prefill_tokens": 10,
                "num_decode_tokens": 2,
                "cached_prefill_tokens": 0,
                "scheduled_prefill_tokens": 10,
                "prefix_cache_query_blocks": 10,
                "prefix_cache_hit_blocks": 0,
                "gpu_prefix_hit_blocks": 0,
                "prefill_replica_id": 0,
                "prefill_dp_id": 0,
                "decode_replica_id": 0,
                "decode_dp_id": 0,
                "ttft_ms": 1,
                "e2e_ms": 2,
            },
            {
                "request_id": 1,
                "session_id": 7,
                "num_prefill_tokens": 100,
                "num_decode_tokens": 3,
                "cached_prefill_tokens": 80,
                "scheduled_prefill_tokens": 20,
                "prefix_cache_query_blocks": 10,
                "prefix_cache_hit_blocks": 8,
                "gpu_prefix_hit_blocks": 8,
                "prefill_replica_id": 0,
                "prefill_dp_id": 1,
                "decode_replica_id": 0,
                "decode_dp_id": 1,
                "ttft_ms": 3,
                "e2e_ms": 4,
            },
        ],
    )
    _write_csv(
        run_dir / "gpu_kv_occupancy.csv",
        [
            {
                "time_s": 0,
                "cluster_type": "PREFILL",
                "replica_id": 0,
                "dp_id": 0,
                "active_blocks": 1,
                "capacity_blocks": 10,
                "active_bytes_per_gpu": 1,
                "hbm_fraction": 0.1,
                "active_fraction_of_kv_budget": 0.1,
                "active_fraction_of_total_hbm": 0.1,
            },
            {
                "time_s": 1,
                "cluster_type": "PREFILL",
                "replica_id": 0,
                "dp_id": 0,
                "active_blocks": 5,
                "capacity_blocks": 10,
                "active_bytes_per_gpu": 5,
                "hbm_fraction": 0.5,
                "active_fraction_of_kv_budget": 0.5,
                "active_fraction_of_total_hbm": 0.5,
            },
        ],
    )

    rows, lanes = collector.collect(tmp_path, [0.05])
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "complete"
    assert row["gpu_prefix_hit_rate_from_csv"] == 0.4
    assert row["migration_sessions_count"] == 1
    assert row["migration_transitions"] == 1
    assert row["manifest_avoidable_recompute_tokens"] == 8
    assert row["prefill_lane_count"] == 8
    assert row["prefill_lane_load_min"] == 0
    assert row["gpu_occupancy_sample_count"] == 2
    assert row["gpu_occupancy_invalid_rows"] == 0
    assert row["gpu_occupancy_prefill_peak_active_kv_budget_fraction"] == 0.5
    assert len([lane for lane in lanes if lane["phase"] == "PREFILL"]) == 8
