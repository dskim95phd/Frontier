#!/usr/bin/env python3
"""Compare the C++ Step 5 example with the Python CPU-store semantics."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--binary", type=Path, required=True)
    result.add_argument("--repo-root", type=Path, required=True)
    return result


def commit(manager, *, session: int, frontier: int, generation: int, time: float):
    reservation = manager.reserve_offload(
        session_id=session,
        desired_frontier_blocks=frontier,
        generation=generation,
        time=time,
    )
    manager.commit_offload(reservation, time=time + 0.001)


def main() -> int:
    args = parser().parse_args()
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo))
    from frontier.config.cpu_kv_cache_config import CPUKVCacheConfig
    from frontier.cpu_kv_cache_transfer import AnalyticalCPUKVCacheTransferEngine
    from frontier.kv_cache import CPUKVCacheManager

    config = repo / "cpp/examples/configs/06_cpu_kv_cache_pdd_online.json"
    workload = repo / "cpp/examples/workloads/06_cpu_kv_cache_sessions.csv"
    def run_cpp(config_path: Path, output_dir: Path) -> dict:
        subprocess.run(
            [
                str(args.binary.resolve()),
                "--config",
                str(config_path),
                "--workload",
                str(workload),
                "--output-dir",
                str(output_dir),
                "--output-mode",
                "full",
            ],
            check=True,
        )
        return json.loads(
            (output_dir / "trace.json").read_text(encoding="utf-8")
        )

    with tempfile.TemporaryDirectory(prefix="frontier-cpu-kv-diff-") as tmp:
        temporary = Path(tmp)
        cpp = run_cpp(config, temporary / "output")
        slow_config = json.loads(config.read_text(encoding="utf-8"))
        slow_config["run_id"] = "cpu-kv-cache-python-diff-slow-h2d"
        slow_config["cpu_kv_cache"]["read_bandwidth_gbps"] = 0.01
        slow_config_path = temporary / "slow.json"
        slow_config_path.write_text(json.dumps(slow_config), encoding="utf-8")
        slow_cpp = run_cpp(slow_config_path, temporary / "slow-output")

    target = cpp["cpu_kv_cache_targets"][0]
    python = CPUKVCacheManager(
        capacity_bytes=target["capacity_bytes"],
        bytes_per_block=target["bytes_per_block"],
        capacity_pressure_policy="prefix_fit",
    )
    # Logical snapshots produced by the example: A[0,2), B[0,3), then A[2,3).
    commit(python, session=7, frontier=2, generation=0, time=0.0)
    commit(python, session=9, frontier=3, generation=0, time=0.01)
    commit(python, session=7, frontier=3, generation=1, time=0.1)
    python.record_lookup(session_id=7, query_blocks=2, hit_blocks=0, lookup_id=0)
    python.record_lookup(session_id=9, query_blocks=3, hit_blocks=0, lookup_id=1)
    python.record_lookup(session_id=7, query_blocks=2, hit_blocks=1, lookup_id=2)
    oracle = python.get_statistics()

    aggregate = cpp["cpu_kv_cache"]
    requests = {record["request_id"]: record for record in cpp["requests"]}
    successor = requests[2]
    normalized_cpp = {
        "completed_request_ids": cpp["completed_request_ids"],
        "prefill_targets": sorted(
            (record["request_id"], record["prefill_replica_id"], record["prefill_dp_id"])
            for record in cpp["requests"]
        ),
        "successor_gpu_hit_blocks": successor["gpu_prefix_hit_blocks"],
        "successor_cpu_query_blocks": successor["cpu_prefix_query_blocks"],
        "successor_cpu_hit_blocks": successor["cpu_prefix_hit_blocks"],
        "successor_cached_tokens": successor["cached_prefill_tokens"],
        "resident_blocks": target["resident_blocks"],
        "reserved_blocks": target["reserved_blocks"],
        "query_blocks": aggregate["query_blocks"],
        "hit_blocks": aggregate["hit_blocks"],
        "offload_operations": aggregate["offload_operations"],
        "offload_blocks": aggregate["offload_blocks"],
        "offload_bytes": aggregate["offload_bytes"],
        "restore_operations": aggregate["restore_operations"],
        "restore_blocks": aggregate["restore_blocks"],
        "restore_bytes": aggregate["restore_bytes"],
        "evicted_blocks": target["evicted_blocks"],
        "truncated_offloads": target["truncated_offloads"],
        "skipped_offloads": target["skipped_offloads"],
        "pdd_transfers": len(cpp["kv_cache_transfers"]),
        "active_restore_leases": target["active_restore_leases"],
        "active_offload_reservations": target["active_offload_reservations"],
        "pending_restore_operations": target["pending_restore_operations"],
        "staged_restore_payloads": target["staged_restore_payloads"],
    }
    normalized_python = {
        "completed_request_ids": [0, 1, 2],
        "prefill_targets": [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
        "successor_gpu_hit_blocks": 1,
        "successor_cpu_query_blocks": 2,
        "successor_cpu_hit_blocks": 1,
        "successor_cached_tokens": 8,
        "resident_blocks": oracle["resident_blocks"],
        "reserved_blocks": oracle["reserved_blocks"],
        "query_blocks": oracle["cpu_query_blocks"],
        "hit_blocks": oracle["cpu_hit_blocks"],
        "offload_operations": 3,
        "offload_blocks": 2 + 3 + 1,
        "offload_bytes": (2 + 3 + 1) * target["bytes_per_block"],
        "restore_operations": 1,
        "restore_blocks": 1,
        "restore_bytes": target["bytes_per_block"],
        "evicted_blocks": oracle["evicted_blocks"],
        "truncated_offloads": oracle["truncated_offloads"],
        "skipped_offloads": oracle["skipped_offloads"],
        "pdd_transfers": 3,
        "active_restore_leases": 0,
        "active_offload_reservations": 0,
        "pending_restore_operations": 0,
        "staged_restore_payloads": 0,
    }
    if normalized_cpp != normalized_python:
        raise AssertionError(
            f"CPU KV logical differential mismatch: C++={normalized_cpp}, "
            f"Python={normalized_python}"
        )

    cpu_config = json.loads(config.read_text(encoding="utf-8"))["cpu_kv_cache"]
    python_transfer_config = CPUKVCacheConfig(
        enable=True,
        capacity_bytes=target["capacity_bytes"],
        write_bandwidth_gbps=cpu_config["write_bandwidth_gbps"],
        write_latency_ms=cpu_config["write_latency_ms"],
        read_bandwidth_gbps=cpu_config["read_bandwidth_gbps"],
        read_latency_ms=cpu_config["read_latency_ms"],
    )
    python_engine = AnalyticalCPUKVCacheTransferEngine(python_transfer_config)
    transfers = sorted(
        cpp["cpu_kv_cache_transfers"],
        key=lambda record: (record["submitted_at_s"], record["transfer_id"]),
    )
    for transfer in transfers:
        direction = "d2h" if transfer["kind"] == "offload" else "h2d"
        expected = python_engine.schedule(
            direction=direction,
            size_bytes=transfer["size_bytes"],
            submitted_at=transfer["submitted_at_s"],
        )
        comparisons = {
            "started_at_s": expected.start_time,
            "completed_at_s": expected.end_time,
            "queue_time_ms": expected.queue_time_ms,
            "service_time_ms": expected.service_time_ms,
        }
        for field, oracle_value in comparisons.items():
            if not math.isclose(
                transfer[field],
                oracle_value,
                rel_tol=1e-9,
                abs_tol=1e-12 if field.endswith("_s") else 1e-9,
            ):
                raise AssertionError(
                    f"CPU transfer timing mismatch for {field}: "
                    f"C++={transfer[field]}, Python={oracle_value}"
                )

    events = cpp["event_trace"]
    if events != sorted(events, key=lambda event: (event["time_s"], event["sequence"])):
        raise AssertionError("C++ CPU event trace is not deterministically ordered")
    for transfer in transfers:
        prefix = "cpu_kv_cache_offload" if transfer["kind"] == "offload" else "cpu_kv_cache_restore"
        matching = [
            event
            for event in events
            if event["type"] in {f"{prefix}_start", f"{prefix}_end"}
            and event.get("transfer_id") == transfer["transfer_id"]
        ]
        if [event["type"] for event in matching] != [
            f"{prefix}_start",
            f"{prefix}_end",
        ]:
            raise AssertionError(
                f"CPU transfer event order mismatch for {transfer['kind']} "
                f"{transfer['transfer_id']}: {matching}"
            )

    fast_successor = next(record for record in cpp["requests"] if record["request_id"] == 2)
    slow_successor = next(
        record for record in slow_cpp["requests"] if record["request_id"] == 2
    )
    slow_transfer_config = CPUKVCacheConfig(
        enable=True,
        capacity_bytes=target["capacity_bytes"],
        read_bandwidth_gbps=0.01,
        read_latency_ms=cpu_config["read_latency_ms"],
    )
    fast_restore_ms = AnalyticalCPUKVCacheTransferEngine(
        python_transfer_config
    ).schedule(
        direction="h2d",
        size_bytes=target["bytes_per_block"],
        submitted_at=0.0,
    ).service_time_ms
    slow_restore_ms = AnalyticalCPUKVCacheTransferEngine(
        slow_transfer_config
    ).schedule(
        direction="h2d",
        size_bytes=target["bytes_per_block"],
        submitted_at=0.0,
    ).service_time_ms
    cpp_ttft_delta_ms = slow_successor["ttft_ms"] - fast_successor["ttft_ms"]
    python_restore_delta_ms = slow_restore_ms - fast_restore_ms
    if not math.isclose(
        cpp_ttft_delta_ms,
        python_restore_delta_ms,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise AssertionError(
            "follow-up TTFT does not include the Python-oracle H2D delta: "
            f"C++={cpp_ttft_delta_ms}, Python={python_restore_delta_ms}"
        )
    print(json.dumps(normalized_cpp, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
