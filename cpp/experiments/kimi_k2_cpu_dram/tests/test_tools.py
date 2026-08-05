"""Fast local checks for the Kimi K2 CPU-DRAM experiment harness."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analyze_results  # noqa: E402
import generate_workload  # noqa: E402
import run_sweep  # noqa: E402


def test_default_workload_contract() -> None:
    rows = generate_workload.generate_rows(seed=20260803)
    assert len(rows) == 1_728 * 6 + 576
    first_turns = [row for row in rows if row.session_turn_index == 0]
    assert len(first_turns) == 1_728
    assert sum(row.num_turns == 6 for row in first_turns) == 1_152
    assert sum(row.num_turns == 7 for row in first_turns) == 576
    assert rows == generate_workload.generate_rows(seed=20260803)
    stats = generate_workload.workload_statistics(rows)
    assert abs(float(stats["final_context_mean_tokens"]) / 65_536 - 1.0) < 0.01
    assert abs(float(stats["input_output_ratio"]) / 8.0 - 1.0) < 0.005
    assert 2.80 <= float(stats["theoretical_extra_ratio"]) <= 3.20
    assert all(row.theoretical_min_prefill_tokens == row.new_input_tokens for row in rows)
    assert all(
        row.implementation_min_prefill_tokens
        == row.new_input_tokens + row.prior_decode_tokens + (16 if row.session_turn_index else 0)
        for row in rows
    )


def test_profiled_arrivals_and_think_times_are_deterministic() -> None:
    kwargs = {
        "seed": 17,
        "session_count": 40,
        "six_turn_sessions": 30,
        "arrival_rate": 0.5,
        "arrival_process": "stratified",
        "think_profile": "balanced_mixed",
    }
    rows = generate_workload.generate_rows(**kwargs)
    assert rows == generate_workload.generate_rows(**kwargs)
    starts = [row.session_start_at for row in rows if row.session_turn_index == 0]
    assert all(start is not None for start in starts)
    assert all(left < right for left, right in zip(starts, starts[1:]))
    assert float(starts[-1]) > 70.0
    gaps = [row.think_time for row in rows if row.session_turn_index > 0]
    assert min(gaps) >= 0.05
    assert max(gaps) <= 1_800.0
    assert len({round(gap, 3) for gap in gaps}) > 20


def test_capacity_contract_and_config() -> None:
    cpu500 = run_sweep.make_capacity_case("cpu500")
    assert cpu500.target_capacity_bytes == 1_000_000_000_000
    assert round(float(cpu500.rho), 2) == 1.32
    config = run_sweep.build_config(
        json.loads((ROOT / "configs" / "base_pdd.json").read_text(encoding="utf-8")),
        run_id="test",
        capacity=cpu500,
        oracle_target_capacity_bytes=10_000_000_000_000,
    )
    assert config["system_architecture"] == "pd-disaggregation"
    assert config["enable_parallel_clusters"] is False
    for name in ("prefill", "decode"):
        parallelism = config["clusters"][name]["parallelism"]
        assert parallelism["tensor_parallel_size"] == 4
        assert parallelism["pipeline_parallel_size"] == 1
        assert parallelism["data_parallel_size"] == 4
        assert parallelism["moe_tensor_parallel_size"] == 1
        assert parallelism["moe_expert_parallel_size"] == 16
        assert parallelism["decode_context_parallel_size"] == (
            4 if name == "decode" else 1
        )
        assert config["clusters"][name]["scheduler"]["num_blocks"] == (
            run_sweep.DECODE_GPU_KV_BLOCKS
            if name == "decode"
            else run_sweep.PREFILL_GPU_KV_BLOCKS
        )
        assert parallelism["num_replicas"] * parallelism["tensor_parallel_size"] * parallelism["pipeline_parallel_size"] * parallelism["data_parallel_size"] == 16
    labels = ["off", "cpu32", "cpu64", "cpu128", "cpu256", "cpu500", "oracle_unbounded_cpu"]
    order = run_sweep._execution_order("coarse", labels, 20260803)
    assert order == run_sweep._execution_order("coarse", labels, 20260803)
    assert order != labels
    assert sorted(order) == sorted(labels)


def test_calibration_uses_compact_scheduled_metric(tmp_path: Path) -> None:
    rows = generate_workload.generate_rows(seed=7, session_count=4, six_turn_sessions=2)
    manifest = tmp_path / "manifest.csv"
    workload = tmp_path / "workload.csv"
    metadata = tmp_path / "metadata.json"
    generate_workload.write_generation(workload, manifest, metadata, rows, seed=7, parameters={})
    requests = tmp_path / "requests.csv"
    with requests.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("request_id", "scheduled_prefill_tokens"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"request_id": row.request_id, "scheduled_prefill_tokens": row.new_input_tokens * 4})
    result = generate_workload.calibrate_from_requests(manifest, requests)
    assert result["actual_extra_ratio"] == 3.0
    assert result["measurement_sources"] == {"scheduled_prefill_tokens": len(rows)}


def test_analyzer_threshold_and_fine_grid() -> None:
    aggregate = [
        {
            "capacity": "cpu128",
            "rho": 0.34,
            "oracle": False,
            "scheduled_prefill_tokens": 1000,
            "cpu_prefix_hit_rate": 0.01,
            "successor_ttft_p90_ms": 100.0,
            "eviction_or_truncation_rate": 0.40,
        },
        {
            "capacity": "cpu256",
            "rho": 0.67,
            "oracle": False,
            "scheduled_prefill_tokens": 980,
            "cpu_prefix_hit_rate": 0.02,
            "successor_ttft_p90_ms": 99.0,
            "eviction_or_truncation_rate": 0.38,
        },
        {
            "capacity": "cpu500",
            "rho": 1.32,
            "oracle": False,
            "scheduled_prefill_tokens": 900,
            "cpu_prefix_hit_rate": 0.10,
            "successor_ttft_p90_ms": 95.0,
            "eviction_or_truncation_rate": 0.20,
        },
    ]
    selection = analyze_results.select_transition(aggregate)
    assert selection["C_low"] == "cpu256"
    assert selection["C_high"] == "cpu500"
    assert 5 <= selection["fine_point_count"] <= 7
    assert selection["fine_capacity_labels"][0] == "cpu256"
    assert selection["fine_capacity_labels"][-1] == "cpu500"
