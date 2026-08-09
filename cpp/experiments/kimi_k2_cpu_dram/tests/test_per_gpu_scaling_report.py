from __future__ import annotations

from cpp.experiments.kimi_k2_cpu_dram import generate_per_gpu_scaling_report as report


def _point(rate: float, dram: float, verified: bool) -> dict[str, object]:
    return {
        "session_injection_rate_per_s": rate,
        "cpu_dram_gb_per_prefill_gpu": dram,
        "physical_cpu_pool_gb_per_two_prefill_gpus": dram * 2,
        "active_sessions_per_prefill_gpu": 100 + 10 * rate,
        "completion_requests_per_s_per_prefill_gpu": 2 + rate,
        "ttft_p90_max_ms": 900.0,
        "tpot_p90_max_ms": 20.0,
        "combined_hit_pct": 95.0,
        "verified_sustainable": verified,
    }


def test_load_envelope_uses_smallest_verified_capacity_and_records_bracket() -> None:
    points = [
        _point(0.4, 500.0, False),
        _point(0.4, 750.0, True),
        _point(0.4, 1_000.0, True),
    ]

    envelope = report.build_load_envelope(points)

    assert envelope[0]["lower_failed_dram_gb_per_gpu"] == 500.0
    assert envelope[0]["verified_dram_gb_per_gpu"] == 750.0
    assert envelope[0]["physical_pool_gb_per_two_prefill_gpus"] == 1_500.0


def test_capacity_summary_reports_highest_tested_verified_rate() -> None:
    points = [
        _point(0.3, 750.0, True),
        _point(0.4, 750.0, True),
        _point(0.45, 750.0, False),
    ]

    summary = report.build_capacity_summary(points)

    assert summary[0]["maximum_tested_verified_rate_per_s"] == 0.4
    assert summary[0]["is_lower_bound"] is True
