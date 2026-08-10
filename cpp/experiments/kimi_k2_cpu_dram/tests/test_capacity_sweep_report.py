from __future__ import annotations

import re

import pytest

from cpp.experiments.kimi_k2_cpu_dram import analyze_r0p4_capacity_sweep_10h as report


def test_ttft_chart_caps_y_axis_and_clips_large_values() -> None:
    cases = [
        {
            "capacity_label": "cpu0500gb",
            "bins": [
                {"ttft_mean_ms": 1_000.0, "end_time_s": 300.0},
                {"ttft_mean_ms": 50_000.0, "end_time_s": 600.0},
            ],
        }
    ]

    rendered = report._svg_chart(
        cases, "ttft_mean_ms", "TTFT mean", " ms", y_max_limit=5_000.0
    )

    assert "Y-axis capped at 5 s; values above the cap are clipped." in rendered
    assert "5 s" in rendered
    path = re.search(r"<path d='([^']+)'", rendered)
    assert path is not None
    y_coordinates = [float(value) for value in re.findall(r",(-?[0-9.]+)", path.group(1))]
    assert y_coordinates
    assert min(y_coordinates) >= 30.0


def test_active_sessions_chart_caps_y_axis_at_four_thousand() -> None:
    cases = [
        {
            "capacity_label": "cpu4000gb",
            "bins": [
                {"active_sessions_time_weighted": 3_000.0, "end_time_s": 300.0},
                {"active_sessions_time_weighted": 5_000.0, "end_time_s": 600.0},
            ],
        }
    ]

    rendered = report._svg_chart(
        cases,
        "active_sessions_time_weighted",
        "Active sessions",
        y_max_limit=4_000.0,
    )

    assert "Y-axis capped at 4,000 sessions" in rendered
    assert ">4e+03<" in rendered


def test_context_length_bucket_boundaries_end_above_200k() -> None:
    assert report._context_length_bucket_index(0) == 0
    assert report._context_length_bucket_index(40_000) == 0
    assert report._context_length_bucket_index(40_001) == 1
    assert report._context_length_bucket_index(200_000) == 4
    assert report._context_length_bucket_index(200_001) == 5


def test_stacked_context_chart_uses_counts_and_all_six_ranges() -> None:
    distribution = {
        "bins": [
            {
                "end_time_s": 300.0,
                "context_le_40k": 3,
                "context_40k_to_80k": 2,
                "context_80k_to_120k": 1,
                "context_120k_to_160k": 1,
                "context_160k_to_200k": 1,
                "context_gt_200k": 1,
            },
            {
                "end_time_s": 600.0,
                "context_le_40k": 4,
                "context_40k_to_80k": 3,
                "context_80k_to_120k": 2,
                "context_120k_to_160k": 1,
                "context_160k_to_200k": 1,
                "context_gt_200k": 2,
            },
        ]
    }

    rendered = report._stacked_context_chart(distribution)

    assert "Requests / 5 min" in rendered
    assert "≤40K" in rendered
    assert "160–200K" in rendered
    assert "&gt;200K" in rendered
    assert rendered.count("<path d=") == 6


def test_final_hour_qos_contains_detailed_weighted_measurements() -> None:
    rows = [
        {
            "start_time_s": 0.0,
            "end_time_s": 300.0,
            "duration_s": 300.0,
            "request_arrivals": 600,
            "request_completions": 570,
            "ttft_count": 100,
            "ttft_mean_ms": 100.0,
            "ttft_p90_ms": 500.0,
            "tpot_count": 100,
            "tpot_mean_ms": 10.0,
            "tpot_p90_ms": 12.0,
            "pool_busy_pct": 50.0,
            "prefix_cache_query_blocks": 1000,
            "gpu_prefix_hit_blocks": 600,
            "cpu_prefix_query_blocks": 400,
            "cpu_prefix_hit_blocks": 300,
            "prefix_cache_hit_blocks": 900,
            "transfer_bytes": 300_000_000_000,
            "active_sessions_time_weighted": 100.0,
            "active_sessions_end": 110,
            "prefill_waiting_queue_count_time_weighted": 2.0,
            "cumulative_backlog_requests": 30,
        },
        {
            "start_time_s": 300.0,
            "end_time_s": 600.0,
            "duration_s": 300.0,
            "request_arrivals": 660,
            "request_completions": 650,
            "ttft_count": 300,
            "ttft_mean_ms": 300.0,
            "ttft_p90_ms": 900.0,
            "tpot_count": 300,
            "tpot_mean_ms": 14.0,
            "tpot_p90_ms": 18.0,
            "pool_busy_pct": 70.0,
            "prefix_cache_query_blocks": 1000,
            "gpu_prefix_hit_blocks": 700,
            "cpu_prefix_query_blocks": 300,
            "cpu_prefix_hit_blocks": 200,
            "prefix_cache_hit_blocks": 900,
            "transfer_bytes": 300_000_000_000,
            "active_sessions_time_weighted": 120.0,
            "active_sessions_end": 125,
            "prefill_waiting_queue_count_time_weighted": 4.0,
            "cumulative_backlog_requests": 40,
        },
    ]

    qos = report._window_qos(rows, 1.0)

    assert qos["request_arrivals_per_s"] == pytest.approx(2.1)
    assert qos["request_completions_per_s"] == pytest.approx(1220 / 600)
    assert qos["ttft_mean_ms"] == pytest.approx(250.0)
    assert qos["ttft_p90_max_ms"] == 900.0
    assert qos["tpot_mean_ms"] == pytest.approx(13.0)
    assert qos["combined_hit_pct"] == pytest.approx(90.0)
    assert qos["cpu_conditional_hit_pct"] == pytest.approx(500 / 700 * 100)
    assert qos["pool_busy_pct_time_weighted"] == pytest.approx(60.0)
    assert qos["waiting_queue_count_time_weighted"] == pytest.approx(3.0)
    assert qos["cumulative_backlog_start_requests"] == 30
    assert qos["cumulative_backlog_end_requests"] == 40
