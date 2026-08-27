from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cpp.experiments.kimi_cpu_dram.lib import capacity_report as report


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


def test_final_hour_statistics_use_source_counts_and_exact_window() -> None:
    rows = [
        {
            "start_time_s": 3_000.0,
            "end_time_s": 3_300.0,
            "duration_s": 300.0,
            "pool_busy_pct": 50.0,
            "active_sessions_time_weighted": 24.0,
            "prefill_waiting_queue_count_time_weighted": 12.0,
            "request_arrivals": 60,
            "request_completions": 48,
            "backlog_delta_requests": 12,
            "cumulative_backlog_requests": 12,
            "ttft_count": 2,
            "ttft_mean_ms": 100.0,
            "ttft_p90_ms": 200.0,
            "tpot_count": 2,
            "tpot_mean_ms": 10.0,
            "tpot_p90_ms": 12.0,
            "prefix_cache_query_blocks": 100,
            "gpu_prefix_hit_blocks": 40,
            "cpu_prefix_query_blocks": 60,
            "cpu_prefix_hit_blocks": 30,
            "prefix_cache_hit_blocks": 70,
            "transfer_bytes": 300_000_000_000,
        },
        {
            "start_time_s": 3_300.0,
            "end_time_s": 3_600.0,
            "duration_s": 300.0,
            "pool_busy_pct": 100.0,
            "active_sessions_time_weighted": 48.0,
            "prefill_waiting_queue_count_time_weighted": 24.0,
            "request_arrivals": 40,
            "request_completions": 52,
            "backlog_delta_requests": -12,
            "cumulative_backlog_requests": 0,
            "ttft_count": 1,
            "ttft_mean_ms": 400.0,
            "ttft_p90_ms": 500.0,
            "tpot_count": 1,
            "tpot_mean_ms": 40.0,
            "tpot_p90_ms": 50.0,
            "prefix_cache_query_blocks": 200,
            "gpu_prefix_hit_blocks": 80,
            "cpu_prefix_query_blocks": 120,
            "cpu_prefix_hit_blocks": 60,
            "prefix_cache_hit_blocks": 140,
            "transfer_bytes": 600_000_000_000,
        },
    ]

    stats = report._window_statistics(rows, 1.0, prefill_lanes=24)

    assert stats["window_start_s"] == 3_000.0
    assert stats["window_end_s"] == 3_600.0
    assert stats["request_arrivals"] == 100
    assert stats["request_completions"] == 100
    assert stats["completion_to_arrival_pct"] == 100.0
    assert stats["completion_requests_per_s_per_prefill_gpu"] == pytest.approx(
        100 / 600 / 24
    )
    assert stats["prefill_busy_pct_time_weighted"] == 75.0
    assert stats["active_sessions_per_prefill_gpu"] == 1.5
    assert stats["waiting_queue_count_per_prefill_gpu"] == 0.75
    assert stats["cumulative_backlog_start_requests"] == 0
    assert stats["cumulative_backlog_end_requests"] == 0
    assert stats["cumulative_backlog_max_requests"] == 12
    assert stats["ttft_mean_ms"] == 200.0
    assert stats["ttft_p90_max_ms"] == 500.0
    assert stats["tpot_mean_ms"] == 20.0
    assert stats["tpot_p90_max_ms"] == 50.0
    assert stats["gpu_hit_pct"] == 40.0
    assert stats["cpu_conditional_hit_pct"] == 50.0
    assert stats["combined_hit_pct"] == 70.0
    assert stats["cpu_transfer_gbps"] == 1.5


def test_hot_session_hbm_statistics_exclude_terminal_sessions_and_include_snapshot(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.normalized.json").write_text(
        json.dumps(
            {
                "clusters": {
                    "prefill": {
                        "parallelism": {
                            "num_replicas": 1,
                            "tensor_parallel_size": 1,
                            "pipeline_parallel_size": 2,
                            "data_parallel_size": 1,
                        },
                        "gpu_memory": {
                            "capacity_bytes_per_gpu": 10_000_000_000,
                            "kv_cache_budget_bytes_per_gpu": 2_000_000_000,
                            "model_weight_bytes_per_gpu": 1_000_000_000,
                            "runtime_reserve_fraction": 0.1,
                            "runtime_reserve_bytes": 0,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "workload.normalized.csv").write_text(
        "session_id,session_turn_index\n0,0\n0,1\n1,0\n",
        encoding="utf-8",
    )
    (tmp_path / "requests.csv").write_text(
        "session_id,num_prefill_tokens,prefill_completed_at_s\n"
        "0,32,3000\n1,999,4000\n0,48,6000\n",
        encoding="utf-8",
    )
    summary = {
        "cpu_kv_cache": {
            "bytes_per_block": 1_000_000_000,
            "kda_snapshot_bytes_per_session": 500_000_000,
        }
    }

    metric = report._hot_session_hbm_statistics(
        tmp_path, summary, horizon_s=7_200.0, prefill_lanes=2
    )

    assert metric["average_hot_sessions"] == pytest.approx(2 / 3)
    assert metric["maximum_hot_sessions"] == 1
    assert metric["average_ordinary_kv_gb_per_gpu"] == pytest.approx(2 / 3)
    assert metric["average_kda_snapshot_gb_per_gpu"] == pytest.approx(1 / 6)
    assert metric["average_hot_kv_gb_per_gpu"] == pytest.approx(5 / 6)
    assert metric["maximum_hot_kv_gb_per_gpu"] == pytest.approx(1.25)
    assert metric["average_hot_kv_pct_of_budget"] == pytest.approx(100 * 5 / 12)
    assert metric["maximum_hot_kv_pct_of_budget"] == pytest.approx(62.5)
    assert metric["average_total_hbm_demand_gb_per_gpu"] == pytest.approx(17 / 6)


def test_report_identity_uses_model_and_hides_decode_for_prefill_only(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.normalized.json").write_text(
        json.dumps(
            {
                "clusters": {
                    "prefill": {
                        "model_name": "moonshotai/Kimi-K2-Instruct",
                        "parallelism": {
                            "num_replicas": 1,
                            "tensor_parallel_size": 1,
                            "pipeline_parallel_size": 1,
                            "data_parallel_size": 8,
                        },
                    },
                    "decode": {
                        "parallelism": {
                            "num_replicas": 1,
                            "tensor_parallel_size": 4,
                            "pipeline_parallel_size": 1,
                            "data_parallel_size": 4,
                        }
                    },
                },
                "prefill_only": {"decode_tokens_per_second": 50.0},
            }
        ),
        encoding="utf-8",
    )

    identity = report._report_identity(tmp_path)

    assert identity == {
        "model_name": "Kimi K2",
        "model_asset_name": "moonshotai/Kimi-K2-Instruct",
        "prefill_gpu_count": 8,
        "decode_gpu_count": 16,
        "prefill_only": True,
    }


def test_html_can_render_final_hour_detailed_statistics() -> None:
    rendered = report._html_report(
        {
            "simulation_rate_per_s": 0.5,
            "report_identity": {
                "model_name": "Kimi K2",
                "prefill_gpu_count": 8,
                "decode_gpu_count": 16,
                "prefill_only": True,
            },
            "report_options": {"include_final_hour_details": True},
            "cases": [
                {
                    "capacity_label": "cpu0500gb",
                    "overall": {},
                    "bins": [],
                    "hot_session_hbm": {
                        "available": True,
                        "window_start_s": 32_400.0,
                        "window_end_s": 36_000.0,
                        "average_hot_sessions": 900.0,
                        "maximum_hot_sessions": 950.0,
                        "average_ordinary_kv_gb_per_gpu": 85.0,
                        "average_kda_snapshot_gb_per_gpu": 9.0,
                        "average_hot_kv_gb_per_gpu": 94.0,
                        "maximum_hot_kv_gb_per_gpu": 98.0,
                        "kv_budget_gb_per_gpu": 192.0,
                        "average_hot_kv_pct_of_budget": 49.0,
                        "maximum_hot_kv_pct_of_budget": 51.0,
                        "average_total_hbm_demand_gb_per_gpu": 191.0,
                        "average_total_hbm_demand_pct": 66.0,
                        "maximum_total_hbm_demand_gb_per_gpu": 194.0,
                        "maximum_total_hbm_demand_pct": 67.0,
                    },
                    "stability": {
                        "final_hour": {
                            "classification": "underloaded-stable",
                            "expected_bin_count": 12,
                            "statistics": {
                                "window_start_s": 32_400.0,
                                "window_end_s": 36_000.0,
                                "observed_bin_count": 12,
                                "request_arrivals": 100,
                                "request_completions": 99,
                                "completion_to_arrival_pct": 99.0,
                            },
                        },
                        "last_2_hours": {},
                    },
                }
            ],
            "heuristic_thresholds": {},
            "recommendations": {},
        }
    )

    assert "Final-hour detailed statistics" in rendered
    assert "9.00–10.00 h" in rendered
    assert "12/12" in rendered
    assert "completed req/s/GPU" in rendered
    assert "TTFT p90 max" in rendered
    assert "underloaded-stable" in rendered
    assert "Final-hour hot-session HBM working set" in rendered
    assert "KDA snapshot avg/GPU" in rendered
    assert "94.00 GB" in rendered
    assert "Kimi K2 — PREFILL-only, 8 PREFILL GPUs" in rendered
    assert "16 DECODE GPUs" not in rendered
