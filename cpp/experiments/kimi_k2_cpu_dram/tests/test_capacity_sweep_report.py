from __future__ import annotations

import re

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
