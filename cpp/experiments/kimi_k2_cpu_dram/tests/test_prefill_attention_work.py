from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "analyze_prefill_attention_work.py"
SPEC = importlib.util.spec_from_file_location("prefill_attention_work", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def test_attention_token_pairs_handles_cached_suffix() -> None:
    # Newly computed positions 5..10 attend to 5+6+...+10 visible tokens.
    assert analysis.attention_token_pairs(10, 4) == 45
    assert analysis.attention_token_pairs(10, 0) == 55
    assert analysis.attention_token_pairs(10, 10) == 0


def test_build_rows_combines_arrival_and_service_work() -> None:
    bucket = analysis.Bucket(start_s=0.0, end_s=10.0)
    bucket.arrived_requests = 1
    bucket.arrived_attention_pairs = 100
    bucket.served_attention_pairs = 60
    bucket.prefill_predicted_execution_ms = 20_000.0
    bucket.ttft_ms = [10.0, 20.0, 30.0]

    row = analysis.build_rows(
        [bucket], prefill_lanes=4, attention_flops_per_pair=2
    )[0]

    assert row["backlog_attention_token_pairs"] == 40
    assert row["prefill_execution_utilization"] == 0.5
    assert row["ttft_p50_ms"] == 20.0
    assert row["ttft_p90_ms"] == 28.0


def test_load_arrival_work_uses_request_arrival_bucket(tmp_path: Path) -> None:
    path = tmp_path / "requests.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "arrived_at_s",
                "num_prefill_tokens",
                "cached_prefill_tokens",
                "scheduled_prefill_tokens",
                "ttft_ms",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "arrived_at_s": "10",
                "num_prefill_tokens": "8",
                "cached_prefill_tokens": "4",
                "scheduled_prefill_tokens": "4",
                "ttft_ms": "12",
            }
        )
    buckets = [analysis.Bucket(0, 10), analysis.Bucket(10, 20)]
    analysis.load_arrival_work(path, buckets, 10)

    assert buckets[0].arrived_requests == 0
    assert buckets[1].arrived_requests == 1
    assert buckets[1].arrived_attention_pairs == 26


def test_all_arrived_summary_replaces_completed_only_pairs() -> None:
    buckets = [analysis.Bucket(0, 120), analysis.Bucket(120, 240)]
    buckets[0].arrived_attention_pairs = 999
    summary = {
        "batch_time_bucket_seconds": 60,
        "prefill_attention_token_pairs_by_arrival_time_bucket": {
            "0": 10,
            "1": 20,
            "2": 30,
        },
    }

    assert analysis.load_all_arrived_attention_work(summary, buckets, 120)
    assert buckets[0].arrived_attention_pairs == 30
    assert buckets[1].arrived_attention_pairs == 30
