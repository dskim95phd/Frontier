"""Fast contract checks for the TraceLab session-arrival sweep runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_tracelab_arrival_sweep as runner  # noqa: E402


def test_default_rates_and_labels() -> None:
    rates = runner.parse_rates(runner.DEFAULT_RATES)
    assert rates == [0.10, 0.09, 0.08, 0.07, 0.06, 0.05]
    assert [runner.rate_label(rate) for rate in rates] == [
        "r0p1",
        "r0p09",
        "r0p08",
        "r0p07",
        "r0p06",
        "r0p05",
    ]
    with pytest.raises(argparse.ArgumentTypeError):
        runner.parse_rates("0.05,0.05")


def test_default_horizons_reproduce_anchor_runs() -> None:
    def end(rate: float) -> float:
        return runner.simulation_end_time_s(
            sample_sessions=3000,
            rate=rate,
            repetitions=2,
            drain_fraction=0.4,
            minimum_drain_seconds=15000.0,
        )

    assert end(0.05) == pytest.approx(144000.0)
    assert end(0.08) == pytest.approx(90000.0)
    assert end(0.10) == pytest.approx(75000.0)


def test_build_config_only_overrides_experiment_routing() -> None:
    template = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    args = argparse.Namespace(
        cache_threshold=0.5,
        balance_abs_threshold=8,
        balance_rel_threshold=1.5,
    )
    config = runner.build_config(template, args, 0.05)
    scheduler = config["cluster_scheduler"]
    assert scheduler == {
        "type": "sticky_round_robin",
        "prefill_type": "cache_aware",
        "decode_type": "vllm_queue_aware",
        "cache_threshold": 0.5,
        "balance_abs_threshold": 8,
        "balance_rel_threshold": 1.5,
    }
    assert config["cpu_kv_cache"]["enabled"] is False
    assert config["clusters"]["prefill"]["parallelism"]["data_parallel_size"] == 8
    assert config["clusters"]["decode"]["parallelism"]["data_parallel_size"] == 3
    assert config["clusters"]["decode"]["parallelism"]["decode_context_parallel_size"] == 8


def test_metadata_validation_rejects_a_different_rate(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "sampling": {
                    "sample_sessions": 3000,
                    "seed": 20260803,
                    "session_repetitions": 2,
                    "session_arrival_rate_per_second": 0.05,
                }
            }
        ),
        encoding="utf-8",
    )
    runner.validate_workload_metadata(
        metadata,
        rate=0.05,
        seed=20260803,
        sample_sessions=3000,
        repetitions=2,
    )
    with pytest.raises(ValueError, match="session_arrival_rate"):
        runner.validate_workload_metadata(
            metadata,
            rate=0.06,
            seed=20260803,
            sample_sessions=3000,
            repetitions=2,
        )
