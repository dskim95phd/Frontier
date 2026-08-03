from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
REQUEST_GENERATOR_DIR = REPO_ROOT / "cpp" / "frontier" / "request_generator"
sys.path.insert(0, str(REQUEST_GENERATOR_DIR))

from workload_generator import (  # noqa: E402
    BoundedLognormalLengthDistribution,
    FixedLengthDistribution,
    GammaIntervalDistribution,
    PoissonIntervalDistribution,
    StaticIntervalDistribution,
    UniformLengthDistribution,
    WorkloadRequest,
    ZipfLengthDistribution,
    generate_request_shapes,
    generate_requests,
    write_workload_csv,
)


def _bounded_lognormal(
    rng: np.random.Generator,
    *,
    median: float,
    sigma: float,
    lower: int,
    upper: int,
) -> int:
    sample = math.exp(rng.normal(math.log(median), sigma))
    return int(min(upper, max(lower, round(sample))))


def test_benchmark_distribution_preserves_seeded_shapes_and_gaps() -> None:
    seed = 20260728
    actual = generate_request_shapes(
        num_requests=4,
        seed=seed,
        length_distribution=BoundedLognormalLengthDistribution(
            prefill_median=8_192,
            prefill_sigma=0.8,
            prefill_min=2_048,
            prefill_max=32_768,
            decode_median=1_024,
            decode_sigma=0.7,
            decode_min=256,
            decode_max=4_096,
        ),
        interval_distribution=PoissonIntervalDistribution(qps=1.0),
    )

    rng = np.random.default_rng(seed)
    expected: list[tuple[int, int, float]] = []
    for _ in range(4):
        expected.append(
            (
                _bounded_lognormal(
                    rng,
                    median=8_192,
                    sigma=0.8,
                    lower=2_048,
                    upper=32_768,
                ),
                _bounded_lognormal(
                    rng,
                    median=1_024,
                    sigma=0.7,
                    lower=256,
                    upper=4_096,
                ),
                float(rng.exponential(1.0)),
            )
        )

    assert [
        (shape.prompt_tokens, shape.output_tokens, shape.unit_exponential_gap)
        for shape in actual
    ] == expected


def test_fixed_static_workload_uses_canonical_csv_contract() -> None:
    requests = generate_requests(
        num_requests=3,
        seed=7,
        length_distribution=FixedLengthDistribution(32, 8),
        interval_distribution=StaticIntervalDistribution(0.25),
    )
    assert [request.session_start_at for request in requests] == [0.0, 0.25, 0.5]
    assert [request.think_time for request in requests] == [0.0, 0.0, 0.0]

    output = io.StringIO()
    write_workload_csv(output, requests)
    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert tuple(rows[0]) == (
        "session_start_at",
        "think_time",
        "num_prefill_tokens",
        "num_decode_tokens",
        "session_id",
        "session_turn_index",
    )
    assert rows == [
        {
            "session_start_at": "0",
            "think_time": "0",
            "num_prefill_tokens": "32",
            "num_decode_tokens": "8",
            "session_id": "",
            "session_turn_index": "",
        },
        {
            "session_start_at": "0.25",
            "think_time": "0",
            "num_prefill_tokens": "32",
            "num_decode_tokens": "8",
            "session_id": "",
            "session_turn_index": "",
        },
        {
            "session_start_at": "0.5",
            "think_time": "0",
            "num_prefill_tokens": "32",
            "num_decode_tokens": "8",
            "session_id": "",
            "session_turn_index": "",
        },
    ]


def test_multi_turn_workload_records_completion_relative_think_time() -> None:
    output = io.StringIO()
    write_workload_csv(
        output,
        [
            WorkloadRequest(0.25, 0.0, 32, 8, 7, 0),
            WorkloadRequest(None, 1.5, 16, 4, 7, 1),
        ],
    )
    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert rows[0]["session_start_at"] == "0.25"
    assert rows[0]["think_time"] == "0"
    assert rows[1]["session_start_at"] == ""
    assert rows[1]["think_time"] == "1.5"


@pytest.mark.parametrize(
    "length_distribution",
    [
        UniformLengthDistribution(16, 64, 3.0),
        ZipfLengthDistribution(
            min_tokens=16,
            max_tokens=64,
            theta=0.6,
            prefill_to_decode_ratio=3.0,
            seed=11,
        ),
    ],
)
@pytest.mark.parametrize(
    "interval_distribution",
    [
        PoissonIntervalDistribution(qps=4.0, max_interval_factor=3.0),
        GammaIntervalDistribution(qps=4.0, cv=0.5),
    ],
)
def test_copied_distributions_produce_valid_requests(
    length_distribution, interval_distribution
) -> None:
    requests = generate_requests(
        num_requests=32,
        seed=11,
        length_distribution=length_distribution,
        interval_distribution=interval_distribution,
        first_arrival_at_zero=False,
    )
    assert all(request.num_prefill_tokens > 0 for request in requests)
    assert all(request.num_decode_tokens > 0 for request in requests)
    assert all(request.session_start_at is not None for request in requests)
    assert all(request.session_start_at >= 0.0 for request in requests)
    assert all(
        left.session_start_at <= right.session_start_at
        for left, right in zip(requests, requests[1:])
    )


def test_cli_generates_a_normalized_workload(tmp_path: Path) -> None:
    config_path = tmp_path / "workload.json"
    output_path = tmp_path / "workload.csv"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "num_requests": 2,
                "seed": 13,
                "first_arrival_at_zero": True,
                "length": {
                    "type": "fixed",
                    "prefill_tokens": 128,
                    "decode_tokens": 16,
                },
                "interval": {"type": "poisson", "qps": 2.0},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REQUEST_GENERATOR_DIR / "generate_workload.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    with output_path.open(encoding="utf-8") as output:
        rows = list(csv.DictReader(output))
    assert len(rows) == 2
    assert rows[0]["session_start_at"] == "0"
    assert rows[0]["think_time"] == "0"
    assert rows[0]["num_prefill_tokens"] == "128"
    assert rows[0]["num_decode_tokens"] == "16"


def test_cli_rejects_non_boolean_first_arrival_setting(tmp_path: Path) -> None:
    config_path = tmp_path / "workload.json"
    output_path = tmp_path / "workload.csv"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "num_requests": 1,
                "first_arrival_at_zero": "false",
                "length": {
                    "type": "fixed",
                    "prefill_tokens": 8,
                    "decode_tokens": 2,
                },
                "interval": {"type": "static"},
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(REQUEST_GENERATOR_DIR / "generate_workload.py"),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "first_arrival_at_zero must be a boolean" in completed.stderr
