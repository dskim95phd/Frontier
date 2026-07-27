from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from frontier.config.config import TraceRequestGeneratorConfig
from frontier.request_generator.trace_replay_request_generator import (
    TraceReplayRequestGenerator,
)


def _write_trace(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _generator(
    trace_file: Path,
    *,
    max_tokens: int = 128,
    prefill_scale_factor: float = 1.0,
    decode_scale_factor: float = 1.0,
    time_scale_factor: float = 1.0,
):
    generator = TraceReplayRequestGenerator(
        TraceRequestGeneratorConfig(
            trace_file=str(trace_file),
            max_tokens=max_tokens,
            prefill_scale_factor=prefill_scale_factor,
            decode_scale_factor=decode_scale_factor,
            time_scale_factor=time_scale_factor,
        )
    )
    generator.configure_session_prefix_incremental_isl(enabled=True)
    return generator


def test_incremental_session_isl_materializes_effective_prompt_lengths(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "incremental.csv"
    _write_trace(
        trace_file,
        [
            {
                "arrived_at": 0.0,
                "num_prefill_tokens": 32,
                "num_decode_tokens": 16,
                "session_id": 7,
            },
            {
                "arrived_at": 1.0,
                "num_prefill_tokens": 24,
                "num_decode_tokens": 8,
                "session_id": 8,
            },
            {
                "arrived_at": 10.0,
                "num_prefill_tokens": 8,
                "num_decode_tokens": 8,
                "session_id": 7,
            },
            {
                "arrived_at": 11.0,
                "num_prefill_tokens": 4,
                "num_decode_tokens": 8,
                "session_id": 8,
            },
        ],
    )

    requests = _generator(trace_file).generate_requests()

    assert [request.num_prefill_tokens for request in requests] == [32, 24, 56, 36]
    assert [request.num_decode_tokens for request in requests] == [16, 8, 8, 8]


def test_incremental_thinking_rounds_accumulate_within_the_session(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "incremental_thinking.csv"
    _write_trace(
        trace_file,
        [
            {
                "arrived_at": 0.0,
                "num_prefill_tokens": 32,
                "num_decode_tokens": 8,
                "session_id": 7,
                "thinking_depth": 2,
                "thinking_round_plans_json": json.dumps(
                    [
                        {
                            "num_prefill_tokens": 48,
                            "num_decode_tokens": 16,
                        },
                        {
                            "num_prefill_tokens": 32,
                            "num_decode_tokens": 8,
                        },
                    ]
                ),
            }
        ],
    )

    request = _generator(trace_file).generate_requests()[0]

    assert request.num_prefill_tokens == 48
    assert request.num_decode_tokens == 16
    assert request.user_facing_num_prefill_tokens == 96
    assert request.user_facing_num_decode_tokens == 8
    assert [
        (plan.num_prefill_tokens, plan.num_decode_tokens)
        for plan in request.thinking_round_plans
    ] == [(48, 16), (96, 8)]


@pytest.mark.parametrize(
    (
        "prefill_scale_factor",
        "decode_scale_factor",
        "raw_round_plans",
        "expected_round_plans",
    ),
    [
        (
            0.5,
            0.5,
            [(48, 16), (32, 8)],
            [(24, 8), (48, 4)],
        ),
        (
            1.5,
            1.5,
            [(5, 3), (3, 1)],
            [(7, 4), (15, 1)],
        ),
    ],
)
def test_incremental_thinking_rounds_use_trace_scale_factors(
    tmp_path: Path,
    prefill_scale_factor: float,
    decode_scale_factor: float,
    raw_round_plans: list[tuple[int, int]],
    expected_round_plans: list[tuple[int, int]],
) -> None:
    final_prefill_tokens, final_decode_tokens = raw_round_plans[-1]
    trace_file = tmp_path / "scaled_incremental_thinking.csv"
    _write_trace(
        trace_file,
        [
            {
                "arrived_at": 2.0,
                "num_prefill_tokens": final_prefill_tokens,
                "num_decode_tokens": final_decode_tokens,
                "session_id": 7,
                "thinking_depth": 2,
                "thinking_round_plans_json": json.dumps(
                    [
                        {
                            "num_prefill_tokens": prefill_tokens,
                            "num_decode_tokens": decode_tokens,
                        }
                        for prefill_tokens, decode_tokens in raw_round_plans
                    ]
                ),
            }
        ],
    )

    request = _generator(
        trace_file,
        prefill_scale_factor=prefill_scale_factor,
        decode_scale_factor=decode_scale_factor,
        time_scale_factor=2.0,
    ).generate_requests()[0]

    assert request.arrived_at == 4.0
    assert [
        (plan.num_prefill_tokens, plan.num_decode_tokens)
        for plan in request.thinking_round_plans
    ] == expected_round_plans


@pytest.mark.parametrize(
    ("field_name", "field_value", "error_pattern"),
    [
        ("arrived_at", -1, "arrived_at must be nonnegative"),
        ("arrived_at", "nan", "arrived_at must be finite"),
        ("arrived_at", "inf", "arrived_at must be finite"),
        (
            "num_prefill_tokens",
            0,
            "num_prefill_tokens must be positive before scaling",
        ),
        (
            "num_prefill_tokens",
            -1,
            "num_prefill_tokens must be positive before scaling",
        ),
        (
            "num_decode_tokens",
            0,
            "num_decode_tokens must be positive before scaling",
        ),
        (
            "num_decode_tokens",
            "-inf",
            "num_decode_tokens must be finite",
        ),
    ],
)
def test_incremental_session_trace_rejects_invalid_raw_values(
    tmp_path: Path,
    field_name: str,
    field_value: object,
    error_pattern: str,
) -> None:
    row: dict[str, object] = {
        "arrived_at": 0.0,
        "num_prefill_tokens": 32,
        "num_decode_tokens": 8,
        "session_id": 7,
    }
    row[field_name] = field_value
    trace_file = tmp_path / "invalid_incremental.csv"
    _write_trace(trace_file, [row])

    with pytest.raises(ValueError, match=error_pattern):
        _generator(trace_file).generate_requests()


def test_incremental_session_trace_rejects_integerized_zero_after_scaling(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "scaled_to_zero.csv"
    _write_trace(
        trace_file,
        [
            {
                "arrived_at": 0.0,
                "num_prefill_tokens": 1,
                "num_decode_tokens": 8,
                "session_id": 7,
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="num_prefill_tokens must remain positive after scaling",
    ):
        _generator(
            trace_file,
            prefill_scale_factor=0.5,
        ).generate_requests()


def test_incremental_session_mode_is_opt_in_for_non_session_prefix_traces(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "legacy_cumulative.csv"
    _write_trace(
        trace_file,
        [
            {
                "arrived_at": 0.0,
                "num_prefill_tokens": 32,
                "num_decode_tokens": 16,
                "session_id": 7,
            },
            {
                "arrived_at": 10.0,
                "num_prefill_tokens": 56,
                "num_decode_tokens": 8,
                "session_id": 7,
            },
        ],
    )
    generator = TraceReplayRequestGenerator(
        TraceRequestGeneratorConfig(
            trace_file=str(trace_file),
            max_tokens=128,
        )
    )

    requests = generator.generate_requests()

    assert [request.num_prefill_tokens for request in requests] == [32, 56]


def test_incremental_session_isl_rejects_expanded_context_instead_of_clipping(
    tmp_path: Path,
) -> None:
    trace_file = tmp_path / "incremental_overflow.csv"
    _write_trace(
        trace_file,
        [
            {
                "arrived_at": 0.0,
                "num_prefill_tokens": 32,
                "num_decode_tokens": 16,
                "session_id": 7,
            },
            {
                "arrived_at": 10.0,
                "num_prefill_tokens": 16,
                "num_decode_tokens": 8,
                "session_id": 7,
            },
        ],
    )

    with pytest.raises(ValueError, match="incremental ISL expands beyond max_tokens"):
        _generator(trace_file, max_tokens=64).generate_requests()
