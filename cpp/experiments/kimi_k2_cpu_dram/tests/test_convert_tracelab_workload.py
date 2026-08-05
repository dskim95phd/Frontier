"""Focused tests for the TraceLab v0.0.2 -> Frontier workload converter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

duckdb = pytest.importorskip("duckdb")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import convert_tracelab_workload as converter  # noqa: E402


def _make_trace_db(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE rounds (
            round_pk BIGINT,
            ingest_seq BIGINT,
            provider VARCHAR,
            project VARCHAR,
            session_id VARCHAR,
            round_index BIGINT,
            model VARCHAR,
            input_tokens_total BIGINT,
            newly_append_tokens BIGINT,
            output_tokens BIGINT,
            trace_key VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE timing_events (
            round_pk BIGINT,
            event_index BIGINT,
            event_type VARCHAR,
            timestamp TIMESTAMP
        )
        """
    )
    con.executemany(
        "INSERT INTO rounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # Public v0.0.2 has no session_file column; trace_key stays
            # round-specific metadata and must not fragment source sessions.
            (1, 1, "codex", "project_a", "session_a", 0, "model_a", 100, 100, 10, "file_a:turn_x:0"),
            (2, 2, "codex", "project_a", "session_a", 1, "model_a", 135, 25, 5, "file_a:turn_x:1"),
            # This row has a negative timing gap but a positive logical ISL;
            # it remains in the session with think_time clamped to zero.
            (3, 3, "codex", "project_a", "session_a", 2, "model_a", 150, 15, 2, "file_a:turn_x:2"),
            # Context reduction starts a new numeric simulator session.
            (4, 4, "codex", "project_a", "session_a", 3, "model_a", 120, 1, 3, "file_a:turn_x:3"),
            # The first index break truncates this row and all later rows.
            (5, 5, "codex", "project_a", "session_a", 5, "model_a", 140, 20, 4, "file_a:turn_x:5"),
            (6, 6, "codex", "project_a", "session_a", 6, "model_a", 160, 20, 4, "file_a:turn_x:6"),
            # Nonpositive OSL rows cannot satisfy the C++ loader contract.
            (7, 7, "codex", "project_a", "session_a", 7, "model_a", 180, 20, 0, "file_a:turn_x:7"),
            (8, 8, "claude", "project_b", "session_b", 0, "model_b", 50, 50, 4, "file_b:round_z"),
        ],
    )
    con.executemany(
        "INSERT INTO timing_events VALUES (?, ?, ?, ?)",
        [
            (1, 1, "user_message", "2026-01-01 00:00:00"),
            (1, 2, "text", "2026-01-01 00:00:02"),
            (2, 1, "tool_result", "2026-01-01 00:00:05"),
            (2, 2, "text", "2026-01-01 00:00:08"),
            # Current input at 7s precedes the prior output at 8s.
            (3, 1, "tool_result", "2026-01-01 00:00:07"),
            (3, 2, "text", "2026-01-01 00:00:09"),
            (4, 1, "tool_result", "2026-01-01 00:00:10"),
            (4, 2, "text", "2026-01-01 00:00:11"),
            (5, 1, "tool_result", "2026-01-01 00:00:12"),
            (5, 2, "text", "2026-01-01 00:00:13"),
            (6, 1, "tool_result", "2026-01-01 00:00:14"),
            (6, 2, "text", "2026-01-01 00:00:15"),
            (8, 1, "user_message", "2026-01-01 00:01:00"),
            (8, 2, "text", "2026-01-01 00:01:01"),
        ],
    )
    con.close()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_gap_clamp_context_split_and_index_truncation(tmp_path: Path) -> None:
    db = tmp_path / "trace.duckdb"
    output = tmp_path / "workload.csv"
    metadata = tmp_path / "workload.json"
    _make_trace_db(db)

    manifest = tmp_path / "workload_manifest.csv"
    summary = converter.convert_database(
        db,
        output,
        metadata,
        manifest,
        session_arrival_rate=1.0,
    )
    rows = _read_rows(output)

    assert len(rows) == 5
    assert rows[0] == {
        "session_start_at": "0.000000000000",
        "think_time": "0.000000000000",
        "num_prefill_tokens": "100",
        "num_decode_tokens": "10",
        "session_id": "0",
        "session_turn_index": "0",
    }
    # 135 - 100 - 10 = 25 logical fresh prompt tokens; 5s - 2s = 3s gap.
    assert rows[1]["session_start_at"] == ""
    assert rows[1]["think_time"] == "3.000000000000"
    assert rows[1]["num_prefill_tokens"] == "25"
    assert rows[1]["session_id"] == "0"
    # Negative timing is clamped to zero while retaining the session.
    assert rows[2]["session_start_at"] == ""
    assert rows[2]["think_time"] == "0.000000000000"
    assert rows[2]["num_prefill_tokens"] == "10"  # 150 - 135 - 5
    assert rows[2]["session_id"] == "0"
    assert rows[2]["session_turn_index"] == "2"
    # Context reduction still starts a new simulator session, but its root is
    # anchored at the original shuffled root plus the observed source-relative
    # elapsed time (10 seconds here), not reset to t=0.
    assert rows[3]["session_start_at"] == "10.000000000000"
    assert rows[3]["num_prefill_tokens"] == "120"
    assert rows[3]["session_id"] == "1"
    assert rows[4]["session_start_at"] == "1.000000000000"
    assert rows[4]["session_id"] == "2"  # Claude source session.

    assert summary["counts"]["source_round_rows_selected"] == 8
    assert summary["counts"]["source_round_rows_emitted"] == 5
    assert summary["counts"]["dropped_nonpositive_output"] == 1
    assert summary["counts"]["rows_discarded_after_index_break"] == 2
    assert summary["split_reasons"]["logical_delta_nonpositive"] == 1
    assert summary["policy_counts"]["negative_gap_clamped"] == 1
    assert summary["policy_counts"]["index_break_truncations"] == 1
    assert summary["sampling"]["before_sampling"]["source_sessions"] == 2
    assert summary["sampling"]["after_sampling"]["source_sessions"] == 2
    assert summary["sampling"]["source_order_shuffled"] is True
    assert summary["sampling"]["session_arrival_rate_per_second"] == 1.0
    manifest_rows = _read_rows(manifest)
    assert manifest_rows[0]["final_context_tokens"] == "152"
    assert manifest_rows[3]["session_start_at"] == "10.000000000000"
    assert summary["distributions"]["think_time_seconds_successors"]["p50"] == 1.5
    written = json.loads(metadata.read_text(encoding="utf-8"))
    assert written["token_assumptions"]["successor_isl"].startswith("current input_tokens_total")
    assert written["arrival_assumptions"]["root_arrival_distribution"].startswith("deterministic stratified")


def test_filters_and_sampling_are_deterministic(tmp_path: Path) -> None:
    db = tmp_path / "trace.duckdb"
    _make_trace_db(db)

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    summary = converter.convert_database(
        db,
        first,
        providers=["codex"],
        session_ids=["session_a"],
        sample_sessions=1,
        seed=19,
        session_arrival_rate=1.0,
    )
    converter.convert_database(
        db,
        second,
        providers=["codex"],
        session_ids=["session_a"],
        sample_sessions=1,
        seed=19,
        session_arrival_rate=1.0,
    )
    assert first.read_bytes() == second.read_bytes()
    rows = _read_rows(first)
    assert rows
    assert {row["session_id"] for row in rows} == {"0", "1"}
    assert summary["sampling"]["source_sessions_after_token_filter"] == 1
    assert summary["sampling"]["after_sampling"]["source_sessions"] == 1
    sampled_all = tmp_path / "sampled_all.csv"
    sampled_summary = converter.convert_database(
        db,
        sampled_all,
        sample_sessions=1,
        seed=19,
        session_arrival_rate=1.0,
    )
    assert sampled_summary["sampling"]["source_sessions_after_token_filter"] == 2
    assert sampled_summary["sampling"]["after_sampling"]["source_sessions"] == 1
    # A comma-separated filter is equivalent to a repeated argparse option.
    filtered = tmp_path / "filtered.csv"
    converter.convert_database(
        db,
        filtered,
        providers=["claude,codex"],
        models=["model_b"],
        session_arrival_rate=1.0,
    )
    assert len(_read_rows(filtered)) == 1


def test_repeated_epochs_get_fresh_session_ids_and_shifted_roots(tmp_path: Path) -> None:
    db = tmp_path / "trace.duckdb"
    _make_trace_db(db)
    output = tmp_path / "repeated.csv"
    summary = converter.convert_database(
        db,
        output,
        sample_sessions=2,
        seed=7,
        session_arrival_rate=1.0,
        session_repetitions=2,
    )

    rows = _read_rows(output)
    assert len(rows) == 10
    roots = [row for row in rows if row["session_start_at"]]
    first_epoch = [row for row in roots if int(row["session_id"]) < 3]
    second_epoch = [row for row in roots if int(row["session_id"]) >= 3]
    assert len(first_epoch) == len(second_epoch) == 3
    assert min(float(row["session_start_at"]) for row in second_epoch) >= 2.0
    assert summary["sampling"]["session_repetitions"] == 2
    assert summary["sampling"]["epoch_duration_seconds"] == 2.0
    assert summary["counts"]["source_session_injections_emitted"] == 4
    assert {item["epoch_index"] for item in summary["session_map"]} == {0, 1}


def test_seeded_session_shuffle_and_arrival_strata(tmp_path: Path) -> None:
    db = tmp_path / "trace.duckdb"
    _make_trace_db(db)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first_summary = converter.convert_database(
        db,
        first,
        session_arrival_rate=2.0,
        seed=0,
    )
    second_summary = converter.convert_database(
        db,
        second,
        session_arrival_rate=2.0,
        seed=1,
    )
    first_order = [item["source_session_id"] for item in first_summary["session_map"] if item["segment_index"] == 0]
    second_order = [item["source_session_id"] for item in second_summary["session_map"] if item["segment_index"] == 0]
    assert first_order != second_order
    assert first_summary["session_map"][0]["segment_root_arrival_seconds"] == 0.0
    assert first_summary["session_map"][1]["segment_root_arrival_seconds"] == 10.0
    assert first_summary["session_map"][-1]["source_root_arrival_seconds"] == 0.5


def test_cli_arrival_rate_has_no_implicit_default() -> None:
    parser = converter._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--db", "trace.duckdb", "--output", "out.csv"])
