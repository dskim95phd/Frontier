#!/usr/bin/env python3
"""Convert a TraceLab v0.0.2 DuckDB trace to Frontier's six-column CSV.

The C++ request loader accepts the following columns::

    session_start_at,think_time,num_prefill_tokens,num_decode_tokens,
    session_id,session_turn_index

TraceLab stores one row per model step in ``rounds``.  A row is not always a
user turn: tool-result continuations are rows too.  This converter deliberately
keeps every usable model step and uses timing events to turn the observed gap
between adjacent steps into Frontier's successor ``think_time``.

Token mapping (the default ``logical_delta`` mode) is intentionally different
from simply copying ``newly_append_tokens``.  Frontier's prefix-cache
materializer adds the prior session context itself.  For an adjacent pair of
TraceLab rows, the fresh logical prompt is therefore::

    current.input_tokens_total - previous.input_tokens_total
        - previous.output_tokens

The first row of each simulator session uses its full ``input_tokens_total``.
If the logical delta is non-positive or gap timing is missing, the current row
starts a new simulator session with its full input as ISL.  Its absolute root
arrival is not reset to zero: it is the seed-shuffled source root arrival plus
the observed source-relative elapsed time to that segment (or the previous
segment's output time when the current input timestamp is unavailable).  A
negative observed gap is clamped to zero while retaining the session.  At the
first non-adjacent source round index, that row and the rest of the source
session are discarded because the intervening requests are not observable.

The CLI requires an explicit ``--session-arrival-rate``.  Retained source
sessions are Fisher-Yates shuffled with ``--seed`` and roots are placed in
deterministic strata at ``index / session_arrival_rate`` seconds.  Optional
session repetitions form consecutive shuffled epochs; every injection gets a
fresh numeric session ID, so repeated templates never share KV state.  This
keeps sampling and arrival ordering reproducible without choosing an
unapproved default arrival rate.

``newly_append_tokens`` remains available as ``--isl-mode raw_append`` for a
diagnostic workload, but it is not the default because it is a provider cache
accounting field and can include replayed/prior-output context when fed through
Frontier's own materializer.

The script only requires DuckDB at runtime.  The checked-in TraceLab virtual
environment under ``outputs/tools/tracelab-venv`` is one way to run it; no
NumPy or pandas dependency is needed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence


CONVERTER_VERSION = "tracelab-v0.0.2-frontier-csv-v2"
CSV_FIELDS = (
    "session_start_at",
    "think_time",
    "num_prefill_tokens",
    "num_decode_tokens",
    "session_id",
    "session_turn_index",
)
MANIFEST_FIELDS = (
    "request_id",
    "session_id",
    "session_turn_index",
    "num_turns",
    "new_input_tokens",
    "decode_tokens",
    "prior_context_tokens",
    "prior_decode_tokens",
    "theoretical_min_prefill_tokens",
    "implementation_min_prefill_tokens",
    "materialized_prefill_tokens",
    "actual_scheduled_prefill_tokens",
    "actual_extra_ratio",
    "session_start_at",
    "think_time",
    "final_context_tokens",
)
MODEL_OUTPUT_EVENTS = ("reasoning", "text", "tool_call")
INPUT_EVENTS = ("user_message", "tool_result")


def _default_db_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "outputs" / "datasets" / "tracelab" / "v0.0.2" / "syfi_coding_trace.duckdb"


def _default_output_path() -> Path:
    return Path(__file__).resolve().parent / "workloads" / "tracelab_v0.0.2.csv"


def _load_duckdb() -> Any:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in CLI environments
        raise RuntimeError(
            "DuckDB is required to read TraceLab; install the duckdb Python "
            "package or run with outputs/tools/tracelab-venv/Scripts/python.exe"
        ) from exc
    return duckdb


def _flatten_filters(values: Sequence[str] | None) -> list[str]:
    """Flatten repeatable and comma-separated argparse filter values."""

    result: list[str] = []
    for value in values or ():
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def _source_key(source_round: "_Round") -> tuple[str, str, str, str]:
    # ``session_id`` is only unique within parts of the public corpus.  Keep
    # provider/project/session-file in the grouping key so a reused source id
    # cannot join two independent traces in Frontier's numeric namespace.
    return (
        source_round.provider,
        source_round.project or "",
        source_round.source_file,
        source_round.source_session_id,
    )


@dataclass
class _Round:
    """A usable TraceLab round and the timing observations needed for a gap."""

    round_pk: int
    ingest_seq: int
    provider: str
    project: str | None
    source_file: str
    source_session_id: str
    source_round_index: int | None
    model: str | None
    input_tokens_total: int
    newly_append_tokens: int
    output_tokens: int
    first_input_at: datetime | None
    first_event_at: datetime | None
    last_output_at: datetime | None
    timing_start_source: str
    timing_end_source: str

    @property
    def timing_start_at(self) -> datetime | None:
        return self.first_input_at or self.first_event_at

    @property
    def timing_end_at(self) -> datetime | None:
        return self.last_output_at or self.first_event_at


@dataclass
class _OutputRow:
    session_id: int
    turn_index: int
    session_start_at: float | None
    think_time: float
    prefill_tokens: int
    decode_tokens: int
    source: _Round
    segment_index: int


def _query_rounds(
    db_path: Path,
    *,
    providers: Sequence[str],
    models: Sequence[str],
    session_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Read rounds and timing aggregates without pandas/numpy."""

    duckdb = _load_duckdb()
    if not db_path.exists():
        raise FileNotFoundError(f"TraceLab DuckDB does not exist: {db_path}")

    predicates: list[str] = []
    parameters: list[Any] = []

    try:
        connection = duckdb.connect(str(db_path), read_only=True)
    except Exception as exc:
        raise RuntimeError(f"could not open TraceLab DuckDB {db_path}: {exc}") from exc
    try:
        schema_rows = connection.execute("PRAGMA table_info('rounds')").fetchall()
        round_columns = {str(row[1]) for row in schema_rows}
        required = {
            "round_pk",
            "ingest_seq",
            "provider",
            "project",
            "session_id",
            "round_index",
            "model",
            "input_tokens_total",
            "newly_append_tokens",
            "output_tokens",
        }
        missing = sorted(required - round_columns)
        if missing:
            raise RuntimeError(f"TraceLab rounds table is missing columns: {', '.join(missing)}")
        source_file_expr = "r.session_file" if "session_file" in round_columns else "NULL"
        trace_key_expr = "r.trace_key" if "trace_key" in round_columns else "NULL"
        group_suffix = ""
        if "trace_key" in round_columns:
            group_suffix += ", r.trace_key"
        if "session_file" in round_columns:
            group_suffix += ", r.session_file"

        def add_in_filter(column: str, values: Sequence[str]) -> None:
            if not values:
                return
            placeholders = ",".join("?" for _ in values)
            predicates.append(f"r.{column} IN ({placeholders})")
            parameters.extend(values)

        add_in_filter("provider", providers)
        add_in_filter("model", models)
        add_in_filter("session_id", session_ids)
        where = f"WHERE {' AND '.join(predicates)}" if predicates else ""

        # Keep the timing expressions explicit.  The canonical input timestamp is
        # the first user_message/tool_result; first_event_at is a deterministic
        # fallback for traces whose event stream starts with another event.  The
        # canonical predecessor completion is the latest model-output event.
        query = f"""
        SELECT
            r.round_pk,
            r.ingest_seq,
            r.provider,
            r.project,
            {source_file_expr} AS source_file,
            {trace_key_expr} AS trace_key,
            r.session_id AS source_session_id,
            r.round_index AS source_round_index,
            r.model,
            r.input_tokens_total,
            r.newly_append_tokens,
            r.output_tokens,
            MIN(t.timestamp) FILTER (
                WHERE t.event_type IN ('user_message', 'tool_result')
            ) AS first_input_at,
            MIN(t.timestamp) AS first_event_at,
            MAX(t.timestamp) FILTER (
                WHERE t.event_type IN ('reasoning', 'text', 'tool_call')
            ) AS last_output_at,
            CASE
                WHEN MIN(t.timestamp) FILTER (
                    WHERE t.event_type IN ('user_message', 'tool_result')
                ) IS NOT NULL THEN 'input_event'
                WHEN MIN(t.timestamp) IS NOT NULL THEN 'first_event_fallback'
                ELSE 'missing'
            END AS timing_start_source,
            CASE
                WHEN MAX(t.timestamp) FILTER (
                    WHERE t.event_type IN ('reasoning', 'text', 'tool_call')
                ) IS NOT NULL THEN 'model_output'
                WHEN MIN(t.timestamp) IS NOT NULL THEN 'first_event_fallback'
                ELSE 'missing'
            END AS timing_end_source
        FROM rounds AS r
        LEFT JOIN timing_events AS t ON t.round_pk = r.round_pk
        {where}
        GROUP BY
            r.round_pk, r.ingest_seq, r.provider, r.project,
            r.session_id, r.round_index, r.model,
            r.input_tokens_total, r.newly_append_tokens, r.output_tokens{group_suffix}
        ORDER BY r.ingest_seq, r.round_pk
        """
        cursor = connection.execute(query, parameters)
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        connection.close()


def _as_int(value: Any, *, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected an integer token/count, got {value!r}") from exc


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return _as_int(value)


def _round_from_record(record: Mapping[str, Any]) -> _Round:
    return _Round(
        round_pk=_as_int(record["round_pk"]),
        ingest_seq=_as_int(record.get("ingest_seq"), default=_as_int(record["round_pk"])),
        provider=str(record.get("provider") or ""),
        project=(None if record.get("project") is None else str(record.get("project"))),
        source_file=str(record.get("source_file") or ""),
        source_session_id=str(record.get("source_session_id") or ""),
        source_round_index=_as_optional_int(record.get("source_round_index")),
        model=(None if record.get("model") is None else str(record.get("model"))),
        input_tokens_total=_as_int(record.get("input_tokens_total")),
        newly_append_tokens=_as_int(record.get("newly_append_tokens")),
        output_tokens=_as_int(record.get("output_tokens")),
        first_input_at=record.get("first_input_at"),
        first_event_at=record.get("first_event_at"),
        last_output_at=record.get("last_output_at"),
        timing_start_source=str(record.get("timing_start_source") or "missing"),
        timing_end_source=str(record.get("timing_end_source") or "missing"),
    )


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


def _describe(values: Sequence[float | int]) -> dict[str, float | int | None]:
    """Return compact, JSON-safe distribution statistics."""

    numeric = [float(value) for value in values]
    if not numeric:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(numeric),
        "min": min(numeric),
        "mean": statistics.fmean(numeric),
        "p50": _percentile(numeric, 0.50),
        "p90": _percentile(numeric, 0.90),
        "p99": _percentile(numeric, 0.99),
        "max": max(numeric),
    }


def _session_order(groups: Mapping[tuple[str, str, str, str], Sequence[_Round]]) -> list[tuple[str, str, str, str]]:
    return sorted(
        groups,
        key=lambda key: (
            groups[key][0].round_pk if groups[key] else math.inf,
            groups[key][0].ingest_seq if groups[key] else math.inf,
            key,
        ),
    )


def _write_csv(path: Path, rows: Sequence[_OutputRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_FIELDS)
        for row in rows:
            start = "" if row.session_start_at is None else f"{row.session_start_at:.12f}"
            writer.writerow(
                (
                    start,
                    f"{row.think_time:.12f}",
                    row.prefill_tokens,
                    row.decode_tokens,
                    row.session_id,
                    row.turn_index,
                )
            )


def _write_manifest(path: Path, rows: Sequence[_OutputRow]) -> None:
    """Write the Kimi-study-compatible request manifest for TraceLab rows.

    The converter's logical-delta rows are fresh input tokens.  Frontier's
    session-prefix materializer adds ``prior_context_tokens``; therefore the
    final context for a split segment is computed independently per emitted
    numeric simulator session.  This manifest is intentionally optional for
    callers that only need the six-column CSV, but enables the shared sweep
    runner's oracle-capacity calculation.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    session_rows: dict[int, list[_OutputRow]] = {}
    for row in rows:
        session_rows.setdefault(row.session_id, []).append(row)
    final_context_by_request: dict[int, int] = {}
    for session_id, values in session_rows.items():
        context = 0
        for row in values:
            context += row.prefill_tokens + row.decode_tokens
        for row in values:
            final_context_by_request[row.source.round_pk] = context

    # Numeric request IDs are the emitted CSV row order.  ``round_pk`` is not
    # necessarily contiguous, so use the same order as _write_csv.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MANIFEST_FIELDS)
        context_by_session: dict[int, int] = {}
        previous_decode_by_session: dict[int, int] = {}
        for request_id, row in enumerate(rows):
            prior_context = context_by_session.get(row.session_id, 0)
            prior_decode = previous_decode_by_session.get(row.session_id, 0)
            materialized = prior_context + row.prefill_tokens
            ratio = (materialized - row.prefill_tokens) / row.prefill_tokens
            final_context = final_context_by_request[row.source.round_pk]
            session_start = "" if row.session_start_at is None else f"{row.session_start_at:.12f}"
            writer.writerow(
                (
                    request_id,
                    row.session_id,
                    row.turn_index,
                    len(session_rows[row.session_id]),
                    row.prefill_tokens,
                    row.decode_tokens,
                    prior_context,
                    prior_decode,
                    row.prefill_tokens,
                    row.prefill_tokens + prior_decode + (16 if row.turn_index > 0 else 0),
                    materialized,
                    materialized,
                    f"{ratio:.12f}",
                    session_start,
                    f"{row.think_time:.12f}",
                    final_context,
                )
            )
            context_by_session[row.session_id] = materialized + row.decode_tokens
            previous_decode_by_session[row.session_id] = row.decode_tokens


def _filter_and_sample_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    sample_sessions: int | None,
    seed: int,
) -> tuple[dict[tuple[str, str, str, str], list[_Round]], dict[str, int]]:
    groups: dict[tuple[str, str, str, str], list[_Round]] = {}
    dropped = {"nonpositive_input": 0, "nonpositive_output": 0}
    for record in records:
        source_round = _round_from_record(record)
        if source_round.input_tokens_total <= 0:
            dropped["nonpositive_input"] += 1
            continue
        if source_round.output_tokens <= 0:
            dropped["nonpositive_output"] += 1
            continue
        groups.setdefault(_source_key(source_round), []).append(source_round)

    for values in groups.values():
        values.sort(key=lambda item: (item.source_round_index is None, item.source_round_index or 0, item.ingest_seq, item.round_pk))

    all_keys = _session_order(groups)
    if sample_sessions is not None:
        if sample_sessions <= 0:
            raise ValueError("sample_sessions must be positive when supplied")
        if sample_sessions < len(all_keys):
            selected = set(random.Random(seed).sample(all_keys, sample_sessions))
            groups = {key: groups[key] for key in all_keys if key in selected}
    return groups, dropped


def _build_output_rows(
    groups: Mapping[tuple[str, str, str, str], Sequence[_Round]],
    *,
    isl_mode: str,
    source_order: Sequence[tuple[str, str, str, str]],
    session_arrival_rate: float,
) -> tuple[
    list[_OutputRow],
    list[dict[str, Any]],
    dict[str, int],
    dict[str, int],
    list[float],
]:
    if isl_mode not in {"logical_delta", "raw_append", "input_total"}:
        raise ValueError(f"unknown isl_mode: {isl_mode}")

    output_rows: list[_OutputRow] = []
    session_map: list[dict[str, Any]] = []
    split_reasons = {
        "timing_missing": 0,
        "logical_delta_nonpositive": 0,
        "raw_append_nonpositive": 0,
    }
    policy_counts = {
        "negative_gap_clamped": 0,
        "index_break_truncations": 0,
        "rows_discarded_after_index_break": 0,
        "segment_arrival_timing_fallbacks": 0,
    }
    gaps: list[float] = []
    next_session_id = 0

    for source_position, source_key in enumerate(source_order):
        source_rows = groups[source_key]
        current_segment: list[_Round] = []
        segment_index = 0
        source_root_at = source_rows[0].timing_start_at if source_rows else None
        source_root_arrival = source_position / session_arrival_rate
        previous_segment_end_at: datetime | None = None

        def flush_segment() -> None:
            nonlocal next_session_id, segment_index, current_segment
            nonlocal previous_segment_end_at
            if not current_segment:
                return
            numeric_id = next_session_id
            next_session_id += 1
            segment_first = current_segment[0]
            arrival_anchor = "stratified_root"
            source_relative_elapsed = 0.0
            if segment_index > 0:
                segment_start_at = segment_first.timing_start_at
                if source_root_at is not None and segment_start_at is not None:
                    source_relative_elapsed = max(
                        0.0,
                        (segment_start_at - source_root_at).total_seconds(),
                    )
                    arrival_anchor = "source_relative_first_input"
                    if previous_segment_end_at is not None:
                        previous_end_elapsed = max(
                            0.0,
                            (previous_segment_end_at - source_root_at).total_seconds(),
                        )
                        if source_relative_elapsed < previous_end_elapsed:
                            # A compaction split cannot arrive before the
                            # previous segment has ended.  Preserve a
                            # negative observed gap as an immediate successor
                            # (zero gap), analogous to successor think-time
                            # clamping within one simulator session.
                            source_relative_elapsed = previous_end_elapsed
                            arrival_anchor = "previous_segment_output_clamped"
                elif previous_segment_end_at is not None and source_root_at is not None:
                    source_relative_elapsed = max(
                        0.0,
                        (previous_segment_end_at - source_root_at).total_seconds(),
                    )
                    arrival_anchor = "previous_segment_output_fallback"
                    policy_counts["segment_arrival_timing_fallbacks"] += 1
                else:
                    arrival_anchor = "stratified_root_fallback"
                    policy_counts["segment_arrival_timing_fallbacks"] += 1
            segment_root_arrival = source_root_arrival + source_relative_elapsed
            for turn_index, item in enumerate(current_segment):
                if turn_index == 0:
                    prefill = item.input_tokens_total
                    think = 0.0
                    start = 0.0
                else:
                    previous = current_segment[turn_index - 1]
                    if isl_mode == "raw_append":
                        prefill = item.newly_append_tokens
                    elif isl_mode == "input_total":
                        prefill = item.input_tokens_total
                    else:
                        prefill = item.input_tokens_total - previous.input_tokens_total - previous.output_tokens
                    previous_end = previous.timing_end_at
                    current_start = item.timing_start_at
                    if current_start is None or previous_end is None:
                        # This branch is normally prevented by segmentation;
                        # retain a safe positive value for defensive callers.
                        think = 0.0
                    else:
                        think = (current_start - previous_end).total_seconds()
                    start = None
                output_rows.append(
                    _OutputRow(
                        session_id=numeric_id,
                        turn_index=turn_index,
                        session_start_at=(
                            segment_root_arrival if start is not None else None
                        ),
                        think_time=max(0.0, float(think)),
                        prefill_tokens=int(prefill),
                        decode_tokens=item.output_tokens,
                        source=item,
                        segment_index=segment_index,
                    )
                )
                if turn_index > 0:
                    gaps.append(max(0.0, float(think)))
            session_map.append(
                {
                    "numeric_session_id": numeric_id,
                    "segment_index": segment_index,
                    "provider": source_key[0],
                    "project": source_key[1] or None,
                    "source_session_file": source_key[2] or None,
                    "source_session_id": source_key[3],
                    "source_round_index_start": current_segment[0].source_round_index,
                    "source_round_index_end": current_segment[-1].source_round_index,
                    "round_count": len(current_segment),
                    "first_round_pk": current_segment[0].round_pk,
                    "source_root_arrival_seconds": source_root_arrival,
                    "segment_root_arrival_seconds": segment_root_arrival,
                    "source_relative_elapsed_seconds": source_relative_elapsed,
                    "segment_arrival_anchor": arrival_anchor,
                    "source_root_timing_start": (
                        source_root_at.isoformat() if source_root_at is not None else None
                    ),
                    "segment_timing_start": (
                        segment_first.timing_start_at.isoformat()
                        if segment_first.timing_start_at is not None
                        else None
                    ),
                    "previous_segment_timing_end": (
                        previous_segment_end_at.isoformat()
                        if previous_segment_end_at is not None
                        else None
                    ),
                }
            )
            previous_segment_end_at = current_segment[-1].timing_end_at
            segment_index += 1
            current_segment = []

        for item_offset, item in enumerate(source_rows):
            if not current_segment:
                current_segment.append(item)
                continue
            previous = current_segment[-1]
            reason: str | None = None
            if (
                previous.source_round_index is None
                or item.source_round_index is None
                or item.source_round_index != previous.source_round_index + 1
            ):
                policy_counts["index_break_truncations"] += 1
                policy_counts["rows_discarded_after_index_break"] += len(source_rows) - item_offset
                flush_segment()
                break
            if item.timing_start_at is None or previous.timing_end_at is None:
                reason = "timing_missing"
            else:
                observed_gap = (item.timing_start_at - previous.timing_end_at).total_seconds()
                if observed_gap < 0.0:
                    policy_counts["negative_gap_clamped"] += 1
            if reason is None and isl_mode == "logical_delta" and (
                item.input_tokens_total - previous.input_tokens_total - previous.output_tokens <= 0
            ):
                reason = "logical_delta_nonpositive"
            elif reason is None and isl_mode == "raw_append" and item.newly_append_tokens <= 0:
                reason = "raw_append_nonpositive"
            if reason is not None:
                split_reasons[reason] += 1
                flush_segment()
            current_segment.append(item)
        flush_segment()

    return output_rows, session_map, split_reasons, policy_counts, gaps


def convert_database(
    db_path: Path,
    output_path: Path,
    metadata_path: Path | None = None,
    manifest_output_path: Path | None = None,
    *,
    providers: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    session_ids: Sequence[str] | None = None,
    sample_sessions: int | None = None,
    seed: int = 0,
    session_arrival_rate: float | None = None,
    session_repetitions: int = 1,
    isl_mode: str = "logical_delta",
) -> dict[str, Any]:
    """Convert a TraceLab DuckDB and return the JSON-serializable summary.

    ``providers``, ``models``, and ``session_ids`` are exact-match filters.
    ``sample_sessions`` samples source sessions without replacement using
    ``seed``.  The retained source-session order is then shuffled with the
    same seed and root arrivals are placed in deterministic strata at
    ``index / session_arrival_rate`` seconds.  ``session_arrival_rate`` is
    intentionally required by the CLI and has no converter default.
    """

    if session_arrival_rate is None or not math.isfinite(session_arrival_rate) or session_arrival_rate <= 0.0:
        raise ValueError("session_arrival_rate must be explicitly supplied as a finite positive sessions/second rate")
    if session_repetitions <= 0:
        raise ValueError("session_repetitions must be positive")

    provider_filter = _flatten_filters(providers)
    model_filter = _flatten_filters(models)
    session_filter = _flatten_filters(session_ids)
    records = _query_rounds(
        Path(db_path),
        providers=provider_filter,
        models=model_filter,
        session_ids=session_filter,
    )
    if not records:
        raise ValueError("no TraceLab rounds matched the requested filters")
    groups, dropped = _filter_and_sample_groups(
        records, sample_sessions=sample_sessions, seed=seed
    )
    if not groups:
        raise ValueError("all TraceLab rounds matched the filters but had nonpositive token counts")
    output_rows: list[_OutputRow] = []
    session_map: list[dict[str, Any]] = []
    split_reasons = {
        "timing_missing": 0,
        "logical_delta_nonpositive": 0,
        "raw_append_nonpositive": 0,
    }
    policy_counts = {
        "negative_gap_clamped": 0,
        "index_break_truncations": 0,
        "rows_discarded_after_index_break": 0,
        "segment_arrival_timing_fallbacks": 0,
    }
    gaps: list[float] = []
    epoch_duration_seconds = len(groups) / session_arrival_rate
    for epoch_index in range(session_repetitions):
        source_order = _session_order(groups)
        random.Random(seed + epoch_index).shuffle(source_order)
        epoch_rows, epoch_map, epoch_splits, epoch_policy, epoch_gaps = _build_output_rows(
            groups,
            isl_mode=isl_mode,
            source_order=source_order,
            session_arrival_rate=session_arrival_rate,
        )
        session_id_offset = len(session_map)
        arrival_offset = epoch_index * epoch_duration_seconds
        for row in epoch_rows:
            row.session_id += session_id_offset
            if row.session_start_at is not None:
                row.session_start_at += arrival_offset
        for item in epoch_map:
            item["numeric_session_id"] = int(item["numeric_session_id"]) + session_id_offset
            item["source_root_arrival_seconds"] = (
                float(item["source_root_arrival_seconds"]) + arrival_offset
            )
            item["segment_root_arrival_seconds"] = (
                float(item["segment_root_arrival_seconds"]) + arrival_offset
            )
            item["epoch_index"] = epoch_index
        output_rows.extend(epoch_rows)
        session_map.extend(epoch_map)
        gaps.extend(epoch_gaps)
        for key, value in epoch_splits.items():
            split_reasons[key] += value
        for key, value in epoch_policy.items():
            policy_counts[key] += value
    if not output_rows:
        raise ValueError("no usable workload rows remained after conversion")

    _write_csv(Path(output_path), output_rows)
    if manifest_output_path is not None:
        _write_manifest(Path(manifest_output_path), output_rows)
    prefill_total = sum(row.prefill_tokens for row in output_rows)
    decode_total = sum(row.decode_tokens for row in output_rows)
    source_input_total = sum(row.source.input_tokens_total for row in output_rows)
    source_append_total = sum(row.source.newly_append_tokens for row in output_rows)
    emitted_isl = [row.prefill_tokens for row in output_rows]
    emitted_osl = [row.decode_tokens for row in output_rows]
    segment_lengths = [int(item["round_count"]) for item in session_map]
    timing_start_fallbacks = sum(
        row.source.timing_start_source != "input_event" for row in output_rows
    )
    timing_end_fallbacks = sum(
        row.source.timing_end_source != "model_output" for row in output_rows
    )
    # Keep sampling counters separate from post-sampling conversion counters.
    # This makes a sampled run auditable without inferring denominators from
    # emitted CSV rows (which can additionally be reduced by index breaks).
    valid_records = [
        _round_from_record(record)
        for record in records
        if _as_int(record.get("input_tokens_total")) > 0
        and _as_int(record.get("output_tokens")) > 0
    ]
    source_sessions_before_sampling = len({_source_key(item) for item in valid_records})
    source_round_rows_before_sampling = len(valid_records)
    source_sessions_after_sampling = len(groups)
    source_round_rows_after_sampling = sum(len(values) for values in groups.values())
    summary: dict[str, Any] = {
        "converter_version": CONVERTER_VERSION,
        "source_db": str(Path(db_path)),
        "output_csv": str(Path(output_path)),
        "manifest_csv": (
            str(Path(manifest_output_path))
            if manifest_output_path is not None
            else None
        ),
        "filters": {
            "provider": provider_filter,
            "model": model_filter,
            "session_id": session_filter,
        },
        "sampling": {
            "sample_sessions": sample_sessions,
            "seed": seed,
            "source_order_shuffled": True,
            "source_order_seed": seed,
            "session_arrival_rate_per_second": session_arrival_rate,
            "session_repetitions": session_repetitions,
            "epoch_duration_seconds": epoch_duration_seconds,
            "measurement_epoch_index": session_repetitions - 1,
            "root_arrival_policy": "stratified index / session_arrival_rate after seed shuffle",
            "before_sampling": {
                "source_sessions": source_sessions_before_sampling,
                "source_round_rows": source_round_rows_before_sampling,
            },
            "after_sampling": {
                "source_sessions": source_sessions_after_sampling,
                "source_round_rows": source_round_rows_after_sampling,
            },
            "sampled_out": {
                "source_sessions": source_sessions_before_sampling - source_sessions_after_sampling,
                "source_round_rows": source_round_rows_before_sampling - source_round_rows_after_sampling,
            },
            # Flat aliases keep this metadata easy to consume from PowerShell
            # and preserve the terminology used in the study notes.
            "source_sessions_before_sampling": source_sessions_before_sampling,
            "source_sessions_after_sampling": source_sessions_after_sampling,
            "source_round_rows_before_sampling": source_round_rows_before_sampling,
            "source_round_rows_after_sampling": source_round_rows_after_sampling,
            # Token filtering happens before optional session sampling, so this
            # legacy flat name must use the pre-sampling denominator.
            "source_sessions_after_token_filter": source_sessions_before_sampling,
        },
        "isl_mode": isl_mode,
        "counts": {
            "source_round_rows_selected": len(records),
            "source_round_rows_emitted": len(output_rows),
            "source_sessions_emitted": len(groups),
            "source_session_injections_emitted": len(groups) * session_repetitions,
            "simulator_sessions_emitted": len(session_map),
            "dropped_nonpositive_input": dropped["nonpositive_input"],
            "dropped_nonpositive_output": dropped["nonpositive_output"],
            "rows_discarded_after_index_break": policy_counts["rows_discarded_after_index_break"],
            "split_segments": sum(split_reasons.values()),
            "think_time_successor_rows": len(gaps),
        },
        "split_reasons": split_reasons,
        "policy_counts": policy_counts,
        "token_totals": {
            "frontier_prefill_tokens": prefill_total,
            "frontier_decode_tokens": decode_total,
            "source_input_tokens_total": source_input_total,
            "source_newly_append_tokens": source_append_total,
            "frontier_prefill_to_decode_ratio": prefill_total / decode_total if decode_total else None,
        },
        "think_time_seconds": {
            "mean": statistics.fmean(gaps) if gaps else None,
            "p50": _percentile(gaps, 0.50),
            "p90": _percentile(gaps, 0.90),
            "p99": _percentile(gaps, 0.99),
            "max": max(gaps) if gaps else None,
        },
        "distributions": {
            "think_time_seconds_successors": _describe(gaps),
            "isl_tokens": _describe(emitted_isl),
            "osl_tokens": _describe(emitted_osl),
            "simulator_segment_rounds": _describe(segment_lengths),
        },
        "timing_assumptions": {
            "successor_gap": "current first user_message/tool_result (fallback first timing event) minus previous latest reasoning/text/tool_call (fallback first timing event)",
            "first_request_arrival": "seed-shuffled source roots are stratified at index / session_arrival_rate seconds",
            "split_segment_arrival": "split roots use shuffled source root arrival plus source-relative elapsed time to the segment first input; when timing is missing, previous segment output is the fallback anchor",
            "cross_session_dependency": "Frontier does not impose completion dependencies between simulator sessions; absolute split roots preserve observed chronology but do not force predecessor completion",
            "negative_gap": "clamp think_time to 0 seconds and retain the simulator session",
            "missing_gap": "split into a new simulator session",
            "round_index_break": "discard the first non-adjacent row and all later rows in that source session",
            "no_think_time_cap": True,
            "input_event_timestamp_fallback_rows": timing_start_fallbacks,
            "model_output_timestamp_fallback_rows": timing_end_fallbacks,
        },
        "token_assumptions": {
            "first_segment_isl": "rounds.input_tokens_total",
            "successor_isl": "current input_tokens_total - previous input_tokens_total - previous output_tokens",
            "raw_append_diagnostic": "rounds.newly_append_tokens is used only by --isl-mode raw_append",
            "osl": "rounds.output_tokens (including reasoning output where TraceLab includes it)",
        },
        "arrival_assumptions": {
            "session_arrival_rate_per_second": session_arrival_rate,
            "root_arrival_distribution": "deterministic stratified strata, one root per 1/rate interval",
            "source_session_order": "seeded Fisher-Yates shuffle after token filtering and optional sampling",
            "epoch_policy": "each repetition reshuffles the sampled source-session pool with seed + epoch_index and assigns fresh simulator session IDs",
            "session_repetitions": session_repetitions,
            "epoch_duration_seconds": epoch_duration_seconds,
            "split_segment_policy": "original shuffled root arrival + source-relative elapsed time",
            "limitations": [
                "Frontier successor timing remains completion-relative within each emitted simulator session.",
                "Different emitted sessions have no completion dependency, so a split root can be scheduled before its source segment has completed under extreme simulator queueing.",
            ],
        },
        "session_map": session_map,
    }
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=_default_db_path(), help="TraceLab v0.0.2 DuckDB path.")
    parser.add_argument("--output", type=Path, default=_default_output_path(), help="Frontier six-column CSV output path.")
    parser.add_argument("--metadata-output", type=Path, help="Summary metadata JSON path (default: output stem + _metadata.json).")
    parser.add_argument("--manifest-output", type=Path, help="Optional Kimi-study-compatible request manifest CSV (default: output stem + _manifest.csv).")
    parser.add_argument("--provider", action="append", help="Exact provider filter; repeat or comma-separate values.")
    parser.add_argument("--model", action="append", help="Exact model filter; repeat or comma-separate values.")
    parser.add_argument("--session-id", "--session", dest="session_id", action="append", help="Exact TraceLab source session_id filter; repeat or comma-separate values.")
    parser.add_argument("--sample-sessions", "--max-sessions", dest="sample_sessions", type=int, help="Deterministically sample this many source sessions without replacement.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used only for --sample-sessions (default: 0).")
    parser.add_argument("--session-arrival-rate", type=float, required=True, help="Explicit stratified source-session root arrival rate in sessions/second; there is intentionally no converter default.")
    parser.add_argument("--session-repetitions", type=int, default=1, help="Number of consecutive shuffled epochs of the sampled source-session pool (default: 1).")
    parser.add_argument("--isl-mode", choices=("logical_delta", "raw_append", "input_total"), default="logical_delta", help="ISL mapping; logical_delta is the prefix-materializer-safe default.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metadata_path = args.metadata_output or args.output.with_name(args.output.stem + "_metadata.json")
    manifest_path = args.manifest_output or args.output.with_name(args.output.stem + "_manifest.csv")
    summary = convert_database(
        args.db,
        args.output,
        metadata_path,
        manifest_path,
        providers=args.provider,
        models=args.model,
        session_ids=args.session_id,
        sample_sessions=args.sample_sessions,
        seed=args.seed,
        session_arrival_rate=args.session_arrival_rate,
        session_repetitions=args.session_repetitions,
        isl_mode=args.isl_mode,
    )
    # A complete trace has tens of thousands of session-map entries.  Keep
    # stdout useful while retaining the full traceability map in metadata.
    console_summary = {key: value for key, value in summary.items() if key != "session_map"}
    console_summary["metadata_json"] = str(metadata_path)
    console_summary["manifest_csv"] = str(manifest_path)
    console_summary["session_map_entries"] = len(summary["session_map"])
    print(json.dumps(console_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
