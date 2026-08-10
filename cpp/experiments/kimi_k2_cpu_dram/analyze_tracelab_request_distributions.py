#!/usr/bin/env python3
"""Export request/turn distributions from the TraceLab v0.0.2 DuckDB.

The output is intentionally CSV-only and exact over the selected DuckDB.  It
separates human-triggered and tool-triggered model requests while retaining
mixed/unknown rows as explicit audit groups.

Definitions used by this analysis:

* human input wait: the previous same-session model-output event to a
  ``user_message`` event;
* tool wall latency: ``tool_calls.result_at - tool_calls.emitted_at``;
* tool-result gap: the previous same-session model-output event to a
  ``tool_result`` event;
* tools between user messages: tool calls in ``[user_i, user_(i+1))``;
* logical ISL: a source-audit estimate using Frontier's logical new-input delta
  for an adjacent request, otherwise the request's full input as a conservative
  reset estimate; this metric does not apply the converter's retention policy;
* request sequence length: ``rounds.input_tokens_total``;
* OSL: ``rounds.output_tokens``.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODEL_OUTPUT_TYPES = ("reasoning", "text", "tool_call")
LATENCY_BOUNDS_SECONDS = (
    0.001,
    0.01,
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
    60.0,
    300.0,
    600.0,
    1_800.0,
    3_600.0,
    21_600.0,
    86_400.0,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_db() -> Path:
    return _repo_root() / "outputs/datasets/tracelab/v0.0.2/syfi_coding_trace.duckdb"


def _default_output_dir() -> Path:
    return _repo_root() / "outputs/datasets/tracelab/v0.0.2/analysis/request_distributions"


def _load_duckdb() -> Any:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "duckdb is required; use outputs/tools/tracelab-venv/Scripts/python.exe"
        ) from exc
    return duckdb


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _percentile(sorted_values: Sequence[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight)


def _summary(metric: str, group: str, unit: str, values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    count = len(ordered)
    total = math.fsum(ordered)
    mean = total / count if count else None
    variance = (
        math.fsum((value - mean) ** 2 for value in ordered) / count
        if count and mean is not None
        else None
    )
    return {
        "metric": metric,
        "group": group,
        "unit": unit,
        "count": count,
        "mean": mean,
        "stddev_population": math.sqrt(variance) if variance is not None else None,
        "min": ordered[0] if ordered else None,
        "p10": _percentile(ordered, 0.10),
        "p25": _percentile(ordered, 0.25),
        "p50": _percentile(ordered, 0.50),
        "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1] if ordered else None,
        "sum": total if ordered else None,
    }


def _add_grouped(store: dict[tuple[str, str], list[float]], metric: str, provider: str, value: Any) -> None:
    if value is None:
        return
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return
    store[(metric, "all")].append(number)
    store[(metric, provider or "<unknown-provider>")].append(number)


def _latency_bin(value: float) -> tuple[float, float | None, str]:
    lower = 0.0
    for upper in LATENCY_BOUNDS_SECONDS:
        if value < upper:
            return lower, upper, f"[{lower:g},{upper:g})"
        lower = upper
    return lower, None, f"[{lower:g},inf)"


def _token_bin(value: float) -> tuple[int, int, str]:
    integer = max(0, int(value))
    if integer == 0:
        return 0, 0, "0"
    exponent = integer.bit_length() - 1
    lower = 1 << exponent
    upper = (1 << (exponent + 1)) - 1
    return lower, upper, f"[{lower},{upper}]"


def _histogram_rows(
    values_by_group: Mapping[tuple[str, str], Sequence[float]], *, kind: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for (metric, group), values in sorted(values_by_group.items()):
        counts: dict[tuple[Any, Any, str], int] = defaultdict(int)
        for value in values:
            key = _latency_bin(value) if kind == "latency" else _token_bin(value)
            counts[key] += 1
        total = len(values)
        cumulative = 0
        for (lower, upper, label), count in sorted(
            counts.items(), key=lambda item: float(item[0][0])
        ):
            cumulative += count
            result.append(
                {
                    "metric": metric,
                    "group": group,
                    "bin_lower_inclusive": lower,
                    "bin_upper_exclusive": upper if kind == "latency" else "",
                    "bin_upper_inclusive": upper if kind == "token" else "",
                    "bin_label": label,
                    "count": count,
                    "share": count / total if total else 0.0,
                    "cumulative_count": cumulative,
                    "cumulative_share": cumulative / total if total else 0.0,
                }
            )
    return result


def _event_gap_values(connection: Any) -> dict[tuple[str, str], list[float]]:
    output_types = ",".join(f"'{value}'" for value in MODEL_OUTPUT_TYPES)
    query = f"""
    WITH events AS (
        SELECT
            r.provider,
            COALESCE(r.project, '') AS project_key,
            r.session_id,
            t.round_pk,
            t.event_index,
            t.event_type,
            t.timestamp,
            MAX(t.timestamp) FILTER (WHERE t.event_type IN ({output_types})) OVER (
                PARTITION BY r.provider, COALESCE(r.project, ''), r.session_id
                ORDER BY t.timestamp, t.round_pk, t.event_index
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS previous_model_output_at
        FROM timing_events AS t
        JOIN rounds AS r ON r.round_pk = t.round_pk
        WHERE t.timestamp IS NOT NULL
    )
    SELECT
        provider,
        event_type,
        EPOCH(timestamp - previous_model_output_at) AS gap_seconds
    FROM events
    WHERE event_type IN ('user_message', 'tool_result')
      AND previous_model_output_at IS NOT NULL
      AND timestamp >= previous_model_output_at
    ORDER BY provider, session_id, timestamp, round_pk, event_index
    """
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for provider, event_type, gap_seconds in connection.execute(query).fetchall():
        metric = (
            "human_input_wait_after_last_output_seconds"
            if event_type == "user_message"
            else "tool_result_after_last_output_seconds"
        )
        _add_grouped(values, metric, str(provider or ""), gap_seconds)
    return values


def _tool_wall_values(connection: Any) -> dict[tuple[str, str], list[float]]:
    query = """
    SELECT r.provider, tc.tool_wall_latency_ms / 1000.0 AS seconds
    FROM tool_calls AS tc
    JOIN rounds AS r ON r.round_pk = tc.round_pk
    WHERE tc.tool_wall_latency_ms IS NOT NULL
      AND tc.tool_wall_latency_ms >= 0
    ORDER BY r.provider, tc.round_pk, tc.tool_index
    """
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for provider, seconds in connection.execute(query).fetchall():
        _add_grouped(values, "agent_tool_wall_latency_seconds", str(provider or ""), seconds)
    return values


def _request_token_values(
    connection: Any,
) -> tuple[dict[tuple[str, str], list[float]], list[dict[str, Any]]]:
    query = """
    WITH ordered AS (
        SELECT
            r.*,
            LAG(r.round_index) OVER session_window AS previous_round_index,
            LAG(r.input_tokens_total) OVER session_window AS previous_input_tokens_total,
            LAG(r.output_tokens) OVER session_window AS previous_output_tokens
        FROM rounds AS r
        WINDOW session_window AS (
            PARTITION BY r.provider, COALESCE(r.project, ''), r.session_id
            ORDER BY r.round_index, r.ingest_seq, r.round_pk
        )
    )
    SELECT
        provider,
        CASE
            WHEN current_user_message_count > 0 AND current_tool_result_count = 0 THEN 'user'
            WHEN current_user_message_count = 0 AND current_tool_result_count > 0 THEN 'agent'
            WHEN current_user_message_count > 0 AND current_tool_result_count > 0 THEN 'mixed'
            ELSE 'unknown'
        END AS trigger_type,
        CASE
            WHEN input_tokens_total < 0 THEN NULL
            WHEN previous_round_index = round_index - 1
             AND input_tokens_total - previous_input_tokens_total - previous_output_tokens > 0
            THEN input_tokens_total - previous_input_tokens_total - previous_output_tokens
            ELSE input_tokens_total
        END AS logical_isl_tokens,
        newly_append_tokens,
        output_tokens,
        input_tokens_total
    FROM ordered
    ORDER BY ingest_seq, round_pk
    """
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    trigger_counts: dict[tuple[str, str], int] = defaultdict(int)
    total = 0
    for provider, trigger, logical_isl, source_append, osl, seq_len in connection.execute(query).fetchall():
        provider = str(provider or "<unknown-provider>")
        trigger = str(trigger)
        total += 1
        trigger_counts[("all", trigger)] += 1
        trigger_counts[(provider, trigger)] += 1
        for metric, value in (
            ("logical_isl_tokens", logical_isl),
            ("source_newly_append_tokens", source_append),
            ("osl_tokens", osl),
            ("request_sequence_length_tokens", seq_len),
        ):
            if value is None or float(value) < 0:
                continue
            number = float(value)
            values[(metric, "all")].append(number)
            values[(metric, provider)].append(number)
            values[(metric, trigger)].append(number)
            values[(metric, f"{provider}:{trigger}")].append(number)
    trigger_rows: list[dict[str, Any]] = []
    totals_by_group: dict[str, int] = defaultdict(int)
    for (group, _trigger), count in trigger_counts.items():
        totals_by_group[group] += count
    for (group, trigger), count in sorted(trigger_counts.items()):
        denominator = totals_by_group[group]
        trigger_rows.append(
            {
                "group": group,
                "trigger_type": trigger,
                "request_count": count,
                "share": count / denominator if denominator else 0.0,
            }
        )
    assert totals_by_group["all"] == total
    return values, trigger_rows


def _tools_between_users(connection: Any) -> dict[tuple[str, str], list[float]]:
    query = """
    WITH event_stream AS (
        SELECT
            r.provider,
            COALESCE(r.project, '') AS project_key,
            r.session_id,
            t.round_pk,
            t.event_index,
            t.event_type,
            t.timestamp,
            SUM(CASE WHEN t.event_type = 'user_message' THEN 1 ELSE 0 END) OVER (
                PARTITION BY r.provider, COALESCE(r.project, ''), r.session_id
                ORDER BY t.timestamp, t.round_pk, t.event_index
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS user_ordinal
        FROM timing_events AS t
        JOIN rounds AS r ON r.round_pk = t.round_pk
        WHERE t.timestamp IS NOT NULL
    ), interval_counts AS (
        SELECT
            provider,
            project_key,
            session_id,
            user_ordinal,
            COUNT(*) FILTER (WHERE event_type = 'tool_call') AS tool_call_count,
            COUNT(DISTINCT round_pk) FILTER (WHERE event_type = 'tool_result')
                AS tool_result_request_count
        FROM event_stream
        WHERE user_ordinal > 0
        GROUP BY provider, project_key, session_id, user_ordinal
    ), closed AS (
        SELECT *, MAX(user_ordinal) OVER (
            PARTITION BY provider, project_key, session_id
        ) AS final_user_ordinal
        FROM interval_counts
    )
    SELECT provider, tool_call_count, tool_result_request_count
    FROM closed
    WHERE user_ordinal < final_user_ordinal
    ORDER BY provider, session_id, user_ordinal
    """
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for provider, tool_calls, tool_requests in connection.execute(query).fetchall():
        provider = str(provider or "")
        _add_grouped(values, "tool_calls_between_user_messages", provider, tool_calls)
        _add_grouped(
            values,
            "tool_triggered_requests_between_user_messages",
            provider,
            tool_requests,
        )
    return values


def _count_distribution_rows(
    values_by_group: Mapping[tuple[str, str], Sequence[float]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (metric, group), values in sorted(values_by_group.items()):
        counts: dict[int, int] = defaultdict(int)
        for value in values:
            counts[int(value)] += 1
        total = len(values)
        cumulative = 0
        for value, count in sorted(counts.items()):
            cumulative += count
            rows.append(
                {
                    "metric": metric,
                    "group": group,
                    "tool_count": value,
                    "interval_count": count,
                    "share": count / total if total else 0.0,
                    "cumulative_interval_count": cumulative,
                    "cumulative_share": cumulative / total if total else 0.0,
                }
            )
    return rows


def analyze(db_path: Path, output_dir: Path) -> list[Path]:
    if not db_path.exists():
        raise FileNotFoundError(f"TraceLab DuckDB does not exist: {db_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    duckdb = _load_duckdb()
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        event_gaps = _event_gap_values(connection)
        tool_wall = _tool_wall_values(connection)
        latency_values = defaultdict(list)
        for source in (event_gaps, tool_wall):
            for key, values in source.items():
                latency_values[key].extend(values)

        token_values, trigger_rows = _request_token_values(connection)
        tool_counts = _tools_between_users(connection)
    finally:
        connection.close()

    summary_rows: list[dict[str, Any]] = []
    for (metric, group), values in sorted(latency_values.items()):
        summary_rows.append(_summary(metric, group, "seconds", values))
    for (metric, group), values in sorted(tool_counts.items()):
        summary_rows.append(_summary(metric, group, "count", values))
    for (metric, group), values in sorted(token_values.items()):
        summary_rows.append(_summary(metric, group, "tokens", values))

    definitions = [
        {
            "metric": "human_input_wait_after_last_output_seconds",
            "definition": "Previous same-session reasoning/text/tool_call timestamp to each user_message timestamp; nonnegative observed pairs only.",
        },
        {
            "metric": "agent_tool_wall_latency_seconds",
            "definition": "Trace-observed tool wall time: tool_calls.tool_wall_latency_ms / 1000.",
        },
        {
            "metric": "tool_result_after_last_output_seconds",
            "definition": "Previous same-session reasoning/text/tool_call timestamp to each tool_result timestamp; nonnegative observed pairs only.",
        },
        {
            "metric": "tool_calls_between_user_messages",
            "definition": "Number of tool_call events in [user_message_i, user_message_(i+1)) within a source session; only closed user-message intervals.",
        },
        {
            "metric": "tool_triggered_requests_between_user_messages",
            "definition": "Distinct rounds carrying tool_result events in [user_message_i, user_message_(i+1)); only closed intervals.",
        },
        {
            "metric": "logical_isl_tokens",
            "definition": "Source-audit estimate: if the previous source round is adjacent and current_input_total - previous_input_total - previous_output > 0, use that delta; otherwise use full input_tokens_total. This does not apply the converter's compaction/truncation retention policy.",
        },
        {
            "metric": "source_newly_append_tokens",
            "definition": "TraceLab provider-side newly_append_tokens, exported as an audit companion to logical ISL.",
        },
        {"metric": "osl_tokens", "definition": "rounds.output_tokens."},
        {
            "metric": "request_sequence_length_tokens",
            "definition": "Prompt sequence length when the model request arrives: rounds.input_tokens_total.",
        },
        {
            "metric": "trigger:user",
            "definition": "current_user_message_count > 0 and current_tool_result_count = 0.",
        },
        {
            "metric": "trigger:agent",
            "definition": "current_user_message_count = 0 and current_tool_result_count > 0.",
        },
        {
            "metric": "trigger:mixed",
            "definition": "Both current_user_message_count and current_tool_result_count are positive.",
        },
        {
            "metric": "trigger:unknown",
            "definition": "Neither current_user_message_count nor current_tool_result_count is positive.",
        },
    ]

    latency_hist = _histogram_rows(latency_values, kind="latency")
    token_hist = _histogram_rows(token_values, kind="token")
    tool_count_rows = _count_distribution_rows(tool_counts)

    paths = {
        "definitions": output_dir / "analysis_definitions.csv",
        "summary": output_dir / "distribution_summary.csv",
        "latency": output_dir / "human_and_agent_interval_distribution.csv",
        "tool_counts": output_dir / "tools_between_user_messages_distribution.csv",
        "isl": output_dir / "isl_distribution.csv",
        "osl": output_dir / "osl_distribution.csv",
        "sequence": output_dir / "request_sequence_length_distribution.csv",
        "triggers": output_dir / "request_trigger_counts.csv",
    }
    _write_csv(paths["definitions"], ("metric", "definition"), definitions)
    _write_csv(
        paths["summary"],
        (
            "metric",
            "group",
            "unit",
            "count",
            "mean",
            "stddev_population",
            "min",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
            "max",
            "sum",
        ),
        summary_rows,
    )
    histogram_fields = (
        "metric",
        "group",
        "bin_lower_inclusive",
        "bin_upper_exclusive",
        "bin_upper_inclusive",
        "bin_label",
        "count",
        "share",
        "cumulative_count",
        "cumulative_share",
    )
    _write_csv(paths["latency"], histogram_fields, latency_hist)
    _write_csv(
        paths["tool_counts"],
        (
            "metric",
            "group",
            "tool_count",
            "interval_count",
            "share",
            "cumulative_interval_count",
            "cumulative_share",
        ),
        tool_count_rows,
    )
    _write_csv(
        paths["isl"],
        histogram_fields,
        [row for row in token_hist if row["metric"] in {"logical_isl_tokens", "source_newly_append_tokens"}],
    )
    _write_csv(
        paths["osl"],
        histogram_fields,
        [row for row in token_hist if row["metric"] == "osl_tokens"],
    )
    _write_csv(
        paths["sequence"],
        histogram_fields,
        [row for row in token_hist if row["metric"] == "request_sequence_length_tokens"],
    )
    _write_csv(
        paths["triggers"],
        ("group", "trigger_type", "request_count", "share"),
        trigger_rows,
    )
    return list(paths.values())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=_default_db())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = analyze(args.db.resolve(), args.output_dir.resolve())
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
