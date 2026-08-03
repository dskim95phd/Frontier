"""Independent Python oracle for the normalized workload contract."""

from __future__ import annotations

import math
import re
from typing import Any


class OracleError(ValueError):
    """Raised when normalized workload input violates the contract."""


_REQUIRED_WORKLOAD_COLUMNS = {
    "session_start_at",
    "think_time",
    "num_prefill_tokens",
    "num_decode_tokens",
}
_OPTIONAL_WORKLOAD_COLUMNS = {"session_id", "session_turn_index"}
_NONNEGATIVE_INTEGER = re.compile(r"[0-9]+\Z")


def _parse_nonnegative_integer(
    text: str,
    field: str,
    *,
    positive: bool = False,
) -> int:
    if not _NONNEGATIVE_INTEGER.fullmatch(text):
        raise OracleError(f"{field} must be an integer")
    value = int(text)
    if positive and value == 0:
        raise OracleError(f"{field} must be positive")
    return value


def parse_normalized_workload(text: str) -> list[dict[str, Any]]:
    if '"' in text:
        raise OracleError("CSV quoting is unsupported")
    lines = text.splitlines()
    if not lines:
        raise OracleError("workload CSV is empty")

    lines[0] = lines[0].removeprefix("\ufeff")
    header = [field.strip(" \t\r") for field in lines[0].split(",")]
    if any(not field for field in header):
        raise OracleError("workload CSV contains an empty header column")
    if len(header) != len(set(header)):
        raise OracleError("workload CSV contains a duplicate column")
    if "block_hash_ids" in header:
        raise OracleError("block_hash_ids is outside the C++ port")

    columns = set(header)
    missing = _REQUIRED_WORKLOAD_COLUMNS - columns
    unknown = columns - (
        _REQUIRED_WORKLOAD_COLUMNS | _OPTIONAL_WORKLOAD_COLUMNS
    )
    if missing:
        raise OracleError(
            f"missing required column {sorted(missing)[0]!r}"
        )
    if unknown:
        raise OracleError(f"unknown column {sorted(unknown)[0]!r}")
    indices = {name: index for index, name in enumerate(header)}

    requests: list[dict[str, Any]] = []
    seen_sessions: set[int] = set()
    for raw_line in lines[1:]:
        if not raw_line.strip(" \t\r"):
            continue
        fields = [field.strip(" \t\r") for field in raw_line.split(",")]
        if len(fields) != len(header):
            raise OracleError(
                "workload CSV field count does not match header"
            )

        start_text = fields[indices["session_start_at"]]
        session_start_at = None
        if start_text:
            try:
                session_start_at = float(start_text)
            except ValueError as error:
                raise OracleError("session_start_at must be numeric") from error
            if not math.isfinite(session_start_at) or session_start_at < 0.0:
                raise OracleError(
                    "session_start_at must be finite and nonnegative"
                )
        try:
            think_time = float(fields[indices["think_time"]])
        except ValueError as error:
            raise OracleError("think_time must be numeric") from error
        if not math.isfinite(think_time) or think_time < 0.0:
            raise OracleError("think_time must be finite and nonnegative")

        def optional_integer(field: str) -> int | None:
            if field not in indices:
                return None
            value = fields[indices[field]]
            if not value:
                return None
            return _parse_nonnegative_integer(value, field)

        session_id = optional_integer("session_id")
        session_turn_index = optional_integer("session_turn_index")
        if session_turn_index is not None and session_id is None:
            raise OracleError(
                "session_turn_index requires session_id"
            )

        first_turn = session_id is None or session_id not in seen_sessions
        if session_id is not None:
            seen_sessions.add(session_id)
        if first_turn and session_start_at is None:
            raise OracleError("a first turn requires session_start_at")
        if first_turn and think_time != 0.0:
            raise OracleError("a first turn must have think_time=0")
        if not first_turn and session_start_at is not None:
            raise OracleError("a successor turn must omit session_start_at")

        requests.append(
            {
                "request_id": len(requests),
                "session_start_at": session_start_at,
                "think_time": think_time,
                "num_prefill_tokens": _parse_nonnegative_integer(
                    fields[indices["num_prefill_tokens"]],
                    "num_prefill_tokens",
                    positive=True,
                ),
                "num_decode_tokens": _parse_nonnegative_integer(
                    fields[indices["num_decode_tokens"]],
                    "num_decode_tokens",
                    positive=True,
                ),
                "session_id": session_id,
                "session_turn_index": session_turn_index,
            }
        )
    return requests
