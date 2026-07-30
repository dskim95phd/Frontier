"""Independent Python oracle for the normalized workload contract."""

from __future__ import annotations

import math
import re
from typing import Any


class OracleError(ValueError):
    """Raised when normalized workload input violates the contract."""


_REQUIRED_WORKLOAD_COLUMNS = {
    "arrived_at",
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
    for raw_line in lines[1:]:
        if not raw_line.strip(" \t\r"):
            continue
        fields = [field.strip(" \t\r") for field in raw_line.split(",")]
        if len(fields) != len(header):
            raise OracleError(
                "workload CSV field count does not match header"
            )

        arrived_text = fields[indices["arrived_at"]]
        try:
            arrived_at = float(arrived_text)
        except ValueError as error:
            raise OracleError("arrived_at must be numeric") from error
        if not math.isfinite(arrived_at) or arrived_at < 0.0:
            raise OracleError(
                "arrived_at must be finite and nonnegative"
            )

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

        requests.append(
            {
                "request_id": len(requests),
                "arrived_at": arrived_at,
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
