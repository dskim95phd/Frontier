"""Independent Python oracle for the C++ Step 1 normalized contracts."""

from __future__ import annotations

import heapq
import json
import math
import re
from dataclasses import dataclass
from typing import Any


class OracleError(ValueError):
    """Raised when normalized Step 1 input violates the Python contract."""


_CONFIG_KEYS = {
    "schema_version",
    "run_id",
    "simulation_mode",
    "system_architecture",
    "enable_parallel_clusters",
    "prefix_cache",
}
_PREFIX_KEYS = {"enabled", "key_mode"}
_REQUIRED_WORKLOAD_COLUMNS = {
    "arrived_at",
    "num_prefill_tokens",
    "num_decode_tokens",
}
_OPTIONAL_WORKLOAD_COLUMNS = {"session_id", "session_turn_index"}
_NONNEGATIVE_INTEGER = re.compile(r"[0-9]+\Z")


def _require_exact_keys(
    value: Any,
    required: set[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OracleError(f"{context} must be a JSON object")
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required)
    if missing:
        raise OracleError(f"{context} is missing required field {missing[0]!r}")
    if unknown:
        raise OracleError(f"{context} contains unknown field {unknown[0]!r}")
    return value


def parse_normalized_config(text: str) -> dict[str, Any]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise OracleError(f"invalid config JSON: {error}") from error
    config = _require_exact_keys(raw, _CONFIG_KEYS, "config")

    if type(config["schema_version"]) is not int:
        raise OracleError("config.schema_version must be an integer")
    if config["schema_version"] != 1:
        raise OracleError("unsupported config schema_version")
    if not isinstance(config["run_id"], str) or not config["run_id"].strip():
        raise OracleError("config.run_id must not be empty")
    if config["simulation_mode"] not in {"offline", "online"}:
        raise OracleError("unsupported simulation_mode")
    if config["system_architecture"] not in {
        "co-location",
        "pd-disaggregation",
    }:
        raise OracleError("unsupported system_architecture")
    if type(config["enable_parallel_clusters"]) is not bool:
        raise OracleError("config.enable_parallel_clusters must be a boolean")
    if config["enable_parallel_clusters"]:
        raise OracleError("parallel clusters are outside the C++ MVP")

    prefix = _require_exact_keys(
        config["prefix_cache"],
        _PREFIX_KEYS,
        "config.prefix_cache",
    )
    if type(prefix["enabled"]) is not bool:
        raise OracleError("config.prefix_cache.enabled must be a boolean")
    if prefix["key_mode"] == "block_hash":
        raise OracleError("block_hash is outside the C++ MVP")
    if prefix["key_mode"] != "session":
        raise OracleError("unsupported prefix-cache key mode")

    return {
        "schema_version": 1,
        "run_id": config["run_id"],
        "simulation_mode": config["simulation_mode"],
        "system_architecture": config["system_architecture"],
        "enable_parallel_clusters": False,
        "prefix_cache": {
            "enabled": prefix["enabled"],
            "key_mode": "session",
        },
    }


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
        raise OracleError("block_hash_ids is outside the C++ MVP")

    columns = set(header)
    missing = _REQUIRED_WORKLOAD_COLUMNS - columns
    unknown = columns - (
        _REQUIRED_WORKLOAD_COLUMNS | _OPTIONAL_WORKLOAD_COLUMNS
    )
    if missing:
        raise OracleError(f"missing required column {sorted(missing)[0]!r}")
    if unknown:
        raise OracleError(f"unknown column {sorted(unknown)[0]!r}")
    indices = {name: index for index, name in enumerate(header)}

    requests: list[dict[str, Any]] = []
    for raw_line in lines[1:]:
        if not raw_line.strip(" \t\r"):
            continue
        fields = [field.strip(" \t\r") for field in raw_line.split(",")]
        if len(fields) != len(header):
            raise OracleError("workload CSV field count does not match header")

        arrived_text = fields[indices["arrived_at"]]
        try:
            arrived_at = float(arrived_text)
        except ValueError as error:
            raise OracleError("arrived_at must be numeric") from error
        if not math.isfinite(arrived_at) or arrived_at < 0.0:
            raise OracleError("arrived_at must be finite and nonnegative")

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
            raise OracleError("session_turn_index requires session_id")

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


def validate_workload_for_config(
    requests: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    if not config["prefix_cache"]["enabled"]:
        return

    last_arrival: dict[int, float] = {}
    last_turn: dict[int, int] = {}
    for request in requests:
        session_id = request["session_id"]
        if session_id is None:
            raise OracleError(
                "session_id is required when prefix caching is enabled"
            )
        if (
            session_id in last_arrival
            and request["arrived_at"] < last_arrival[session_id]
        ):
            raise OracleError("session arrivals must be nondecreasing")
        last_arrival[session_id] = request["arrived_at"]

        turn = request["session_turn_index"]
        if turn is not None:
            if session_id in last_turn and turn <= last_turn[session_id]:
                raise OracleError("session turns must be strictly increasing")
            last_turn[session_id] = turn


@dataclass(order=True, frozen=True)
class _QueuedEvent:
    time_s: float
    sequence: int
    event_type: str
    request_id: int


def run_foundation_oracle(
    config: dict[str, Any],
    workload: list[dict[str, Any]],
    *,
    service_time_ms: float = 1.0,
) -> dict[str, Any]:
    if config["system_architecture"] != "co-location":
        raise OracleError("foundation lifecycle supports only co-location")
    if config["prefix_cache"]["enabled"]:
        raise OracleError("foundation lifecycle does not implement prefix caching")
    if not math.isfinite(service_time_ms) or service_time_ms < 0.0:
        raise OracleError("service_time_ms must be finite and nonnegative")
    validate_workload_for_config(workload, config)

    queue: list[_QueuedEvent] = []
    next_sequence = 1
    for expected_id, request in enumerate(workload):
        if request["request_id"] != expected_id:
            raise OracleError("request IDs must be contiguous")
        heapq.heappush(
            queue,
            _QueuedEvent(
                request["arrived_at"],
                next_sequence,
                "request_arrival",
                request["request_id"],
            ),
        )
        next_sequence += 1

    states = ["pending"] * len(workload)
    event_trace: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    while queue:
        event = heapq.heappop(queue)
        event_trace.append(
            {
                "time_s": event.time_s,
                "sequence": event.sequence,
                "type": event.event_type,
                "request_id": event.request_id,
            }
        )
        request = workload[event.request_id]
        if event.event_type == "request_arrival":
            if states[event.request_id] != "pending":
                raise OracleError("request arrived more than once")
            states[event.request_id] = "arrived"
            completion_time = event.time_s + service_time_ms / 1e3
            if not math.isfinite(completion_time):
                raise OracleError("foundation completion time is nonfinite")
            heapq.heappush(
                queue,
                _QueuedEvent(
                    completion_time,
                    next_sequence,
                    "foundation_completion",
                    event.request_id,
                ),
            )
            next_sequence += 1
        else:
            if states[event.request_id] != "arrived":
                raise OracleError("request completed without one arrival")
            states[event.request_id] = "completed"
            latency_ms = (
                event.time_s - request["arrived_at"]
            ) * 1e3
            completed.append(
                {
                    "request_id": event.request_id,
                    "arrived_at_s": request["arrived_at"],
                    "prefill_completed_at_s": event.time_s,
                    "completed_at_s": event.time_s,
                    "ttft_ms": latency_ms,
                    "e2e_ms": latency_ms,
                }
            )

    return {
        "schema_version": 1,
        "run": {
            "run_id": config["run_id"],
            "simulation_mode": config["simulation_mode"],
            "system_architecture": config["system_architecture"],
            "timestamp_unit": "seconds",
            "latency_unit": "milliseconds",
            "metrics_semantics": "foundation-placeholder",
        },
        "completed_request_ids": [
            request["request_id"] for request in completed
        ],
        "requests": completed,
        "event_trace": event_trace,
        "analytical_diagnostics": [],
    }
