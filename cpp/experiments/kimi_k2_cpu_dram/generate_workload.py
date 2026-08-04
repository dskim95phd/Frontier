#!/usr/bin/env python3
"""Generate the deterministic Kimi K2 / GB300 multi-turn workload.

The C++ simulator consumes the small, normalized six-column workload CSV.  A
second, request-level manifest is emitted alongside it so that a simulator
run can be joined back to the fresh input, the previous session context, and
the two PREFILL lower bounds described in the capacity-study document.

This file intentionally uses only the Python standard library.  It therefore
works in the lightweight C++ build environment on Windows as well as in the
full Python simulator environment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import random
import statistics
from typing import Iterable, Mapping, Sequence


GENERATOR_VERSION = "kimi-k2-cpu-dram-v4-profile-capable"
SESSION_COUNT = 1_728
SIX_TURN_SESSIONS = 1_152
SEVEN_TURN_SESSIONS = 576
CONTEXT_MEAN_TOKENS = 65_536
CONTEXT_MIN_TOKENS = 49_152
CONTEXT_MAX_TOKENS = 98_304
CONTEXT_CV = 0.15
DIRICHLET_ALPHA = 20.0
ARRIVAL_RATE_PER_SECOND = 100.0
THINK_MEDIAN_SECONDS = 1_000.0
THINK_LOG_SIGMA = 0.001
THINK_MIN_SECONDS = 990.0
THINK_MAX_SECONDS = 1_010.0
BLOCK_SIZE = 16
INPUT_OUTPUT_RATIO = 8
ARRIVAL_PROCESSES = ("poisson", "stratified")
THINK_PROFILES: dict[str, tuple[tuple[float, float, float, float, float], ...]] = {
    # probability, log-normal median, sigma, minimum, maximum (seconds)
    "agentic_mixed": (
        (0.65, 0.2, 0.8, 0.01, 2.0),
        (0.25, 5.0, 1.0, 0.5, 60.0),
        (0.10, 120.0, 0.8, 10.0, 600.0),
    ),
    "balanced_mixed": (
        (0.45, 1.0, 0.8, 0.05, 10.0),
        (0.40, 30.0, 1.0, 2.0, 300.0),
        (0.15, 300.0, 0.9, 30.0, 1_800.0),
    ),
    "long_mixed": (
        (0.25, 2.0, 0.8, 0.1, 20.0),
        (0.45, 60.0, 0.9, 5.0, 600.0),
        (0.30, 600.0, 0.8, 60.0, 3_600.0),
    ),
    # A long-running agent/session sensitivity profile.  It deliberately
    # retains a small interactive tail while making parked sessions common
    # enough to put pressure on the GPU-resident session cache.
    "parked_mixed": (
        (0.10, 2.0, 0.9, 0.1, 30.0),
        (0.30, 300.0, 1.0, 30.0, 1_800.0),
        (0.60, 1_200.0, 0.8, 120.0, 7_200.0),
    ),
}
THINK_PROFILE_NAMES = ("single_lognormal", *THINK_PROFILES)
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


@dataclass(frozen=True)
class WorkloadRow:
    """One request in the normalized C++ workload and its manifest values."""

    request_id: int
    session_id: int
    session_turn_index: int
    num_turns: int
    new_input_tokens: int
    decode_tokens: int
    prior_context_tokens: int
    prior_decode_tokens: int
    theoretical_min_prefill_tokens: int
    implementation_min_prefill_tokens: int
    materialized_prefill_tokens: int
    actual_scheduled_prefill_tokens: int | None
    actual_extra_ratio: float | None
    session_start_at: float | None
    think_time: float
    final_context_tokens: int

    def workload_values(self) -> tuple[object, ...]:
        """Return the columns accepted by ``frontier_sim``."""

        return (
            "" if self.session_start_at is None else self.session_start_at,
            self.think_time,
            self.new_input_tokens,
            self.decode_tokens,
            self.session_id,
            self.session_turn_index,
        )

    def manifest_values(self) -> tuple[object, ...]:
        return tuple(getattr(self, field) for field in MANIFEST_FIELDS)


def _round_block(value: float, block_size: int = BLOCK_SIZE) -> int:
    if not math.isfinite(value):
        raise ValueError("token value must be finite")
    return max(block_size, int(round(value / block_size)) * block_size)


def _allocate_integer(total: int, weights: Sequence[float], *, minimum: int = 1) -> list[int]:
    """Allocate ``total`` in proportion to weights using largest remainder."""

    if total < minimum * len(weights):
        raise ValueError("total is too small for the requested allocation")
    if not weights or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
        raise ValueError("allocation weights must be finite and positive")
    weight_total = math.fsum(weights)
    ideal = [total * weight / weight_total for weight in weights]
    result = [max(minimum, math.floor(value)) for value in ideal]
    # If enforcing a positive minimum consumed too many tokens, start from a
    # minimum allocation and distribute the residual by the same weights.
    used = sum(result)
    if used > total:
        result = [minimum] * len(weights)
        used = sum(result)
    residual = total - used
    fractions = sorted(
        range(len(weights)),
        key=lambda index: (ideal[index] - math.floor(ideal[index]), -index),
        reverse=True,
    )
    for index in range(residual):
        result[fractions[index % len(fractions)]] += 1
    return result


def _sample_final_contexts(rng: random.Random, count: int) -> list[int]:
    """Sample bounded log-normal contexts and rescale to the target mean.

    Contexts are represented as ``9 * output_tokens``.  Making output tokens
    multiples of 16 keeps the resulting context and every prefix-cache block
    aligned while preserving the requested bounded log-normal shape.
    """

    sigma = math.sqrt(math.log1p(CONTEXT_CV * CONTEXT_CV))
    raw = [rng.lognormvariate(math.log(CONTEXT_MEAN_TOKENS), sigma) for _ in range(count)]
    scale = CONTEXT_MEAN_TOKENS / statistics.fmean(raw)
    contexts = [
        min(CONTEXT_MAX_TOKENS, max(CONTEXT_MIN_TOKENS, _round_block(sample * scale)))
        for sample in raw
    ]
    # Context = 9 * output.  Round the output, rather than the context, so the
    # aggregate input:output contract is exact for every generated workload.
    outputs = [
        max(1, int(round(context / 9.0 / BLOCK_SIZE)) * BLOCK_SIZE)
        for context in contexts
    ]
    min_output = math.ceil(CONTEXT_MIN_TOKENS / 9 / BLOCK_SIZE) * BLOCK_SIZE
    max_output = math.floor(CONTEXT_MAX_TOKENS / 9 / BLOCK_SIZE) * BLOCK_SIZE
    outputs = [min(max_output, max(min_output, output)) for output in outputs]
    target_sum = int(round(count * CONTEXT_MEAN_TOKENS / 9 / BLOCK_SIZE)) * BLOCK_SIZE
    # A deterministic residual correction makes the mean stable enough for a
    # gate while retaining the random distribution and the stated bounds.
    residual = target_sum - sum(outputs)
    order = sorted(range(count), key=lambda i: (outputs[i], i))
    direction = 1 if residual >= 0 else -1
    while residual:
        changed = False
        for index in (order if direction > 0 else list(reversed(order))):
            candidate = outputs[index] + direction * BLOCK_SIZE
            if min_output <= candidate <= max_output:
                outputs[index] = candidate
                residual -= direction * BLOCK_SIZE
                changed = True
                if not residual:
                    break
        if not changed:
            # This should not happen for the documented range, but avoid an
            # infinite loop if a caller supplies a very small custom range.
            break
    return [output * 9 for output in outputs]


def _sample_turn_weights(
    rng: random.Random, turns: int, alpha: float = DIRICHLET_ALPHA
) -> list[float]:
    # gammavariate(alpha, 1) is a standard-library Dirichlet construction.
    values = [rng.gammavariate(alpha, 1.0) for _ in range(turns)]
    total = math.fsum(values)
    return [value / total for value in values]


def _think_time(
    rng: random.Random,
    *,
    median_seconds: float = THINK_MEDIAN_SECONDS,
    log_sigma: float = THINK_LOG_SIGMA,
    minimum_seconds: float = THINK_MIN_SECONDS,
    maximum_seconds: float = THINK_MAX_SECONDS,
) -> float:
    value = rng.lognormvariate(math.log(median_seconds), log_sigma)
    return min(maximum_seconds, max(minimum_seconds, value))


def _profiled_think_time(rng: random.Random, profile: str) -> float:
    classes = THINK_PROFILES.get(profile)
    if classes is None:
        raise ValueError(f"unknown mixed think-time profile: {profile}")
    draw = rng.random()
    cumulative = 0.0
    selected = classes[-1]
    for gap_class in classes:
        cumulative += gap_class[0]
        if draw < cumulative:
            selected = gap_class
            break
    _, median, sigma, minimum, maximum = selected
    return _think_time(
        rng,
        median_seconds=median,
        log_sigma=sigma,
        minimum_seconds=minimum,
        maximum_seconds=maximum,
    )


def generate_rows(
    *,
    seed: int = 20260803,
    session_count: int = SESSION_COUNT,
    six_turn_sessions: int = SIX_TURN_SESSIONS,
    context_mean_tokens: int = CONTEXT_MEAN_TOKENS,
    context_min_tokens: int = CONTEXT_MIN_TOKENS,
    context_max_tokens: int = CONTEXT_MAX_TOKENS,
    context_cv: float = CONTEXT_CV,
    dirichlet_alpha: float = DIRICHLET_ALPHA,
    arrival_rate: float = ARRIVAL_RATE_PER_SECOND,
    arrival_process: str = "poisson",
    think_profile: str = "single_lognormal",
    think_median_seconds: float = THINK_MEDIAN_SECONDS,
    think_log_sigma: float = THINK_LOG_SIGMA,
    think_min_seconds: float = THINK_MIN_SECONDS,
    think_max_seconds: float = THINK_MAX_SECONDS,
) -> list[WorkloadRow]:
    """Generate deterministic session rows for one seed.

    The keyword parameters are exposed for calibration tests, but the default
    values are the frozen study workload contract.  The normal CLI uses the
    defaults and records every override in its JSON manifest.
    """

    if session_count <= 0 or six_turn_sessions < 0 or six_turn_sessions > session_count:
        raise ValueError("invalid session and six-turn counts")
    seven_turn_sessions = session_count - six_turn_sessions
    if context_mean_tokens <= 0 or context_min_tokens <= 0 or context_max_tokens < context_min_tokens:
        raise ValueError("invalid context bounds")
    if context_cv < 0.0 or dirichlet_alpha <= 0.0 or arrival_rate <= 0.0:
        raise ValueError("distribution parameters must be positive")
    if arrival_process not in ARRIVAL_PROCESSES:
        raise ValueError(f"unknown arrival process: {arrival_process}")
    if think_profile not in THINK_PROFILE_NAMES:
        raise ValueError(f"unknown think-time profile: {think_profile}")
    if think_median_seconds <= 0.0 or think_log_sigma < 0.0:
        raise ValueError("think-time parameters are invalid")
    if think_min_seconds < 0.0 or think_max_seconds < think_min_seconds:
        raise ValueError("think-time bounds are invalid")

    # Use a local RNG and local constants so that importing this module never
    # perturbs any simulator or test process global random stream.
    rng = random.Random(seed)
    contexts = _sample_final_contexts(rng, session_count)
    # The helper above intentionally uses the frozen constants.  Apply custom
    # context parameters here only for explicit calibration calls.
    if (
        context_mean_tokens != CONTEXT_MEAN_TOKENS
        or context_min_tokens != CONTEXT_MIN_TOKENS
        or context_max_tokens != CONTEXT_MAX_TOKENS
        or context_cv != CONTEXT_CV
    ):
        sigma = math.sqrt(math.log1p(context_cv * context_cv))
        raw = [rng.lognormvariate(math.log(context_mean_tokens), sigma) for _ in range(session_count)]
        scale = context_mean_tokens / statistics.fmean(raw)
        contexts = [
            min(context_max_tokens, max(context_min_tokens, _round_block(sample * scale)))
            for sample in raw
        ]
        outputs = [max(BLOCK_SIZE, int(round(context / 9 / BLOCK_SIZE)) * BLOCK_SIZE) for context in contexts]
        contexts = [output * 9 for output in outputs]

    # Draw all first-turn arrivals before constructing rows.  This is a
    # Poisson process at one new-session/s by default, independent of turn
    # length and think-time draws.
    arrivals: list[float] = []
    if arrival_process == "poisson":
        arrival = 0.0
        for session_id in range(session_count):
            # Follow the simulator workload convention: the first Poisson
            # event is anchored at t=0, and subsequent starts use exponential
            # gaps.
            if session_id > 0:
                arrival += rng.expovariate(arrival_rate)
            arrivals.append(arrival)
    else:
        # Put exactly one session in each 1/rate-second stratum.  This keeps
        # the requested average offered load while avoiding an artificial
        # all-at-once burst; the within-stratum offset remains stochastic.
        arrivals = [
            (session_id + rng.random()) / arrival_rate
            for session_id in range(session_count)
        ]
        anchor = arrivals[0]
        arrivals = [arrival - anchor for arrival in arrivals]

    rows: list[WorkloadRow] = []
    request_id = 0
    for session_id in range(session_count):
        turns = 6 if session_id < six_turn_sessions else 7
        output_total = contexts[session_id] // 9
        input_total = output_total * INPUT_OUTPUT_RATIO
        weights = _sample_turn_weights(rng, turns, dirichlet_alpha)
        output_tokens = _allocate_integer(output_total, weights)
        input_tokens = _allocate_integer(input_total, weights)
        prior_context = 0
        prior_decode = 0
        for turn_index, (new_input, decode) in enumerate(zip(input_tokens, output_tokens)):
            start_at = arrivals[session_id] if turn_index == 0 else None
            if turn_index == 0:
                think_time = 0.0
            elif think_profile == "single_lognormal":
                think_time = _think_time(
                    rng,
                    median_seconds=think_median_seconds,
                    log_sigma=think_log_sigma,
                    minimum_seconds=think_min_seconds,
                    maximum_seconds=think_max_seconds,
                )
            else:
                think_time = _profiled_think_time(rng, think_profile)
            theoretical_min = new_input
            implementation_min = new_input + prior_decode + (BLOCK_SIZE if turn_index > 0 else 0)
            materialized = new_input + prior_context
            extra_ratio = (materialized - new_input) / new_input
            rows.append(
                WorkloadRow(
                    request_id=request_id,
                    session_id=session_id,
                    session_turn_index=turn_index,
                    num_turns=turns,
                    new_input_tokens=new_input,
                    decode_tokens=decode,
                    prior_context_tokens=prior_context,
                    prior_decode_tokens=prior_decode,
                    theoretical_min_prefill_tokens=theoretical_min,
                    implementation_min_prefill_tokens=implementation_min,
                    materialized_prefill_tokens=materialized,
                    actual_scheduled_prefill_tokens=materialized,
                    actual_extra_ratio=extra_ratio,
                    session_start_at=start_at,
                    think_time=think_time,
                    final_context_tokens=contexts[session_id],
                )
            )
            request_id += 1
            prior_context += new_input + decode
            prior_decode = decode
    return rows


def _write_workload(path: Path, rows: Sequence[WorkloadRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "session_start_at",
                "think_time",
                "num_prefill_tokens",
                "num_decode_tokens",
                "session_id",
                "session_turn_index",
            )
        )
        for row in rows:
            values = row.workload_values()
            writer.writerow(
                (
                    "" if values[0] == "" else f"{float(values[0]):.12f}",
                    f"{float(values[1]):.12f}",
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                )
            )


def write_manifest_csv(path: Path, rows: Sequence[WorkloadRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(MANIFEST_FIELDS)
        for row in rows:
            writer.writerow(row.manifest_values())


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _int_or_none(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def read_manifest(path: Path) -> dict[int, dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in MANIFEST_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest is missing fields: {', '.join(missing)}")
        result: dict[int, dict[str, object]] = {}
        for raw in reader:
            request_id = int(raw["request_id"])
            row: dict[str, object] = dict(raw)
            for key in (
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
                "final_context_tokens",
            ):
                row[key] = int(float(raw[key]))
            row["actual_scheduled_prefill_tokens"] = _int_or_none(raw.get("actual_scheduled_prefill_tokens"))
            row["actual_extra_ratio"] = _float_or_none(raw.get("actual_extra_ratio"))
            row["session_start_at"] = _float_or_none(raw.get("session_start_at"))
            row["think_time"] = float(raw["think_time"])
            result[request_id] = row
    return result


def _scheduled_from_output(
    output_row: Mapping[str, object], manifest_row: Mapping[str, object]
) -> tuple[int, str]:
    """Extract actual scheduled PREFILL from current or older C++ outputs."""

    for key in (
        "actual_scheduled_prefill_tokens",
        "scheduled_prefill_tokens",
        "prefill_scheduled_tokens",
    ):
        if key in output_row and output_row[key] not in (None, ""):
            return int(float(output_row[key])), key
    # Current C++ requests.csv carries the fresh prompt and cache hits, while
    # the compact scheduled-PREFILL field is optional.  Reconstruct the
    # materialized prompt from the manifest and subtract any prefix hit.
    effective = int(manifest_row["materialized_prefill_tokens"])
    cached = int(float(output_row.get("cached_prefill_tokens", 0) or 0))
    scheduled = max(0, effective - cached)
    # A future compact metric may report recomputed work separately.
    recomputed = output_row.get("preemption_recomputed_prefill_tokens")
    if recomputed not in (None, "") and "scheduled_prefill_tokens" not in output_row:
        scheduled += int(float(recomputed))
    return scheduled, "manifest_materialized_minus_cached"


def calibrate_from_requests(
    manifest_path: Path,
    requests_path: Path,
    *,
    output_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Join one simulator ``requests.csv`` with a workload manifest.

    The returned ``actual_extra_ratio`` is exactly the study definition::

        (sum(actual_scheduled_prefill_tokens) - sum(new_input_tokens)) /
        sum(new_input_tokens)

    If an older simulator does not emit a compact scheduled-PREFILL column,
    the materialized prompt from the manifest minus ``cached_prefill_tokens``
    is used and the result records that fallback in ``measurement_sources``.
    """

    manifest = read_manifest(manifest_path)
    with requests_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "request_id" not in reader.fieldnames:
            raise ValueError("requests output must contain request_id")
        output_rows = list(reader)

    updated: dict[int, dict[str, object]] = {}
    sources: dict[str, int] = {}
    scheduled_total = 0
    new_input_total = 0
    for raw in output_rows:
        request_id = int(float(raw["request_id"]))
        if request_id not in manifest:
            raise ValueError(f"requests output references unknown request_id={request_id}")
        row = dict(manifest[request_id])
        scheduled, source = _scheduled_from_output(raw, row)
        row["actual_scheduled_prefill_tokens"] = scheduled
        new_input = int(row["new_input_tokens"])
        row["actual_extra_ratio"] = (scheduled - new_input) / new_input
        scheduled_total += scheduled
        new_input_total += new_input
        sources[source] = sources.get(source, 0) + 1
        updated[request_id] = row
    if len(updated) != len(manifest):
        missing = sorted(set(manifest) - set(updated))
        raise ValueError(f"requests output is missing {len(missing)} manifest rows")
    ratio = (scheduled_total - new_input_total) / new_input_total if new_input_total else 0.0
    gate = 2.80 <= ratio <= 3.20
    result = {
        "generator_version": GENERATOR_VERSION,
        "manifest": str(manifest_path),
        "requests_output": str(requests_path),
        "request_count": len(updated),
        "new_input_tokens": new_input_total,
        "actual_scheduled_prefill_tokens": scheduled_total,
        "actual_extra_ratio": ratio,
        "scheduled_over_theoretical_minimum": scheduled_total / new_input_total if new_input_total else 0.0,
        "gate_2_80_to_3_20": gate,
        "measurement_sources": sources,
    }
    if output_manifest_path is not None:
        output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with output_manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for request_id in sorted(updated):
                row = updated[request_id]
                writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
        result["updated_manifest"] = str(output_manifest_path)
    return result


def workload_statistics(rows: Sequence[WorkloadRow]) -> dict[str, object]:
    sessions = {row.session_id for row in rows}
    final_contexts = [row.final_context_tokens for row in rows if row.session_turn_index == row.num_turns - 1]
    input_total = sum(row.new_input_tokens for row in rows)
    output_total = sum(row.decode_tokens for row in rows)
    materialized_total = sum(row.materialized_prefill_tokens for row in rows)
    implementation_total = sum(row.implementation_min_prefill_tokens for row in rows)
    ratio = (materialized_total - input_total) / input_total if input_total else 0.0
    return {
        "request_count": len(rows),
        "session_count": len(sessions),
        "six_turn_sessions": sum(row.num_turns == 6 for row in rows if row.session_turn_index == 0),
        "seven_turn_sessions": sum(row.num_turns == 7 for row in rows if row.session_turn_index == 0),
        "final_context_mean_tokens": statistics.fmean(final_contexts) if final_contexts else 0.0,
        "final_context_min_tokens": min(final_contexts) if final_contexts else 0,
        "final_context_max_tokens": max(final_contexts) if final_contexts else 0,
        "input_tokens": input_total,
        "output_tokens": output_total,
        "input_output_ratio": input_total / output_total if output_total else 0.0,
        "materialized_prefill_tokens": materialized_total,
        "theoretical_extra_ratio": ratio,
        "implementation_min_prefill_tokens": implementation_total,
        "implementation_min_extra_ratio": (implementation_total - input_total) / input_total if input_total else 0.0,
    }


def write_generation(
    output_csv: Path,
    manifest_csv: Path,
    metadata_path: Path,
    rows: Sequence[WorkloadRow],
    *,
    seed: int,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    _write_workload(output_csv, rows)
    write_manifest_csv(manifest_csv, rows)
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "parameters": dict(parameters),
        "statistics": workload_statistics(rows),
        "manifest_csv": str(manifest_csv),
        "workload_csv": str(output_csv),
        "theoretical_minimum_definition": "fresh new_input_tokens only",
        "implementation_minimum_definition": "new input + immediately preceding decode + one 16-token block on successor turns",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "workloads")
    parser.add_argument("--output", type=Path, help="Explicit workload CSV path (overrides --output-dir naming).")
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--sessions", type=int, default=SESSION_COUNT)
    parser.add_argument("--six-turn-sessions", type=int, default=SIX_TURN_SESSIONS)
    parser.add_argument("--context-mean", type=int, default=CONTEXT_MEAN_TOKENS)
    parser.add_argument("--context-min", type=int, default=CONTEXT_MIN_TOKENS)
    parser.add_argument("--context-max", type=int, default=CONTEXT_MAX_TOKENS)
    parser.add_argument("--context-cv", type=float, default=CONTEXT_CV)
    parser.add_argument("--dirichlet-alpha", type=float, default=DIRICHLET_ALPHA)
    parser.add_argument("--arrival-rate", type=float, default=ARRIVAL_RATE_PER_SECOND)
    parser.add_argument("--arrival-process", choices=ARRIVAL_PROCESSES, default="poisson")
    parser.add_argument("--think-profile", choices=THINK_PROFILE_NAMES, default="single_lognormal")
    parser.add_argument("--think-median", type=float, default=THINK_MEDIAN_SECONDS)
    parser.add_argument("--think-log-sigma", type=float, default=THINK_LOG_SIGMA)
    parser.add_argument("--think-min", type=float, default=THINK_MIN_SECONDS)
    parser.add_argument("--think-max", type=float, default=THINK_MAX_SECONDS)
    parser.add_argument("--calibrate-manifest", type=Path, help="Manifest CSV to calibrate from a requests output.")
    parser.add_argument("--calibrate-requests", type=Path, help="frontier_sim requests.csv for --calibrate-manifest.")
    parser.add_argument("--calibrate-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if (args.calibrate_manifest is None) != (args.calibrate_requests is None):
        raise SystemExit("--calibrate-manifest and --calibrate-requests must be supplied together")
    if args.calibrate_manifest is not None:
        result = calibrate_from_requests(
            args.calibrate_manifest,
            args.calibrate_requests,
            output_manifest_path=args.calibrate_output,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["gate_2_80_to_3_20"] else 2

    output_csv = args.output or (args.output_dir / f"seed_{args.seed}.csv")
    manifest_csv = args.manifest_output or output_csv.with_name(output_csv.stem + "_manifest.csv")
    metadata_path = args.metadata_output or output_csv.with_name(output_csv.stem + "_metadata.json")
    rows = generate_rows(
        seed=args.seed,
        session_count=args.sessions,
        six_turn_sessions=args.six_turn_sessions,
        context_mean_tokens=args.context_mean,
        context_min_tokens=args.context_min,
        context_max_tokens=args.context_max,
        context_cv=args.context_cv,
        dirichlet_alpha=args.dirichlet_alpha,
        arrival_rate=args.arrival_rate,
        arrival_process=args.arrival_process,
        think_profile=args.think_profile,
        think_median_seconds=args.think_median,
        think_log_sigma=args.think_log_sigma,
        think_min_seconds=args.think_min,
        think_max_seconds=args.think_max,
    )
    parameters = {
        "sessions": args.sessions,
        "six_turn_sessions": args.six_turn_sessions,
        "seven_turn_sessions": args.sessions - args.six_turn_sessions,
        "context_mean": args.context_mean,
        "context_min": args.context_min,
        "context_max": args.context_max,
        "context_cv": args.context_cv,
        "dirichlet_alpha": args.dirichlet_alpha,
        "arrival_rate": args.arrival_rate,
        "arrival_process": args.arrival_process,
        "think_profile": args.think_profile,
        "think_median": args.think_median,
        "think_log_sigma": args.think_log_sigma,
        "think_min": args.think_min,
        "think_max": args.think_max,
        "block_size": BLOCK_SIZE,
        "input_output_ratio": INPUT_OUTPUT_RATIO,
    }
    metadata = write_generation(output_csv, manifest_csv, metadata_path, rows, seed=args.seed, parameters=parameters)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
