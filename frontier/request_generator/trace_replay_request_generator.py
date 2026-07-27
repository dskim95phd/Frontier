import logging
import math
import json
from typing import List

import pandas as pd

from frontier.config import TraceRequestGeneratorConfig
from frontier.entities import Request, RequestRoundPlan
from frontier.request_generator.base_request_generator import BaseRequestGenerator

logger = logging.getLogger(__name__)


def _parse_block_hash_ids(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        delimiter = "|" if "|" in stripped else ","
        return [int(part.strip()) for part in stripped.split(delimiter) if part.strip()]
    if isinstance(value, int):
        return [int(value)]
    raise ValueError(f"Unsupported block_hash_ids value: {value!r}")


def _is_missing_value(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _parse_optional_int(value) -> int | None:
    if _is_missing_value(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def _parse_optional_float(value) -> float | None:
    if _is_missing_value(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _finite_float(value, *, field_name: str, context: str) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be numeric. {context}; value={value!r}."
        ) from exc
    if not math.isfinite(numeric_value):
        raise ValueError(
            f"{field_name} must be finite. {context}; value={value!r}."
        )
    return numeric_value


def scale_positive_token_count(
    value,
    scale_factor,
    *,
    field_name: str,
    context: str,
) -> int:
    """Apply trace token scaling with strict session-prefix validation."""
    raw_value = _finite_float(
        value,
        field_name=field_name,
        context=context,
    )
    if raw_value <= 0:
        raise ValueError(
            f"{field_name} must be positive before scaling. "
            f"{context}; value={value!r}."
        )

    factor_field_name = {
        "num_prefill_tokens": "prefill_scale_factor",
        "num_decode_tokens": "decode_scale_factor",
    }.get(field_name, f"{field_name}_scale_factor")
    factor = _finite_float(
        scale_factor,
        field_name=factor_field_name,
        context=context,
    )
    if factor <= 0:
        raise ValueError(
            f"{factor_field_name} must be positive. "
            f"{context}; value={scale_factor!r}."
        )

    scaled_value = raw_value * factor
    if not math.isfinite(scaled_value) or scaled_value <= 0:
        raise ValueError(
            f"{field_name} must be finite and positive after scaling. "
            f"{context}; value={value!r}; scale_factor={scale_factor!r}; "
            f"scaled_value={scaled_value!r}."
        )

    integer_value = int(scaled_value)
    if integer_value < 1:
        raise ValueError(
            f"{field_name} must remain positive after scaling and integerization. "
            f"{context}; value={value!r}; scale_factor={scale_factor!r}; "
            f"scaled_value={scaled_value!r}; integer_value={integer_value}."
        )
    return integer_value


def scale_nonnegative_arrival_time(
    value,
    scale_factor,
    *,
    context: str,
) -> float:
    """Apply arrival-time scaling while rejecting invalid session timestamps."""
    raw_value = _finite_float(
        value,
        field_name="arrived_at",
        context=context,
    )
    if raw_value < 0:
        raise ValueError(
            f"arrived_at must be nonnegative before scaling. "
            f"{context}; value={value!r}."
        )

    factor = _finite_float(
        scale_factor,
        field_name="time_scale_factor",
        context=context,
    )
    if factor <= 0:
        raise ValueError(
            f"time_scale_factor must be positive. "
            f"{context}; value={scale_factor!r}."
        )

    scaled_value = raw_value * factor
    if not math.isfinite(scaled_value) or scaled_value < 0:
        raise ValueError(
            "arrived_at must be finite and nonnegative after scaling. "
            f"{context}; value={value!r}; scale_factor={scale_factor!r}; "
            f"scaled_value={scaled_value!r}."
        )
    return scaled_value


def parse_thinking_round_plans(
    value,
    *,
    prefill_scale_factor: float = 1.0,
    decode_scale_factor: float = 1.0,
    context: str = "thinking_round_plans_json",
) -> list[RequestRoundPlan] | None:
    if _is_missing_value(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        raw_round_plans = json.loads(stripped)
    else:
        raw_round_plans = value

    if not isinstance(raw_round_plans, list):
        raise ValueError(
            "thinking_round_plans_json must decode to a list of round plans."
        )

    round_plans: list[RequestRoundPlan] = []
    for round_index, raw_round_plan in enumerate(raw_round_plans, start=1):
        if not isinstance(raw_round_plan, dict):
            raise ValueError(
                "Each thinking round plan must be a dict with prefill/decode token counts."
            )
        round_context = f"{context}; round={round_index}"
        missing_fields = sorted(
            {
                "num_prefill_tokens",
                "num_decode_tokens",
            }
            - set(raw_round_plan)
        )
        if missing_fields:
            raise ValueError(
                "Thinking round plan is missing token fields. "
                f"{round_context}; missing_fields={missing_fields}."
            )
        round_plans.append(
            RequestRoundPlan(
                num_prefill_tokens=scale_positive_token_count(
                    raw_round_plan["num_prefill_tokens"],
                    prefill_scale_factor,
                    field_name="num_prefill_tokens",
                    context=round_context,
                ),
                num_decode_tokens=scale_positive_token_count(
                    raw_round_plan["num_decode_tokens"],
                    decode_scale_factor,
                    field_name="num_decode_tokens",
                    context=round_context,
                ),
            )
        )
    return round_plans


def materialize_incremental_session_round_plans(
    *,
    initial_context_tokens: int,
    incremental_round_plans: list[RequestRoundPlan],
    max_tokens: int,
    context: str,
) -> tuple[list[RequestRoundPlan], int]:
    """Expand new-token ISLs into full prompt lengths for one session."""
    context_tokens = int(initial_context_tokens)
    effective_round_plans: list[RequestRoundPlan] = []
    for round_index, round_plan in enumerate(incremental_round_plans, start=1):
        new_prefill_tokens = int(round_plan.num_prefill_tokens)
        decode_tokens = int(round_plan.num_decode_tokens)
        if new_prefill_tokens < 1 or decode_tokens < 1:
            raise ValueError(
                "Session prefix incremental ISL requires positive prefill and "
                f"decode token counts. {context}; round={round_index}; "
                f"new_prefill_tokens={new_prefill_tokens}; "
                f"decode_tokens={decode_tokens}."
            )

        effective_prefill_tokens = context_tokens + new_prefill_tokens
        resulting_context_tokens = effective_prefill_tokens + decode_tokens
        if resulting_context_tokens > int(max_tokens):
            raise ValueError(
                "Session prefix incremental ISL expands beyond max_tokens. "
                f"{context}; round={round_index}; "
                f"prior_context_tokens={context_tokens}; "
                f"new_prefill_tokens={new_prefill_tokens}; "
                f"effective_prefill_tokens={effective_prefill_tokens}; "
                f"decode_tokens={decode_tokens}; max_tokens={max_tokens}."
            )

        effective_round_plans.append(
            RequestRoundPlan(
                num_prefill_tokens=effective_prefill_tokens,
                num_decode_tokens=decode_tokens,
            )
        )
        context_tokens = resulting_context_tokens

    return effective_round_plans, context_tokens


class TraceReplayRequestGenerator(BaseRequestGenerator):
    """
    Reads a trace csv file containing request arrival time, its prompt and completion token values to generate
    inter-request times, number of tokens.
    """

    def __init__(self, config: TraceRequestGeneratorConfig):
        super().__init__(config)

        # load into a pd dataframe
        self.trace_df = pd.read_csv(config.trace_file)
        for column_name in (
            "arrived_at",
            "num_prefill_tokens",
            "num_decode_tokens",
        ):
            if column_name not in self.trace_df.columns:
                raise ValueError(
                    f"Trace file {config.trace_file} is missing required "
                    f"column {column_name!r}."
                )
            self.trace_df[f"_raw_{column_name}"] = self.trace_df[column_name]
            for row_index, raw_value in self.trace_df[column_name].items():
                _finite_float(
                    raw_value,
                    field_name=column_name,
                    context=(
                        f"Trace file: {config.trace_file}; "
                        f"row_index={row_index}"
                    ),
                )

        for factor_name in (
            "prefill_scale_factor",
            "decode_scale_factor",
            "time_scale_factor",
        ):
            _finite_float(
                getattr(config, factor_name),
                field_name=factor_name,
                context=f"Trace file: {config.trace_file}",
            )

        # scale prefill and decode tokens
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] * config.prefill_scale_factor
        )
        self.trace_df["num_decode_tokens"] = (
            self.trace_df["num_decode_tokens"] * config.decode_scale_factor
        )
        for column_name in ("num_prefill_tokens", "num_decode_tokens"):
            for row_index, scaled_value in self.trace_df[column_name].items():
                _finite_float(
                    scaled_value,
                    field_name=column_name,
                    context=(
                        f"Trace file: {config.trace_file}; "
                        f"row_index={row_index}; after scaling"
                    ),
                )

        # make sure all the prefill and decode counts are integers
        self.trace_df["num_prefill_tokens"] = self.trace_df[
            "num_prefill_tokens"
        ].astype(int)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].astype(
            int
        )

        # make sure that there is at least one prefill and decode token
        self.trace_df["num_prefill_tokens"] = self.trace_df["num_prefill_tokens"].clip(
            lower=1
        )
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].clip(
            lower=1
        )

        # make sure the total does not exceed the max tokens, adjust the prefill tokens if needed
        total_tokens = (
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
        )
        if "thinking_round_plans_json" in self.trace_df.columns:
            overflowing_multi_round_rows = []
            for row_index, row in self.trace_df.loc[
                total_tokens > config.max_tokens
            ].iterrows():
                row_thinking_depth = _parse_optional_int(row.get("thinking_depth"))
                row_thinking_round_plans = parse_thinking_round_plans(
                    row.get("thinking_round_plans_json"),
                    prefill_scale_factor=config.prefill_scale_factor,
                    decode_scale_factor=config.decode_scale_factor,
                    context=(
                        f"Trace file: {config.trace_file}; "
                        f"row_index={row_index}"
                    ),
                )
                is_multi_round_row = (
                    row_thinking_round_plans is not None
                    or (
                        row_thinking_depth is not None
                        and row_thinking_depth > 1
                    )
                )
                if not is_multi_round_row:
                    continue
                session_id = _parse_optional_int(row.get("session_id"))
                row_label = (
                    f"session_id={session_id}"
                    if session_id is not None
                    else f"row_index={row_index}"
                )
                overflowing_multi_round_rows.append(row_label)
            if overflowing_multi_round_rows:
                raise ValueError(
                    "Trace replay multi-round rows exceed "
                    f"max_tokens={config.max_tokens}: "
                    f"{', '.join(overflowing_multi_round_rows)}. "
                    "Increase trace_request_generator_config_max_tokens "
                    "instead of relying on prefill clipping."
                )
        diff_tokens = total_tokens - config.max_tokens
        diff_tokens = diff_tokens.clip(lower=0)
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] - diff_tokens
        )

        assert all(
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
            <= config.max_tokens
        )

        # rescale the time to change QPS
        self.trace_df["arrived_at"] = (
            self.trace_df["arrived_at"] * config.time_scale_factor
        )
        for row_index, scaled_arrived_at in self.trace_df["arrived_at"].items():
            _finite_float(
                scaled_arrived_at,
                field_name="arrived_at",
                context=(
                    f"Trace file: {config.trace_file}; "
                    f"row_index={row_index}; after scaling"
                ),
            )

        logger.info(
            f"Loaded trace file {config.trace_file} with {len(self.trace_df)} requests"
        )
        # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles
        pd_ratio = (
            self.trace_df["num_prefill_tokens"] / self.trace_df["num_decode_tokens"]
        )
        logger.debug(
            f"Prompt/decode token ratio stats\n:{pd_ratio.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])}"
        )

    def generate_requests(self) -> List[Request]:
        if self._session_prefix_incremental_isl:
            return self._generate_incremental_session_requests()

        requests = []

        for _, row in self.trace_df.iterrows():
            # Read priority from trace file if available, otherwise default to 0
            priority = int(row.get("priority", 0))
            session_id = _parse_optional_int(row.get("session_id"))
            cohort = row.get("cohort")
            if _is_missing_value(cohort):
                cohort = None
            elif isinstance(cohort, str):
                cohort = cohort.strip() or None
            block_hash_ids = _parse_block_hash_ids(row.get("block_hash_ids"))
            row_thinking_depth = _parse_optional_int(row.get("thinking_depth"))
            row_tool_call_latency = _parse_optional_float(row.get("tool_call_latency"))
            row_thinking_round_plans = parse_thinking_round_plans(
                row.get("thinking_round_plans_json"),
                prefill_scale_factor=self.config.prefill_scale_factor,
                decode_scale_factor=self.config.decode_scale_factor,
                context=(
                    f"Trace file: {self.config.trace_file}; "
                    f"row_index={row.name}"
                ),
            )

            if (
                row_thinking_depth is None
                and row_tool_call_latency is None
                and row_thinking_round_plans is None
            ):
                request = self._build_request(
                    arrived_at=row["arrived_at"],
                    num_prefill_tokens=row["num_prefill_tokens"],
                    num_decode_tokens=row["num_decode_tokens"],
                    priority=priority,
                    block_hash_ids=block_hash_ids,
                    session_id=session_id,
                    cohort=cohort,
                )
                requests.append(request)
                continue

            if row_thinking_round_plans is not None:
                inferred_thinking_depth = len(row_thinking_round_plans)
            else:
                inferred_thinking_depth = 1
            thinking_depth = (
                row_thinking_depth
                if row_thinking_depth is not None
                else inferred_thinking_depth
            )
            if thinking_depth < 1:
                raise ValueError(
                    f"thinking_depth must be >= 1 for trace row, got={thinking_depth}"
                )
            if row_thinking_round_plans is None and thinking_depth != 1:
                raise ValueError(
                    "Trace rows with thinking_depth > 1 must provide thinking_round_plans_json."
                )
            if row_tool_call_latency is None:
                row_tool_call_latency = 0.001

            request = Request(
                arrived_at=float(row["arrived_at"]),
                num_prefill_tokens=int(row["num_prefill_tokens"]),
                num_decode_tokens=int(row["num_decode_tokens"]),
                priority=priority,
                block_hash_ids=block_hash_ids,
                session_id=session_id,
                cohort=cohort,
                thinking_depth=thinking_depth,
                tool_call_latency=row_tool_call_latency,
                thinking_round_plans=row_thinking_round_plans,
            )

            requests.append(request)

        return requests

    def _generate_incremental_session_requests(self) -> List[Request]:
        requests: list[Request] = []
        context_tokens_by_session: dict[int, int] = {}

        for row_index, row in self.trace_df.iterrows():
            row_context = (
                f"Trace file: {self.config.trace_file}; row_index={row_index}"
            )
            priority = int(row.get("priority", 0))
            session_id = _parse_optional_int(row.get("session_id"))
            if session_id is None:
                raise ValueError(
                    "session_id is required for session prefix incremental ISL. "
                    f"row_index={row_index}."
                )

            cohort = row.get("cohort")
            if _is_missing_value(cohort):
                cohort = None
            elif isinstance(cohort, str):
                cohort = cohort.strip() or None
            block_hash_ids = _parse_block_hash_ids(row.get("block_hash_ids"))

            row_thinking_depth = _parse_optional_int(row.get("thinking_depth"))
            row_tool_call_latency = _parse_optional_float(
                row.get("tool_call_latency")
            )
            explicit_round_plans = parse_thinking_round_plans(
                row.get("thinking_round_plans_json"),
                prefill_scale_factor=self.config.prefill_scale_factor,
                decode_scale_factor=self.config.decode_scale_factor,
                context=row_context,
            )
            final_incremental_plan = RequestRoundPlan(
                num_prefill_tokens=scale_positive_token_count(
                    row["_raw_num_prefill_tokens"],
                    self.config.prefill_scale_factor,
                    field_name="num_prefill_tokens",
                    context=row_context,
                ),
                num_decode_tokens=scale_positive_token_count(
                    row["_raw_num_decode_tokens"],
                    self.config.decode_scale_factor,
                    field_name="num_decode_tokens",
                    context=row_context,
                ),
            )
            arrived_at = scale_nonnegative_arrival_time(
                row["_raw_arrived_at"],
                self.config.time_scale_factor,
                context=row_context,
            )

            has_row_thinking_override = (
                row_thinking_depth is not None
                or row_tool_call_latency is not None
                or explicit_round_plans is not None
            )
            preserve_round_plans = explicit_round_plans is not None
            if explicit_round_plans is not None:
                thinking_depth = (
                    row_thinking_depth
                    if row_thinking_depth is not None
                    else len(explicit_round_plans)
                )
                if len(explicit_round_plans) != thinking_depth:
                    raise ValueError(
                        "thinking_round_plans length must match thinking_depth, "
                        f"got len={len(explicit_round_plans)} "
                        f"depth={thinking_depth}; row_index={row_index}."
                    )
                if explicit_round_plans[-1] != final_incremental_plan:
                    raise ValueError(
                        "The final incremental thinking round plan must match "
                        "the trace row's num_prefill_tokens and "
                        f"num_decode_tokens; row_index={row_index}."
                    )
                incremental_round_plans = explicit_round_plans
                tool_call_latency = (
                    row_tool_call_latency
                    if row_tool_call_latency is not None
                    else 0.001
                )
            elif has_row_thinking_override:
                thinking_depth = (
                    row_thinking_depth if row_thinking_depth is not None else 1
                )
                if thinking_depth != 1:
                    raise ValueError(
                        "Trace rows with thinking_depth > 1 must provide "
                        f"thinking_round_plans_json; row_index={row_index}."
                    )
                incremental_round_plans = [final_incremental_plan]
                tool_call_latency = (
                    row_tool_call_latency
                    if row_tool_call_latency is not None
                    else 0.001
                )
            else:
                hidden_round_plans = self._build_hidden_round_plans(
                    final_prefill_tokens=final_incremental_plan.num_prefill_tokens,
                    final_decode_tokens=final_incremental_plan.num_decode_tokens,
                )
                incremental_round_plans = hidden_round_plans + [
                    final_incremental_plan
                ]
                thinking_depth = (
                    self._thinking_depth if self._thinking_mode_enabled else 1
                )
                tool_call_latency = self._tool_call_latency
                preserve_round_plans = bool(hidden_round_plans)

            initial_context_tokens = context_tokens_by_session.get(session_id, 0)
            effective_round_plans, resulting_context_tokens = (
                materialize_incremental_session_round_plans(
                    initial_context_tokens=initial_context_tokens,
                    incremental_round_plans=incremental_round_plans,
                    max_tokens=int(self.config.max_tokens),
                    context=f"{row_context}; session_id={session_id}",
                )
            )
            context_tokens_by_session[session_id] = resulting_context_tokens
            final_effective_plan = effective_round_plans[-1]

            requests.append(
                Request(
                    arrived_at=arrived_at,
                    num_prefill_tokens=final_effective_plan.num_prefill_tokens,
                    num_decode_tokens=final_effective_plan.num_decode_tokens,
                    priority=priority,
                    block_hash_ids=block_hash_ids,
                    session_id=session_id,
                    cohort=cohort,
                    thinking_depth=thinking_depth,
                    tool_call_latency=tool_call_latency,
                    thinking_round_plans=(
                        effective_round_plans
                        if preserve_round_plans or thinking_depth > 1
                        else None
                    ),
                )
            )

        return requests
