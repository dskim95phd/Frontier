"""Standalone workload generation for the C++ Frontier simulator.

This module deliberately does not import the Python Frontier simulator.  It
produces the normalized CSV records consumed by ``workload.cc`` and can also
be imported by benchmark drivers that need to reuse request shapes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
from typing import Mapping, Protocol, Sequence, TextIO

import numpy as np


@dataclass(frozen=True)
class WorkloadRequest:
    session_start_at: float | None
    think_time: float
    num_prefill_tokens: int
    num_decode_tokens: int
    session_id: int | None = None
    session_turn_index: int | None = None


@dataclass(frozen=True)
class RequestShape:
    num_prefill_tokens: int
    num_decode_tokens: int
    unit_interval: float

    @property
    def prompt_tokens(self) -> int:
        return self.num_prefill_tokens

    @property
    def output_tokens(self) -> int:
        return self.num_decode_tokens

    @property
    def unit_exponential_gap(self) -> float:
        return self.unit_interval


class RandomStreams:
    """Seeded streams matching the source distributions' random APIs."""

    def __init__(self, seed: int):
        self.seed = seed
        self.python = random.Random(seed)
        self.numpy = np.random.default_rng(seed)


class LengthDistribution(Protocol):
    def sample(self, streams: RandomStreams) -> tuple[int, int]: ...


class IntervalDistribution(Protocol):
    def sample(self, streams: RandomStreams) -> float: ...


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be a finite number")
    return parsed


def _positive_float(value: object, field: str) -> float:
    parsed = _finite_float(value, field)
    if parsed <= 0.0:
        raise ValueError(f"{field} must be positive")
    return parsed


@dataclass(frozen=True)
class FixedLengthDistribution:
    prefill_tokens: int
    decode_tokens: int

    def __post_init__(self) -> None:
        _positive_int(self.prefill_tokens, "prefill_tokens")
        _positive_int(self.decode_tokens, "decode_tokens")

    def sample(self, streams: RandomStreams) -> tuple[int, int]:
        del streams
        return self.prefill_tokens, self.decode_tokens


@dataclass(frozen=True)
class UniformLengthDistribution:
    min_tokens: int
    max_tokens: int
    prefill_to_decode_ratio: float

    def __post_init__(self) -> None:
        minimum = _positive_int(self.min_tokens, "min_tokens")
        maximum = _positive_int(self.max_tokens, "max_tokens")
        if minimum > maximum:
            raise ValueError("min_tokens cannot exceed max_tokens")
        _positive_float(self.prefill_to_decode_ratio, "prefill_to_decode_ratio")

    def sample(self, streams: RandomStreams) -> tuple[int, int]:
        total_tokens = streams.python.uniform(self.min_tokens, self.max_tokens)
        decode_tokens = math.ceil(
            total_tokens / (1.0 + self.prefill_to_decode_ratio)
        )
        prefill_tokens = max(1, int(total_tokens - decode_tokens))
        return prefill_tokens, decode_tokens


class ZipfLengthDistribution:
    """Bounded YCSB-style Zipf distribution used by Python Frontier."""

    def __init__(
        self,
        *,
        min_tokens: int,
        max_tokens: int,
        theta: float,
        prefill_to_decode_ratio: float,
        scramble: bool = False,
        seed: int = 42,
    ) -> None:
        self.min_tokens = _positive_int(min_tokens, "min_tokens")
        self.max_tokens = _positive_int(max_tokens, "max_tokens")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens cannot exceed max_tokens")
        self.theta = _finite_float(theta, "theta")
        if self.theta < 0.0 or self.theta >= 1.0:
            raise ValueError("theta must be in [0, 1)")
        self.prefill_to_decode_ratio = _positive_float(
            prefill_to_decode_ratio, "prefill_to_decode_ratio"
        )
        if not isinstance(scramble, bool):
            raise ValueError("scramble must be a boolean")
        self.scramble = scramble
        self.seed = _nonnegative_int(seed, "seed")
        if self.seed > 2**32 - 1:
            raise ValueError("seed must fit in an unsigned 32-bit integer")
        self._items = self.max_tokens - self.min_tokens + 1
        self._zeta_2 = self._zeta(2)
        self._alpha = 1.0 / (1.0 - self.theta)
        self._zetan = self._zeta(self._items)
        self._eta = (
            1.0 - math.pow(2.0 / self._items, 1.0 - self.theta)
        ) / (1.0 - self._zeta_2 / (self._zetan + 1e-8))
        self._rng = np.random.RandomState(seed)

    def _zeta(self, count: int) -> float:
        return float(
            np.sum(1.0 / np.power(np.arange(1, count), self.theta))
        )

    def _next_total_tokens(self) -> int:
        u = float(self._rng.random_sample())
        uz = u * self._zetan
        if uz < 1.0:
            value = self.min_tokens
        elif uz < 1.0 + math.pow(0.5, self.theta):
            value = self.min_tokens + 1
        else:
            value = self.min_tokens + int(
                self._items
                * math.pow(self._eta * u - self._eta + 1.0, self._alpha)
            )
        value = min(value, self.max_tokens)
        if not self.scramble:
            return value

        # Python's salted hash is not reproducible across processes.  Preserve
        # the old intent with a stable seed-dependent permutation instead.
        digest = hashlib.blake2b(
            f"{value}:{self.seed}".encode("ascii"), digest_size=8
        ).digest()
        offset = int.from_bytes(digest, "little") % self._items
        return self.min_tokens + offset

    def sample(self, streams: RandomStreams) -> tuple[int, int]:
        del streams
        total_tokens = self._next_total_tokens()
        decode_tokens = math.ceil(
            total_tokens / (1.0 + self.prefill_to_decode_ratio)
        )
        prefill_tokens = max(1, int(total_tokens - decode_tokens))
        return prefill_tokens, decode_tokens


@dataclass(frozen=True)
class BoundedLognormalLengthDistribution:
    prefill_median: float
    prefill_sigma: float
    prefill_min: int
    prefill_max: int
    decode_median: float
    decode_sigma: float
    decode_min: int
    decode_max: int

    def __post_init__(self) -> None:
        _positive_float(self.prefill_median, "prefill_median")
        _positive_float(self.decode_median, "decode_median")
        for field, value in (
            ("prefill_sigma", self.prefill_sigma),
            ("decode_sigma", self.decode_sigma),
        ):
            if _finite_float(value, field) < 0.0:
                raise ValueError(f"{field} must be nonnegative")
        for prefix, lower, upper in (
            ("prefill", self.prefill_min, self.prefill_max),
            ("decode", self.decode_min, self.decode_max),
        ):
            minimum = _positive_int(lower, f"{prefix}_min")
            maximum = _positive_int(upper, f"{prefix}_max")
            if minimum > maximum:
                raise ValueError(f"{prefix}_min cannot exceed {prefix}_max")

    @staticmethod
    def _sample(
        streams: RandomStreams,
        *,
        median: float,
        sigma: float,
        lower: int,
        upper: int,
    ) -> int:
        sample = math.exp(streams.numpy.normal(math.log(median), sigma))
        return int(min(upper, max(lower, round(sample))))

    def sample(self, streams: RandomStreams) -> tuple[int, int]:
        return (
            self._sample(
                streams,
                median=self.prefill_median,
                sigma=self.prefill_sigma,
                lower=self.prefill_min,
                upper=self.prefill_max,
            ),
            self._sample(
                streams,
                median=self.decode_median,
                sigma=self.decode_sigma,
                lower=self.decode_min,
                upper=self.decode_max,
            ),
        )


@dataclass(frozen=True)
class StaticIntervalDistribution:
    interval_seconds: float = 0.0

    def __post_init__(self) -> None:
        if _finite_float(self.interval_seconds, "interval_seconds") < 0.0:
            raise ValueError("interval_seconds must be nonnegative")

    def sample(self, streams: RandomStreams) -> float:
        del streams
        return float(self.interval_seconds)


@dataclass(frozen=True)
class PoissonIntervalDistribution:
    qps: float
    max_interval_factor: float | None = None

    def __post_init__(self) -> None:
        _positive_float(self.qps, "qps")
        if self.max_interval_factor is not None:
            _positive_float(self.max_interval_factor, "max_interval_factor")

    def sample(self, streams: RandomStreams) -> float:
        interval = float(streams.numpy.exponential(1.0 / self.qps))
        if self.max_interval_factor is not None:
            interval = min(interval, self.max_interval_factor / self.qps)
        return interval


@dataclass(frozen=True)
class GammaIntervalDistribution:
    qps: float
    cv: float

    def __post_init__(self) -> None:
        _positive_float(self.qps, "qps")
        _positive_float(self.cv, "cv")

    def sample(self, streams: RandomStreams) -> float:
        shape = 1.0 / (self.cv**2)
        scale = 1.0 / (self.qps * shape)
        return float(streams.numpy.gamma(shape, scale))


def generate_request_shapes(
    *,
    num_requests: int,
    seed: int,
    length_distribution: LengthDistribution,
    interval_distribution: IntervalDistribution,
) -> list[RequestShape]:
    _positive_int(num_requests, "num_requests")
    _nonnegative_int(seed, "seed")
    streams = RandomStreams(seed)
    shapes: list[RequestShape] = []
    for _ in range(num_requests):
        prefill_tokens, decode_tokens = length_distribution.sample(streams)
        interval = interval_distribution.sample(streams)
        if prefill_tokens <= 0 or decode_tokens <= 0:
            raise ValueError("length distribution produced nonpositive tokens")
        if not math.isfinite(interval) or interval < 0.0:
            raise ValueError("interval distribution produced an invalid interval")
        shapes.append(
            RequestShape(prefill_tokens, decode_tokens, float(interval))
        )
    return shapes


def materialize_requests(
    shapes: Sequence[RequestShape],
    *,
    interval_scale: float = 1.0,
    fixed_interval_seconds: float | None = None,
    first_arrival_at_zero: bool = True,
) -> list[WorkloadRequest]:
    scale = _positive_float(interval_scale, "interval_scale")
    if fixed_interval_seconds is not None:
        fixed_interval_seconds = _finite_float(
            fixed_interval_seconds, "fixed_interval_seconds"
        )
        if fixed_interval_seconds < 0.0:
            raise ValueError("fixed_interval_seconds must be nonnegative")

    session_start_at = 0.0
    requests: list[WorkloadRequest] = []
    for index, shape in enumerate(shapes):
        interval = (
            fixed_interval_seconds
            if fixed_interval_seconds is not None
            else shape.unit_interval * scale
        )
        if index > 0 or not first_arrival_at_zero:
            session_start_at += interval
        requests.append(
            WorkloadRequest(
                session_start_at=session_start_at,
                think_time=0.0,
                num_prefill_tokens=shape.num_prefill_tokens,
                num_decode_tokens=shape.num_decode_tokens,
            )
        )
    return requests


def generate_requests(
    *,
    num_requests: int,
    seed: int,
    length_distribution: LengthDistribution,
    interval_distribution: IntervalDistribution,
    first_arrival_at_zero: bool = True,
) -> list[WorkloadRequest]:
    shapes = generate_request_shapes(
        num_requests=num_requests,
        seed=seed,
        length_distribution=length_distribution,
        interval_distribution=interval_distribution,
    )
    return materialize_requests(
        shapes, first_arrival_at_zero=first_arrival_at_zero
    )


def write_workload_csv(
    destination: Path | TextIO,
    requests: Sequence[WorkloadRequest],
    *,
    time_decimal_places: int | None = None,
) -> None:
    should_close = isinstance(destination, Path)
    if should_close:
        destination.parent.mkdir(parents=True, exist_ok=True)
        output = destination.open("w", encoding="utf-8", newline="")
    else:
        output = destination

    try:
        writer = csv.writer(output, lineterminator="\n")
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
        seen_sessions: set[int] = set()
        for request in requests:
            if request.session_start_at is not None and (
                not math.isfinite(request.session_start_at)
                or request.session_start_at < 0.0
            ):
                raise ValueError(
                    "request session_start_at must be finite and nonnegative"
                )
            if not math.isfinite(request.think_time) or request.think_time < 0.0:
                raise ValueError("request think_time must be finite and nonnegative")
            _positive_int(request.num_prefill_tokens, "num_prefill_tokens")
            _positive_int(request.num_decode_tokens, "num_decode_tokens")
            if request.session_turn_index is not None and request.session_id is None:
                raise ValueError("session_turn_index requires session_id")
            if request.session_id is not None:
                _nonnegative_int(request.session_id, "session_id")
            if request.session_turn_index is not None:
                _nonnegative_int(
                    request.session_turn_index, "session_turn_index"
                )
            first_turn = (
                request.session_id is None or request.session_id not in seen_sessions
            )
            if request.session_id is not None:
                seen_sessions.add(request.session_id)
            if first_turn and request.session_start_at is None:
                raise ValueError("a first turn requires session_start_at")
            if first_turn and request.think_time != 0.0:
                raise ValueError("a first turn must have think_time=0")
            if not first_turn and request.session_start_at is not None:
                raise ValueError("a successor turn must omit session_start_at")
            session_start_at = ""
            if request.session_start_at is not None:
                session_start_at = (
                    format(request.session_start_at, ".17g")
                    if time_decimal_places is None
                    else f"{request.session_start_at:.{time_decimal_places}f}"
                )
            think_time = (
                format(request.think_time, ".17g")
                if time_decimal_places is None
                else f"{request.think_time:.{time_decimal_places}f}"
            )
            writer.writerow(
                (
                    session_start_at,
                    think_time,
                    request.num_prefill_tokens,
                    request.num_decode_tokens,
                    "" if request.session_id is None else request.session_id,
                    ""
                    if request.session_turn_index is None
                    else request.session_turn_index,
                )
            )
    finally:
        if should_close:
            output.close()


def _required(config: Mapping[str, object], field: str) -> object:
    if field not in config:
        raise ValueError(f"missing required field: {field}")
    return config[field]


def length_distribution_from_config(
    config: Mapping[str, object], *, seed: int
) -> LengthDistribution:
    distribution_type = config.get("type")
    if distribution_type == "fixed":
        return FixedLengthDistribution(
            _positive_int(_required(config, "prefill_tokens"), "prefill_tokens"),
            _positive_int(_required(config, "decode_tokens"), "decode_tokens"),
        )
    if distribution_type == "uniform":
        return UniformLengthDistribution(
            _positive_int(_required(config, "min_tokens"), "min_tokens"),
            _positive_int(_required(config, "max_tokens"), "max_tokens"),
            _positive_float(
                config.get("prefill_to_decode_ratio", 20.0),
                "prefill_to_decode_ratio",
            ),
        )
    if distribution_type == "zipf":
        scramble = config.get("scramble", False)
        if not isinstance(scramble, bool):
            raise ValueError("scramble must be a boolean")
        return ZipfLengthDistribution(
            min_tokens=_positive_int(
                _required(config, "min_tokens"), "min_tokens"
            ),
            max_tokens=_positive_int(
                _required(config, "max_tokens"), "max_tokens"
            ),
            theta=_finite_float(config.get("theta", 0.6), "theta"),
            prefill_to_decode_ratio=_positive_float(
                config.get("prefill_to_decode_ratio", 20.0),
                "prefill_to_decode_ratio",
            ),
            scramble=scramble,
            seed=seed,
        )
    if distribution_type == "bounded_lognormal":
        return BoundedLognormalLengthDistribution(
            prefill_median=_positive_float(
                _required(config, "prefill_median"), "prefill_median"
            ),
            prefill_sigma=_finite_float(
                _required(config, "prefill_sigma"), "prefill_sigma"
            ),
            prefill_min=_positive_int(
                _required(config, "prefill_min"), "prefill_min"
            ),
            prefill_max=_positive_int(
                _required(config, "prefill_max"), "prefill_max"
            ),
            decode_median=_positive_float(
                _required(config, "decode_median"), "decode_median"
            ),
            decode_sigma=_finite_float(
                _required(config, "decode_sigma"), "decode_sigma"
            ),
            decode_min=_positive_int(
                _required(config, "decode_min"), "decode_min"
            ),
            decode_max=_positive_int(
                _required(config, "decode_max"), "decode_max"
            ),
        )
    raise ValueError(
        "length.type must be fixed, uniform, zipf, or bounded_lognormal"
    )


def interval_distribution_from_config(
    config: Mapping[str, object],
) -> IntervalDistribution:
    distribution_type = config.get("type")
    if distribution_type == "static":
        return StaticIntervalDistribution(
            _finite_float(config.get("interval_seconds", 0.0), "interval_seconds")
        )
    if distribution_type == "poisson":
        max_factor = config.get("max_interval_factor")
        return PoissonIntervalDistribution(
            qps=_positive_float(_required(config, "qps"), "qps"),
            max_interval_factor=(
                None
                if max_factor is None
                else _positive_float(max_factor, "max_interval_factor")
            ),
        )
    if distribution_type == "gamma":
        return GammaIntervalDistribution(
            qps=_positive_float(_required(config, "qps"), "qps"),
            cv=_positive_float(_required(config, "cv"), "cv"),
        )
    raise ValueError("interval.type must be static, poisson, or gamma")
