from __future__ import annotations

from dataclasses import dataclass

from frontier.config.cpu_kv_cache_config import CPUKVCacheConfig


@dataclass(frozen=True)
class CPUKVCacheTransferTiming:
    direction: str
    size_bytes: int
    submitted_at: float
    start_time: float
    end_time: float
    queue_time_ms: float
    service_time_ms: float

    @property
    def transfer_time_ms(self) -> float:
        return self.service_time_ms


class AnalyticalCPUKVCacheTransferEngine:
    """One full-duplex, direction-serialized transfer engine per cache target."""

    def __init__(self, config: CPUKVCacheConfig) -> None:
        self._config = config
        self._next_d2h_available_at = 0.0
        self._next_h2d_available_at = 0.0

    @staticmethod
    def _service_time_ms(
        size_bytes: int,
        *,
        bandwidth_gbps: float,
        latency_ms: float,
    ) -> float:
        if size_bytes < 0:
            raise ValueError(f"size_bytes must be >= 0, got={size_bytes}")
        bandwidth_bytes_per_ms = (float(bandwidth_gbps) * 1e9) / (8 * 1000)
        return float(latency_ms) + (int(size_bytes) / bandwidth_bytes_per_ms)

    def schedule(
        self,
        *,
        direction: str,
        size_bytes: int,
        submitted_at: float,
    ) -> CPUKVCacheTransferTiming:
        direction = str(direction).lower()
        submitted_at = float(submitted_at)
        if direction == "d2h":
            available_at = self._next_d2h_available_at
            service_time_ms = self._service_time_ms(
                size_bytes,
                bandwidth_gbps=self._config.write_bandwidth_gbps,
                latency_ms=self._config.write_latency_ms,
            )
        elif direction == "h2d":
            available_at = self._next_h2d_available_at
            service_time_ms = self._service_time_ms(
                size_bytes,
                bandwidth_gbps=self._config.read_bandwidth_gbps,
                latency_ms=self._config.read_latency_ms,
            )
        else:
            raise ValueError(
                f"CPU KV-cache transfer direction must be d2h or h2d, got={direction!r}"
            )

        start_time = max(submitted_at, available_at)
        end_time = start_time + service_time_ms * 1e-3
        if direction == "d2h":
            self._next_d2h_available_at = end_time
        else:
            self._next_h2d_available_at = end_time
        return CPUKVCacheTransferTiming(
            direction=direction,
            size_bytes=int(size_bytes),
            submitted_at=submitted_at,
            start_time=start_time,
            end_time=end_time,
            queue_time_ms=(start_time - submitted_at) * 1e3,
            service_time_ms=service_time_ms,
        )
