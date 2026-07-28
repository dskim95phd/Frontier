import math
from dataclasses import dataclass, field


@dataclass
class CpuKVCacheConfig:
    """Configuration for the prefill-side CPU KV-cache tier."""

    enable: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable the prefill-side CPU KV-cache tier for sequential "
                "pd-disaggregation."
            )
        },
    )
    capacity_bytes: int = field(
        default=0,
        metadata={
            "help": "CPU KV-cache capacity in bytes for each prefill replica/DP lane."
        },
    )
    write_bandwidth_gbps: float = field(
        default=64.0,
        metadata={
            "help": (
                "Aggregate GPU-to-CPU KV-cache copy bandwidth in Gbps for "
                "one prefill replica/DP cache target."
            )
        },
    )
    write_latency_ms: float = field(
        default=0.01,
        metadata={"help": "Fixed GPU-to-CPU KV-cache copy latency in milliseconds."},
    )
    read_bandwidth_gbps: float = field(
        default=64.0,
        metadata={
            "help": (
                "Aggregate CPU-to-GPU KV-cache copy bandwidth in Gbps for "
                "one prefill replica/DP cache target."
            )
        },
    )
    read_latency_ms: float = field(
        default=0.01,
        metadata={"help": "Fixed CPU-to-GPU KV-cache copy latency in milliseconds."},
    )
    eviction_policy: str = field(
        default="session_lru_suffix",
        metadata={
            "help": "CPU eviction policy. MVP supports session_lru_suffix."
        },
    )
    capacity_pressure_policy: str = field(
        default="prefix_fit",
        metadata={
            "help": (
                "CPU capacity-pressure policy: prefix_fit keeps the largest "
                "leading prefix that fits; skip_offload requires the full delta."
            ),
            "choices": ["prefix_fit", "skip_offload"],
        },
    )
    transfer_concurrency: str = field(
        default="full_duplex_serialized",
        metadata={
            "help": (
                "CPU transfer concurrency model. MVP supports one serialized "
                "queue per direction."
            )
        },
    )

    def __post_init__(self) -> None:
        if self.capacity_bytes < 0:
            raise ValueError(
                f"cpu_kv_cache_config.capacity_bytes must be >= 0, got={self.capacity_bytes}"
            )
        if self.enable and self.capacity_bytes <= 0:
            raise ValueError(
                "cpu_kv_cache_config.capacity_bytes must be > 0 when CPU "
                "KV-cache offloading is enabled"
            )
        if (
            not math.isfinite(self.write_bandwidth_gbps)
            or self.write_bandwidth_gbps <= 0
        ):
            raise ValueError(
                "cpu_kv_cache_config.write_bandwidth_gbps must be > 0, "
                f"got={self.write_bandwidth_gbps}"
            )
        if (
            not math.isfinite(self.read_bandwidth_gbps)
            or self.read_bandwidth_gbps <= 0
        ):
            raise ValueError(
                "cpu_kv_cache_config.read_bandwidth_gbps must be > 0, "
                f"got={self.read_bandwidth_gbps}"
            )
        if not math.isfinite(self.write_latency_ms) or self.write_latency_ms < 0:
            raise ValueError(
                "cpu_kv_cache_config.write_latency_ms must be >= 0, "
                f"got={self.write_latency_ms}"
            )
        if not math.isfinite(self.read_latency_ms) or self.read_latency_ms < 0:
            raise ValueError(
                "cpu_kv_cache_config.read_latency_ms must be >= 0, "
                f"got={self.read_latency_ms}"
            )
        if self.eviction_policy != "session_lru_suffix":
            raise ValueError(
                "cpu_kv_cache_config.eviction_policy must be "
                f"'session_lru_suffix', got={self.eviction_policy!r}"
            )
        if self.capacity_pressure_policy not in {"prefix_fit", "skip_offload"}:
            raise ValueError(
                "cpu_kv_cache_config.capacity_pressure_policy must be one of "
                "['prefix_fit', 'skip_offload'], "
                f"got={self.capacity_pressure_policy!r}"
            )
        if self.transfer_concurrency != "full_duplex_serialized":
            raise ValueError(
                "cpu_kv_cache_config.transfer_concurrency must be "
                "'full_duplex_serialized', "
                f"got={self.transfer_concurrency!r}"
            )


# Compatibility alias. FlatDataclass uses the underlying class name, whose
# casing yields the documented `cpu_kv_cache_config_*` CLI prefix.
CPUKVCacheConfig = CpuKVCacheConfig
