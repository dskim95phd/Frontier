import math
from dataclasses import dataclass, field, replace


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
    static_slice_per_gpu: bool = field(
        default=False,
        metadata={
            "help": (
                "Derive one CPU KV-cache target from a fixed per-GPU capacity "
                "and bandwidth slice. The target is aggregated over attention TP."
            )
        },
    )
    capacity_bytes_per_gpu: int = field(
        default=750_000_000_000,
        metadata={"help": "Static CPU DRAM capacity assigned to one GPU in bytes."},
    )
    read_bandwidth_gbps_per_gpu: float = field(
        default=4_800.0,
        metadata={"help": "Static CPU-to-GPU bandwidth assigned to one GPU in Gbps."},
    )
    write_bandwidth_gbps_per_gpu: float = field(
        default=4_800.0,
        metadata={"help": "Static GPU-to-CPU bandwidth assigned to one GPU in Gbps."},
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
        if self.enable and not self.static_slice_per_gpu and self.capacity_bytes <= 0:
            raise ValueError(
                "cpu_kv_cache_config.capacity_bytes must be > 0 when CPU "
                "KV-cache offloading is enabled"
            )
        if self.capacity_bytes_per_gpu <= 0:
            raise ValueError(
                "cpu_kv_cache_config.capacity_bytes_per_gpu must be > 0, "
                f"got={self.capacity_bytes_per_gpu}"
            )
        for field_name in (
            "read_bandwidth_gbps_per_gpu",
            "write_bandwidth_gbps_per_gpu",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"cpu_kv_cache_config.{field_name} must be > 0, got={value}"
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

    def resolve_for_target(self, num_gpus: int) -> "CpuKVCacheConfig":
        """Resolve a static per-GPU slice into one aggregate cache target."""

        resolved_num_gpus = int(num_gpus)
        if resolved_num_gpus <= 0:
            raise ValueError(
                f"CPU KV-cache target num_gpus must be > 0, got={num_gpus}"
            )
        if not self.static_slice_per_gpu:
            return self
        return replace(
            self,
            static_slice_per_gpu=False,
            capacity_bytes=self.capacity_bytes_per_gpu * resolved_num_gpus,
            read_bandwidth_gbps=(
                self.read_bandwidth_gbps_per_gpu * resolved_num_gpus
            ),
            write_bandwidth_gbps=(
                self.write_bandwidth_gbps_per_gpu * resolved_num_gpus
            ),
        )


# Compatibility alias. FlatDataclass uses the underlying class name, whose
# casing yields the documented `cpu_kv_cache_config_*` CLI prefix.
CPUKVCacheConfig = CpuKVCacheConfig
