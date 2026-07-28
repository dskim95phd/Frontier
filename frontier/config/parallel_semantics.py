from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class FrontierParallelismMapping:
    cluster_num_replicas: int
    attn_tensor_parallel_size: int
    attn_data_parallel_size: int
    moe_tensor_parallel_size: int
    moe_expert_parallel_size: int

    @property
    def attention_parallel_size(self) -> int:
        return self.attn_tensor_parallel_size * self.attn_data_parallel_size

    @property
    def moe_parallel_size(self) -> int:
        return self.moe_tensor_parallel_size * self.moe_expert_parallel_size

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CollectiveSimParallelLayout:
    tp: int
    cp: int
    dp: int
    ep: int

    @property
    def world_size(self) -> int:
        return self.tp * self.cp * self.dp * self.ep

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CollectiveSimPhysicalTopology:
    servers: int
    gpus_per_server: int

    @property
    def world_size(self) -> int:
        return self.servers * self.gpus_per_server

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RackLocalReplicaPlacement:
    num_racks: int
    gpus_per_rack: int
    replica_gpus: int
    num_replicas: int
    replicas_per_rack: int
    active_gpus: int
    idle_gpus: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _validate_positive(name: str, value: int) -> int:
    resolved_value = int(value)
    if resolved_value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return resolved_value


def build_rack_local_replica_placement(
    *,
    num_replicas: int,
    replica_gpus: int,
    gpus_per_rack: int,
    configured_num_racks: int | None = None,
) -> RackLocalReplicaPlacement:
    """Pack whole replicas into cluster-exclusive racks.

    A replica, and therefore every DP lane and model-parallel group inside it,
    remains within one rack. Unused GPUs in the final or fragmented rack stay
    idle and cannot be assigned to another cluster.
    """

    resolved_num_replicas = _validate_positive("num_replicas", num_replicas)
    resolved_replica_gpus = _validate_positive("replica_gpus", replica_gpus)
    resolved_gpus_per_rack = _validate_positive(
        "gpus_per_rack",
        gpus_per_rack,
    )
    if resolved_replica_gpus > resolved_gpus_per_rack:
        raise ValueError(
            "Rack-local placement requires one replica to fit in one rack: "
            f"replica_gpus={resolved_replica_gpus}, "
            f"gpus_per_rack={resolved_gpus_per_rack}"
        )

    replicas_per_rack = resolved_gpus_per_rack // resolved_replica_gpus
    required_num_racks = (
        resolved_num_replicas + replicas_per_rack - 1
    ) // replicas_per_rack
    if configured_num_racks is None:
        resolved_num_racks = required_num_racks
    else:
        resolved_num_racks = _validate_positive(
            "configured_num_racks",
            configured_num_racks,
        )
        if resolved_num_racks < required_num_racks:
            raise ValueError(
                "Configured rack count cannot hold all rack-local replicas: "
                f"configured_num_racks={resolved_num_racks}, "
                f"required_num_racks={required_num_racks}, "
                f"num_replicas={resolved_num_replicas}, "
                f"replica_gpus={resolved_replica_gpus}, "
                f"replicas_per_rack={replicas_per_rack}"
            )

    active_gpus = resolved_num_replicas * resolved_replica_gpus
    return RackLocalReplicaPlacement(
        num_racks=resolved_num_racks,
        gpus_per_rack=resolved_gpus_per_rack,
        replica_gpus=resolved_replica_gpus,
        num_replicas=resolved_num_replicas,
        replicas_per_rack=replicas_per_rack,
        active_gpus=active_gpus,
        idle_gpus=(resolved_num_racks * resolved_gpus_per_rack) - active_gpus,
    )


def validate_frontier_shared_parallel_domains(
    mapping: FrontierParallelismMapping,
) -> None:
    if mapping.attention_parallel_size != mapping.moe_parallel_size:
        raise ValueError(
            "Frontier shared attention/MoE parallel domain requires "
            "attn_tp*attn_dp == moe_tp*moe_ep"
        )


def resolve_frontier_parallelism_mapping(
    *,
    model_profile: str,
    tensor_parallel_size: int,
    data_parallel_size: int,
    enable_expert_parallel: bool,
) -> FrontierParallelismMapping:
    normalized_model_profile = str(model_profile).strip().lower()
    if normalized_model_profile not in {"dense", "moe"}:
        raise ValueError(
            f"Unsupported model_profile={model_profile!r}; expected 'dense' or 'moe'"
        )

    resolved_tp = _validate_positive("tensor_parallel_size", tensor_parallel_size)
    resolved_dp = _validate_positive("data_parallel_size", data_parallel_size)

    if normalized_model_profile == "dense":
        if enable_expert_parallel:
            raise ValueError("Dense models do not support expert parallel")
        return FrontierParallelismMapping(
            cluster_num_replicas=resolved_dp,
            attn_tensor_parallel_size=resolved_tp,
            attn_data_parallel_size=1,
            moe_tensor_parallel_size=1,
            moe_expert_parallel_size=1,
        )

    if enable_expert_parallel:
        mapping = FrontierParallelismMapping(
            cluster_num_replicas=1,
            attn_tensor_parallel_size=resolved_tp,
            attn_data_parallel_size=resolved_dp,
            moe_tensor_parallel_size=1,
            moe_expert_parallel_size=resolved_tp * resolved_dp,
        )
    else:
        mapping = FrontierParallelismMapping(
            cluster_num_replicas=1,
            attn_tensor_parallel_size=resolved_tp,
            attn_data_parallel_size=resolved_dp,
            moe_tensor_parallel_size=resolved_tp * resolved_dp,
            moe_expert_parallel_size=1,
        )

    validate_frontier_shared_parallel_domains(mapping)
    return mapping


def build_collective_sim_layout(
    *,
    mapping: FrontierParallelismMapping,
    num_pipeline_stages: int,
    domain: str,
) -> CollectiveSimParallelLayout:
    resolved_pp = _validate_positive("num_pipeline_stages", num_pipeline_stages)
    normalized_domain = str(domain).strip().lower()

    if normalized_domain == "attention":
        return CollectiveSimParallelLayout(
            tp=_validate_positive(
                "attn_tensor_parallel_size", mapping.attn_tensor_parallel_size
            ),
            cp=resolved_pp,
            dp=_validate_positive(
                "attention_dp",
                mapping.attn_data_parallel_size * mapping.cluster_num_replicas,
            ),
            ep=1,
        )

    if normalized_domain == "moe":
        return CollectiveSimParallelLayout(
            tp=_validate_positive(
                "moe_tensor_parallel_size", mapping.moe_tensor_parallel_size
            ),
            cp=resolved_pp,
            dp=_validate_positive("cluster_num_replicas", mapping.cluster_num_replicas),
            ep=_validate_positive(
                "moe_expert_parallel_size", mapping.moe_expert_parallel_size
            ),
        )

    raise ValueError(
        f"Unsupported domain={domain!r}; expected 'attention' or 'moe'"
    )


def build_collective_sim_physical_topology(
    *,
    cluster_total_devices: int,
    num_devices_per_node: int,
) -> CollectiveSimPhysicalTopology:
    resolved_cluster_total_devices = _validate_positive(
        "cluster_total_devices",
        cluster_total_devices,
    )
    resolved_num_devices_per_node = _validate_positive(
        "num_devices_per_node",
        num_devices_per_node,
    )
    if resolved_cluster_total_devices <= resolved_num_devices_per_node:
        return CollectiveSimPhysicalTopology(
            servers=1,
            gpus_per_server=resolved_cluster_total_devices,
        )
    if resolved_cluster_total_devices % resolved_num_devices_per_node != 0:
        raise ValueError(
            "collective-sim physical topology requires cluster_total_devices "
            f"{resolved_cluster_total_devices} to be divisible by node size "
            f"{resolved_num_devices_per_node}"
        )
    return CollectiveSimPhysicalTopology(
        servers=resolved_cluster_total_devices // resolved_num_devices_per_node,
        gpus_per_server=resolved_num_devices_per_node,
    )


@lru_cache(maxsize=None)
def _get_collective_sim_profile_gpus_per_server(scenario_profile: str) -> int:
    normalized_profile = str(scenario_profile).strip()
    if not normalized_profile:
        raise ValueError("scenario_profile must be non-empty")

    python_root = (
        Path(__file__).resolve().parents[1]
        / "cc_backend"
        / "backends"
        / "collective-sim"
        / "python"
    )
    python_root_str = str(python_root)
    if python_root_str not in sys.path:
        sys.path.insert(0, python_root_str)

    scenario_profiles = importlib.import_module("collective_sim_core.scenario_profiles")
    schema_module = importlib.import_module("collective_sim_core.schema")
    builder = scenario_profiles.get_scenario_profile_builder(normalized_profile)
    prototype = builder(
        servers=1,
        collective=schema_module.CollectiveConfig(),
        parallelism=schema_module.ParallelismConfig(),
    )
    return _validate_positive(
        "profile_gpus_per_server",
        int(prototype.cluster.gpus_per_server),
    )


def resolve_collective_sim_physical_topology(
    *,
    cluster_total_devices: int,
    num_devices_per_node: int,
    scenario_profile: str | None = None,
) -> CollectiveSimPhysicalTopology:
    resolved_cluster_total_devices = _validate_positive(
        "cluster_total_devices",
        cluster_total_devices,
    )
    normalized_profile = str(scenario_profile).strip() if scenario_profile else ""
    if not normalized_profile:
        return build_collective_sim_physical_topology(
            cluster_total_devices=resolved_cluster_total_devices,
            num_devices_per_node=num_devices_per_node,
        )

    profile_gpus_per_server = _get_collective_sim_profile_gpus_per_server(
        normalized_profile
    )
    if resolved_cluster_total_devices <= profile_gpus_per_server:
        return CollectiveSimPhysicalTopology(
            servers=1,
            gpus_per_server=resolved_cluster_total_devices,
        )
    if resolved_cluster_total_devices % profile_gpus_per_server != 0:
        raise ValueError(
            "collective-sim scenario profile requires cluster_total_devices "
            f"{resolved_cluster_total_devices} to be divisible by profile node size "
            f"{profile_gpus_per_server}"
        )
    return CollectiveSimPhysicalTopology(
        servers=resolved_cluster_total_devices // profile_gpus_per_server,
        gpus_per_server=profile_gpus_per_server,
    )
