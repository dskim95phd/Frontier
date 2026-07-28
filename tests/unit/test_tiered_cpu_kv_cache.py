from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.config.cpu_kv_cache_config import CPUKVCacheConfig
from frontier.config.config import SimulationConfig
from frontier.cpu_kv_cache_transfer import (
    AnalyticalCPUKVCacheTransferEngine,
)
from frontier.entities import Request
from frontier.entities.cpu_kv_cache_transfer_info import (
    CPUKVCacheOffloadInfo,
)
from frontier.kv_cache import CPUKVCacheManager
from frontier.kv_cache.replica_kv_cache_manager import ReplicaKVCacheManager
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import (
    ClusterSchedulerType,
    ClusterType,
    ReplicaSchedulerType,
)


BLOCK_SIZE = 16
BYTES_PER_BLOCK = 1024


def _request(
    *,
    session_id: int = 7,
    prefill_tokens: int = 48,
    decode_tokens: int = 8,
) -> Request:
    return Request(
        arrived_at=0.0,
        num_prefill_tokens=prefill_tokens,
        num_decode_tokens=decode_tokens,
        session_id=session_id,
    )


def _gpu_manager(num_blocks: int = 16) -> ReplicaKVCacheManager:
    return ReplicaKVCacheManager(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=num_blocks,
        enable_caching=True,
        caching_hash_algo="builtin",
        caching_key_mode="session",
        num_preallocate_tokens=0,
    )


def _cpu_manager(num_blocks: int = 16) -> CPUKVCacheManager:
    return CPUKVCacheManager(
        capacity_bytes=num_blocks * BYTES_PER_BLOCK,
        bytes_per_block=BYTES_PER_BLOCK,
    )


def _seed_gpu(
    manager: ReplicaKVCacheManager,
    *,
    session_id: int,
    blocks: int,
) -> None:
    request = _request(
        session_id=session_id,
        prefill_tokens=blocks * BLOCK_SIZE,
    )
    assert manager.allocate_slots(request, blocks * BLOCK_SIZE) is not None
    request._num_processed_tokens = blocks * BLOCK_SIZE
    manager.mark_blocks_computed(request)
    manager.free(request)


def _seed_cpu(
    manager: CPUKVCacheManager,
    *,
    session_id: int,
    blocks: int,
    generation: int = 1,
    time: float = 1.0,
) -> None:
    reservation = manager.reserve_offload(
        session_id=session_id,
        desired_frontier_blocks=blocks,
        generation=generation,
        time=time,
    )
    manager.commit_offload(reservation, time=time + 0.1)


def _scheduler(
    *,
    gpu_manager: ReplicaKVCacheManager | None = None,
    cpu_manager: CPUKVCacheManager | None = None,
) -> VLLMv1EngineReplicaScheduler:
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._cluster_type = ClusterType.PREFILL
    scheduler._replica_id = 3
    scheduler._dp_id = 1
    scheduler._config = SimpleNamespace(block_size=BLOCK_SIZE)
    scheduler._watermark_blocks = 0
    scheduler._kv_cache_manager = gpu_manager or _gpu_manager()
    scheduler._cpu_kv_cache_manager = cpu_manager or _cpu_manager()
    scheduler._cpu_kv_cache_transfer_engine = (
        AnalyticalCPUKVCacheTransferEngine(
            CPUKVCacheConfig(
                enable=True,
                capacity_bytes=16 * BYTES_PER_BLOCK,
                write_bandwidth_gbps=8.0,
                read_bandwidth_gbps=8.0,
                write_latency_ms=0.01,
                read_latency_ms=0.01,
            )
        )
    )
    scheduler._pending_cpu_restore_operations = {}
    scheduler._cpu_restore_waiting_requests = {}
    scheduler._cpu_restore_ready_plans = {}
    scheduler._pending_auxiliary_events = []
    scheduler._allocation_map = {}
    scheduler._num_allocated_blocks = 0
    scheduler._pending_kv_transfer_requests = set()
    scheduler._pending_prefill_export_kinds = {}
    scheduler._pending_cpu_offload_operations = {}
    scheduler._cpu_offload_generation_by_session = {}
    return scheduler


def _simulation_validation_subject() -> SimulationConfig:
    simulation = object.__new__(SimulationConfig)
    simulation.cpu_kv_cache_config = CPUKVCacheConfig(
        enable=True,
        capacity_bytes=BYTES_PER_BLOCK,
    )
    simulation.sys_arch = "pd-disaggregation"
    simulation.enable_parallel_clusters = False
    simulation.enable_thinking_mode = False
    simulation.cluster_config = SimpleNamespace(
        cluster_scheduler_config=SimpleNamespace(
            get_type=lambda: ClusterSchedulerType.STICKY_ROUND_ROBIN
        ),
        replica_scheduler_config=SimpleNamespace(
            get_type=lambda: ReplicaSchedulerType.VLLM_V1,
            enable_prefix_caching=True,
            prefix_caching_key_mode="session",
        ),
        prefill_replica_scheduler_config_type=None,
    )
    return simulation


def test_cpu_offload_simulation_validation_accepts_supported_pdd_mode() -> None:
    _simulation_validation_subject()._validate_cpu_kv_cache_config()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: setattr(cfg, "sys_arch", "co-location"), "pd-disaggregation"),
        (
            lambda cfg: setattr(cfg, "enable_parallel_clusters", True),
            "sequential PDD",
        ),
        (
            lambda cfg: setattr(
                cfg.cluster_config.cluster_scheduler_config,
                "get_type",
                lambda: ClusterSchedulerType.STICKY_LOR,
            ),
            "sticky_round_robin",
        ),
        (
            lambda cfg: setattr(
                cfg.cluster_config.replica_scheduler_config,
                "get_type",
                lambda: ReplicaSchedulerType.SARATHI,
            ),
            "vllm_v1",
        ),
        (
            lambda cfg: setattr(
                cfg.cluster_config.replica_scheduler_config,
                "enable_prefix_caching",
                False,
            ),
            "enable_prefix_caching",
        ),
        (
            lambda cfg: setattr(
                cfg.cluster_config.replica_scheduler_config,
                "prefix_caching_key_mode",
                "block_hash",
            ),
            "prefix_caching_key_mode",
        ),
        (
            lambda cfg: setattr(cfg, "enable_thinking_mode", True),
            "Thinking Mode",
        ),
    ],
)
def test_cpu_offload_simulation_validation_rejects_unsupported_modes(
    mutation, message: str
) -> None:
    simulation = _simulation_validation_subject()
    simulation.enable_thinking_mode = False
    mutation(simulation)

    with pytest.raises(ValueError, match=message):
        simulation._validate_cpu_kv_cache_config()


def test_tiered_plan_gpu_only_hit() -> None:
    gpu = _gpu_manager()
    _seed_gpu(gpu, session_id=7, blocks=2)
    scheduler = _scheduler(gpu_manager=gpu)

    plan = scheduler._build_tiered_prefix_plan(_request())

    assert plan.gpu_hit_blocks == 2
    assert plan.cpu_hit_blocks == 0
    assert plan.cpu_query_blocks == 1
    assert plan.hit_frontier_blocks == 2
    assert plan.num_new_tokens == 16


def test_tiered_plan_cpu_only_hit() -> None:
    cpu = _cpu_manager()
    _seed_cpu(cpu, session_id=7, blocks=2)
    scheduler = _scheduler(cpu_manager=cpu)

    plan = scheduler._build_tiered_prefix_plan(_request())

    assert plan.gpu_hit_blocks == 0
    assert plan.cpu_block_indices == [0, 1]
    assert plan.cpu_query_blocks == 3
    assert plan.hit_frontier_blocks == 2
    assert plan.num_new_tokens == 16


def test_tiered_plan_combines_gpu_prefix_and_cpu_suffix() -> None:
    gpu = _gpu_manager()
    cpu = _cpu_manager()
    _seed_gpu(gpu, session_id=7, blocks=2)
    _seed_cpu(cpu, session_id=7, blocks=4)
    scheduler = _scheduler(gpu_manager=gpu, cpu_manager=cpu)

    plan = scheduler._build_tiered_prefix_plan(
        _request(prefill_tokens=5 * BLOCK_SIZE)
    )

    assert sorted(plan.gpu_blocks_by_index) == [0, 1]
    assert plan.cpu_block_indices == [2, 3]
    assert plan.hit_frontier_blocks == 4
    assert plan.num_new_tokens == BLOCK_SIZE


def test_cpu_can_fill_gap_before_later_gpu_hit() -> None:
    gpu = _gpu_manager()
    cpu = _cpu_manager()
    _seed_gpu(gpu, session_id=7, blocks=3)
    _seed_cpu(cpu, session_id=7, blocks=3)
    block_zero = gpu.get_cached_block_for_key(("session", 7, 0))
    assert block_zero is not None
    cached = gpu.block_pool.cached_block_hash_to_block[block_zero.block_hash]
    del cached[block_zero.block_id]
    del gpu.block_pool.cached_block_hash_to_block[block_zero.block_hash]
    block_zero.reset_hash()
    scheduler = _scheduler(gpu_manager=gpu, cpu_manager=cpu)

    plan = scheduler._build_tiered_prefix_plan(
        _request(prefill_tokens=4 * BLOCK_SIZE)
    )

    assert plan.cpu_block_indices == [0]
    assert sorted(plan.gpu_blocks_by_index) == [1, 2]
    assert plan.hit_frontier_blocks == 3


def test_first_combined_miss_ignores_later_gpu_and_cpu_blocks() -> None:
    gpu = _gpu_manager()
    cpu = _cpu_manager()
    _seed_gpu(gpu, session_id=7, blocks=3)
    _seed_cpu(cpu, session_id=7, blocks=3)
    for block_index in (0,):
        block = gpu.get_cached_block_for_key(("session", 7, block_index))
        assert block is not None
        cached = gpu.block_pool.cached_block_hash_to_block[block.block_hash]
        del cached[block.block_id]
        del gpu.block_pool.cached_block_hash_to_block[block.block_hash]
        block.reset_hash()
        cpu_entry = cpu._sessions[7]
        cpu_block = cpu_entry.blocks.pop(block_index)
        cpu._free_block(cpu_block)
        cpu_entry.committed_frontier_blocks = 0
    scheduler = _scheduler(gpu_manager=gpu, cpu_manager=cpu)

    plan = scheduler._build_tiered_prefix_plan(_request())

    assert plan.hit_frontier_blocks == 0
    assert plan.gpu_blocks_by_index == {}
    assert plan.cpu_block_indices == []
    assert plan.num_new_tokens == 48


def test_fully_cached_prompt_demotes_final_block_before_restore() -> None:
    cpu = _cpu_manager()
    _seed_cpu(cpu, session_id=7, blocks=2)
    scheduler = _scheduler(cpu_manager=cpu)

    plan = scheduler._build_tiered_prefix_plan(
        _request(prefill_tokens=2 * BLOCK_SIZE)
    )

    assert plan.hit_frontier_blocks == 1
    assert plan.cpu_block_indices == [0]
    assert plan.num_new_tokens == BLOCK_SIZE


def test_restore_reserves_gpu_pages_and_publishes_only_on_completion() -> None:
    cpu = _cpu_manager()
    _seed_cpu(cpu, session_id=7, blocks=2)
    scheduler = _scheduler(cpu_manager=cpu)
    request = _request()
    plan = scheduler._build_tiered_prefix_plan(request)

    assert scheduler._begin_cpu_kv_cache_restore(
        request=request, plan=plan, time=2.0
    )
    restore_info = scheduler._pending_cpu_restore_operations[request.id]
    assert scheduler._kv_cache_manager.get_num_blocks_for_request(request) == 3
    assert all(
        block.block_hash is None
        for block in restore_info.restore_blocks_by_index.values()
    )
    assert all(block.pin_count == 1 for block in restore_info.cpu_lease.blocks)

    scheduler.complete_cpu_kv_cache_restore(
        restore_info.timing.end_time, restore_info
    )

    assert all(
        block.block_hash == ("session", 7, index)
        for index, block in restore_info.restore_blocks_by_index.items()
    )
    assert all(block.pin_count == 0 for block in restore_info.cpu_lease.blocks)
    assert request.cpu_prefix_cache_hit_blocks == 2
    assert request.cpu_prefix_cache_restored_blocks == 2
    assert request.cpu_prefix_cache_restored_tokens == 32
    assert request.num_prefill_tokens_cached == 32
    assert request.id in scheduler._cpu_restore_ready_plans


def test_restore_allocation_failure_does_not_pin_or_attach_blocks() -> None:
    cpu = _cpu_manager()
    _seed_cpu(cpu, session_id=7, blocks=2)
    gpu = _gpu_manager(num_blocks=1)
    scheduler = _scheduler(gpu_manager=gpu, cpu_manager=cpu)
    request = _request()
    plan = scheduler._build_tiered_prefix_plan(request)

    assert not scheduler._begin_cpu_kv_cache_restore(
        request=request, plan=plan, time=2.0
    )
    assert gpu.get_num_blocks_for_request(request) == 0
    assert cpu.reserved_blocks == 0
    assert all(block.pin_count == 0 for block in cpu._blocks.values())
    assert cpu.cpu_query_blocks == 0
    assert cpu.cpu_hit_blocks == 0


def test_restore_atomically_reserves_prompt_suffix_against_competing_request() -> None:
    cpu = _cpu_manager()
    _seed_cpu(cpu, session_id=7, blocks=2)
    gpu = _gpu_manager(num_blocks=4)
    scheduler = _scheduler(gpu_manager=gpu, cpu_manager=cpu)
    restoring = _request()
    plan = scheduler._build_tiered_prefix_plan(restoring)

    assert scheduler._begin_cpu_kv_cache_restore(
        request=restoring, plan=plan, time=2.0
    )
    assert gpu.get_num_blocks_for_request(restoring) == 3

    competing = _request(session_id=8, prefill_tokens=32)
    assert not gpu.can_allocate_slots(competing, 32)
    assert gpu.get_num_blocks_for_request(restoring) == 3


@pytest.mark.parametrize("decode_first", [True, False])
def test_prefill_export_barrier_releases_after_both_branches(
    decode_first: bool,
) -> None:
    cpu = _cpu_manager()
    scheduler = _scheduler(cpu_manager=cpu)
    request = _request(prefill_tokens=32)
    reservation = cpu.reserve_offload(
        session_id=7,
        desired_frontier_blocks=2,
        generation=1,
        time=1.0,
    )
    timing = scheduler._cpu_kv_cache_transfer_engine.schedule(
        direction="d2h",
        size_bytes=2 * BYTES_PER_BLOCK,
        submitted_at=1.0,
    )
    info = CPUKVCacheOffloadInfo(
        request=request,
        replica_id=3,
        dp_id=1,
        reservation=reservation,
        timing=timing,
        desired_frontier_blocks=2,
    )
    scheduler._pending_kv_transfer_requests = {request.id}
    scheduler._pending_prefill_export_kinds = {
        request.id: {"decode_transfer", "cpu_offload"}
    }
    scheduler._pending_cpu_offload_operations = {request.id: info}
    scheduler._allocation_map = {request.id: 2}
    freed = []

    def free_resources(freed_request: Request) -> None:
        freed.append(freed_request.id)
        scheduler._allocation_map.pop(freed_request.id, None)

    scheduler._free_request_resources = free_resources

    if decode_first:
        scheduler.record_decode_kv_transfer_completion(1.0, [request])
        scheduler.complete_kv_transfer_for_requests([request])
        assert freed == []
        scheduler.complete_cpu_kv_cache_offload(timing.end_time, info)
        assert info.source_gpu_hold_time_ms == pytest.approx(
            max(0.0, (timing.end_time - 1.0) * 1e3)
        )
    else:
        scheduler.complete_cpu_kv_cache_offload(timing.end_time, info)
        assert freed == []
        scheduler.complete_kv_transfer_for_requests([request])
        assert info.source_gpu_hold_time_ms == 0.0

    assert freed == [request.id]
    assert scheduler._allocation_map == {}
    assert scheduler._pending_prefill_export_kinds == {}
    assert cpu.get_committed_frontier(7) == 2


def test_terminal_cpu_snapshot_does_not_add_export_barrier_branch() -> None:
    cpu = _cpu_manager()
    _seed_cpu(cpu, session_id=7, blocks=2)
    scheduler = _scheduler(cpu_manager=cpu)
    request = _request(prefill_tokens=32)
    scheduler._pending_kv_transfer_requests = {request.id}
    scheduler._pending_prefill_export_kinds = {
        request.id: {"decode_transfer"}
    }
    scheduler._allocation_map = {request.id: 2}
    freed = []
    scheduler._free_request_resources = lambda req: (
        freed.append(req.id),
        scheduler._allocation_map.pop(req.id, None),
    )

    assert scheduler.start_cpu_kv_cache_offload(
        time=2.0, request=request
    ) is None
    scheduler.complete_kv_transfer_for_requests([request])

    assert freed == [request.id]
    assert scheduler._pending_prefill_export_kinds == {}


def test_cpu_offload_generation_is_monotonic_across_session_requests() -> None:
    cpu = _cpu_manager()
    scheduler = _scheduler(cpu_manager=cpu)
    first = _request(session_id=7, prefill_tokens=32)
    second = _request(session_id=7, prefill_tokens=64)

    first_event = scheduler.start_cpu_kv_cache_offload(time=1.0, request=first)
    second_event = scheduler.start_cpu_kv_cache_offload(time=1.1, request=second)

    assert first_event is not None
    assert second_event is not None
    first_info = first_event._offload_info
    second_info = second_event._offload_info
    assert first_info.reservation.generation == 1
    assert second_info.reservation.generation == 2
    cpu.commit_offload(second_info.reservation, time=2.0)
    cpu.commit_offload(first_info.reservation, time=3.0)
    assert cpu.stale_generation_completions == 1


def test_restore_reservation_preserves_scheduler_watermark() -> None:
    cpu = _cpu_manager(num_blocks=10)
    _seed_cpu(cpu, session_id=7, blocks=8)
    scheduler = _scheduler(
        gpu_manager=_gpu_manager(num_blocks=10),
        cpu_manager=cpu,
    )
    scheduler._watermark_blocks = 2
    request = _request(session_id=7, prefill_tokens=10 * BLOCK_SIZE)
    plan = scheduler._build_tiered_prefix_plan(request)

    assert not scheduler._begin_cpu_kv_cache_restore(
        request=request,
        plan=plan,
        time=2.0,
    )
    assert scheduler._kv_cache_manager.num_used_blocks == 0
    assert scheduler._pending_cpu_restore_operations == {}
    assert scheduler._pending_auxiliary_events == []
    assert all(block.pin_count == 0 for block in cpu._blocks.values())


def test_restore_reservation_succeeds_at_watermark_boundary() -> None:
    cpu = _cpu_manager(num_blocks=10)
    _seed_cpu(cpu, session_id=7, blocks=6)
    scheduler = _scheduler(
        gpu_manager=_gpu_manager(num_blocks=10),
        cpu_manager=cpu,
    )
    scheduler._watermark_blocks = 2
    request = _request(session_id=7, prefill_tokens=8 * BLOCK_SIZE)
    plan = scheduler._build_tiered_prefix_plan(request)

    assert scheduler._begin_cpu_kv_cache_restore(
        request=request,
        plan=plan,
        time=2.0,
    )
    assert scheduler._kv_cache_manager.num_used_blocks == 8
    assert scheduler._kv_cache_manager.block_pool.get_num_free_blocks() == 2


def test_restore_schedule_failure_rolls_back_gpu_and_cpu_state() -> None:
    cpu = _cpu_manager()
    _seed_cpu(cpu, session_id=7, blocks=2)
    scheduler = _scheduler(cpu_manager=cpu)
    request = _request(session_id=7, prefill_tokens=48)
    plan = scheduler._build_tiered_prefix_plan(request)
    scheduler._cpu_kv_cache_transfer_engine = SimpleNamespace(
        schedule=lambda **_: (_ for _ in ()).throw(RuntimeError("h2d fault"))
    )

    with pytest.raises(RuntimeError, match="h2d fault"):
        scheduler._begin_cpu_kv_cache_restore(
            request=request,
            plan=plan,
            time=2.0,
        )

    assert scheduler._kv_cache_manager.num_used_blocks == 0
    assert scheduler._pending_cpu_restore_operations == {}
    assert all(block.pin_count == 0 for block in cpu._blocks.values())


def test_restore_cancellation_cleans_up_exactly_once() -> None:
    cpu = _cpu_manager()
    _seed_cpu(cpu, session_id=7, blocks=2)
    scheduler = _scheduler(cpu_manager=cpu)
    request = _request(session_id=7, prefill_tokens=48)
    plan = scheduler._build_tiered_prefix_plan(request)
    assert scheduler._begin_cpu_kv_cache_restore(
        request=request,
        plan=plan,
        time=2.0,
    )

    assert scheduler.cancel_cpu_kv_cache_restore(request.id, time=2.1)
    assert not scheduler.cancel_cpu_kv_cache_restore(request.id, time=2.2)
    assert scheduler._kv_cache_manager.num_used_blocks == 0
    assert scheduler._pending_cpu_restore_operations == {}
    assert scheduler._cpu_restore_waiting_requests == {}
    assert scheduler._pending_auxiliary_events == []
    assert all(block.pin_count == 0 for block in cpu._blocks.values())


def test_offload_schedule_failure_aborts_cpu_reservation() -> None:
    cpu = _cpu_manager()
    scheduler = _scheduler(cpu_manager=cpu)
    request = _request(session_id=7, prefill_tokens=32)
    scheduler._cpu_kv_cache_transfer_engine = SimpleNamespace(
        schedule=lambda **_: (_ for _ in ()).throw(RuntimeError("d2h fault"))
    )

    with pytest.raises(RuntimeError, match="d2h fault"):
        scheduler.start_cpu_kv_cache_offload(time=1.0, request=request)

    assert cpu.reserved_blocks == 0
    assert cpu._reservations == {}
    assert scheduler._pending_cpu_offload_operations == {}


def test_offload_abort_closes_only_cpu_barrier_branch() -> None:
    cpu = _cpu_manager()
    scheduler = _scheduler(cpu_manager=cpu)
    request = _request(session_id=7, prefill_tokens=32)
    scheduler._pending_kv_transfer_requests = {request.id}
    scheduler._pending_prefill_export_kinds = {
        request.id: {"decode_transfer"}
    }
    scheduler._allocation_map = {request.id: 2}
    event = scheduler.start_cpu_kv_cache_offload(time=1.0, request=request)
    assert event is not None

    assert scheduler.abort_cpu_kv_cache_offload(event._offload_info)
    assert not scheduler.abort_cpu_kv_cache_offload(event._offload_info)
    assert cpu.reserved_blocks == 0
    assert scheduler._pending_prefill_export_kinds == {
        request.id: {"decode_transfer"}
    }
