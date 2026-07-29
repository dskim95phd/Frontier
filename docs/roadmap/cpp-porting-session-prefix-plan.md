# C++ Simulator Porting Plan: Session Prefix Cache MVP

## Goal

Build a fast C++ discrete-event simulation core for Frontier while keeping the
existing Python profiling tools and the Python simulator as the behavioral
oracle during migration.

The first C++ milestone supports only the serving path required for
co-location and sequential PDD (`pd-disaggregation`), with vLLM-style
continuous batching, analytical execution/communication modeling, and
session-scoped prefix caching.

This is a new implementation, not a line-by-line translation of the Python
object model. The implementation should use compact ID-based state and a
single deterministic event queue.

The port should nevertheless preserve the Python simulator's domain boundaries
and state-machine semantics where they define observable behavior. In
particular, the scheduler and KV-cache implementation should have an explicit
semantic mapping to the corresponding Python queues, block states,
reservations, leases, and metrics. Python object references become stable C++
IDs and contiguous storage; the behavior is not independently redesigned.

## MVP Scope

### Included

- `co-location` with one monolithic cluster.
- Sequential `pd-disaggregation` with a `PREFILL` cluster, a unified `DECODE`
  cluster, and analytical KV-cache-transfer events.
- One vLLM-v1-style scheduler:
  - continuous batching;
  - chunked prefill where required by the current model;
  - token budget and KV-block admission;
  - preemption only when needed for vLLM-compatible memory admission.
- Analytical execution-time prediction, including the current NVL72-oriented
  model and the selected analytical communication backend.
- Basic workloads:
  - synthetic fixed/uniform request lengths and arrivals;
  - CSV trace replay with `arrived_at`, `num_prefill_tokens`,
    `num_decode_tokens`, and optional `session_id` / `session_turn_index`.
- GPU-resident, session-scoped prefix caching:
  - `prefix_caching_key_mode=session`;
  - complete-block lookup/allocation and eviction;
  - per-request and system hit/query metrics;
  - affinity to the `(replica_id, dp_id)` target that owns the cache.
- CPU KV-cache tiering after the initial GPU-cache validation:
  - analytical CPU offload and restore transfer events;
  - CPU-resident session-prefix blocks and capacity/eviction management;
  - GPU/CPU hit, restore, queue-time, and transfer-time metrics.
- Machine-readable JSON/CSV inputs and metrics outputs sufficient for
  Python-vs-C++ parity tests.

### Explicitly excluded from the MVP

- `pd-af-disaggregation`, M2N transfers, `DECODE_ATTN` / `DECODE_FFN`, and
  ping-pong pipeline events.
- Speculative decoding / MTP and all speculative batch/request state.
- Thinking Mode and multi-round requests. A request has exactly one prefill
  phase and one decode phase.
- Profile-trained execution-time predictors, predictor training, and runtime
  profiling. Profiling remains Python-only.
- Other replica schedulers (SGLang, Sarathi, Orca, SJ2Q, etc.), other
  communication backends, parallel-cluster execution, and W&B/plot export.
- `prefix_caching_key_mode=block_hash` and explicit request
  `block_hash_ids`. The C++ MVP supports only session-derived prefix keys.

## Semantics of Session Prefix Caching

Session prefix caching reuses prefill KV blocks across separate requests in a
conversation. It is independent of Thinking Mode: consecutive requests can
share a `session_id` while each remains a normal single-round request.

Example:

```text
session 7, turn 0: prompt A              -> populate complete KV blocks
session 7, turn 1: prompt A + B          -> reuse A blocks, compute B blocks
session 7, turn 2: prompt A + B + C      -> reuse A+B blocks, compute C blocks
```

The cache is local to a scheduling target. Therefore, when
`num_replicas * data_parallel_size > 1`, all requests for a cache-owning
session must be routed to the same `(replica_id, dp_id)`. Round-robin affinity
is the baseline policy. `sticky_lor` is not part of the PDD MVP.

The initial cache key is conceptually:

```text
(session_id, complete_block_index)
```

The cache must never treat blocks from different session IDs as equivalent.
Partial final blocks are not cache hits; they are computed normally and become
cacheable only after a complete block is available.

## Proposed C++ Layout

```text
cpp/
  CMakeLists.txt
  frontier/
    config/
      config.h
      config.cc
    core/
      ids.h
      event.h
      event_queue.h
      event_queue.cc
    entities/
      request.h
      request.cc
      batch.h
      cluster.h
    scheduler/
      scheduler.h
      vllm_v1_scheduler.h
      vllm_v1_scheduler.cc
    kv_cache/
      kv_cache_block.h
      block_pool.h
      block_pool.cc
      prefix_cache.h
      prefix_cache.cc
      cpu_cache.h
      cpu_cache.cc
      tiered_prefix_plan.h
    execution_time_predictor/
      analytical_model.h
      analytical_model.cc
    cc_backend/
      analytical_model.h
      analytical_model.cc
    kv_cache_transfer/
      analytical_transfer.h
      analytical_transfer.cc
    request_generator/
      workload.h
      synthetic.cc
      csv_trace.cc
    metrics/
      metrics.h
      metrics.cc
    simulator/
      simulator.h
      simulator.cc
    main.cc
  tests/
    core/
    scheduler/
    kv_cache/
    analytical_model/
    parity/
```

Headers and implementation files live together in functional directories.
This mirrors the navigational boundaries of the Python package without copying
its inheritance hierarchy or object graph. A separate public `include/` tree is
unnecessary while the MVP is an internal executable rather than a distributed
C++ library.

The directory structure is a target layout, not a requirement to create every
file before the corresponding milestone is implemented. Only directories
needed by an implemented vertical slice should be created.

### Core data structures

| Concept | C++ representation | Reason |
| --- | --- | --- |
| Request | `std::vector<Request>` plus `RequestId` | Stable IDs avoid Python-style object graphs and make events compact. |
| Event | value type `{time, sequence, type, payload}` in a min-heap | Deterministic DES ordering without event-object allocation. |
| Batch | ephemeral value containing `std::vector<RequestId>` and scheduled token counts | It has no need to own request lifetime. |
| Scheduler queues | `std::deque<RequestId>` for waiting, vectors for running work | Matches vLLM-style admission and minimizes allocations. |
| GPU KV block | contiguous block pool of `{BlockId, ref_count, optional<SessionBlockKey>, prev_free, next_free}` | Preserves Python `KVCacheBlock` semantics with ID-based links. |
| GPU free blocks | intrusive queue over block IDs | Mirrors `FreeKVCacheBlockQueue` and provides allocation/eviction order parity. |
| Session cache | `std::unordered_map<SessionBlockKey, std::set<BlockId>>` plus the intrusive free/eviction queue | Preserves the Python ability to associate more than one physical block with a key and deterministically select the lowest block ID. |
| CPU KV block | `{BlockId, FREE/RESERVED/COMMITTED, session, block_index, pin_count, reservation_id}` | Direct semantic mapping of Python `CPUKVCacheBlock`. |
| CPU session state | session entry plus offload reservations and restore leases referenced by IDs | Preserves committed frontier, capacity reservation, pinning, cancellation, and generation behavior. |
| Cluster routing | `SessionId -> ReplicaTarget` affinity map | Necessary because prefix cache state is target-local. |

Events should reference IDs and a generation/epoch number rather than owning
`Request` or `Batch` pointers. A stale event is ignored when its stored epoch
does not match the request's current epoch.

The initial implementation should port the observable Python state transitions,
not replace them with a new cache policy:

- GPU blocks retain Python-compatible reference counting, cache-key assignment,
  free-list ordering, touch, release, and eviction behavior.
- CPU blocks retain `FREE`, `RESERVED`, and `COMMITTED` states.
- CPU session entries retain committed/reserved frontiers, generation checks,
  active reservation IDs, restore pins, and session-level eviction behavior.
- `CPUOffloadReservation`, `CPURestoreLease`, and `TieredPrefixPlan` remain
  explicit value types. Their references to requests and blocks use IDs.

## Input, Output, and Determinism Contracts

Freeze a small, versioned C++ input schema before implementing scheduler
behavior. Do not reproduce the full flattened Python dataclass CLI.

- A normalized JSON configuration contains a required `schema_version`.
- CSV trace input supports `arrived_at`, `num_prefill_tokens`,
  `num_decode_tokens`, and optional `session_id` / `session_turn_index`.
  `block_hash_ids` is intentionally unsupported.
- Invalid nonfinite/negative arrival times and nonfinite/nonpositive token
  counts are rejected rather than clipped.
- Output JSON/CSV schemas define timestamp and latency units, float
  serialization precision, optional-field behavior, and the canonical TTFT
  definition.
- Unsupported architectures, key modes, and feature combinations fail fast
  with clear errors.

The event queue ordering key is `(time, sequence)`, where `sequence` is a
monotonically increasing event-creation ID. Floating-point event times are
compared exactly for queue ordering; numerical tolerances apply only to
reported analytical metrics. Equal-time ordering fixtures must verify the same
creation and processing order as Python.

## Architecture Simplification

Use one global event queue for both architectures.

```text
co-location:
  arrival -> schedule -> batch end -> request completion

sequential PDD:
  arrival -> PREFILL schedule -> PREFILL batch end
          -> KV transfer complete -> DECODE schedule -> decode iterations
          -> request completion
```

Do not reproduce `ClusterSimulator` threads or Python's parallel-cluster
control path. Public PDD already requires sequential execution, so a single
queue both matches the release contract and removes synchronization overhead.

## Implementation Milestones

1. **Contracts, parity harness, and simulator foundation**
   - Establish the CMake build, JSON configuration reader, deterministic event
     queue, and basic request/batch/metrics types.
   - Freeze the normalized input and output schemas and add fixture validation.
   - Add a differential runner that invokes Python and C++ with the same input,
     normalizes their outputs, and reports field-level differences.
   - Add equal-time event-order and basic request-lifecycle golden fixtures.
   - Port only the analytical roofline and communication formulas needed by the
     selected dense model and NVL72 topology.
   - Keep formulas side-effect-free and test them independently of the DES.
   - Follow the detailed
     [Step 1 foundation plan](cpp-porting-step1-foundation.md).

2. **Co-location scheduler (complete)**
   - Implement and validate in small vertical slices:
     1. one fixed-latency request;
     2. multiple requests and continuous batching;
     3. token budgeting and ordinary KV-block admission;
     4. decode iterations;
     5. chunked prefill; and
     6. memory-pressure preemption.
   - Require request completion count, TTFT, E2E latency, queue state, and block
     accounting parity at each slice before proceeding.
   - Follow the detailed
     [Step 2 co-location scheduler plan](cpp-porting-step2-colocation-scheduler.md).

2.5. **Dense co-location event and parallelism parity (complete)**
   - Replace the flattened Step 2 event loop with the production Python
     global/cluster/replica/stage event boundaries.
   - Make the existing scheduler ownership hierarchy functional.
   - Support dense TP, PP, DP, and multiple replicas with per-target scheduler,
     KV, in-flight batch, and stage state.
   - Refactor whole-batch analytical timing into per-stage compute, TP
     collective, and PP transfer timing.
   - Require target, event, scheduler, batch-stage, and request-metric parity
     before beginning PDD.
   - Follow the detailed
     [Step 2.5 dense co-location event and parallelism plan](cpp-porting-step2-5-colocation-parallelism-events.md).

3. **Sequential PDD**
   - Add the prefill-to-decode handoff and analytical KV transfer.
   - Validate baseline PDD completion ordering, TTFT/E2E metrics, and KV
     transfer counts against Python before adding cache behavior.

4. **Session prefix cache**
   - Add session affinity, complete-block GPU cache lookup, allocation,
     eviction, and metrics to the validated co-location and PDD paths.
   - Port the Python GPU block-pool, reference-counting, free-list, and
     cache-admission semantics using IDs rather than object references.
   - Validate repeated turns in one session, interleaved sessions, and
     multiple replica/DP targets.
   - Reject `prefix_caching_key_mode=block_hash`.

5. **CPU KV-cache tiering**
   - Add analytical GPU-to-CPU offload and CPU-to-GPU restore events only after
     the GPU-resident session cache has parity.
   - Add a bounded CPU block pool, session-aware CPU cache lookup/eviction,
     restore admission, and tiered-cache metrics.
   - Preserve the Python reservation, lease, pin, generation, cancellation,
     and committed-frontier state transitions.
   - Validate GPU-only hit, CPU-cache hit plus restore, miss, eviction, and
     capacity-pressure cases independently from PDD.

6. **Hardening and performance**
   - Add differential/regression tests, fixed seeds, profiling, allocation
     reduction, and benchmark workloads.

## Parity Contract

Python remains the oracle until a C++ case is accepted. Each deterministic
test compares normalized inputs and outputs, not implementation internals.
The parity harness is introduced in milestone 1 and is a gate for every later
milestone; it is not deferred to final hardening.

Required checks:

- completed request IDs and completion count;
- session-to-target affinity;
- GPU/CPU prefix-cache query and hit blocks, cached prefill tokens, restores,
  queue/transfer times, and evictions;
- KV block allocation never exceeds capacity;
- PDD KV-transfer count and completion ordering;
- TTFT, E2E, and throughput within a documented numerical tolerance for the
  analytical model;
- byte-for-byte equal event ordering where all event times are equal.

The existing Python references for the session-cache behavior are primarily:

- `frontier/kv_cache/base_kv_cache_manager.py`;
- `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`;
- `frontier/entities/request.py`;
- `tests/e2e/test_session_prefix_cache_runtime.py`;
- `tests/unit/test_session_incremental_isl.py`.

## Definition of Done for the MVP

- A C++ executable runs deterministic co-location and sequential PDD inputs.
- It uses only the analytical predictor and supports session-scoped GPU and
  CPU-tiered prefix caching, including analytical offload/restore.
- It supports only `prefix_caching_key_mode=session` and clearly rejects
  `block_hash`.
- It rejects all excluded modes with clear errors.
- It passes the agreed Python-vs-C++ parity suite for dense session-prefix
  workloads, including multi-target affinity and PDD handoff cases.
- Profiling workflows continue to run from the existing Python code without a
  C++ dependency.
