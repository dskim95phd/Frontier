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
  include/frontier/
    config.h
    ids.h
    event.h
    request.h
    batch.h
    cluster.h
    scheduler.h
    kv_cache.h
    analytical_model.h
    metrics.h
    simulator.h
  src/
    event_queue.cc
    scheduler.cc
    kv_cache.cc
    analytical_model.cc
    simulator.cc
    workload.cc
    metrics.cc
    main.cc
  tests/
```

The directory structure is a target layout, not a requirement to create every
file before the corresponding milestone is implemented.

### Core data structures

| Concept | C++ representation | Reason |
| --- | --- | --- |
| Request | `std::vector<Request>` plus `RequestId` | Stable IDs avoid Python-style object graphs and make events compact. |
| Event | value type `{time, sequence, type, payload}` in a min-heap | Deterministic DES ordering without event-object allocation. |
| Batch | ephemeral value containing `std::vector<RequestId>` and scheduled token counts | It has no need to own request lifetime. |
| Scheduler queues | `std::deque<RequestId>` for waiting, vectors for running work | Matches vLLM-style admission and minimizes allocations. |
| KV blocks | contiguous block pool plus per-request block counts | Fast ordinary allocation/free. |
| Session cache | `unordered_map<SessionBlockKey, BlockId>` plus intrusive/LRU eviction list | O(1) lookup while preserving an explicit eviction policy. |
| Cluster routing | `SessionId -> ReplicaTarget` affinity map | Necessary because prefix cache state is target-local. |

Events should reference IDs and a generation/epoch number rather than owning
`Request` or `Batch` pointers. A stale event is ignored when its stored epoch
does not match the request's current epoch.

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

1. **Analytical model and simulator foundation**
   - Establish the CMake build, JSON configuration reader, deterministic event
     queue, and basic request/batch/metrics types.
   - Port only the analytical roofline and communication formulas needed by the
     selected dense model and NVL72 topology.
   - Keep formulas side-effect-free and test them independently of the DES.

2. **Co-location scheduler**
   - Implement vLLM-style waiting/running queues, token budgeting, KV block
     allocation, batching, decode iterations, and preemption rules.
   - Match request completion count, TTFT, E2E latency, and block accounting
     for deterministic Python workloads.

3. **Sequential PDD**
   - Add the prefill-to-decode handoff and analytical KV transfer.
   - Validate baseline PDD completion ordering, TTFT/E2E metrics, and KV
     transfer counts against Python before adding cache behavior.

4. **Session prefix cache**
   - Add session affinity, complete-block GPU cache lookup, allocation,
     eviction, and metrics to the validated co-location and PDD paths.
   - Validate repeated turns in one session, interleaved sessions, and
     multiple replica/DP targets.

5. **CPU KV-cache tiering**
   - Add analytical GPU-to-CPU offload and CPU-to-GPU restore events only after
     the GPU-resident session cache has parity.
   - Add a bounded CPU block pool, session-aware CPU cache lookup/eviction,
     restore admission, and tiered-cache metrics.
   - Validate GPU-only hit, CPU-cache hit plus restore, miss, eviction, and
     capacity-pressure cases independently from PDD.

6. **Hardening and performance**
   - Add differential/regression tests, fixed seeds, profiling, allocation
     reduction, and benchmark workloads.

## Parity Contract

Python remains the oracle until a C++ case is accepted. Each deterministic
test compares normalized inputs and outputs, not implementation internals.

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
- It rejects all excluded modes with clear errors.
- It passes the agreed Python-vs-C++ parity suite for dense session-prefix
  workloads, including multi-target affinity and PDD handoff cases.
- Profiling workflows continue to run from the existing Python code without a
  C++ dependency.
