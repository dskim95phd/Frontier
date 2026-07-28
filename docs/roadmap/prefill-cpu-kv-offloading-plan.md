# Prefill-Side CPU KV-Cache Offloading Implementation Plan

## Status

- **State:** Implemented and validated
- **Target release:** `pre-release-v0.2`
- **Validation date:** 2026-07-27
- **Primary architecture:** Sequential `pd-disaggregation`
- **Primary scheduler:** `vllm_v1`
- **Cache identity mode:** `session`
- **CPU cache location:** Local to each prefill `(replica_id, dp_id)` cache target

## Summary

This proposal adds a CPU DRAM tier below the existing prefill GPU KV cache.
After a request finishes prefill, Frontier copies the full, reusable prefill
block frontier to CPU DRAM on the same prefill cache target. A later request
from the same session can restore CPU-resident blocks to GPU before computing
the remaining prompt suffix.

The MVP deliberately does not offload KV blocks created while decoding. In
sequential PDD, the decode cluster continues to release its request KV state
after decoding finishes. On the next turn, the prior decode output is part of
the new prompt and is recomputed by the prefill cluster. Once that prefill
finishes, those tokens become part of the newer prefill-side CPU snapshot.

The resulting behavior is an extension of Frontier's current PDD session
prefix-cache contract:

```text
turn 1:
  prefill computes A
  A -> decode GPU
  A -> prefill-side CPU DRAM
  decode produces B
  B is not returned to prefill or CPU

turn 2 prompt:
  A + B + new input C

turn 2 execution:
  restore A from prefill GPU or CPU
  recompute B + C in prefill
  store the new A + B + C prefill frontier in prefill-side CPU DRAM
```

Decode-to-CPU offload, decode-to-prefill KV return, and a shared remote CPU
cache are follow-up features and require separate topology and transfer-cost
contracts.

## Motivation

The current session prefix cache is local to a GPU cache target. When a
request releases its blocks, full cached blocks remain reusable while their
physical pages have not been selected for reuse. Under GPU memory pressure,
those pages are eventually evicted.

Multi-turn sessions often have enough time between turns for their GPU blocks
to be evicted. A larger CPU DRAM tier can preserve the prefill-created prefix
without requiring the simulator to assume that decode-created KV state is
returned to the prefill cluster.

The feature must model both sides of the tradeoff:

- lower future prefill compute from CPU cache hits;
- GPU-to-CPU offload latency and bandwidth;
- CPU-to-GPU restore latency and bandwidth;
- source GPU pages retained while an asynchronous copy is in flight;
- finite CPU capacity and session-aware eviction; and
- GPU capacity consumed by restored blocks.

## Decisions

The MVP makes the following explicit decisions.

1. **Only prefill-side KV is offloaded.**
   Decode-created KV is not copied to CPU.
2. **The CPU cache is local to a prefill cache target.**
   Each `(prefill_replica_id, dp_id)` owns an independent CPU store.
3. **Session affinity remains required.**
   CPU offloading does not replace `sticky_round_robin`.
4. **Only full blocks are reusable and offloadable.**
   Partial blocks are recomputed on a later turn.
5. **The CPU store preserves a contiguous prefix per session.**
   It never reports a hit after the first missing block.
6. **CPU restore completes before prefill execution begins for that request.**
7. **PREFILL-to-DECODE and PREFILL-to-CPU copies may overlap.**
   Decode can start after its own KV transfer completes; it does not wait for
   the CPU copy.
8. **Prefill GPU pages are released only after all required exports finish.**
   For PDD with CPU offloading enabled, that means both the decode transfer and
   CPU offload have reached a terminal state.
9. **Released GPU cache pages remain normally evictable.**
   CPU offload does not force immediate deletion of the GPU cache key.
10. **The CPU snapshot is monotonic for an append-only session.**
    An older asynchronous completion must not shrink or overwrite a newer
    committed frontier.

## Goals

1. Preserve prefill-created session KV blocks in finite prefill-side CPU DRAM.
2. Restore a contiguous CPU prefix to GPU with explicit latency and bandwidth.
3. Combine GPU hits, CPU hits, and prefill recomputation in one admission plan.
4. Preserve current sequential PDD semantics for prior decode output tokens.
5. Model the GPU residency extension caused by an in-flight CPU offload.
6. Evict CPU blocks using a session-aware policy that preserves useful
   prefixes.
7. Export request-level and system-level tiered-cache metrics.
8. Keep CPU offloading disabled by default and preserve existing behavior when
   it is disabled.

## Non-goals

The MVP will not model:

- decode-to-CPU KV offload;
- decode-to-prefill KV return;
- reuse of decode-created KV before it has been recomputed by prefill;
- a cluster-wide or remote CPU KV-cache pool;
- cache migration between prefill replicas or DP lanes;
- CPU compression, decompression, or KV dtype conversion;
- NVMe or other storage tiers;
- branching conversations under one `session_id`;
- context truncation or middle-of-prompt edits;
- cross-session content sharing;
- AFD / `pd-af-disaggregation`;
- closed-loop turn generation; or
- Thinking Mode rounds sharing one live `Request` object; or
- detailed PCIe, NUMA, DMA-engine, and CPU memory-controller contention.

Co-location support can reuse the store and restore machinery later, but is
not required for the sequential PDD MVP acceptance criteria.

## Baseline PDD Semantics

Without CPU offloading, sequential PDD behaves as follows:

1. The prefill cluster computes the current prompt KV.
2. Full completed blocks may be published in the prefill GPU prefix cache.
3. The full current prompt KV is transferred to the decode cluster.
4. The prefill request pages are released after the decode transfer completes.
5. The decode cluster extends the KV state while producing output tokens.
6. The decode request pages are released at request completion.
7. Decode-created KV is not returned to the prefill cluster.

For a session:

```text
turn 1: prompt A, output B
turn 2: prompt A + B + C
```

the first follow-up turn can reuse `A` only if it is still present in the
prefill GPU cache. The prefill cluster must recompute `B + C`. CPU offloading
changes the residency of `A`, not this architectural boundary.

With the proposed CPU tier:

- GPU hit for `A`: reuse directly;
- CPU hit for `A`: restore, then reuse;
- miss for `A`: recompute it;
- `B + C`: recompute in every case because `B` was created on decode and `C`
  is new input.

After turn 2 prefill completes, full blocks from `A + B + C` are eligible for
the new CPU snapshot. Decode output therefore becomes CPU-cacheable one turn
later, after it has passed through prefill.

## Terminology

| Term | Meaning |
| --- | --- |
| GPU cache target | One prefill `(replica_id, dp_id)` with an independent GPU `BlockPool`. |
| CPU cache target | CPU DRAM store paired one-to-one with a GPU cache target. |
| GPU hit block | A requested prefix block already present in the local GPU `BlockPool`. |
| CPU hit block | A requested prefix block absent from GPU but committed in the paired CPU store. |
| Miss block | The first requested prefix block absent from both tiers, or any block after it. |
| CPU frontier | Number of consecutive committed full blocks starting at block index zero. |
| Offload | GPU-to-CPU copy after prefill completion. |
| Restore | CPU-to-GPU copy before request admission. |
| Export barrier | Completion state that keeps prefill request pages alive until all required outgoing copies finish. |

## Workload and Routing Contract

The feature builds on the existing session-prefix workload contract:

- `session_id` identifies one append-only linear conversation;
- trace ISL values are newly appended input tokens;
- Frontier materializes each request's effective full prompt;
- prompt and output lengths remain positive and finite;
- expanded context must not exceed `max_tokens`; and
- session IDs are not reused for unrelated conversations.

The CPU store is local, so a session must return to the same prefill cache
target. The MVP therefore requires:

```text
cluster_scheduler_config_type = sticky_round_robin
prefix_caching_key_mode = session
```

The cache-target count remains:

```text
num_cache_targets = num_replicas * data_parallel_size
```

Affinity is required whenever this value is greater than one.

Open-loop arrivals retain their current meaning. If a later turn arrives
before an earlier turn has published or committed its prefix, the later turn
may observe a smaller hit frontier and recompute more work. CPU offloading
does not add a causal dependency between turns.

## Cache Identity and Snapshot Contract

CPU blocks use the same logical session keys as the GPU prefix cache:

```python
("session", session_id, block_index)
```

For a prefill block size of 16, a 50-token completed prefill frontier has three
offloadable full blocks:

```text
block 0: tokens 0..15
block 1: tokens 16..31
block 2: tokens 32..47
partial: tokens 48..49, not offloaded as a reusable block
```

The desired CPU snapshot after prefill is:

```text
[0, floor(num_prefill_tokens / block_size))
```

The store may transfer only missing blocks as an optimization, but the
committed result must behave as a contiguous snapshot from block zero.

### Incremental snapshot update

Suppose a session's current CPU frontier is 2 blocks and a later prefill
finishes with a 5-block frontier:

```text
CPU before: [0, 1]
desired:    [0, 1, 2, 3, 4]
copy:             [2, 3, 4]
CPU after:  [0, 1, 2, 3, 4]
```

If an eviction reduced the CPU frontier to one block, the copy starts at block
one. If the entire session was evicted, the copy starts at block zero.

An offload operation records a monotonic snapshot generation or source
request epoch. Its completion may extend an existing CPU frontier but must
never shrink a longer frontier committed by a newer operation.

## CPU Store Data Model

Add a CPU cache manager local to each prefill cache target.

Suggested types:

```python
@dataclass
class CPUKVCacheBlock:
    block_id: int
    key: Hashable | None
    session_id: int | None
    block_index: int | None
    state: CPUBlockState
    pin_count: int


@dataclass
class CPUSessionCacheEntry:
    session_id: int
    committed_frontier_blocks: int
    reserved_frontier_blocks: int
    last_access_time: float
    last_commit_time: float
    latest_generation: int
    blocks: dict[int, CPUKVCacheBlock]
```

Suggested block states:

| State | Meaning |
| --- | --- |
| `FREE` | Capacity is available. |
| `RESERVED` | Destination capacity is reserved for an in-flight offload. |
| `COMMITTED` | Block can satisfy a CPU prefix lookup. |
| `RESTORING` | Committed source block is pinned for CPU-to-GPU transfer. |

Only `COMMITTED` blocks participate in lookups. `RESERVED` blocks remain
invisible until the full transfer operation commits.

The store exposes operations such as:

```python
def get_committed_frontier(session_id: int) -> int:
    ...

def reserve_offload(
    *,
    session_id: int,
    desired_frontier_blocks: int,
    generation: int,
    time: float,
) -> CPUOffloadReservation:
    ...

def commit_offload(reservation: CPUOffloadReservation, time: float) -> None:
    ...

def abort_offload(reservation: CPUOffloadReservation, time: float) -> None:
    ...

def pin_restore_blocks(
    *,
    session_id: int,
    block_indices: list[int],
    time: float,
) -> CPURestoreLease:
    ...
```

## Tiered Prefix Lookup

The current GPU lookup stops at the first missing block. CPU offloading
requires a tier-aware lookup that considers the union of local GPU and CPU
state while preserving the same contiguous-prefix rule.

For requested keys `K[0:q]`, scan block indices in order:

1. If `K[i]` is present on GPU, classify it as a GPU hit.
2. Otherwise, if `K[i]` is committed on CPU, classify it as a CPU hit.
3. Otherwise, stop. Block `i` and every later block are misses.

Example:

```text
requested: [0, 1, 2, 3, 4, 5]
GPU:       [0, 1,    3]
CPU:             [2,    4]

combined hit frontier: [0, 1, 2, 3, 4]
CPU restore blocks:    [2, 4]
prefill starts at:      block 5
```

If block zero is missing from both tiers, the hit frontier is zero even if
later blocks exist.

Suggested result type:

```python
@dataclass
class TieredPrefixPlan:
    query_blocks: int
    gpu_hit_blocks: list[KVCacheBlock]
    cpu_restore_indices: list[int]
    hit_frontier_blocks: int
    hit_tokens: int
    num_new_tokens: int
```

The existing fully-cached-prompt rule remains in force. If the plan would
produce zero new prefill tokens, demote the final reusable block before
starting a CPU transfer so Frontier does not restore a block that it will
immediately recompute.

## Restore Admission Flow

A CPU hit is not immediately equivalent to a computed GPU block. The request
must wait for capacity reservation and transfer completion.

Recommended flow:

1. Route the request to its sticky prefill cache target.
2. Build a `TieredPrefixPlan`.
3. Pin all GPU-hit blocks participating in the plan.
4. Reserve physical GPU pages for CPU restore blocks.
5. Verify that restored pages plus the planned prefill suffix satisfy normal
   allocation and watermark rules.
6. If reservation succeeds, remove the request from normal admission and put
   it in `WAITING_FOR_CPU_RESTORE`.
7. Pin the source CPU blocks.
8. Emit a CPU restore start event.
9. On restore completion:
   - attach the restored physical GPU blocks to the request;
   - publish their existing logical session keys on GPU;
   - release the CPU restore lease;
   - apply the combined cached-token frontier to the request;
   - return the request to schedulable prefill work; and
   - emit a replica schedule event.
10. Prefill computes only `num_new_tokens`.

GPU pages must be reserved before the transfer starts. Deferring allocation
until transfer completion would allow unrelated work to consume the space and
leave a completed restore without a valid destination.

A request waiting for restore should not block unrelated requests at the head
of the normal waiting queue. Track restore-waiting requests separately and
reinsert them according to the existing scheduling priority once the restore
completes.

## Prefill Offload Flow

After a prefill batch advances a request to its completed prefill frontier:

1. Publish newly completed full blocks to the local GPU prefix cache using the
   existing `mark_blocks_computed()` path.
2. Determine the desired CPU frontier.
3. Reserve CPU capacity and select the missing block range.
4. If there is new CPU work, create a GPU-to-CPU offload operation.
5. In PDD, create the normal PREFILL-to-DECODE transfer independently.
6. Register both operations with the request's export barrier.
7. Allow each operation to complete independently.
8. Deliver the request to decode immediately after P-to-D completion.
9. Release the prefill request's GPU allocation only after every required
   export has completed, failed, or been explicitly skipped.

If no CPU blocks need copying, the CPU branch of the export barrier is
immediately complete.

If CPU capacity policy skips an offload, the skip is terminal for barrier
purposes; it must not leak or indefinitely pin source GPU pages.

## Export Barrier and GPU Lifetime

The current PDD implementation tracks pending source transfers as a set of
request IDs. CPU offload introduces more than one outgoing operation for the
same request, so the pending state must become a counter or explicit barrier.

Suggested model:

```python
@dataclass
class PrefillExportBarrier:
    request_id: int
    pending_operations: set[PrefillExportKind]
    source_replica_id: int
    source_dp_id: int


class PrefillExportKind(Enum):
    DECODE_TRANSFER = "decode_transfer"
    CPU_OFFLOAD = "cpu_offload"
```

State transition:

```text
prefill complete
    pending = {DECODE_TRANSFER, CPU_OFFLOAD}

P->D completes
    route request to decode
    pending = {CPU_OFFLOAD}
    keep prefill GPU allocation

P->CPU completes
    commit CPU blocks
    pending = {}
    release prefill GPU allocation
```

The reverse completion order must produce the same final state.

The existing `KVCacheTransferEndEvent` assumes that its target is a compute
cluster and invokes `on_kv_cache_arrival()`. CPU storage is not a compute
cluster, so the CPU copy must use separate events rather than adding a fake
CPU `ClusterType`.

Suggested events:

- `CPUKVCacheOffloadStartEvent`
- `CPUKVCacheOffloadEndEvent`
- `CPUKVCacheRestoreStartEvent`
- `CPUKVCacheRestoreEndEvent`

## Transfer Model

Add a CPU-cache transfer predictor separate from the PDD network KV transfer
predictor.

For a copy of `size_bytes`:

```text
service_time_ms =
    fixed_latency_ms
    + size_bytes / bandwidth_bytes_per_ms
```

Direction-specific parameters are required:

- GPU-to-CPU write bandwidth and latency;
- CPU-to-GPU read bandwidth and latency.

The byte size must be derived from the same runtime KV layout used for GPU
block planning. Avoid maintaining a second formula that can diverge for
dense attention, MLA, TP, PP, or KV quantization.

Conceptually:

```text
per_worker_block_bytes =
    runtime_kv_layout_bytes_per_worker * num_layers

bytes_per_kv_block =
    per_worker_block_bytes * attention_tensor_parallel_size

transfer_bytes = num_transferred_blocks * bytes_per_kv_block
```

`bytes_per_kv_block` is aggregate physical storage for one
`(replica_id, dp_id)` target. It includes TP KV-head replication where the
runtime maps fewer KV heads than TP workers. The configured bandwidth is the
aggregate effective bandwidth of that target. PP is not multiplied again
because the per-worker formula above already spans all model layers.

`vllm_v1_scheduler_config.kv_cache_dtype` is the canonical GPU and CPU KV
precision. `auto` follows the model `torch_dtype`; explicit FP32, FP16, BF16,
FP8, INT8, FP4, and INT4 modes are supported. Packed sub-byte formats round
the final page size up to a whole byte.

The MVP should model one serialized queue per direction and CPU cache target:

```python
next_d2h_available_at: float
next_h2d_available_at: float
```

For an operation submitted at `time`:

```text
actual_start = max(time, next_direction_available_at)
end = actual_start + service_time
next_direction_available_at = end
```

This represents a full-duplex local path with one active copy per direction.
The P-to-D network transfer remains independent from the local CPU copy.
Detailed shared PCIe, CPU DRAM, and DMA contention is follow-up work.

## CPU Capacity and Eviction

CPU capacity is finite and accounted in KV blocks and bytes. The CPU store
must never overcommit capacity, including blocks reserved by in-flight
offloads.

Capacity is configured per aggregate prefill `(replica_id, dp_id)` target,
not per TP worker.

### Session selection

Use session LRU:

```text
oldest session = unpinned session with the smallest last_access_time
```

Update `last_access_time` on:

- successful CPU restore use; and
- successful offload commit.

Merely querying a session without using a CPU block does not protect it.

### Eviction within a session

Evict the largest block index first:

```text
[0, 1, 2, 3, 4] -> evict 4, then 3, then 2
```

This suffix-first rule preserves the longest useful prefix. Evicting block zero
first would strand all later blocks because prefix matching stops at the first
missing block.

If "top block" is used in logs or documentation, it means the session's
highest block index.

### Pinned state

Do not evict:

- blocks reserved as an in-flight offload destination;
- blocks pinned as an in-flight restore source; or
- a session entry being atomically committed.

If no unpinned capacity can be reclaimed, the store keeps the largest prefix
that fits or skips the new offload according to the configured pressure
policy. It must not store a discontiguous suffix.

If one session's desired frontier is larger than total CPU capacity, retain
only the leading prefix that fits.

## Configuration Design

Add a dedicated configuration rather than overloading
`kv_cache_transfer_config`.

Suggested dataclass:

```python
@dataclass
class CPUKVCacheConfig:
    enable: bool = False
    capacity_bytes: int = 0

    write_bandwidth_gbps: float = 64.0
    write_latency_ms: float = 0.01
    read_bandwidth_gbps: float = 64.0
    read_latency_ms: float = 0.01

    eviction_policy: str = "session_lru_suffix"
    capacity_pressure_policy: str = "prefix_fit"
    transfer_concurrency: str = "full_duplex_serialized"
```

Validation:

- `capacity_bytes > 0` when enabled;
- bandwidth values are positive;
- latency values are nonnegative;
- only `session_lru_suffix` is accepted for the MVP;
- only `prefix_fit` or `skip_offload` pressure policy is accepted;
- CPU offloading requires prefix caching;
- CPU offloading requires `prefix_caching_key_mode=session`;
- CPU offloading requires sequential `pd-disaggregation`;
- CPU offloading requires `sticky_round_robin`;
- CPU offloading requires a supported vLLM V1 prefill scheduler; and
- CPU offloading rejects Thinking Mode until export barriers and GPU
  allocations are scoped by request execution epoch; and
- CPU offloading fails fast for guarded or parallel-cluster paths.

Illustrative CLI:

```bash
--cpu_kv_cache_config_enable
--cpu_kv_cache_config_capacity_bytes 137438953472
--cpu_kv_cache_config_write_bandwidth_gbps 64
--cpu_kv_cache_config_write_latency_ms 0.01
--cpu_kv_cache_config_read_bandwidth_gbps 64
--cpu_kv_cache_config_read_latency_ms 0.01
--cpu_kv_cache_config_eviction_policy session_lru_suffix
```

Final CLI names should follow the existing flattened-dataclass convention.

## Request State and Metrics

### Request-level state

Add tier-specific runtime and reporting fields:

```text
cpu_prefix_cache_query_blocks
cpu_prefix_cache_hit_blocks
cpu_prefix_cache_restored_blocks
cpu_prefix_cache_restored_tokens
cpu_kv_cache_restore_bytes
cpu_kv_cache_restore_queue_time
cpu_kv_cache_restore_transfer_time
cpu_kv_cache_offload_bytes
cpu_kv_cache_offload_queue_time
cpu_kv_cache_offload_transfer_time
```

Request time values use Frontier's standard contract: they are held in
seconds internally and exported to `request_metrics.csv` in milliseconds.

The existing prefix-cache totals should represent the combined reusable
frontier:

```text
total_hit_blocks = gpu_hit_blocks + cpu_hit_blocks within the contiguous prefix
```

Add tier breakdown fields rather than changing the meaning of existing
request-prefix metrics.

CPU restore time contributes naturally to request waiting time and TTFT
because prefill cannot start until restore completes.

### System-level metrics

Add `cpu_kv_cache_statistics` containing:

- capacity bytes and blocks;
- current and peak resident bytes and blocks;
- current and peak reserved bytes and blocks;
- offload operations, blocks, bytes, queue time, and transfer time;
- restore operations, blocks, bytes, queue time, and transfer time;
- CPU query and hit blocks;
- CPU hit ratio;
- sessions with CPU hits;
- session eviction count;
- evicted block and byte count;
- skipped or truncated offload count;
- stale-generation completion count;
- current resident session count; and
- source GPU hold time attributable to CPU offload.

Do not merge these values into the existing PDD
`kv_cache_transfer_statistics`; that metric describes inter-cluster transfer.

## Failure and Race Handling

### Offload reservation failure

If CPU capacity cannot be reserved after allowed eviction:

- apply the configured prefix-fit or skip policy;
- mark the CPU export branch terminal;
- allow the GPU export barrier to finish normally; and
- record the event in metrics.

### Restore GPU allocation failure

Do not start the restore until GPU destination pages are reserved. If the
request cannot reserve them:

- leave it in the normal waiting path;
- release temporary GPU and CPU pins;
- retry under the existing scheduling policy; and
- do not record a successful CPU hit admission yet.

### Stale completion

An offload completion from an older request may arrive after a newer session
snapshot has committed. The older completion:

- may fill a still-missing block if its key is consistent with the append-only
  session contract;
- must not reduce `committed_frontier_blocks`;
- must not replace a newer reservation; and
- increments a stale-generation metric when discarded.

### Restore cancellation

The MVP should avoid preempting a request while its CPU restore is in flight.
If cancellation becomes necessary, release:

- CPU source pins;
- reserved GPU destination blocks; and
- request restore-wait state

exactly once.

### Same-time session turns

CPU `RESERVED` blocks are not visible to another request. Only committed
blocks can hit. A same-session request arriving before an offload commit may
use still-resident GPU blocks or recompute the missing prefix, matching the
existing completed-frontier rule.

## Detailed Code Changes

### 1. Configuration

Files:

- `frontier/config/cpu_kv_cache_config.py` (new)
- `frontier/config/config.py`
- `frontier/config/utils.py`, if required by flat CLI naming

Changes:

- add and validate `CPUKVCacheConfig`;
- attach it to `SimulationConfig`;
- expose flattened CLI fields;
- reject unsupported architecture, scheduler, cache mode, and affinity
  combinations during configuration validation.

### 2. CPU block store

Files:

- `frontier/kv_cache/cpu_kv_cache_block.py` (new)
- `frontier/kv_cache/cpu_kv_cache_manager.py` (new)
- `frontier/kv_cache/__init__.py`

Changes:

- implement fixed CPU capacity;
- implement per-session committed frontiers;
- implement reservation, commit, abort, pin, and release operations;
- implement session-LRU/suffix-first eviction;
- enforce full-block and contiguous-prefix invariants;
- expose occupancy and eviction statistics.

### 3. Shared KV sizing

Files:

- `frontier/attention/memory.py`
- `frontier/scheduler/utils/memory_planner.py`
- `frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py`

Changes:

- expose one canonical `bytes_per_kv_block` calculation;
- use the same attention-family, KV-head, head-size, layer, parallelism, and
  precision semantics for GPU planning and CPU transfer accounting;
- add dense, MLA, TP, and PP sizing regression tests.

### 4. CPU transfer predictor and engine

Files:

- `frontier/cpu_kv_cache_transfer/` (new package)

Changes:

- calculate direction-specific analytical service time;
- calculate queue delay per cache target and direction;
- track direction availability timestamps;
- return transfer byte, queue, start, and end information.

### 5. Events and operation records

Files:

- `frontier/entities/cpu_kv_cache_transfer_info.py` (new)
- `frontier/events/cpu_kv_cache_offload_start_event.py` (new)
- `frontier/events/cpu_kv_cache_offload_end_event.py` (new)
- `frontier/events/cpu_kv_cache_restore_start_event.py` (new)
- `frontier/events/cpu_kv_cache_restore_end_event.py` (new)
- `frontier/events/__init__.py`
- `frontier/types/event_type.py`

Changes:

- add CPU-specific event types;
- keep CPU storage separate from `ClusterType`;
- make end events commit or release leases exactly once;
- emit replica rescheduling when restored pages become ready or offload
  completion releases GPU memory.

### 6. Prefill export barrier

Files:

- `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`
- `frontier/events/cluster_batch_end_event.py`
- `frontier/events/kv_cache_transfer_end_event.py`

Changes:

- replace the single pending-transfer set with an explicit multi-operation
  export barrier;
- register decode and CPU exports independently;
- preserve immediate decode arrival on P-to-D completion;
- defer prefill source release until all required exports are terminal;
- handle either completion order;
- preserve current behavior when CPU offloading is disabled.

### 7. Tiered admission

Files:

- `frontier/kv_cache/base_kv_cache_manager.py`
- `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`

Changes:

- add non-breaking GPU key probes needed by the tiered planner;
- build one contiguous GPU+CPU prefix plan;
- reserve restore pages before emitting a transfer;
- maintain a restore-waiting request collection;
- publish restored GPU block keys only after transfer completion;
- apply existing cached-token and fully-cached-prompt semantics;
- avoid lookup metric double counting across failed admission retries.

### 8. Metrics and traces

Files:

- `frontier/entities/request.py`
- `frontier/metrics/constants.py`
- `frontier/metrics/metrics_store.py`
- `frontier/metrics/op_trace_utils.py`, if CPU copies are added to op tracing

Changes:

- add request tier-breakdown fields;
- add system CPU cache statistics;
- add CPU offload/restore trace events;
- keep PDD inter-cluster transfer metrics separate;
- include queue time separately from transfer service time.

### 9. Examples and documentation

Files:

- `examples/architecture/pdd/offline/`
- `examples/architecture/pdd/online/`
- `examples/fixtures/`
- `examples/architecture/README.md`
- `docs/cli/README.md`

Changes:

- add an analytical CPU-offload PDD example;
- use a session trace with at least two turns and two sessions;
- document that prior decode output is recomputed on the next prefill;
- document local CPU capacity and transfer fields;
- document expected GPU-hit, CPU-hit, and miss boundaries.

## Testing Plan

### Configuration tests

- CPU offloading is disabled by default.
- Enabled mode requires positive capacity and bandwidth.
- Negative latency is rejected.
- Non-session key mode is rejected.
- Non-sticky multi-target routing is rejected.
- `sticky_lor`, parallel PDD, co-location MVP opt-in, and guarded AFD paths fail
  with focused errors.

### CPU manager unit tests

- same-session blocks are found by position;
- different sessions never share blocks;
- only committed blocks hit;
- lookup stops at the first missing block;
- reservation accounts against capacity;
- abort returns reserved capacity;
- LRU chooses the oldest unpinned session;
- suffix-first eviction preserves block zero and the longest prefix;
- pinned restore and offload blocks cannot be evicted;
- a session larger than CPU capacity retains only a leading prefix;
- an older completion cannot shrink a newer frontier;
- incremental offload transfers only CPU-missing blocks.

### Tiered lookup tests

- GPU-only hit;
- CPU-only hit;
- GPU prefix followed by CPU suffix;
- CPU fills a gap between two GPU blocks;
- first missing block terminates the combined hit frontier;
- later GPU or CPU blocks after a miss are ignored;
- partial final blocks miss;
- fully cached prompt demotes one block before restore;
- query, GPU-hit, CPU-hit, and total-hit metrics agree.

### Restore scheduling tests

- GPU pages are reserved before restore starts;
- restored blocks remain invisible until transfer completion;
- unrelated requests can schedule while one request waits for restore;
- restore completion makes the request schedulable;
- allocation failure leaves no CPU or GPU pin leak;
- H2D queue time follows per-direction serialization;
- CPU restore time contributes to TTFT.

### PDD export-barrier tests

Test both completion orders:

```text
P->D first, P->CPU second
P->CPU first, P->D second
```

Verify:

- decode arrival occurs immediately after P-to-D completion;
- CPU completion does not enqueue a request in a compute cluster;
- prefill GPU resources remain allocated after only one branch completes;
- resources release exactly once after both branches are terminal;
- a skipped CPU offload does not block release;
- source replica scheduling is retriggered when offload completion frees memory;
- CPU-disabled behavior matches the existing single-transfer contract.

### End-to-end semantic test

Use block size 16 and a session trace such as:

```csv
arrived_at,num_prefill_tokens,num_decode_tokens,session_id
0.0,32,16,7
100.0,16,8,7
1.0,32,16,8
101.0,16,8,8
```

For each session:

```text
turn 1:
  effective prefill = 32
  CPU snapshot after prefill = 2 blocks / 32 tokens
  decode output = 16, not offloaded

turn 2:
  effective prefill = 32 + 16 + 16 = 64
  reusable prefill-side prefix = 2 blocks / 32 tokens
  recomputed tokens = prior decode 16 + new input 16
  CPU snapshot after prefill = 4 blocks / 64 tokens
```

Force GPU eviction between turns so the second-turn hit is observably from
CPU, not GPU.

Verify:

- each first turn has zero CPU hit blocks;
- each second turn restores two CPU blocks;
- each second turn recomputes 32 prefill tokens;
- no decode-to-CPU transfer is emitted;
- P-to-D still transfers the full current prompt KV;
- CPU offload transfers the expected missing snapshot blocks;
- session 7 never reuses session 8 blocks;
- final request and system metrics match the expected byte and block counts.

### Regression tests

Run:

- existing session-prefix manager tests;
- existing session-prefix PDD runtime tests;
- PDD KV transfer completion-contract tests;
- PDD example smoke tests;
- block-hash prefix-cache regression tests; and
- release architecture guard tests.

CPU-disabled runs must preserve existing metrics and event behavior.

## Implementation Sequence

### Phase 1: Store and policy

1. Add configuration and validation.
2. Add canonical KV block byte sizing.
3. Implement the CPU block store.
4. Add capacity, reservation, LRU, suffix-eviction, and generation tests.

### Phase 2: Restore path

1. Add tiered prefix planning.
2. Add GPU destination reservation.
3. Add CPU-to-GPU transfer events.
4. Add restore-waiting request handling.
5. Add request and system restore metrics.

At the end of this phase, tests may seed the CPU store directly to validate
restore behavior.

### Phase 3: Prefill offload path

1. Add GPU-to-CPU transfer events.
2. Create CPU snapshot reservations after completed prefill.
3. Replace the single PDD pending-transfer set with the export barrier.
4. Connect CPU commit and GPU release.
5. Add offload, barrier-order, and source-memory-pressure tests.

### Phase 4: End-to-end PDD surface

1. Add offline and online examples.
2. Add end-to-end session fixtures.
3. Add CLI and architecture documentation.
4. Run the complete prefix-cache and PDD regression suites.

## Acceptance Criteria

The MVP is complete when:

1. Sequential PDD can run with prefill-side CPU offloading enabled through the
   public CLI.
2. A prefill-completed full block can be committed to finite CPU capacity and
   restored to the same prefill cache target on a later turn.
3. GPU, CPU, and miss blocks form one correct contiguous prefix plan.
4. Prior decode output is recomputed on the next turn and is not reported as a
   same-turn CPU hit.
5. Once recomputed by a later prefill, that output can become part of the next
   CPU snapshot.
6. P-to-D completion can start decode without waiting for P-to-CPU completion.
7. Prefill source GPU pages are released only after all required exports are
   terminal.
8. CPU capacity never exceeds its configured limit, including reservations.
9. Eviction preserves a contiguous per-session prefix and never evicts pinned
   blocks.
10. Request CSV and system JSON distinguish GPU hits, CPU hits, misses,
    offload cost, restore cost, and eviction.
11. CPU-disabled runs preserve the current PDD behavior.

## Implementation and Validation Record

All four implementation phases and the acceptance criteria above are covered
by the current implementation.

- The finite per-target CPU store, session-LRU suffix eviction, reservations,
  restore leases, monotonic session generations, and full-duplex analytical
  transfer queues are implemented under `frontier/kv_cache/` and
  `frontier/cpu_kv_cache_transfer/`.
- CPU capacity and transfer bytes aggregate all attention TP worker shards in
  one `(replica_id, dp_id)` target and use the configured runtime KV
  precision. Session restore pins are tracked in O(1), and an LRU victim's
  required suffix is evicted in one linear bulk operation.
- Sequential PDD `vllm_v1` integrates mixed GPU/CPU/recompute planning,
  watermark-aware atomic GPU reservation, H2D restore events, D2H snapshot
  events, and the two-branch source export barrier.
- Failure paths roll back CPU reservations and pins, GPU destination pages,
  pending-operation maps, and export-barrier branches exactly once.
- Request CSV time fields follow Frontier's millisecond export contract.
  System JSON reports CPU occupancy, transfer, hit, eviction, and attributable
  source-GPU hold metrics.
- Offline and online example wrappers and a multi-session E2E fixture are
  included under `examples/architecture/pdd/` and `examples/fixtures/`.

Validation included five independent review-and-fix cycles. Those reviews
found and drove fixes for zero-block reservation leaks, unsupported Thinking
Mode overlap, GPU-only metric classification, request time units, attributable
GPU hold time, out-of-order and aborted reservation suffixes, generation
ordering, large-capacity complexity, nonfinite transfer settings, watermark
preservation, transactional failure cleanup, no-op LRU updates, and
preemption-safe metric deduplication.

The final stress suite performs randomized reserve/commit/abort/pin/release
and eviction sequences under both capacity policies, checks capacity and
ownership invariants after every operation, exercises thousands of
full-duplex transfer submissions, and compares E2E runs to confirm that CPU
restore latency contributes to follow-up TTFT.

## Follow-up Work

Potential follow-ups include:

- decode-to-CPU delta offload;
- decode-to-prefill KV return;
- a CPU snapshot containing the full completed turn without next-turn
  recomputation;
- co-location support;
- shared or remote CPU KV-cache pools;
- cross-replica session migration;
- direct CPU-to-decode restore;
- multi-tier CPU plus NVMe caching;
- compression and dtype conversion;
- explicit session TTL and termination;
- cost-aware admission and offload bypass;
- shared PCIe, NUMA, DMA, and memory-controller contention;
- cache-aware cluster scheduling without strict sticky affinity; and
- closed-loop multi-turn arrivals.
