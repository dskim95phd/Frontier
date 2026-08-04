# C++ Porting Step 5: Prefill-Side CPU KV-Cache Tiering

## Status

- **State:** Implemented (2026-08-03)
- **Milestone:** Step 5 of the C++ simulator port
- **Primary architecture:** Sequential `pd-disaggregation`
- **Primary scheduler:** PREFILL `vllm_v1`
- **Cache identity:** Session-scoped, append-only prefix ranges
- **CPU cache ownership:** One independent store per PREFILL
  `(replica_id, dp_id)` target
- **Python reference:** The implemented prefill-side CPU KV-cache path under
  `frontier/kv_cache/`, `frontier/cpu_kv_cache_transfer/`, and the production
  vLLM V1 replica scheduler

### Implementation result

The C++ Step 5 path is implemented for the supported sequential-PDD contract.
It includes the target-local finite CPU store, explicit reservation/lease/pin
state, analytical full-duplex D2H/H2D queues, deferred restore admission, the
two-branch PREFILL export barrier, request/target/system metrics, online and
offline examples, stress/topology coverage, and a Python-manager logical
  differential test. The differential also replays the C++ transfer submissions
  through Python's analytical engine and checks H2D contribution to TTFT.

The implementation intentionally retains the planned analytical GPU range
model. Python/C++ validation therefore compares logical frontiers, ownership,
traffic, metrics, and terminal state rather than physical block IDs or exact
object graphs. Integer counts, IDs, target affinity, byte totals, and
deliberately equal-time event ordering are exact. CPU analytical timing, when
compared across implementations, uses `1e-9` absolute and relative tolerance
in milliseconds; end-to-end tests additionally verify causal ordering and
that slower H2D service increases follow-up TTFT.

Validation completed on 2026-08-03:

- clean MSVC C++17 build completed for all targets;
- all 31 CTest entries passed, including CPU manager, transfer, scheduler,
  online/offline integration, stress, topology, and Python logical
  differential gates;
- all 67 focused Python CPU KV-cache unit/stress/tiered tests passed; and
- both `cpu-kv-online` and `cpu-kv-offline` examples completed with three
  offloads, one restore, and zero terminal reservation/lease state.

### Extended stress matrix and CPU ON/OFF comparison

`frontier_simulator.cpu_kv_cache_stress_matrix` adds deterministic runs across
direct CPU capacities of 1, 2, and 8 blocks, `prefix_fit` and `skip_offload`,
online and offline PDD, fast and slow transfers, TP2/PP2 static-slice capacity,
and four independent PREFILL `(replica_id, dp_id)` targets. Every run checks
completion, affinity, PDD transfer identity, finite capacity accounting, and
zero terminal reservation/lease/pending/staged state.

The following CPU toggle results use the same 8-session, 2-turn fixed-latency
workload. Times are simulator results for this deterministic fixture, not
hardware performance predictions.

| Mode | CPU tier | PREFILL scheduled tokens | Cached tokens | Successor TTFT avg (ms) | Makespan (ms) | Offload / restore | Block / session evictions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Online fast | OFF | 160 | 48 | 2.38554 | 666.829 | 0 / 0 | 0 / 0 |
| Online fast | ON | 128 | 80 | 2.40269 | 666.863 | 16 / 4 | 20 / 5 |
| Online slow | OFF | 160 | 48 | 2.38554 | 666.829 | 0 / 0 | 0 / 0 |
| Online slow | ON | 128 | 80 | 4.35770 | 670.773 | 16 / 4 | 20 / 5 |
| Offline | OFF | 214 | 0 | 11.3778 | 25.7910 | 0 / 0 | 0 / 0 |
| Offline | ON | 216 | 0 | 11.2578 | 26.0247 | 16 / 0 | 40 / 13 |

For the online fixture, CPU tiering reduces scheduled PREFILL work by 20% and
raises cached tokens from 48 to 80. The fast analytical tier is approximately
latency-neutral in this workload, while the deliberately slow tier raises
successor TTFT because H2D service dominates the saved compute. The offline
burst performs D2H snapshots but no restores, demonstrating a workload where
offloading adds traffic without reuse.

Additional pressure outcomes include:

- 1-block `prefix_fit`: 16 offloads, 15 session evictions, 16 truncations;
- 2-block `prefix_fit`: 16 offloads, 30 block evictions, 12 truncations;
- 1-block `skip_offload`: 16 skips and no transfer;
- 8-block mixed-fit `skip_offload`: 14 commits, 2 skips, and 35 block evictions;
- TP2/PP2 static-slice: four physical slices, 16 offloads, and 36 block
  evictions; and
- four target-local CPU stores: 24 offloads and 26 block evictions, with every
  target's independent transfer engine exercised.

## Summary

Step 5 adds a finite CPU DRAM tier below the C++ simulator's existing
session-scoped PREFILL GPU KV cache. After PREFILL completes, reusable full KV
blocks may be copied from the PREFILL GPU target to its paired CPU store. A
later request in the same session can restore a CPU-resident prefix before
computing the remaining prompt suffix.

The implementation should preserve the observable Python state transitions
without replacing the C++ port's analytical GPU cache with Python's physical
GPU block object graph. The selected split is:

- retain the current C++ GPU representation as a session-level contiguous
  prefix range;
- use explicit CPU blocks, reservations, restore leases, pins, and generations
  where those states affect capacity, cancellation, or event ordering; and
- represent mixed GPU/CPU prefix plans using block-index ranges and counts.

Exact event timestamps and every Python diagnostic field are not required to
match. The C++ implementation must, however, preserve the Python feature's
capacity, liveness, reuse, transfer, and source-ownership semantics.

## Relationship to Existing Plans

This document is the detailed implementation plan for Step 5 in
[`cpp-porting-session-prefix-plan.md`](cpp-porting-session-prefix-plan.md).

The Python behavior and its design history are documented in:

- [`prefill-cpu-kv-offloading-plan.md`](prefill-cpu-kv-offloading-plan.md);
- [`cpu-kv-restore-deferred-allocation-plan.md`](cpu-kv-restore-deferred-allocation-plan.md);
  and
- [`cpu-kv-offload-restore-hol-deadlock.md`](cpu-kv-offload-restore-hol-deadlock.md).

The deferred-allocation decision is normative for the C++ port. CPU restore
completion creates a logical staged payload; it does not allocate
scheduler-managed PREFILL GPU blocks.

## Current C++ Baseline

The current C++ implementation already provides the prerequisites required by
Step 5:

- deterministic typed events ordered by `(time, sequence)`;
- sequential PDD with per-request PREFILL-to-DECODE KV transfers;
- per-target vLLM V1 schedulers;
- session affinity through `sticky_round_robin`;
- a target-local analytical GPU session prefix cache;
- complete-block lookup and all-hit demotion;
- session-suffix GPU LRU reclamation;
- PREFILL source KV retention until the PDD transfer completes;
- per-request and system prefix-cache metrics; and
- online/offline, dense/MoE, TP/PP/DP, and multi-replica tests.

The existing GPU cache is intentionally analytical. It stores a session's
resident prefix length rather than one object per physical GPU KV page. This
representation remains the Step 5 default.

The following CPU-tier elements are currently absent:

- CPU cache configuration and validation;
- a finite CPU block store;
- CPU session frontier and eviction state;
- D2H/H2D transfer queues;
- CPU offload and restore events;
- staged restore state;
- a two-branch PREFILL export barrier; and
- CPU-tier request/system metrics.

## Supported Runtime Contract

CPU KV-cache tiering is enabled only when all of the following are true:

```text
system_architecture = pd-disaggregation
enable_parallel_clusters = false
cluster_scheduler.type = sticky_round_robin
prefix_cache.enabled = true
prefix_cache.key_mode = session
PREFILL scheduler.type = vllm_v1
cpu_kv_cache.enabled = true
```

The feature remains disabled by default.

### Supported modes

- sequential PDD online simulation;
- sequential PDD offline simulation;
- dense and currently supported MoE models;
- multiple PREFILL replicas and DP lanes;
- TP and PP topologies already accepted by the C++ simulator;
- fixed and analytical execution-time models; and
- `prefix_fit` and `skip_offload` CPU capacity-pressure policies.

### Explicitly excluded

- co-location CPU KV-cache tiering;
- parallel PDD clusters;
- decode-to-CPU offload;
- decode-to-PREFILL KV return;
- decode-created KV reuse before a later PREFILL recomputes it;
- shared, remote, or cross-target CPU pools;
- session migration between cache targets;
- block-hash prefix caching;
- branching or edited sessions;
- AFD/M2N paths;
- Thinking Mode export overlap;
- speculative receiver-side GPU staging capacity;
- NVMe or additional storage tiers; and
- detailed PCIe, NUMA, DMA-engine, and memory-controller contention.

## Behavioral Contract

For an append-only session:

```text
turn 1 prompt: A
turn 1 output: B
turn 2 prompt: A + B + C
```

the expected behavior is:

```text
turn 1:
  PREFILL computes A
  A -> DECODE GPU
  A -> paired PREFILL CPU store
  DECODE produces B
  B is not returned to PREFILL or CPU

turn 2:
  reuse A from PREFILL GPU or CPU when available
  recompute B + C on PREFILL
  offload the new full-block frontier of A + B + C
```

Only full blocks are reusable and offloadable. If `block_size=16`, a 50-token
PREFILL frontier exposes blocks 0, 1, and 2. Tokens 48 and 49 remain a partial
block and are recomputed on the next turn.

## Core Invariants

The implementation must enforce the following invariants in runtime validation
and tests.

### CPU capacity

```text
resident_cpu_blocks + reserved_cpu_blocks <= capacity_cpu_blocks
```

Reservations count against capacity but are not visible to CPU lookup until
committed.

### Contiguous session frontier

For every session, CPU lookup may report only the committed range:

```text
[0, committed_frontier_blocks)
```

A committed block beyond a gap is not reusable until the gap closes.

### Restore source ownership

```text
RESTORE_PENDING:
  CPU source blocks are pinned

RESTORE_STAGED:
  CPU source lease is released
  staged payload remains logically valid after later CPU eviction
```

### Restore destination ownership

```text
normal PREFILL waiting request: no request-owned GPU allocation
RESTORE_PENDING request:        no request-owned GPU allocation
RESTORE_STAGED request:         no request-owned GPU allocation
RUNNING request:                GPU allocation exists
```

GPU allocation and transition to `RUNNING` must succeed in the same admission
transaction.

### PREFILL export ownership

When a real CPU offload is scheduled, PREFILL source pages are retained until
both required export branches are terminal:

```text
decode_transfer complete AND cpu_offload complete
```

PREFILL-to-DECODE completion may admit DECODE immediately. It does not wait for
CPU offload completion.

### Terminal state

At simulator finalization:

- no CPU reservation is active;
- no restore lease is active;
- no CPU transfer operation is pending;
- no staged restore payload is unconsumed;
- no PREFILL export branch is pending; and
- resident CPU cache contents may remain as persistent cache state.

## Configuration Model

Add a top-level `cpu_kv_cache` object to the normalized C++ configuration.

Suggested shape:

```json
{
  "cpu_kv_cache": {
    "enabled": false,
    "capacity_bytes": 0,
    "static_slice_per_gpu": false,
    "capacity_bytes_per_gpu": 750000000000,
    "dram_bandwidth_gbps_per_gpu": 4800.0,
    "c2c_bandwidth_gbps_per_gpu": 3600.0,
    "write_bandwidth_gbps": 64.0,
    "write_latency_ms": 0.01,
    "read_bandwidth_gbps": 64.0,
    "read_latency_ms": 0.01,
    "eviction_policy": "session_lru_suffix",
    "capacity_pressure_policy": "prefix_fit",
    "transfer_concurrency": "full_duplex_serialized"
  }
}
```

### Compatibility decision

To avoid invalidating every existing schema-version-1 input, the parser should
accept an omitted `cpu_kv_cache` object as the disabled default. The normalized
serializer should always emit the object. Unknown fields and unsupported
values inside the object remain errors.

If the project instead requires every top-level field to be present, update all
fixtures and examples in the same change. Do not accept two ambiguous partial
shapes.

### Validation

Validate all integer and floating-point fields before constructing scheduler
state:

- enabled direct capacity must be greater than zero;
- capacity must hold at least one logical KV block;
- per-GPU capacity must be positive;
- bandwidths must be finite and positive;
- fixed latencies must be finite and nonnegative;
- only `session_lru_suffix` eviction is accepted;
- only `prefix_fit` and `skip_offload` pressure policies are accepted;
- only `full_duplex_serialized` transfer concurrency is accepted; and
- unsupported architecture/scheduler/cache combinations fail before events
  are created.

### Static per-GPU slice

For one PREFILL `(replica_id, dp_id)` target:

```text
physical_slices = attention_tensor_parallel_size
                * pipeline_parallel_size

resolved_capacity = capacity_bytes_per_gpu * physical_slices

resolved_bandwidth_per_direction =
    min(dram_bandwidth_gbps_per_gpu, c2c_bandwidth_gbps_per_gpu)
    * physical_slices
```

This mirrors the Python target-resolution contract.

### Bytes per CPU block

Use the existing model-aware KV size calculation for one logical block:

```text
bytes_per_block = model_kv_cache_size_bytes(
    block_size,
    prefill_model,
    kv_cache_dtype_size_bytes)
```

The result represents all KV shards owned by one PREFILL replica/DP target. It
must follow the same dense/MLA layout and runtime KV precision as the PDD
transfer model.

Use checked arithmetic for capacity, block counts, and transfer sizes.

## CPU Store Data Model

Add `cpp/frontier/kv_cache/cpu_kv_cache_manager.{h,cc}`.

### Strong IDs

Add distinct strong IDs where ownership can outlive a local call:

```cpp
CpuBlockId
CpuOffloadReservationId
CpuRestoreLeaseId
CpuKvTransferId
```

Session snapshot generation may use a dedicated `CpuOffloadGeneration` or a
clearly documented `Generation` value. Do not use an untyped integer in event
payloads when a stale operation could mutate newer state.

### Block state

```cpp
enum class CpuBlockState {
    kReserved,
    kCommitted,
};
```

The final C++ representation materializes only owned blocks. Free capacity is
implicit in `capacity_blocks - resident_blocks - reserved_blocks`, while
released materialized IDs are recycled through the free-ID queue. Consequently
there is no live `kFree` block record; this preserves the planned lazy-memory
contract for very large CPU capacities.

Each materialized CPU block records:

- block ID;
- state;
- session ID;
- logical block index;
- pin count; and
- owning reservation ID when reserved.

### Session state

Each CPU session entry records:

- `committed_frontier_blocks`;
- `reserved_frontier_blocks`;
- last access time;
- last commit time;
- latest committed generation;
- logical-index to CPU-block ownership;
- active reservation IDs; and
- aggregate restore pin count.

A session is not evictable while it has an active offload reservation or
restore pin.

### Offload reservation

An offload reservation records:

- reservation ID;
- session ID;
- generation;
- desired frontier;
- admitted frontier;
- reserved block indices and IDs;
- submission time;
- skipped/truncated flags; and
- terminal state.

### Restore lease

A restore lease records:

- lease ID;
- session ID;
- pinned block indices and IDs;
- start time; and
- released state.

### Lazy materialization

Do not allocate one C++ object per configured CPU block at construction time.
Large DRAM capacities can represent hundreds of thousands or millions of
logical blocks. Materialize block records only when reserved, and recycle freed
IDs through a free-ID queue.

Construction and idle memory usage should be independent of configured
capacity.

## CPU Store Operations

### Lookup

CPU lookup is a non-mutating probe used to build a tiered prefix plan. Failed
scheduler retries must not inflate CPU query/hit metrics.

Record logical lookup metrics only after successful admission, deduplicated by
request ID.

### Reserve offload

Given `desired_frontier_blocks`:

1. compute the base frontier from committed and already reserved ranges;
2. reserve only the missing suffix;
3. exclude the same session from eviction while extending its snapshot;
4. evict unpinned LRU session suffixes when necessary;
5. apply `prefix_fit` or `skip_offload` policy;
6. mark new CPU blocks `RESERVED`; and
7. update the session's reserved frontier.

A zero-block, skipped, or fully satisfied reservation is immediately terminal
and does not create a D2H event or export-barrier branch.

### Commit offload

Commit converts reserved blocks to committed blocks and advances the contiguous
frontier across every now-closed gap.

An older completion must not shrink a newer frontier. Out-of-order completions
may commit blocks beyond a temporary gap, but those blocks remain invisible
until the gap is filled.

### Abort offload

Abort must:

- free the reservation's CPU blocks;
- remove it from active reservation state;
- cancel active dependent suffix reservations;
- reclaim already-completed but unreachable committed suffixes beyond the new
  gap; and
- recompute the reserved frontier.

The operation is idempotent after reaching terminal state.

### Pin and release restore

Pinning succeeds only for blocks inside the committed frontier. A partial pin
failure rolls back pins already acquired by that call.

Lease release decrements every block pin and the session aggregate exactly
once. A successful H2D completion refreshes session LRU access time; a
cancellation does not.

### Session-LRU suffix eviction

Victim order is:

```text
(last_access_time, last_commit_time, session_id)
```

Evict only the suffix required to satisfy current pressure. Never evict the
session being extended, a pinned session, or a session with active
reservations.

## Analytical CPU Transfer Engine

Add `cpp/frontier/cpu_kv_cache_transfer/analytical_transfer.{h,cc}`.

Each PREFILL cache target owns one transfer engine with separate next-available
times:

```text
next_d2h_available_at
next_h2d_available_at
```

Transfers in one direction serialize. D2H and H2D may overlap.

For a transfer:

```text
bytes_per_ms = bandwidth_gbps * 1e9 / 8 / 1000
service_time_ms = fixed_latency_ms + size_bytes / bytes_per_ms
start_time = max(submitted_at, direction_available_at)
end_time = start_time + service_time_ms / 1000
queue_time_ms = (start_time - submitted_at) * 1000
```

The timing value type records:

- direction;
- size bytes;
- submission time;
- actual start and end time;
- queue time; and
- service time.

Scheduling a transfer advances only the corresponding direction's availability
time.

## Transfer and Staged-Payload Entities

Add `cpp/frontier/entities/cpu_kv_cache_transfer_info.{h,cc}`.

### Offload operation

An in-flight D2H operation records:

- CPU transfer ID;
- request ID;
- PREFILL replica/DP target;
- CPU reservation ID;
- transfer timing;
- desired frontier;
- session generation; and
- optional PREFILL-to-DECODE completion time.

The attributable source hold metric is:

```text
max(0, cpu_offload_end - decode_transfer_end)
```

### Restore operation

An in-flight H2D operation records:

- CPU transfer ID;
- request ID;
- PREFILL replica/DP target;
- restore lease ID;
- transfer-start tiered plan snapshot;
- transfer timing; and
- request/runtime generation needed for stale-event validation.

It does not own GPU blocks.

### Staged restore

Restore completion creates an immutable logical payload containing:

- request and session ID;
- target replica and DP ID;
- transferred CPU block-index range;
- lookup-time GPU and CPU frontier observations;
- query block count;
- block and prompt size; and
- completed transfer timing.

The staged payload is independent of later CPU eviction.

## Event Model

Extend `EventType`, `EventPayload`, dispatcher declarations, event serialization,
and event handlers with:

```text
CPU_KV_CACHE_OFFLOAD_START
CPU_KV_CACHE_OFFLOAD_END
CPU_KV_CACHE_RESTORE_START
CPU_KV_CACHE_RESTORE_END
```

Payloads should carry IDs and target coordinates, not owning pointers.

### Offload events

Offload start:

- validate that the operation remains pending;
- record start/trace metrics; and
- enqueue offload end at the operation's end time.

Offload end:

- validate operation identity and generation;
- commit the CPU reservation;
- close the CPU export branch;
- release PREFILL GPU source state if no branch remains;
- record transfer and source-hold metrics; and
- wake the PREFILL replica if waiting work exists.

On failure, abort the CPU reservation and close only the CPU branch.

### Restore events

Restore start:

- validate the pending operation;
- record start/trace metrics; and
- enqueue restore end.

Restore end:

- validate operation identity and generation;
- release the CPU lease with `used=true`;
- create the staged payload;
- remove pending restore ownership;
- requeue the request exactly once;
- record completed transfer metrics; and
- enqueue a PREFILL replica schedule event.

On failure, cancel the matching restore exactly once without GPU cleanup.

## Tiered Prefix Planning

The Python GPU manager can expose per-index physical hits. The C++ analytical
GPU cache exposes a contiguous session frontier. Step 5 should exploit this
representation instead of introducing physical GPU blocks.

Suggested plan fields:

```cpp
struct TieredPrefixPlan {
    std::uint64_t query_blocks;
    std::uint64_t cpu_query_blocks;
    std::uint64_t gpu_hit_frontier_blocks;
    std::uint64_t cpu_begin_block;
    std::uint64_t cpu_end_block;
    std::uint64_t hit_frontier_blocks;
    std::uint64_t block_size;
    std::uint64_t prompt_tokens;
};
```

For the analytical contiguous-range model:

```text
GPU source range = [0, gpu_frontier)
CPU source range = [0, cpu_committed_frontier)
```

CPU restore is needed only for the contiguous range after the GPU frontier and
before the first miss.

### First-gap rule

At restore admission, current GPU state may differ from transfer-start state.
Walk the logical range from block zero using:

```text
current GPU data first
then matching staged CPU data
stop at the first index supplied by neither source
```

Although the analytical GPU cache cannot represent arbitrary holes, the staged
range may no longer join the current GPU frontier. In that case, staged blocks
after the gap are transferred traffic but do not save computation.

### Fully cached prompt demotion

If the reusable frontier would cover the entire prompt, demote the last full
block so the scheduler executes at least one block of PREFILL work. Apply this
rule after the tiered source plan is built and again after admission-time
revalidation.

## Deferred Restore Allocation

The scheduler must not reserve PREFILL GPU pages when H2D starts.

### Restore start state transition

```text
NORMAL_WAITING
  -> pin CPU source
  -> schedule H2D
  -> remove request from runnable deque
  -> RESTORE_PENDING
```

State in `RESTORE_PENDING`:

- request remains logically `WAITING` for latency accounting;
- request is absent from ordinary runnable deques;
- one restore end event provides liveness;
- CPU lease is held; and
- request owns zero GPU blocks.

### Restore completion state transition

```text
RESTORE_PENDING
  -> release CPU lease
  -> create staged payload
  -> reinsert request into runnable PREFILL queue
  -> RESTORE_STAGED
```

State in `RESTORE_STAGED`:

- request remains logically `WAITING`;
- request has exactly one runnable queue entry;
- staged payload exists;
- CPU lease is released; and
- request owns zero GPU blocks.

### Admission-time revalidation

When a staged request reaches normal PREFILL admission:

1. query the current GPU session frontier;
2. combine it with the staged CPU range;
3. stop at the first gap;
4. apply fully cached prompt demotion;
5. compute required GPU pages for the reusable prefix and scheduled suffix;
6. apply ordinary token-budget, batch-cap, watermark, and capacity checks;
7. atomically allocate and publish restored blocks;
8. restore request token frontiers and record tiered metrics;
9. remove the staged payload;
10. remove the request from waiting; and
11. append it to `RUNNING` and the scheduled batch.

Temporary capacity failure leaves the staged payload and queue position intact
and allocates no GPU blocks.

## GPU Cache Manager Extension

Extend `ReplicaKVCacheManager` with an admission-only tiered materialization
primitive.

Possible API:

```cpp
bool can_admit_tiered(
    RequestId request_id,
    SessionId session_id,
    std::uint64_t reusable_frontier_blocks,
    std::uint64_t scheduled_tokens) const;

void admit_tiered(
    RequestId request_id,
    SessionId session_id,
    std::uint64_t reusable_frontier_blocks,
    std::uint64_t scheduled_tokens);
```

The operation must account for:

- currently resident GPU prefix blocks already owned by the session;
- staged CPU blocks that require new GPU capacity;
- prompt suffix or recompute blocks;
- the existing GPU watermark; and
- LRU reclamation of other inactive GPU sessions.

On success:

- the request allocation owns all required pages;
- the session's published frontier includes used restored blocks;
- the request can transition to `RUNNING`; and
- cache accounting remains valid.

On any mutation failure, roll back the whole admission and leave the request
with zero GPU allocation and its staged payload intact.

## Scheduler State

Add target-local state to `VllmV1Scheduler`:

- optional CPU manager;
- optional CPU transfer engine;
- per-session offload generation;
- pending CPU offload operations by request;
- pending CPU restore operations by request;
- restore-suspended waiting requests;
- staged restore payloads;
- pending auxiliary CPU start events; and
- pending PREFILL export branches by request.

`contains_request()`, `has_pending_work()`, `idle()`, waiting diagnostics, and
runtime validation must include these side states where appropriate.

Resident CPU cache entries do not make a scheduler non-idle. Pending
reservations, leases, operations, staged payloads, and export branches do.

### Auxiliary events

When waiting-queue scheduling discovers a CPU restore, it must be able to emit a
restore-start event without producing a compute batch. Add a small scheduler
API that drains pending auxiliary `EventPayload` values after each schedule
call, or carry them explicitly in `ScheduleResult`.

The preferred option is a separate drain method so scheduler-trace output does
not need to own event payload variants.

## PREFILL Export Barrier

Replace the single pending-PDD-transfer release condition with a request-local
set or bit mask:

```text
decode_transfer
cpu_offload
```

### PREFILL completion

At completed PREFILL batch handling:

1. mark `decode_transfer` pending;
2. request a CPU offload reservation;
3. if the reservation contains blocks, mark `cpu_offload` pending;
4. enqueue the CPU offload start event;
5. create and enqueue the ordinary PDD transfer start event; and
6. retain the source GPU allocation.

For equal-time parity with Python, enqueue CPU offload start before ordinary PDD
transfer start.

### Decode-transfer completion

PDD completion must:

- deliver the request to DECODE immediately;
- close `decode_transfer`;
- record the completion time in a pending CPU offload operation; and
- release PREFILL source GPU pages only when no export branch remains.

### CPU-offload completion

CPU offload completion must:

- commit the reservation;
- close `cpu_offload`; and
- release PREFILL source GPU pages only when no export branch remains.

### No-op and failure paths

A no-op, skipped, or zero-block reservation does not add a CPU branch. A D2H
scheduling failure aborts the reservation and closes only the CPU branch. It
must not cancel the ordinary PDD transfer.

## Request Metrics

Extend the C++ request entity and request output record with:

- GPU prefix hit blocks;
- CPU prefix query blocks;
- CPU prefix hit blocks consumed at admission;
- CPU restore transferred blocks;
- CPU restore consumed blocks;
- CPU restore discarded blocks;
- CPU restored tokens consumed at admission;
- restore bytes;
- restore queue time;
- restore service time;
- offload bytes;
- offload queue time; and
- offload service time.

Keep transfer traffic distinct from compute reuse:

```text
restore_transferred_blocks >= restore_consumed_blocks
```

Transfer time and byte metrics are recorded from the scheduled operation.
CPU hit and cached-token metrics are recorded only from the admission-revalidated
plan.

Internal request durations remain seconds. CSV and JSON latency fields with
`_ms` suffixes are exported in milliseconds.

## System and Target Metrics

Add one record per PREFILL CPU target containing:

- target replica and DP ID;
- capacity bytes/blocks;
- bytes per block;
- resident, reserved, and free bytes/blocks;
- peak resident and reserved occupancy;
- resident session count;
- evicted sessions/blocks/bytes;
- skipped and truncated offloads;
- stale generation completions;
- CPU query/hit blocks;
- sessions with CPU hits;
- pending restore operations;
- staged restore payloads; and
- active leases/reservations for final diagnostics.

Aggregate system metrics include:

- cache target count;
- offload/restore operations, blocks, and bytes;
- cumulative D2H/H2D queue and service time;
- CPU query/hit ratio;
- source GPU hold time attributable to CPU offload; and
- sums of the target occupancy/eviction fields.

Add detailed CPU transfer records to full output and retain the four typed CPU
events in the event trace. Summary and request-only output modes should not need
detailed event retention to compute aggregates.

## Failure and Cancellation Semantics

### Restore scheduling failure

- release the just-created CPU lease with `used=false`;
- remove the suspended request side state;
- leave GPU allocation unchanged; and
- propagate a deterministic error.

### Pending restore cancellation

- match the exact operation ID/generation;
- remove the pending operation;
- release the CPU lease once;
- remove any not-yet-drained auxiliary start payload;
- restore or remove request queue ownership as specified by the caller; and
- perform no GPU cleanup.

### Staged restore cancellation

- remove the staged payload;
- remove its runnable queue entry exactly once; and
- perform no GPU cleanup.

### Offload scheduling failure

- abort the matching CPU reservation;
- remove pending offload state;
- close only the CPU export branch; and
- free PREFILL source pages only if the decode branch is already terminal.

### Stale events

A stale or duplicate event must not:

- release a newer lease;
- commit a newer reservation;
- requeue a request twice;
- consume another staged payload;
- close another request's export branch; or
- free GPU pages owned by newer work.

## File-Level Change Plan

### New files

```text
cpp/frontier/kv_cache/cpu_kv_cache_manager.h
cpp/frontier/kv_cache/cpu_kv_cache_manager.cc
cpp/frontier/cpu_kv_cache_transfer/analytical_transfer.h
cpp/frontier/cpu_kv_cache_transfer/analytical_transfer.cc
cpp/frontier/entities/cpu_kv_cache_transfer_info.h
cpp/frontier/entities/cpu_kv_cache_transfer_info.cc
cpp/frontier/events/cpu_kv_cache_offload_start_event.cc
cpp/frontier/events/cpu_kv_cache_offload_end_event.cc
cpp/frontier/events/cpu_kv_cache_restore_start_event.cc
cpp/frontier/events/cpu_kv_cache_restore_end_event.cc
cpp/tests/kv_cache/cpu_kv_cache_manager_test.cc
cpp/tests/cpu_kv_cache_transfer/analytical_transfer_test.cc
cpp/tests/simulator/cpu_kv_cache_integration_test.cc
cpp/tests/simulator/cpu_kv_cache_stress_test.cc
```

### Existing files with material changes

```text
cpp/CMakeLists.txt
cpp/frontier/config/config.h
cpp/frontier/config/config_parse.cc
cpp/frontier/config/config_serialize.cc
cpp/frontier/core/ids.h
cpp/frontier/core/event.h
cpp/frontier/entities/request.h
cpp/frontier/entities/request.cc
cpp/frontier/events/event_handlers.h
cpp/frontier/events/event_dispatcher.cc
cpp/frontier/events/cluster_batch_end_event.cc
cpp/frontier/events/kv_cache_transfer_end_event.cc
cpp/frontier/events/replica_schedule_event.cc
cpp/frontier/kv_cache/replica_kv_cache_manager.h
cpp/frontier/kv_cache/replica_kv_cache_manager.cc
cpp/frontier/metrics/metrics_store.h
cpp/frontier/metrics/metrics_store.cc
cpp/frontier/metrics/output_contract.h
cpp/frontier/metrics/output_contract.cc
cpp/frontier/scheduler/global_scheduler/*
cpp/frontier/scheduler/cluster_scheduler/*
cpp/frontier/scheduler/replica_scheduler/base_replica_scheduler.*
cpp/frontier/scheduler/replica_scheduler/replica_scheduler_factory.*
cpp/frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.*
cpp/frontier/simulator/simulator.*
cpp/tests/config/config_test.cc
cpp/tests/entities/request_test.cc
cpp/tests/metrics/output_contract_test.cc
cpp/tests/scheduler/vllm_v1_scheduler_test.cc
cpp/tests/simulator/sequential_pdd_test.cc
```

### Documentation and examples

```text
docs/roadmap/README.md
docs/roadmap/cpp-porting-session-prefix-plan.md
cpp/README.md
cpp/examples/README.md
cpp/examples/configs/<cpu-offload-pdd-config>.json
cpp/examples/workloads/<multi-session-cpu-restore>.csv
```

## Implementation Milestones

### 5.1 Configuration and byte-size contract

- add `CpuKVCacheConfig`;
- parse, validate, resolve, and serialize it;
- calculate aggregate target bytes per block;
- propagate disabled/enabled config to PREFILL replica targets; and
- add config round-trip and invalid-combination tests.

Gate: all current config and analytical KV layout tests pass.

### 5.2 Standalone CPU cache manager

- implement block/session/reservation/lease state;
- implement capacity policies and suffix LRU;
- implement monotonic generation and dependent-abort behavior;
- expose statistics and a full invariant checker; and
- add deterministic and randomized manager tests.

Gate: CPU manager passes independently of PDD and scheduler code.

### 5.3 Analytical transfer engine and typed events

- implement per-direction serialized queues;
- add timing and transfer-operation entities;
- add the four event types and output names;
- add start/end dispatch handlers; and
- validate equal-time ordering and full-duplex behavior.

Gate: transfer tests and existing event-order tests pass.

### 5.4 Deferred CPU restore

- build tiered lookup plans;
- suspend restore-pending requests outside runnable queues;
- pin/release CPU source blocks;
- stage completed restore payloads;
- revalidate prefix sources at admission;
- atomically allocate/publish GPU state; and
- add cancellation, watermark, and liveness validation.

Gate: GPU-only, CPU-only, mixed, miss, first-gap, fully cached demotion, and
small-GPU head-of-line tests pass.

### 5.5 CPU offload and export barrier

- reserve incremental CPU snapshot suffixes at PREFILL completion;
- schedule D2H events independently of PDD transfer;
- add monotonic per-session generations;
- retain source GPU allocation across both branches;
- handle either completion order; and
- add exact-once rollback behavior.

Gate: both export orderings, no-op offloads, schedule failure, and stale
generation tests pass without GPU or CPU reservation leaks.

### 5.6 Metrics, output, examples, and differential validation

- extend request and system output contracts;
- add per-target and transfer records;
- add online/offline example configs and workloads;
- add Python-vs-C++ logical differential checks; and
- run the full C++ regression suite.

Gate: Step 5 acceptance matrix and all existing Step 1-4 tests pass.

## Unit Test Matrix

### CPU store

- commits only full contiguous frontiers;
- reserves only a missing incremental suffix;
- aborted reservations are invisible and return capacity;
- session LRU evicts the suffix of the oldest eligible session;
- restore pins prevent eviction;
- `prefix_fit` truncates a snapshot larger than capacity;
- `skip_offload` admits no partial delta;
- out-of-order commits never shrink a frontier;
- aborting an earlier reservation reclaims dependent suffixes;
- no-op offload does not refresh LRU;
- lookup metrics deduplicate scheduler retries;
- empty session metadata does not accumulate;
- zero-block truncation is terminal and does not pin a session;
- huge configured capacity uses constant-size initial state; and
- randomized reserve/commit/abort/pin/release/evict sequences preserve all
  invariants.

### Transfer engine

- D2H operations serialize;
- H2D operations serialize;
- D2H and H2D overlap;
- queued operations report positive queue time;
- zero-byte timing follows the chosen API contract;
- nonfinite and invalid bandwidth/latency fail; and
- large timestamps and sizes do not overflow.

### Tiered plan and restore

- GPU-only hit needs no restore;
- CPU-only hit creates one restore range;
- GPU prefix plus CPU suffix forms one contiguous frontier;
- a gap stops later reuse;
- fully cached prompts demote the last block;
- restore start allocates no GPU pages;
- restore start does not consume GPU watermark;
- pending restore does not block a competing GPU admission;
- completion stages payload without GPU ownership;
- admission revalidates the current GPU frontier;
- staged CPU blocks replaced by newer GPU hits are not double-allocated;
- staged blocks beyond a new gap are discarded;
- insufficient admission capacity keeps staged state intact; and
- cancellation is exact-once.

### Export barrier

- decode transfer completes before CPU offload;
- CPU offload completes before decode transfer;
- source GPU pages release only after the last branch;
- a terminal/no-op CPU reservation adds no branch;
- offload scheduling failure rolls back the CPU reservation;
- abort closes only the CPU branch;
- per-session offload generations are monotonic; and
- an older completion cannot shrink a newer CPU frontier.

## End-to-End Validation Matrix

### Basic runtime

Use a multi-session, two-turn PDD workload:

- first turns miss and schedule D2H snapshots;
- second turns restore CPU-resident prefixes;
- cached tokens reduce PREFILL work;
- every request still performs one PDD transfer; and
- all schedulers reach quiescence.

### CPU-disabled control

Run the same workload with CPU tiering disabled. Existing Step 4 behavior,
request completion, affinity, GPU cache metrics, and PDD transfer behavior must
remain unchanged apart from additive zero/default output fields.

### Head-of-line regression

Use at least four sessions, three turns per session, a five-block PREFILL GPU
cache, and arrivals close enough to overlap restore demand.

Acceptance:

- all requests complete;
- at least one H2D restore occurs;
- pending/staged restore counts end at zero;
- CPU reservations and leases end at zero; and
- event queue exhaustion cannot leave a GPU-owning waiting request.

### CPU eviction and truncation

- eight-block CPU tier with repeated multi-session churn;
- one-block CPU tier with growing session snapshots; and
- both `prefix_fit` and `skip_offload` policies.

Acceptance:

- capacity is never exceeded;
- eviction preserves contiguous prefixes;
- truncation/skip metrics are nonzero in their intended cases; and
- request/system byte totals agree.

### Slow transfer queues

Use very low D2H/H2D bandwidth and simultaneous demand.

Acceptance:

- same-direction queue time becomes positive;
- D2H and H2D remain full duplex;
- source GPU hold time is attributable only after decode transfer completion;
- simulation TTFT includes H2D delay for restored requests; and
- no reservation or lease leaks remain.

### Multiple cache targets

Run with multiple PREFILL DP lanes and replicas.

Acceptance:

- CPU target count equals `num_replicas * data_parallel_size`;
- session affinity returns each turn to the same target;
- capacity and transfer queues are independent per target; and
- aggregate metrics equal the sum of target metrics.

### Model and topology coverage

Cover at least:

- dense and MoE;
- fixed and analytical execution models;
- online and offline PDD;
- TP1/TP2 or larger;
- PP1 and PP greater than one; and
- DP1 and multi-DP PREFILL.

## Python-vs-C++ Differential Contract

Python remains the behavioral oracle, but Step 5 differential tests should
compare normalized logical outcomes rather than every internal object or exact
floating-point event timestamp.

Required comparisons:

- completed request IDs and completion count;
- session-to-PREFILL-target affinity;
- GPU hit, CPU query, and CPU hit block counts;
- cached PREFILL tokens;
- offload and restore operation counts;
- offload and restore bytes;
- CPU eviction, truncation, and skip outcomes;
- PDD transfer count;
- final zero reservation/lease/pending-operation state;
- restore latency contribution to follow-up TTFT; and
- deterministic event ordering for deliberately equal-time fixtures where the
  two implementations expose equivalent events.

Document tolerances for analytical latency fields. Do not make internal CPU
block IDs or allocation order part of the parity contract.

## Known Fidelity Boundaries

### Analytical GPU ranges

The C++ GPU cache cannot represent arbitrary per-block holes or physical free
queue ordering. This is an intentional continuation of Step 4. CPU store state
is explicit because reservation, pin, and generation behavior requires it;
GPU state remains range-based for simulation speed.

### Logical H2D staging

H2D completion does not reserve a real destination buffer. It creates a
logical payload that consumes GPU capacity only at admission. This avoids the
known restore head-of-line deadlock and matches the selected Python simulator
abstraction, but it does not model receiver-side staging pressure.

### Transfer contention

The engine models one serialized queue per direction and fixed bandwidth plus
latency. It does not model shared PCIe switches, NUMA effects, multiple DMA
engines, or interference with PDD network transfers.

### Same-session overlap

The normal C++ think-time workload contract serializes successor turns after
predecessor completion. CPU manager generations must still support
out-of-order operations in direct tests so future open-loop or overlapping
workloads cannot corrupt a session frontier.

## Acceptance Criteria

Step 5 is complete when all of the following hold:

1. Sequential PDD runs with CPU KV-cache tiering enabled through normalized
   C++ JSON configuration.
2. A completed PREFILL full block can be committed to finite target-local CPU
   capacity.
3. A later same-session request can restore that block and reduce PREFILL work.
4. GPU, staged CPU, and recompute ranges form one correct contiguous admission
   frontier.
5. Prior DECODE output is recomputed by the next PREFILL and is not reported as
   a same-turn CPU hit.
6. Once recomputed by PREFILL, that output may join the next CPU snapshot.
7. PDD transfer completion can admit DECODE without waiting for CPU offload.
8. PREFILL source GPU pages remain allocated until every required export branch
   is terminal.
9. Restore-pending and staged requests own zero PREFILL GPU pages.
10. GPU allocation, restored-prefix publication, and admission are atomic.
11. CPU resident plus reserved blocks never exceed capacity.
12. CPU eviction preserves contiguous prefixes and never evicts pinned state.
13. Generation ordering and cancellation cannot shrink or corrupt newer
    session snapshots.
14. Request and system outputs distinguish traffic transferred from prefix
    blocks actually consumed.
15. CPU-disabled runs preserve Step 4 behavior.
16. The small-GPU restore head-of-line regression reaches quiescence.
17. The full existing C++ test suite and the new Step 5 matrix pass.

### Acceptance evidence

The numbered criteria above are enforced by the following checked-in gates:

| Criterion | Evidence |
| --- | --- |
| 1 | `config_test.cc` validates JSON normalization, supported PDD/scheduler combinations, model-aware block bytes, and invalid surfaces; both CPU examples execute the normalized configuration. |
| 2 | `cpu_kv_cache_manager_test.cc` covers incremental reservation/commit and finite lazy capacity; the online/offline integration test observes committed resident blocks. |
| 3 | `cpu_kv_cache_integration_test.cc` checks a same-session CPU suffix hit, transferred restore block, cached tokens, and reduced PREFILL frontier. |
| 4 | `vllm_v1_scheduler_test.cc` covers first-gap stopping, all-hit demotion, GPU+CPU tier composition, staged revalidation, and atomic admission. |
| 5 | The tiered-plan scheduler test and Python differential require prior DECODE output to remain outside the same-turn CPU-hit range. |
| 6 | Incremental manager tests plus the E2E successor snapshot verify recomputed full blocks can extend the following CPU snapshot. |
| 7 | The export-barrier test exercises both completion orders and verifies the DECODE branch progresses independently of D2H completion. |
| 8 | Export-barrier and slow-transfer integration tests verify source GPU hold until both branches are terminal, including failure rollback. |
| 9 | Deferred-restore, cancellation, scheduling-failure, completion-failure, and simulator finalization checks require zero GPU ownership and zero transient CPU state. |
| 10 | Deferred restore admission tests inject changed GPU frontiers, insufficient capacity, cancellation, and completion failure while checking exact rollback. |
| 11 | Prefix-fit/skip, randomized manager invariants, the exact five-GPU/eight-CPU-block stress case, and final target diagnostics enforce capacity. |
| 12 | Manager LRU suffix/pin tests and randomized invariants verify contiguous eviction and pin protection. |
| 13 | Out-of-order commit, dependent abort, stale completion, and scheduler failure tests protect newer generations and exact-once cleanup. |
| 14 | Integration and topology tests sum request traffic and every target counter into system aggregates while separating transferred and consumed restore blocks. Compact output is checked to retain these aggregates. |
| 15 | The CPU-disabled integration control completes the same workload with the Step 4 PDD transfer count and additive zero CPU fields. |
| 16 | Scheduler head-of-line coverage and the five-GPU/eight-CPU-block stress workload reach quiescence with restore activity and zero reservations, leases, pending restores, or staged payloads. |
| 17 | The 31-entry CTest suite includes manager, transfer, scheduler, online/offline, stress and stress-matrix, dense/MoE/fixed/analytical/topology, CPU ON/OFF, and Python differential gates; focused Python tests and both examples are additional release checks. |

`cpu_kv_cache_topology_matrix_test.cc` additionally proves stable session
affinity, independent per-target capacity and transfer queues, cross-target
overlap, and exact target-to-system sums. The Python differential compares
logical frontiers, affinity, traffic, final ownership, analytical queue/service
timestamps, deterministic event order, and the slow-versus-fast H2D TTFT delta.

## Recommended Implementation Order

Implement Step 5 as small reviewable vertical slices:

1. configuration and block-byte calculation;
2. standalone CPU manager and randomized invariants;
3. analytical transfer engine and typed events;
4. restore planning and logical staging;
5. admission-time GPU materialization;
6. D2H offload and the two-branch export barrier;
7. request/target/system metrics;
8. E2E examples and Python differential checks; and
9. full regression, stress, and documentation pass.

Do not begin export-barrier integration until the standalone CPU manager and
restore deferred-allocation tests are stable. Those two components contain the
highest-risk capacity and liveness invariants.
