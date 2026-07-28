# CPU KV-Cache Restore Deferred GPU Allocation Plan

## Status

- **State:** Implemented; targeted and supported-path regression validation
  complete
- **Decision:** Model CPU-to-prefill restore using the same deferred target
  allocation abstraction as the existing PDD prefill-to-decode transfer
- **Baseline commit:** `8db430f`
- **Target architecture:** Sequential `pd-disaggregation`
- **Target scheduler:** `vllm_v1`
- **Affected cache mode:** Session prefix caching with prefill-side CPU KV-cache
  offloading
- **Primary issue:** Restore-ready requests can own prefill GPU blocks while
  remaining in the scheduler's waiting state

## Decision Summary

Frontier will stop reserving prefill GPU KV blocks before a CPU restore starts.
An H2D restore completion will mean that the transferred payload has arrived in
a logical staging state, not that it already occupies scheduler-managed GPU KV
blocks.

GPU KV blocks will be allocated only when the restored request is admitted by
the normal prefill scheduler. Allocation, restored-prefix publication, suffix
reservation, and transition to `RUNNING` must occur atomically in the same
scheduling iteration.

The selected flow is:

```text
CPU prefix hit
  -> pin CPU source blocks
  -> model H2D transfer
  -> record a logically staged restore payload
  -> return the request to the runnable prefill waiting queue
  -> revalidate the prefix at admission time
  -> atomically allocate GPU blocks and admit the request
```

This replaces the current flow:

```text
CPU prefix hit
  -> reserve restored-prefix and prompt-suffix GPU blocks
  -> model H2D transfer
  -> leave the GPU-owning request in WAITING
  -> attempt admission later
```

## Why This Model Was Selected

The existing PDD prefill-to-decode path does not reserve decode-side GPU KV
blocks while a transfer is in flight. The transfer completes logically, the
request arrives at the decode scheduler, and decode GPU allocation occurs
during normal scheduler admission.

Using the same abstraction for CPU restore has four important benefits:

1. It preserves the vLLM scheduler invariant that ordinary waiting requests do
   not own scarce GPU KV blocks.
2. It structurally removes the CPU-restore head-of-line dependency in which a
   waiting request both holds the blocks and waits for another queue entry to
   stop requesting those blocks.
3. It avoids adding an `admitted-but-not-runnable` scheduler state, admission
   credits, and special Phase 0 priority rules in the current release.
4. It keeps the CPU restore model consistent with Frontier's existing PDD
   transfer abstraction.

This is a simulator modeling decision, not a claim that a real H2D copy can
finish without a physical destination buffer.

## Problem Being Removed

The current implementation can create this state:

```text
request A:
  state = restore ready, scheduler WAITING
  owns most prefill GPU KV blocks
  has no remaining restore event

request B:
  state = WAITING or partial-prefill RUNNING
  needs an additional prefill GPU KV block

scheduler:
  stops or preempts at B
  does not admit A
  produces no batch and no future event
```

This violates the scheduling contract assumed by the ordinary vLLM two-phase
algorithm:

```text
GPU block owner -> RUNNING work -> future batch event -> eventual release
```

Deferred allocation restores that contract:

```text
restore pending/staged request -> zero GPU blocks
GPU block owner                -> RUNNING request
```

## Goals

1. A pending or completed CPU restore must not change scheduler-managed prefill
   GPU KV occupancy.
2. A staged restore must not appear in the request allocation map before
   admission.
3. Restore completion must always produce either a runnable queued request or a
   terminal cancellation/error.
4. Admission must atomically:
   - revalidate current GPU hits;
   - determine which staged CPU blocks remain usable;
   - reserve all required restored-prefix and suffix/recompute blocks;
   - publish restored prefix blocks;
   - attach the allocation to the request; and
   - move the request to `RUNNING`.
5. CPU source leases, staged payloads, and GPU allocations must each have clear
   and exactly-once ownership.
6. Transfer metrics must remain distinct from the amount of computation
   actually saved at later admission.
7. The original small-GPU P1 reproducer and repeated CPU-eviction stress must
   complete without reservation leaks or non-empty terminal scheduler state.

## Non-Goals

This change will not:

- add a finite receiver-side staging buffer;
- model GPU DMA destination pages during the H2D interval;
- add target-side transfer backpressure;
- make PDD transfers reserve decode GPU memory;
- implement a general vLLM scheduler liveness redesign;
- solve same-session generation races that are independent of GPU reservation;
- enable parallel PDD clusters; or
- change the release-supported architecture and scheduler matrix.

## Required Scheduler Invariants

After this change, the following invariants are mandatory.

### Waiting ownership

```text
request in normal prefill WAITING:
  request-owned GPU KV blocks == 0

request in RESTORE_PENDING:
  request-owned GPU KV blocks == 0

request with staged restore payload:
  request-owned GPU KV blocks == 0
```

### Admission ownership

```text
request-owned GPU allocation created
  <=> allocation and RUNNING admission succeed in the same transaction
```

No successfully allocated restored request may return to ordinary waiting
without first rolling back the entire allocation.

### Event liveness

Every restore request that is not runnable must be covered by one of:

- a scheduled restore end event;
- a runnable queue entry;
- a cancellation path; or
- a deterministic terminal error.

A completed restore payload must not exist only in a side map without an
associated runnable request.

## State Model

The intended state machine is:

```text
NORMAL_WAITING
  |
  | CPU restore selected
  v
RESTORE_PENDING
  - CPU source lease held
  - H2D end event scheduled
  - no GPU allocation
  |
  | H2D completion
  v
RESTORE_STAGED
  - CPU source lease released
  - logical payload retained
  - request returned to runnable prefill queue
  - no GPU allocation
  |
  | scheduler admission succeeds
  v
RUNNING
  - GPU prefix/suffix allocation attached
  - restored prefix published
  - staged payload consumed
```

Cancellation can transition `RESTORE_PENDING` or `RESTORE_STAGED` to a
terminal state. Only cancellation after successful admission uses ordinary GPU
request cleanup.

## Detailed Design

### 1. Split restore transfer from GPU materialization

`VllmV1EngineReplicaScheduler._begin_cpu_kv_cache_restore()` currently performs
both operations:

1. reserve destination and suffix GPU pages; and
2. schedule the H2D latency.

It must perform only the second operation after this change.

Remove from the restore-start path:

```python
kv_cache_manager.can_reserve_tiered_prefix(...)
kv_cache_manager.reserve_tiered_prefix(...)
_sync_prefix_cache_allocation_state(request)
```

The new restore-start path is:

```python
cpu_lease = cpu_manager.pin_restore_blocks(...)
timing = transfer_engine.schedule(direction="h2d", ...)
pending_restores[request.id] = restore_info
pending_restore_requests[request.id] = request
emit CPUKVCacheRestoreStartEvent
```

Starting a restore must not depend on prefill GPU free-block capacity.

### 2. Remove pending restores from the runnable waiting queue

A request with an in-flight restore is not runnable. It should not be repeatedly
removed from and appended to the local waiting deque on every scheduler pass.

When restore starts:

- remove the request from the scheduler's runnable waiting order;
- retain the request in `_cpu_restore_waiting_requests`; and
- include that side state in pending/idle diagnostics.

When restore completes:

- remove it from `_cpu_restore_waiting_requests`;
- create the staged payload;
- reinsert the request into the runnable prefill waiting queue exactly once; and
- emit a `ReplicaScheduleEvent`.

The request's waiting-time accounting should continue across the transfer. It
must not receive duplicate enter-waiting transitions merely because its
eligibility was temporarily suspended.

### 3. Replace destination blocks with an immutable transfer snapshot

`CPUKVCacheRestoreInfo` must not contain preallocated destination
`KVCacheBlock` objects.

The transfer snapshot must contain enough information to determine what arrived
without retaining mutable GPU cache objects:

```python
request
replica_id
dp_id
session_id
block_keys
transferred_cpu_block_indices
lookup_query_blocks
lookup_cpu_query_blocks
lookup_hit_frontier_blocks
block_size
prompt_tokens
cpu_lease
timing
```

The exact dataclass split may use:

- one in-flight `CPUKVCacheRestoreInfo`; and
- one immutable `StagedCPUKVCacheRestore` produced at completion.

The staged object represents transferred contents by block key/index, not GPU
storage ownership.

### 4. Restore completion creates staging, not residency

`complete_cpu_kv_cache_restore()` must:

1. validate that the completion matches the pending operation and request
   generation;
2. release the CPU source lease with `used=True`;
3. remove the pending operation and pending request entries;
4. record transfer completion metrics;
5. create the staged restore payload;
6. return the request to the runnable prefill queue; and
7. leave GPU allocation state unchanged.

It must not call:

```python
publish_restored_prefix(...)
kv_cache_manager.free(...)
_sync_prefix_cache_allocation_state(request)
```

Immediately after completion:

```text
request allocation-map entry = absent
request-owned GPU blocks      = 0
CPU restore lease             = released
staged payload                = present
runnable queue entry          = present
```

### 5. Revalidate the prefix at admission time

The cache state observed before H2D is not stable. GPU cache hits can be evicted
while the transfer is pending or while the staged request waits for admission.

Admission must walk the request prefix from block zero and select a source for
each index:

```text
if the block key is currently present in GPU cache:
  use the current GPU block
else if that index/key exists in the staged CPU payload:
  use staged CPU data
else:
  stop the reusable contiguous prefix
```

Blocks after the first gap cannot contribute to prefix reuse, even if they were
transferred successfully.

Example:

```text
restore-start snapshot:
  GPU hits          = {0, 1}
  transferred CPU  = {2, 3}

admission-time GPU:
  GPU hits          = {1}
  staged CPU        = {2, 3}

index 0 is absent from both sources
  -> effective hit frontier = 0
  -> blocks 0..3 are recomputed
  -> staged blocks 2 and 3 are discarded for this admission
```

This can waste modeled transfer bytes, but it must never overstate reusable
prefix tokens.

Current GPU data should take precedence over staged data when both provide the
same key. That avoids allocating a duplicate restored block.

### 6. Build an admission-time tiered plan

Add a helper conceptually equivalent to:

```python
_build_staged_restore_admission_plan(
    request,
    staged_restore,
) -> TieredPrefixAdmissionPlan
```

The result should include:

```python
effective_hit_frontier_blocks
current_gpu_blocks_by_index
staged_cpu_indices_to_materialize
staged_cpu_indices_discarded
suffix_or_recompute_block_count
block_keys
prompt_tokens
block_size
```

The helper must not mutate GPU state.

### 7. Allocate and publish atomically during admission

After normal token-budget, max-running-request, and fast-lane eligibility checks
have passed, Phase 2 should perform one admission transaction:

1. verify capacity for:
   - staged CPU blocks inside the effective frontier; and
   - every block from the effective frontier through the required prompt
     reservation;
2. attach current GPU hit blocks;
3. allocate new blocks for staged CPU contents;
4. allocate suffix/recompute blocks;
5. publish only the staged blocks inside the effective prefix frontier;
6. synchronize the request allocation map;
7. advance scheduler-computed token state;
8. remove the staged payload;
9. remove the request from waiting; and
10. append the request to `RUNNING` and the scheduled batch.

If any mutation fails, all mutations from that transaction must be rolled back.
The request must retain its staged payload and return to zero request-owned GPU
blocks.

The existing `reserve_tiered_prefix()` API can be refactored into an
admission-only allocation primitive, but its name and documentation should no
longer imply that restore-start reservation is supported.

### 8. Handle insufficient GPU memory like ordinary waiting admission

If admission capacity is temporarily insufficient:

- do not consume the staged payload;
- allocate no request-owned GPU blocks;
- leave the request in waiting; and
- apply the normal scheduling policy.

An FCFS `break` is safe with respect to the CPU-restore deadlock once the staged
request holds no GPU blocks. The memory shortage must be attributable to
running work that can produce a later event and release memory.

Separately, if a request cannot fit on an otherwise empty target, Frontier
should fail deterministically rather than drain the event queue. That generic
request-fit validation may be implemented in a separate change if it is outside
the CPU restore patch.

### 9. Preserve CPU source safety

CPU source blocks remain pinned only while the modeled H2D copy is in flight.

```text
RESTORE_PENDING:
  source CPU blocks pinned

RESTORE_STAGED:
  source lease released
  logical payload independent of later CPU eviction
```

CPU eviction after restore completion must not invalidate an already staged
payload.

### 10. Simplify cancellation and stale-event cleanup

Pending cancellation:

- remove the matching pending operation;
- release the CPU lease with `used=False`;
- remove the pending request entry;
- remove a not-yet-emitted auxiliary start event when applicable; and
- perform no GPU cleanup.

Staged cancellation:

- remove the staged payload;
- remove the runnable queue entry; and
- perform no GPU cleanup.

Post-admission cancellation uses the existing request GPU cleanup path.

Duplicate or stale completion must not:

- release a newer lease;
- enqueue the request twice;
- consume another request's staged payload; or
- mutate GPU allocation state.

## Metrics Contract

Deferred allocation separates “bytes transferred” from “computation saved.”
The metrics contract must reflect that distinction.

### Transfer-time metrics

Record from the completed H2D operation:

- restore operation count;
- transferred blocks;
- transferred bytes;
- queue time;
- service time; and
- completion time.

These values describe modeled traffic even if some payload is later discarded.

### Admission-time reuse metrics

Record after prefix revalidation:

- current GPU blocks used;
- staged CPU blocks materialized and used;
- effective reusable prefix frontier;
- cached/reused tokens; and
- staged blocks discarded because of a prefix gap or a newer GPU hit.

The following relationship should be allowed:

```text
transferred CPU blocks > staged CPU blocks used
```

Request and system metrics must not report transferred blocks as saved
computation unless those blocks are inside the effective admission-time
frontier.

If existing metrics cannot express this distinction clearly, add counters
equivalent to:

```text
cpu_kv_cache_restore_transferred_blocks
cpu_kv_cache_restore_consumed_blocks
cpu_kv_cache_restore_discarded_blocks
```

Existing byte metrics should continue to represent actual modeled transfer
bytes for backward compatibility.

## File-Level Change Plan

### Replica scheduler

`frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`

- remove GPU capacity checks and reservations from restore start;
- suspend pending restore requests outside the runnable waiting queue;
- stage payloads and requeue requests at restore completion;
- add admission-time prefix revalidation;
- add transactional staged-restore admission;
- remove pending-restore skip/tail-reinsertion behavior;
- include pending and staged restore states in idle diagnostics; and
- keep request waiting-time transitions exactly once.

### GPU KV-cache manager

`frontier/kv_cache/base_kv_cache_manager.py`

- replace restore-start reservation semantics with admission-only allocation;
- add a non-mutating capacity check for the refreshed admission plan;
- allocate staged-prefix and suffix blocks atomically;
- publish staged blocks only after allocation succeeds; and
- provide complete rollback on failure.

### Transfer and staged-payload entities

`frontier/entities/cpu_kv_cache_transfer_info.py`

- remove destination `KVCacheBlock` ownership from in-flight restore info;
- store an immutable transfer snapshot;
- optionally introduce `StagedCPUKVCacheRestore`; and
- retain generation/identity information needed for stale-event validation.

### Events

`frontier/events/cpu_kv_cache_restore_start_event.py`

- retain transfer timing behavior.

`frontier/events/cpu_kv_cache_restore_end_event.py`

- redefine completion as logical payload arrival;
- ensure the request is requeued exactly once; and
- continue to wake the target prefill replica.

### Metrics

`frontier/entities/request.py`

`frontier/metrics/constants.py`

`frontier/metrics/metrics_store.py`

- separate transfer-time traffic from admission-time consumed reuse;
- add discarded-staging accounting if needed; and
- keep request/system byte totals consistent.

### Documentation

`docs/roadmap/prefill-cpu-kv-offloading-plan.md`

- add a concise design-decision section that links to this document.

`docs/roadmap/cpu-kv-offload-restore-hol-deadlock.md`

- link the selected remediation plan now;
- after validation, record the resolving commit and regression results.

`docs/cli/README.md`

- add a user-facing modeling note describing logical staging and the absence of
  receiver-side capacity/backpressure.

## Unit Test Plan

### Restore start does not allocate GPU memory

Assert before and after restore start:

```text
GPU used blocks unchanged
request absent from allocation map
request owns zero GPU blocks
CPU source lease held
restore end event scheduled
```

### Restore completion does not allocate GPU memory

Assert after restore end:

```text
GPU used blocks unchanged
CPU source lease released
staged payload present
request in runnable prefill queue exactly once
request owns zero GPU blocks
```

### Admission success materializes once

With sufficient GPU memory:

```text
current GPU hits attached
staged CPU blocks allocated and published
suffix/recompute blocks allocated
staged payload removed
request added to RUNNING
request scheduled in the same iteration
```

### Admission failure is non-mutating

With temporary GPU pressure:

```text
request remains WAITING
staged payload remains present
request owns zero GPU blocks
other request allocations unchanged
no partial prefix publication
```

### GPU churn between transfer and admission

Cover:

- an original GPU hit remains resident;
- an original GPU hit is evicted;
- a new GPU hit appears;
- a gap truncates the effective prefix;
- transferred blocks beyond a gap are discarded; and
- the full-prompt last-block demotion rule remains correct.

### CPU eviction after completion

Complete restore, evict the source CPU session, then admit the request. The
staged payload must remain valid.

### Cancellation and stale completion

Cover:

- pending cancellation;
- staged cancellation;
- duplicate restore completion;
- completion from an older request/session generation;
- admission exception rollback; and
- final zero CPU leases and zero staged payloads.

## End-to-End and Stress Test Plan

### Original head-of-line reproducer

Configuration:

- four distinct sessions;
- three turns per session;
- 100 ms intra-round arrivals;
- five prefill GPU KV blocks;
- large CPU tier; and
- CPU offloading enabled.

Acceptance:

- all 12 requests complete;
- CPU restores occur;
- the event queue terminates normally;
- final request-owned GPU allocations are zero;
- final CPU restore leases are zero; and
- final staged payload count is zero.

### CPU-offload-disabled control

Run the same trace without CPU offloading. All requests must continue to
complete, proving that the request shapes fit the configured GPU target.

### Repeated CPU eviction

Configuration:

- at least 16 sessions;
- at least three turns;
- five prefill GPU KV blocks; and
- an eight-block CPU tier.

Acceptance:

- repeated D2H offloads and H2D restores;
- multiple session/block evictions;
- no deadlock;
- CPU resident plus reserved blocks never exceed capacity;
- no final leases or staged payloads; and
- request/system transfer byte totals match.

### One-block CPU tier

Validate repeated truncation and recomputation:

- no prefix hit crosses a missing block;
- staged reuse never crosses an admission-time gap;
- no negative capacity state;
- all requests that individually fit complete; and
- all final resources return to zero.

### Slow serialized transfer queue

Use very low D2H/H2D bandwidth and simultaneous follow-ups:

- transfer queue time must be positive;
- transfers must follow the configured serialization model;
- pending restores must not change GPU used blocks;
- CPU leases must release at completion; and
- requests must eventually materialize or cancel.

### Multiple cache targets

Exercise multiple prefill DP targets and replicas:

- each staged payload returns to its original `(replica_id, dp_id)`;
- CPU capacity remains target-local;
- GPU admission remains target-local;
- session affinity remains intact; and
- aggregate metrics equal the per-target sum.

## Documentation of Modeling Limitations

The implementation and user documentation must state the following explicitly:

> CPU restore completion represents arrival into an abstract logical staging
> area. It does not represent completion into scheduler-managed prefill GPU KV
> pages.

Consequences:

1. The staging area has no modeled capacity limit.
2. Retaining a staged payload until scheduler admission has zero modeled memory
   cost.
3. An H2D transfer can complete while the prefill GPU KV pool is full.
4. There is no target-side transfer credit or backpressure.
5. GPU materialization at admission has zero additional transfer latency.
6. Cache churn can make some transferred blocks unusable before admission.
7. Transfer and compute overlap may be more optimistic than on real hardware.
8. The model is internally consistent with current PDD target allocation but is
   not a physical DMA-buffer model.

These limitations are accepted for the current release in exchange for
scheduler consistency and deterministic liveness. A higher-fidelity receiver
model can replace logical staging later without changing the external CPU-cache
configuration surface.

## Alternatives Considered

### GPU reservation plus a restore-ready fast lane

Rejected as the primary fix because queue promotion only addresses waiting
head-of-line ordering. It does not by itself handle Phase 1 preemption that
suppresses Phase 2 admission.

### GPU reservation plus admission credit

This is the preferred future high-fidelity direction. Starting a restore would
reserve:

- destination GPU pages;
- a running/admission slot; and
- guaranteed scheduler priority after completion.

It was not selected for the current release because it requires a new
`admitted-but-transfer-pending` state, token/slot reservation semantics,
preemption-policy changes, and more complex rollback.

### Restore only CPU blocks and reserve suffix later

Rejected because a completed restore could still occupy destination GPU pages
without enough memory to compute the suffix. It reduces reservation size but
does not remove the resource-owning waiting state.

### Increase GPU capacity

Rejected because it only moves the failure boundary and does not repair the
scheduler invariant.

## Implementation Sequence

1. Add invariant-focused tests that demonstrate pending and completed restores
   currently own GPU blocks.
2. Introduce the immutable staged-payload representation.
3. Remove restore-start GPU allocation and adjust cancellation.
4. Suspend pending restores outside the runnable waiting queue.
5. Requeue staged requests on restore completion.
6. Implement admission-time prefix revalidation.
7. Implement transactional GPU materialization and admission.
8. Split transfer and consumed-reuse metrics.
9. Run targeted unit tests.
10. Run the original P1 reproducer and CPU-capacity churn matrix.
11. Update user-facing limitations and the P1 resolution record.
12. Run the full supported sequential-PDD regression suite.

Each step should leave CPU offloading disabled-by-default behavior unchanged.

## Implementation and Validation Result

The scheduler now uses an immutable staged restore payload and does not reserve
or publish prefill GPU KV blocks at restore start or restore completion.
Admission revalidates the current GPU prefix, combines it with the still-usable
staged CPU indices, checks the complete prompt reservation plus watermark, and
materializes the result only in the successful admission path.

Implemented diagnostics expose both `pending_restore_operations` and
`staged_restore_payloads`. Both counters, CPU restore reservations, and
request-owned GPU allocations return to zero at normal simulation completion.

Validation completed on 2026-07-28:

- 106 targeted CPU KV-cache, transfer-completion, tiered-cache, and session
  prefix-cache tests passed;
- the five-block GPU reproducer completed for 4 sessions x 3 rounds;
- the eight-block CPU tier completed for 4 and 16 sessions with repeated
  eviction;
- the formal E2E matrix completed with prefill `DP=2`, two independent cache
  targets, and an eight-block CPU tier per target;
- the existing slow-H2D latency E2E test passed and attributed the added
  latency to follow-up TTFT;
- 70 adjacent prefix-cache, layout, example-PDD, and session-prefix regression
  tests passed, with one environment-conditional skip; and
- Python compilation and whitespace validation passed for the changed files.

The repository-wide unit collection still contains unrelated baseline and
environment failures, including a pre-existing duplicate `from __future__`
syntax error, unavailable optional `torch` profiling dependencies, missing
debug/config-optimizer fixtures, and Windows shell/path assumptions. These are
outside this change; the affected and release-supported paths above are green.

## Acceptance Criteria

The plan is complete when:

1. Restore start and restore completion do not change prefill GPU used blocks.
2. Pending and staged restore requests have no allocation-map entry.
3. GPU allocation is created only during successful scheduler admission.
4. Allocation, publish, staged-payload consumption, and transition to
   `RUNNING` are atomic.
5. Admission-time cache churn cannot overstate the reusable prefix frontier.
6. The original five-block P1 reproducer completes.
7. The eight-block CPU eviction and one-block truncation stress cases complete.
8. Slow transfer and multi-target stress cases complete.
9. CPU leases, staged payloads, and request-owned GPU allocations are zero at
   terminal state.
10. Transfer byte metrics remain exact and consumed-reuse metrics reflect the
    admission-time frontier.
11. Existing CPU-offload-disabled, prefix-cache, and sequential-PDD tests remain
    green.
12. The logical-staging rationale and limitations are documented in both the
    roadmap and user-facing CLI guide.
