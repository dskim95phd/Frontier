# C++ Porting Step 3: Dense Sequential PDD Parity

## Status

Complete (2026-07-30).

Validation evidence:

- the 17-case Step 3 production-Python/C++ differential matrix passes;
- an additional 44-case configuration/workload stress matrix passes across
  fixed and analytical timing, online and offline arrivals, TP/PP/DP,
  multi-replica routing, transfer pressure, and preemption pressure;
- the full checked-in differential suite passes 66/66, including all 49
  Step 1-2.5 regression cases;
- the Debug CTest suite passes 19/19, including the dedicated sequential-PDD
  integration target;
- a clean WSL GCC build completes without compiler warnings; and
- the AddressSanitizer plus UndefinedBehaviorSanitizer CTest suite passes
  19/19 with leak detection and halt-on-error enabled.

The implementation keeps the shared dense handlers in
`co_location_events.*`; the optional mechanical rename described below was
not needed for parity.

Step 3 starts from the completed Step 2.5 implementation:

- production-shaped global, cluster, replica, and replica-stage scheduling;
- the nine typed dense scheduling events;
- multiple replicas and independent DP targets;
- dense TP, PP, and DP execution;
- fixed per-stage and analytical stage timing;
- target-local KV accounting, preemption, and stale-event protection; and
- exact Python/C++ parity over the Step 2.5 matrix.

Step 3 must extend those contracts to sequential `pd-disaggregation`. It must
not build a second flattened PDD simulator beside the Step 2.5 event and
scheduler hierarchy.

## Goal

End Step 3 with deterministic dense PDD runs in which:

1. requests enter a `PREFILL` cluster;
2. prefill execution uses the same vLLM V1, TP, PP, DP, and multi-replica
   machinery validated in Step 2.5;
3. completed prefill KV remains allocated at its source target until transfer
   completion;
4. one analytical KV transfer is emitted for each decode-bound request;
5. the request enters a unified `DECODE` cluster only after that transfer
   completes;
6. decode uses an independently configured topology and scheduler state;
7. online and offline decode admission match production Python; and
8. event, routing, scheduler, stage, transfer, request-metric, and final-state
   outputs match Python.

“Sequential” means that C++ uses one deterministic discrete-event queue, as
the release-supported Python path does when `enable_parallel_clusters=False`.
It does not mean that all prefill work must finish before all decode work in
online mode. Events from both clusters may be interleaved on the simulated
timeline; they are processed serially in `(time, creation_sequence)` order.

## Scope

### Included

- `system_architecture=pd-disaggregation`;
- `enable_parallel_clusters=false`;
- dense Llama-2-7B;
- FP16;
- vLLM V1 replica scheduling;
- FCFS request scheduling;
- round-robin cluster routing;
- fixed per-stage execution timing;
- the existing analytical dense execution model;
- analytical PREFILL-to-DECODE KV transfer;
- offline and online simulation modes;
- independent PREFILL and DECODE:
  - replica counts,
  - TP sizes,
  - PP sizes,
  - DP sizes,
  - scheduler budgets,
  - KV block capacities, and
  - fixed or analytical execution configurations;
- continuous batching, chunked prefill, and ordinary recompute preemption;
- multiple requests completing prefill in one batch;
- simultaneous and staggered KV transfers;
- deterministic transfer and cluster traces; and
- schema v1-v3 regression preservation.

### Explicitly Deferred

- parallel cluster threads;
- `pd-af-disaggregation`;
- M2N transfers;
- prefix caching of any kind;
- `block_hash` input;
- CPU KV-cache offload or restore;
- transfer compression;
- transfer latency hiding;
- request-payload transfer modeling;
- non-analytical KV transfer predictors;
- MoE, EP, and MoE synchronization events;
- speculative decoding / MTP;
- Thinking Mode and multiple prefill/decode rounds;
- CUDA Graph timing;
- SGLang and other replica schedulers;
- sticky, LOR, random, or load-aware cluster routing;
- profile-trained execution predictors; and
- heterogeneous model families or devices.

These are not prerequisites for proving the dense sequential PDD contract.
They must not be folded into Step 3 merely because production Python has
branches for them.

## Python Runtime Contract

The production Python implementation remains the oracle. The baseline path is:

```text
RequestArrival(PREFILL)
  -> GlobalSchedule
  -> ClusterSchedule(PREFILL)
  -> ReplicaSchedule(PREFILL)
  -> BatchStageArrival(PREFILL)
  -> ReplicaStageSchedule(PREFILL)
  -> BatchStageEnd(PREFILL)
  -> ClusterBatchEnd(PREFILL)
       -> KVCacheTransferStart(PREFILL -> DECODE), once per request
       -> ReplicaSchedule(PREFILL)
  -> KVCacheTransferEnd(PREFILL -> DECODE)
       -> release retained source KV
       -> add request to DECODE cluster queue
       -> ClusterSchedule(DECODE), subject to mode policy
  -> ReplicaSchedule(DECODE)
  -> BatchStageArrival(DECODE)
  -> ReplicaStageSchedule(DECODE)
  -> BatchStageEnd(DECODE)
  -> ClusterBatchEnd(DECODE)
  -> GlobalBatchEnd(DECODE)
       -> ReplicaSchedule(DECODE)
       -> repeat until request completion
```

The same nine Step 2.5 scheduling event types remain shared by both clusters.
Step 3 adds only the two inter-cluster event types:

- `KVCacheTransferStartEvent`; and
- `KVCacheTransferEndEvent`.

Every scheduling, batch, stage, and completion event must additionally carry a
`cluster_type`. A PREFILL event must never resolve state from the DECODE
cluster, even when replica and DP IDs happen to be equal.

### PREFILL Completion

At `ClusterBatchEndEvent(PREFILL)`:

1. validate the batch generation and request snapshots using the existing
   Step 2.5 stale-event contract;
2. advance prefill request state;
3. call PREFILL replica-scheduler batch completion;
4. remove a fully prefetched request from the runnable set;
5. retain its allocated source KV blocks;
6. add the request to a pending-transfer ownership set;
7. predict full-request KV size and transfer duration;
8. emit one transfer-start event per request; and
9. schedule more PREFILL work if capacity permits.

Chunked prefill does not emit a transfer until the request's entire prefill is
complete.

When several requests complete prefill in the same batch, Python creates
separate one-request transfer batches and emits transfers in batch request
order. C++ must preserve the same observable request and transfer ordering,
while using stable IDs instead of Python object references.

### Transfer Start

At `KVCacheTransferStartEvent`:

1. validate that the request is owned by the declared PREFILL target and is in
   pending-transfer state;
2. create a transfer record with a stable `TransferId`;
3. record source cluster, source replica, source DP, target cluster, request,
   byte count, predicted milliseconds, and start time;
4. mark the request transfer as started;
5. emit `KVCacheTransferEndEvent` at
   `start_time + transfer_time_ms * 1e-3`, matching Python's floating-point
   operation order; and
6. leave source KV allocated.

Zero network latency is valid. Equal-time start and end events still obey event
creation sequence.

### Transfer End

At `KVCacheTransferEndEvent`:

1. validate the transfer generation and current source ownership;
2. mark the transfer complete exactly once;
3. record request transfer end and duration;
4. enqueue the request at the DECODE cluster;
5. release the source PREFILL KV allocation exactly once;
6. clear the PREFILL pending-transfer state;
7. reschedule a blocked PREFILL target when the released blocks make progress
   possible; and
8. emit DECODE scheduling according to online/offline policy.

The target arrival and source release happen at the same event time. Their
mutation and emitted-event order must match the Python oracle. A duplicate or
stale transfer end must not double-release blocks or enqueue the request
twice.

### Online Decode Admission

In online mode, each transfer completion adds its request to the DECODE queue
and immediately emits a DECODE `ClusterScheduleEvent`. Decode may therefore
overlap later PREFILL work on the simulated timeline even though the event loop
itself is single-threaded.

### Offline Decode Admission

In offline mode, Python buffers DECODE arrivals until all generated
decode-bound requests have completed transfer. The request generator records
that expected count before simulation starts.

C++ must:

- derive the expected decode-bound count from the normalized workload;
- reject an arrival count greater than the expected count;
- emit no DECODE cluster schedule before the count is reached; and
- emit one DECODE cluster schedule when the final expected transfer arrives.

This barrier is admission behavior, not a reason to give PREFILL and DECODE
separate event queues.

### Decode Completion and First-Decode-Token Timing

DECODE uses the shared stage pipeline. Dense
`ClusterBatchEndEvent(DECODE)` emits `GlobalBatchEndEvent(DECODE)`.

The first DECODE token timestamp follows Python's PDD rule: when a valid decode
completion is finishing decode token index 1, the first-decode-token timestamp
is stamped before request token state is mutated.

Do not conflate this field with Frontier's canonical TTFT. Production Python
defines canonical TTFT as `prefill_completed_at - arrived_at`; KV transfer and
the first DECODE iteration are tracked separately through transfer and
first-decode-token timestamps. C++ must preserve both contracts exactly rather
than redefining TTFT for PDD.

## Configuration Contract

Introduce normalized schema v4. Schemas v1, v2, and v3 remain frozen and retain
their existing parsers, serializers, output fields, and simulator paths.

An illustrative schema v4 configuration is:

```json
{
  "schema_version": 4,
  "run_id": "step3-dense-pdd",
  "simulation_mode": "online",
  "system_architecture": "pd-disaggregation",
  "enable_parallel_clusters": false,
  "prefix_cache": {
    "enabled": false,
    "key_mode": "session"
  },
  "cluster_scheduler": {
    "type": "round_robin"
  },
  "clusters": {
    "prefill": {
      "parallelism": {
        "num_replicas": 2,
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 2,
        "data_parallel_size": 2
      },
      "scheduler": {
        "type": "vllm_v1",
        "scheduling_policy": "fcfs",
        "batch_size_cap": 8,
        "max_tokens_in_batch": 128,
        "enable_preemption": true,
        "enable_chunked_prefill": true,
        "long_prefill_token_threshold": 0,
        "block_size": 16,
        "num_blocks": 128,
        "watermark_blocks_fraction": 0.0,
        "num_preallocate_tokens": 0
      },
      "execution_model": {
        "type": "fixed",
        "stage_latencies_ms": [1.0, 2.0]
      }
    },
    "decode": {
      "parallelism": {
        "num_replicas": 1,
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 4,
        "data_parallel_size": 2
      },
      "scheduler": {
        "type": "vllm_v1",
        "scheduling_policy": "fcfs",
        "batch_size_cap": 8,
        "max_tokens_in_batch": 64,
        "enable_preemption": true,
        "enable_chunked_prefill": false,
        "long_prefill_token_threshold": 0,
        "block_size": 16,
        "num_blocks": 256,
        "watermark_blocks_fraction": 0.0,
        "num_preallocate_tokens": 0
      },
      "execution_model": {
        "type": "fixed",
        "stage_latencies_ms": [1.0, 1.0, 2.0, 1.0]
      }
    }
  },
  "kv_cache_transfer": {
    "type": "analytical",
    "network_bandwidth_gbps": 200.0,
    "network_latency_ms": 0.5,
    "kv_cache_dtype_size_bytes": 2,
    "enable_compression": false
  }
}
```

The analytical execution-model form is the schema-v3 form inside each cluster.
Each cluster's `parallelism.tensor_parallel_size` remains the source of truth
for its stage predictor.

### Validation

- schema v4 requires `system_architecture=pd-disaggregation`;
- `enable_parallel_clusters` must be false;
- exactly `prefill` and `decode` clusters must exist;
- all parallelism dimensions must be positive;
- each cluster independently satisfies layer/TP/PP topology rules;
- each cluster independently fits the supported hardware-domain constraint;
- fixed stage latency count must equal that cluster's PP size;
- scheduler and KV capacity apply independently to every target in that
  cluster;
- PREFILL and DECODE may have different replica, TP, PP, and DP sizes;
- only `round_robin`, `vllm_v1`, `fcfs`, and analytical KV transfer are
  accepted;
- transfer bandwidth must be finite and positive;
- transfer latency must be finite and nonnegative;
- KV dtype size must be finite and positive;
- compression must be false;
- prefix caching must be disabled;
- `block_hash` workload fields remain rejected; and
- every unsupported Step 3 feature combination fails before simulator
  construction.

## KV Transfer Formula

Step 1 already provides the pure C++ analytical transfer helper. Step 3 wires
it into runtime events instead of introducing a second formula.

For the dense Python oracle:

```text
size_bytes =
  num_prefill_tokens
  * num_layers
  * runtime_num_kv_heads
  * runtime_head_size
  * 2
  * dtype_size_bytes

bandwidth_bytes_per_ms =
  network_bandwidth_gbps * 1e9 / (8 * 1000)

transfer_time_ms =
  network_latency_ms + size_bytes / bandwidth_bytes_per_ms
```

The transfer is for the request's full prefill KV, not only the last prefill
chunk. The production predictor currently derives the full runtime KV layout
from the model configuration. C++ must freeze Python goldens across multiple
PREFILL TP and PP values rather than infer a new per-worker sharding rule.

Integer byte counts compare exactly. Only transfer duration is a
floating-point tolerance field.

## C++ Architecture

Headers and implementations remain colocated in functional directories. Step
3 should move toward the following shape:

```text
cpp/frontier/
  entities/
    request.*
    batch.*
    batch_stage.*
    cluster.*
    replica.*
    kv_cache_transfer_info.*
  events/
    base_event.h
    dense_scheduling_events.*
    kv_cache_transfer_events.*
  scheduler/
    global_scheduler/
    cluster_scheduler/
    replica_scheduler/
    replica_stage_scheduler/
  kv_cache_transfer/
    analytical_transfer.*
  simulator/
    simulation_context.*
    co_location_simulator.*
    sequential_pdd_simulator.*
```

This preserves the agreed functional layout and does not introduce separate
`include/` and `src/` trees.

The existing `co_location_events.*` handlers should be generalized, not copied
into a PDD event file. A mechanical rename to `dense_scheduling_events.*` may
be done after both architectures use the handlers. The only PDD-specific event
module is the inter-cluster KV transfer module.

### Cluster Identity

Extend `ClusterType` with:

- `kMonolithic`;
- `kPrefill`; and
- `kDecode`.

Add cluster identity to event payloads, batch ownership, stage records,
scheduler traces, and lookup APIs. `ReplicaId` and `DataParallelId` are only
unique within a cluster, so a complete runtime target is:

```text
(cluster_type, replica_id, dp_id)
```

No state table may use only `(replica_id, dp_id)` when both PDD clusters are
active.

### Simulation Context

Generalize `SimulationContext` from one `cluster_`, one topology, and one
request target into:

- a deterministic cluster map containing PREFILL and DECODE runtimes;
- cluster-local topology and execution models;
- cluster-local scheduler ownership;
- per-cluster request target assignments;
- shared request, batch, stage, and transfer arenas;
- globally unique batch and transfer IDs;
- expected and arrived offline decode counts; and
- one shared deterministic event queue.

The context must expose typed lookup functions that require `ClusterType`.
Avoid nullable “current cluster” state.

`SequentialPddSimulator` should only construct the context, seed arrival
events, dispatch events, and finalize. It must not own scheduler or request
transition logic.

### Global Scheduler

Promote the current co-location global scheduler into a cluster-keyed global
scheduler:

- one deterministic request queue per cluster type;
- explicit `get_cluster_scheduler(cluster_type)`;
- deterministic cluster iteration order;
- PREFILL routing for initial PDD arrivals; and
- no direct PREFILL-to-DECODE route that bypasses a transfer end.

Do not use unordered-set iteration to create equal-time cluster schedule
events. Step 2.5 already established that oracle-visible ordering must be
stable.

### Cluster Scheduler

Generalize the co-location round-robin cluster scheduler to own a
`ClusterType`. Instantiate it independently for PREFILL and DECODE.

Each instance has:

- its own round-robin cursor;
- its own replica/DP target matrix;
- its own request queue;
- its own scheduler configuration; and
- its own target assignment trace.

The DECODE scheduler additionally exposes the KV-arrival entry point and the
offline admission barrier. Transfer completion is the only caller of that
entry point in Step 3.

### Replica Scheduler

Pass `ClusterType` into each vLLM V1 scheduler instance.

Shared Step 2.5 behavior remains:

- token budgets;
- batch-size caps;
- running/waiting/preempted queues;
- target-local block accounting;
- multi-in-flight PP batches;
- request/batch epochs; and
- recompute preemption.

PREFILL-specific behavior:

- schedule only remaining prefill tokens;
- keep partial chunked prefills runnable;
- remove fully prefetched requests from the runnable set;
- keep their block allocations in a pending-transfer set; and
- free those allocations only on successful transfer end.

DECODE-specific behavior:

- accept requests only after KV arrival;
- preserve already processed prefill token state;
- allocate/account for the imported KV footprint according to Python vLLM V1
  admission behavior;
- schedule decode iterations without re-running prefill; and
- free DECODE blocks only on completion or recompute preemption.

Source and target block pools are independent. Releasing PREFILL blocks must
not mutate DECODE capacity, and importing KV at DECODE must not reuse a source
allocation object.

### Request State

Keep one logical request entity but add explicit cluster-scoped lifecycle
state:

- initial PREFILL arrival time;
- PREFILL owner target;
- PREFILL schedule/execution epochs;
- prefill completion time;
- transfer state, start, end, duration, and byte count;
- DECODE arrival time;
- DECODE owner target;
- DECODE schedule/execution epochs;
- first DECODE token time; and
- final completion time.

The request must transition through a validated lifecycle:

```text
PREFILL_WAITING
  -> PREFILL_RUNNING
  -> PREFILL_TRANSFER_PENDING
  -> KV_TRANSFER_IN_FLIGHT
  -> DECODE_WAITING
  -> DECODE_RUNNING
  -> COMPLETED
```

Preemption is a cluster-local sub-transition and must not erase the completed
PREFILL frontier after handoff.

### Transfer Entity

Add an ID-based `KVCacheTransferInfo` arena record containing:

- `TransferId`;
- request ID;
- transfer batch ID if required for Python trace parity;
- source cluster, replica, and DP;
- target cluster;
- source request runtime/execution epoch snapshots;
- source allocation generation;
- exact KV byte count;
- predicted transfer milliseconds;
- start time;
- optional end time; and
- pending/completed state.

Events carry `TransferId`, not pointers. Completion validates the stored
generation before mutating source or target state.

## Output Contract

Schema v4 output adds cluster and transfer visibility while preserving v1-v3
serialization.

### Request Records

- request ID;
- original arrival;
- PREFILL replica and DP;
- PREFILL completion time;
- transfer ID;
- KV transfer bytes;
- transfer start, end, and duration;
- DECODE arrival;
- DECODE replica and DP;
- first DECODE token time;
- completion time;
- processed-token count;
- preemption count split or traceable by cluster; and
- canonical TTFT and E2E metrics.

### Batch and Stage Records

Add `cluster_type` to:

- batches;
- batch stages;
- scheduler traces; and
- all relevant event records.

Batch IDs remain globally unique. Scheduler iteration IDs remain local to a
cluster target and are compared together with the complete target key.

### Transfer Records

Add a `kv_cache_transfers[]` collection containing:

- transfer and request IDs;
- source and target identifiers;
- exact bytes;
- predicted milliseconds;
- start and end timestamps; and
- completion state.

### Exact Versus Tolerant Comparison

Compare exactly:

- IDs;
- cluster types;
- target assignments;
- event order;
- queue and completion order;
- token and block counts;
- transfer counts;
- transfer byte counts;
- batch membership;
- stage indices; and
- final empty-state counts.

Use tolerance only for named floating-point fields:

- event and metric timestamps;
- fixed/analytical execution components;
- transfer duration; and
- derived latency and throughput values.

An integer on one side and a floating-point value on the other is a type
mismatch for ID, count, and byte fields.

## Implementation Slices

### Step 3A: Freeze the PDD Oracle and Schema v4

- add normalized schema-v4 fixtures;
- add Python oracle export for cluster, transfer, and cluster-scoped request
  fields;
- freeze a one-request TP1/PP1/DP1 fixed-timing PDD trace;
- freeze online and offline admission traces;
- freeze analytical KV-size and transfer-time goldens; and
- add negative config fixtures.

Acceptance:

- the oracle invokes production Python events and schedulers;
- it does not reproduce PDD routing or transfer behavior in the test harness;
- equal-time Python event order is deterministic;
- schema v1-v3 goldens remain byte-stable; and
- unsupported feature combinations fail before simulation.

### Step 3B: Generalize Cluster-Scoped Runtime Ownership

- extend `ClusterType`;
- key all target/state lookups by cluster;
- replace the single cluster/topology in `SimulationContext` with a
  deterministic cluster map;
- instantiate independent PREFILL and DECODE entities and schedulers;
- generalize the shared nine event handlers; and
- add `cluster_type` to batch, stage, scheduler, and event traces.

Acceptance:

- co-location schema v3 still passes unchanged;
- identical numeric replica/DP IDs in two clusters cannot alias;
- both cluster trees are independently queryable; and
- the simulator loop remains free of event-type business logic.

### Step 3C: PREFILL-Only PDD Execution

- seed all initial PDD requests into PREFILL;
- implement PREFILL-only scheduling and chunked prefill;
- stop request token progression at the prefill boundary;
- retain source block allocations after prefill completion;
- create pending-transfer ownership state; and
- verify PREFILL rescheduling under retained-KV pressure.

Acceptance:

- no request enters DECODE yet;
- no decode token is scheduled in PREFILL;
- partial prefills do not transfer;
- fully prefetched requests leave the runnable queue but keep source blocks;
- retained blocks can temporarily block later PREFILL requests; and
- PREFILL event, stage, and scheduler traces match Python.

### Step 3D: Analytical Transfer Entity and Events

- add `TransferId` and the transfer arena;
- connect the Step 1 analytical transfer helper;
- add start and end event handlers;
- emit one transfer per completed request in batch order;
- record transfer metrics; and
- release source allocations on valid transfer end.

Acceptance:

- exact byte and transfer-count parity;
- duration parity within tolerance;
- source KV remains allocated throughout the transfer;
- source KV is released exactly once at completion;
- a stale-generation completion changes no state, while a duplicate completion
  for the current generation fails before any second mutation;
- simultaneous transfers have deterministic order; and
- a transfer completion can unblock PREFILL work.

### Step 3E: Online DECODE Admission and Execution

- enqueue each request into DECODE at transfer end;
- immediately trigger online DECODE cluster scheduling;
- route with an independent round-robin cursor;
- allocate/account for imported KV at the DECODE target;
- run decode batches through the shared PP stage pipeline;
- implement PDD first-token timing; and
- complete and release DECODE requests.

Acceptance:

- DECODE scheduling never precedes transfer end;
- DECODE never recomputes PREFILL in the ordinary path;
- PREFILL and DECODE work may interleave on one event timeline;
- independent topologies route and execute correctly; and
- request completion, TTFT, E2E, and target traces match Python.

### Step 3F: Offline DECODE Barrier

- derive `num_decode_bound_requests` from the workload;
- buffer DECODE arrivals;
- trigger no DECODE scheduling before the final expected arrival;
- trigger scheduling when the barrier is satisfied; and
- add over-arrival and quiescence diagnostics.

Acceptance:

- the first offline DECODE schedule occurs at the final transfer-end time;
- arrival and queue order match Python;
- the barrier counts requests, not transfer events or batches; and
- all offline requests complete with empty cluster state.

### Step 3G: Independent Parallelism and Capacity Pressure

Run nontrivial independent PREFILL/DECODE configurations:

```text
P: replicas=1, DP=1, PP=1, TP=1
D: replicas=1, DP=1, PP=1, TP=1

P: replicas=2, DP=2, PP=2, TP=2
D: replicas=1, DP=2, PP=4, TP=4

P: replicas=1, DP=2, PP=4, TP=2
D: replicas=2, DP=1, PP=2, TP=4

P: replicas=1, DP=1, PP=8, TP=1
D: replicas=1, DP=2, PP=4, TP=2
```

Workloads include:

- same-time bursts larger than PREFILL capacity;
- staggered online arrivals while both clusters have live work;
- mixed prefill and decode lengths;
- chunked prefills;
- multiple requests finishing prefill in one batch;
- nonzero transfer latency with overlapping transfers;
- zero transfer latency and equal-time ordering;
- tight PREFILL blocks that remain pinned during transfer;
- tight DECODE blocks that trigger recompute preemption;
- asymmetric stage latencies; and
- PP4/PP8 fill and terminal drain.

Acceptance:

- target, event, scheduler, batch, stage, transfer, and request parity;
- no cross-cluster block or in-flight-batch accounting;
- no target starvation caused by another target's local capacity;
- repeated C++ output is byte-stable; and
- every successful run drains both cluster trees.

### Step 3H: Analytical Matrix and Hardening

Run the primary analytical topology matrix with independent cluster values:

```text
PREFILL TP: 1, 2, 4, 8
PREFILL PP: 1, 2, 4
DECODE TP:  1, 2, 4, 8
DECODE PP:  1, 2, 4
DP:         1, 2 where topology permits
replicas:   1, 2 in selected cross-products
```

The full Cartesian product is not required. Select pairwise and boundary cases
that cover every TP/PP value on both sides, asymmetric layouts, multiple
targets, and the Rubin domain limit.

Acceptance:

- analytical stage components match Python within documented tolerance;
- transfer bytes match exactly across topology variants;
- transfer duration matches Python within tolerance;
- Step 2.5's full 49-case differential suite still passes;
- the Step 3 fixed and analytical differential matrix passes;
- all CTest suites pass;
- GCC WSL build passes;
- AddressSanitizer and UndefinedBehaviorSanitizer pass; and
- schemas v1-v4 reject malformed integer, topology, timing, and feature input.

## Primary Test Matrix

| Case | Mode | PREFILL | DECODE | Transfer | Purpose |
| --- | --- | --- | --- | --- | --- |
| PDD-01 | online | R1/TP1/PP1/DP1 | R1/TP1/PP1/DP1 | fixed, nonzero | minimal event graph |
| PDD-02 | offline | R1/TP1/PP1/DP1 | R1/TP1/PP1/DP1 | fixed, nonzero | decode arrival barrier |
| PDD-03 | online | R1/TP1/PP1/DP1 | R1/TP1/PP1/DP1 | zero latency | equal-time event order |
| PDD-04 | online | R1/TP2/PP2/DP1 | R1/TP4/PP4/DP1 | fixed | asymmetric TP/PP |
| PDD-05 | online | R2/TP2/PP2/DP2 | R1/TP4/PP2/DP2 | fixed | independent routing |
| PDD-06 | offline | R1/TP2/PP4/DP2 | R2/TP2/PP2/DP1 | fixed | barrier plus target remap |
| PDD-07 | online | tight blocks | roomy blocks | slow | source retention pressure |
| PDD-08 | online | roomy blocks | tight blocks | fast | decode preemption pressure |
| PDD-09 | online | chunked prefill | normal decode | nonzero | transfer only after final chunk |
| PDD-10 | online | multi-request batch | normal decode | simultaneous | one transfer per request |
| PDD-11 | online | PP4 slow middle | PP4 asymmetric | nonzero | pipeline backpressure |
| PDD-12 | offline | PP8 | PP4/DP2 | fixed | terminal drain |
| PDD-13 | online | analytical TP1 | analytical TP8 | analytical | topology/transfer golden |
| PDD-14 | online | analytical TP8 | analytical TP1 | analytical | reverse asymmetry |
| PDD-15 | offline | analytical TP8/PP4 | analytical TP1/PP1 | analytical | mutable PP-stage request context |
| PDD-16 | online | R1/TP2/PP2/DP1 | tight decode blocks | fixed | decode preemption waiting order |
| PDD-17 | online | R2/TP2/PP2/DP2 | R2/TP4/PP4/DP2 | simultaneous | equal-time decode routing order |

Each case compares the production Python oracle and C++ output. Focused C++
unit tests supplement but do not replace the differential cases.

## Invariants

- Every PDD request has exactly one PREFILL owner.
- A request obtains a DECODE owner only after its transfer ends.
- A complete target key is `(cluster_type, replica_id, dp_id)`.
- A request cannot be runnable in PREFILL and DECODE at the same time.
- A partial prefill never emits a transfer.
- Each decode-bound request emits exactly one P-to-D transfer in Step 3.
- Transfer bytes are based on the request's full prefill length.
- Source KV remains allocated from prefill completion through transfer end.
- Source KV is released exactly once.
- Target DECODE KV accounting is independent of source allocation ownership.
- DECODE does not execute prefill tokens in the ordinary handoff path.
- Online DECODE scheduling may start on each transfer arrival.
- Offline DECODE scheduling starts only when its request-count barrier is met.
- Batch IDs and transfer IDs are globally unique.
- Scheduler iteration IDs are local to their complete cluster target.
- A batch belongs to one cluster target for its entire lifetime.
- Request token state advances only after the final PP stage.
- Stale events cannot release capacity, enqueue requests, or advance tokens
  more than once.
- Per-target KV use never exceeds that target's capacity.
- Successful termination means:
  - global queues are empty;
  - both cluster queues are empty;
  - every replica scheduler is idle;
  - every stage is idle and has no tickets;
  - PREFILL pending-transfer sets are empty;
  - every transfer is complete;
  - all source and target block allocations are released; and
  - every request is completed exactly once.

## Risks and Decisions

### Do Not Duplicate the Step 2.5 Pipeline

A PDD-only simulator with separate request and batch logic would immediately
create two implementations of TP/PP/DP, preemption, stage overlap, and stale
events. Cluster-scoping the existing implementation is required before
transfer work.

### Source Retention Is Part of Scheduling Correctness

KV transfer is not only a timing edge. Retained PREFILL blocks can prevent
later admissions. Releasing them at prefill completion would produce plausible
latencies while violating Python scheduling and capacity behavior.

### Offline and Online Are Different Contracts

The only planned mode branch at DECODE arrival is:

- online: schedule on each arrival;
- offline: wait for all decode-bound arrivals.

The rest of the event and scheduler pipeline remains shared.

### Preserve Python Semantics Before Improving Them

If Python behavior appears stricter or looser than an earlier prose plan,
record the discrepancy and use production Python behavior for Step 3 parity.
Any all-or-nothing stale-event strengthening or transfer-model correction must
be a separately approved Python-and-C++ change.

### Determinism Is Observable

Cluster maps, transfer creation, and emitted events must use stable iteration
order. Final metrics matching is insufficient when event or scheduler traces
differ.

## Definition of Done

Step 3 is complete when the C++ simulator runs dense sequential PDD through
the generalized Step 2.5 hierarchy and matches production Python across:

- PREFILL and DECODE routing;
- independent replica/TP/PP/DP topology;
- continuous batching, chunking, and preemption;
- source KV retention and release;
- per-request analytical KV transfers;
- online immediate and offline barrier admission;
- stage overlap and timing;
- first-token, TTFT, E2E, and completion metrics;
- exact event, scheduler, batch, stage, and transfer ordering; and
- complete two-cluster quiescence.

Passing only request completion or final latency is insufficient. The
cluster-scoped event graph, transfer lifecycle, scheduler decisions, memory
ownership, and final empty state are all part of the Step 3 parity contract.
