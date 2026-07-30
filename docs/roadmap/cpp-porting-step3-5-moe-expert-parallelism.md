# C++ Porting Step 3.5: MoE and Expert-Parallel Parity

## Status

Implemented and validated on 2026-07-30.

Step 3.5 follows the completed dense co-location and sequential PDD
implementation and is the validated baseline for Step 4 session-prefix
caching.

The production Python simulator remains the behavioral oracle. The C++ design
does not copy Python object ownership or the large cluster-scheduler methods
verbatim; it reproduces their observable configuration, event, scheduling,
timing, routing, and completion contracts with typed IDs and composed runtime
components.

### Implementation Result

The completed implementation includes:

- the `Phi-tiny-MoE-instruct` model contract and final unified schema;
- attention TP/DP and MoE TP/EP shared-domain validation;
- local MoE and synchronized MoE paths for co-location and sequential PDD;
- deterministic balanced, random, skewed, and Zipf routing with exact NumPy
  PCG64/SeedSequence golden-vector parity;
- contiguous expert ownership, analytical grouped expert execution, TP/EP
  communication, and critical-lane timing;
- all four typed prefill/decode synchronization event families;
- generation-safe barriers, deterministic idle participants, real-over-idle
  replacement, duplicate-event idempotence, and strict final quiescence;
- model-derived Phi KV-transfer sizing; and
- normalized routing, component, stage, synchronization, request, and
  transfer output.

The production-Python differential contract treats simultaneous,
causally-independent events as an unordered timestamp group. Request IDs,
replica/DP/stage/layer IDs, synchronization phases and groups, token counts,
routing arrays, completion order, and all state-changing identities remain
exact. Raw event-queue sequence numbers and allocator-local IDs of temporary
idle batches are intentionally excluded: Python assigns those from global
process counters affected by unrelated temporary object creation.

Final validation:

| Gate | Result |
| --- | --- |
| Debug CTest | 18/18 passed |
| Release `/O2` CTest | 18/18 passed |
| MSVC ASan CTest | 18/18 passed |
| GCC UBSan MoE fixtures | analytical co-location, local co-location, and sequential PDD passed |
| MoE Python/C++ tests | 28/28 passed |
| Existing dense differential suite | 47/47 passed |
| Full production simulator parity | 6 co-location and 5 sequential-PDD configuration/workload matrix cases matched requests, semantic event timestamp groups, scheduler traces, routing allocations, stage timing, completion order, and transfers |
| Release performance | 119.24x on the two-request fixture and 45.15x on the 12-request longer-prefill fixture, three-run median with process startup included |

The expanded production-oracle matrix covers:

- local MoE, MoE TP-only, EP-only, TP+EP, DP+EP, PP2, PP4, and two-replica
  topologies;
- asymmetric PREFILL/DECODE parallel domains in sequential PDD;
- online and offline admission, simultaneous and staggered arrivals, long
  contexts, uneven request sizes, chunking, block pressure, and slow KV
  transfer;
- top-k 1 and 2; and
- balanced, `uniform_legacy`, `uniform_random`, random, skewed, and Zipf
  routing.

The comparison is not limited to final request latency. It requires exact
request state, event type/count, timestamp-group semantics, scheduler token
decisions, stage identity, expert and EP-lane token arrays, critical-lane
selection, completion order, and PDD transfer identity. Floating tolerance is
used only for explicitly named timing fields.

### Production Python Oracle Limitations

Two attempted PDD stress combinations cannot currently serve as differential
oracles: DECODE attention DP2 combined with PP2, and two DECODE replicas
combined with PP2. In both cases the production Python simulator drains its
event queue with requests stopped at 30 of 32 completed layers. The passing
matrix therefore uses PP1 for those two DECODE variants while retaining DP2,
EP, asymmetric topology, multiple replicas, and PP coverage in other cases.
This is an oracle/runtime limitation, not a C++ parity result, so those exact
combinations are not claimed as validated.

Python's public dense-PDD configuration validation also rejects DP greater
than one. The differential oracle first constructs a public-valid topology and
then installs the independently validated PREFILL and DECODE private configs,
matching the production runtime path. This workaround is isolated to the test
harness; the configuration contract should eventually be aligned on both
sides.

## Why This Is a Separate Step

MoE support is not an execution-predictor-only change. Expert parallelism adds
a layer-level distributed state machine inside one pipeline stage:

```text
attention
  -> pre-MoE synchronization
  -> expert routing and lane-local expert execution
  -> post-MoE synchronization / communication
  -> next transformer layer
```

Before Step 3.5, the C++ stage scheduler predicted and completed an entire
pipeline stage in one operation. It had no representation for:

- separate attention and MoE parallel domains;
- expert ownership and token routing;
- a batch correlation ID shared by synchronization participants;
- idle lanes that complete a collective domain;
- per-layer pre-MoE and post-MoE barriers;
- EP dispatch/combine communication;
- per-EP-lane execution times or critical-lane delay; or
- MoE-specific execution and synchronization metrics.

Adding only MoE FLOP formulas would produce plausible numbers but would not
match Python when DP or EP is greater than one.

## Goal

End Step 3.5 with deterministic `Phi-tiny-MoE-instruct` simulations in which:

1. co-location and sequential PDD both support MoE;
2. attention TP/DP, MoE TP/EP, PP, multiple replicas, online/offline arrivals,
   chunked prefill, continuous batching, and ordinary preemption compose
   correctly;
3. expert routing conserves `tokens * router_topk` exactly;
4. each EP lane executes only its owned experts;
5. the slowest required lane controls collective progress where Python uses a
   critical-lane barrier;
6. missing synchronization participants are represented deterministically and
   never mutate request or KV state;
7. each real request advances once even when synchronization records refer to
   the same request through more than one participant;
8. PDD retains the Step 3 source-KV and transfer contract with model-derived KV
   dimensions; and
9. normalized event, routing, scheduler, timing, transfer, request, and final
   state outputs match production Python.

## Python Contract Audit

The implementation must be based on the following live Python paths rather
than on older design documents.

| Contract | Primary Python source |
| --- | --- |
| MoE model fields | `frontier/config/model_config.py` |
| Replica MoE and EP fields | `frontier/config/config.py::ReplicaConfig` |
| Shared attention/MoE domain | `frontier/config/parallel_semantics.py` |
| Model architecture capabilities | `frontier/model_architectures.py` |
| Batch `is_moe`, `is_idle`, and global IDs | `frontier/entities/batch.py` |
| Stage-path selection | `frontier/events/replica_stage_schedule_event.py` |
| Prefill sync arrival | `frontier/events/prefill_sync_event.py` |
| Prefill collective | `frontier/events/prefill_sync_collective_event.py` |
| Decode sync arrival | `frontier/events/decode_sync_event.py` |
| Decode collective | `frontier/events/decode_sync_collective_event.py` |
| Barrier and idle-lane behavior | `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py` |
| Analytical routing and lane timing | `frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.py` |
| Dummy MoE component timing | `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py` |
| MoE time composition | `frontier/entities/execution_time.py` and `time_components.py` |
| PDD completion and transfer | `frontier/events/cluster_batch_end_event.py` |

Before C++ implementation begins, small Python oracle tests must freeze the
observable behavior of these paths. This is important because several Python
details are non-obvious:

- MoE detection comes from `model_config.is_moe`, not from EP size or a
  user-supplied expert count.
- The shared-domain invariant is
  `attn_tp * attn_dp == moe_tp * moe_ep`.
- `total_expert_num` controls routing ownership, while the model's declared
  expert count is still used by some analytical operators such as the gating
  projection. The public Phi example currently overrides 16 model experts with
  8 runtime experts; C++ must reproduce the frozen Python result unless the
  Python contract is deliberately corrected first.
- Prefill synchronization is keyed by a batch creation/global ordinal across
  DP lanes.
- Co-location pure-decode MoE uses a lane-scoped decode synchronization ID,
  distinct from the ordinary batch ID.
- Python creates idle participants in sorted lane order. An idle participant
  cannot replace a real participant already in the waiting room.
- Prefill and decode do not compose MoE communication at exactly the same
  event boundary. The oracle fixtures must pin where each cost is paid instead
  of deriving a cleaner but different timeline.
- The default analytical MoE predictor models EP dispatch and combine as two
  all-to-all collectives. The generic architecture-profile all-gather path used
  by other Python branches is not automatically the oracle for this Step 3.5
  runtime.

## Supported Scope

### Included

- `co-location`;
- sequential `pd-disaggregation` with `enable_parallel_clusters=false`;
- offline and online simulation modes;
- `Phi-tiny-MoE-instruct` as the reference MoE model:
  - 32 transformer layers,
  - hidden size 4096,
  - MoE intermediate size 448,
  - 16 query heads,
  - 4 KV heads,
  - head dimension 128,
  - gated MoE,
  - model expert count 16, and
  - model router top-k 2;
- explicit runtime expert-count and router-top-k overrides, matching Python;
- vLLM V1 and FCFS;
- round-robin cluster routing;
- fixed/dummy MoE timing for deterministic state-machine tests;
- the analytical roofline and analytical communication backend;
- attention TP and DP;
- MoE TP and EP;
- PP and multiple replicas;
- EP=1 local MoE and EP>1 synchronized MoE;
- `simulation`, `uniform_legacy`, and `uniform_random` routing modes;
- balanced, random, skewed, and Zipf simulation distributions;
- deterministic routing seeds and stage-local layer IDs matching Python;
- expert-load imbalance and critical EP-lane timing;
- prefill and decode synchronization events;
- deterministic idle synchronization participants;
- mixed prefill/decode batches in co-location;
- chunked prefill, continuous batching, KV pressure, preemption, and stale-event
  handling inherited from Steps 2.5 and 3;
- asymmetric PREFILL and DECODE MoE topologies in PDD, provided each topology
  independently satisfies the shared-domain invariant; and
- model-derived dense-attention KV transfer sizing for the Phi MoE model.

### Explicitly Deferred

- `pd-af-disaggregation`, `DECODE_ATTN`, and `DECODE_FFN`;
- `EPBatchGroup`, M2N transfer, and AFD-specific expert regrouping;
- parallel cluster threads;
- profile-trained sklearn execution predictors;
- collective-sim and ASTRA-Sim integration;
- mixed dense/MoE-layer models;
- shared experts;
- Step3Text/Step2Mini architecture-specific operations;
- architecture-specific EP all-gather versus all-to-all selection beyond the
  reference Phi analytical path;
- MLA, MFA, latent KV layouts, and linear attention;
- speculative decoding/MTP;
- Thinking Mode;
- CUDA Graph timing;
- SGLang;
- prefix caching, block hashes, and session affinity;
- CPU KV-cache offload/restore; and
- GPU kernel execution or real distributed inference.

These deferred features require additional model-architecture contracts. They
must not be silently accepted and approximated by the generic Phi path.

## Configuration Contract

Step 3.5 extends the single normalized schema version 1. It does not introduce
or retain schema v2/v3/v4 compatibility paths.

The model and topology must be visible to scheduler, entity, predictor, KV
accounting, and transfer code. Model structure therefore cannot remain hidden
inside `AnalyticalExecutionModelConfig`.

The normalized C++ shape should be equivalent to:

```json
{
  "schema_version": 1,
  "system_architecture": "co-location",
  "cluster": {
    "model": {
      "name": "Phi-tiny-MoE-instruct",
      "runtime_total_experts": 8,
      "router_topk": 2
    },
    "parallelism": {
      "num_replicas": 1,
      "pipeline_parallel_size": 1,
      "attention": {
        "tensor_parallel_size": 4,
        "data_parallel_size": 1
      },
      "moe": {
        "tensor_parallel_size": 2,
        "expert_parallel_size": 2
      }
    },
    "moe_routing": {
      "mode": "simulation",
      "distribution": "balanced",
      "seed": 42
    }
  }
}
```

Dense configurations use the same attention topology and set the MoE domain to
its neutral value. The parser must not infer a MoE model merely because an MoE
field is present.

For PDD, PREFILL and DECODE each own their topology, scheduler, routing state,
and execution model. They must use the same model structure and runtime expert
contract, but may use different:

- replica counts;
- attention TP/DP;
- MoE TP/EP;
- PP;
- scheduler budgets;
- block capacities;
- fixed/dummy component time; and
- analytical network parameters.

### Validation

Reject a configuration before constructing runtime state when:

- any parallel size is zero;
- `num_layers % pipeline_parallel_size != 0`;
- a dense model requests EP or MoE TP behavior;
- an MoE model has fewer than two runtime experts;
- `router_topk == 0` after model-default resolution;
- `router_topk > runtime_total_experts`;
- `runtime_total_experts % moe_expert_parallel_size != 0`;
- `attn_tp * attn_dp != moe_tp * moe_ep`;
- a routing seed is negative;
- a routing mode or distribution is unknown;
- the fixed per-lane timing vector does not match the required lane domain;
- PDD PREFILL and DECODE model/expert contracts differ;
- parallel PDD is selected; or
- any explicitly deferred model/runtime capability is requested.

All integer parsing must retain the existing checked-range behavior before
converting to the C++ field type.

## Parallel-Domain Representation

Replace the current single TP/DP interpretation with a shared physical domain:

```text
one replica and one pipeline stage

attention view: attn_dp lanes x attn_tp ranks
MoE view:       moe_ep lanes  x moe_tp ranks

attn_dp * attn_tp == moe_ep * moe_tp
```

The accelerator count is:

```text
num_replicas * pipeline_parallel_size *
shared_parallel_domain_size
```

It is not the product of attention and MoE parallel sizes.

Replica schedulers remain attached to attention-DP targets
`(replica_id, dp_id)`, as in Python. A small immutable
`ParallelDomainLayout` maps physical ranks to:

- attention DP lane and TP rank;
- MoE EP lane and TP rank;
- experts owned by each EP lane; and
- the synchronization participant domain used by each path.

Expert ownership is contiguous:

```text
experts_per_lane = runtime_total_experts / moe_ep
ep_id             = global_expert_id / experts_per_lane
local_expert_id   = global_expert_id % experts_per_lane
```

## IDs and Entity State

Do not overload globally unique `BatchId` with Python's lane-local `global_id`.
Add typed IDs with the existing `-1` invalid sentinel:

- `MoESyncGroupId`;
- `MoEParticipantId`; and
- `LayerId` if a generic layer ID is not already available.

Add:

```text
ModelKind: dense | moe
BatchKind: work | moe_idle
MoESyncPhase: pre_moe | post_moe
MoESyncPath: prefill | decode
```

A real MoE batch records:

- its ordinary `BatchId`;
- its synchronization group ID;
- model kind;
- replica/DP ownership;
- stage-local layer progress;
- generation;
- request snapshots; and
- stage start time.

An idle synchronization batch:

- has a unique `BatchId` for event-trace parity;
- has no requests and no scheduled tokens;
- carries a synchronization group and participant ID;
- never enters the replica request scheduler;
- never allocates or releases KV blocks;
- never creates a transfer;
- never produces a normal batch-completion record; and
- is destroyed or retired when its barrier entry is consumed.

Repeated request references within one synchronization group must be
deduplicated by `RequestId` before any per-layer or completion mutation.

## Synchronization Group IDs

The Python oracle uses more than one correlation rule. C++ must make these
rules explicit:

- PREFILL and monolithic mixed/prefill MoE use the lane-local batch creation
  ordinal as the group correlation key across attention-DP lanes.
- Unified PDD DECODE follows the Python decode waiting-key rule.
- Co-location pure-decode MoE uses a dedicated lane-scoped decode ID generated
  from the lane decode counter and EP participant lane. It must not reuse
  `BatchId`.

ID creation must be deterministic, checked for overflow, and local to the
owning replica/cluster domain. A stale generation must not be allowed to join a
new group that happens to reuse the same ordinal.

## Event Contract

Add four typed event payloads and four event handler files:

- `PrefillSyncEvent`;
- `PrefillSyncCollectiveEvent`;
- `DecodeSyncEvent`; and
- `DecodeSyncCollectiveEvent`.

The arrival payloads carry:

- cluster type;
- replica ID;
- stage ID;
- batch ID;
- DP/participant ID;
- synchronization group ID;
- sync phase;
- stage-local layer ID;
- elapsed component time; and
- batch generation.

The collective payloads carry the barrier key, sync phase, layer ID, and
generation. No optional catch-all event payload is introduced.

The event queue continues to order by exact `(time, sequence)`. Missing idle
participants are created by ascending participant ID. Post-MoE events preserve
Python's real-batch-before-idle insertion rule. Differential normalization
compares causally-independent events at the same timestamp as a semantic
multiset rather than requiring Python and C++ allocator sequence numbers to
match.

### Co-location Flow

Local MoE (`attn_dp=1` and `moe_ep=1`) retains the direct stage path:

```text
ReplicaStageSchedule
  -> predict whole local MoE stage
  -> BatchStageEnd
```

Synchronized prefill or mixed prefill:

```text
ReplicaStageSchedule
  -> first-layer attention
  -> PrefillSync(pre_moe)
  -> PrefillSyncCollective(pre_moe)
  -> critical expert-lane/post-attention work
  -> PrefillSync(post_moe)
  -> PrefillSyncCollective(post_moe)
  -> next-layer attention ... or BatchStageEnd
```

Pure decode uses the corresponding decode events and Python's decode
synchronization-group rule.

### Sequential PDD Flow

PREFILL uses the prefill state machine. The final real PREFILL stage completion
then follows the existing Step 3 path:

```text
BatchStageEnd(PREFILL)
  -> ClusterBatchEnd(PREFILL)
  -> one KVCacheTransferStart per completed real request
```

Idle synchronization participants do not reach this path.

Unified DECODE uses the decode state machine and ends in the existing
`ClusterBatchEnd(DECODE) -> GlobalBatchEnd(DECODE)` path.

## Barrier Coordinator

Python stores nested waiting rooms directly on `BaseClusterScheduler`. C++
keeps the same cluster-local ownership while using composition for the
waiting-room implementation:

```text
BaseClusterScheduler
  owns MoEBarrierCoordinator

RoundRobinClusterScheduler
  inherits BaseClusterScheduler
  implements only request-to-replica/DP selection
```

`SimulationContext` owns only simulation-wide entity storage, the event queue,
output, and the global scheduler. It does not own cluster-local MoE groups,
barriers, batch ordinals, or synchronization state. Event handlers resolve the
target cluster through the global scheduler and invoke the corresponding
`BaseClusterScheduler`, matching the Python ownership chain.

This keeps the scheduler hierarchy aligned with Python while allowing the
waiting-room invariants to remain isolated in a small C++ value component.

The barrier key is:

```text
(cluster_type,
 replica_id,
 stage_id,
 sync_group_id,
 layer_id,
 sync_phase,
 generation)
```

Each entry contains an ordered participant map with:

- participant ID;
- real or idle batch ID;
- arrival time; and
- elapsed component time.

Required behavior:

1. validate the participant domain and generation;
2. reject an out-of-range participant;
3. accept each real participant at most once;
4. allow a real participant to win over a pending idle participant exactly
   where Python permits it;
5. never allow an idle participant to overwrite a real participant;
6. create missing pre-MoE idle participants in ascending ID order;
7. emit one collective at the maximum participant arrival time;
8. make duplicate collective events idempotent;
9. erase the consumed phase entry;
10. erase empty group/layer containers; and
11. require all waiting rooms to be empty at simulator quiescence.

Co-location shared-domain decode compacts missing idle lanes directly into the
waiting room at `pre_moe`, so no idle `DecodeSyncEvent` is emitted for that
phase. At `post_moe`, it emits lane-timed idle `DecodeSyncEvent`s. Prefill and
sequential-PDD decode use explicit idle sync events for missing participants.

## Routing and Load Imbalance

Routing is a pure deterministic component independent of the event loop.

For each layer:

```text
total_routed_tokens = effective_tokens * router_topk
```

The global allocation must conserve that integer exactly.

### Deterministic Ratio Discretization

For balanced, skewed, Zipf, and ratio-based random distributions:

1. normalize expert weights;
2. compute exact fractional token counts;
3. assign each expert its floor;
4. sort remainder candidates by:
   `(-fractional_part, -normalized_weight, expert_id)`; and
5. distribute the remainder in that order.

Integer expert IDs, token counts, lane allocations, and array order are exact
parity fields.

### Random Compatibility

The Step 3.5 analytical Python oracle uses
`numpy.random.default_rng(seed + layer_id)` PCG64 for both:

- `simulation + random` weights; and
- `uniform_random` expert sampling.

Older sklearn/disaggregation predictor internals still contain legacy
`numpy.random.seed` routing helpers, but those are outside the selected
analytical Step 3.5 predictor contract.

C++ must reproduce the corresponding NumPy sampling transformation, not only
select a generator with the same family name and call the result deterministic.
Either:

- implement the required NumPy PCG64/SeedSequence subset with checked golden
  vectors; or
- check in route-allocation fixtures generated by the Python oracle and use a
  separately specified portable generator whose contract is intentionally
  adopted by both Python and C++.

The first option preserves current Python behavior and is the default Step 3.5
plan. Random modes are not complete until multi-seed, multi-layer allocation
fixtures match exactly.

## Execution-Time Model

Refactor `BatchExecutionModel` so synchronized MoE code can request
single-layer components without reverse-engineering an aggregate stage time.

The predictor interface needs equivalent operations for:

- one attention layer;
- one MoE layer for an explicit local expert-token map;
- all EP-lane MoE times for one global routed allocation;
- attention TP all-reduce;
- MoE TP all-reduce;
- EP dispatch and combine;
- DP input/output collective terms used by the Python path;
- PP boundary transfer; and
- a whole local/dense stage fast path.

Use nested C++ value types rather than one Python-shaped class with dozens of
loosely related fields:

```text
AttentionLayerTime
MoELayerTime
CommunicationTime
LayerExecutionPrediction
StageExecutionPrediction
```

`MoELayerTime` includes:

- gating linear;
- routing top-k;
- grouped up projection;
- grouped down projection;
- shuffling; and
- post-attention norm.

`CommunicationTime` keeps distinct:

- attention TP;
- MoE TP;
- EP dispatch;
- EP combine;
- DP input;
- DP output; and
- PP.

The serializer can flatten these values into the normalized output contract.
The runtime must never count an EP or TP component both inside lane compute and
again at a collective event.

### Analytical Phi Model

Port the Python analytical formulas for:

- Phi attention dimensions and GQA KV heads;
- gating projection;
- routing streaming work;
- grouped expert up/down GEMMs;
- gated projection multiplicity;
- shuffling;
- norm and residual work;
- attention and MoE TP collectives;
- two EP all-to-all operations for dispatch/combine; and
- PP communication.

Each EP lane is predicted from only its local expert allocation. Where Python
uses critical-lane timing:

```text
critical_lane_time = max(per_lane_time)
```

Do not use mean tokens, mean time, or a perfectly balanced approximation for
an imbalanced routing mode.

### Fixed/Dummy Model

Add a MoE-aware fixed component model matching Python's dummy behavior:

- the base dummy time populates each enabled compute component;
- TP, EP, and DP communication is zero when its domain size is one;
- the PP boundary term is zero on the last pipeline stage; and
- MoE fields are populated while dense MLP fields remain zero.

This model is used for exact event-order and barrier tests. Analytical timing
is still required for production parity.

## Stage Scheduler and Completion

At `ReplicaStageScheduleEvent`:

1. keep the existing stale-ticket validation;
2. choose dense, local-MoE, prefill-sync, or decode-sync path;
3. mark the stage busy once;
4. create one real `BatchStage` lifecycle record;
5. retain stage ownership through all layer sync events; and
6. release the stage only at the final real `BatchStageEndEvent`.

The sync path must not enqueue the same batch into the stage scheduler for
every layer.

At each completed decode MoE layer, advance each unique real `RequestId` once
where Python advances its completed-layer counter. Idle participants are
excluded.

At final completion:

- actual stage wall time includes synchronization waiting;
- predicted operator components remain separately auditable;
- only real batches emit normal stage/batch completion;
- PP handoff occurs once;
- request token state mutates once;
- in-flight batch accounting decrements once; and
- stale completion behavior remains the Python-compatible request-level
  behavior already selected in Step 2.

## PDD KV Contract

The current C++ Step 3 transfer size is hard-coded to Llama-2 dimensions. Step
3.5 must derive it from the selected model:

```text
tokens
* num_layers
* runtime_num_kv_heads
* head_dim
* kv_factor
* kv_dtype_bytes
```

For the reference Phi dense-attention family:

- `runtime_num_kv_heads = 4`;
- `head_dim = 128`; and
- `kv_factor = 2`.

This follows the production transfer predictor, which uses the model runtime
KV head count for the transferred request. It must not use MoE expert count,
EP size, or the local expert layout.

All Step 3 retention, one-transfer-per-request, online admission, offline
barrier, source release, and transfer ordering contracts remain unchanged.

## Output Contract

Extend normalized schema version 1 with enough information to diagnose parity
without exposing internal pointers or map layout.

### Event Trace

MoE sync events include:

- event type and time;
- cluster, replica, stage, and participant;
- batch ID and sync-group ID;
- layer ID;
- sync phase;
- idle flag; and
- generation.

### Batch and Stage Metrics

Real MoE batch/stage records include:

- model kind;
- runtime expert count and top-k;
- synchronization group ID;
- attention/MoE topology;
- per-component execution time;
- synchronization wait time;
- critical EP lane and critical-lane time; and
- actual stage wall time.

Idle participants remain visible in the event trace but are excluded from
ordinary request, batch-completion, throughput, and KV metrics.

### Routing Diagnostics

Record deterministic diagnostics per real sync group and layer:

- total input tokens;
- total routed tokens;
- global expert token counts in expert-ID order;
- per-EP-lane local expert counts;
- per-lane predicted time;
- critical lane; and
- routing mode, distribution, and seed.

IDs and counts compare exactly. Numerical tolerance is permitted only for
named analytical time fields. The differential comparator must not treat
integer IDs or token counts as floating-point-compatible values.

## C++ File Layout

Keep the existing feature-oriented layout; do not introduce separate
`include/` and `src/` trees.

Suggested additions:

```text
cpp/frontier/
  moe/
    parallel_domain.h/.cc
    routing.h/.cc
    barrier_coordinator.h/.cc
    analytical_model.h/.cc
  events/
    prefill_sync_event.cc
    prefill_sync_collective_event.cc
    decode_sync_event.cc
    decode_sync_collective_event.cc
```

Existing files that require contract changes include:

- `config/config.h`, parser, validator, and serializer;
- `core/ids.h` and `core/event.h`;
- `entities/batch.h`, `batch_stage.h`, `execution_time.h`, and `replica.h`;
- `execution_time_predictor/batch_execution_model.*`;
- `scheduler/cluster_scheduler/*`;
- `scheduler/replica_scheduler/*`;
- `scheduler/replica_stage_scheduler/*`;
- `simulator/entity_arena.*` and `simulation_context.*`;
- `kv_cache_transfer/analytical_transfer.*`; and
- `metrics/output_contract.*`.

## Implementation Slices

Each slice ends with green unit tests and all existing dense differential tests.
Do not postpone dense regression checks until the final slice.

### Step 3.5A: Freeze the Python Oracle and Configuration

1. Add Python contract tests for:
   - model versus runtime expert count;
   - shared-domain validation;
   - local-expert ownership;
   - sync-group ID creation;
   - idle participant order;
   - dummy component enable/skip rules; and
   - Phi KV transfer size.
2. Add normalized MoE config fixtures.
3. Extend C++ model and parallelism configuration.
4. Add all fail-fast validation.
5. Update dense fixtures to the final unified topology shape.

Exit gate:

- C++ round-trips the final schema;
- invalid topology/expert/routing cases fail deterministically; and
- all dense configuration and differential tests remain green.

### Step 3.5B: Routing and Analytical MoE Kernels

1. Implement expert ratio discretization.
2. Implement EP ownership partitioning.
3. Add NumPy-compatible random golden vectors.
4. Add Phi gating, routing, grouped-GEMM, shuffling, and norm formulas.
5. Add lane-local timing and critical-lane selection.
6. Add TP/EP communication components.

Exit gate:

- exact expert token allocations match Python across routing modes, seeds,
  layers, token counts, top-k values, and EP sizes;
- token conservation is property-tested;
- lane times and critical lane match analytical Python within named tolerance;
  and
- grouped projections are counted once per active local expert group.

### Step 3.5C: Entity, Event, and Barrier Foundation

1. Add typed IDs and enums.
2. Add real/idle MoE batch state.
3. Add the four event payloads, handlers, serializers, and dispatcher entries.
4. Implement `MoEBarrierCoordinator`.
5. Add duplicate, stale, overwrite, ordering, and quiescence checks.

Exit gate:

- synthetic two- and four-participant barriers match Python event order;
- idle lanes never mutate request/KV state;
- duplicate collective events are idempotent; and
- no completed test leaves barrier state alive.

### Step 3.5D: Local MoE Fast Path

1. Select MoE from model configuration.
2. Run EP=1/DP=1 without sync events.
3. Aggregate all stage layers once.
4. Preserve TP and PP communication.
5. Add MoE execution and metrics output.

Exit gate:

- local MoE co-location matches Python for prefill, decode, mixed batches,
  PP, chunking, and preemption; and
- no MoE sync event appears when both sync domains are local.

### Step 3.5E: Prefill Synchronization

1. Start a stage with first-layer attention.
2. Add pre-MoE barrier and deterministic idle lanes.
3. Aggregate routed tokens as Python does.
4. replace representative expert time with the critical EP-lane time where
   the Python path does so;
5. add post-MoE barrier;
6. iterate stage-local layers;
7. finish real stages once and skip idle stages.

Exit gate:

- co-location prefill/mixed-prefill and PDD PREFILL match event, timing,
  routing, stage, and scheduler traces for balanced and imbalanced loads.

### Step 3.5F: Decode Synchronization

1. Add Python-compatible decode sync-group generation.
2. Add shared-domain participant mapping.
3. Add direct idle compaction versus explicit idle events as required by the
   oracle.
4. Build virtual global token views for prediction without creating runnable
   request batches.
5. schedule lane-local expert work and post-MoE communication;
6. deduplicate per-layer request mutation;
7. complete real stages once.

Exit gate:

- co-location pure decode and unified PDD DECODE match Python for EP1/EP>1,
  DP1/DP>1, uneven lane occupancy, multiple layers, and equal-time events.

### Step 3.5G: Full Co-location and PDD Integration

1. Exercise independent PDD PREFILL/DECODE topologies.
2. replace hard-coded KV dimensions with model-derived dimensions;
3. combine MoE with multiple replicas, PP, chunked prefill, preemption, and
   transfer pressure;
4. verify online overlap and offline decode admission;
5. add complete MoE outputs and differential normalization.

Exit gate:

- the required matrix below passes production-Python/C++ differential tests;
- all dense Step 1-3 cases still pass; and
- every run reaches strict quiescence.

### Step 3.5H: Hardening and Performance

1. Run Debug, Release `-O2`, ASan, and UBSan suites.
2. Add overflow and nonfinite-time tests.
3. Reserve barrier/routing storage by known domain size.
4. avoid copying request objects into virtual global batches;
5. benchmark long-context, high-top-k, high-EP, and many-layer cases;
6. confirm no completed barrier or idle batch leaks.

Exit gate:

- sanitizer suites are clean;
- Release has no correctness-only code-path divergence;
- C++ remains faster than the production Python oracle on the agreed MoE
  workloads; and
- Step 3.5 is committed before Step 4 cache state is added.

## Differential Test Matrix

At minimum, cover:

### Topologies

1. local MoE: `attn TP1/DP1`, `MoE TP1/EP1`, PP1;
2. MoE TP only: `attn TP2/DP1`, `MoE TP2/EP1`;
3. EP only: `attn TP2/DP1`, `MoE TP1/EP2`;
4. shared TP+EP: `attn TP4/DP1`, `MoE TP2/EP2`;
5. DP+EP: `attn TP2/DP2`, `MoE TP1/EP4`;
6. PP2 and PP4 variants;
7. two replicas with uneven request distribution; and
8. asymmetric PDD PREFILL/DECODE combinations selected from the above.

### Workloads

- one request;
- simultaneous multi-request arrivals;
- staggered online arrivals;
- uneven DP-lane occupancy;
- prefill-only stage work;
- pure decode iterations;
- mixed prefill/decode co-location batches;
- chunked long prefill;
- multiple decode iterations;
- top-k 1 and 2;
- balanced, random, skewed, and Zipf expert load;
- seeds and layer IDs that change the critical lane;
- KV pressure without preemption;
- KV pressure with preemption and stale completion;
- equal-time real and idle participant arrivals;
- multiple PREFILL requests completing and transferring together; and
- long context with asymmetric PDD topology.

### Exact Semantic Fields

- request, real batch, stage, replica, DP, participant, transfer, and
  sync-group IDs where the Python oracle exposes semantic identity;
- event types, payload integer fields, and chronological timestamp groups;
- request completion order;
- scheduler decisions and token budgets;
- scheduled/prefill/decode/routed token counts;
- global and lane-local expert allocations;
- idle participant IDs and ascending creation policy;
- critical-lane ID;
- KV block and transfer byte accounting; and
- final empty/busy/barrier state.

Python-global event sequence numbers and allocator-local `Batch.id` values for
temporary idle batches are diagnostic implementation details, not semantic
parity fields. Within one exact timestamp, independently runnable events are
compared as a multiset of their semantic fields.

### Tolerant Fields

Only explicitly named floating-point fields:

- analytical operator milliseconds;
- collective milliseconds;
- event times derived from analytical components;
- request latency metrics; and
- throughput.

The tolerance and operation order are documented per field. Integer-versus-
floating coercion is never used for IDs, counts, or array elements.

## Definition of Done

Step 3.5 is complete only when:

- the reference Phi MoE model runs in co-location and sequential PDD;
- attention TP/DP, MoE TP/EP, PP, and multiple replicas compose under the
  shared-domain invariant;
- local MoE avoids unnecessary sync events;
- synchronized MoE uses all four typed event families;
- expert routing and lane ownership match Python exactly;
- random routing matches frozen NumPy golden vectors;
- idle participants are deterministic and side-effect free;
- critical-lane timing matches Python;
- real requests, stages, batches, KV blocks, and transfers complete exactly
  once;
- model-derived Phi KV transfer sizes match Python;
- fixed/dummy component and state-machine contracts pass, and analytical
  production-simulator differential matrices pass;
- all dense Step 1-3 tests pass unchanged in behavior;
- strict final quiescence includes empty MoE waiting rooms;
- Debug, Release `-O2`, ASan, and UBSan gates pass; and
- all excluded features fail fast with clear messages.
