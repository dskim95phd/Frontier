# C++ DP-Attention / EP-FFN Lockstep Plan

## Status

PDD and co-location pure-`DECODE` milestones implemented and validated,
2026-08-06.

The Python/C++ differential harness referenced below (`cpp/tests/parity/`,
`cpp/tests/differential/`, `FRONTIER_CPP_BINARY`, `FRONTIER_CPP_RUNNER`) has
since been retired: the port goal is met and the C++ core is now the primary
implementation. Those instructions are kept as a record of how the milestone
was validated at the time and are no longer runnable. Current gates are the
CTest suite and the contracts in `cpp/README.md`.


The implementation keeps the existing `DecodeSyncEvent` and
`DecodeSyncCollectiveEvent` flow. It adds coordinator-owned generations for
both decode architectures, a stage-lifetime shared-domain reservation, and
persistent dummy participants. Unequal real DP lanes now enter one group and
release the pre-MoE collective at the latest real attention completion time. The Debug
test suite passes 31/31, including dedicated DP2/EP2 regressions for the
PDD slow-lane barrier, PDD dummy-lane reservation, and co-location lockstep.

Analytical decode group latency now sums the realized expert-token counts from
all real DP participants, repartitions the combined counts over the EP domain,
and invokes the MoE predictor once for that integrated allocation. Fixed mode
retains the prior lane-time sum as its deterministic fallback. In the DP2/EP2
regression, the old time-sum critical path and observed duration were both
`0.064669 ms`; aggregate-token prediction is `0.033120 ms`, and the corrected
observed duration is also `0.033120 ms`. The co-location mixed/prefill and PDD
prefill path audit remain follow-up work.

`first_layer_scaled` decode now scales the group-level repeated critical path,
not the pre-group batch-local estimate. The Kimi K2 DP2/EP2 regression uses
unequal contexts and separates the first pre-MoE attention delta from the
subsequent repeated-layer delta. Before the fix, the scaled synchronization
wait gap was `0.000098 ms` while detailed mode and the calculated 60-layer
expectation were `0.002996 ms`; after the fix, scaled and detailed both report
`0.002996 ms`.

This plan corrects the C++ simulator's MoE timing semantics when attention
uses data parallelism and the following expert FFN uses expert parallelism on
the same physical accelerator domain. The first acceptance target was the
sequential PDD `DECODE` cluster with both `attention_dp > 1` and
`moe_ep > 1`; pure-decode co-location now uses the same coordinator contract.

The selected contract is:

> DP lanes may finish attention at different simulated times, but every DP
> lane in the aligned forward must reach the same pre-MoE synchronization
> point before any lane starts expert dispatch or expert FFN. The shared FFN
> therefore starts at the latest participating attention completion time.

Dual Batch Overlap (DBO) is intentionally out of scope. Without DBO, another
forward cannot use an overlapping attention view of the physical ranks while
the aligned forward owns those ranks.

## Why This Change Is Required

The current C++ implementation already validates that attention and MoE are
two views of one physical domain:

```text
attention_tp * attention_dp == moe_tp * moe_ep
```

Its barrier also selects the maximum arrival time when all participants have
joined one synchronization group. The incorrect behavior occurs earlier, in
group construction:

- PDD decode candidates are joined only when their
  `initial_pre_arrival` timestamps are exactly equal.
- Different sequence-length sums produce different attention times, so the DP
  lanes are split into different synchronization generations.
- At the first arrival of each resulting group, the missing lanes are filled
  with immediately ready idle participants.
- The early lane can consequently enter the expert phase without the slower
  real lane, even though both attention and EP layouts describe the same GPUs.
- The current PDD expert prediction then derives the critical EP time from one
  representative real batch instead of routing the aggregate work of all real
  DP participants.

This is optimistic and physically inconsistent: an active slow DP lane cannot
be replaced by an idle participant for the same aligned forward. In a real
distributed runtime, all ranks in an EP collective must issue the matching
collective in the same order. An early rank waits in or before that collective
until the slow rank arrives. A rank with no request still executes an empty or
dummy forward so that it can participate.

The relevant implementation areas are:

- shared-domain validation in `cpp/frontier/config/config_parse.cc`;
- MoE group construction, idle participant creation, barrier release, and
  expert continuation in
  `cpp/frontier/scheduler/cluster_scheduler/base_cluster_scheduler.cc`;
- stage launch in `cpp/frontier/events/replica_stage_schedule_event.cc`;
- synchronization event payloads in `cpp/frontier/core/event.h`;
- typed identifiers in `cpp/frontier/core/ids.h`;
- stage wall-time and synchronization metrics in
  `cpp/frontier/entities/batch_stage.*`; and
- grouped routing and lane timing in the execution-time predictors.

## Runtime Semantics to Model

vLLM's documented DP-attention plus EP-MoE configuration treats the DP ranks
as a coordinated forward domain rather than independent engines. Forward
passes are aligned across DP ranks; a rank with no scheduled request runs an
empty forward; and expert layers synchronize on every forward. The expert
parallel group spans the shared DP/TP rank domain, whereas each attention
group covers the TP ranks within one DP lane.

The simulator does not need to imitate vLLM's Python process orchestration or
NCCL API calls. It must reproduce the observable timing and resource
constraints:

1. Attention work remains lane-local and may complete at different times.
2. Every aligned forward has exactly one participant per attention-DP lane:
   a real batch or a side-effect-free dummy participant.
3. The first expert dispatch begins only after the last participant reaches
   the corresponding pre-MoE point.
4. Routing and expert work use the aggregate tokens of all real participants.
5. The slowest EP lane determines expert completion.
6. All participants pass the post-MoE synchronization before advancing.
7. The participant set stays fixed for every MoE layer in that forward.
8. No second forward overlaps the shared physical domain while its EP phase is
   active, because DBO is not modeled.

References:

- [vLLM data-parallel deployment](https://docs.vllm.ai/en/v0.22.0/serving/data_parallel_deployment/)
- [vLLM expert-parallel deployment](https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md)
- [NCCL collective operation ordering](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)

## Scope

### Required in the first implementation

- sequential PDD with `enable_parallel_clusters=false`;
- unified `DECODE` clusters;
- MoE models with `attention_dp > 1` and `moe_ep > 1`;
- offline and online admission;
- real and dummy DP participants;
- unequal attention times caused by different scheduled-token and context
  totals;
- all MoE layers in a pipeline stage;
- fixed/dummy and analytical execution-time predictors;
- PP1 as the primary differential gate and a standalone C++ PP2 state-machine
  test; and
- deterministic metrics and event traces for the new synchronization
  contract.

### Follow-on audit using the same coordinator

After the PDD decode acceptance gate is green, apply the same invariants to:

- co-location pure-decode MoE (completed 2026-08-06);
- co-location mixed/prefill MoE; and
- sequential PDD `PREFILL`.

These paths should share one group/coordinator abstraction. They may retain
path-specific event types and timing composition where the operator sequence
really differs.

### Explicitly excluded

- Dual Batch Overlap or any attention/FFN microbatch overlap;
- `pd-af-disaggregation`, `DECODE_ATTN`, and `DECODE_FFN`;
- parallel PDD cluster execution;
- changes to request routing, batching policy, KV-transfer policy, cache
  affinity, or preemption policy except where needed to preserve the shared
  domain reservation;
- real CUDA streams, kernels, NCCL execution, or network contention; and
- matching an existing Python behavior when that behavior conflicts with this
  explicitly selected physical/vLLM contract.

## Correctness Invariants

Let `G` be one physical `(cluster, replica, pipeline stage)` domain, `g` one
aligned forward generation, `l` one stage-local MoE layer, and `d` an
attention-DP lane.

### Group and participant invariants

1. `g` contains exactly `attention_dp` participant slots.
2. A slot is either `real(batch_id)` or `dummy`; it is never both.
3. A real batch registered for `(G, g, d)` cannot be replaced by a dummy.
4. A dummy is permitted only when lane `d` launched no real work for `g`.
5. Group membership becomes immutable before the first expert dispatch and
   remains immutable until the forward/stage completes.
6. A real batch arriving after group closure is assigned to a later
   generation; it must not replace an in-flight dummy.
7. Duplicate or stale events cannot add a participant, advance a request, or
   release a resource twice.

### Timing invariants

For every group and MoE layer:

```text
ffn_start(G, g, l) = max(attention_ready(G, g, l, d))
                     over all DP participant lanes d
```

For a dummy participant, `attention_ready` is its aligned-forward readiness
time. It is not an unconditional timestamp-zero arrival and cannot stand in
for a registered real batch whose attention is still executing.

After aggregate routing:

```text
expert_finish = ffn_start
              + dispatch_time(group_tokens)
              + max(expert_lane_time[ep_id])
              + combine_time(group_tokens)
```

All real participants advance from the post-MoE point together. Their earlier
lane-local attention times remain visible in metrics, while their shared
expert critical path is identical.

### Resource invariants without DBO

1. The aligned forward holds an exclusive reservation on the shared physical
   domain from group launch, before any participant starts attention, through
   final stage completion.
2. The reservation permits the registered DP participants to execute their
   lane-local attention concurrently and then switch to the common EP view. A
   different forward that maps to any rank in the domain cannot begin
   attention or expert work until the reservation is released.
3. A dummy participant reserves its DP lane for the entire aligned forward;
   it is not destroyed and recreated at each MoE layer.
4. Requests arriving at a dummy-reserved lane remain schedulable for the next
   generation but cannot enter the current generation.
5. Releasing the reservation deterministically wakes pending stage work; the event
   loop must not poll by repeatedly rescheduling zero-progress events.

### Routing and state invariants

1. Expert routing receives the sum of scheduled tokens from every real DP
   participant in `g`.
2. Token conservation holds exactly:

   ```text
   sum(global_expert_token_counts)
     == total_real_group_tokens * router_topk
   ```

3. Each EP lane executes only its owned experts; the maximum predicted EP-lane
   time is the group critical time.
4. Dummy participants never allocate/release KV blocks, mutate requests,
   create transfers, or emit normal batch completion.
5. Every real request and batch advances exactly once at each state boundary.
6. Completed simulations contain no live group, barrier, dummy participant,
   or shared-domain reservation.

## Proposed Design

### 1. Use a stable aligned-forward identity

Remove `initial_pre_arrival` from synchronization-group identity and matching.
Time is an output of execution, not a correlation key.

Introduce or formalize a typed `MoEForwardId`/generation scoped by:

```text
(cluster_type, replica_id, stage_id, sync_path, forward_ordinal)
```

The existing lane-local `BatchGlobalId` cannot be reused: it is generated from
per-target counters and can diverge across DP lanes. Add an explicit ordinal
owned by the shared-domain coordinator. Do not overload the globally unique
`BatchId` and do not compare floating-point `SimTime` values to identify a
forward.

Generation counters must be monotonic within their owner, checked for
overflow, and included in stale-event validation.

### 2. Register the aligned forward at the existing stage-launch boundary

No new DES event is required for group launch. Keep
`ReplicaStageScheduleEvent -> begin_moe_stage()` as the point at which a real
lane starts attention, but make registration domain-aware:

- the first real lane creates the coordinator-owned open forward generation
  and acquires its domain reservation;
- another DP lane may join that same generation while the group is
  `collecting` and its participant slot is empty;
- another batch from an already occupied DP lane is deferred; and
- once the group closes, later real work is deferred to the next generation.

The reservation permits participant registration and attention for the
current generation but blocks creation of a second generation on the same
`(cluster, replica, stage)` domain.

Registration records:

- forward ID and generation;
- cluster, replica, stage, and DP lane;
- real batch ID;
- stage-local layer progress;
- lane attention start time and predicted first pre-MoE arrival; and
- the batch generation/ticket used for stale-event checks.

This makes every slow but active lane visible before its pre-MoE event fires.
When another lane reaches pre-MoE early,
`ensure_moe_group_participants()` sees a registered real slot and waits instead of
inserting an idle batch.

### 3. Close a group explicitly

Add a small shared-domain coordinator, either as a focused component owned by
`BaseClusterScheduler` or as clearly separated private state there. Suggested
state:

```text
MoEForwardGroup
  key
  generation
  state: collecting | pre_moe_wait | expert_active | post_moe_wait | complete
  participant[attention_dp]
  current_layer
  active_domain_reservation
  aggregate_routing
  pending_event_generation
```

The group closes at the first `pre_moe` arrival, after all real lanes whose
attention was already launched have registered. A registered real lane remains
real and pending; an unregistered lane becomes dummy. Work whose stage launch
occurs after closure belongs to the next forward generation.

This keeps the existing DES structure and uses neither an arbitrary timeout
nor timestamp equality. Add a zero-duration regression test to prove that
same-timestamp `ReplicaStageScheduleEvent` registration is processed before a
newly enqueued `DecodeSyncEvent(pre_moe)`. If that ordering cannot be guaranteed
by the existing `(time, sequence)` contract, add a launch-coalescing event only
as a narrowly justified follow-up; it is not part of the initial design.

### 4. Keep dummy participants for the full forward

Replace the current layer-local idle insertion/release behavior with a
persistent dummy participant owned by `MoEForwardGroup`.

The dummy carries identity and barrier readiness only. It does not need a
mutable fake request-bearing `Batch`. If compatibility with existing event
payloads temporarily requires an idle batch ID, keep it as a trace-only handle
owned by the group and retire it only when the group completes.

Real-over-dummy replacement is allowed only while the group is still
`collecting`. After closure, new real work is assigned to the next forward.

### 5. Add an aligned-forward domain reservation

Represent the no-DBO resource constraint explicitly with a
`AlignedForwardReservation` or equivalent state keyed by
`(cluster, replica, stage)`.

- Acquire it when the group launches and before any attention starts.
- Hold it through every attention, dispatch, expert-compute, combine, and
  synchronization phase in the stage.
- Release it only after the aligned forward's final stage completion has been
  scheduled exactly once for every real participant.
- While held, defer `ReplicaStageScheduleEvent` work that would use the same
  physical domain.
- On release, enqueue the deferred scheduling events in stable lane/order
  order.

The first implementation deliberately reserves the domain for the whole
multi-layer stage. Shortening the interval would be an optimization with new
overlap semantics and is deferred together with DBO.

### 6. Predict expert work from aggregate real tokens

Implementation note: analytical decode implements the group-layer predictor
API described below. It preserves the realized per-DP routing decisions by
summing their `global_expert_tokens`; it does not rerun the router with a new
random stream. The combined allocation is then predicted once per aligned
forward/layer. Predictors without a token-sensitive model may return the
lane-time-sum fallback carried in the same input.

Keep lane-local attention prediction unchanged. At the pre-MoE barrier,
construct an immutable `MoEGroupLayerInput` from all real participants:

```text
MoEGroupLayerInput
  forward/group identity
  model and topology
  stage-local/global layer ID
  total scheduled tokens
  per-real-batch token slices needed by routing
  routing seed/mode/distribution
```

Extend the fixed and analytical predictors with a group-layer API that returns:

```text
MoEGroupLayerPrediction
  global expert token counts
  per-EP-lane owned-expert counts
  dispatch time
  per-EP-lane expert time
  critical EP lane and time
  combine time
```

Do not create a runnable synthetic request batch just to call the predictor.
The routing allocation is produced once per group/layer. Every real batch then
references the same shared expert prediction while retaining its own attention
prediction and barrier wait.

### 7. Make synchronization events generation-safe

Keep the typed prefill/decode arrival and collective events, but ensure each
payload carries enough information to reject:

- an event for a completed forward;
- an event for a previous layer;
- a duplicate participant arrival;
- a collective release for the wrong generation; and
- a post-MoE event after the reservation or group has already advanced.

`maybe_ready()` should continue to compute the maximum participant arrival.
Its caller must schedule exactly one collective release per phase/generation.

### 8. Define metric ownership

Record group-level data once per `(forward, layer)`:

- all participant DP lanes and real/dummy flags;
- per-lane attention-ready time;
- pre-MoE barrier release/FFN start time;
- total real input and routed tokens;
- expert and EP-lane token arrays;
- dispatch, critical expert, and combine times;
- critical EP lane;
- post-MoE release time; and
- shared-domain reservation interval.

For each real batch/stage, retain:

- its lane-local predicted attention time;
- `pre_moe_wait_ms = ffn_start - own_attention_ready`;
- the shared expert critical-path duration;
- its actual stage wall time; and
- its final completion/request metrics.

Group-level expert time must not be summed once per DP batch in aggregate
cluster utilization output unless that metric explicitly represents occupied
rank-time. Document the unit and multiplicity of every affected metric.

## Target Event Flow

The first implementation should add no new DES event type. It reuses the
existing decode synchronization events and changes the group state they act
on:

| Event | Cardinality | Responsibility |
| --- | --- | --- |
| `ReplicaStageScheduleEvent` | One per scheduled real lane | Through `begin_moe_stage()`, join or create the open coordinator-owned generation before scheduling that lane's attention completion. The first real participant acquires the stage-lifetime domain reservation; other DP lanes may join the same collecting group. |
| `DecodeSyncEvent(pre_moe)` | One per participant per layer | Represent that lane's arrival at the pre-MoE barrier. A real lane arrives at `layer_attention_start + lane_attention_time`; a dummy arrives at the aligned layer start. |
| `DecodeSyncCollectiveEvent(pre_moe)` | One per group per layer | Fire at the maximum pre-MoE arrival, aggregate tokens across real participants, route once, predict dispatch and every EP lane, and set the group FFN completion from the critical EP lane. |
| `DecodeSyncEvent(post_moe)` | One per participant per layer | Represent completion of the shared dispatch, expert FFN, combine, and any path-specific communication charged at this boundary. These events normally share one timestamp because the expert phase is group-level. |
| `DecodeSyncCollectiveEvent(post_moe)` | One per group per layer | Advance every real participant exactly once. Start the next layer's lane-local attention for the same fixed participant set, or schedule final real-batch stage completion. |
| `BatchStageEndEvent` | One per real batch at final layer | Complete the existing real stage lifecycle. Notify the group coordinator; the last real completion releases the domain reservation and wakes deferred scheduling work. Dummies never receive this event. |

No separate `AttentionEndEvent` is needed: the simulator represents attention
execution by placing `DecodeSyncEvent(pre_moe)` at its predicted completion
time. No separate expert-end event is needed either: the pre-MoE collective
places the participant `DecodeSyncEvent(post_moe)` events at the predicted
group expert completion time.

The important change is therefore not a new event. It is that
`begin_moe_stage()` assigns a coordinator-owned generation without comparing
`BatchGlobalId` or `initial_pre_arrival`, and records the real participant
before enqueuing its `DecodeSyncEvent(pre_moe)`. At the first pre-MoE arrival,
`ensure_moe_group_participants()` may fill only still-unregistered lanes with
persistent dummies. It cannot replace any registered slow lane.

For two real DP lanes whose attention times differ:

```text
DP0 ReplicaStageScheduleEvent      DP1 ReplicaStageScheduleEvent
          |                                  |
 begin_moe_stage(): create g       begin_moe_stage(): join g
 reserve shared domain             register real DP1
 register real DP0                           |
          |                                  |
          +-------+-------------------------+
          |                                 |
     DP0 attention                     DP1 attention
          |                                 |
DecodeSyncEvent(pre_moe)       DecodeSyncEvent(pre_moe)
          |                                 |
          +---------------+-----------------+
                          |
          DecodeSyncCollectiveEvent(pre_moe)
             at max(DP0_ready, DP1_ready)
                                   |
                         continue under forward reservation
                                   |
                         aggregate route + EP dispatch
                                   |
                         max(EP lane expert times)
                                   |
                              EP combine
                                   |
               DecodeSyncEvent(post_moe) per participant
                                   |
              DecodeSyncCollectiveEvent(post_moe)
                                   |
                         retain forward reservation
                                   |
                   next layer for the same fixed participants
                                   |
                   final stage completion for all real participants
                                   |
                       BatchStageEndEvent per real batch
                                   |
                  last real completion releases reservation
                  and wakes the next domain generation
```

For one real lane and one truly empty lane, the empty slot is closed as a
dummy participant. The dummy participates in every barrier for that forward.
A later request on that DP lane waits for the next forward generation.

## Implementation Slices

### Slice 0: Freeze the failing behavior

1. Add a deterministic PDD decode fixture with `DP2/EP4` and two real batches
   whose context/scheduled-token sums produce clearly different analytical
   attention times.
2. Capture the current incorrect trace: different group/generation IDs or an
   idle replacement for a real active lane, and expert start before the slower
   attention finishes.
3. Add assertions against semantic event fields rather than allocator-local
   temporary batch IDs.

Exit gate: the new regression test fails for the intended reason on the
current implementation.

### Slice 1: Stable forward groups and stage-launch registration

1. Add the coordinator-owned typed forward identity and group lifecycle state.
2. Register each real lane in the open generation from `begin_moe_stage()`
   before enqueuing its attention-completion event.
3. Remove `initial_pre_arrival` equality from group matching.
4. Stop using lane-local `BatchGlobalId` as the cross-DP correlation key.
5. Close participant membership at the first pre-MoE arrival and materialize
   dummies only for lanes that never registered.
6. Preserve duplicate/stale-event idempotence and quiescence checks.

Exit gate: unequal real DP lanes join the same generation, and the pre-MoE
collective is scheduled exactly once at their maximum arrival time.

### Slice 2: Persistent dummy and no-DBO resource reservation

1. Replace per-layer idle creation/release with a forward-lifetime dummy.
2. Implement the aligned-forward reservation and pending-stage wakeup.
3. Queue late real work into the next generation.
4. Add deadlock diagnostics that print group, lane, generation, and reservation
   ownership if final quiescence fails.

Exit gate: a truly empty lane participates without side effects, a registered
slow lane is never replaced, and no overlapping group uses the domain while
the reservation is held.

### Slice 3: Aggregate routing and expert prediction

1. Add `MoEGroupLayerInput` and group prediction result types.
2. Aggregate all real DP token work once per layer.
3. Run routing once and calculate every EP lane's work.
4. Use the maximum EP-lane time for the shared critical path.
5. Adapt fixed predictor fixtures and analytical predictor formulas.

Exit gate: routing conserves aggregate tokens exactly and changing work on
either real DP lane can change the critical EP lane/time.

### Slice 4: Metrics and normalized output

1. Add group/layer synchronization diagnostics.
2. Attribute per-batch wait and wall time without double-counting group work.
3. Include reservation state in debug traces and final quiescence.
4. Update the normalized output comparator for the intentionally changed
   synchronization contract.

Exit gate: the fast lane's recorded wait equals the difference between its
attention-ready time and the shared FFN start, and aggregate expert work is
reported once at group level.

### Slice 5: Integration and path audit

1. Run the PDD decode matrix below.
2. Audit co-location decode, co-location prefill/mixed, and PDD prefill for the
   same timestamp-keyed grouping or transient-idle problem.
3. Migrate each affected path to the common coordinator.
4. Update the older Step 3.5 roadmap's parity claims or add an implementation
   record explaining that physical/vLLM semantics intentionally supersede the
   former production-Python edge behavior.
5. Run Debug, Release, ASan, UBSan, and differential gates.

Exit gate: all required C++ tests pass, intentionally changed Python
differential cases use the new contract, and all unaffected dense and MoE
cases remain green.

## Test Plan

### Focused unit tests

- unequal real arrivals in one group release at the maximum arrival;
- equal arrivals still release once and deterministically;
- a registered real participant is never idle-replaced;
- a genuinely empty lane receives one persistent dummy;
- a dummy remains reserved across all layers in the forward;
- a real batch arriving after closure enters the next generation;
- exact timestamp equality is not required for group identity;
- duplicate and stale arrivals/collectives are idempotent;
- the domain reservation blocks another overlapping group and wakes it once in stable
  order;
- aggregate routing conserves `tokens * topk`;
- the predicted critical lane is the maximum of all EP-lane predictions;
- dummy participants cannot mutate request, KV, transfer, or completion state;
- group completion releases every barrier, participant, and reservation; and
- generation/ordinal overflow and invalid transitions fail explicitly.

### PDD integration tests

1. `DP2/EP4`, two real batches, highly unequal context totals:
   - different attention-ready times;
   - same forward/group and sync generation;
   - no dummy for either active lane;
   - FFN start equals the slower arrival; and
   - both batches share the aggregate critical expert duration.
2. `DP2/EP4`, one real batch and one empty lane:
   - one persistent dummy;
   - synchronized FFN entry; and
   - no dummy request/KV side effects.
3. A request arrives on a dummy-reserved lane during the active forward:
   - it is queued for the next generation; and
   - it never joins the closed group.
4. Multiple MoE layers:
   - fixed membership at every layer;
   - repeated pre/post barriers; and
   - no collective ordering inversion; and
   - one domain reservation spans the whole multi-layer stage.
5. Multiple replicas and PP stages:
   - groups and reservations are isolated by replica/stage;
   - PP1 is required for the first end-to-end gate; and
   - PP2 has an independent C++ state-machine test even if the Python oracle
     cannot complete the same topology.
6. Slow KV transfer, prefix-cache hit/miss, block pressure, and preemption:
   - admission timing may change, but an active group remains generation-safe;
   - requests and KV ownership still complete exactly once.
7. Online staggered arrivals and offline batches:
   - no starvation or indefinite open group;
   - deterministic group closure and completion.

### Regression gates

- `DP1/EP1` local-MoE fast path unchanged;
- dense models and dense PDD unchanged;
- existing MoE routing golden vectors unchanged where their input is already
  a complete group;
- co-location mixed/prefill and PDD prefill cases unchanged until deliberately
  migrated;
- strict final simulator quiescence;
- Debug and Release CTest;
- MSVC ASan and available GCC/Clang UBSan coverage; and
- targeted Python/C++ differential tests, with the old unequal-DP timing
  expectation replaced rather than preserved.

Suggested test locations:

- `cpp/tests/simulator/dp_ep_lockstep_test.cc` for focused coordinator tests;
- `cpp/tests/simulator/moe_integration_test.cc` for end-to-end PDD cases;
- `cpp/tests/parity/test_moe_differential.py` for normalized differential
  coverage; and
- `cpp/CMakeLists.txt` for test registration.

## Acceptance Criteria

The implementation is complete only when all of the following are true:

- for every tested group/layer,
  `ffn_start == max(participant_attention_ready)` within the named timing
  tolerance;
- no expert dispatch, expert compute, or combine event begins before the last
  registered real DP participant reaches the matching pre-MoE point;
- no active real DP lane is represented by an idle/dummy participant;
- the participant set remains fixed for the full aligned forward;
- without DBO, no other forward overlaps the reserved shared physical domain;
- routing uses all real DP batches and conserves aggregate routed tokens;
- the slowest aggregate EP lane determines expert completion;
- fast lanes report the correct synchronization wait;
- dummy participants have no request, KV, transfer, throughput, or normal
  completion effects;
- every real request/batch/stage advances and completes exactly once;
- late and stale events cannot enter a newer generation;
- completed simulations have no live group, barrier, dummy, deferred schedule,
  or reservation state; and
- unaffected dense, local-MoE, routing, transfer, cache, and scheduler tests
  remain green.

## Risks and Required Design Decisions

### Aligned-forward closure

Independent lane schedulers can drift in ordinal or become temporarily empty.
The implementation must define the scheduler-visible point at which a forward
is closed. A simulated timeout is not acceptable because it makes correctness
depend on workload timing.

### Reservation granularity

The first implementation holds the exclusive reservation for the complete
aligned stage. This is intentionally conservative and matches the decision to
exclude DBO. A future optimization may shorten the interval only under a new,
explicit overlap contract with rank-ownership and collective-ordering tests.

### Pipeline parallelism

The Python oracle has a known PDD decode MoE issue for some `PP > 1`
topologies. PP correctness must therefore have a direct C++ state-machine
test, and lack of a completed Python run must not justify copying broken or
undefined behavior.

### Metric multiplicity

The group has one physical expert critical path but several real batches
experience it. Request latency may reference the same duration per batch;
aggregate utilization/compute metrics must not accidentally count the same
group work once per batch.

### Predictor compatibility

Some predictor paths currently accept only a single `Batch`. The group input
API must preserve all fields needed for first-layer scaling, context-dependent
attention, deterministic routing seed derivation, and communication payload
sizing without introducing a mutable synthetic batch.

### Intentional parity change

The completed Step 3.5 record treated production Python as the oracle. For
this specific unequal-DP shared-domain case, the selected behavior is the
hardware/vLLM lockstep contract. Differential fixtures and documentation must
make this exception explicit so a future parity cleanup does not reintroduce
timestamp-keyed group splitting or active-lane idle replacement.

## Files Expected to Change During Implementation

The exact split may evolve, but the implementation should remain concentrated
in these areas:

```text
cpp/frontier/core/ids.h
cpp/frontier/core/event.h
cpp/frontier/entities/batch.h
cpp/frontier/entities/batch.cc
cpp/frontier/entities/batch_stage.h
cpp/frontier/entities/batch_stage.cc
cpp/frontier/events/replica_stage_schedule_event.cc
cpp/frontier/events/decode_sync_event.cc
cpp/frontier/events/decode_sync_collective_event.cc
cpp/frontier/scheduler/cluster_scheduler/base_cluster_scheduler.h
cpp/frontier/scheduler/cluster_scheduler/base_cluster_scheduler.cc
cpp/frontier/scheduler/replica_stage_scheduler/*
cpp/frontier/execution_time_predictor/base_execution_time_predictor.h
cpp/frontier/execution_time_predictor/fixed_execution_time_predictor.*
cpp/frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.*
cpp/frontier/metrics/*
cpp/tests/simulator/dp_ep_lockstep_test.cc
cpp/tests/simulator/moe_integration_test.cc
cpp/tests/parity/test_moe_differential.py
cpp/CMakeLists.txt
```

Do not broaden the change into AFD, DBO, or general scheduler-policy redesign.
The implemented decode milestone is successful when unequal PDD or
co-location decode DP lanes wait for one another and enter one aggregate
expert FFN phase together.
