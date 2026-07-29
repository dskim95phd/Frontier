# C++ Porting Step 2.5: Dense Co-location Event and Parallelism Parity

## Status

Completed on 2026-07-30, including the expanded configuration-matrix
hardening.

The implementation is available through normalized schema v3. Schema v1 and
v2 remain frozen and continue to use their original execution paths.

Completion evidence:

- the production Python `Simulator`, event classes, round-robin cluster
  scheduler, vLLM V1 replica scheduler, and analytical predictor serve as the
  differential oracle;
- fixed online `replicas=2, DP=2, PP=2, TP=2`, fixed offline
  `DP=2, PP=4, TP=2`, analytical offline `DP=2, PP=2, TP=4`, and target-local
  DP pressure/preemption cases match on request ownership, event order,
  batches, stages, timings, completion state, and every target-local scheduler
  iteration (decisions, budgets, KV blocks, and queue counts);
- the analytical unit matrix covers TP 1/2/4/8 crossed with PP 1/2/4;
- CTest passes 18/18 and the full Python/C++ differential suite passes 49/49;
- the expanded 14-case configuration matrix passes 14/14, including complex
  fixed and analytical PP4/PP8 terminal-drain cases; and
- clean GCC 13.3, AddressSanitizer, and UndefinedBehaviorSanitizer builds pass.

Expanded-matrix fixes:

1. When a completed request still has a pending MONOLITHIC+PP terminal-release
   boundary, C++ now blocks waiting admission until the release is
   materialized, matching Python.
2. When one `ReplicaScheduleEvent` produces one or more batches and then ends
   with an empty internal scheduling iteration, C++ now leaves the terminal
   follow-up poll pending. It consumes the poll only when the whole event
   produced no batches, matching Python.
3. On an empty iteration that releases terminal state, the C++ trace currently
   snapshots queue/KV counters after the release, matching Python's
   `iteration_end` diagnostic.

Step 2.5 is a required compatibility milestone between the completed
single-target Step 2 scheduler and Step 3 sequential PDD. It replaces the
flattened Step 2 simulator loop with the production Python event and scheduler
boundaries, then expands dense co-location to real TP, PP, DP, and
multi-replica execution.

Step 2 remains a valid completed vertical slice. Schema v1 and v2 behavior must
remain frozen while Step 2.5 introduces a new schema version and a wider
execution path.

## Why Step 2.5 Exists

Step 2 intentionally collapsed the Python event pipeline into:

```text
RequestArrival
  -> SchedulerPoll
  -> BatchCompletion
```

It also fixed the topology to one replica, one DP target, and one pipeline
stage. This was enough to validate the vLLM V1 token/KV scheduling algorithm,
but it left two structural gaps:

1. `CoLocationSimulator` still owns event behavior that belongs to explicit
   event handlers.
2. The global, cluster, replica, and replica-stage scheduler hierarchy exists,
   but the upper layers are pass-through skeletons and the replica scheduler
   permits only one in-flight batch.

These gaps must be closed before PDD, prefix-cache affinity, or more complex
parallel workloads are added.

## Goal

End Step 2.5 with deterministic dense co-location runs whose event transitions,
routing decisions, batches, per-stage execution, request metrics, and
analytical timing match the production Python simulator for:

- tensor parallelism (TP);
- pipeline parallelism (PP);
- data parallelism (DP);
- more than one replica; and
- mixed online/offline workloads using the vLLM V1 scheduler.

The target execution shape is:

```text
RequestArrivalEvent
  -> GlobalScheduleEvent
  -> ClusterScheduleEvent
  -> ReplicaScheduleEvent
  -> BatchStageArrivalEvent(stage=0)
  -> ReplicaStageScheduleEvent(stage=0)
  -> BatchStageEndEvent(stage=0)
       -> BatchStageArrivalEvent(stage=1) -> ...
  -> ClusterBatchEndEvent
  -> GlobalBatchEndEvent
  -> ReplicaScheduleEvent
```

## Meaning of Parallelism in This Milestone

The three forms of parallelism have different simulator semantics and must not
be represented by one generic multiplier.

### Tensor parallelism

- One logical stage execution represents a TP worker group.
- TP does not create independent request schedulers or per-rank DES events for
  dense co-location.
- Compute shapes are sharded by TP size.
- Attention and dense-MLP TP collectives are included in the stage execution
  prediction.
- TP1 has no TP collective cost.

This matches the production Python dense path: TP is an execution-time and
communication concern, not a request-routing concern.

### Pipeline parallelism

- A replica target owns `PP` independent `ReplicaStageScheduler` objects.
- Each stage serializes its own work while different stages may overlap.
- A batch moves through every stage in order.
- The replica scheduler may keep up to `PP` batches in flight, matching the
  Python monolithic scheduling capacity.
- Request token state mutates once, after the final stage; it does not mutate
  at each stage.
- A non-final stage includes the configured PP send/receive cost before the
  next-stage arrival.

### Data parallelism

- Each `(replica_id, dp_id)` is an independent scheduling and KV-capacity
  target.
- Each DP target owns a full TP x PP model-parallel group.
- Waiting/running/preempted queues, block accounting, iteration counters,
  in-flight batches, and stage queues are target-local.
- Dense DP lanes do not synchronize or perform DP collectives.
- The cluster scheduler routes each request to exactly one target.

### Multiple replicas

Multiple replicas are included even though they are not another model-parallel
axis. Without them, the cluster scheduler would remain mostly a skeleton and
the later prefix-cache affinity work would still lack its real target space.

The routing target set is:

```text
num_replicas x data_parallel_size
```

## Fixed Scope

### Included

- `co-location` / `MONOLITHIC`;
- dense Llama-2-7B;
- vLLM V1 replica scheduling with FCFS;
- deterministic round-robin cluster routing;
- `num_replicas >= 1`;
- TP, PP, and DP sizes greater than or equal to one;
- explicit per-target KV-block capacity;
- continuous batching, token budgeting, chunked prefill, and recompute
  preemption from Step 2;
- multiple in-flight batches required by PP;
- fixed per-stage timing for structural tests;
- analytical dense compute, TP collective, and PP transfer timing;
- explicit event classes for the dense co-location execution path;
- per-target and per-stage metrics/traces;
- offline and online trace semantics; and
- deterministic Python/C++ differential tests over combined parallelism.

### Excluded

- PDD and every AFD surface;
- parallel-cluster threads or multiple cluster event queues;
- MoE, EP, expert imbalance, idle synchronization batches, and MoE DP
  collectives;
- prefix caching, sticky routing, block hashes, and CPU KV-cache tiering;
- speculative decoding / MTP;
- Thinking Mode;
- CUDA Graph timing;
- profile-trained predictors;
- SGLang, Sarathi, Orca, SJ2Q, and other replica schedulers;
- random, sticky, and load-aware cluster policies; and
- multi-node or heterogeneous device placement.

Round-robin is the only cluster policy required to prove correct target
ownership and parallel execution. LOR and sticky policies should be added only
when their own feature milestone needs them.

## Configuration Contract

Introduce normalized schema v3. Schema v1 and v2 remain byte-compatible and
continue to run through their existing contracts.

Schema v3 adds one source of truth for topology:

```json
{
  "schema_version": 3,
  "run_id": "step2-5-dense-parallel",
  "simulation_mode": "offline",
  "system_architecture": "co-location",
  "enable_parallel_clusters": false,
  "parallelism": {
    "num_replicas": 1,
    "tensor_parallel_size": 8,
    "pipeline_parallel_size": 2,
    "data_parallel_size": 2
  },
  "cluster_scheduler": {
    "type": "round_robin"
  },
  "prefix_cache": {
    "enabled": false,
    "key_mode": "session"
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
    "type": "analytical",
    "device": "rubin",
    "model": "llama2-7b",
    "precision": "fp16",
    "num_layers": 32,
    "network_bandwidth_gbps": 400.0,
    "network_latency_us": 1.0,
    "intra_node_bandwidth_gbps": 14400.0
  }
}
```

In schema v3, `tensor_parallel_size` is removed from `execution_model`;
`parallelism.tensor_parallel_size` is authoritative.

The fixed structural-test variant uses:

```json
{
  "type": "fixed",
  "stage_latencies_ms": [1.0, 3.0]
}
```

The array length must equal `pipeline_parallel_size`. This makes pipeline
overlap and backpressure observable without analytical-model noise.

Validation:

- every parallelism dimension is positive;
- `num_layers % pipeline_parallel_size == 0`;
- fixed `stage_latencies_ms` has exactly `PP` finite, nonnegative entries;
- model hidden dimensions and attention heads are divisible by TP as required
  by the Python `Replica` contract;
- `num_replicas * DP * PP * TP` fits the supported single Rubin NVL72 domain;
- scheduler and KV capacity values apply independently to every
  `(replica_id, dp_id)` target;
- `enable_parallel_clusters` remains false because model parallelism is not
  Python's parallel-cluster execution mode;
- prefix caching remains disabled; and
- unsupported models, placements, policies, and feature combinations fail
  before simulator construction.

## C++ Layout

Headers and implementations remain together in functional directories:

```text
cpp/frontier/
  core/
    event.h
    event_queue.h
  entities/
    request.*
    batch.*
    batch_stage.*
    cluster.*
    replica.*
    execution_time.*
  events/
    base_event.h
    co_location_events.*
  scheduler/
    global_scheduler/
    cluster_scheduler/
    replica_scheduler/
    replica_stage_scheduler/
  execution_time_predictor/
  metrics/
  simulator/
    simulation_context.*
```

`co_location_events.*` groups the nine named concrete handlers in one
functional module; the handlers remain separate types registered by
`EventDispatcher`.

Event payloads continue to carry stable IDs and generation/epoch snapshots, not
raw pointers. Concrete event types may share a `BaseEvent` interface, but event
ownership must remain explicit and deterministic. The queue ordering contract
remains `(time, creation_sequence)`.

`SimulationContext` exposes the request, batch, and batch-stage arenas, the
global scheduler, metrics, and event emission. `CoLocationSimulator` only:

1. initializes the context and arrival events;
2. pops the next event;
3. invokes its handler; and
4. enqueues returned events.

It must not contain request/scheduler/stage state-transition switches.

## Entity Work

### Cluster

- Own the replica set and cluster type.
- Expose deterministic replica iteration order.
- Own or expose the execution/communication model shared by compatible
  replica targets.

### Replica

- Store replica ID and TP/PP/DP topology.
- Validate layer, hidden-size, Q-head, and KV-head divisibility.
- Define `num_layers_per_pipeline_stage`.
- Provide the physical size of one replica as `TP * PP * DP`.

### Batch

Extend the Step 2 batch with:

- globally unique batch ID;
- replica ID and DP ID;
- schedule iteration and schedule epoch;
- current/next stage metadata;
- per-request execution and mutation snapshots;
- scheduled/completed timestamps; and
- per-stage execution records.

Batch completion at intermediate stages must not advance request-visible token
state.

### BatchStage and ExecutionTime

Add explicit values for:

- batch ID, replica ID, DP ID, and stage ID;
- stage arrival/start/end timestamps;
- number of layers in the stage;
- dense compute time;
- TP collective time;
- PP send/receive time; and
- total stage time.

## Event Contract

### `RequestArrivalEvent`

- Mark request arrival.
- Queue it in the global scheduler for `MONOLITHIC`.
- Record arrival metrics.
- Emit `GlobalScheduleEvent`.

### `GlobalScheduleEvent`

- Drain global queued requests into the monolithic cluster scheduler.
- Clear only entries included in this scheduling decision.
- Emit one `ClusterScheduleEvent` for each affected cluster.

### `ClusterScheduleEvent`

- Run round-robin routing over the ordered
  `(replica_id, dp_id)` target set.
- Add each request to exactly one replica scheduler.
- Emit one `ReplicaScheduleEvent` per affected target, with duplicates
  coalesced deterministically.

### `ReplicaScheduleEvent`

- Call the selected target's `on_schedule(time)`.
- Permit zero or more batches.
- Mark every returned batch scheduled.
- Emit `BatchStageArrivalEvent(stage=0)` for each batch.

### `BatchStageArrivalEvent`

- Validate batch schedule epoch.
- Enqueue the batch in the selected stage scheduler.
- Emit a stage schedule event only when the stage is idle.

### `ReplicaStageScheduleEvent`

- Pop one live batch if the stage is idle.
- Drop stale stage tickets.
- Predict stage execution time.
- Mark the batch stage started.
- Emit `BatchStageEndEvent` at `time + stage_duration`.

### `BatchStageEndEvent`

- Validate batch/request snapshots.
- Mark the stage idle and record stage metrics.
- If queued work remains, emit a same-stage schedule event.
- If this is not the last stage, emit the next stage arrival.
- Otherwise emit `ClusterBatchEndEvent`.

### `ClusterBatchEndEvent`

- Perform monolithic cluster-end bookkeeping.
- Emit `GlobalBatchEndEvent`.

### `GlobalBatchEndEvent`

- Advance valid request token state once.
- Apply the production Python per-request stale-snapshot behavior.
- Release or retain per-target KV blocks as Python does.
- Decrement the target's in-flight batch count.
- Record batch/request completion metrics.
- Emit a target-local `ReplicaScheduleEvent`.

`BatchEndEvent`, sync/collective events, KV-transfer events, periodic events,
and Thinking events are not used by this dense co-location slice.

## Scheduler Work

### Global scheduler

Port the functional Python responsibilities:

- queued `(RequestId, ClusterType)` entries;
- deterministic `schedule()` grouping;
- queue clearing after dispatch;
- cluster scheduler lookup; and
- full emptiness/debug-state reporting.

### Round-robin cluster scheduler

- Own a scheduler for every `(replica_id, dp_id)` target.
- Match production Python's deterministic replica-first batch distribution:
  distribute a scheduling cycle across replicas, then split each replica's
  slice evenly across its DP lanes.
- Route each request once and advance the round-robin cursor.
- Expose target and stage scheduler lookup.
- Report per-target queue, block, in-flight, and stage state.

### vLLM V1 replica scheduler

Keep the Step 2 scheduling algorithm, then add the Python functionality needed
by PP:

- replace `has_in_flight_batch` with an in-flight count and batch registry;
- allow `on_schedule()` to return batches until
  `num_running_batches == PP`;
- track request IDs active in in-flight batches;
- do not schedule a request already active in another PP batch;
- preserve scheduler-computed token frontiers across multiple outstanding
  batches;
- keep allocation, preemption, and rollback target-local;
- release active-request markers at final batch completion;
- preserve Python monolithic PP terminal resource-release semantics where they
  affect observable scheduling; and
- make iteration IDs target-local while batch IDs remain globally unique.

### Replica stage scheduler

- One instance per `(replica_id, dp_id, stage_id)`.
- Serialize execution within a stage.
- Order queued batches by `(global_batch_id, insertion_sequence)`.
- Store batch schedule epoch in each ticket.
- Drop stale tickets without leaving the stage busy.
- Expose queue/busy state for liveness diagnostics.

## Execution-Time and Communication Work

Refactor whole-batch prediction into stage prediction:

```text
predict_stage(batch, replica_id, dp_id, stage_id)
  -> StageExecutionPrediction
```

For analytical dense execution:

- parameterize dense matrix shapes by TP instead of using a hard-coded TP8
  model;
- compute only `num_layers / PP` layers per stage;
- use no TP collective at TP1;
- add the Python-compatible attention and dense-MLP TP collective costs for
  TP > 1;
- add PP activation send/receive cost on every non-final stage;
- avoid adding any dense DP collective;
- expose compute, TP communication, PP communication, and total duration
  separately; and
- validate every duration as finite and nonnegative.

The initial physical placement is one homogeneous Rubin NVL72 domain.
Multi-node rank placement and topology-dependent link selection remain
separate work.

## Metrics and Output Contract

Schema v3 extends normalized output with:

```text
requests[]
  replica_id
  dp_id

batches[]
  replica_id
  dp_id
  num_pipeline_stages

batch_stages[]
  batch_id
  replica_id
  dp_id
  stage_id
  arrived_at_s
  started_at_s
  completed_at_s
  dense_compute_ms
  tp_communication_ms
  pp_communication_ms
  duration_ms

event_trace[]
  sequence
  simulation_time_s
  event_type
  optional request/batch/replica/dp/stage IDs

scheduler_trace[]
  replica_id
  dp_id
  target_iteration_id
  existing Step 2 decision fields
```

IDs, target assignments, stages, counts, queue order, event order, batch order,
and token counts compare exactly. Tolerance is used only for explicit
floating-point timing fields.

## Implementation Slices

### Step 2.5A: Contract and Python Oracle

- Freeze schema v3 and invalid-topology fixtures.
- Extend the production Python oracle output with target, event, batch-stage,
  and timing breakdown fields.
- Add a deterministic fixed per-stage predictor for structural parity.
- Freeze a TP1/PP1/DP1 event trace before changing C++ execution.

Acceptance:

- The oracle invokes production Python events and schedulers.
- It does not reimplement routing, PP, or vLLM V1 scheduling.
- Schema v1/v2 golden files remain unchanged.

### Step 2.5B: Typed Event Pipeline at 1 x 1 x 1

- Add `SimulationContext` and the nine dense co-location events.
- Move all schema-v3 state transitions out of `CoLocationSimulator`.
- Run the existing single-target Step 2 behavior through the new pipeline.

Acceptance:

- Python/C++ event type and equal-time order match.
- Step 2 scheduler decisions and metrics remain unchanged.
- The simulator loop contains no event-type business-logic switch.

### Step 2.5C: Entities and Functional Scheduler Ownership

- Add Cluster, Replica, BatchStage, and ExecutionTime.
- Make global scheduler queueing/dispatch functional.
- Implement deterministic round-robin cluster routing.
- Instantiate the full `(replica_id, dp_id)` target matrix.
- Give every target independent replica/stage schedulers and block state.

Fixtures:

- DP2 same-time arrivals;
- DP2 mid-flight arrivals;
- two replicas with DP2;
- per-target block pressure; and
- target-local quiescence failure diagnostics.

Acceptance:

- Target assignments match Python exactly.
- No request appears in two targets.
- One target's block pressure does not affect another target.
- All target queues and allocations drain to zero.

### Step 2.5D: Multi-in-flight Replica Scheduling

- Replace the single in-flight flag with Python-compatible batch accounting.
- Add active-batch request tracking.
- Fill up to PP pipeline capacity without duplicate request scheduling.
- Preserve preemption and stale completion behavior with several live batches.

Acceptance:

- `0 <= num_running_batches <= PP`.
- A request is active in at most one live PP batch.
- Batch completion order may differ from creation order only when stage timing
  makes Python do so.
- Stale completion never leaks an in-flight slot.

### Step 2.5E: Pipeline Stage Execution

- Instantiate PP stage schedulers.
- Implement stage arrival/schedule/end forwarding.
- Add fixed per-stage timing and PP overlap tests.
- Add PP terminal resource-release behavior required by Python.

Fixtures:

- PP2 with at least three batches;
- PP4 pipeline fill and drain;
- a slow middle stage;
- simultaneous completion and arrival at one stage;
- preemption while other batches are in the pipeline; and
- stale batch tickets at an intermediate stage.

Acceptance:

- A stage never executes two batches simultaneously.
- Different stages overlap when Python overlaps them.
- Request progress changes only at global batch end.
- Stage timestamps and final request timestamps match Python.

### Step 2.5F: Configurable TP and Analytical Stage Timing

- Generalize Llama-2-7B dense formulas from TP8 to the supported TP matrix.
- Add per-stage layer count.
- Add TP collective and PP send/receive components.
- Generate analytical golden data from production Python.

Primary matrix:

```text
TP: 1, 2, 4, 8
PP: 1, 2, 4
```

Acceptance:

- Stage compute/TP/PP components match Python within documented tolerance.
- TP1 communication is exactly zero.
- Final-stage PP communication is exactly zero.
- Summed stage components equal stage duration.

### Step 2.5G: Combined Parallelism and Hardening

Run nontrivial combined cases, including:

```text
replicas=1, DP=1, PP=1, TP=8
replicas=1, DP=2, PP=2, TP=4
replicas=2, DP=2, PP=2, TP=2
replicas=1, DP=2, PP=4, TP=2
```

Workloads must include:

- same-time bursts larger than total target capacity;
- staggered arrivals while every target has in-flight work;
- mixed long prefills and decodes;
- chunked prefills;
- tight token budgets;
- per-target KV pressure and recompute preemption; and
- unequal stage durations that exercise pipeline backpressure.

Acceptance:

- normalized Python/C++ output parity;
- byte-stable repeated C++ output;
- exact integer/ID/order comparison;
- float tolerance only on declared timing values;
- no live batches, busy stages, queued tickets, allocated blocks, or pending
  requests at successful termination;
- CTest and differential pytest pass;
- clean GCC WSL build;
- AddressSanitizer and UndefinedBehaviorSanitizer pass; and
- no regression in schema v1/v2 suites.

## Invariants

- Every request has exactly one `(replica_id, dp_id)` owner after cluster
  routing.
- Every batch belongs to exactly one target.
- Batch IDs are globally unique; scheduler iteration IDs are target-local.
- A batch occupies at most one stage at a time.
- A stage executes at most one batch at a time.
- Request token state advances only after the batch leaves the final stage.
- KV accounting is local to a target and never exceeds that target's capacity.
- PP capacity is based on pipeline stages, not DP size.
- TP and PP sizes affect execution timing; DP size does not divide a batch's
  tokens or model weights.
- Successful termination means all global, cluster, replica, and stage state is
  empty.

## Additional Work Required Beyond Events and Schedulers

PP, TP, and DP support would be incomplete without the following:

1. **Topology entities and validation** so TP/PP/DP are not disconnected
   integers.
2. **Multi-replica routing** so the cluster layer becomes real and later
   prefix affinity has the correct target space.
3. **Stage-level prediction and communication breakdown** because the current
   C++ predictor returns one whole-model batch duration.
4. **Multi-in-flight request protection** because the current C++ vLLM V1
   scheduler has a single in-flight boolean.
5. **Per-target KV and liveness accounting** so DP lanes cannot accidentally
   share scheduler or memory state.
6. **Stage/event metrics and parity traces** so a matching final timestamp
   cannot hide a different pipeline execution.
7. **A new schema version** so the frozen Step 2 contract is not silently
   reinterpreted.

## Definition of Done

Step 2.5 is complete when the C++ simulator executes the production Python
dense co-location event graph, supports real TP/PP/DP and multiple replicas,
and matches Python on routing, scheduling, stage overlap, timing breakdown,
request completion, and final empty-state invariants over the combined
parallelism matrix.

Passing only final request metrics is insufficient. Event, target, batch,
stage, and scheduler traces are part of the parity contract.
