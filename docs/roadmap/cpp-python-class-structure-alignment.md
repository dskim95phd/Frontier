# C++ / Python Class-Structure Alignment Checklist

## Status

Implemented on the C++ port branch before Step 4.

- [x] Runtime `Cluster` / `Replica` graph
- [x] Simulator-owned per-cluster predictors and cluster-owned communication backend
- [x] `(batch.global_id, insertion_order)` stage priority
- [x] Runtime `MetricsStore`
- [x] KV-transfer and communication-backend interfaces
- [x] Single concrete `GlobalScheduler`
- [x] Python-aligned scheduler construction chain
- [x] `SimulationContext` absorbed into `Simulator`
- [x] Python-aligned `AnalyticalRooflineExecutionTimePredictor` responsibility
- [x] Shared `BaseExecutionTimePredictor` contract and predictor factory
- [x] Common replica lifecycle/state in `BaseReplicaScheduler`
- [x] Replica-scheduler construction isolated behind a factory
- [x] MoE synchronization state returned to `BaseClusterScheduler`
- [x] MoE barrier hidden behind cluster-scheduler event entry points

The default rule is to preserve the production Python ownership and class
responsibility boundaries unless C++ requires a materially different design.
Language-level adaptations such as typed IDs, value types, and event payload
variants are not considered structural drift.

## 1. Connect `Cluster` and `Replica` to the Runtime Graph

### Previous Difference

Python constructs and uses the following runtime entity graph:

```text
Simulator
  -> Cluster
     -> Replica
```

The C++ `Cluster` and `Replica` classes previously existed without being used
by the simulation runtime. The old `SimulationContext` also constructed
replica schedulers directly from configuration.

### Target

- Make each runtime cluster own its `Replica` entities.
- Make cluster schedulers operate against the corresponding `Cluster`.
- Make replica schedulers refer to their owning `Replica`.
- Keep `EntityArena` for stable `Request`, `Batch`, and `BatchStage` storage;
  it should complement rather than replace the cluster/replica entity graph.
- Remove or consolidate any duplicate topology state after the entity graph is
  authoritative.

### Exit Gate

- No production runtime `Cluster` or `Replica` class is dead code.
- Cluster and replica identity/topology have one authoritative owner.
- Existing dense and MoE differential suites remain unchanged.

Implemented: `Simulator` owns cluster entities, clusters own replicas, and
cluster/replica schedulers keep lifetime-safe references to those entities.
`GlobalScheduler` constructs cluster schedulers, and `BaseClusterScheduler`
constructs replica schedulers, matching the production Python chain. The
duplicate `cluster_parallelism_` map and the separate `SimulationContext`
facade were removed.

## 2. Align Predictor and Backend Ownership

### Previous Difference

Python creates one predictor per cluster and injects it through the scheduler
hierarchy. C++ previously created an execution model per replica scheduler and
cloned it for pipeline stages.

### Target

- Construct one execution-time predictor per cluster in `Simulator`.
- Let each cluster own its communication backend.
- Inject each predictor through global → cluster → replica → stage schedulers.
- Let replica and stage schedulers hold a lifetime-safe reference or
  `shared_ptr<const ...>` instead of independent clones.
- Preserve immutable, thread-safe prediction interfaces where possible.
- Keep fixed and analytical implementations behind the same public predictor
  contract needed by future trained predictors.

### Exit Gate

- Predictor/backend construction occurs once per intended Python ownership
  domain.
- PP stages do not silently duplicate stateful predictor caches.
- Fixed and analytical parity remains unchanged.

Implemented: `Simulator` owns the per-cluster predictor map, matching Python's
`Simulator._predictors`, while each `Cluster` owns its communication backend.
`GlobalScheduler` injects the matching predictor into each cluster scheduler;
replica and PP-stage schedulers share it rather than cloning it.
`BaseExecutionTimePredictor` is the scheduler-facing contract, while a factory
selects fixed or analytical implementations. The former public
`AnalyticalBatchExecutionModel` and `frontier/moe/` calculation modules were
consolidated behind the Python-aligned
`AnalyticalRooflineExecutionTimePredictor` entry point. Its dense roofline,
routing, EP ownership, and MoE timing implementation now lives in one `.cc`
file rather than being exposed as a separate runtime subsystem.

## 2.1 Align Replica-Scheduler Ownership

Python's base replica scheduler owns request admission, in-flight batch
lifecycle, pipeline capacity, and shared request state. The C++ base now owns
those same cross-policy responsibilities. `VllmV1Scheduler` retains only the
vLLM-specific batching, preemption, KV accounting policy, and completion
mutation hooks.

`BaseClusterScheduler` constructs replica schedulers through
`make_replica_scheduler()`. Adding another scheduler therefore requires a new
derived policy and one factory case, without changing cluster scheduling.

## 3. Match `ReplicaStageScheduler` Queue Ordering

### Previous Difference

Python uses a priority queue ordered by:

```text
(batch.global_id, insertion_order)
```

C++ currently uses a FIFO `deque`. Existing covered workloads match, but the
Python ordering exists to prevent circular waits during EP synchronization.

### Target

- Use the same global-ID priority and FIFO tie-break contract.
- Preserve schedule epoch validation and stale-ticket removal.
- Keep ordering fields typed and checked for overflow.
- Add a focused test in which FIFO and global-ID priority would choose
  different batches.

### Exit Gate

- Stage dequeue order matches Python exactly.
- EP synchronization stress cases cannot deadlock due to queue ordering.
- Existing event-order and scheduler-trace parity remains green.

Implemented with a min-priority queue and an overflow-checked insertion
counter. A focused scheduler-hierarchy test covers insertion order that
conflicts with global-ID order.

## 4. Introduce the `MetricsStore` Responsibility

### Previous Difference

Python records runtime metrics through `MetricsStore`. C++ event handlers and
the former `SimulationContext::finalize()` previously appended or
reconstructed normalized output records directly.

### Target

- Add a C++ runtime metrics collector corresponding to Python `MetricsStore`.
- Route request, batch, stage, scheduler, transfer, MoE, and future
  prefix-cache observations through it.
- Keep JSON/CSV serialization in the output-contract layer.
- Do not port pandas, Plotly, or presentation-only Python code into the
  simulator core.
- Keep orchestration and simulation-wide ownership in `Simulator`.

### Exit Gate

- Event handlers do not directly own normalized output bookkeeping.
- Finalization validates state and asks the metrics collector for output.
- Existing output schema and differential comparisons remain unchanged.

Implemented: event handlers record observations through `MetricsStore`;
request record materialization and final output ownership also live there.

MoE synchronization follows the same ownership rule: sync event files only
delegate typed payloads to `BaseClusterScheduler`. Barrier keys, participants,
waiting rooms, idle compaction, and collective continuation are private
cluster-scheduler details implemented in `base_cluster_scheduler.cc`.

## 5. Add KV-Transfer and Communication-Backend Interfaces

### Previous Difference

Python uses `BaseKVCacheTransferPredictor` and `BaseCCBackend` hierarchies.
C++ currently invokes analytical transfer and communication functions
directly.

### Target

- Add explicit KV-transfer predictor and communication-backend interfaces.
- Let the appropriate cluster/runtime owner construct them from config.
- Inject them into schedulers or predictors following the Python ownership
  path.
- Keep analytical implementations first; add other backends only when their
  feature step requires them.
- Avoid embedding backend selection in event orchestration.

### Exit Gate

- Analytical behavior is unchanged.
- Backend selection is isolated behind typed interfaces.
- Adding another backend does not require changing event orchestration.

Implemented with `BaseCCBackend` and `BaseKVCacheTransferPredictor`.
`Cluster` owns the communication backend; `Simulator` owns the cross-cluster
KV-transfer predictor and injects it through the scheduler hierarchy. The
analytical implementations preserve current behavior.

## 6. Remove Unnecessary Global-Scheduler Double Abstraction

### Previous Difference

Production Python instantiates `BaseGlobalScheduler` as the effective global
scheduler. C++ currently has both an abstract `BaseGlobalScheduler` and a
single concrete `GlobalScheduler`.

### Target

Choose one of the following based on the actual supported policy surface:

1. make `GlobalScheduler` the single concrete class; or
2. move common ownership and behavior into `BaseGlobalScheduler`, retaining
   concrete subclasses only when multiple real policies exist.

Do not retain an abstract layer solely for hypothetical future variants.

### Exit Gate

- Global queue and cluster-scheduler ownership have one clear implementation
  home.
- No forwarding-only class remains.
- Global scheduling parity and PDD routing remain unchanged.

Implemented by retaining only the concrete `GlobalScheduler`; the
forwarding-only `BaseGlobalScheduler` interface was removed.

## Intentional C++ Differences

The following should remain unless evidence shows a correctness problem:

- `std::variant` event payloads and typed event dispatch;
- strong IDs with an invalid sentinel;
- `EntityArena` for stable dynamic-entity storage;
- typed MoE waiting-room storage owned by `BaseClusterScheduler`;
- structs and validation functions in place of Python dataclass mechanics;
- predictor-internal analytical helper functions; and
- compile-time factories where the supported implementation set is small.

## Deferred, Not Structural Refactoring

The following Python hierarchies are absent because their feature surfaces
have not yet been ported:

- LOR, random, and sticky cluster schedulers;
- non-vLLM-V1 replica schedulers;
- synthetic request-generator families;
- trained/sklearn execution predictors;
- CPU KV-cache support;
- speculative decoding and Thinking Mode;
- prefix caching;
- AFD/M2N and `EPBatchGroup`; and
- non-analytical communication backends.

When these features are implemented, their Python ownership boundaries should
be the starting point rather than adding another simulation-wide context
facade.
