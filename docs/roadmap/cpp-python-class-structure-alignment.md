# C++ / Python Class-Structure Alignment Checklist

## Status

Implemented on the C++ port branch before Step 4.

- [x] Runtime `Cluster` / `Replica` graph
- [x] Cluster-owned shared predictor and communication backend
- [x] `(batch.global_id, insertion_order)` stage priority
- [x] Runtime `MetricsStore`
- [x] KV-transfer and communication-backend interfaces
- [x] Single concrete `GlobalScheduler`

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

The C++ `Cluster` and `Replica` classes currently exist but are not used by
the simulation runtime. `SimulationContext` constructs replica schedulers
directly from configuration.

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

Implemented: `SimulationContext` owns cluster entities, clusters own replicas,
and cluster/replica schedulers keep lifetime-safe references to those entities.
The duplicate `cluster_parallelism_` map was removed.

## 2. Move Predictor and Backend Ownership to the Cluster

### Previous Difference

Python creates one predictor per cluster and injects it through the scheduler
hierarchy. C++ currently creates a `BatchExecutionModel` per replica scheduler
and clones it for pipeline stages.

### Target

- Construct execution-time predictors at cluster initialization.
- Let the cluster own or share the predictor and communication backend.
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

Implemented: each `Cluster` owns one immutable execution model and one
communication backend. Replica and PP-stage schedulers share the model rather
than cloning it.

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
`SimulationContext::finalize()` currently append or reconstruct normalized
output records directly.

### Target

- Add a C++ runtime metrics collector corresponding to Python `MetricsStore`.
- Route request, batch, stage, scheduler, transfer, MoE, and future
  prefix-cache observations through it.
- Keep JSON/CSV serialization in the output-contract layer.
- Do not port pandas, Plotly, or presentation-only Python code into the
  simulator core.
- Reduce `SimulationContext` to orchestration and simulation-wide ownership.

### Exit Gate

- Event handlers do not directly own normalized output bookkeeping.
- Finalization validates state and asks the metrics collector for output.
- Existing output schema and differential comparisons remain unchanged.

Implemented: event handlers record observations through `MetricsStore`;
request record materialization and final output ownership also live there.

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
- Avoid embedding backend selection inside `SimulationContext`.

### Exit Gate

- Analytical behavior is unchanged.
- Backend selection is isolated behind typed interfaces.
- Adding another backend does not require changing event orchestration.

Implemented with `BaseCCBackend` and `BaseKVCacheTransferPredictor`.
`Cluster` owns the communication backend; the cross-cluster runtime owns the
KV-transfer predictor. The analytical implementations preserve current
behavior.

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
- `BarrierCoordinator` as a composed waiting-room component;
- structs and validation functions in place of Python dataclass mechanics;
- pure analytical helper functions; and
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
be the starting point rather than extending `SimulationContext`.
