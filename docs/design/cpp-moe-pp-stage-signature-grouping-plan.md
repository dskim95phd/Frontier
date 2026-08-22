# C++ MoE PP Stage-Signature Grouping Implementation Plan

## Status

Implemented, 2026-08-10.

Implemented:

- canonical timing and memory signatures with deterministic stage groups;
- exact stage/rank-local weight, KV-block, and KDA-snapshot profiles;
- one logical GPU KV capacity constrained by the limiting physical profile;
- stage-aware GPU occupancy bytes while retaining full-model CPU/PDD payloads;
- the restore-before-PP0 and final-stage-before-offload contracts, covered for
  K3 PP1 and PP4; and
- `stage_group_scaled`, including KDA/MLA family separation and safe detailed
  fallback for layer-dependent routing or a non-contiguous MoE suffix;
- a thread-safe, batch-scoped cross-stage prediction cache keyed by timing
  group and released with the batch entity;
- mandatory positive per-GPU HBM capacity for manual and automatic configs;
- exact physical validation of manually configured logical block counts; and
- an opt-in pipeline-exclusive K3 experiment contract (`PP>1`, `DP=1`,
  `MoE EP=1`, `MoE TP=attention TP`) without changing the general hybrid
  PP+DP+EP surface.

The following observability work is now implemented:

- `pipeline_memory_diagnostics` in the simulation JSON contains per-cluster
  capacity/block/snapshot limiting fields and per-stage layer ranges, KDA/MLA/
  MoE counts, rank-local F/B/S vectors, and timing/memory group IDs and
  multiplicities;
- `frontier_pp_stage_group_benchmark` is a non-gating CMake target that runs
  PP1/PP4/PP24 and prints wall time, DES event count, stage/predictor calls,
  unique timing groups, and peak event queue size.

This plan extends the existing C++ pipeline-parallel execution path. It does
not replace the current stage state machine or introduce a global PP barrier.
The main change is to derive an exact static profile for every pipeline stage,
group stages only when their canonical signatures are equivalent, and reuse
timing work within those groups.

The same exact stage profiles become the source of truth for GPU KV capacity,
KDA snapshot admission, CPU offload payloads, diagnostics, and tests.

## Benchmark Note (non-gating)

`frontier_pp_stage_group_benchmark` was run once in a Debug build on the
Windows development host with the analytical Phi-tiny MoE fixture, balanced
routing, and the same workload for PP1/PP4/PP24.  These values are diagnostic
only and are not release thresholds:

| PP | wall time (ms) | DES events | stage predictions | unique groups | cache hits / misses | peak queue |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.914 | 390 | 33 | 1 | 2 / 31 | 4 |
| 4 | 47.172 | 1,488 | 168 | 3 | 111 / 57 | 17 |
| 24 | 241.273 | 8,175 | 1,008 | 4 | 932 / 76 | 17 |

## Pipeline-Exclusive K3 Validation Matrix

The pipeline-exclusive path is covered by a bounded pairwise matrix rather
than only one uniform PP layout:

| Area | Covered points |
|---|---|
| topology | `(TP,PP,DCP) = (1,2,1), (2,3,1), (4,4,1), (8,8,1), (8,24,8)` |
| execution model | fixed and analytical `stage_group_scaled` |
| workload shape | prefill-heavy, decode-heavy, concurrent batches, staggered arrivals |
| stage causality | analytical PP3, PP8, and PP24 timelines |
| PDD/offload | K3 PP2, PP4, and PP8; concurrent/staggered multi-turn sessions |

The matrix asserts that every batch visits every physical stage, intervals on
one stage never overlap, and pipeline-exclusive runs emit no EP/MoE
synchronization events. It also verifies positive stage-aggregate KV footprints and
stage-local HBM/snapshot constraints. With DCP, an individual rank may own
zero tokens for a small logical block, so non-zero KV is intentionally checked
at the stage aggregate rather than incorrectly required on every rank.

For CPU offload, the matrix preserves the asymmetric contract: GPU admission
uses stage-local profiles, while a CPU transfer contains one full-model KDA
snapshot. Publication starts only after the final PP stage and a restored
batch cannot enter PP0 until the atomic H2D transfer completes. Invalid
pipeline-exclusive DP>1 and MoE EP>1 configurations are rejected at parse
time.

## Decisions

1. PP stages are not classified by position such as `start`, `middle`, and
   `end`.
2. Every stage receives one or more canonical signatures derived from its
   actual layer range and resident state.
3. Stages are grouped only when the signature relevant to the operation is
   equal.
4. Timing and memory use separate signatures because timing equivalence does
   not imply memory equivalence, and vice versa.
5. Static memory accounting is calculated exactly for every stage. Grouping is
   used only to deduplicate equivalent profiles and summarize the limiting
   group.
6. The existing stage-level PP event flow remains the correctness baseline:
   a batch moves to stage `s + 1` when stage `s` finishes.
7. General dense and attention layers do not gain DES events. Layer-level MoE
   critical-path work is aggregated inside the predictor; PP boundaries remain
   the externally visible events.
8. GPU KV admission remains one logical cache-manager decision across all PP
   stages, constrained by the physical stage/rank with the least capacity.
9. GPU snapshot admission uses the limiting stage profile. CPU capacity and
   transfer payloads remain full-model aggregates across all stage shards.
10. The original Kimi K3 metadata remains authoritative. The implementation
    must support the real 69 KDA / 24 MLA distribution without changing it to
    manufacture uniform PP stages. A synthetic normalized model, if desired,
    must be a separate explicit model configuration.

## Existing Behavior to Preserve

The following mechanisms already exist and should be reused:

- `config::pipeline_stage_layer_range()` assigns each PP stage one contiguous
  `[begin, end)` layer range, including uneven partitions.
- `BaseReplicaScheduler` creates one logical `ReplicaKVCacheManager` per
  `(replica_id, dp_id)` and multiple stage schedulers beneath it.
- `BatchStageArrivalEvent` and `BatchStageEndEvent` move a batch through PP
  stages.
- the analytical and fixed predictors already receive `StageId` and inspect
  the corresponding local layer range;
- the analytical predictor adds point-to-point PP communication on non-final
  stages and LM-head work on the final stage; and
- CPU KDA snapshots are published and restored atomically by the CPU cache
  manager.

The plan corrects the following PP-incompatible assumptions:

- automatic GPU KV bytes per block currently use all model KV-bearing layers;
- GPU KDA snapshot charge currently uses the full-model KDA layer count;
- GPU KV occupancy metrics repeat the full-model rank-local footprint;
- `first_layer_scaled` reuses the first detailed MoE lane prediction across
  later model layers even when layer-dependent routing can differ; and
- the predictor exposes a first physical layer diagnostic that is not a
  representative signature for an arbitrary PP stage.

## Terminology

Let:

- `P` be the pipeline-parallel size;
- `s` be a physical PP stage in `[0, P)`;
- `R_s` be the stage-local contiguous model-layer range;
- `r` be a rank-local memory shard within one logical `(replica, DP)` cache
  target when relevant; replicated DP targets are resolved independently;
- `F_s,r` be HBM available for KV and KDA state after weights and reserves;
- `B_s,r` be bytes consumed on that physical rank by one logical KV block;
- `S_s,r` be bytes consumed on that physical rank by one session's KDA
  snapshot shard; and
- `N` be the shared logical KV-block capacity exposed by
  `ReplicaKVCacheManager`.

A **stage group** is a set of stages with an equal canonical signature for a
specific purpose. There is no single universal group identity.

## Signature Model

### 1. Common static layer descriptor

Introduce a canonical static descriptor for every layer in a stage:

```cpp
enum class AttentionFamily {
    kStandard,
    kMla,
    kKda,
};

struct LayerStaticSignature {
    AttentionFamily attention_family;
    bool is_moe;
    bool has_dense_mlp;
};
```

Model-wide operator dimensions and precisions do not need to be repeated per
layer when they are already part of the owning model/execution configuration
identity.

### 2. Timing signature

Timing grouping must preserve ordered execution semantics, not only counts.
Two stages with the same number of KDA, MLA, and MoE layers can still differ if
those layers appear in a different order around MoE barriers.

```cpp
struct StageTimingSignature {
    std::vector<LayerStaticSignature> ordered_layers;
    bool owns_input_embedding;
    bool owns_final_norm;
    bool owns_lm_head;
    bool emits_pp_send;
};
```

The actual key should use a stable value object or collision-safe canonical
serialization. A hash may accelerate lookup, but equality must compare the
canonical contents rather than trusting the hash alone.

The signature excludes dynamic batch properties such as token count, context
length, prefill/decode phase, and realized routing allocation. Those belong to
the timing cache key or are recomputed for the current batch.

### 3. Memory profile and memory signature

Memory must first be derived exactly for each stage and physical shard:

```cpp
struct PipelineStageMemoryProfile {
    StageId stage_id;
    PipelineStageLayerRange layers;
    std::uint64_t resident_weight_bytes;
    std::uint64_t reserved_bytes;
    std::uint64_t free_bytes;
    std::vector<std::uint64_t> kv_bytes_per_block_by_rank;
    std::vector<std::uint64_t> kda_snapshot_bytes_by_rank;
};
```

Equivalent numeric profiles may be deduplicated with a memory signature:

```cpp
struct StageMemorySignature {
    std::uint64_t resident_weight_bytes;
    std::uint64_t reserved_bytes;
    std::vector<std::uint64_t> kv_bytes_per_block_by_rank;
    std::vector<std::uint64_t> kda_snapshot_bytes_by_rank;
};
```

Boundary weights such as embeddings, final norm, and LM head are naturally
represented by `resident_weight_bytes`; no positional label is required.

### 4. Deterministic group catalogue

Build a catalogue during runtime-config resolution or predictor construction:

```cpp
struct PipelineStageGroupCatalogue {
    std::vector<StageTimingSignature> timing_groups;
    std::vector<StageMemorySignature> memory_groups;
    std::vector<std::uint32_t> stage_to_timing_group;
    std::vector<std::uint32_t> stage_to_memory_group;
    std::vector<std::uint64_t> timing_group_multiplicity;
    std::vector<std::uint64_t> memory_group_multiplicity;
};
```

Group IDs must be deterministic for tests and diagnostics. Assign IDs in first
stage occurrence order after canonical equality lookup.

## GPU KV Capacity

### Exact physical constraint

One logical KV block exists across the pipeline, but stage `s` stores only the
state for `R_s`. Therefore the ordinary KV capacity is:

```text
N = min over (s, r) of floor(F_s,r / B_s,r), for B_s,r > 0
```

Stages with `B_s,r == 0` do not constrain ordinary KV-block count. They can
still constrain KDA snapshot capacity and must remain in the profile set.

The implementation must not combine `max(weight)` and `max(block bytes)` from
different stages and present that combination as an exact physical stage.
Compute `floor(F_s,r / B_s,r)` jointly for every real profile, then select the
minimum.

### Logical manager contract

`ReplicaKVCacheManager` continues to allocate integer logical blocks. Its
configured block capacity becomes `N`; it does not need a separate cache
manager per PP stage.

The public/runtime meaning of a block count is logical. Physical byte metrics
must use the stage profiles rather than multiplying logical blocks by the
former full-model `kv_cache_bytes_per_block` scalar.

### Occupancy metrics

For `K` allocated logical blocks:

```text
stage_physical_kv_bytes[s, r] = K * B_s,r
aggregate_pipeline_kv_bytes  = sum over (s, r) stage_physical_kv_bytes[s, r]
max_rank_kv_bytes            = max over (s, r) stage_physical_kv_bytes[s, r]
```

Expose both the aggregate and maximum-rank views where existing metrics need
different interpretations. Do not report one as the other.

## GPU KDA Snapshot Admission

### Physical invariant

For `K` ordinary logical blocks and `M` resident snapshots, every physical
stage/rank must satisfy:

```text
K * B_s,r + M * S_s,r <= F_s,r
```

The initial implementation may keep the cache manager's existing scalar
`kda_snapshot_blocks_per_session`, but the scalar charge must be derived from
all real profiles after `N` is known.

A safe normalized logical-block charge that also handles a KDA-only stage with
`B_s,r == 0` is:

```text
q_s,r = ceil(N * S_s,r / F_s,r)
q     = max over (s, r) q_s,r
```

The manager then enforces:

```text
ordinary_logical_blocks + q * resident_snapshots <= N
```

Use checked wide arithmetic for `N * S_s,r`. If `S_s,r > F_s,r`, configuration
must fail because one snapshot cannot fit on that stage/rank.

This normalized charge can be conservative because one scalar represents a
multi-dimensional physical constraint. Exact vector admission is a possible
future refinement, not required for the first implementation.

### Snapshot diagnostics

Record independently:

- the stage/rank limiting ordinary KV capacity;
- the stage/rank producing the largest normalized snapshot charge;
- `B_s,r`, `S_s,r`, and `F_s,r` for every stage group; and
- the final logical values `N` and `q`.

The KV limiting group and snapshot limiting group are allowed to differ.

## CPU Offload and Restore

### Capacity and transfer payload

The CPU target remains a full-model aggregate. It must not use only the worst
GPU stage shard.

For one snapshot, the existing full-model analytical helper remains the
canonical CPU payload calculation:

```text
cpu_snapshot_payload_bytes = full-model KDA snapshot bytes
```

As a validation identity, the same result may be reconstructed by summing the
non-replicated PP/TP shards that constitute one logical target:

```text
sum over memory groups g of multiplicity[g] * snapshot_bytes[g]
```

The multiplicity in this identity must describe only those non-replicated
shards; it must not multiply the payload by DP or another replicated cache
target. The result must equal the full-model analytical snapshot payload for
the same precision and sharding contract.

Ordinary CPU KV blocks retain the existing full-model target-physical layout.
GPU stage-local accounting and CPU aggregate accounting intentionally use
different byte views.

### Restore barrier

The first implementation uses a correctness-first global restore barrier:

1. atomically reserve the logical GPU KV blocks and KDA snapshot capacity;
2. pin the CPU restore lease;
3. transfer the full aggregate payload representing all PP shards;
4. publish all GPU snapshot shards atomically;
5. release the CPU lease; and
6. only then admit the batch to PP stage 0.

No stage-ready transfer/compute overlap is modeled initially.

### Offload publication

Offload starts only after the relevant batch/session frontier has completed
the final PP stage. The CPU cache publishes the logical snapshot only after the
full aggregate payload completes. A partially transferred set of PP shards is
never visible or independently evictable.

## Timing Approximation

### Two levels of reuse

The new approximation has two distinct levels:

1. **within a stage signature:** reuse representative operator-family costs
   across compatible repeated local layers;
2. **across stages:** reuse the same stage timing template only for stages with
   an equal `StageTimingSignature` while the same immutable batch traverses PP.

There is no rule that stage 0 represents every later stage.

### Batch-scoped timing key

All dynamic predictor inputs are fixed in the batch's request snapshots, so
cross-stage reuse needs only the batch identity and static timing group:

```cpp
struct StageTimingCacheKey {
    BatchId batch_id;
    std::uint32_t timing_group_id;
};
```

Different batches do not share templates even when their request shapes happen
to match. The predictor retains one small timing-group map per in-flight batch,
and `Simulator::release_batch()` removes that map together with the batch
entity.

### Within-group analytical time

For compatible layers, a stage template may use:

```text
stage time =
    sum(attention-family representative time * compatible layer count)
  + sum(MoE critical-path representative time * compatible layer count)
  + dense prefix/suffix work
  + TP/DP/EP communication
  + PP boundary communication
  + final norm / LM-head work
```

KDA, MLA, and standard attention representatives remain distinct. Ordered
layer signatures ensure prefix, suffix, and MoE-barrier placement are not
silently changed by grouping.

### Routing compatibility

`route_tokens(..., model_layer)` makes routing potentially layer dependent.
Therefore:

- balanced or otherwise explicitly layer-invariant routing may reuse one MoE
  lane critical path across compatible layers and stages;
- random, skewed, Zipf, or any layer-dependent routing must recompute routing
  allocation and the EP critical lane for every logical layer, or fall back to
  detailed prediction; and
- caching static roofline/operator costs is still allowed when routing is
  recomputed.

The mode must never silently reuse the first model layer's realized routing
allocation for a non-equivalent layer.

### Configuration transition

Add an explicit mode rather than silently changing legacy experiment meaning:

```text
moe_layer_event_mode = detailed
moe_layer_event_mode = first_layer_scaled   # legacy behavior
moe_layer_event_mode = stage_group_scaled   # new behavior
```

`stage_group_scaled` is the PP-compatible approximation introduced by this
plan. Existing `first_layer_scaled` tests remain compatibility gates until a
separate deprecation decision is made.

For PP1, `stage_group_scaled` should match the compatible representative-family
behavior of `first_layer_scaled` within numerical tolerance.

## PP Event Semantics

Stage grouping changes prediction reuse, not pipeline causality.

For batch `b` and stage `s`, the existing event flow continues to implement:

```text
start[b, s] = max(finish[b, s - 1], stage_available[s])
finish[b, s] = start[b, s] + predicted_stage_time[b, s]
```

There is no all-stage barrier after every local layer. MoE synchronization is
local to the participating physical domain for the same aligned layer.

`BatchStageArrivalEvent` and `BatchStageEndEvent` preserve this causal flow for
every visited PP stage.

## Proposed Implementation Areas

### Configuration and static profiles

Primary files:

- `cpp/frontier/config/config.h`
- `cpp/frontier/config/config.cc`
- `cpp/frontier/config/config_parse.cc`

Tasks:

1. Add stage layer/timing signature builders.
2. Refactor per-GPU weight calculation to retain every stage's bytes instead of
   returning only the maximum.
3. Derive stage/rank-local KV block bytes from `R_s`.
4. Derive stage/rank-local KDA snapshot bytes from KDA layers in `R_s`.
5. Build deterministic timing and memory group catalogues.
6. Resolve logical block capacity `N` from exact stage profiles.
7. Resolve normalized snapshot charge `q` after `N`.
8. Validate overflows, zero-byte edge cases, and one-snapshot fit.
9. Parse and validate `stage_group_scaled`.

### KV and KDA byte helpers

Primary files:

- `cpp/frontier/kv_cache_transfer/analytical_transfer.h`
- `cpp/frontier/kv_cache_transfer/analytical_transfer.cc`

Tasks:

1. Add helpers accepting a `PipelineStageLayerRange` or explicit layer counts.
2. Preserve existing full-model helpers for CPU/PDD aggregate payloads.
3. Name APIs so `stage_rank_local` and `target_physical_full_model` cannot be
   confused at call sites.
4. Test KDA-only stages where ordinary per-token KV bytes are zero but snapshot
   bytes are nonzero.

### Execution-time predictor

Primary files:

- `cpp/frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h`
- `cpp/frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.cc`
- `cpp/frontier/execution_time_predictor/fixed_execution_time_predictor.*`

Tasks:

1. Map `StageId` to its timing group.
2. Add bounded or batch-local reuse keyed by timing group and actual predictor
   features.
3. Retain separate standard/MLA/KDA representatives.
4. Recompute or reject layer-dependent routing reuse.
5. Replace ambiguous first-physical-layer diagnostics with explicit group and
   stage totals while retaining compatibility fields where required.
6. Keep PP communication and final-stage work in the signature/template so
   stages with different boundaries cannot be grouped accidentally.

### Scheduler and CPU offload

Primary files:

- `cpp/frontier/scheduler/replica_scheduler/base_replica_scheduler.*`
- `cpp/frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.cc`
- `cpp/frontier/kv_cache/replica_kv_cache_manager.*`
- `cpp/frontier/kv_cache/cpu_kv_cache_manager.*`

Tasks:

1. Keep one logical GPU manager and install the derived `N` and `q`.
2. Preserve atomic reservation/publication semantics across all PP shards.
3. Require full restore completion before PP0 admission.
4. Require final PP completion and full transfer completion before CPU
   publication.
5. Avoid multiplying or dividing the CPU full-model payload by PP a second
   time.

### Metrics

Primary files:

- `cpp/frontier/simulator/simulator.cc`
- `cpp/frontier/metrics/metrics_store.*`

Tasks:

1. Stop reconstructing GPU occupancy with full-model rank-local block bytes.
2. Report logical allocated blocks separately from physical stage bytes.
3. Emit group ID, multiplicity, layer range/counts, `F`, `B`, `S`, capacity,
   and limiting stage/rank diagnostics.
4. Preserve full-model CPU occupancy and transfer metrics.

## Test Plan

### Signature and partition unit tests

- PP1 produces one stage and one timing/memory group when boundary state is
  colocated.
- equal ordered layer signatures group deterministically;
- equal counts with different ordered KDA/MLA/MoE sequences do not timing-group;
- equal timing signatures with different boundary weights do not memory-group;
- uneven partitions reconstruct every model layer exactly once; and
- original K3 retains 69 KDA and 24 MLA layers across the union of stage
  profiles.

### GPU capacity tests

- dense and MLA models use stage-local KV-bearing layer counts;
- logical capacity equals `min floor(F_s,r / B_s,r)`;
- the limiting stage is selected from a real joint profile rather than a
  synthetic `max(weight) + max(B)` combination;
- K3 PP2, PP4, and PP24 produce deterministic stage profiles;
- a KDA-only stage with `B == 0` does not divide by zero;
- a KDA-only stage can still limit snapshot admission; and
- occupancy metrics reconstruct aggregate and maximum-rank bytes correctly.

### Snapshot and CPU tests

- `q` satisfies the physical HBM inequality for every stage/rank at the
  logical admission boundary;
- the ordinary-KV limiting group and snapshot limiting group may differ;
- CPU snapshot bytes equal the sum of all stage shards and the existing
  full-model analytical helper;
- PP does not multiply the full-model CPU payload twice;
- restore cannot admit PP0 before the atomic payload completes; and
- offload cannot publish a partial PP snapshot.

### Timing tests

- `stage_group_scaled` and `detailed` agree for a synthetic uniform,
  layer-invariant-routing model;
- stages with equal signatures in the same batch reuse one timing template;
- different batches never share timing templates, even for equal shapes;
- stages with different boundary state or ordered layer sequence do not reuse;
- K3 PP4 and PP24 retain separate KDA/MLA contributions;
- random/skewed/Zipf routing recomputes per-layer routing or explicitly falls
  back to detailed mode; and
- the existing PP2 timeline remains causal: a downstream stage starts at the
  maximum of upstream completion and downstream availability.

### Event-count and performance benchmark

Add a non-gating benchmark for PP1, PP4, and PP24 at fixed completed-request
count. Record:

- DES events processed;
- predictor calls;
- unique timing-group predictions;
- simulator wall time; and
- peak event-queue size.

## Implementation Phases

### Phase 1: Static stage profiles and groups

- implement canonical signatures and deterministic group catalogues;
- retain exact per-stage weight, KV, and snapshot profiles; and
- add partition/signature diagnostics and unit tests.

Acceptance gate: every layer belongs to exactly one stage; original K3 counts
reconstruct exactly; grouping is deterministic.

### Phase 2: Correct GPU capacity and metrics

- derive `N` from stage/rank-local KV footprints;
- derive `q` from the normalized snapshot pressure;
- install both values in the existing logical cache manager; and
- replace PP-unaware GPU occupancy metrics.

Acceptance gate: every admitted logical state satisfies the physical profile
inequality for all tested stages/ranks.

### Phase 3: Atomic PP-aware CPU offload

- preserve full-model CPU payloads;
- enforce the restore-before-PP0 barrier;
- enforce final-stage-before-offload publication; and
- add PP2/PP4 K3 CPU integration tests.

Acceptance gate: no partial snapshot becomes schedulable or evictable.

### Phase 4: Stage-group timing approximation

- add `stage_group_scaled`;
- introduce group/batch timing reuse;
- guard layer-dependent routing; and
- add detailed-versus-grouped timing tests.

Acceptance gate: compatible models match detailed mode within tolerance, and
incompatible routing never silently reuses a representative allocation.

### Phase 5: Performance decision

- benchmark PP1/PP4/PP24 event and predictor overhead; and
- optimize signature/template caching if predictor work dominates.

## Completion Criteria

The implementation is complete when:

1. PP execution continues to use causal stage-to-stage events without a global
   layer barrier.
2. Every stage maps to deterministic timing and memory groups based on actual
   equivalence rather than position.
3. GPU logical KV capacity and KDA snapshot admission are safe for every
   physical stage/rank profile.
4. CPU snapshot capacity and transfer bytes remain full-model aggregates.
5. PP0 cannot begin a restore-dependent batch before the atomic restore
   completes.
6. `stage_group_scaled` preserves KDA/MLA family costs and rejects or recomputes
   layer-dependent routing.
7. Original K3 metadata works for uneven PP partitions without artificial KDA
   count changes.
8. Metrics identify the actual limiting stage/group for ordinary KV and KDA
   snapshot pressure.
9. PP1, PP4, and PP24 correctness tests pass, and simulator wall-time results
   are recorded.
