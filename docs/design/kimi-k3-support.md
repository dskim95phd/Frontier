# Kimi K3 Support Design Decisions

- **Status:** Accepted
- **Date:** 2026-08-10
- **Scope:** C++ simulator (`cpp/`)
- **Model asset:** [`moonshotai__Kimi-K3.json`](../../data/config/models/moonshotai__Kimi-K3.json)

## 1. Purpose

This document is the decision record for Kimi K3 support in the Frontier C++
simulator. It defines the intended behavior of model parsing, analytical
execution-time prediction, mixed precision, KDA snapshot caching, GPU/CPU
capacity management, and sequential PDD transfer.

The document is normative for the K3-specific behavior described below. The
implementation and tests remain the executable source of truth when this
document and code disagree.

## 2. Model representation

Kimi K3 is represented as one hybrid decoder model rather than separate KDA
and MLA submodels.

The checked-in model metadata preserves the official nested `text_config`
shape and records:

- 93 decoder layers;
- 69 KDA (linear-attention) layers;
- 24 full-attention MLA layers;
- 96 attention/KDA heads with head dimension 128;
- one dense prefix layer followed by MoE layers;
- 896 routed experts, top-16 routing, and two shared experts;
- latent routed-expert hidden size 3584;
- 12-layer AttnRes block size;
- Gated MLA and NoPE enabled.

### 2.1 Supported KDA topology

The current KDA implementation deliberately supports the symmetric Q/K/V
topology published for Kimi K3 only:

```text
num_heads == num_k_heads == num_v_heads
head_dim == key_head_dim == value_head_dim
```

The parser retains the separately named fields because that is how generalized
Kimi-Linear metadata may describe them, but a model with unequal values fails
fast during model loading. Automatic GPU weight accounting and the analytical
roofline also defend this contract when called with an in-memory model.

This restriction prevents the K3 projection, short-convolution, gating, output,
and recurrent-state formulas from being silently applied to a topology whose
Q/K/V head grouping semantics have not been defined. Supporting an asymmetric
Kimi-Linear variant requires those mappings and all corresponding memory and
latency formulas to be generalized together.

The official layer lists in the asset are one-based. Model loading normalizes
them to zero-based C++ layer indices. A legacy model with `use_mla=true` and no
explicit hybrid lists remains an all-MLA model, preserving Kimi K2 behavior.

Relevant implementation:

- [`model_config.cc`](../../cpp/frontier/config/model_config.cc)
- [`config.h`](../../cpp/frontier/config/config.h)
- [`config_test.cc`](../../cpp/tests/config/config_test.cc)

## 3. Analytical attention modeling

### 3.1 KDA layers

KDA is modeled as its own attention family. A KDA layer charges analytical
roofline work for:

1. Q, K, and V projections;
2. causal short convolutions;
3. decay and beta projections;
4. recurrent state updates;
5. gated normalization/output work;
6. output projection.

Prefill recurrent work scales with the number of processed tokens. Decode uses
the fixed-size recurrent state and therefore does not grow with past-context
length. This is intentionally different from MLA decode, whose attention work
reads a context-dependent cache.

On a hybrid decode replica, DCP reuses ranks inside the TP domain and applies
only to MLA layers. KDA layers retain the replica-level DCP metadata but ignore
it: their projections and recurrent state remain sharded by the full TP size,
and they issue no DCP collectives. For example, `TP=8, DCP=8` uses eight GPUs,
not 64; KDA runs as TP8 while MLA shards its sequence context over the same
eight ranks.

KDA recurrent state is not represented as per-token scheduler KV. It is
represented as one fixed snapshot per session when prefix caching is enabled.

K3 AttnRes metadata, including the 12-layer block size, is retained to describe
the architecture but contributes zero analytical latency and zero automatic
GPU weight-memory overhead. The operation is treated as negligible or fused.
The previous block-width streaming approximation was removed because it had no
measured kernel basis and could overstate HBM traffic.

### 3.2 Gated MLA

K3 MLA layers enable a full-rank output gate. The analytical path charges:

- the TP-sharded gate projection from the hidden state to local value-head
  channels; and
- the fused sigmoid and elementwise multiplication before the output
  projection.

Legacy MLA models leave this feature disabled unless explicitly configured.

### 3.3 NoPE

NoPE removes only the rotary transform cost. It does **not** remove the
configured shared Q/K channel from:

- QK attention FLOPs;
- persistent cache layout;
- decode-context-parallel traffic; or
- PDD transfer payloads.

This distinction prevents NoPE from incorrectly shrinking the attention state
or communication volume.

### 3.4 MLA phase-specific execution

MLA prefill and decode retain the existing phase-specific modeling:

- Prefill uses unabsorbed MHA, including KV up-projection and temporary
  head-specific workspace traffic.
- Decode uses absorbed MQA, reading the latent cache directly and expanding
  the result on the output side.

The temporary prefill workspace affects roofline time but does not consume
scheduler KV-block capacity.

Relevant implementation:

- [`analytical_roofline_execution_time_predictor.cc`](../../cpp/frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.cc)
- [`analytical_roofline_execution_time_predictor.h`](../../cpp/frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.h)
- [`analytical_model_test.cc`](../../cpp/tests/analytical_model/analytical_model_test.cc)

## 4. Representative-layer scaling

`moe_layer_event_mode="first_layer_scaled"` avoids replaying every repeated
layer event while preserving K2 and K3 attention differences.

The representative values are separated by attention implementation family:

- standard attention;
- MLA;
- KDA.

Each pipeline stage counts how many layers of each family it owns and scales
one representative attention prediction for that family. The MoE critical
path is also represented once and reused for the remaining compatible MoE
layers.

Consequences:

- Kimi K2 retains one MLA representative per stage.
- Kimi K3 retains distinct MLA and KDA totals even when both appear in the
  same pipeline stage.
- Inter-layer synchronization and congestion changes after the
  representative event are intentionally omitted.

## 5. Native mixed-precision policy

K3 analytical configurations install the following defaults unless a new
field or its documented legacy fallback was explicitly set:

| Operator/state | Default |
|---|---:|
| Routed-expert weight | MXFP4 |
| Routed-expert activation | MXFP8 |
| Stable LatentMoE down/up projection weight | BF16 |
| Stable LatentMoE down/up projection activation | BF16 |
| Shared-expert weight | BF16 |
| Shared-expert activation | BF16 |
| Dense MLP weight | BF16 |
| Dense MLP activation | BF16 |
| Router weight storage | BF16 |
| Router input activation storage | BF16 |
| Router compute/logits | FP32 |
| LM head | BF16 |
| MLA KV cache latent component | FP8 |
| KDA snapshot | BF16 |

MXFP storage includes one E8M0 scale byte per 32 values. Analytical HBM
traffic therefore uses 1.03125 bytes/value for MXFP8 and 0.53125 bytes/value
for MXFP4.

`routed_expert_down_proj` and `routed_expert_up_proj` are dense projections
outside the quantized routed-expert bank. Their HBM storage and roofline use
the separate `latent_moe_projection_weight` and
`latent_moe_projection_activation` fields. Models without Stable LatentMoE do
not consume these fields. For older/custom latent-MoE configurations that omit
them, both fields fall back to the routed-expert weight/activation precision,
preserving the previous simulator behavior.

### 5.1 Router storage and compute are separate

`router_weight_storage` controls resident GPU weight memory and router-weight
HBM reads. `moe_router_activation` controls input activation storage.
`router_compute` controls the router GEMM/TopK roofline ceiling and logits
width.

For the native K3 policy, BF16 weights and BF16 input are consumed by an FP32
router computation that produces FP32 logits. BF16-to-FP32 conversion is
assumed fused and free: no separate cast buffer or cast latency is charged.

The legacy `moe_router_weight` field remains a fallback for both historical
storage and compute behavior when the canonical fields are absent. A canonical
`router_weight_storage` override does not suppress K3's FP32 compute default.

### 5.2 Persistent MLA cache layout

MLA cache storage is component-wise:

- the latent component uses `operator_precisions.kv_cache`;
- the decoupled RoPE component remains BF16.

Scale metadata is included for MXFP storage. Quantization/dequantization time
is not modeled.

## 6. KDA snapshot representation

### 6.1 One latest, atomic snapshot

Each session owns at most one latest KDA snapshot. A partial snapshot has no
meaning and is never retained. Snapshot eviction and transfer therefore occur
as one atomic group rather than block by block.

The snapshot charge is converted to KV-equivalent blocks so it shares the same
finite GPU and CPU capacity pools as ordinary KV. The K3 metadata implies, per
KDA layer:

- 1,572,864 recurrent-state elements;
- 110,592 short-convolution state elements;
- 1,683,456 total state elements.

Across 69 KDA layers this is 116,158,464 elements, or 232,316,928 bytes at the
native BF16 snapshot precision, before conversion to the configured
KV-equivalent block charge. Fixed execution has no per-operator precision
declaration and therefore follows this native BF16 snapshot contract.

### 6.2 Free reverse-recovery assumption

The simulator assumes KDA recurrence can be reversed at zero execution cost.
The latest snapshot is therefore sufficient to recover useful state even when
some ordinary per-token KV has been suffix-evicted.

This is a deliberate simplification. No inverse-kernel latency, temporary
buffer, numerical error, or recomputation traffic is modeled.

## 7. GPU admission and snapshot lifecycle

Snapshot capacity is a required admission resource, not an opportunistic cache
allocation after compute.

### 7.1 Empty-replica feasibility

`full_sequence_fits_empty()` is a static feasibility test. For KDA prefix
caching it checks:

```text
full prompt KV blocks + watermark blocks + one snapshot charge
    <= fixed replica capacity
```

The comparison uses fixed `capacity_blocks_`, not current reclaimable
capacity. Current active requests must not make an otherwise valid request
look permanently impossible.

### 7.2 Dynamic admission

The dynamic `can_commit()`/`can_admit()` paths include a snapshot charge only
for a cold session. An existing same-session snapshot is already charged and
must not be charged twice.

Successful cold admission atomically reserves the snapshot group before
compute by creating an unpublished frontier-zero entry. Snapshot entries keep
an explicit `published` bit because frontier zero has two distinct meanings:

- `published=false`: cold capacity ownership only;
- `published=true`: a valid computed snapshot for a prompt shorter than one
  ordinary KV block.

Neither state produces an ordinary block prefix hit, but the published
snapshot-only state must survive normal request release and remain available
for CPU offload or later recovery. The first completed batch publishes into
the already-reserved group, so normal publication cannot fail for capacity
reasons.

If asynchronous restore/admission is rolled back before a reusable frontier is
published, only the unpublished reservation is released.

The scheduler treats a failed publication after successful reservation as an
accounting invariant violation. It must never ignore the return value of
`publish_kda_snapshot()`.

### 7.3 Example

With capacity 4 blocks, snapshot charge 2 blocks, and watermark 0:

- a cold 3-block prompt is rejected because `3 + 2 > 4`;
- a cold 2-block prompt is admitted and reserves all four blocks atomically;
- a later turn for the same session can reuse the existing two-block snapshot
  charge rather than allocating another one.

Relevant implementation:

- [`replica_kv_cache_manager.cc`](../../cpp/frontier/kv_cache/replica_kv_cache_manager.cc)
- [`replica_kv_cache_manager.h`](../../cpp/frontier/kv_cache/replica_kv_cache_manager.h)
- [`vllm_v1_engine_replica_scheduler.cc`](../../cpp/frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.cc)

## 8. Session-scoped eviction policy

Capacity pressure selects inactive sessions in unified LRU order. For each
selected victim session:

1. evict its ordinary KV from the suffix;
2. if more capacity is still required after that session's KV reaches zero,
   evict that session's whole snapshot;
3. only then advance to the next LRU session.

This does **not** mean that every session's ordinary KV is globally drained
before any snapshot can be removed. The KV-before-snapshot rule is scoped to
the selected victim session.

Atomic snapshot removal may reclaim more blocks than the immediate deficit.
The surplus returns to blank capacity and must not cause eviction to spill into
the next session.

Active sessions and transfer-pinned snapshots are not evictable.

## 9. CPU offload and restore

CPU KV offload uses the same session and snapshot semantics as GPU caching:

- ordinary KV and the KDA snapshot share one finite CPU capacity;
- a snapshot is reserved, transferred, and committed atomically;
- only a newer useful snapshot generation/frontier is transferred;
- restore leases pin the snapshot until success or cancellation;
- GPU restore publication is transactional: the scheduler checkpoints the
  previous presence, frontier, and publication state, then restores that
  checkpoint if CPU lease termination or later completion steps fail;
- a prompt shorter than one KV block transfers the snapshot by itself; its
  empty KV range remains a zero-block/zero-token prefix hit;
- an offload whose ordinary and snapshot frontiers are unchanged is a true
  no-op and changes neither skipped nor truncated-offload metrics;
- pressure completes one victim session's ordinary-KV-then-snapshot chain
  before selecting another session;
- discard waits for in-flight pins and reaps the session safely afterward.

The snapshot byte payload is separate from ordinary KV bytes in transfer
accounting. A transfer is valid when it contains ordinary KV, a KDA snapshot,
or both. Empty restore ranges remain invalid unless accompanied by a snapshot
payload. CPU offload is supported for K3 and is covered by dedicated manager,
scheduler, and integration tests.

Relevant implementation:

- [`cpu_kv_cache_manager.cc`](../../cpp/frontier/kv_cache/cpu_kv_cache_manager.cc)
- [`analytical_transfer.cc`](../../cpp/frontier/kv_cache_transfer/analytical_transfer.cc)
- [`k3_cpu_kv_cache_integration_test.cc`](../../cpp/tests/simulator/k3_cpu_kv_cache_integration_test.cc)

## 10. Sequential PDD contract

K3 snapshot transfer follows the release-supported sequential PDD path.

The PREFILL and DECODE sides must agree on KDA support and snapshot
representation:

- analytical/analytical requires the exact same snapshot precision name;
- BF16 and FP16 are considered different despite equal byte width;
- fixed execution implies a BF16 snapshot;
- analytical/fixed is valid only when the analytical side also uses BF16.

The transfer predictor receives the validated common snapshot element size. It
must not silently use only the PREFILL precision.

The snapshot payload is added to the normal KV transfer payload and is moved
atomically. Precision conversion cost is not modeled.

Relevant implementation:

- [`config.cc`](../../cpp/frontier/config/config.cc)
- [`config_parse.cc`](../../cpp/frontier/config/config_parse.cc)
- [`simulator.cc`](../../cpp/frontier/simulator/simulator.cc)

## 11. Compatibility decisions

- Kimi K2 remains an all-MLA model and does not enable Gated MLA or NoPE by
  default.
- Non-KDA models have zero snapshot charge and retain the legacy KV admission
  and eviction behavior.
- Existing precision-family fields remain supported through documented
  fallbacks.
- Existing snapshot replacement is in-place and does not allocate a second
  atomic group.
- KDA-specific behavior is active only when the model has KDA and prefix
  caching owns the relevant cache.

## 12. Deliberate simplifications and non-goals

The current implementation intentionally does not model:

- KDA inverse/recovery execution time or numerical stability;
- more than one historical KDA snapshot per session;
- partial snapshot storage, eviction, or transfer;
- AttnRes latency or weight-memory overhead;
- router cast buffers or BF16-to-FP32 conversion latency;
- quantization/dequantization kernel time;
- precision-conversion cost during PDD transfer;
- KV-block fragmentation inside an atomic snapshot charge;
- per-layer congestion changes hidden by `first_layer_scaled` mode.

These assumptions should be revisited if measured K3 kernels, memory traces,
or a more detailed recurrent-state recovery policy become available.

## 13. Verification requirements

Changes to this design should preserve tests for:

- nested K3 model parsing and KDA/MLA layer normalization;
- symmetric KDA Q/K/V topology acceptance and asymmetric topology rejection;
- KDA projection, short-convolution, and recurrent roofline costs, with
  AttnRes metadata retained and its modeled cost fixed at zero;
- Gated MLA cost and NoPE invariants;
- native mixed-precision defaults and router storage/compute separation;
- distinct KDA and MLA representative-layer scaling;
- cold snapshot admission, existing-snapshot reuse, and rollback cleanup;
- per-session KV-before-snapshot eviction and atomic surplus reclaim;
- CPU snapshot offload/restore;
- exact PDD snapshot precision agreement.

The current release gate is the complete C++ CTest suite in `cpp/build`.
