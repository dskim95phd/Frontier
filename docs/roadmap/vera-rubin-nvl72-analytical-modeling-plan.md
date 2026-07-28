# NVIDIA Vera Rubin NVL72 Analytical Modeling Plan

## Status

- **State:** Initial MVP implemented; calibration/backtest remains
- **Plan date:** 2026-07-28
- **Target architectures:** `co-location` and sequential `pd-disaggregation`
- **Execution-time source:** Analytical roofline model
- **Scale-up network:** Existing ASTRA-Sim-inspired analytical backend
- **CPU offload scope:** Existing prefill-side CPU KV-cache tier

## Summary

This plan adds an initial NVIDIA Vera Rubin NVL72 model to Frontier without
requiring Rubin profiling data. The MVP intentionally favors a small,
auditable model over a detailed reconstruction of hardware that is not yet
available for direct profiling.

The model has three independent parts:

1. A per-operator analytical execution-time predictor based on FLOPs, HBM
   traffic, hardware ceilings, shape-dependent efficiency, launch overhead,
   and incomplete compute/memory overlap.
2. A single logical switch connecting all 72 Rubin GPUs through the existing
   ASTRA-Sim-inspired analytical communication backend.
3. A static split of each Vera CPU's DRAM capacity and bandwidth between its
   two attached Rubin GPUs.

The MVP is intended for relative comparisons such as:

- batch-size and context-length sensitivity;
- prefill versus decode behavior;
- tensor, pipeline, data, and expert parallelism choices;
- CPU KV-cache offload tradeoffs; and
- communication sensitivity inside one NVL72 scale-up domain.

It is not intended to claim exact Rubin kernel latency before hardware and
software measurements are available.

## Public Hardware Inputs

The initial hardware profile uses NVIDIA's published preliminary values.
NVIDIA marks these specifications as preliminary and subject to change.

| Component | MVP input |
| --- | ---: |
| Rubin GPUs per NVL72 | 72 |
| Vera CPUs per NVL72 | 36 |
| Rubin GPUs per Vera CPU | 2 |
| HBM4 capacity per GPU | 288 GB |
| HBM4 bandwidth per GPU | 22 TB/s |
| FP16/BF16 dense peak per GPU | 4 PFLOPS |
| FP8/FP6 dense peak per GPU | 17.5 PFLOPS |
| NVFP4 inference peak per GPU | 50 PFLOPS |
| LPDDR5X capacity per Vera CPU | 1.5 TB |
| LPDDR5X bandwidth per Vera CPU | 1.2 TB/s |
| NVLink-C2C bandwidth per superchip | 1.8 TB/s bidirectional |
| NVLink 6 bandwidth per GPU | 3.6 TB/s bidirectional |
| NVLink 6 aggregate bandwidth per rack | 260 TB/s bidirectional |

Primary sources:

- [NVIDIA Vera Rubin NVL72 specifications](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)
- [Inside the NVIDIA Vera Rubin Platform](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)
- [NVIDIA NVLink: The Scale-Up Network for AI Factories](https://developer.nvidia.com/blog/nvidia-nvlink-the-scale-up-network-for-ai-factories/)

The 50-PFLOPS NVFP4 inference number is not treated as a generic dense-compute
ceiling. The predictor must distinguish the published sparse/Transformer
Engine ceiling from workloads that do not satisfy its assumptions.

## MVP Decisions

### 1. CPU DRAM Is Statically Partitioned Per GPU

One Vera CPU is paired with two Rubin GPUs. The MVP assigns half of the Vera
CPU memory and half of its memory bandwidth to each GPU:

```text
CPU capacity per GPU = 1.5 TB / 2 = 750 GB
CPU bandwidth per GPU = 1.2 TB/s / 2 = 600 GB/s
```

Frontier's CPU KV-cache transfer configuration uses gigabits per second:

```text
600 GB/s * 8 = 4,800 Gbps
```

The default physical slice for one Rubin GPU is therefore:

```text
capacity_bytes       = 750_000_000_000
read_bandwidth_gbps  = 4_800
write_bandwidth_gbps = 4_800
```

The existing CPU KV-cache target is one `(replica_id, dp_id)` and accounts for
the physical KV shards of all attention-TP workers in that target. If a target
contains `N` Rubin GPUs, its aggregate settings must be derived as:

```text
target_capacity_bytes = 750_000_000_000 * N
target_read_bw_gbps   = 4_800 * N
target_write_bw_gbps  = 4_800 * N
```

The published NVLink-C2C value is bidirectional. Interpreting it as roughly
900 GB/s per direction leaves the statically assigned 600 GB/s LPDDR slice as
the MVP bottleneck. The first implementation therefore does not need a
separate C2C limiter.

The following effects are deferred:

- dynamic bandwidth sharing between the two GPUs;
- LPDDR read/write contention;
- NUMA or memory-controller effects;
- C2C protocol and coherency overhead;
- contention between CPU offload and GPU HBM traffic; and
- capacity borrowing between the two static GPU slices.

### 2. NVL72 Is One Logical 72-GPU Switch Domain

The MVP uses the existing `astra_sim_analytical` backend. All 72 GPUs are
placed in one logical server and connected through one `Switch` topology:

```text
cluster_servers             = 1
cluster_gpus_per_server     = 72
intra_server_topology       = Switch
intra_server_bandwidth_gbps = 14_400
intra_server_latency_us     = 1.0
```

The bandwidth conversion is:

```text
3.6 TB/s bidirectional
-> 1.8 TB/s in one direction
-> 1,800 GB/s
-> 14,400 Gbps
```

`intra_server_latency_us=1.0` is an explicit initial assumption, not a
published Rubin latency. In the current ASTRA analytical implementation a
`Switch` path has two link hops, so the fixed path contribution becomes
approximately 2 microseconds before serialization and collective rounds.

The physical compute tray contains four GPUs, but the MVP must not use four as
`num_devices_per_node`. Doing so would make the backend classify communication
between trays as `inter_server` traffic even though the NVLink 6 fabric is a
uniform rack-scale domain.

The implementation should add a logical node/network SKU such as:

```text
vera_rubin_nvl72_domain
num_devices_per_node = 72
```

This SKU describes the scale-up communication domain, not the mechanical
compute-tray boundary.

The MVP continues to use the backend's existing ring-style collective
formulas. It does not model:

- the 36 individual NVLink 6 switch chips;
- the nine physical switch trays;
- SHARP in-network reductions;
- simultaneous-collective contention;
- NCCL channel and algorithm selection;
- link or switch failures;
- compute/communication overlap; or
- scale-out traffic between NVL72 racks.

### 3. Layer Runtime Uses an Analytical Roofline Predictor

For each physical operator, the predictor computes:

```text
T_compute = FLOPs / (precision_peak * compute_efficiency)
T_memory  = HBM_bytes / (HBM_bandwidth * memory_efficiency)

T_kernel = launch_latency
         + max(T_compute, T_memory)
         + overlap_penalty * min(T_compute, T_memory)
```

Where:

- `compute_efficiency` is in `(0, 1]`;
- `memory_efficiency` is in `(0, 1]`;
- `overlap_penalty` is in `[0, 1]`;
- `overlap_penalty=0` represents ideal compute/memory overlap; and
- `overlap_penalty=1` represents fully additive compute and memory service.

This formulation preserves the standard roofline lower bound for large,
well-pipelined kernels while allowing small or poorly occupied kernels to pay
part of both costs.

The initial Rubin ridge points are approximately:

```text
BF16:  4.0 PFLOPS / 22 TB/s = 182 FLOP/byte
FP8:  17.5 PFLOPS / 22 TB/s = 795 FLOP/byte
NVFP4: 50 PFLOPS / 22 TB/s = 2,273 FLOP/byte
```

The NVFP4 value is only a ceiling for a workload that satisfies the published
NVFP4 inference assumptions.

## Operator Cost Model

### Dense Linear Projections

For a local matrix multiplication with shapes `M x K` and `K x N`:

```text
FLOPs = 2 * M * N * K

HBM_bytes =
    input_elements  * input_bytes
  + weight_elements * weight_bytes
  + output_elements * output_bytes
  + quantization_metadata_bytes
```

The predictor must use the local tensor shapes after TP sharding. It must also
distinguish gated and non-gated MLPs.

For decode with a small number of tokens, weight reads generally dominate.
Weights are assumed to come from HBM unless an operator-specific cache rule is
added later.

### Dense Prefill Attention

Prefill attention must be computed from each request's query length and live
KV context. It must not materialize an `S x S` score tensor in the HBM byte
count because the supported attention path is IO-aware and tiled.

The MVP calculates:

- QK and probability-times-V FLOPs;
- Q, K, V, output, and KV-cache HBM traffic;
- request-specific existing context;
- GQA/MQA head sharing;
- attention-TP local head counts; and
- KV-cache precision.

Mixed request lengths are summed request by request rather than represented by
one average sequence length.

### Dense Decode Attention

Decode attention is modeled primarily from KV reads:

```text
KV read bytes =
    sum_over_requests(
        live_context_tokens
        * local_kv_heads
        * head_dim
        * K_and_V_factor
        * kv_bytes_per_element
    )
```

The model also includes Q/output traffic and attention FLOPs. Long-context
decode is expected to be HBM-bound for most practical batches.

### Memory-Dominated Operators

The following operators initially use streaming-byte models:

- input and post-attention normalization;
- RoPE;
- residual additions;
- activation functions;
- KV-cache writes; and
- token reshapes/shuffling where applicable.

Fusion must be reflected in the operator graph so that fused add+norm or fused
activation paths do not double-count launches and memory traffic.

### MoE

MoE grouped GEMM uses the existing `per_expert_tokens` information when
available. Runtime is based on the most heavily loaded participating GPU, not
only the global average token count.

The weight-byte calculation includes only experts activated on the local GPU.
The first model does not assume cross-request expert-weight cache residency.

MoE routing, TopK, and local shuffling are modeled as separate small
projection or memory operators. Expert-parallel communication continues to
use the configured communication backend.

## Efficiency Model

The MVP avoids one global efficiency constant. It defines a small set of
configurable operator classes:

```text
large_gemm
small_gemm
prefill_attention
decode_attention
streaming_memory
moe_grouped_gemm
routing_topk
```

Each class has:

- compute efficiency;
- memory efficiency;
- launch latency;
- overlap penalty; and
- optional shape-saturation thresholds.

The initial values should be fitted or sanity-checked against the repository's
existing H100/H800 kernel-only profiling CSVs. This does not make the Rubin
model profile-based: the target hardware ceilings remain Rubin specifications,
while existing profiles provide architecture-independent priors for kernel
occupancy, launch cost, and small-shape efficiency loss.

All coefficients must remain visible in configuration and exported metadata.
No coefficient should be hidden inside a trained estimator.

## Prediction Diagnostics

Each analytical operator prediction should retain the following diagnostic
fields:

```text
operator_name
phase
precision
local_shape
flops
hbm_bytes
compute_efficiency
memory_efficiency
compute_time_ms
memory_time_ms
launch_time_ms
overlap_penalty
predicted_time_ms
bottleneck
```

`bottleneck` should use at least:

```text
COMPUTE
HBM
LAUNCH
NETWORK
CPU_DRAM
```

These values should be available in debug traces or a dedicated analytical
prediction CSV. They are necessary to audit future-hardware predictions.

## Configuration Model

### Device SKU

Add a Rubin device SKU with at least:

```text
device = rubin
total_memory_gb = 288
hbm_bandwidth_tbps = 22
fp16_bf16_dense_tflops = 4_000
fp8_fp6_dense_tflops = 17_500
nvfp4_inference_tflops = 50_000
```

The existing `fp16_tflops` field may remain for compatibility, but the
analytical predictor must select a precision-specific peak.

### Logical Node SKU

Add:

```text
network_device = vera_rubin_nvl72_domain
num_devices_per_node = 72
```

This is required so ASTRA analytical topology materialization produces one
72-GPU switch domain.

### Predictor Type

Add an execution-time predictor type and configuration such as:

```text
execution_time_predictor_config_type = analytical_roofline
```

The predictor should derive directly from `BaseExecutionTimePredictor` rather
than inheriting the sklearn implementation. This prevents profiling-file
initialization and avoids double-applying the current profile-oriented
precision scaling.

### Illustrative Runtime Configuration

The final flag names may differ after flat-dataclass integration, but the
intended configuration is:

```text
--device rubin
--network_device vera_rubin_nvl72_domain
--execution_time_predictor_config_type analytical_roofline
--cc_backend_config_type astra_sim_analytical
--astra_sim_analytical_cc_backend_config_intra_server_topology Switch
--astra_sim_analytical_cc_backend_config_intra_server_bandwidth_gbps 14400
--astra_sim_analytical_cc_backend_config_intra_server_latency_us 1.0
```

For CPU KV-cache offload, the user-facing configuration may either:

1. accept already aggregated target values; or
2. enable a Rubin static-slice mode that derives target values from the number
   of GPUs in the prefill cache target.

The second option is preferred because it prevents a one-GPU capacity from
being applied accidentally to a multi-GPU TP target.

## Integration Points

The expected implementation touches:

- `frontier/types/execution_time_predictor_type.py`
- `frontier/types/device_sku_type.py`
- `frontier/types/node_sku_type.py`
- `frontier/config/device_sku_config.py`
- `frontier/config/node_sku_config.py`
- `frontier/config/config.py`
- `frontier/execution_time_predictor/execution_time_predictor_registry.py`
- a new analytical predictor module under
  `frontier/execution_time_predictor/`
- CPU KV-cache target materialization or validation code; and
- runnable examples under `examples/architecture/`.

The predictor should return Frontier's existing `AttentionTime`, `MLPTime`,
`MoETime`, communication timing, and `ExecutionTime` structures. It should use
the canonical operator families so operation-level traces remain compatible
with the current metrics and comparison tools.

No new communication backend is required for the MVP. The existing
ASTRA-Sim-inspired analytical backend is configured to represent the logical
NVL72 switch domain.

## Implementation Phases

### Phase 1: Hardware and Configuration

1. Add Rubin device and logical NVL72 node SKUs.
2. Add precision-specific hardware ceilings and HBM bandwidth.
3. Add the analytical predictor configuration and registry entry.
4. Validate that a 72-GPU cluster materializes as one ASTRA switch domain.

### Phase 2: Dense Analytical Predictor

1. Implement dense projections.
2. Implement normalization, residual, activation, RoPE, and KV writes.
3. Implement dense prefill and decode attention.
4. Compose single-layer and pipeline-stage `ExecutionTime`.
5. Export per-operator analytical diagnostics.

### Phase 3: MoE

1. Implement gating linear and TopK costs.
2. Implement local shuffling costs.
3. Implement grouped GEMM using `per_expert_tokens`.
4. Preserve the existing EP communication path.

### Phase 4: CPU Static Slices

1. Add or document the Rubin per-GPU CPU slice.
2. Derive aggregate target capacity and bandwidth from target GPU count.
3. Preserve the existing full-duplex serialized transfer-engine behavior.
4. Export the resolved per-target values in configuration/metrics metadata.

### Phase 5: Examples and Validation

1. Add one co-location Rubin example.
2. Add one sequential-PDD Rubin example with CPU KV offload.
3. Add unit tests for formulas and topology materialization.
4. Backtest the analytical predictor on existing hardware profiles.

## Validation Plan

### Formula Unit Tests

Tests must verify:

- non-negative execution times;
- monotonicity with FLOPs and HBM bytes;
- inverse scaling with peak compute and HBM bandwidth;
- correct `max + overlap * min` behavior;
- correct TP-local shapes;
- correct GQA/MQA KV traffic;
- correct precision byte widths;
- correct per-expert MoE load handling; and
- no duplicate timing from fused operators.

### Hardware-Independent Backtest

Use existing H100/H800 kernel-only CSVs to run the same analytical formulas
with H100/H800 hardware profiles. Report at least:

- median absolute percentage error;
- P90 absolute percentage error;
- error by operator class;
- error by token/shape bucket; and
- bottleneck-classification agreement where it can be inferred.

The objective is not to tune one coefficient per CSV row. The backtest should
use a small shared coefficient set and test whether it transfers across
unseen shapes.

### Network Tests

Tests must verify:

- all 72 GPUs resolve inside one logical switch domain;
- participant groups do not use `inter_server` parameters;
- collective latency increases with payload and participant count;
- one-device collectives return zero; and
- the configured 14,400-Gbps directional bandwidth is used.

### CPU Slice Tests

Tests must verify:

- one GPU resolves to 750 GB and 4,800 Gbps;
- an `N`-GPU target resolves to `N` times those values;
- capacity accounting still uses aggregate physical TP KV shards;
- existing CPU offload/restore event ordering is unchanged; and
- disabling CPU offload preserves existing simulation behavior.

### End-to-End Smoke Tests

The examples must complete with:

- no Rubin profiling CSVs;
- analytical execution time enabled;
- ASTRA analytical communication enabled;
- CSV/JSON metrics enabled; and
- deterministic output for a fixed workload seed.

## Acceptance Criteria

The MVP is complete when:

1. A Rubin NVL72 simulation runs without target-hardware profiling data.
2. All 72 GPUs are represented as one logical ASTRA `Switch` domain.
3. Layer timing is calculated from FLOPs, bytes, hardware ceilings, and
   explicit efficiency/overlap parameters.
4. The predictor supports dense prefill and decode; MoE support may land as a
   separately tracked phase but must use the same analytical framework.
5. CPU KV-cache capacity and bandwidth are statically divided per GPU and
   correctly aggregated for multi-GPU cache targets.
6. Per-operator bottleneck diagnostics are exported.
7. Existing non-Rubin predictors and examples retain their behavior.
8. Unit, backtest, and end-to-end smoke tests pass.

## Non-Goals

The MVP does not include:

- exact Rubin kernel timing;
- a Rubin profiling dataset;
- L1/L2/shared-memory roofline ceilings;
- detailed CUDA occupancy or tile-wave simulation;
- exact CUDA Graph launch behavior;
- GPU thermal or power throttling;
- SHARP in-network compute;
- detailed NCCL algorithm/channel selection;
- communication contention between concurrent replicas;
- switch-chip-level routing;
- scale-out networking between racks;
- dynamic CPU bandwidth or capacity sharing;
- C2C/LPDDR/HBM resource contention;
- general weight or expert offloading; or
- CPU KV-cache modes beyond the release-supported sequential-PDD path.

## Follow-Up Fidelity Upgrades

The MVP is structured so the following can be added without replacing its
operator contracts:

1. Add cache-level rooflines and tile-aware occupancy.
2. Replace constant efficiency values with transparent shape curves.
3. Add SHARP-aware collective formulas.
4. Promote NVLink bandwidth to a shared DES resource for contention.
5. Introduce a `HostComplex` containing one Vera CPU and two Rubin GPUs.
6. Replace static CPU slices with dynamic LPDDR and C2C arbitration.
7. Model overlap between compute, communication, and offload.
8. Calibrate the coefficients with Rubin measurements when hardware becomes
   available.

Until then, studies using this model should report configuration values and
prefer sensitivity ranges over claims of exact absolute latency.
