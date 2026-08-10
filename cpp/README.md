# Frontier C++ Core

This directory contains the deterministic C++ port of Frontier's dense
co-location and sequential PDD simulation paths.

Implemented behavior includes:

- hierarchical global, cluster, replica, and pipeline-stage scheduling;
- typed discrete events for arrivals, scheduling, pipeline execution,
  completions, and PDD KV-cache transfers;
- FCFS vLLM V1-style continuous batching, chunked prefill, KV-block
  accounting, and recompute preemption;
- multiple replicas plus TP, PP, and DP;
- fixed per-stage and configurable Rubin/GB300 analytical execution models;
- sequential prefill/decode clusters with analytical KV-cache transfer;
- finite target-local PREFILL CPU KV-cache offload/restore for sequential PDD;
- strict JSON configuration and CSV workload contracts; and
- CTest plus production-Python differential tests.

Session prefix caching and MoE/expert parallelism are implemented. Block-hash
prefix caching, parallel PDD clusters, and topology-aware communication
backends remain outside the current C++ surface.

## Build in WSL

Use the Windows worktree as the source and a WSL-native build directory:

```bash
cd /mnt/c/Users/jklpr/Desktop/project/Frontier-cxx-port
export PATH="$HOME/frontier-tools/bin:$PATH"

cmake -S cpp \
  -B "$HOME/frontier-build/cxx-port" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON

cmake --build "$HOME/frontier-build/cxx-port"
ctest \
  --test-dir "$HOME/frontier-build/cxx-port" \
  --output-on-failure
```

The baseline is C++17, GCC 11+ or Clang 14+, CMake 3.24+, and Ninja 1.10+.

## Core value contract

Strong IDs and `SimTime` are value types with an invalid default sentinel.
Their underlying value is `-1`; zero and positive values are valid. Persistent
entity and simulator state stores these values directly and checks `valid()`
instead of wrapping them in `std::optional`. Optional return values remain
appropriate for operations whose result may be absent, such as selecting a
preemption victim.

## Run

Dense co-location:

```bash
"$HOME/frontier-build/cxx-port/frontier_sim" \
  --config cpp/tests/fixtures/config/fixed_parallel_colocation.json \
  --workload cpp/tests/fixtures/workloads/step25_parallel.csv
```

Sequential PDD:

```bash
"$HOME/frontier-build/cxx-port/frontier_sim" \
  --config cpp/tests/fixtures/config/fixed_sequential_pdd.json \
  --workload cpp/tests/fixtures/workloads/step3_pdd_small.csv
```

For user-facing recipes, start with `cpp/examples` instead of the test
fixtures:

```powershell
python cpp/examples/run_example.py hello `
  --binary .build-step5/frontier_sim.exe

python cpp/examples/run_example.py pdd `
  --binary .build-step5/frontier_sim.exe

python cpp/examples/run_example.py cpu-kv-online `
  --binary .build-step5/frontier_sim.exe
```

The example runner writes normalized inputs and analysis-ready artifacts under
`outputs/cpp_examples/`. See `cpp/examples/README.md` for dense analytical,
KV-pressure, MoE, PDD, and session-prefix-cache recipes.

### Analysis-ready output

Pass an output directory to avoid emitting the entire detailed trace to
stdout:

```powershell
.build-step5/frontier_sim.exe `
  --config cpp/examples/configs/03_sequential_pdd.json `
  --workload cpp/examples/workloads/00_tiny.csv `
  --output-dir outputs/my-pdd-run `
  --output-mode requests
```

Every output-directory run writes `config.normalized.json`,
`workload.normalized.csv`, and `summary.json`. Mode `requests` also writes
`requests.csv`; mode `full` additionally retains detailed runtime records and
writes `trace.json`. Summary and requests modes disable detailed event,
scheduler, batch, and analytical traces during the run.

`summary.json` reports request/token throughput, latency mean/p50/p90/p99,
preemptions, cluster batch distributions, KV-transfer latency, and prefix-cache
hit rate.

Read-only normalization:

```bash
"$HOME/frontier-build/cxx-port/frontier_sim" \
  --normalize-config \
  cpp/tests/fixtures/config/fixed_parallel_colocation.json

"$HOME/frontier-build/cxx-port/frontier_sim" \
  --normalize-workload \
  cpp/tests/fixtures/workloads/session_prefix.csv
```

## Configuration contract

There is one current configuration contract: `schema_version: 1`. Older
development-step schemas are not accepted.

Common top-level fields are:

```json
{
  "schema_version": 1,
  "run_id": "example",
  "simulation_mode": "online",
  "system_architecture": "co-location",
  "enable_parallel_clusters": false,
  "prefix_cache": {
    "enabled": false,
    "key_mode": "session"
  },
  "cluster_scheduler": {
    "type": "round_robin"
  },
  "clusters": {}
}
```

The fields shown above are required. The additional top-level `cpu_kv_cache`
object may be omitted and then normalizes to the disabled default. Unknown
fields and unsupported values are rejected.
`enable_parallel_clusters` must currently be `false`. Session prefix caching
is supported with `prefix_cache.enabled=true`, `key_mode="session"`, and
either `sticky_round_robin` or `cache_aware` when more than one replica/DP
target is available. `cache_aware` queries the actual GPU-resident prefix and
may migrate to a least-loaded target; migration discards the old target's
session KV.

### GPU HBM and KV-block capacity

Every cluster must provide its physical per-GPU HBM capacity. Analytical
clusters may either keep an explicit `scheduler.num_blocks` value or omit it
and request automatic block sizing:

```json
"gpu_memory": {
  "capacity_bytes_per_gpu": 288000000000,
  "auto_calculate_num_blocks": true,
  "runtime_reserve_fraction": 0.10,
  "runtime_reserve_bytes": 0,
  "weight_overhead_fraction": 0.0
}
```

The parser computes the maximum per-pipeline-stage weight footprint on one
GPU, accounting for attention TP, MoE TP/EP, embeddings/LM head, model weight
precisions (including FP4), and replicated weights. It then subtracts weights
and the runtime reserve from HBM and divides the remainder by the rank-local KV
block size. KV precision, MLA/GQA layout, block size, and DCP token sharding
therefore all affect the resolved block count. `config.normalized.json` emits
`model_weight_bytes_per_gpu`, `kv_cache_budget_bytes_per_gpu`,
`kv_cache_bytes_per_block`, and the resolved `scheduler.num_blocks` for audit.

For manual sizing, retain `scheduler.num_blocks` and set
`auto_calculate_num_blocks` to `false` (or omit that flag; the parser infers
manual mode when `num_blocks` is present). The configured block count is still
checked against the stage/rank-local physical capacity. Fixed-latency models
must use manual sizing because they do not carry a weight-precision contract.

### CPU KV-cache tiering

CPU tiering is an opt-in sequential-PDD feature. It requires session prefix
caching, `sticky_round_robin` or `cache_aware`, a PREFILL `vllm_v1` scheduler,
and `enable_parallel_clusters=false`. Each PREFILL `(replica_id, dp_id)` target
owns an independent finite store and one serialized queue per transfer
direction; D2H and H2D may overlap.

Start from
`cpp/examples/configs/06_cpu_kv_cache_pdd_online.json` or the matching offline
config. The normalized top-level object selects either direct capacity or a
static per-GPU slice and configures analytical transfer bandwidth/latency plus
`prefix_fit` or `skip_offload` pressure behavior. Full output includes
`cpu_kv_cache`, `cpu_kv_cache_targets`, and `cpu_kv_cache_transfers`; request
records distinguish restored blocks transferred, consumed, and discarded.

### Co-location clusters

Co-location has exactly one `monolithic` cluster:

```json
{
  "clusters": {
    "monolithic": {
      "parallelism": {
        "num_replicas": 2,
        "tensor_parallel_size": 2,
        "decode_context_parallel_size": 1,
        "pipeline_parallel_size": 2,
        "data_parallel_size": 2,
        "moe_tensor_parallel_size": 1,
        "moe_expert_parallel_size": 1
      },
      "scheduler": {
        "type": "vllm_v1",
        "scheduling_policy": "fcfs",
        "batch_size_cap": 4,
        "max_tokens_in_batch": 8,
        "enable_preemption": true,
        "enable_chunked_prefill": true,
        "long_prefill_token_threshold": 0,
        "block_size": 4,
        "num_blocks": 16,
        "watermark_blocks_fraction": 0.0,
        "num_preallocate_tokens": 0
      },
      "gpu_memory": {"capacity_bytes_per_gpu": 288000000000},
      "execution_model": {
        "type": "fixed",
        "stage_latencies_ms": [1.0, 3.0]
      },
      "model_name": "meta-llama/Llama-2-7b-hf",
      "total_expert_num": 1,
      "router_topk": 1,
      "moe_routing": {
        "mode": "simulation",
        "distribution": "balanced",
        "seed": 42
      }
    }
  }
}
```

For MLA models, set `decode_context_parallel_size` to a divisor of
`tensor_parallel_size` to model vLLM-style token-interleaved DCP. DCP reuses
the TP ranks and does not add GPUs. A token at global position `i` is stored on
DCP rank `i % decode_context_parallel_size`; for example, TP4/DCP4 removes the
four-way latent-KV replication while adding decode attention collectives.
The analytical diagnostics keep `kv_cache_bytes_per_token_per_layer` as the
logical one-copy size and expose the per-GPU value separately as
`kv_cache_rank_local_bytes_per_token_per_layer`.
For `pd-disaggregation`, PREFILL must set
`decode_context_parallel_size` to `1`; DCP is allowed only on the DECODE
cluster. PDD KV-transfer sizing follows the DECODE target layout.

The number of fixed stage latencies must equal
`pipeline_parallel_size`.

`model_name` follows Python's `ReplicaConfig.model_name` contract. C++ first
looks in the same built-in model registry as Python, then falls back to
`data/config/models/<model-name-with-slashes-replaced-by-__>.json`. For
example, `moonshotai/Kimi-K2-Instruct` resolves to
`data/config/models/moonshotai__Kimi-K2-Instruct.json`. Set
`FRONTIER_MODEL_CONFIG_DIR` to use an additional model-asset directory.

For MoE models, `total_expert_num` and `router_topk` default to the model
asset's `num_experts` and `num_experts_per_tok`. They may be provided at the
cluster level as runtime overrides. Dense models always use `1` for both.

### Sequential PDD clusters

PDD uses the same cluster object for `prefill` and `decode`, plus a transfer
model:

```json
{
  "system_architecture": "pd-disaggregation",
  "clusters": {
    "prefill": {
      "parallelism": {},
      "scheduler": {},
      "gpu_memory": {"capacity_bytes_per_gpu": 288000000000},
      "execution_model": {},
      "model_name": "meta-llama/Llama-2-7b-hf",
      "moe_routing": {}
    },
    "decode": {
      "parallelism": {},
      "scheduler": {},
      "gpu_memory": {"capacity_bytes_per_gpu": 288000000000},
      "execution_model": {},
      "model_name": "meta-llama/Llama-2-7b-hf",
      "moe_routing": {}
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

The abbreviated cluster objects above indicate shape only; every field shown
in the co-location cluster example remains required.

### Analytical execution

An analytical cluster replaces its fixed execution model with:

```json
{
  "type": "analytical",
  "device": "rubin",
  "precision": "fp16",
  "moe_layer_event_mode": "detailed",
  "operator_precisions": {
    "attention": "fp8",
    "dense": "fp8",
    "moe_expert": "fp8",
    "moe_expert_weight": "fp4",
    "moe_expert_activation": "fp8",
    "moe_router": "fp8",
    "lm_head": "fp8",
    "kv_cache": "fp8",
    "communication": "fp8"
  },
  "network_bandwidth_gbps": 400.0,
  "network_latency_us": 1.0,
  "intra_node_bandwidth_gbps": 14400.0
}
```

`device` selects a built-in hardware ceiling preset. The available presets are
`rubin` and `gb300`. `gb300` uses dense, non-sparse per-GPU ceilings derived
from the public GB300 NVL72 rack totals: 8 TB/s HBM, 83.333 TFLOPS FP32,
2,500 TFLOPS FP16/BF16, 5,000 TFLOPS FP8/INT8, and 15,000 TFLOPS FP4/INT4.

Any preset ceiling can be overridden independently without copying the other
values:

```json
{
  "device": "gb300",
  "device_overrides": {
    "hbm_bandwidth_tbps": 7.5,
    "fp8_tflops": 4750.0
  }
}
```

The configurable ceiling fields are `hbm_bandwidth_tbps`, `fp32_tflops`,
`fp16_tflops`, `fp8_tflops`, and `fp4_tflops`. Use `device: "custom"` to
define hardware without a preset; custom devices must provide all five fields.
BF16 uses the FP16 ceiling, INT8 uses the FP8 ceiling, and INT4 uses the FP4
ceiling.

`operator_precisions` is optional. Any omitted field inherits `precision`, so
existing single-precision configs retain their behavior. Supported values are
`fp32`, `fp16`, `bf16`, `fp8`, `mxfp8`, `int8`, `fp4`, `mxfp4`, and `int4`.
MXFP storage includes one E8M0 scale byte per 32 values, so analytical HBM
traffic uses 1.03125 bytes/value for MXFP8 and 0.53125 for MXFP4. The KV-cache value
controls cached-KV reads and writes; in PDD it must also match
`kv_cache_transfer.kv_cache_dtype_size_bytes` in both clusters.
Each compute family (`attention`, `dense`, `moe_expert`, `moe_router`, and
`lm_head`) can be split further with `_weight` and `_activation` suffixes.
The unsuffixed value remains the fallback for both, so the example above
models W4A8 experts while retaining the compact syntax elsewhere.

Kimi K3 analytical configs install the model-native policy below when neither
the new field nor its legacy family fallback is explicitly set:

The complete K3 design decision record is
[`docs/design/kimi-k3-support.md`](../docs/design/kimi-k3-support.md).

```json
{
  "routed_expert_weight": "mxfp4",
  "routed_expert_activation": "mxfp8",
  "latent_moe_projection_weight": "bf16",
  "latent_moe_projection_activation": "bf16",
  "shared_expert_weight": "bf16",
  "shared_expert_activation": "bf16",
  "dense_mlp_weight": "bf16",
  "dense_mlp_activation": "bf16",
  "router_weight_storage": "bf16",
  "moe_router_activation": "bf16",
  "router_compute": "fp32",
  "lm_head": "bf16",
  "kv_cache": "fp8",
  "kda_snapshot": "bf16"
}
```

Explicit per-family values take precedence. Legacy `moe_expert_*`, `dense_*`,
and `moe_router_*` fields remain valid fallbacks.

Stable LatentMoE's dense down/up projections have their own weight and
activation precision fields. Non-latent models ignore them. Existing latent-
MoE configs that omit them inherit `routed_expert_weight` and
`routed_expert_activation`, so their prior memory and latency behavior is
unchanged; Kimi K3 installs BF16/BF16 explicitly.

Router weight/input storage and compute are separate. K3 reads BF16 resident
router weights and BF16 hidden-state inputs, produces FP32 logits, and uses the
FP32 roofline ceiling for routing. BF16-to-FP32 conversion is assumed fused
into the router kernel, so no separate cast buffer or cast time is charged.
`moe_router_weight` remains a legacy fallback for
`router_weight_storage`.

Kimi K3's 24 MLA layers also enable `mla_use_output_gate` and
`mla_use_nope`. The analytical MLA path charges the TP-sharded full-rank gate
projection plus its fused sigmoid/multiply before the output projection. NoPE
removes only the rotary transform cost: the configured 64-wide shared Q/K
component remains part of attention work, persistent cache storage, and DCP
traffic, matching the published K3 forward path. Legacy MLA models leave both
flags disabled.

K3 AttnRes metadata is retained, but AttnRes contributes zero analytical
latency and zero automatic GPU weight-memory overhead. It is treated as a
negligible or fused operation until a measured kernel model is available.

When KDA prefix caching is enabled, each session owns at most one latest
snapshot. Snapshot capacity is charged in KV-equivalent blocks, but the
snapshot itself is atomic. GPU and CPU pressure select sessions in LRU order;
within one victim session they evict ordinary KV from the suffix first. If
additional capacity is still required after that KV reaches zero, they evict
that same session's whole snapshot before advancing to the next session.
An atomic snapshot eviction may therefore reclaim more blocks than the current
deficit. Active or transfer-pinned sessions remain non-evictable.
For prompts shorter than one cache block, CPU caching transfers the KDA
snapshot as a standalone payload while keeping reusable KV blocks and cached
tokens at zero.
Cold PREFILL admission includes one snapshot charge in both empty-capacity and
current-capacity checks, then atomically reserves that group before compute.
The first batch completion publishes into the reserved group, while an
existing same-session snapshot is updated without a second charge. A failed
asynchronous admission releases an unpublished frontier-zero reservation.

Sequential PDD requires the PREFILL and DECODE KDA snapshot precisions to
match exactly; equal byte widths such as BF16 and FP16 are not interchangeable.
Fixed execution implies an FP32 snapshot, so a mixed analytical/fixed PDD
configuration is valid only when the analytical side also selects FP32.

For MLA models, the analytical path uses distinct phase-specific execution
modes. Prefill uses compute-friendly unabsorbed MHA: cached latent KV is
expanded through the KV up-projection, the head-specific NoPE K/V result is
materialized in a temporary HBM workspace, and attention uses the model's
ordinary QK/V head dimensions. Decode uses data-movement-friendly absorbed
MQA: the query is transformed into latent space, attention reads the latent
cache directly, and the latent result is expanded on the output side. The
temporary prefill workspace contributes projection FLOPs and HBM write/read
time but does not consume scheduler KV-block capacity.

Persistent MLA cache layout is component-wise rather than a uniform tensor:
the latent component uses `operator_precisions.kv_cache`, while the decoupled
RoPE component remains BF16. MXFP scale metadata is included; quantization time
is omitted.
For Kimi K2 with FP8 KV this is `512 * 1 + 64 * 2 = 640` bytes per token per
layer. The same layout is used for cache writes, decode reads, and PDD transfer;
for MLA, `kv_cache_transfer.kv_cache_dtype_size_bytes` therefore specifies the
latent component size and the BF16 RoPE bytes are added automatically.

The analytical predictor derives the model and layer count from the cluster's
`model_name`; they are not repeated in `execution_model`. TP comes from the
cluster's `parallelism` object. The current model validates TP 1/2/4/8 and
allows uneven contiguous PP partitions (for example, 61 layers over PP4 become
16/15/15/15). MoE assets may configure a dense prefix and shared experts;
shared experts are replicated across EP lanes and sharded only by MoE TP. The
last PP stage also models the vocabulary-parallel LM-head projection. Supported
attention families include dense-KV MHA/GQA/MQA, Step3Text MFA's shared-Q path,
and latent-cache MLA. Frozen DSA remains unsupported.

`moe_layer_event_mode` is optional and defaults to `detailed`. In this mode the
analytical predictor evaluates one MoE layer only when that layer is about to
run; the scheduler then records its routing, executes its synchronization/event
sequence, and accumulates its execution-time components into the batch stage.
It does not predict and retain the whole PP stage again at every layer.
`first_layer_scaled` runs the first MoE layer through the same routing,
expert-lane, barrier, and communication events, then reuses that MoE critical
path for the remaining contiguous MoE layers. Attention is scaled separately
per implementation family: standard attention, MLA, and KDA each retain one
representative roofline time and a PP-stage-local layer count. Kimi K2 therefore
keeps its single MLA representative, while hybrid Kimi K3 stages preserve their
distinct KDA and MLA totals. Execution-time component totals remain unchanged.
The mode deliberately omits inter-layer synchronization and congestion changes
after the representative MoE event. A dense prefix is supported, but a dense
layer after the first MoE layer in the same PP stage is rejected.

`stage_group_scaled` is the PP-aware approximation mode. It derives a
deterministic signature from each stage's ordered layer families and boundary
work, groups only identical signatures, and retains separate standard/MLA/KDA
attention totals. Representative MoE lane reuse is enabled only for
layer-invariant routing (legacy uniform or balanced simulation routing).
Layer-dependent routing and non-contiguous MoE layouts automatically use the
detailed per-layer prediction path instead of reusing a mismatched first
layer. PP stage arrival/end events remain unchanged.

## Workload contract

Required CSV columns:

```text
session_start_at,think_time,num_prefill_tokens,num_decode_tokens
```

Optional columns:

```text
session_id,session_turn_index
```

Request IDs follow row order from zero. A standalone request or the first turn
of a session records `session_start_at` and uses `think_time=0`. A later turn
leaves `session_start_at` empty and records its delay after the previous turn's
terminal completion in `think_time`:

```text
actual_arrival = predecessor_completion + think_time
```

Both time fields must be finite and nonnegative when present; token counts must
be positive integers. Duplicate, missing, unknown, quoted, legacy `arrived_at`,
and block-hash columns are rejected.

Generate a workload without importing the Python simulator:

```powershell
python cpp/frontier/request_generator/generate_workload.py `
  --config cpp/frontier/request_generator/example_workload.json `
  --output outputs/workload.csv
```

The generator implements fixed, uniform, Zipf, and bounded log-normal request
lengths plus static, Poisson, and Gamma initial-arrival distributions. Its
single-turn output records `session_start_at` with zero think time and uses the
same normalized CSV contract parsed by `frontier_sim`.

## Output contract

The single output schema is also version 1. It always contains canonical
request, batch, batch-stage, scheduler, event, and analytical diagnostic
records. Co-location records identify `(replica_id, dp_id)` targets. PDD
records additionally identify prefill/decode owners, cluster types, and
KV-cache transfers.

Timestamps use seconds with `_s` suffixes. Latencies use milliseconds with
`_ms` suffixes. TTFT is measured from request arrival to completion of the
first generated token. Arrival-to-Prefill completion is reported separately as
`prefill_latency_ms`.
IDs, token counts, arrays, and event order are exact; floating-point parity
comparisons use `1e-12` absolute and relative tolerance.

## JSON dependency

The contracts use `nlohmann/json` 3.11.3. CMake resolves an installed package,
`FRONTIER_NLOHMANN_JSON_SOURCE_DIR`, or the pinned upstream archive when
`FRONTIER_FETCH_DEPENDENCIES=ON`.

## Python/C++ differential gate

From Windows PowerShell:

```powershell
$env:FRONTIER_CPP_BINARY = "/home/dskim/frontier-build/cxx-port/frontier_sim"
$env:FRONTIER_CPP_RUNNER = '["wsl","-d","Ubuntu","-e"]'
$env:FRONTIER_CPP_PATH_STYLE = "wsl"
python -m pytest cpp/tests/parity/test_differential.py -q -p no:cacheprovider
```

The matrix covers offline and online workloads, fixed and analytical timing,
multiple replica/TP/PP/DP combinations, chunking, watermark and preallocation
settings, pressure/preemption, sequential PDD routing, and transfer timing.
