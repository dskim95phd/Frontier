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
- strict JSON configuration and CSV workload contracts; and
- CTest plus production-Python differential tests.

Session prefix caching, block-hash prefix caching, MoE/expert parallelism,
parallel PDD clusters, and topology-aware communication backends remain
outside the current C++ surface.

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

All fields are required. Unknown fields and unsupported values are rejected.
`enable_parallel_clusters` and `prefix_cache.enabled` must currently be
`false`.

### Co-location clusters

Co-location has exactly one `monolithic` cluster:

```json
{
  "clusters": {
    "monolithic": {
      "parallelism": {
        "num_replicas": 2,
        "tensor_parallel_size": 2,
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
      "execution_model": {},
      "model_name": "meta-llama/Llama-2-7b-hf",
      "moe_routing": {}
    },
    "decode": {
      "parallelism": {},
      "scheduler": {},
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
`fp32`, `fp16`, `bf16`, `fp8`, `int8`, `fp4`, and `int4`. The KV-cache value
controls cached-KV reads and writes; in PDD it must also match
`kv_cache_transfer.kv_cache_dtype_size_bytes` in both clusters.
Each compute family (`attention`, `dense`, `moe_expert`, `moe_router`, and
`lm_head`) can be split further with `_weight` and `_activation` suffixes.
The unsuffixed value remains the fallback for both, so the example above
models W4A8 experts while retaining the compact syntax elsewhere.

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
RoPE component remains BF16. Scale metadata and quantization time are omitted.
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
expert-lane, barrier, and communication events, then waits for the first-layer
time multiplied by the remaining contiguous MoE-layer count. Execution-time
component totals remain unchanged. This mode is intended for repeated balanced
analytical layers; it deliberately omits inter-layer synchronization and
congestion changes after the representative first layer. A dense prefix is
supported, but a dense layer after the first MoE layer in the same PP stage is
rejected.

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
`_ms` suffixes. TTFT is measured from request arrival to prefill completion.
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
