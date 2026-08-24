# Frontier C++ Core

This directory contains Frontier's deterministic C++ simulation core for the
co-location and sequential PDD architectures. It is the primary
implementation; see [`../AGENTS.md`](../AGENTS.md) for how it relates to the
Python simulator.

Implemented behavior includes:

- hierarchical global, cluster, replica, and pipeline-stage scheduling;
- typed discrete events for arrivals, scheduling, pipeline execution,
  completions, and PDD KV-cache transfers;
- FCFS vLLM V1-style continuous batching, chunked prefill, KV-block
  accounting, and recompute preemption;
- multiple replicas plus TP, PP, DP, and decode context parallelism;
- MoE with expert parallelism, routing/imbalance modeling, and lockstep
  synchronization;
- fixed per-stage and configurable Rubin/GB300 analytical execution models
  with per-operator precisions down to MXFP4;
- session prefix caching;
- sequential prefill/decode clusters with analytical KV-cache transfer;
- finite target-local PREFILL CPU KV-cache offload/restore for sequential PDD;
- automatic KV block sizing from physical per-GPU HBM; and
- strict JSON configuration and CSV workload contracts.

Block-hash prefix caching, `pd-af-disaggregation`, and topology-aware
communication backends remain outside the C++ surface.

## Build

On Linux or WSL, configure a build from the repository root:

```bash
cmake -S cpp \
  -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON

cmake --build build
ctest --test-dir build --output-on-failure
```

The baseline is C++17, GCC 11+ or Clang 14+, CMake 3.24+, and Ninja 1.10+.
Windows is tested with current MSVC. When building a Windows worktree from
WSL, keep the build directory on the WSL filesystem for substantially faster
compilation.

Preset users can run the following from `cpp/`:

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

For an installable release tree and archive:

```bash
cmake --preset release
cmake --build --preset release
cmake --install ../build/release --prefix ../stage
cpack --config ../build/release/CPackConfig.cmake -B ../packages
```

The installed layout contains `bin/frontier_sim`, model assets under
`share/frontier/models`, and the public JSON Schema under
`share/frontier/schema`. It can be moved as a unit; model discovery is relative
to the executable. `FRONTIER_MODEL_CONFIG_DIR` remains available as the
highest-priority override for custom assets.

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
build/frontier_sim \
  --config cpp/tests/fixtures/config/fixed_parallel_colocation.json \
  --workload cpp/tests/fixtures/workloads/step25_parallel.csv
```

Sequential PDD:

```bash
build/frontier_sim \
  --config cpp/tests/fixtures/config/fixed_sequential_pdd.json \
  --workload cpp/tests/fixtures/workloads/step3_pdd_small.csv
```

For user-facing recipes, start with `cpp/examples` instead of the test
fixtures:

```bash
python cpp/examples/run_example.py hello \
  --binary build/frontier_sim

python cpp/examples/run_example.py pdd \
  --binary build/frontier_sim

python cpp/examples/run_example.py cpu-kv-online \
  --binary build/frontier_sim
```

The example runner writes normalized inputs and analysis-ready artifacts under
`outputs/cpp_examples/`. See `cpp/examples/README.md` for dense analytical,
KV-pressure, MoE, PDD, and session-prefix-cache recipes.

### Analysis-ready output

Pass an output directory to avoid emitting the entire detailed trace to
stdout:

```bash
build/frontier_sim \
  --config cpp/examples/configs/03_sequential_pdd.json \
  --workload cpp/examples/workloads/00_tiny.csv \
  --output-dir outputs/my-pdd-run \
  --output-mode requests
```

Every output-directory run writes `config.normalized.json`,
`workload.normalized.csv`, `summary.json`, and `gpu_kv_occupancy.csv`. Mode
`requests` also writes `requests.csv`; mode `full` additionally retains
detailed runtime records and writes `trace.json`. Summary and requests modes
disable detailed event, scheduler, batch, and analytical traces during the
run. Occupancy samples are compact change events and are retained in every
mode.

`summary.json` reports request/token throughput, latency mean/p50/p90/p99,
preemptions, cluster batch distributions, KV-transfer latency, and prefix-cache
hit rate.

### Run options

| Option | Default | Effect |
| --- | --- | --- |
| `--output-mode summary\|requests\|full` | `summary` | Which artifacts to write. Requires `--output-dir`. |
| `--runtime-validation true\|false` | `true` | Per-iteration scheduler, KV-accounting, and CPU-ownership invariant checks. Disable for throughput once a configuration is trusted. |
| `--gpu-kv-occupancy true\|false` | `true` | Whether to record the GPU KV occupancy sample stream. |
| `--wall-progress-interval-s <seconds>` | unset | Emit `simulation_progress_s=<seconds>` to stderr after each wall-clock interval. |
| `--simulation-end-time-s <seconds>` | unset | Stop at a bounded observation horizon instead of running to quiescence. |

`--simulation-end-time-s` runs a bounded experiment: only requests that
reached canonical completion are exported, quiescence validation is skipped,
and cache/transfer diagnostics are snapshotted at the horizon. The output
then carries `observation_window_seconds`, and `summary.json` computes rates
against that window rather than against first-arrival-to-last-completion.
For online runs, complete sessions whose external start time is after the
horizon are omitted before simulator entities are constructed. Retained
requests keep their original workload-row `request_id`; offline runs are not
pruned because they preload root requests at simulation time zero.

Progress reporting is disabled unless `--wall-progress-interval-s` is set.
For example, `--wall-progress-interval-s 60` prints the latest processed
simulation time approximately once per wall-clock minute. Progress goes to
stderr so stdout remains valid deterministic JSON when `--output-dir` is not
used.

Read-only normalization:

```bash
build/frontier_sim \
  --normalize-config \
  cpp/tests/fixtures/config/fixed_parallel_colocation.json

build/frontier_sim \
  --normalize-workload \
  cpp/tests/fixtures/workloads/session_prefix.csv
```

## Configuration contract

Two input contracts are accepted. `schema_version: 1` is the standalone,
fully resolved runtime contract. `schema_version: 2` is a modular scenario
that references named GPU, cluster, and inter-cluster link assets. The loader
resolves v2 to v1 before constructing simulator state, so both forms have the
same validation, normalized output, and numerical behavior. Older
development-step schemas are not accepted.

The machine-readable contracts are
[`schema/config-v1.schema.json`](schema/config-v1.schema.json),
[`schema/config-v2.schema.json`](schema/config-v2.schema.json), and the three
`*-asset-v1.schema.json` files in the same directory. Release archives install
them under `share/frontier/schema`.

### Modular configuration (`schema_version: 2`)

A modular scenario keeps experiment-specific model, parallelism, scheduler,
and precision choices in the main file. Physical GPU memory and analytical
device identity come from a GPU asset. A cluster asset selects a GPU profile,
an exact accelerator count, and the cluster-internal network. It may also set
deployment-level `gpu_memory` defaults such as runtime reservation, while the
GPU asset remains authoritative for physical HBM capacity. PDD transfer
bandwidth and latency come from a link asset. The reference shape below
abbreviates the unchanged v1 `parallelism` and `scheduler` objects; use the
linked example for a runnable document.

```jsonc
{
  "schema_version": 2,
  "run_id": "modular-pdd",
  "simulation_mode": "online",
  "system_architecture": "pd-disaggregation",
  "model": "meta-llama/Llama-2-7b-hf",
  "clusters": {
    "prefill": {
      "profile": "gb300-1gpu",
      "parallelism": { /* existing v1 parallelism fields */ },
      "scheduler": { /* existing v1 scheduler fields */ },
      "execution_model": {
        "type": "fixed",
        "stage_latencies_ms": [0.5]
      }
    },
    "decode": {
      "profile": "gb300-1gpu",
      "parallelism": { /* existing v1 parallelism fields */ },
      "scheduler": { /* existing v1 scheduler fields */ },
      "execution_model": {
        "type": "fixed",
        "stage_latencies_ms": [0.8]
      }
    }
  },
  "kv_cache_transfer": {
    "link": "ib-200g",
    "kv_cache_dtype_size_bytes": 2,
    "enable_compression": false
  }
}
```

The complete runnable form is
[`examples/configs/08_modular_sequential_pdd.json`](examples/configs/08_modular_sequential_pdd.json).
Optional `prefix_cache` and `cluster_scheduler` fields default to disabled
session caching and `round_robin`; `moe_routing` defaults to balanced
simulation with seed 42. `cpu_kv_cache` remains optional and uses the v1
object unchanged.

Analytical v2 deployments select a named execution-precision profile (or set
`precision` directly) and specify only scenario-specific kernel policy. Device
and network fields still come from the cluster/GPU assets:

```json
"execution_model": {
  "type": "analytical",
  "precision_profile": "kimi-k3-native",
  "moe_layer_event_mode": "stage_group_scaled"
}
```

Precision profiles live under `precision_profiles/` and contain `precision`
plus optional `operator_precisions`. A scenario may provide either field
locally; local values override the named profile, and local operator entries
are merged by key rather than replacing the whole profile.

Those fields are supplied by assets shaped as follows:

`gpus/gb300.json`:

```json
{
  "asset_schema_version": 1,
  "name": "gb300",
  "memory": {"capacity_bytes_per_gpu": 288000000000},
  "analytical": {
    "device": "gb300",
    "device_overrides": {
      "hbm_bandwidth_tbps": 8.0,
      "fp32_tflops": 83.33333333333333,
      "fp16_tflops": 2500.0,
      "fp8_tflops": 5000.0,
      "fp4_tflops": 15000.0
    }
  }
}
```

`clusters/gb300-1gpu.json`:

```json
{
  "asset_schema_version": 1,
  "name": "gb300-1gpu",
  "gpu": {"profile": "gb300", "count": 1},
  "network": {
    "network_bandwidth_gbps": 400.0,
    "network_latency_us": 1.0,
    "intra_node_bandwidth_gbps": 14400.0
  }
}
```

Asset names contain only letters, digits, `.`, `_`, and `-`. The loader looks
first in `assets/{gpus,clusters,links,precision_profiles}` beside the scenario,
then in nearby typed directories, `FRONTIER_CONFIG_ASSET_DIR`, installed
assets, and the repository `data/config` catalogue. A cluster profile describes
the exact allocation used by that simulated cluster. Its `gpu.count` must equal
the product of `num_replicas`, `data_parallel_size`,
`pipeline_parallel_size`, and `tensor_parallel_size`. DCP reuses TP ranks and
MoE shares the validated parallel domain, so neither adds accelerators to this
count.

`frontier_sim --normalize-config` resolves either input version to a complete,
self-contained schema-v1 document. Output-directory runs likewise write the
resolved form to `config.normalized.json`, making results reproducible without
the original asset catalogue.

### Standalone configuration (`schema_version: 1`)

The standalone common top-level fields are:

```json
{
  "schema_version": 1,
  "run_id": "example",
  "simulation_mode": "online",
  "system_architecture": "co-location",
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

`cluster_scheduler.type` selects how requests are routed to a
`(replica_id, dp_id)` target:

| Type | Routing policy |
| --- | --- |
| `round_robin` | Cyclic assignment across targets |
| `sticky_round_robin` | Cyclic, but pins a session to its first target |
| `cache_aware` | Queries each target's actual GPU-resident prefix |
| `kv_aware` | Compares projected KV-block pressure per target |
| `vllm_queue_aware` | Smallest observable outstanding queue (running + waiting) |

`type` is always required. For PDD, the optional `prefill_type` and
`decode_type` override it per cluster; an absent override falls back to
`type`. The optional `cache_threshold` (default `0.5`, in `[0, 1]`),
`balance_abs_threshold` (default `32`), and `balance_rel_threshold`
(default `1.1`, `>= 1`) tune when `cache_aware` prefers prefix affinity over
migrating to a less loaded target.

Session prefix caching is supported with `prefix_cache.enabled=true`,
`key_mode="session"`, and either `sticky_round_robin` or `cache_aware` when
more than one replica/DP target is available. `cache_aware` may migrate a
session to a least-loaded target; migration discards the old target's
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
caching, `sticky_round_robin` or `cache_aware`, and a PREFILL `vllm_v1`
scheduler. Each PREFILL `(replica_id, dp_id)` target owns an independent finite
store and one serialized queue per transfer direction; D2H and H2D may overlap.

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
        "moe_expert_parallel_size": 1,
        "pipeline_exclusive": false
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
        "seed": 42,
        "layer_scope": "shared"
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

#### PREFILL-only PDD

PDD experiments that analyze PREFILL capacity can replace DECODE scheduling
with an external fixed-rate lifecycle:

```json
"prefill_only": {
  "decode_tokens_per_second": 50.0
}
```

This optional top-level object is valid only with
`system_architecture="pd-disaggregation"`. PREFILL execution and the
analytical KV-cache transfer remain fully simulated. When the transfer ends,
the request generates output independently at the configured per-request
rate; it never enters a DECODE replica scheduler. Consequently, the mode
emits no DECODE batches or batch stages and assumes unlimited aggregate
external DECODE capacity.

For a transfer ending at `decode_start`, the first generated token completes
at `decode_start + 1 / decode_tokens_per_second`, and terminal completion is
`decode_start + num_decode_tokens / decode_tokens_per_second`. In online
session workloads, the next turn then arrives at terminal completion plus its
recorded `think_time`. PDD request records retain PREFILL ownership and
transfer fields, while `decode_replica_id` and `decode_dp_id` are `null` in
JSON (empty in CSV). Summary TTFT, TPOT, E2E, and decode-token throughput
therefore include the configured synthetic assumption rather than measured
DECODE capacity. Full JSON output also includes `prefill_completions`; bounded
online summaries use those boundary records for PREFILL counts, prompt-token
throughput, and PREFILL latency even when synthetic output remains in flight
at the observation horizon.

Run the included example with:

```bash
build/frontier_sim \
  --config cpp/examples/configs/07_prefill_only_pdd.json \
  --workload cpp/examples/workloads/00_tiny.csv
```

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

Kimi K3 analytical configs may select the bundled `kimi-k3-native` precision
profile. The model-aware resolver also installs the same defaults for older
standalone configs when neither the new field nor its legacy family fallback
is explicitly set:

The bundled `kimi-k3-lmsys-day0` profile reproduces the 2026 LMSYS serving
experiment instead: BF16 MLA KV, BF16 hidden-state collectives and combine,
and FP32 KDA recurrent state, while retaining native MXFP4/MXFP8 routed
experts. The public MegaMoE backend uses that routed activation dtype for
post-quant dispatch independently of the BF16 communication/combine dtype.

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

Router weight/input storage and logits are separate. K3 reads BF16 resident
router weights and BF16 hidden-state inputs on the BF16 Tensor Core GEMM
roofline, accumulates and writes FP32 logits, and performs Top-K over those
FP32 logits. BF16-to-FP32 conversion is assumed fused into the router kernel,
so no separate cast buffer or cast time is charged.
`moe_router_weight` remains a legacy fallback for
`router_weight_storage`.

The K3 kernel profiles also reproduce SGLang's fused LatentMoE front. EP paths
merge the 896-column router gate and 3584-column latent down projection into
one BF16 `[M,7168] x [7168,4480]` GEMM. Plain TP additionally folds in the
TP-local shared-expert gate/up projection. A front-specific TGV efficiency
applies only below the existing small-GEMM token threshold; large-M work
retains the ordinary large-GEMM roofline. Precision-incompatible custom
configurations fall back to separate kernels.

Numerical precision and CUDA scheduling are independent contracts. Selecting
the native K3 precision profile alone does not enable fused kernels. On decode,
`k3_sglang_mxfp4` also models fused route+quant, shared/routed overlap, TP8
column-sharded latent-up GEMM plus multicast all-gather for `M <= 12`, the
fused KDA update chain and projection overlap, and MLA output-gate overlap.
These switches do not alter prefill or unsupported batch/topology shapes.

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
allows uneven contiguous PP partitions. By default, stages are filled from the
front with `ceil(num_layers / PP)` layers while preserving at least one layer
for every stage (for example, 61 layers over PP4 become 16/16/16/13, and Kimi
K3's 93 layers over PP24 become 23 four-layer stages followed by one one-layer
stage). To override that front-filled partition, set the optional
`parallelism.pipeline_stage_layer_counts` array. It must contain exactly one
positive layer count per PP stage and sum to the model layer count. MoE assets
may configure a dense prefix and shared experts;
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
Because this mode is defined by one representative expert path, it always uses
one batch-shared routing draw even when `moe_routing.layer_scope` is
`per_layer`.
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

`execution_model.moe_communication_backend` is optional and defaults to
`generic`, which preserves the configured collective model. The opt-in
`sm100_megamoe_public` profile models the fused MegaMoE
dispatch → expert GEMMs → combine critical path on a GB200/GB300 NVL72 domain.
It uses linear fits of the public one-sided A2A latency measurements, charges
the logical unique-token copies after deduplicating multiple expert routes
from one token to the same destination lane, uses post-quant routed-activation
bytes for dispatch and BF16 bytes for combine, and
overlaps communication with the expert kernels. At an aligned DP barrier, each
active DP batch contributes one source row of destination-EP routed and
unique-token counts. Column sums form the receiver expert loads, while the
maximum logical row sum bounds rank-local dispatch/combine latency. Idle DP
participants contribute no traffic. For
DP-attention + EP-expert layouts,
dispatch/combine are the
layout transition, so this profile does not add a second DP input/output
all-reduce. It is a documented prior, not a workload-fitted calibration.
The profile may also be selected with the `rubin` device preset as an explicit
forward projection. In that mode arithmetic and HBM work use Rubin roofline
ceilings, the expert grid uses 224 rather than 160 SMs, and A2A payload time
scales by the public 3.6/1.8 TB/s NVLink ratio. The precision-correct GB300
refit selected no extra per-cluster residual, so the unidentifiable fitted
grid-latency term was removed from the model and configuration surface. Rubin
counted writes and tile-level dependent triggering may improve overlap, but
there is no K3 MegaMoE measurement to quantify it, so fixed startup and the
selected overlap residual also remain the GB300 priors. Block-M, block-N,
two-CTA clustering, and the functional form remain public SM100 priors rather
than Rubin measurements.
The Rubin A2A payload scale belongs to the device plus communication backend,
not to `kernel_profile`: explicitly selecting `sm100_megamoe_public` with a
generic kernel profile still uses the Rubin payload bandwidth. The ordinary
`generic` collective backend ignores this MegaMoE-specific scale.

`execution_model.kernel_profile` is also optional and defaults to `generic`,
which preserves the portable roofline efficiencies. Three Kimi-K3 profiles for
GB300, or explicit forward projection onto Rubin, make serving-stack
assumptions explicit instead of silently changing the device ceilings:

- `k3_flashkda_prefill` is an isolated prefill ablation. It keeps portable
  GEMM/MoE efficiencies but replaces KDA's conservative per-token state
  traffic with FlashKDA's 16-token K1/K2 pipeline, exact 13,824-byte
  chunk/head workspace, and public GB200 sequence/head scheduling envelope.
- `k3_sglang_mxfp4` models the published Blackwell K3 local-kernel path: W4A8
  SiTU expert kernels, the topology-specific two/three-way MoE front, fused
  route+quant, shared/routed overlap, TP8 latent-up GEMM+all-gather, fused KDA
  decode and auxiliary projection overlap, FlashKDA two-stage prefill, MLA
  gate/attention overlap,
  small-M TGV BF16 GEMMs, fused collective finalize work, and reduced launch
  overhead. Its batch-1 target is the published non-speculative approximately
  113 tok/s endpoint. This profile is required on every local K3 decode arm;
  it is not implied by an experiment label or by the precision profile.
- `k3_deepgemm_megamoe` applies only to the DP-source-composed destination-EP
  group prediction. It applies the public DeepGEMM SM100 block-M selection
  policy to the exact destination-lane expert histogram, charges tensor-core
  work and activation IO for padded M blocks. After correcting the router GEMM
  and dispatch/combine precision paths, no independent grid-latency term was
  identifiable; the DP2/EP16 points remain a topology holdout. DP-group
  recomposition includes routed destination-expert work only: router, latent
  projection, replicated shared expert, normalization, and finalize work keep
  each DP source's local token count. The shared-expert overlap component is
  carried from that first source prediction into the group barrier rather
  than recomputing a full MoE layer per source. In the MegaMoE profile, the EP
  side-stream overlap exposes half of the ideal `min(shared, routed)` credit,
  matching SGLang's reported 4-5% GB300 output-throughput gain without
  assuming contention-free streams. It also
  exposes the complete dispatch/GEMM/combine producer-consumer critical path.
  The former
  workload-fitted constant expert slowdown is not used; the A2A transport
  envelope remains the separate public NVIDIA prior described above.

  For controlled sensitivity or hardware refitting, this profile alone also
  accepts optional `mega_moe_tail_io_fraction` and
  `mega_moe_wave_exposure` overrides. Omitting them selects the documented
  device-profile priors.

Both profiles require `device="gb300"` or `device="rubin"` and a Kimi K3 model
asset. They are opt-in so existing analytical configs and numeric baselines
remain unchanged.
The model choice, equations, calibration split, and rejected alternatives are
recorded in
[`docs/design/kimi-k3-wide-ep-expert-kernel-model.md`](../docs/design/kimi-k3-wide-ep-expert-kernel-model.md).

Timing-template reuse is scoped to one immutable batch: equivalent PP stages
of that batch share one template per timing group. Different batches never
share templates, even when their token and context shapes match, and all
templates are released when the batch entity leaves the simulator.

For experiments that use PP specifically to partition large layer-resident
weights such as Kimi K3 KDA, set `parallelism.pipeline_exclusive=true`. This
mode requires `PP > 1`, `DP = 1`, `MoE EP = 1`, and `MoE TP = attention TP`.
It intentionally removes DP/EP MoE barriers while retaining TP collectives,
PP activation transfers, stage-local HBM/KV/snapshot accounting, and optional
DCP inside the attention TP group. The flag is opt-in; hybrid PP+DP+EP
configurations remain supported when it is false.

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
IDs, token counts, arrays, and event order are exact; floating-point
comparisons use `1e-12` absolute and relative tolerance.

## JSON dependency

The contracts use `nlohmann/json` 3.11.3. CMake resolves an installed package,
`FRONTIER_NLOHMANN_JSON_SOURCE_DIR`, or the pinned upstream archive when
`FRONTIER_FETCH_DEPENDENCIES=ON`.

## MoE routing across layers

`moe_routing.distribution` shapes a single expert assignment.
`moe_routing.layer_scope` decides how many assignments a batch draws, and the
two are independent:

| `layer_scope` | meaning |
| --- | --- |
| `shared` | every MoE layer routes the batch to the same experts. Routing and the expert roofline are evaluated once per batch and reused, so the compressed `moe_layer_event_mode` values stay exact. |
| `per_layer` | each layer re-draws its assignment, seeded by the layer index, so imbalance decorrelates with depth. Forces per-layer routing and per-layer expert predictions in `detailed`; `stage_group_scaled` falls back to that path. `first_layer_scaled` deliberately overrides this scope with one batch-shared representative draw. |

Omitting the field keeps the behavior that the mode and distribution implied
before it existed: `uniform_legacy` and `simulation`/`balanced` resolve to
`shared`, everything else to `per_layer`. The normalized config always states
the resolved value.

`skewed` and `zipf` never consumed the layer seed, so selecting `shared` for
them changes no number — it only stops the predictor recomputing an assignment
it already has. That is worth doing: at K3 (93 layers, 896 experts) each
redundant draw sorts 896 weights.

For Kimi K3, use `mode="uniform_random"`, `distribution="balanced"`, and
`layer_scope="shared"` to represent no-aux-loss routing as balanced in
expectation but stochastic for every finite batch. One token-level top-k draw
includes the batch ID in the deterministic seed and is reused by every MoE
layer, so two same-sized batches do not repeat the same expert histogram
without expanding the DES into one routing/barrier sequence per layer. Routing diagnostics
also expose `lane_routed_tokens`, `lane_active_experts`, and
`lane_unique_tokens`; the last field counts one source token once per
destination EP lane and can therefore drive source-aware A2A traffic models.

## Determinism

One configuration and workload always produce the same output: the same
event order, the same timestamps, and the same MoE routing. This holds
across platforms and standard-library implementations, so a run on Linux and
a run on Windows are comparable.

The routing RNG is a `std::mt19937_64` engine with conversions implemented
in-tree rather than taken from `<random>`'s distributions, whose sequences
are implementation-defined. Routing golden vectors in
`tests/execution_time_predictor/` pin that behavior; regenerate them only
for a deliberate, documented routing change.

`moe_routing.mode="uniform_random"` now draws each input token's `router_topk`
experts **without replacement**, as a router does, instead of taking
`input_tokens * router_topk` independent samples. Expert loads for
`router_topk > 1` therefore differ from revisions before this change and are
not comparable at the same seed. The golden vectors use `router_topk = 1`,
where both formulations consume the RNG identically, so they still hold and
did not need regenerating.

Earlier revisions reproduced NumPy's PCG64 bit stream so results could be
diffed against the Python simulator. That correspondence is no longer
maintained, and the Python differential harness has been retired. Analytical
MoE runs using `moe_routing.mode="uniform_random"` or
`distribution="random"` are therefore not comparable across that change at
the same seed; the other routing modes never consumed the RNG and are
unaffected.
