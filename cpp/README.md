# Frontier C++ Core

This directory contains the ID-based deterministic C++ simulation core being
ported from Frontier's Python behavioral oracle.

The current implementation covers the complete Step 1 foundation, Step 2
single-target scheduler, and Step 2.5 dense co-location parallel path:

- CMake/CTest skeleton;
- deterministic `(time, sequence)` event queue;
- versioned normalized configuration;
- strict session-oriented CSV workload input;
- versioned JSON plus request-metrics CSV output contracts;
- Rubin roofline and dense Llama-2-7B TP1/2/4/8 execution formulas;
- simple analytical communication formulas exercised at 72 devices;
- dense KV-cache size and transfer formulas;
- the frozen scheduler-free schema-v1 lifecycle;
- canonical schema-v2 Request and Batch entities;
- FCFS vLLM V1-style continuous batching and token budgeting;
- ordinary non-prefix KV-block admission and watermark accounting;
- monolithic prefill/decode progression and chunked prefill;
- recompute preemption with stale-completion protection;
- functional Python-aligned global, cluster, replica, and replica-stage
  scheduler ownership;
- explicit request-arrival, global/cluster/replica schedule, stage
  arrival/schedule/end, cluster-end, and global-end events;
- deterministic multi-replica and per-replica DP routing;
- PP stage serialization, cross-stage overlap, multi-in-flight batches, and
  Python-compatible PP terminal resource release;
- fixed per-stage and analytical stage execution models with dense compute,
  TP collective, and PP transfer breakdowns;
- schema-v3 target, batch-stage, scheduler, and event traces;
- a normalized config/workload CLI that emits deterministic JSON;
- independent config and workload normalization commands; and
- a pytest-driven Python/C++ differential gate.

It does not yet implement session prefix caching, physical cache-block
identity, or PDD handoff. The topology-aware `astra_sim_analytical` placement
model is also deferred; schema v3 currently supports dense
Rubin/Llama-2-7B/FP16 with validated TP 1/2/4/8, PP 1/2/4/8, DP, multiple
replicas, and the simple analytical communication backend.

## Build in WSL

Use the existing Windows worktree as the source and a WSL-native out-of-tree
build directory:

```bash
cd /mnt/c/Users/jklpr/Desktop/project/Frontier-cxx-port
export PATH="$HOME/frontier-tools/bin:$PATH"

cmake -S cpp \
  -B "$HOME/frontier-build/cxx-port" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build "$HOME/frontier-build/cxx-port"
ctest \
  --test-dir "$HOME/frontier-build/cxx-port" \
  --output-on-failure
```

The initial toolchain baseline is C++20, GCC 11 or newer (or Clang 14 or
newer), CMake 3.24 or newer, and Ninja 1.10 or newer.

## Run the Foundation Lifecycle

The Step 1 lifecycle consumes the normalized schemas and writes output JSON to
stdout:

```bash
"$HOME/frontier-build/cxx-port/frontier_sim" \
  --config cpp/tests/fixtures/config/minimal_foundation_colocation.json \
  --workload cpp/tests/fixtures/workloads/session_prefix.csv
```

This is intentionally the frozen schema-v1 foundation path. Every arrival
schedules one completion at a fixed 1 ms delay, independently of token counts.

## Run the Step 2 Co-location Scheduler

Run the fixed-latency canonical scheduler fixture:

```bash
"$HOME/frontier-build/cxx-port/frontier_sim" \
  --config cpp/tests/fixtures/config/step2_fixed_colocation.json \
  --workload cpp/tests/fixtures/workloads/step2_single_request.csv
```

Run the same workload with analytical batch timing:

```bash
"$HOME/frontier-build/cxx-port/frontier_sim" \
  --config cpp/tests/fixtures/config/step2_analytical_colocation.json \
  --workload cpp/tests/fixtures/workloads/step2_single_request.csv
```

Schema v2 supports one dense co-location replica, one DP target, one pipeline
stage, the FCFS vLLM V1 scheduler, explicit ordinary KV-block capacity, and
fixed or Rubin analytical execution. Offline and online are metadata modes
over the same trace-driven event semantics.

## Run Step 2.5 Dense Parallel Co-location

Run the fixed online fixture with two replicas, DP2, TP2, and PP2:

```bash
"$HOME/frontier-build/cxx-port/frontier_sim" \
  --config cpp/tests/fixtures/config/step25_fixed_parallel_colocation.json \
  --workload cpp/tests/fixtures/workloads/step25_parallel.csv
```

Run the analytical DP2/TP4/PP2 fixture:

```bash
"$HOME/frontier-build/cxx-port/frontier_sim" \
  --config cpp/tests/fixtures/config/step25_analytical_parallel_colocation.json \
  --workload cpp/tests/fixtures/workloads/step25_parallel.csv
```

Schema v3 uses production Python arrival semantics: online mode replays trace
arrival times, while offline mode places every request in the monolithic
scheduler at time zero and begins with one global schedule event.

The CLI also exposes read-only normalization commands used by parity tests:

```bash
"$HOME/frontier-build/cxx-port/frontier_sim" \
  --normalize-config cpp/tests/fixtures/config/minimal_foundation_colocation.json

"$HOME/frontier-build/cxx-port/frontier_sim" \
  --normalize-workload cpp/tests/fixtures/workloads/session_prefix.csv
```

## JSON Dependency

The config and output contracts use `nlohmann/json` 3.11.3.

CMake resolves the dependency in this order:

1. an installed `nlohmann_json` 3.11.3 package;
2. `FRONTIER_NLOHMANN_JSON_SOURCE_DIR`; or
3. the pinned upstream release archive when
   `FRONTIER_FETCH_DEPENDENCIES=ON`.

The release URL is protected by a checked-in SHA-256. For an offline build,
prepare the exact source release and configure with:

```bash
cmake -S cpp \
  -B "$HOME/frontier-build/cxx-port-offline" \
  -G Ninja \
  -DFRONTIER_FETCH_DEPENDENCIES=OFF \
  -DFRONTIER_NLOHMANN_JSON_SOURCE_DIR=/path/to/nlohmann-json-3.11.3
```

## Normalized Configuration Schema v1

All fields are required. Unknown fields and unsupported values are rejected.

```json
{
  "schema_version": 1,
  "run_id": "example",
  "simulation_mode": "offline",
  "system_architecture": "co-location",
  "enable_parallel_clusters": false,
  "prefix_cache": {
    "enabled": true,
    "key_mode": "session"
  }
}
```

Supported values:

- `simulation_mode`: `offline`, `online`;
- `system_architecture`: `co-location`, sequential `pd-disaggregation`;
- `enable_parallel_clusters`: must be `false`; and
- `prefix_cache.key_mode`: must be `session`.

The C++ MVP explicitly rejects `block_hash`, `block_hash_ids`, parallel PDD,
and unsupported architectures.

## Normalized Configuration Schema v2

Schema v2 selects the Step 2 scheduler. All fields are required and unknown
fields are rejected:

```json
{
  "schema_version": 2,
  "run_id": "step2-example",
  "simulation_mode": "offline",
  "system_architecture": "co-location",
  "enable_parallel_clusters": false,
  "prefix_cache": {
    "enabled": false,
    "key_mode": "session"
  },
  "scheduler": {
    "type": "vllm_v1",
    "scheduling_policy": "fcfs",
    "batch_size_cap": 8,
    "max_tokens_in_batch": 128,
    "enable_preemption": false,
    "enable_chunked_prefill": false,
    "long_prefill_token_threshold": 0,
    "block_size": 16,
    "num_blocks": 128,
    "watermark_blocks_fraction": 0.0,
    "num_preallocate_tokens": 0
  },
  "execution_model": {
    "type": "fixed",
    "batch_latency_ms": 1.0
  }
}
```

The analytical execution object is:

```json
{
  "type": "analytical",
  "device": "rubin",
  "model": "llama2-7b",
  "precision": "fp16",
  "tensor_parallel_size": 8,
  "num_layers": 32,
  "network_bandwidth_gbps": 400.0,
  "network_latency_us": 1.0,
  "intra_node_bandwidth_gbps": 14400.0
}
```

Schema v2 rejects PDD, parallel clusters, prefix caching, block-hash keys,
priority scheduling, and analytical model values outside the implemented
surface before entering the event loop.

## Normalized Configuration Schema v3

Schema v3 adds authoritative topology and cluster-routing objects:

```json
{
  "schema_version": 3,
  "run_id": "step25-example",
  "simulation_mode": "online",
  "system_architecture": "co-location",
  "enable_parallel_clusters": false,
  "parallelism": {
    "num_replicas": 2,
    "tensor_parallel_size": 2,
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
  }
}
```

For analytical execution, TP comes only from `parallelism`; `num_layers` must
be 32 and divisible by PP. Each `(replica_id, dp_id)` owns independent request
queues, in-flight batches, KV accounting, and PP stage schedulers. The current
cluster policy matches Python's replica-first batch distribution followed by
an even split across each replica's DP lanes.

## Normalized Workload CSV

Required columns:

```text
arrived_at,num_prefill_tokens,num_decode_tokens
```

Optional columns:

```text
session_id,session_turn_index
```

Example:

```csv
arrived_at,num_prefill_tokens,num_decode_tokens,session_id,session_turn_index
0,32,8,7,0
1.5,16,4,7,1
```

Contract:

- request IDs are assigned from zero in row order;
- `arrived_at` is finite, nonnegative, and measured in seconds;
- token counts are positive integers;
- session and turn IDs are nonnegative integers;
- `session_turn_index` requires `session_id`;
- session prefix caching requires a session ID on every request;
- arrivals are nondecreasing within each session;
- explicit turn indices are strictly increasing within each session;
- duplicate, missing, and unknown columns are rejected; and
- normalized input does not support quoted fields.

## Output Schema v1

The JSON output uses stable insertion order and contains:

```text
schema_version
run
completed_request_ids
requests
event_trace
analytical_diagnostics
```

Timestamps use seconds and have an `_s` suffix. Latencies use milliseconds and
have an `_ms` suffix. Run metadata contains one of:

- `metrics_semantics: canonical` for scheduler-derived metrics; or
- `metrics_semantics: foundation-placeholder` for the Step 1E fixed-latency
  lifecycle.

Canonical TTFT is:

```text
request arrival -> request prefill completion
```

In the foundation lifecycle only, `prefill_completed_at` and `completed_at`
both refer to the single placeholder completion event, so TTFT equals E2E.

The request CSV header is frozen to:

```csv
request_id,arrived_at_s,prefill_completed_at_s,completed_at_s,ttft_ms,e2e_ms
```

JSON fields use deterministic insertion order. CSV floating-point values use
`max_digits10` precision in the classic locale.

## Output Schema v2

Schema v2 retains the stable top-level run/request/event fields and adds:

```text
requests[].first_scheduled_at_s
requests[].first_token_completed_at_s
requests[].scheduling_delay_ms
requests[].num_processed_tokens
requests[].preemption_count
batches[]
scheduler_trace[]
```

Batch rows contain stable batch/iteration IDs, ordered request IDs and token
counts, prefill/decode totals, scheduling/completion timestamps, and predicted
execution time. Scheduler rows preserve iteration order, production decision
order, token budgets, available block counts, queue counts, and emitted batch
order. Analytical runs also contain one component summary per batch.

Successful schema-v2 termination requires every request to complete and all
waiting/running state, in-flight work, and ordinary KV allocations to be
empty. Unschedulable workloads fail with a deterministic quiescence report.

## Output Schema v3

Schema v3 retains schema-v2 request and batch records and adds target IDs,
`num_pipeline_stages`, `batch_stages[]`, and target-local scheduler fields.
Each stage record contains arrival/start/completion timestamps plus dense,
TP-communication, PP-communication, and total milliseconds. The event trace
uses the explicit nine-event co-location pipeline.

Successful termination requires all global, cluster, replica, KV, in-flight,
and stage-queue state to be empty.

## Analytical Golden Fixture

`tests/golden/generate_analytical_golden.py` and
`generate_step2_analytical_batch_golden.py` invoke the production Python
analytical implementation as the numerical oracle. Their JSON outputs are
checked in under `tests/fixtures/analytical/` and consumed by C++ tests.

From the repository root:

```bash
python cpp/tests/golden/generate_analytical_golden.py
python cpp/tests/golden/generate_step2_analytical_batch_golden.py
```

Execution, communication, and transfer latency comparisons use `1e-12`
absolute and relative tolerances. Integral byte counts are compared exactly.

## Python/C++ Differential Gate

The parity gate compares normalized config and workload behavior, rejection
cases, the frozen foundation lifecycle, Step 2 scheduler decisions and state,
equal-time event ordering, analytical batch golden values, and Step 2.5
production Python `Simulator` runs covering online multi-replica/DP/PP,
offline PP4 drain, analytical TP/PP stage components, and target-local KV
pressure/preemption. Step 2.5 comparisons include every target-local scheduler
iteration, including decisions, token budgets, KV blocks, queue counts, and
batch membership. The expanded matrix adds 14 topology/scheduler combinations;
all 14 pass exactly, including complex fixed and analytical PP4/PP8 terminal
drains. From Windows PowerShell, invoke the WSL build with:

```powershell
$env:FRONTIER_CPP_BINARY = "/home/dskim/frontier-build/cxx-port/frontier_sim"
$env:FRONTIER_CPP_RUNNER = '["wsl","-d","Ubuntu","--"]'
$env:FRONTIER_CPP_PATH_STYLE = "wsl"
python -m pytest cpp/tests/parity -q
```

When Python runs natively in Linux or WSL, only the binary is required:

```bash
export FRONTIER_CPP_BINARY="$HOME/frontier-build/cxx-port/frontier_sim"
python -m pytest cpp/tests/parity -q
```

The harness parses JSON before comparison, uses `1e-12` absolute and relative
tolerances for floats, and compares IDs, counts, and event order exactly. On a
failure it preserves Python output, C++ stdout/stderr, and a field-level
difference. Set `FRONTIER_PARITY_ARTIFACT_DIR` to choose a persistent artifact
directory.
