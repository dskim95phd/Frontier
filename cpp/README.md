# Frontier C++ Core

This directory contains the ID-based deterministic C++ simulation core being
ported from Frontier's Python behavioral oracle.

The current implementation covers the complete Step 1 foundation:

- CMake/CTest skeleton;
- deterministic `(time, sequence)` event queue;
- versioned normalized configuration;
- strict session-oriented CSV workload input;
- versioned JSON plus request-metrics CSV output contracts;
- Rubin roofline and dense Llama-2-7B TP8 execution formulas;
- simple analytical communication formulas exercised at 72 devices;
- dense KV-cache size and transfer formulas;
- scheduler-free request arrival/completion lifecycle;
- a normalized config/workload CLI that emits deterministic JSON;
- independent config and workload normalization commands; and
- a pytest-driven Python/C++ differential gate.

It does not yet implement scheduling, KV-cache behavior, or PDD handoff. The
topology-aware `astra_sim_analytical` placement model is also deferred; Step
1D uses the release example default `analytical` backend.

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

This is intentionally not a scheduler simulation. Every arrival schedules one
completion at a fixed 1 ms delay, independently of token counts. It accepts
only co-location with prefix caching disabled. Sequential PDD, scheduling, and
prefix caching are introduced by later milestones.

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

## Analytical Golden Fixture

`tests/golden/generate_analytical_golden.py` invokes the Python implementation
as the numerical oracle. Its JSON output is checked in as
`tests/fixtures/analytical/analytical_v1.json` and consumed by the C++ test.

From the repository root:

```bash
python cpp/tests/golden/generate_analytical_golden.py
```

Execution, communication, and transfer latency comparisons use `1e-12`
absolute and relative tolerances. Integral byte counts are compared exactly.

## Python/C++ Differential Gate

The parity gate compares normalized config and workload behavior, rejection
cases, foundation lifecycle output, equal-time event ordering, and analytical
golden values. From Windows PowerShell, invoke the WSL build with:

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
