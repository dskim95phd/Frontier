# CPU KV-Cache Restore Head-of-Line Deadlock

## Status

- **State:** Open
- **Severity:** P1
- **Discovered:** 2026-07-28
- **Affected architecture:** Sequential `pd-disaggregation`
- **Affected scheduler:** `vllm_v1`
- **Affected cache mode:** Session prefix caching with CPU KV-cache offloading
- **Primary trigger:** Small prefill GPU KV cache with multiple follow-up
  requests waiting for CPU-to-GPU restore

## Summary

When several follow-up requests are waiting on the same prefill cache target,
a request whose CPU restore has completed can retain reserved GPU blocks while
it sits behind another request that has not started restore.

If the request at the front of the waiting queue cannot reserve enough GPU
blocks for its restore, the scheduler stops scanning the queue. It therefore
never reaches the restore-ready request behind it. Because the ready request
already owns GPU prefix and suffix reservations, the resource shortage cannot
resolve by itself. The event queue eventually drains while requests remain in
the scheduler.

The deadlock is specific to the CPU-offload admission path:

- the same workload completes when CPU offloading is disabled;
- every individual request fits in the configured GPU cache;
- the workload completes when arrivals are sufficiently serialized; and
- the workload completes when the GPU cache is made large enough.

This is distinct from:

- overlapping restores for the **same** session; and
- the PDD DECODE MoE synchronization bug involving `PP > 1`.

The reproducer in this document uses different session IDs and keeps the
decode cluster at `DP=TP=PP=EP=1`.

## User-Visible Impact

A supported sequential-PDD simulation can terminate with:

```text
Sequential simulation ended with non-empty scheduler state
```

The terminal state has the following characteristics:

- the simulator event queue is empty;
- the prefill scheduler still contains waiting requests;
- at least one restore-ready request owns GPU blocks;
- no future event can schedule that ready request or release its reservation;
  and
- request/system metrics are not written because the simulation does not
  complete.

This is a P1 issue because small GPU capacity and concurrent CPU restores are
core use cases for CPU KV-cache offloading, not malformed configurations.

## Minimal Reproduction

### Prerequisites

Run from the repository root with the development environment activated:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:WANDB_DISABLED = "true"
$env:VIDUR_DISABLE_WANDB = "1"
```

The reproducer uses the existing E2E argument builders so it stays aligned
with the supported PDD CPU-offload configuration.

### Reproduction script

The workload has four independent sessions and three turns per session.
Each turn appends 16 input tokens and produces one output token. Turns within
the same round arrive 100 ms apart. The prefill GPU cache has five blocks, so
every request fits individually.

```powershell
@'
import runpy
import tempfile
from pathlib import Path

import pandas as pd

ns = runpy.run_path("tests/e2e/test_session_prefix_cache_runtime.py")
output_root = Path(tempfile.mkdtemp(prefix="frontier_restore_hol_"))
trace_file = output_root / "restore_hol_trace.csv"

rows = []
orders = [
    [0, 1, 2, 3],
    [3, 2, 1, 0],
    [0, 1, 2, 3],
]
for round_id, order in enumerate(orders):
    for position, session_index in enumerate(order):
        rows.append(
            {
                "arrived_at": round_id * 100 + position * 0.1,
                "num_prefill_tokens": 16,
                "num_decode_tokens": 1,
                "session_id": 1000 + session_index,
            }
        )

pd.DataFrame(rows).to_csv(trace_file, index=False)


def run_case(label, cpu_offload_enabled):
    if cpu_offload_enabled:
        cache_args = ns["_cpu_offload_args"]()
    else:
        cache_args = [
            "--vllm_v1_scheduler_config_num_blocks",
            "5",
        ]

    try:
        request_metrics, _ = ns["_run_simulation"](
            output_root=output_root / label,
            run_id=label,
            architecture_args=[
                *ns["_pdd_args"](),
                *cache_args,
            ],
            trace_file=trace_file,
            key_mode="session",
        )
    except AssertionError as error:
        deadlocked = (
            "Sequential simulation ended with non-empty scheduler state"
            in str(error)
        )
        print(label, "FAIL", "deadlock=", deadlocked)
        return

    print(label, "PASS", "completed_requests=", len(request_metrics))


run_case("cpu_off_control", cpu_offload_enabled=False)
run_case("cpu_on_reproducer", cpu_offload_enabled=True)
print("artifacts:", output_root)
'@ | .\.venv\Scripts\python.exe -
```

### Expected result before the fix

```text
cpu_off_control PASS completed_requests= 12
cpu_on_reproducer FAIL deadlock= True
```

The CPU-offload-disabled run is the control. It demonstrates that five GPU
blocks are sufficient for the workload when restore reservations are not
involved.

## Reproduction Matrix

The issue was reproduced with four, eight, twelve, and sixteen sessions.
With `num_blocks=5`, every CPU-offload-enabled case deadlocked while every
CPU-offload-disabled control completed.

| Sessions | Turns per session | GPU blocks | CPU offload OFF | CPU offload ON |
|---:|---:|---:|---|---|
| 4 | 3 | 5 | Pass | Deadlock |
| 8 | 3 | 5 | Pass | Deadlock |
| 12 | 3 | 5 | Pass | Deadlock |
| 16 | 3 | 5 | Pass | Deadlock |

Holding the four-session trace constant and changing only GPU capacity
produced the following boundary:

| GPU blocks | Result with CPU offloading |
|---:|---|
| 5 | Deadlock |
| 6 | Deadlock |
| 8 | Deadlock |
| 10 | Pass |
| 12 | Pass |
| 16 | Pass; all useful prefixes remained on GPU |

The exact pass boundary depends on request shape and queue order. It is not a
safe minimum-capacity recommendation. It only shows that the failure tracks
aggregate restore reservations rather than whether one request fits.

Arrival spacing also changes the result:

| Intra-round arrival spacing | Result |
|---:|---|
| 1 ms | Deadlock |
| 10 ms | Deadlock |
| 100 ms | Deadlock |
| 1 s | Pass |

The dummy predictor makes one prefill stage take approximately 390 ms in this
reproducer. At one-second spacing, requests are sufficiently serialized that
the problematic queue state does not form.

## Failure Timeline

A representative event trace is:

```text
t=100.000
  follow-up A enters PREFILL
  A starts/resumes prefill work

t=100.100 ... 100.300
  follow-ups B, C, and D enter the same prefill target
  scheduling attempts produce no batch while earlier work is active

t≈101.170
  one CPU restore starts and completes
  its request now owns restored-prefix and suffix GPU reservations
  CPUKVCacheRestoreEndEvent emits ReplicaScheduleEvent

t≈101.171
  ReplicaScheduleEvent returns no batch
  the restored request remains behind another waiting request

later
  no CPU restore, prefill execution, or reservation-release event remains
  event queue drains
  scheduler still contains unfinished requests
```

The final simulator diagnostic reports:

```text
event_queue_length: 0
global_scheduler_is_empty: false
message: "Sequential simulation ended with non-empty scheduler state"
```

## Root Cause

### 1. Restore admission reserves GPU blocks before transfer

`VllmV1EngineReplicaScheduler._begin_cpu_kv_cache_restore()` calls
`reserve_tiered_prefix()`. This attaches the existing GPU hit blocks and
allocates unpublished GPU pages for:

- CPU-resident prefix blocks being restored; and
- the uncached prompt suffix.

The reservation is intentionally retained after the H2D transfer completes.
It must remain valid until the request is admitted and computes its suffix.

Relevant files:

- `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`
- `frontier/kv_cache/base_kv_cache_manager.py`

### 2. A request that starts restore is moved out of its queue position

In `_schedule_waiting_requests()`:

```python
if self._begin_cpu_kv_cache_restore(...):
    waiting_queue.popleft()
    skipped_waiting_requests.append(request)
    continue
```

The request is removed from its original FCFS position and placed in
`skipped_waiting_requests`.

### 3. FCFS reinsertion appends the request to the tail

At the end of the scheduling iteration:

```python
# prepend skipped queue back to waiting queue.
...
waiting_queue.extend(skipped_waiting_requests)
```

Despite the comment saying "prepend", `extend()` appends the skipped requests.
A restored request can therefore sit behind requests that have not yet
reserved restore space.

### 4. Restore admission failure stops the entire queue scan

For a later request at the front of the queue:

```python
if tiered_plan.needs_restore:
    if self._begin_cpu_kv_cache_restore(...):
        ...
        continue
    break
```

If the earlier ready request is holding most of the small GPU pool,
`_begin_cpu_kv_cache_restore()` returns `False`. The unconditional `break`
prevents the scheduler from reaching the restore-ready request behind it.

### 5. No event can resolve the cycle

The restore-ready request needs scheduling to consume its plan and eventually
release its blocks. The request at the front needs those blocks before it can
start restore. Since the scheduler does not scan past the blocked request,
neither side can progress.

The dependency cycle is:

```text
restore-ready request A
  holds GPU reservation
  waits behind request B

request B
  needs additional GPU reservation
  causes queue scan to break

scheduler
  never reaches A
  emits no useful future event
```

## Why Existing Tests Did Not Catch It

The initial E2E CPU-offload fixture has:

- three sessions;
- two turns per session;
- one-second spacing between requests within a round; and
- only one restore wave.

That is sufficient for basic metric and transfer validation, but it does not
create sustained contention between:

- a restore-ready request holding GPU blocks; and
- a different non-ready request ahead of it.

Topology stress over TP, DP, PP, and EP also does not substitute for cache
churn stress. Most topology cases used the same short three-session fixture.

## Additional Churn Results

The following cases were run after serializing arrivals enough to avoid the
known head-of-line deadlock. These results validate the underlying eviction,
truncation, accounting, and transfer-queue mechanics independently of the
open scheduling issue.

### Repeated GPU pressure with a large CPU tier

Configuration:

- 16 sessions;
- 3 turns per session;
- 48 completed requests;
- prefill GPU cache: 5 blocks;
- CPU cache: 1 GiB.

Result:

- 48 offload operations;
- 29 restore operations;
- 44 restored blocks;
- no CPU eviction;
- no final reserved blocks; and
- request/system byte totals matched.

### Repeated CPU eviction

Configuration:

- same 16-session, 3-turn workload;
- prefill GPU cache: 5 blocks;
- CPU cache: 8 blocks.

Result:

- 48 offload operations;
- 86 incrementally offloaded blocks;
- 4 restore operations;
- 38 evicted sessions;
- 78 evicted blocks;
- resident blocks never exceeded 8;
- no final reserved blocks; and
- request/system byte totals matched.

### CPU cache smaller than a growing snapshot

Configuration:

- same 16-session, 3-turn workload;
- prefill GPU cache: 5 blocks;
- CPU cache: 1 block.

Result:

- all 48 requests completed;
- 46 D2H offload transfers;
- 45 evicted sessions and blocks;
- 32 truncated offload attempts;
- no restore hits;
- resident blocks never exceeded 1; and
- no final reserved blocks.

Two completed requests did not schedule a new D2H copy because the session's
existing one-block CPU frontier occupied the entire CPU cache. The session
itself is excluded from eviction while extending its snapshot, so the missing
block could not be admitted and the operation terminated as a truncation.
This is expected under the current partial-admission policy.

### Sharded target churn

Configuration:

- 32 sessions;
- 3 turns per session;
- 96 completed requests;
- prefill `DP=4`;
- four independent CPU cache targets;
- GPU cache: 5 blocks per target;
- CPU cache: 2 blocks per target.

Result:

- aggregate CPU capacity: 8 blocks;
- 92 offload transfers;
- 84 evicted sessions;
- 140 evicted blocks;
- 32 truncated offloads;
- no final reserved blocks;
- all 96 PDD transfers completed; and
- request/system byte totals matched.

### Slow serialized DRAM transfer queue

Configuration:

- 64 cold sessions followed by 8 simultaneous follow-ups;
- GPU cache: 32 blocks;
- CPU cache: 1 GiB;
- D2H and H2D bandwidth: 0.01 Gbps;
- D2H and H2D fixed latency: 5 ms.

Result:

- 72 completed requests;
- 72 offload operations;
- 8 restore operations;
- about 1,224 seconds of cumulative offload queue time;
- about 23.6 seconds of cumulative restore queue time;
- 32 peak CPU-reserved blocks;
- zero final reserved blocks; and
- request/system byte totals matched.

This validates transfer serialization and queue-time accounting when GPU
capacity is large enough to avoid the open restore-admission deadlock.

## Proposed Fix Direction

The fix must guarantee that a request holding a completed restore reservation
cannot be blocked behind a request that needs additional restore capacity.

The preferred scheduling contract is:

1. Process `_cpu_restore_ready_plans` before starting new restores on the same
   cache target.
2. Preserve the original scheduler order among restore-ready requests.
3. If a non-ready request cannot reserve restore space, continue scanning for
   restore-ready work instead of unconditionally breaking.
4. Start no additional restore whose reservation would prevent already-ready
   work from being admitted.
5. Keep restore lease, GPU block, request allocation, and metric finalization
   exactly-once under success, cancellation, and stale events.

Possible implementations include:

- reinserting skipped restore requests at the front rather than the tail;
- explicitly partitioning the queue into restore-ready, restore-pending, and
  not-started requests;
- adding a ready-request fast lane before normal FCFS admission; or
- limiting in-flight GPU restore reservations and waking admission when the
  current ready request is consumed.

Simply increasing GPU capacity is not a fix. It only moves the failure
boundary.

## Required Regression Tests

### 1. Minimal head-of-line reproduction

Add an E2E test with:

- four distinct sessions;
- three turns per session;
- 100 ms intra-round arrivals;
- GPU cache of five blocks;
- large CPU cache; and
- CPU offloading enabled.

Acceptance:

- all 12 requests complete;
- at least one CPU restore occurs;
- final GPU/CPU reservations are zero; and
- the simulation does not exit with non-empty scheduler state.

### 2. CPU-offload-disabled control

Run the same trace with CPU offloading disabled.

Acceptance:

- all 12 requests complete; and
- the result proves each request fits the GPU configuration.

### 3. Repeated CPU eviction

Use at least 16 sessions and three turns with an eight-block CPU tier.

Acceptance:

- multiple session and block evictions occur;
- multiple offload generations for the same session complete;
- resident plus reserved blocks never exceed capacity;
- final reserved blocks are zero; and
- request and system byte totals match.

### 4. Snapshot truncation

Use a one-block CPU tier with sessions whose reusable frontier grows past one
block.

Acceptance:

- truncation is reported;
- CPU hits never cross a missing block;
- no negative/free-capacity inconsistency appears; and
- final reservations are zero.

### 5. Slow transfer queues

Use low H2D/D2H bandwidth and simultaneous requests.

Acceptance:

- both offload and restore queue time are positive;
- transfers remain serialized according to the configured engine;
- source GPU hold time is recorded;
- all requests complete; and
- no reservation leak remains.

### 6. Multiple cache targets

Run the churn trace with multiple prefill DP targets and/or replicas.

Acceptance:

- target count equals `num_replicas * attn_data_parallel_size`;
- capacity is enforced independently per target;
- session affinity is preserved; and
- aggregate metrics equal the sum of per-target metrics.

### 7. Separate same-session concurrency test

Overlapping restores for one session require a separate test because this
document's P1 reproducer deliberately uses distinct sessions. Fixing the
head-of-line issue must not be treated as fixing same-session restore
generation and ownership races.

## Acceptance Criteria for Closing This Issue

This issue can be closed when:

1. the minimal CPU-on reproducer completes with GPU `num_blocks=5`;
2. the CPU-off control continues to pass;
3. all requests individually and collectively release GPU reservations;
4. CPU restore leases return to zero;
5. request/system byte and operation metrics remain consistent;
6. the eight-block CPU eviction and one-block truncation stresses pass;
7. the slow H2D/D2H queue stress passes;
8. multi-target DP stress passes; and
9. the existing targeted CPU KV-cache unit and E2E suites pass.

Recommended targeted suite:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/unit/test_cpu_kv_cache_manager.py `
  tests/unit/test_cpu_kv_cache_stress.py `
  tests/unit/test_tiered_cpu_kv_cache.py `
  tests/e2e/test_session_prefix_cache_runtime.py `
  -q
```

## Notes

- No production fix is included in this document.
- Stress artifacts were generated under the operating system temporary
  directory and are not repository fixtures.
- The numerical queue-time totals above are cumulative simulator metrics, not
  wall-clock test duration.
