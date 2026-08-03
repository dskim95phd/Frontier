# C++ Port Step 4: Analytical Session Prefix Cache

## Status

Implemented. The C++ vLLM V1 replica scheduler supports an analytical,
session-scoped GPU prefix cache for co-location and sequential PDD.

This design deliberately models logical session-prefix ranges rather than
physical vLLM block identities. It is intended for large simulation sweeps,
where visiting every cached block on every turn would make a long append-only
conversation quadratic in its number of turns.

## Supported Contract

The implementation assumes:

- cache keys are `(session_id, logical_block_index)`;
- a session is append-only and does not branch;
- at most one turn of a session executes at a time;
- different sessions never share KV, even if their token content matches;
- only complete blocks are reusable;
- cache state is local to one `(replica_id, dp_id)` target;
- exact physical free-queue ordering and duplicate physical copies are not
  observable requirements.

`block_hash` cross-session sharing remains unsupported. A future physical
allocator model should be a separately named fidelity mode rather than being
mixed into this analytical state.

## Why the Physical Model Was Removed

The first design represented every physical block, its reference count, its
free-queue position, and the logical session key stored in that block. That is
useful for exact vLLM allocator parity, but it is unnecessarily expensive for
the supported session-only contract.

With a cumulative conversation of 100, 200, 300, ... blocks, touching the
entire reused prefix during lookup, admission, and release yields work
proportional to:

```text
100 + 200 + 300 + ... = O(number_of_turns^2)
```

The analytical manager stores one prefix frontier per session and one
allocation record per active request. Its memory does not grow with the number
of logical blocks in a session.

## State Model

Each cache-owning scheduler target owns one `ReplicaKVCacheManager`.

Conceptually, session state is:

```cpp
struct SessionCacheEntry {
    uint64_t resident_prefix_blocks;
    RequestId active_request;
    bool in_evictable_lru;
    list<SessionId>::iterator lru_position;
};
```

Active request state is:

```cpp
struct RequestKVAllocation {
    SessionId session_id;
    uint64_t allocated_blocks;
    uint64_t published_blocks;
};
```

The target also keeps aggregate counters:

```text
capacity_blocks
active_blocks
blank_blocks
evictable_blocks
resident_blocks
evictable_sessions
sessions_with_nonzero_frontier
```

The main capacity invariant is:

```text
active_blocks + blank_blocks + evictable_blocks == capacity_blocks
```

Resident blocks owned by an active request are included in `active_blocks`.
Resident blocks of inactive sessions are included in `evictable_blocks`.
`resident_blocks` is a cache-validity diagnostic and may overlap the active
count; it is not another capacity partition.

## Operations and Complexity

### Lookup

For a prompt containing `prompt_blocks` complete blocks:

```text
hit_blocks = min(prompt_blocks, session.resident_prefix_blocks)
cached_tokens = hit_blocks * block_size
```

Lookup is O(1). The simulator injects at most one turn of a session at a time,
so a successor cannot reach lookup while its predecessor is active.

### Admission

Admission converts the inactive session range into active ownership and
allocates only the additional suffix. It does not materialize a vector of
block IDs or increment a reference count once per hit block.

The vLLM V1 all-hit rule still demotes the final hit block so at least one
token block is recomputed. The analytical manager reuses that logical slot;
it does not create a duplicate physical copy.

Admission is O(1) when blank capacity is sufficient. If eviction is required,
it is O(number of victim sessions), not O(number of evicted blocks).

### Publication

After batch completion, the manager publishes:

```text
floor(num_processed_tokens / block_size)
```

bounded by the request's allocation. A partial final block is never exposed as
a hit. An append-only publication frontier cannot move backwards.

### Release and Preemption

Freeing request ownership is not GPU eviction.

On normal completion or preemption:

- active ownership is released;
- complete published KV remains GPU-resident;
- the inactive session is appended to the session LRU;
- uncomputed/private capacity becomes blank;
- cache validity changes only if a later allocation actually evicts the
  session range.

This preserves the required distinction between entering a free/reclaimable
queue and disappearing from GPU memory.

Release is O(1).

### Session-Suffix Eviction

The free queue is session-level. Allocation reclaims the oldest inactive
session first and subtracts only the number of suffix blocks required. If the
victim retains a nonzero prefix, it stays at the front of the LRU. If its
frontier reaches zero, its session entry is removed.

The simulator therefore preserves a contiguous valid prefix for every
session, while intentionally not reproducing physical block-level ordering.

## Preemption Replay Semantics

Preemption reconstructs the known context exactly as the Python request model
does:

```text
total_tokens = old_prefill_tokens + old_decode_tokens
new_prefill_tokens = num_processed_tokens
new_decode_tokens = total_tokens - num_processed_tokens
num_processed_tokens = 0
```

This applies both during prefill and after decode has begun. Cache lookup then
uses the reconstructed prefill length.

Example with block size 4:

```text
original prompt = 5
generated before preemption = 4
known context = 9
complete resident blocks = 2
replay lookup = 8 cached tokens + 1 recomputed token
remaining decode = 1 token
```

If preemption occurs partway through prefill, any complete published blocks
that remain resident are likewise restored; they are not recomputed merely
because request ownership was released.

## Same-Session Causality and Think Time

Turns are ordered by request materialization order within each session. Only a
session's first turn is initially injected. Terminal completion schedules its
successor at:

```text
predecessor_completion + successor.think_time
```

The successor therefore never occupies a scheduler waiting queue while its
predecessor is active. Replica schedulers do not build predecessor maps or scan
blocked turns.

For co-location, terminal completion occurs in the monolithic replica. For
sequential PDD it occurs only after DECODE completes, not when PREFILL or KV
transfer completes.

Offline PDD normally uses an all-arrivals barrier for DECODE scheduling. With
session prefix caching enabled, each completed transfer is scheduled
immediately; otherwise a blocked successor PREFILL and the barrier can form a
deadlock.

## Routing

Prefix-cache state is target-local. `sticky_round_robin` maps a `session_id`
to a stable `(replica_id, dp_id)` owner. This is required whenever the number
of cache-owning targets is greater than one.

The release keeps the existing policy constraint:

- co-location: supported sticky policy;
- sequential PDD: `sticky_round_robin` only;
- `sticky_lor`: rejected for PDD.

## Metrics and Output Contract

Aggregate JSON identifies the model explicitly:

```json
{
  "prefix_cache": {
    "storage_model": "analytical_session",
    "key_mode": "session",
    "block_size": 16,
    "successful_admissions": 0,
    "query_blocks": 0,
    "hit_blocks": 0,
    "hit_rate": 0.0,
    "evicted_blocks": 0,
    "evicted_sessions": 0
  }
}
```

Each target reports:

- `capacity_blocks`;
- `available_blocks`;
- `active_blocks`;
- `resident_blocks`;
- `evictable_blocks`;
- `evictable_sessions`;
- `sessions_with_nonzero_frontier`.

Physical allocator metrics are intentionally absent. Request-level query,
hit, and cached-token metrics continue to accumulate across preemption without
double-counting the same admission attempt.

## Implementation Files

Runtime:

- `cpp/frontier/kv_cache/replica_kv_cache_manager.h`
- `cpp/frontier/kv_cache/replica_kv_cache_manager.cc`
- `cpp/frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.*`
- `cpp/frontier/scheduler/cluster_scheduler/sticky_round_robin_cluster_scheduler.*`
- `cpp/frontier/events/global_batch_end_event.cc`
- `cpp/frontier/simulator/simulator.cc`
- `cpp/frontier/metrics/metrics_store.cc`
- `cpp/frontier/metrics/output_contract.*`

The physical `kv_cache_block`, `block_pool`, and `BlockId` implementation was
removed from the C++ target.

## Validation Coverage

Focused tests cover:

- cold, partial-hit, and full-hit admission;
- all-hit final-block recomputation without duplicate storage;
- normal release versus actual suffix eviction;
- preemption during prefill and decode replay;
- same-session serialization without unrelated-session HOL blocking;
- PDD successor gating until terminal DECODE completion;
- deterministic session-suffix LRU behavior;
- randomized 20,000-turn capacity churn;
- a 100,000,000-block logical prefix reused for 5,000 turns, proving metadata
  size and iteration count are independent of prefix length;
- JSON output fields and removal of physical allocator metrics;
- co-location and sequential PDD across dense/MoE and TP/DP/PP/EP variants;
- low-capacity preemption and eviction stress.

The full gate is Debug, Release, and AddressSanitizer CTest. Any change to
cache ownership, session ordering, or PDD wake-up behavior must run all three.

## Fidelity Limits

This model does not represent:

- physical block IDs;
- per-block reference counts;
- concurrent turns sharing one session prefix;
- duplicate copies from concurrent misses;
- exact vLLM free-block queue order;
- content-hash sharing across sessions;
- branching conversations;
- swap/offload block movement.

Those are explicit boundaries, not accidental omissions. If a future study
requires them, add a separate physical-cache fidelity mode and keep
`analytical_session` as the fast default.
