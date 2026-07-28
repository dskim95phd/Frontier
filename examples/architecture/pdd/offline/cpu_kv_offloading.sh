#!/bin/bash
# Sequential PDD prefill-side CPU KV-cache offloading example.
#
# The third session forces the five-block prefill GPU cache to evict an older
# two-block prefix before follow-up turns arrive. Those full prefill blocks are
# restored from the cache target's local CPU DRAM. Decode-created tokens are
# still recomputed by prefill on the next turn.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export PREFILL_REPLICAS="${PREFILL_REPLICAS:-1}"
export DECODE_REPLICAS="${DECODE_REPLICAS:-1}"
export PREFIX_CACHING_KEY_MODE=session
export TRACE_FILE="${TRACE_FILE:-$REPO_ROOT/examples/fixtures/cpu_kv_offload_session_trace.csv}"
export EXPECTED_TRACE_REQUESTS="${EXPECTED_TRACE_REQUESTS:-6}"
export NUM_BLOCKS="${NUM_BLOCKS:-5}"
export MAX_TOKENS="${MAX_TOKENS:-128}"
export RUN_ID="${RUN_ID:-cpu_kv_offloading}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

exec bash "$SCRIPT_DIR/moe_prefix_caching.sh" -- \
  --vllm_v1_scheduler_config_kv_cache_dtype "$KV_CACHE_DTYPE" \
  --cpu_kv_cache_config_enable \
  --cpu_kv_cache_config_capacity_bytes "${CPU_KV_CAPACITY_BYTES:-137438953472}" \
  --cpu_kv_cache_config_write_bandwidth_gbps "${CPU_KV_WRITE_BANDWIDTH_GBPS:-64}" \
  --cpu_kv_cache_config_write_latency_ms "${CPU_KV_WRITE_LATENCY_MS:-0.01}" \
  --cpu_kv_cache_config_read_bandwidth_gbps "${CPU_KV_READ_BANDWIDTH_GBPS:-64}" \
  --cpu_kv_cache_config_read_latency_ms "${CPU_KV_READ_LATENCY_MS:-0.01}" \
  --cpu_kv_cache_config_capacity_pressure_policy "${CPU_KV_CAPACITY_POLICY:-prefix_fit}"
