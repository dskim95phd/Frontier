#!/bin/bash
# Sequential PDD Rubin NVL72 analytical example with static Vera CPU slices.
# Prefill and decode each own one rack and use 36 GPUs by default; the
# remaining GPUs stay idle rather than being shared across clusters.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PREFILL_DEVICE=rubin
export DECODE_DEVICE=rubin
export PREFILL_REPLICAS="${PREFILL_REPLICAS:-18}"
export DECODE_REPLICAS="${DECODE_REPLICAS:-18}"
export PREFILL_RACKS="${PREFILL_RACKS:-1}"
export DECODE_RACKS="${DECODE_RACKS:-1}"
export PREFILL_ATTN_TP="${PREFILL_ATTN_TP:-2}"
export DECODE_ATTN_TP="${DECODE_ATTN_TP:-2}"
export PREFILL_MOE_TP="${PREFILL_MOE_TP:-1}"
export DECODE_MOE_TP="${DECODE_MOE_TP:-1}"
export PREFILL_MOE_EP="${PREFILL_MOE_EP:-2}"
export DECODE_MOE_EP="${DECODE_MOE_EP:-2}"
export ENABLE_DUMMY_MODE=false
export PREFIX_CACHING_KEY_MODE=session
export DECODE_CUDA_GRAPH_MODE=none
export RUN_ID="${RUN_ID:-vera_rubin_nvl72_cpu_offload_analytical}"

exec bash "$SCRIPT_DIR/moe_prefix_caching.sh" -- \
  --cluster_scheduler_config_type sticky_round_robin \
  --cluster_config_prefill_cluster_num_racks "$PREFILL_RACKS" \
  --cluster_config_decode_cluster_num_racks "$DECODE_RACKS" \
  --cluster_config_prefill_replica_config_network_device vera_rubin_nvl72_domain \
  --cluster_config_decode_replica_config_network_device vera_rubin_nvl72_domain \
  --execution_time_predictor_config_type analytical_roofline \
  --cc_backend_config_type astra_sim_analytical \
  --astra_sim_analytical_cc_backend_config_intra_server_topology Switch \
  --astra_sim_analytical_cc_backend_config_intra_server_bandwidth_gbps 14400 \
  --astra_sim_analytical_cc_backend_config_intra_server_latency_us 1.0 \
  --cpu_kv_cache_config_enable \
  --cpu_kv_cache_config_static_slice_per_gpu \
  --cpu_kv_cache_config_capacity_bytes_per_gpu 750000000000 \
  --cpu_kv_cache_config_dram_bandwidth_gbps_per_gpu 4800 \
  --cpu_kv_cache_config_c2c_bandwidth_gbps_per_gpu 3600 \
  "$@"
