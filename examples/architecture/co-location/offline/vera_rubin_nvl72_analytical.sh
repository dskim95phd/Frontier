#!/bin/bash
# Rubin NVL72 profile-free analytical example.
# Nine TP=8 replicas are packed into one cluster-exclusive NVL72 rack.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export DEVICE=rubin
export NUM_REPLICAS="${NUM_REPLICAS:-9}"
export NUM_RACKS="${NUM_RACKS:-1}"
export ATTN_TP="${ATTN_TP:-8}"
export ENABLE_DUMMY_MODE=false
export DECODE_CUDA_GRAPH_MODE=none
export CC_BACKEND=astra_sim_analytical
export RUN_ID="${RUN_ID:-vera_rubin_nvl72_analytical}"

exec bash "$SCRIPT_DIR/dense_model_basic.sh" -- \
  --cluster_config_num_racks "$NUM_RACKS" \
  --replica_config_network_device vera_rubin_nvl72_domain \
  --execution_time_predictor_config_type analytical_roofline \
  --astra_sim_analytical_cc_backend_config_intra_server_topology Switch \
  --astra_sim_analytical_cc_backend_config_intra_server_bandwidth_gbps 14400 \
  --astra_sim_analytical_cc_backend_config_intra_server_latency_us 1.0 \
  "$@"
