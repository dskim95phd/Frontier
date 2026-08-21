# Test targets and CTest registration live separately so the product build,
# installation, and packaging contract remain easy to audit.

add_test(
  NAME frontier_release.metadata
  COMMAND
    ${CMAKE_COMMAND}
    "-DFRONTIER_REPOSITORY_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/.."
    "-DFRONTIER_VERSION=${PROJECT_VERSION}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_release_metadata.cmake
)
add_test(
  NAME frontier_release.install_smoke
  COMMAND
    ${CMAKE_COMMAND}
    "-DFRONTIER_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    "-DFRONTIER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DFRONTIER_INSTALL_DIR=${CMAKE_CURRENT_BINARY_DIR}/install-smoke"
    "-DFRONTIER_BUILD_CONFIG=$<CONFIG>"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_smoke.cmake
)
set_tests_properties(
  frontier_release.install_smoke
  PROPERTIES
    LABELS packaging
    RUN_SERIAL TRUE
    TIMEOUT 60
)

add_test(
  NAME frontier_sim.version
  COMMAND $<TARGET_FILE:frontier_sim> --version
)
set_tests_properties(
  frontier_sim.version
  PROPERTIES PASS_REGULAR_EXPRESSION "^frontier_sim 0\\.2\\.0"
)
add_test(
  NAME frontier_sim.cli
  COMMAND
    $<TARGET_FILE:frontier_sim>
    --config
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/config/fixed_parallel_colocation.json"
    --workload
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/workloads/step25_parallel.csv"
)
set_tests_properties(
  frontier_sim.cli
  PROPERTIES PASS_REGULAR_EXPRESSION "global_batch_end"
)
add_test(
  NAME frontier_sim.wall_progress
  COMMAND
    $<TARGET_FILE:frontier_sim>
    --config
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/config/fixed_parallel_colocation.json"
    --workload
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/workloads/step25_parallel.csv"
    --output-dir
    "${CMAKE_CURRENT_BINARY_DIR}/wall-progress-test-output"
    --output-mode
    summary
    --wall-progress-interval-s
    0.000001
)
set_tests_properties(
  frontier_sim.wall_progress
  PROPERTIES PASS_REGULAR_EXPRESSION "simulation_progress_s=[0-9]"
)
add_test(
  NAME frontier_load_benchmark.summary
  COMMAND
    $<TARGET_FILE:frontier_load_benchmark_summary>
    --config
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/config/fixed_parallel_colocation.json"
    --workload
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/workloads/step25_parallel.csv"
)
set_tests_properties(
  frontier_load_benchmark.summary
  PROPERTIES PASS_REGULAR_EXPRESSION "\"requests\": 8"
)
add_test(
  NAME frontier_load_benchmark.pdd_summary
  COMMAND
    $<TARGET_FILE:frontier_load_benchmark_summary>
    --config
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/config/fixed_sequential_pdd.json"
    --workload
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/workloads/step3_pdd_small.csv"
)
set_tests_properties(
  frontier_load_benchmark.pdd_summary
  PROPERTIES PASS_REGULAR_EXPRESSION "\"kv_cache_transfers\": 2"
)
add_test(
  NAME frontier_sim.normalize_config
  COMMAND
    $<TARGET_FILE:frontier_sim>
    --normalize-config
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/config/fixed_parallel_colocation.json"
)
set_tests_properties(
  frontier_sim.normalize_config
  PROPERTIES PASS_REGULAR_EXPRESSION "\"schema_version\": 1"
)
add_test(
  NAME frontier_sim.normalize_workload
  COMMAND
    $<TARGET_FILE:frontier_sim>
    --normalize-workload
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/workloads/session_prefix.csv"
)
set_tests_properties(
  frontier_sim.normalize_workload
  PROPERTIES PASS_REGULAR_EXPRESSION "session_turn_index"
)
add_test(
  NAME frontier_sim.normalize_modular_config
  COMMAND
    $<TARGET_FILE:frontier_sim>
    --normalize-config
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/configs/08_modular_sequential_pdd.json"
)
set_tests_properties(
  frontier_sim.normalize_modular_config
  PROPERTIES
    PASS_REGULAR_EXPRESSION "\"network_bandwidth_gbps\": 200\\.0"
)
add_test(
  NAME frontier_sim.normalize_kimi_experiment_configs
  COMMAND
    ${CMAKE_COMMAND}
    "-DFRONTIER_BINARY=$<TARGET_FILE:frontier_sim>"
    "-DFRONTIER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
    -P
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/normalize_kimi_experiment_configs.cmake"
)

add_executable(
  frontier_event_queue_test
  tests/core/event_queue_test.cc
)
target_link_libraries(frontier_event_queue_test PRIVATE frontier_core)
frontier_enable_warnings(frontier_event_queue_test)
add_test(
  NAME frontier_core.event_queue
  COMMAND $<TARGET_FILE:frontier_event_queue_test>
)

add_executable(
  frontier_runtime_paths_test
  tests/core/runtime_paths_test.cc
)
target_link_libraries(frontier_runtime_paths_test PRIVATE frontier_core)
frontier_enable_warnings(frontier_runtime_paths_test)
add_test(
  NAME frontier_core.runtime_paths
  COMMAND $<TARGET_FILE:frontier_runtime_paths_test>
)

add_executable(
  frontier_config_contract_test
  tests/config/config_test.cc
)
target_compile_definitions(
  frontier_config_contract_test
  PRIVATE
    FRONTIER_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(frontier_config_contract_test PRIVATE frontier_core)
frontier_enable_warnings(frontier_config_contract_test)
add_test(
  NAME frontier_contract.config
  COMMAND $<TARGET_FILE:frontier_config_contract_test>
)

add_executable(
  frontier_workload_contract_test
  tests/request_generator/workload_test.cc
)
target_compile_definitions(
  frontier_workload_contract_test
  PRIVATE
    FRONTIER_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(frontier_workload_contract_test PRIVATE frontier_core)
frontier_enable_warnings(frontier_workload_contract_test)
add_test(
  NAME frontier_contract.workload
  COMMAND $<TARGET_FILE:frontier_workload_contract_test>
)

add_executable(
  frontier_output_contract_test
  tests/metrics/output_contract_test.cc
)
target_link_libraries(
  frontier_output_contract_test
  PRIVATE
    frontier_core
    nlohmann_json::nlohmann_json
)
frontier_enable_warnings(frontier_output_contract_test)
add_test(
  NAME frontier_contract.output
  COMMAND $<TARGET_FILE:frontier_output_contract_test>
)

add_executable(
  frontier_analytical_model_test
  tests/analytical_model/analytical_model_test.cc
)
target_compile_definitions(
  frontier_analytical_model_test
  PRIVATE
    FRONTIER_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(
  frontier_analytical_model_test
  PRIVATE
    frontier_core
    nlohmann_json::nlohmann_json
)
frontier_enable_warnings(frontier_analytical_model_test)
add_test(
  NAME frontier_analytical.model
  COMMAND $<TARGET_FILE:frontier_analytical_model_test>
)

add_executable(
  frontier_k3_feature_fidelity_test
  tests/analytical_model/k3_feature_fidelity_test.cc
)
target_link_libraries(
  frontier_k3_feature_fidelity_test
  PRIVATE
    frontier_core
    nlohmann_json::nlohmann_json
)
frontier_enable_warnings(frontier_k3_feature_fidelity_test)
add_test(
  NAME frontier_analytical.k3_feature_fidelity
  COMMAND $<TARGET_FILE:frontier_k3_feature_fidelity_test>
)

add_executable(
  frontier_request_entity_test
  tests/entities/request_test.cc
)
target_link_libraries(frontier_request_entity_test PRIVATE frontier_core)
frontier_enable_warnings(frontier_request_entity_test)
add_test(
  NAME frontier_entities.request
  COMMAND $<TARGET_FILE:frontier_request_entity_test>
)

add_executable(
  frontier_batch_entity_test
  tests/entities/batch_test.cc
)
target_link_libraries(frontier_batch_entity_test PRIVATE frontier_core)
frontier_enable_warnings(frontier_batch_entity_test)
add_test(
  NAME frontier_entities.batch
  COMMAND $<TARGET_FILE:frontier_batch_entity_test>
)

add_executable(
  frontier_analytical_roofline_moe_test
  tests/execution_time_predictor/analytical_roofline_execution_time_predictor_moe_test.cc
)
target_link_libraries(
  frontier_analytical_roofline_moe_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_analytical_roofline_moe_test)
add_test(
  NAME frontier_analytical_roofline.moe
  COMMAND $<TARGET_FILE:frontier_analytical_roofline_moe_test>
)

add_executable(
  frontier_kv_block_accounting_test
  tests/scheduler/kv_block_accounting_test.cc
)
target_link_libraries(
  frontier_kv_block_accounting_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_kv_block_accounting_test)
add_test(
  NAME frontier_scheduler.kv_block_accounting
  COMMAND $<TARGET_FILE:frontier_kv_block_accounting_test>
)

add_executable(
  frontier_replica_kv_cache_manager_test
  tests/kv_cache/replica_kv_cache_manager_test.cc
)
target_link_libraries(
  frontier_replica_kv_cache_manager_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_replica_kv_cache_manager_test)
add_test(
  NAME frontier_kv_cache.replica_manager
  COMMAND $<TARGET_FILE:frontier_replica_kv_cache_manager_test>
)

add_executable(
  frontier_cpu_kv_cache_manager_test
  tests/kv_cache/cpu_kv_cache_manager_test.cc
)
target_link_libraries(
  frontier_cpu_kv_cache_manager_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_cpu_kv_cache_manager_test)
add_test(
  NAME frontier_kv_cache.cpu_manager
  COMMAND $<TARGET_FILE:frontier_cpu_kv_cache_manager_test>
)

add_executable(
  frontier_cpu_kv_cache_transfer_test
  tests/cpu_kv_cache_transfer/analytical_transfer_test.cc
)
target_link_libraries(
  frontier_cpu_kv_cache_transfer_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_cpu_kv_cache_transfer_test)
add_test(
  NAME frontier_cpu_kv_cache_transfer.analytical
  COMMAND $<TARGET_FILE:frontier_cpu_kv_cache_transfer_test>
)

add_executable(
  frontier_vllm_v1_scheduler_test
  tests/scheduler/vllm_v1_scheduler_test.cc
)
target_link_libraries(
  frontier_vllm_v1_scheduler_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_vllm_v1_scheduler_test)
add_test(
  NAME frontier_scheduler.vllm_v1
  COMMAND $<TARGET_FILE:frontier_vllm_v1_scheduler_test>
)

add_executable(
  frontier_scheduler_hierarchy_test
  tests/scheduler/scheduler_hierarchy_test.cc
)
target_link_libraries(
  frontier_scheduler_hierarchy_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_scheduler_hierarchy_test)
add_test(
  NAME frontier_scheduler.hierarchy
  COMMAND $<TARGET_FILE:frontier_scheduler_hierarchy_test>
)

add_executable(
  frontier_moe_barrier_coordinator_test
  tests/scheduler/moe_barrier_coordinator_test.cc
)
target_link_libraries(
  frontier_moe_barrier_coordinator_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_moe_barrier_coordinator_test)
add_test(
  NAME frontier_scheduler.moe_barrier_coordinator
  COMMAND $<TARGET_FILE:frontier_moe_barrier_coordinator_test>
)

add_executable(
  frontier_parallel_colocation_test
  tests/simulator/parallel_colocation_test.cc
)
target_compile_definitions(
  frontier_parallel_colocation_test
  PRIVATE
    FRONTIER_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(
  frontier_parallel_colocation_test
  PRIVATE
    frontier_core
    nlohmann_json::nlohmann_json
)
frontier_enable_warnings(frontier_parallel_colocation_test)
add_test(
  NAME frontier_simulator.parallel_colocation
  COMMAND $<TARGET_FILE:frontier_parallel_colocation_test>
)

add_executable(
  frontier_sequential_pdd_test
  tests/simulator/sequential_pdd_test.cc
)
target_compile_definitions(
  frontier_sequential_pdd_test
  PRIVATE
    FRONTIER_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(
  frontier_sequential_pdd_test
  PRIVATE
    frontier_core
    nlohmann_json::nlohmann_json
)
frontier_enable_warnings(frontier_sequential_pdd_test)
add_test(
  NAME frontier_simulator.sequential_pdd
  COMMAND $<TARGET_FILE:frontier_sequential_pdd_test>
)

add_executable(
  frontier_moe_integration_test
  tests/simulator/moe_integration_test.cc
)
target_compile_definitions(
  frontier_moe_integration_test
  PRIVATE
    FRONTIER_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(
  frontier_moe_integration_test
  PRIVATE
    frontier_core
    nlohmann_json::nlohmann_json
)
frontier_enable_warnings(frontier_moe_integration_test)
add_test(
  NAME frontier_simulator.moe_integration
  COMMAND $<TARGET_FILE:frontier_moe_integration_test>
)

add_executable(
  frontier_prefix_cache_integration_test
  tests/simulator/prefix_cache_integration_test.cc
)
target_compile_definitions(
  frontier_prefix_cache_integration_test
  PRIVATE
    FRONTIER_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(
  frontier_prefix_cache_integration_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_prefix_cache_integration_test)
add_test(
  NAME frontier_simulator.prefix_cache
  COMMAND $<TARGET_FILE:frontier_prefix_cache_integration_test>
)

add_executable(
  frontier_cpu_kv_cache_integration_test
  tests/simulator/cpu_kv_cache_integration_test.cc
)
target_compile_definitions(
  frontier_cpu_kv_cache_integration_test
  PRIVATE
    FRONTIER_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples"
)
target_link_libraries(
  frontier_cpu_kv_cache_integration_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_cpu_kv_cache_integration_test)
add_test(
  NAME frontier_simulator.cpu_kv_cache
  COMMAND $<TARGET_FILE:frontier_cpu_kv_cache_integration_test>
)

add_executable(
  frontier_k3_cpu_kv_cache_integration_test
  tests/simulator/k3_cpu_kv_cache_integration_test.cc
)
target_compile_definitions(
  frontier_k3_cpu_kv_cache_integration_test
  PRIVATE
    FRONTIER_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples"
)
target_link_libraries(
  frontier_k3_cpu_kv_cache_integration_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_k3_cpu_kv_cache_integration_test)
add_test(
  NAME frontier_simulator.k3_cpu_kv_cache
  COMMAND $<TARGET_FILE:frontier_k3_cpu_kv_cache_integration_test>
)

add_executable(
  frontier_k3_pipeline_exclusive_offload_matrix_test
  tests/simulator/k3_pipeline_exclusive_offload_matrix_test.cc
)
target_compile_definitions(
  frontier_k3_pipeline_exclusive_offload_matrix_test
  PRIVATE
    FRONTIER_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples"
)
target_link_libraries(
  frontier_k3_pipeline_exclusive_offload_matrix_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_k3_pipeline_exclusive_offload_matrix_test)
add_test(
  NAME frontier_simulator.k3_pipeline_exclusive_offload_matrix
  COMMAND $<TARGET_FILE:frontier_k3_pipeline_exclusive_offload_matrix_test>
)

add_executable(
  frontier_pipeline_exclusive_matrix_test
  tests/simulator/pipeline_exclusive_matrix_test.cc
)
target_compile_definitions(
  frontier_pipeline_exclusive_matrix_test
  PRIVATE
    FRONTIER_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples"
)
target_link_libraries(
  frontier_pipeline_exclusive_matrix_test
  PRIVATE
    frontier_core
    nlohmann_json::nlohmann_json
)
frontier_enable_warnings(frontier_pipeline_exclusive_matrix_test)
add_test(
  NAME frontier_simulator.pipeline_exclusive_matrix
  COMMAND $<TARGET_FILE:frontier_pipeline_exclusive_matrix_test>
)

add_executable(
  frontier_cpu_kv_cache_stress_test
  tests/simulator/cpu_kv_cache_stress_test.cc
)
target_compile_definitions(
  frontier_cpu_kv_cache_stress_test
  PRIVATE
    FRONTIER_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples"
)
target_link_libraries(
  frontier_cpu_kv_cache_stress_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_cpu_kv_cache_stress_test)
add_test(
  NAME frontier_simulator.cpu_kv_cache_stress
  COMMAND $<TARGET_FILE:frontier_cpu_kv_cache_stress_test>
)
set_tests_properties(
  frontier_simulator.cpu_kv_cache_stress
  PROPERTIES
    LABELS stress
    RUN_SERIAL TRUE
    TIMEOUT 180
)

add_executable(
  frontier_cpu_kv_cache_topology_matrix_test
  tests/simulator/cpu_kv_cache_topology_matrix_test.cc
)
target_compile_definitions(
  frontier_cpu_kv_cache_topology_matrix_test
  PRIVATE
    FRONTIER_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples"
    FRONTIER_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(
  frontier_cpu_kv_cache_topology_matrix_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_cpu_kv_cache_topology_matrix_test)
add_test(
  NAME frontier_simulator.cpu_kv_cache_topology_matrix
  COMMAND $<TARGET_FILE:frontier_cpu_kv_cache_topology_matrix_test>
)

add_executable(
  frontier_cpu_kv_cache_stress_matrix_test
  tests/simulator/cpu_kv_cache_stress_matrix_test.cc
)
target_compile_definitions(
  frontier_cpu_kv_cache_stress_matrix_test
  PRIVATE
    FRONTIER_EXAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples"
)
target_link_libraries(
  frontier_cpu_kv_cache_stress_matrix_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_cpu_kv_cache_stress_matrix_test)
add_test(
  NAME frontier_simulator.cpu_kv_cache_stress_matrix
  COMMAND $<TARGET_FILE:frontier_cpu_kv_cache_stress_matrix_test>
)
set_tests_properties(
  frontier_simulator.cpu_kv_cache_stress_matrix
  PROPERTIES
    LABELS stress
    RUN_SERIAL TRUE
    TIMEOUT 300
)

add_executable(
  frontier_prefix_cache_topology_matrix_test
  tests/simulator/prefix_cache_topology_matrix_test.cc
)
target_compile_definitions(
  frontier_prefix_cache_topology_matrix_test
  PRIVATE
    FRONTIER_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(
  frontier_prefix_cache_topology_matrix_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_prefix_cache_topology_matrix_test)
add_test(
  NAME frontier_simulator.prefix_cache_topology_matrix
  COMMAND $<TARGET_FILE:frontier_prefix_cache_topology_matrix_test>
)

add_executable(
  frontier_prefix_cache_stress_test
  tests/simulator/prefix_cache_stress_test.cc
)
target_compile_definitions(
  frontier_prefix_cache_stress_test
  PRIVATE
    FRONTIER_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(
  frontier_prefix_cache_stress_test
  PRIVATE frontier_core
)
frontier_enable_warnings(frontier_prefix_cache_stress_test)
add_test(
  NAME frontier_simulator.prefix_cache_stress
  COMMAND $<TARGET_FILE:frontier_prefix_cache_stress_test>
)
set_tests_properties(
  frontier_simulator.prefix_cache_stress
  PROPERTIES
    LABELS stress
    RUN_SERIAL TRUE
    TIMEOUT 180
)

# Optional, non-gating performance probe for PP stage grouping.  It is
# intentionally a standalone target rather than a CTest test: wall-clock
# samples are machine-dependent, while the output exposes event and
# predictor counts for regression analysis.
add_executable(
  frontier_pp_stage_group_benchmark
  tests/benchmarks/pp_stage_group_benchmark.cc
)
target_compile_definitions(
  frontier_pp_stage_group_benchmark
  PRIVATE
    FRONTIER_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
)
target_link_libraries(
  frontier_pp_stage_group_benchmark
  PRIVATE
    frontier_core
    nlohmann_json::nlohmann_json
)
frontier_enable_warnings(frontier_pp_stage_group_benchmark)
