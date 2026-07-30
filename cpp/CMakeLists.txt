cmake_minimum_required(VERSION 3.24)

project(
  FrontierCpp
  VERSION 0.1.0
  DESCRIPTION "C++ discrete-event simulation core for Frontier"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(
  FRONTIER_FETCH_DEPENDENCIES
  "Allow CMake to download pinned C++ dependencies"
  ON
)
set(
  FRONTIER_NLOHMANN_JSON_SOURCE_DIR
  ""
  CACHE PATH
  "Offline path to a nlohmann/json 3.11.3 source tree"
)

find_package(nlohmann_json 3.11.3 EXACT CONFIG QUIET)

if(NOT TARGET nlohmann_json::nlohmann_json)
  if(FRONTIER_NLOHMANN_JSON_SOURCE_DIR)
    add_subdirectory(
      "${FRONTIER_NLOHMANN_JSON_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/_deps/nlohmann_json-build"
      EXCLUDE_FROM_ALL
    )
  elseif(FRONTIER_FETCH_DEPENDENCIES)
    include(FetchContent)
    FetchContent_Declare(
      nlohmann_json
      URL
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz"
      URL_HASH
        "SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(nlohmann_json)
  else()
    message(
      FATAL_ERROR
      "nlohmann/json 3.11.3 is required. Install it, set "
      "FRONTIER_NLOHMANN_JSON_SOURCE_DIR, or enable "
      "FRONTIER_FETCH_DEPENDENCIES."
    )
  endif()
endif()

function(frontier_enable_warnings target_name)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /permissive-)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
      ${target_name}
      PRIVATE -Wall -Wextra -Wpedantic
    )
  endif()
endfunction()

add_library(
  frontier_core
  STATIC
  frontier/config/config.cc
  frontier/config/config_parse.cc
  frontier/config/config_serialize.cc
  frontier/cc_backend/analytical_model.cc
  frontier/core/event_queue.cc
  frontier/entities/request.cc
  frontier/entities/batch.cc
  frontier/entities/batch_stage.cc
  frontier/entities/cluster.cc
  frontier/entities/replica.cc
  frontier/entities/kv_cache_transfer_info.cc
  frontier/events/batch_stage_arrival_event.cc
  frontier/events/batch_stage_end_event.cc
  frontier/events/cluster_batch_end_event.cc
  frontier/events/cluster_schedule_event.cc
  frontier/events/event_dispatcher.cc
  frontier/events/event_helpers.cc
  frontier/events/global_batch_end_event.cc
  frontier/events/global_schedule_event.cc
  frontier/events/kv_cache_transfer_end_event.cc
  frontier/events/kv_cache_transfer_start_event.cc
  frontier/events/replica_schedule_event.cc
  frontier/events/replica_stage_schedule_event.cc
  frontier/events/request_arrival_event.cc
  frontier/events/prefill_sync_event.cc
  frontier/events/prefill_sync_collective_event.cc
  frontier/events/decode_sync_event.cc
  frontier/events/decode_sync_collective_event.cc
  frontier/execution_time_predictor/execution_time_predictor_factory.cc
  frontier/execution_time_predictor/fixed_execution_time_predictor.cc
  frontier/execution_time_predictor/analytical_roofline_execution_time_predictor.cc
  frontier/kv_cache_transfer/analytical_transfer.cc
  frontier/metrics/output_contract.cc
  frontier/metrics/metrics_store.cc
  frontier/request_generator/workload.cc
  frontier/scheduler/cluster_scheduler/base_cluster_scheduler.cc
  frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.cc
  frontier/scheduler/global_scheduler/global_scheduler.cc
  frontier/scheduler/kv_block_accounting.cc
  frontier/scheduler/replica_scheduler/base_replica_scheduler.cc
  frontier/scheduler/replica_scheduler/replica_scheduler_factory.cc
  frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.cc
  frontier/scheduler/replica_stage_scheduler/replica_stage_scheduler.cc
  frontier/simulator/entity_arena.cc
  frontier/simulator/simulator.cc
)
target_include_directories(frontier_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(frontier_core PRIVATE nlohmann_json::nlohmann_json)
frontier_enable_warnings(frontier_core)

add_executable(frontier_sim frontier/main.cc)
target_compile_definitions(
  frontier_sim
  PRIVATE FRONTIER_VERSION="${PROJECT_VERSION}"
)
target_link_libraries(frontier_sim PRIVATE frontier_core)
frontier_enable_warnings(frontier_sim)

include(CTest)

if(BUILD_TESTING)
  add_test(
    NAME frontier_sim.version
    COMMAND $<TARGET_FILE:frontier_sim> --version
  )
  set_tests_properties(
    frontier_sim.version
    PROPERTIES PASS_REGULAR_EXPRESSION "^frontier_sim 0\\.1\\.0"
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
endif()
