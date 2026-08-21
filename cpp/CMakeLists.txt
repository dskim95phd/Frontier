cmake_minimum_required(VERSION 3.24)

project(
  FrontierCpp
  VERSION 0.2.0
  DESCRIPTION "C++ discrete-event simulation core for Frontier"
  LANGUAGES CXX
)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(
  FRONTIER_WARNINGS_AS_ERRORS
  "Treat compiler warnings as build errors"
  OFF
)

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
    if(FRONTIER_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE /WX)
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
      ${target_name}
      PRIVATE -Wall -Wextra -Wpedantic
    )
    if(FRONTIER_WARNINGS_AS_ERRORS)
      target_compile_options(${target_name} PRIVATE -Werror)
    endif()
  endif()
endfunction()

add_library(
  frontier_core
  STATIC
  frontier/attention/mla.cc
  frontier/attention/model_binding.cc
  frontier/config/config.cc
  frontier/config/config_compose.cc
  frontier/config/config_memory_resolution.cc
  frontier/config/config_memory_profile.cc
  frontier/config/model_config.cc
  frontier/config/model_registry.cc
  frontier/config/config_parse.cc
  frontier/config/config_parse_cluster.cc
  frontier/config/config_parse_execution.cc
  frontier/config/config_parse_primitives.cc
  frontier/config/config_serialize.cc
  frontier/cc_backend/analytical_model.cc
  frontier/core/event_queue.cc
  frontier/core/runtime_paths.cc
  frontier/entities/request.cc
  frontier/entities/batch.cc
  frontier/entities/batch_stage.cc
  frontier/entities/cluster.cc
  frontier/entities/replica.cc
  frontier/entities/kv_cache_transfer_info.cc
  frontier/entities/cpu_kv_cache_transfer_info.cc
  frontier/kv_cache/replica_kv_cache_manager.cc
  frontier/kv_cache/replica_kv_cache_admission.cc
  frontier/kv_cache/replica_kv_cache_commitments.cc
  frontier/kv_cache/replica_kv_cache_kda.cc
  frontier/kv_cache/cpu_kv_cache_manager.cc
  frontier/kv_cache/cpu_kv_cache_offload.cc
  frontier/kv_cache/cpu_kv_cache_restore.cc
  frontier/kv_cache/cpu_kv_cache_validation.cc
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
  frontier/events/cpu_kv_cache_offload_start_event.cc
  frontier/events/cpu_kv_cache_offload_end_event.cc
  frontier/events/cpu_kv_cache_restore_start_event.cc
  frontier/events/cpu_kv_cache_restore_end_event.cc
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
  frontier/execution_time_predictor/analytical_roofline_cache.cc
  frontier/execution_time_predictor/analytical_roofline_group_moe.cc
  frontier/execution_time_predictor/analytical_roofline_internal.cc
  frontier/execution_time_predictor/analytical_roofline_prediction.cc
  frontier/execution_time_predictor/analytical_dense_layer.cc
  frontier/execution_time_predictor/analytical_attention_work.cc
  frontier/execution_time_predictor/analytical_roofline_primitives.cc
  frontier/execution_time_predictor/analytical_expert_parallel.cc
  frontier/execution_time_predictor/analytical_moe_layer.cc
  frontier/execution_time_predictor/analytical_moe_communication.cc
  frontier/execution_time_predictor/analytical_moe_routing.cc
  frontier/execution_time_predictor/analytical_moe_stage.cc
  frontier/execution_time_predictor/analytical_stage_diagnostics.cc
  frontier/kv_cache_transfer/analytical_transfer.cc
  frontier/cpu_kv_cache_transfer/analytical_transfer.cc
  frontier/metrics/output_contract.cc
  frontier/metrics/output_json_trace.cc
  frontier/metrics/output_json_records.cc
  frontier/metrics/output_contract_validation.cc
  frontier/metrics/output_csv.cc
  frontier/metrics/metrics_store.cc
  frontier/metrics/metrics_store_cache_activity.cc
  frontier/metrics/metrics_store_cache_targets.cc
  frontier/request_generator/workload.cc
  frontier/scheduler/cluster_scheduler/base_cluster_scheduler.cc
  frontier/scheduler/cluster_scheduler/base_cluster_scheduler_moe_begin.cc
  frontier/scheduler/cluster_scheduler/base_cluster_scheduler_moe_domain.cc
  frontier/scheduler/cluster_scheduler/base_cluster_scheduler_moe_progress.cc
  frontier/scheduler/cluster_scheduler/base_cluster_scheduler_sync_events.cc
  frontier/scheduler/cluster_scheduler/moe_barrier_coordinator.cc
  frontier/scheduler/cluster_scheduler/actual_cache_aware_cluster_scheduler.cc
  frontier/scheduler/cluster_scheduler/kv_aware_cluster_scheduler.cc
  frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.cc
  frontier/scheduler/cluster_scheduler/sticky_round_robin_cluster_scheduler.cc
  frontier/scheduler/cluster_scheduler/vllm_queue_aware_cluster_scheduler.cc
  frontier/scheduler/global_scheduler/global_scheduler.cc
  frontier/scheduler/kv_block_accounting.cc
  frontier/scheduler/replica_scheduler/base_replica_scheduler.cc
  frontier/scheduler/replica_scheduler/replica_scheduler_factory.cc
  frontier/scheduler/replica_scheduler/tiered_prefix_plan.cc
  frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.cc
  frontier/scheduler/replica_scheduler/vllm_v1_cpu_kv_cache.cc
  frontier/scheduler/replica_scheduler/vllm_v1_request_lifecycle.cc
  frontier/scheduler/replica_scheduler/vllm_v1_scheduling.cc
  frontier/scheduler/replica_stage_scheduler/replica_stage_scheduler.cc
  frontier/simulator/entity_arena.cc
  frontier/simulator/simulator.cc
  frontier/simulator/simulator_finalization.cc
  frontier/simulator/simulator_memory_layout.cc
  frontier/simulator/simulator_run.cc
)
target_include_directories(frontier_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(frontier_core PRIVATE nlohmann_json::nlohmann_json)
add_custom_target(
  frontier_config_assets ALL
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/../data/config/models"
    "${CMAKE_CURRENT_BINARY_DIR}/share/frontier/models"
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/../data/config/gpus"
    "${CMAKE_CURRENT_BINARY_DIR}/share/frontier/gpus"
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/../data/config/clusters"
    "${CMAKE_CURRENT_BINARY_DIR}/share/frontier/clusters"
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/../data/config/links"
    "${CMAKE_CURRENT_BINARY_DIR}/share/frontier/links"
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/../data/config/precision_profiles"
    "${CMAKE_CURRENT_BINARY_DIR}/share/frontier/precision_profiles"
  COMMENT "Copying Frontier configuration assets"
)
add_dependencies(frontier_core frontier_config_assets)
frontier_enable_warnings(frontier_core)

add_executable(frontier_sim frontier/main.cc)
target_compile_definitions(
  frontier_sim
  PRIVATE FRONTIER_VERSION="${PROJECT_VERSION}"
)
target_link_libraries(frontier_sim PRIVATE frontier_core)
frontier_enable_warnings(frontier_sim)

add_executable(
  frontier_load_benchmark_summary
  frontier/load_benchmark_main.cc
)
target_link_libraries(
  frontier_load_benchmark_summary
  PRIVATE
    frontier_core
    nlohmann_json::nlohmann_json
)
frontier_enable_warnings(frontier_load_benchmark_summary)

install(
  TARGETS frontier_sim
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../data/config/models/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/frontier/models"
  FILES_MATCHING PATTERN "*.json"
)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../data/config/gpus/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/frontier/gpus"
  FILES_MATCHING PATTERN "*.json"
)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../data/config/clusters/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/frontier/clusters"
  FILES_MATCHING PATTERN "*.json"
)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../data/config/links/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/frontier/links"
  FILES_MATCHING PATTERN "*.json"
)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../data/config/precision_profiles/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/frontier/precision_profiles"
  FILES_MATCHING PATTERN "*.json"
)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/schema/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/frontier/schema"
  FILES_MATCHING PATTERN "*.schema.json"
)

set(CPACK_PACKAGE_NAME "frontier-simulator")
set(CPACK_PACKAGE_VENDOR "Frontier contributors")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")
set(CPACK_PACKAGE_CHECKSUM SHA256)
if(WIN32)
  set(CPACK_GENERATOR ZIP)
else()
  set(CPACK_GENERATOR TGZ)
endif()
include(CPack)

include(CTest)

if(BUILD_TESTING)
  include(cmake/tests.cmake)
endif()
