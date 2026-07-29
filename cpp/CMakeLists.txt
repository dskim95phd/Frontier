cmake_minimum_required(VERSION 3.24)

project(
  FrontierCpp
  VERSION 0.1.0
  DESCRIPTION "C++ discrete-event simulation core for Frontier"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
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
  frontier/cc_backend/analytical_model.cc
  frontier/core/event_queue.cc
  frontier/execution_time_predictor/analytical_model.cc
  frontier/kv_cache_transfer/analytical_transfer.cc
  frontier/metrics/output_contract.cc
  frontier/request_generator/workload.cc
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
    NAME frontier_sim.foundation_cli
    COMMAND
      $<TARGET_FILE:frontier_sim>
      --config
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/config/minimal_foundation_colocation.json"
      --workload
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/workloads/session_prefix.csv"
  )
  set_tests_properties(
    frontier_sim.foundation_cli
    PROPERTIES PASS_REGULAR_EXPRESSION "foundation_completion"
  )
  add_test(
    NAME frontier_sim.normalize_config
    COMMAND
      $<TARGET_FILE:frontier_sim>
      --normalize-config
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/config/minimal_foundation_colocation.json"
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
  target_link_libraries(frontier_output_contract_test PRIVATE frontier_core)
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
    frontier_foundation_simulator_test
    tests/simulator/simulator_test.cc
  )
  target_compile_definitions(
    frontier_foundation_simulator_test
    PRIVATE
      FRONTIER_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
  )
  target_link_libraries(
    frontier_foundation_simulator_test
    PRIVATE
      frontier_core
      nlohmann_json::nlohmann_json
  )
  frontier_enable_warnings(frontier_foundation_simulator_test)
  add_test(
    NAME frontier_simulator.foundation_lifecycle
    COMMAND $<TARGET_FILE:frontier_foundation_simulator_test>
  )
endif()
