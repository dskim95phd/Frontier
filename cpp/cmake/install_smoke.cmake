if(NOT DEFINED FRONTIER_BUILD_DIR OR NOT DEFINED FRONTIER_SOURCE_DIR OR
   NOT DEFINED FRONTIER_INSTALL_DIR)
  message(FATAL_ERROR "install smoke requires build, source, and install paths")
endif()

set(install_command
  "${CMAKE_COMMAND}" --install "${FRONTIER_BUILD_DIR}"
  --prefix "${FRONTIER_INSTALL_DIR}"
)
if(DEFINED FRONTIER_BUILD_CONFIG AND NOT FRONTIER_BUILD_CONFIG STREQUAL "")
  list(APPEND install_command --config "${FRONTIER_BUILD_CONFIG}")
endif()

execute_process(
  COMMAND ${install_command}
  RESULT_VARIABLE install_result
  OUTPUT_VARIABLE install_stdout
  ERROR_VARIABLE install_stderr
)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR
    "frontier install failed (${install_result})\n"
    "${install_stdout}\n${install_stderr}"
  )
endif()

if(CMAKE_HOST_WIN32)
  set(frontier_executable "${FRONTIER_INSTALL_DIR}/bin/frontier_sim.exe")
else()
  set(frontier_executable "${FRONTIER_INSTALL_DIR}/bin/frontier_sim")
endif()

set(smoke_output "${FRONTIER_INSTALL_DIR}/install-smoke-output")
file(MAKE_DIRECTORY "${smoke_output}")
execute_process(
  COMMAND
    "${frontier_executable}"
    --config
    "${FRONTIER_SOURCE_DIR}/examples/configs/04_moe_expert_parallel.json"
    --workload
    "${FRONTIER_SOURCE_DIR}/examples/workloads/00_tiny.csv"
    --output-dir
    "${smoke_output}"
    --output-mode
    summary
  WORKING_DIRECTORY "${FRONTIER_INSTALL_DIR}"
  RESULT_VARIABLE smoke_result
  OUTPUT_VARIABLE smoke_stdout
  ERROR_VARIABLE smoke_stderr
)
if(NOT smoke_result EQUAL 0)
  message(FATAL_ERROR
    "installed frontier_sim failed (${smoke_result})\n"
    "${smoke_stdout}\n${smoke_stderr}"
  )
endif()
if(NOT EXISTS "${smoke_output}/summary.json")
  message(FATAL_ERROR "installed frontier_sim did not write summary.json")
endif()
if(NOT EXISTS
   "${FRONTIER_INSTALL_DIR}/share/frontier/models/Phi-tiny-MoE-instruct.json")
  message(FATAL_ERROR "installed Frontier model assets are missing")
endif()
if(NOT EXISTS
   "${FRONTIER_INSTALL_DIR}/share/frontier/schema/config-v1.schema.json")
  message(FATAL_ERROR "installed Frontier config schema is missing")
endif()
if(NOT EXISTS
   "${FRONTIER_INSTALL_DIR}/share/frontier/schema/config-v2.schema.json")
  message(FATAL_ERROR "installed Frontier modular config schema is missing")
endif()
if(NOT EXISTS
   "${FRONTIER_INSTALL_DIR}/share/frontier/schema/precision-profile-asset-v1.schema.json")
  message(FATAL_ERROR "installed Frontier precision profile schema is missing")
endif()
if(NOT EXISTS
   "${FRONTIER_INSTALL_DIR}/share/frontier/gpus/gb300.json" OR
   NOT EXISTS
   "${FRONTIER_INSTALL_DIR}/share/frontier/clusters/gb300-1gpu.json" OR
   NOT EXISTS "${FRONTIER_INSTALL_DIR}/share/frontier/links/ib-200g.json" OR
   NOT EXISTS
   "${FRONTIER_INSTALL_DIR}/share/frontier/precision_profiles/kimi-k3-native.json" OR
   NOT EXISTS
   "${FRONTIER_INSTALL_DIR}/share/frontier/precision_profiles/kimi-k2-k3-native.json" OR
   NOT EXISTS
   "${FRONTIER_INSTALL_DIR}/share/frontier/precision_profiles/kimi-k3-w4a8.json")
  message(FATAL_ERROR "installed Frontier modular config assets are missing")
endif()
