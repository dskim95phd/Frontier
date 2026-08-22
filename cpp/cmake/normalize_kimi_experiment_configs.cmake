if(NOT DEFINED FRONTIER_BINARY OR NOT DEFINED FRONTIER_SOURCE_DIR)
  message(FATAL_ERROR "Kimi config check requires binary and source directory")
endif()

file(
  GLOB kimi_configs
  "${FRONTIER_SOURCE_DIR}/experiments/kimi_k2_cpu_dram/configs/*.json"
  "${FRONTIER_SOURCE_DIR}/experiments/kimi_k3_cpu_dram/configs/*.json"
)
list(LENGTH kimi_configs config_count)
if(NOT config_count EQUAL 12)
  message(FATAL_ERROR "expected 12 checked-in Kimi configs, found ${config_count}")
endif()

foreach(config IN LISTS kimi_configs)
  execute_process(
    COMMAND "${FRONTIER_BINARY}" --normalize-config "${config}"
    WORKING_DIRECTORY "${FRONTIER_SOURCE_DIR}/.."
    RESULT_VARIABLE normalize_result
    OUTPUT_QUIET
    ERROR_VARIABLE normalize_error
  )
  if(NOT normalize_result EQUAL 0)
    message(FATAL_ERROR
      "failed to normalize ${config} (${normalize_result})\n${normalize_error}"
    )
  endif()
endforeach()
