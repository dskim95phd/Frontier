if(NOT DEFINED FRONTIER_REPOSITORY_ROOT OR NOT DEFINED FRONTIER_VERSION)
  message(FATAL_ERROR "release metadata check requires repository root and version")
endif()

file(READ "${FRONTIER_REPOSITORY_ROOT}/pyproject.toml" pyproject)
string(FIND "${pyproject}" "version = \"${FRONTIER_VERSION}\"" py_version)
if(py_version EQUAL -1)
  message(FATAL_ERROR
    "pyproject.toml does not declare version ${FRONTIER_VERSION}"
  )
endif()

file(READ "${FRONTIER_REPOSITORY_ROOT}/README.md" readme)
string(FIND "${readme}" "release-v${FRONTIER_VERSION}-green" readme_version)
if(readme_version EQUAL -1)
  message(FATAL_ERROR
    "README.md release badge does not match ${FRONTIER_VERSION}"
  )
endif()
