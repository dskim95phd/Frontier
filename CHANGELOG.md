# Changelog

Frontier follows semantic versioning for release artifacts. Configuration and
output schemas have their own explicit `schema_version` and may remain stable
across executable releases.

## Unreleased

### Added

- Cross-platform C++ CI for GCC, Clang, MSVC, ASan, and UBSan.
- CMake presets, install rules, CPack release archives, and relocatable model
  asset discovery.
- A strict, installable JSON Schema for configuration version 1, validated
  against every public example and contract fixture in CI.
- Focused tests for executable-relative runtime paths and the MoE barrier
  coordinator.
- Contributor, security, and community policies.

### Changed

- The root quick start now points to the primary C++ implementation.
- C++ and Python package versions are aligned at 0.2.0.
- MoE collective coordination and tiered prefix planning now have standalone,
  unit-testable ownership boundaries instead of living inside large scheduler
  implementations.
- The C++ core's configuration, scheduling, KV-cache, simulator, metrics, and
  test-build responsibilities are split into focused implementation modules;
  public headers and runtime/output contracts remain unchanged.
- Analytical prediction now separates configuration validation, stage timing
  caches, group-MoE prediction, roofline primitives, attention work, and MoE
  communication; model registries and JSON record serializers likewise have
  dedicated internal ownership.

## 0.2.0

- Added deterministic co-location and sequential PDD simulation in C++.
- Added TP, PP, DP, MoE TP/EP, DCP, analytical execution models, session prefix
  caching, finite CPU KV-cache tiering, Kimi K2/K3 modeling, and deterministic
  JSON/CSV output contracts.

## 0.1.0

- Initial Frontier Python simulator and early C++ core foundations.
