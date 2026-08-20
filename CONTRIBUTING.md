# Contributing to Frontier

Thank you for helping improve Frontier. New simulator work belongs in the C++
core under `cpp/`; the Python simulator is maintained for capabilities that do
not yet exist in C++ and for profiling workflows.

## Development setup

Frontier requires CMake 3.24+, Ninja, and a C++17 compiler. From `cpp/`:

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

Linux contributors should also run the sanitizer preset before requesting
review:

```bash
cmake --preset sanitizers
cmake --build --preset sanitizers
ctest --preset sanitizers
```

The CI matrix covers GCC, Clang, MSVC, Debug, Release, ASan, and UBSan. Please
do not submit changes that introduce compiler warnings. C++ formatting is
checked with `clang-format` 22.1.8 using the repository's `.clang-format` file.

## Design expectations

- State changes must be driven by typed events.
- Keep the global, cluster, replica, and pipeline-stage scheduling layers
  distinct.
- Prefer composition and existing helpers over adding feature branches to an
  unrelated scheduler layer.
- Configuration additions must be optional unless accompanied by a schema
  version change and migration notes.
- Preserve deterministic event ordering and numerical results. If a fidelity
  correction changes results, identify the affected configurations in the
  pull request and `CHANGELOG.md`.
- The C++ implementation does not need compatibility code whose only purpose
  is to copy historical Python behavior. When Python behavior represents the
  modeled runtime contract, document that reason next to the implementation.

## Tests

Every behavior change should include the narrowest useful unit test and, when
it crosses scheduling layers, an integration or topology-matrix case. New
features should cover applicable combinations of:

- dense and MoE models;
- online and offline modes;
- co-location and sequential PDD;
- varied request lengths, counts, and arrival rates;
- enabled and disabled feature states; and
- representative model configurations.

Run the complete CTest suite. User-visible CLI, configuration, workload, or
output changes must also update `cpp/README.md` and a runnable example where
appropriate.

## Pull requests

Keep changes focused and describe:

1. the modeled contract or maintenance problem;
2. the ownership boundary selected for the implementation;
3. whether numerical or trace results change;
4. the exact commands used for verification; and
5. any compatibility or release-note impact.

Do not commit build directories, generated output, local caches, credentials,
or profiling datasets that are not explicitly part of a fixture.
