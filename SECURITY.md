# Security policy

## Supported versions

Security fixes are made on the default branch and included in the next tagged
release. Pre-release branches may receive fixes without a separate patch
release.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature when it is
available. Otherwise email the maintainers listed in `README.md` with
`[SECURITY]` in the subject. Do not open a public issue for an undisclosed
vulnerability.

Include the affected revision, reproduction steps, expected impact, and any
suggested mitigation. The maintainers will acknowledge a report within seven
days and will coordinate disclosure after a fix or mitigation is available.

Frontier is a simulator and does not execute model-generated code, but its CLI
parses untrusted JSON/CSV input and writes output files. Reports involving
input-driven crashes, resource exhaustion, path handling, dependency supply
chain risks, or release archive integrity are in scope.
