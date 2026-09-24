# Developer Documentation

How PyPTO is built: the IR, the pass pipeline, code generation, and the
infrastructure around them.

This is documentation for people working *on* the compiler. If you are writing
PyPTO programs, start with the [User Manual](../user/index.md).

## Sub-chapters

| Chapter | What it covers |
| ------- | -------------- |
| [IR](ir/index.md) | Node hierarchy, type system, operators, builder, parser, serialization, structural comparison |
| [Passes](passes/index.md) | The pass framework and every pass in the default pipeline, numbered in execution order |
| [Language](language/index.md) | The Python DSL syntax specification and external C++ kernel integration |
| [Code Generation](codegen/index.md) | Lowering IR to PTO-ISA dialect MLIR and to orchestration C++ |
| [Backend](backend/index.md) | Per-architecture dispatch through `BackendHandler` |
| [Debug](debug/index.md) | Lowering IR to an executable PyTorch script for numerical validation |

## Top-level topics

| Page | What it covers |
| ---- | -------------- |
| [PTO Project Ecosystem](00-ecosystem.md) | The multi-repo toolchain — PyPTO, PTOAS, pto-isa, simpler, pypto-lib — and how they fit together |
| [Compile Profiling](01-compile-profiling.md) | Built-in wall-clock timing of the compilation pipeline |
| [Error Handling](02-error-handling.md) | `CHECK` vs `INTERNAL_CHECK`, PyPTO exception types, IR source locations in failures |
| [Logging](03-logging.md) | The two independent logging subsystems and which one a message came from |
| [Runtime DFX Flags](03-runtime-dfx.md) | The five runtime diagnostic sub-features exposed through `RunConfig` |
| [Replaying an Existing `build_output`](03-runtime-replay.md) | Re-run, edit, and re-measure a compiled build directory without recompiling |
| [Simulator Trace Cleaning](04-simulator-trace-cleaning.md) | Converting MindStudio Insight binary dumps into readable traces |
| [Per-Task Ring Sizing](05-runtime-ring-sizing.md) | The three ring-size overrides on `RunConfig` and when to tune them |
| [Persistent L3 execution](06-persistent-l3.md) | Reusing one worker across prepared distributed programs |
| [Memory Map](07-memory-map.md) | Rendering a pass dump into an interactive HTML map of on-chip memory |
| [Compile and Execution Entry Points](08-entry-points.md) | Every compile and execution entry point, the layer it belongs to, and when to reach for it |
| [Distributed Operators](distributed_ops.md) | The N6 distributed op family — typed DSL access to collectives and primitives |
| [PTOAS Op Status Matrix](ptoas-op-status.md) | Which public and compatibility PTOAS ops the compiler currently emits |
| [FP4](fp4.md) | Logical vs packed FP4, cast policy, and hand-written `FP4E2M1X2` guidance |

## Automated PR Review

The `Codex Review` GitHub Actions workflow reviews non-draft pull requests when
they are opened, reopened, updated, or marked ready for review. It uses the
trusted workflow from the default branch, checks out the pull request head as
untrusted input, and posts the result from a separate GitHub-hosted job.
The generated review is advisory; it never approves or merges the pull request.

Administrators enable or disable reviews with the repository Actions variable
`CODEX_REVIEW_ENABLED`. Set it to `true` to enable reviews; unset it or use any
other value to disable them immediately.

The review job requires a dedicated self-hosted runner labelled `Linux`,
`ARM64`, and `cpu-codex`. The runner provides the following host-managed
resources, none of which come from the pull request:

- `/home/ci-runner/.codex-ci/auth.json`, readable only by the runner account
- the digest-pinned review and proxy images in the local registry
- the `pypto-codex-egress` Docker network
- a healthy `pypto-codex-proxy-relay` container on relay port `17895`
- a loopback-only mixed proxy on `127.0.0.1:7895`

Codex runs as UID/GID `1002:1003` in a read-only container with a read-only
repository mount, dropped capabilities, resource limits, and an internal Docker
network. The GitHub token with pull-request write access is available only to
the separate comment job. Review output is rejected if it contains an exact
long-form value from the Codex credential and is labelled as automated,
untrusted content when posted.

ChatGPT-managed Codex authentication is refreshed by a trusted weekly or manual
maintenance run that has no checkout and uses an empty temporary directory.
The refresh is serialized with reviews by a host lock and updates the persistent
credential in place. Pull-request review containers receive only an ephemeral
snapshot; they never mount or write the persistent credential.

## See Also

- [PTO ISA reference](../reference/index.md) — the hardware model the backend targets.
- [Runtime documentation](https://www.pypto.ai/simpler/) — the scheduler that executes compiled programs.
