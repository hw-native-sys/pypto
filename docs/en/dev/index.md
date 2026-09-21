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

The `PR Agent` GitHub Actions workflow reviews non-draft PRs when they are
opened, reopened, updated, or marked ready, including PRs from forks. Repository
owners, organization members, and collaborators can also post exactly `/review`
on an open PR to request a review. Findings update a persistent PR comment.

Administrators enable the workflow in **Settings → Secrets and variables → Actions**:

1. Add the DeepSeek API key as the repository secret `OPENAI_KEY`.
2. Optionally set `PR_AGENT_API_BASE` (default `https://api.deepseek.com`) and
   `PR_AGENT_MODEL` (default `deepseek-flash`) as repository variables. Use the
   provider's model ID without an `openai/` prefix; the workflow adds it for
   OpenAI-compatible routing. Reviews use a 128,000-token context budget.
3. Set the repository variable `PR_AGENT_ENABLED` to `true` after the workflow
   is merged into the default branch. Unset it or set it to `false` to disable reviews.

PR content is sent to the configured model service and consumes API credits.
The workflow reads PR data through GitHub's API without checking out PR code.
It enables review only, uses workflow-owned settings, and does not automatically
rewrite PR descriptions, apply code changes, or approve merges. Bot-triggered
events are skipped; a maintainer can request `/review` on a bot-authored PR.

## See Also

- [PTO ISA reference](../reference/index.md) — the hardware model the backend targets.
- [Runtime documentation](https://www.pypto.ai/simpler/) — the scheduler that executes compiled programs.
