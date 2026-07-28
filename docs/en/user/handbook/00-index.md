# Feature Handbook

Task-oriented guides that string PyPTO's existing tools into workflows. Each
chapter follows **symptom → tool → steps → how to read the output**, and
cross-links the reference docs rather than duplicating them. If you know your
symptom, use the index below; otherwise read the pillar workflow pages.

## Symptom → Tool Index

| Symptom | Go to |
| ------- | ----- |
| Output values are wrong / diverge from torch | [Precision › Workflow](precision/00-workflow.md) |
| Need to find the first pass whose IR diverges | [Precision › Pass-IR Bisection](precision/02-pass-ir-bisection.md) |
| Want to compare one tensor on-device vs golden | [Precision › Selective Dump](precision/03-selective-dump.md) |
| Kernel is correct but too slow | [Perf › Workflow](perf/00-workflow.md) |
| Loop / pipeline not overlapping | [Perf › Loop & Pipeline](perf/01-loop-pipeline.md) |
| Underutilized cores / want to parallelize | [Perf › Split & Parallel](perf/02-split-parallel.md) |
| Tasks serialize when they could overlap | [Perf › Dependency & Dispatch](perf/03-dependency-dispatch.md) |
| Data in the wrong memory level | [Perf › Memory Placement](perf/04-memory-placement.md) |
| Need to measure where time goes | [Perf › Measuring Impact](perf/05-measuring-impact.md) |
| Want the per-task timeline / see where cores stall | [DFX › Swimlane](dfx/01-swimlane.md) |
| Want to inspect scheduling / dependency graph | [DFX › Diagnostics Render](dfx/02-diagnostics-render.md) |
| Need verbose compile / runtime logs | [DFX › Logging](dfx/04-logging.md) |
| Reproduce from an existing `build_output` | [DFX › Replay](dfx/03-replay.md) |

## The Three Pillars

1. **[Precision Localization](precision/00-workflow.md)** — "my output is wrong →
   here is how to bisect where it diverges."
2. **[Performance-Tuning Syntax](perf/00-workflow.md)** — the DSL knobs, *when*
   and *how* to apply each, and how to measure the effect.
3. **[DFX Features](dfx/00-flag-matrix.md)** — diagnostics, rendering, replay,
   and logging.
