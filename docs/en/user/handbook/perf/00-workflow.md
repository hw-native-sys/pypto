# Performance Tuning: Workflow

**Symptom:** your kernel is correct but too slow. This page is the tuning loop —
measure, find the bottleneck, apply the matching knob, re-measure.

## The Golden Rule: Measure First

Never tune blind. Before touching any knob, profile to find where time actually
goes ([Measuring Impact](05-measuring-impact.md)):

- **Benchmark** — `pypto.runtime.benchmark(compiled, args)` gives the headline
  end-to-end device wall-clock (steady state over many launches). Start here to
  know *whether* a change helped; use the lenses below to know *why*.
- **Compile-time profiling** — `ir.compile(..., profiling=True)` (or the
  `PYPTO_COMPILE_PROFILING=1` env var) writes
  `output_dir/report/pipeline_profile.{txt,json}`. Note this measures
  **compilation** wall-clock (per-pass, per-kernel codegen), i.e. "why is
  *compiling* slow" — for on-chip kernel performance use the two tools below.
- **In-core profiling** — cycle-accurate per-kernel traces via the Ascend msprof
  op-simulator (the `incore-profiling` skill). This is your **in-core** lens —
  what happens *inside* one kernel/task, including cube/vector overlap.
- **On-device swimlane** — `enable_l2_swimlane` captures a per-task timeline
  showing where separate tasks stall vs overlap ([DFX › Swimlane](../dfx/01-swimlane.md)).
  This is your **inter-core** lens.
- **Codegen comparison** — diff `.pto` / pass dumps across branches to see what a
  change actually produced (the `compare-codegen` skill).

Then change **one knob at a time** and re-measure. Keep the before/after numbers.

## Two axes: in-core vs inter-core

Tuning splits into two questions, each with its own knobs and its own measurement
lens:

| Axis | Goal | Knob chapters | Measure with |
| ---- | ---- | ------------- | ------------ |
| **In-core** | make one kernel (one task) run full — cube+vector overlap, loop pipeline, memory | Loop & Pipeline, Mixed Kernel, Memory Placement | in-core profiling (msprof) |
| **Inter-core** | make separate tasks / cores overlap | SPMD multi-block, Dependency & Dispatch | swimlane (per-task timeline) |

> A **mixed kernel** (1 cube + 2 vector, co-scheduled) is **one task** — it shows
> as a single block on the swimlane. Its internal cube/vector overlap (including
> `pl.pipeline`'s C→V→C skew) is therefore an **in-core** concern, read from the
> in-core profile, not the swimlane.

## Tuning Loop

```text
Too slow → measure first (chapter 05)
│
├── IN-CORE (one kernel/task underutilized; read the in-core profile)
│   ├─ same-core load/compute/store not overlapping?   → chapter 01
│   │     pl.pipeline · pl.unroll · pl.parallel
│   ├─ cube (AIC) and vector (AIV) not overlapping?     → chapter 02
│   │     mixed kernel: pl.split(SplitMode) · pl.split_aiv · sync_set/wait
│   └─ memory-bound / data at the wrong level?          → chapter 04
│         memory-space hints · L1 reuse · L0 tiling · 512B align
│
└── INTER-CORE (separate tasks/cores idle or serialized; read the swimlane)
    ├─ work not spread across blocks / cores?           → chapter 02
    │     on-chip: pl.spmd · pl.cluster · pl.at · syncall
    └─ tasks serialize when they could overlap?         → chapter 03
          deps= · no_dep · dummy_task · allow_early_resolve · predicate
```

## Knob Index

**In-core** (measure with in-core profiling):

| Concern | Knobs | Chapter |
| ------- | ----- | ------- |
| Loop / pipeline overlap (same-core) | `pl.pipeline`, `pl.unroll`, `pl.parallel` | [01](01-loop-pipeline.md) |
| Mixed kernel — cube + vector overlap (one task) | `pl.split`(`SplitMode`), `pl.split_aiv`, `pl.system.sync_set`/`pl.system.sync_wait` | [02](02-split-parallel.md) |
| Memory placement | `target_memory` / `pl.Mem` hints (auto L1 reuse & L0 tiling) | [04](04-memory-placement.md) |

**Inter-core** (measure with the swimlane):

| Concern | Knobs | Chapter |
| ------- | ----- | ------- |
| On-chip SPMD multi-block | `pl.spmd`, `pl.cluster`, `pl.at`, `pl.spmd_submit`, `pl.system.syncall` | [02](02-split-parallel.md) |
| Dependency & dispatch control | `deps=`, `pl.no_dep`, `dummy_task`, `allow_early_resolve`, `predicate`, `pl.manual_scope`/`pl.submit`, `TaskId` | [03](03-dependency-dispatch.md) |

**Shared:** [Measuring Impact](05-measuring-impact.md) — benchmark (headline
wall-clock), compile profiling, in-core msprof, swimlane, codegen compare.

## How to Read This Section

Each knob chapter is organized **when to use / how to write / effect / how to
verify**, so you can jump straight to the knob your measurement points at. Two
inter-core families interact:

- On-chip dispatch (`pl.spmd`, `pl.at`, `pl.spmd_submit` in
  [chapter 02](02-split-parallel.md)) accepts the *scheduling* parameters
  (`deps=`, `allow_early_resolve=`, `predicate=`) documented in
  [chapter 03](03-dependency-dispatch.md).
- After changing dependencies, re-inspect the task graph via
  [DFX › Diagnostics Render](../dfx/02-diagnostics-render.md) (`deps.json` → HTML)
  before trusting the wall-clock number.

## See Also

- Developer reference: [`dev/01-compile-profiling.md`](../../../dev/01-compile-profiling.md), [`dev/03-runtime-dfx.md`](../../../dev/03-runtime-dfx.md)
