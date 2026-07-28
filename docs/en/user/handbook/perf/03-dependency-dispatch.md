# Performance Tuning: Dependency & Dispatch Control

> **Status:** DRAFT skeleton. Shape the task graph so independent work overlaps
> and conditional work is skipped. These knobs control *scheduling*, not compute.

## Task Graph Basics

*TODO:*

- `pl.manual_scope` / `pl.scope` / `ScopeMode` — manual task-graph scopes.
- `pl.submit(...)` / `pl.spmd_submit(...)` — launch a task, capture its
  `Scalar[TASK_ID]`.
- `pl.TaskId` / `pl.TASK_ID` — the dependency handle type.

## Explicit Dependencies

*TODO:*

- `deps=[tid, ...]` — declare producer edges explicitly (union'd with auto-deps).
- `pl.no_dep(arg)` / `pl.adir.no_dep` — opt a single argument out of automatic
  dependency inference.

> **Correctness, not just performance — lost WAR dependencies.** In AUTO scopes
> the runtime currently omits **write-after-read** edges for loop-carried buffers,
> which silently corrupts results (**issue #2058**). Until auto-detection lands you
> must add the edge by hand so `reader(N)` finishes before `writer(N+1)` overwrites
> the buffer:
>
> ```python
> _, tid_read  = pl.submit(self.reader, buf, ...)
> _, tid_write = pl.submit(self.writer, buf, ..., deps=[tid_read])
> ```
>
> Full explanation: [Precision › Lost WAR Dependencies](../precision/00-workflow.md#known-gotcha-lost-war-dependencies-loop-carried-buffers).

## Anchor / Barrier Tasks

*TODO:*

- `dummy_task` — a dependency anchor / barrier with no compute; when and why to
  insert one.

## Speculative & Conditional Dispatch

*TODO:*

- `allow_early_resolve=True` — let the scheduler pre-stage a task's consumers
  before it completes (speculative early dispatch). Trade-offs and correctness
  conditions.
- `predicate=(...)` — a single-comparison **dispatch predicate** that skips a
  task entirely when the gate is false (e.g. skip an expert whose row count is
  0). Pair with `deps=` to avoid reading a stale gate value.

## How These Interact

*TODO — table: `deps` vs `no_dep` vs `allow_early_resolve` vs `predicate`,
combination constraints, common pitfalls (e.g. `predicate` without `deps`).*

## Verifying the Effect

Inspect the resulting graph via [DFX › Diagnostics Render](../dfx/02-diagnostics-render.md)
(`deps.json` → HTML), then re-measure with [Measuring Impact](05-measuring-impact.md).
