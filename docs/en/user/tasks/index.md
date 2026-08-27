# Tasks and Ordering

What must finish before a task starts — how the runtime works that out on its own, and the
interfaces for saying it yourself when it cannot.

> **Prerequisites:** [Scopes and Placement](../language/04-scopes.md) and
> [Types § parameter directions](../language/00-types.md#parameter-directions).

## What this chapter is

[Scopes and Placement](../language/04-scopes.md) answers **where** code runs. This chapter
answers **when** — the shape of the dependency graph the runtime actually executes.

Most programs never need it. The runtime derives the graph from the buffers each task
touches and the direction each parameter declares, and that inference is correct by
default. You come here when it is not enough:

- the runtime inferred an edge that is not a real dependency, and serialized work that
  could have overlapped;
- it could not infer a real edge, because the relationship is not visible as a buffer
  overlap;
- or the work itself is conditional, and you want the task skipped rather than dispatched.

## Contents

| Page | Covers |
| ---- | ------ |
| [The dependency model](00-model.md) | What a task is, how edges are derived, why statement order expresses nothing |
| [Runtime scopes](01-scopes.md) | `pl.scope` / `pl.manual_scope` / `ScopeMode` — the auto-tracking boundary |
| [Declaring an edge](02-submit.md) | TaskIds and `deps=` — via `pl.at` blocks or `pl.submit`, and fan-in through a TaskId array |
| [Refining the graph](03-tuning.md) | `pl.no_dep`, `predicate=`, `allow_early_resolve=`, `pl.system.task_dummy` |

## Reading order

Read [The dependency model](00-model.md) first — every interface in the other pages is
either a way to switch that inference off, or a way to add to it:

```text
00-model ──► 01-scopes ──► 02-submit ──► 03-tuning
  what the      where the      how to say      how to refine
  runtime       inference      an edge         what you said
  infers        is on/off      yourself
```

A useful thing to know before you start: **explicit edges and automatic tracking are not
alternatives.** `deps=` works inside an ordinary auto scope, and the final wait set is the
union of both. Reaching for `pl.manual_scope` is a separate, heavier decision — it turns
the inference off entirely and makes every edge yours to declare.

## See Also

- [Scopes and Placement](../language/04-scopes.md) — the sibling question: which hardware runs the code.
- [Types § parameter directions](../language/00-types.md#parameter-directions) — what the runtime reads to infer an edge.
- [Operations](../ops/01-catalog.md) — one-line entries for the operators named here.
- [AutoDeriveTaskDependencies](../../dev/passes/39-auto_derive_task_dependencies.md) — the pass that emits the compiler-side edges.
