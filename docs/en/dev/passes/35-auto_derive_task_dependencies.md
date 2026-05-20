# AutoDeriveTaskDependencies Pass

## Overview

`AutoDeriveTaskDependencies` derives conservative task-to-task dependency
edges inside `with pl.manual_scope():` regions. It runs after
[`DeriveCallDirections`](33-derive_call_directions.md), reads the resolved
`Call.attrs["arg_directions"]`, and writes compiler-owned producer TaskId
edges to `Call.attrs["compiler_manual_dep_edges"]`.

User-written `pl.submit(..., deps=[...])` edges remain in
`Call.attrs["manual_dep_edges"]`. The two attrs are intentionally separate so
IR dumps preserve provenance; orchestration codegen merges and deduplicates
them immediately before emitting `Arg::set_dependencies(...)`.

## Position in the pipeline

```text
... -> DeriveCallDirections -> AutoDeriveTaskDependencies -> CollectCommGroups -> Simplify (final)
```

The pass only changes manual runtime scopes. Auto scopes keep the runtime
OverlapMap behaviour unchanged.

## Algorithm

For each function body:

1. Build a conservative storage-root map for tensor Vars. Direct aliases,
   loop carries, tuple elements, `tensor.slice`, `tensor.assemble`, and
   cross-function outputs inherit the same root when the root can be traced.
2. Preserve storage lineage through `IfStmt.return_vars` when both branches
   yield the same traced storage root.
3. Treat MemRef-backed shaped values as aliases when `MemRef::MayAlias` reports
   the same base allocation with overlapping or symbolic byte ranges.
4. Collect statically bound producer TaskIds from `pl.submit` tuple tails.
5. Walk each `RuntimeScopeStmt(manual=true)` in source order, maintaining prior
   accesses for that manual scope only.
6. For every non-builtin call with resolved `arg_directions`, classify tensor
   arguments as read, write, or read-write. Accesses to the same storage root,
   or to MemRef roots that may alias, are treated as may-overlap.
7. Add a compiler edge from any prior producer TaskId when RAW, WAR, or WAW
   hazards exist. Read-read pairs do not produce edges. User-written edges are
   respected and not duplicated.

If a hazard depends on a prior producer whose TaskId was not statically bound,
the pass raises a targeted `ValueError` asking the user to submit the producer
as `out, tid = pl.submit(...)`.

## Properties

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `SplitIncoreOrch`, `CallDirectionsResolved` | `CallDirectionsResolved` | — |

The pass preserves `CallDirectionsResolved`: it rewrites only dependency attrs,
not call arguments or `arg_directions`.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::AutoDeriveTaskDependencies()` | `passes.auto_derive_task_dependencies()` | Program-level |

## References

- Source: [src/ir/transforms/auto_derive_task_dependencies_pass.cpp](../../../../src/ir/transforms/auto_derive_task_dependencies_pass.cpp)
- Proposal: [Automatic Task Dependency Derivation](../proposals/auto_task_dependencies.md)
- Lowering: [Orchestration Code Generation](../codegen/01-orchestration_codegen.md#manual-scope-and-taskid-lowering)
