# AutoDeriveTaskDependencies Pass

## Overview

`AutoDeriveTaskDependencies` derives conservative task-to-task dependency
edges inside AUTO runtime scopes when explicitly enabled. It runs after
[`DeriveCallDirections`](34-derive_call_directions.md), reads the resolved
`Call.attrs["arg_directions"]`, and writes compiler-owned producer TaskId
edges to `Call.attrs["compiler_manual_dep_edges"]`. When a tensor slot's
runtime dependency lookup is fully covered by explicit user or compiler edges
in the same successfully analyzed scope, the pass can also rewrite selected
call-site directions:

- read-only `ArgDirection::Input` -> `ArgDirection::NoDep`;
- read-write `ArgDirection::InOut` -> `ArgDirection::OutputExisting`.

User-written `pl.submit(..., deps=[...])` edges remain in
`Call.attrs["manual_dep_edges"]`. The two attrs are intentionally separate so
IR dumps preserve provenance; orchestration codegen merges and deduplicates
them immediately before emitting `Arg::set_dependencies(...)`.

## Position in the pipeline

```text
... -> DeriveCallDirections
    -> AutoDeriveTaskDependencies
    -> ExpandManualPhaseFence
    -> CollectCommGroups
    -> Simplify (final)
```

User-written MANUAL regions keep their scheduling contract: explicit
`deps=[...]` remain the only task dependency edges, and the pass never adds
compiler deps, demotes the scope, or rewrites call-site directions to `NoDep`
or `OutputExisting`. AUTO regions are analyzed only when the compile-time
`analyze_auto_scopes_for_deps` switch is enabled. Hand-placed AUTO
`RuntimeScopeStmt` nodes keep `manual=false` in the output IR. For default
`auto_scope=True` orchestration functions, the pass runs before
`MaterializeRuntimeScopes`; when AUTO analysis is enabled, it uses an
analysis-only virtual AUTO region around the function body and does not insert
or move scope wrappers itself. When analysis proves a whole default-mode
function body or a future for/if partition is fully covered by compiler-derived
deps, it records compiler-only markers that `MaterializeRuntimeScopes` consumes
to emit a compiler-owned MANUAL `RuntimeScopeStmt` for that region. Otherwise
`MaterializeRuntimeScopes` emits normal AUTO wrappers and runtime
OverlapMap/TensorMap tracking remains enabled. Compiler-derived edges, when
statically encodable, are emitted on top through `Arg::set_dependencies(...)`.

## Algorithm

For each function body:

1. Build a conservative storage-location map for tensor Vars. Direct aliases,
   loop carries, tuple elements, `tensor.assemble`, and cross-function outputs
   inherit the same storage root and region when the root can be traced.
2. Preserve storage lineage through `IfStmt.return_vars` by merging finite
   branch root sets. Different branch roots are retained as alternatives, not
   dropped. Matching regions are preserved; differing regions for the same root
   widen to unknown.
3. Preserve loop and while return lineage by merging the initial carried value
   with the trailing body `pl.yield_()` value, then widening regions to unknown
   because the final iteration source is control-flow dependent.
4. Track constant rectangular `tensor.slice` windows as regions relative to the
   storage root only for bare tensors or packed ND `TensorView` tensors. Slices
   with symbolic shape/offset, strided views, non-ND layouts, `valid_shape`, or
   padding fall back to an unknown region and overlap conservatively.
5. Treat MemRef-backed shaped values as aliases when `MemRef::MayAlias` reports
   the same base allocation with overlapping or symbolic byte ranges.
6. Collect statically bound producer TaskIds from `pl.submit` tuple tails.
   `Array[TASK_ID]` dependency values are expanded conservatively when their
   lineage is known: direct scalar writes, dynamic loop writes, and synthesized
   per-element `arr[i]` dep arrays can cover user-written hazards only when all
   static array slots are covered. Unknown or partially covered arrays remain
   unexpanded so missing dependencies still trigger fallback.
7. Walk each `RuntimeScopeStmt` in source order, maintaining prior accesses for
   the current analyzable region. When a scope finishes without fallback, export
   its access records to the enclosing region so later sibling or parent-scope
   consumers can depend on the producer TaskIds. For default `auto_scope=True`
   orchestration functions with no materialized scope yet, use the whole
   function body as a virtual AUTO analysis region. For AUTO scopes this is
   analysis-only; the final scope mode remains AUTO unless
   `MaterializeRuntimeScopes` later consumes compiler auto-manual markers for a
   fully covered default-mode region.
8. For every non-builtin call with resolved `arg_directions`, classify tensor
   arguments as read, write, or read-write. Accesses to the same storage root,
   or to MemRef roots that may alias, are considered for region overlap.
9. Skip dependency edges for statically proven disjoint regions. Otherwise, add
   a compiler edge from any prior producer TaskId when RAW, WAR, or WAW hazards
   exist. Read-read pairs do not produce edges. User-written edges are respected
   and not duplicated.
10. For read-only `Input` arguments in analyzed AUTO scopes whose precise
    static region (`Full` or `Box`) or known-root conservative region has RAW
    hazards covered by user or compiler explicit deps inside the same
    successfully analyzed runtime scope, rewrite the call-site direction to
    `NoDep` so codegen emits `add_no_dep(...)`.
11. For `InOut` arguments in analyzed AUTO scopes, if every overlapping RAW,
    WAR, or WAW hazard for that slot is covered by user or compiler explicit
    deps, rewrite the call-site direction to `OutputExisting`. This skips the
    slot's TensorMap lookup while still publishing the write as an output.
    MANUAL scopes, `OutputExisting` slots, dynamic-window regions whose slice
    window depends on runtime values, dynamic-producer cases without complete
    user fan-in coverage, and cross-scope cases keep their original directions.

If dependency-relevant tensor access cannot be represented as bounded static
roots plus fixed TaskId deps in an analyzed AUTO scope, the pass strips any
partial compiler-derived deps from the whole enclosing region and leaves it as
AUTO.
Implemented fallback triggers include:

- a required hazard whose prior producer TaskId was not statically bound;
- a prior producer inside a loop whose TaskId cannot be represented by a fixed
  loop-return value or by a statically sized TaskId array fan-in;
- dynamic gather/scatter-like tensor values whose accessed region depends on
  runtime indices;
- root-set lineage that exceeds the pass cap for static alternatives;
- tensor arguments with read/write directions whose storage location cannot be
  resolved by the current lineage analysis.

This leaves the entire AUTO region on runtime OverlapMap/TensorMap tracking
instead of mixing partial compiler deps with runtime state at scope boundaries.
User-written MANUAL scopes are skipped by this pass, so AUTO fallback handling
does not apply inside them.

## Default-Path Changes

- MANUAL scopes do not get compiler-derived dependency edges or automatic
  direction rewrites. User-written `deps=[...]` remain the only dependency
  source inside `pl.manual_scope()`, and the scope stays MANUAL.
- AUTO-scope analysis is opt-in. With the default switch value, AUTO runtime
  scope mode and TensorMap/OverlapMap tracking remain unchanged.
- Dead scalar assignment elimination preserves TaskId tuple-element extracts in
  all builds. This can leave a cheap scalar TaskId local that older pipelines
  might have removed, so dependency derivation/codegen can recover producer task
  ids.

## Properties

| Required | Produced | Invalidated |
| -------- | -------- | ----------- |
| `SplitIncoreOrch`, `CallDirectionsResolved` | `CallDirectionsResolved` | — |

The pass preserves `CallDirectionsResolved`: it does not rewrite call arguments,
and any `arg_directions` changes remain verifier-legal direction demotions for
tensor slots whose explicit dependencies are already present in analyzed AUTO
scopes: `Input -> NoDep` for covered read-only inputs and
`InOut -> OutputExisting` for covered read-write slots.

## API

- C++: `pass::AutoDeriveTaskDependencies()`
- Python: `passes.auto_derive_task_dependencies()`
- Level: program-level

## References

- Source: [pass source][pass-source]
- Proposal: [Automatic Task Dependency Derivation](../proposals/auto_task_dependencies.md)
- Lowering: [Orchestration Code Generation][orchestration-lowering]

[pass-source]: ../../../../src/ir/transforms/auto_derive_task_dependencies_pass.cpp
[orchestration-lowering]: ../codegen/01-orchestration_codegen.md#manual-scope-and-taskid-lowering
