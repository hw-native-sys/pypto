# ExpandManualPhaseFence Pass

## Overview

`ExpandManualPhaseFence` compresses profitable full-array `TaskId`
dependencies produced by explicit `pl.submit(..., deps=[tids])` edges. It is a
narrow orchestration-only pass: when a manual-scope consumer fanout depends on
one stable, read-only `Array[TASK_ID]`, the pass inserts one dependency-only
`system.task_dummy` barrier and rewrites the covered consumers to depend on the
barrier `TaskId`.

The dependency shape changes from repeated all-to-all fanout:

```text
tids[N] -> consumers[M]
```

to one explicit phase fence:

```text
tids[N] -> system.task_dummy -> consumers[M]
```

The pass does not change kernel execution semantics. It only rewrites
`Call.attrs["manual_dep_edges"]` on selected consumer calls and adds the marked
dummy-task call that codegen lowers to `rt_submit_dummy_task(...)`.

## Position in the pipeline

```text
... -> DeriveCallDirections -> ExpandManualPhaseFence -> CollectCommGroups -> Simplify (final)
```

`DeriveCallDirections` must run first so calls carry resolved
`arg_directions` and parser/outline-produced `manual_dep_edges` are already
visible. `ExpandManualPhaseFence` runs before the final distributed metadata
collection and before orchestration codegen observes `manual_dep_edges`.

## Algorithm

For each orchestration function, the pass visits `RuntimeScopeStmt(manual=true)`
regions and analyzes each loop body:

1. **Find candidate arrays.** A candidate consumer must have exactly one
   `manual_dep_edges` entry, and that entry must be an `Array[TASK_ID]`.
2. **Estimate benefit.** The pass compares direct fanout (`N * M`) with the
   barrier shape (`N + M`). Low-benefit shapes such as `N -> 1` and `2 -> 2`
   stay direct.
3. **Reject unsafe shapes.** The pass skips mixed deps, scalar deps, unresolved
   arrays, non-loop-carried arrays defined or updated inside the same loop body,
   non-manual scopes, and non-orchestration functions.
4. **Insert a barrier.** For a profitable safe candidate, the pass creates a
   fresh `Scalar[TASK_ID]` variable and assigns it from `system.task_dummy` with
   `attrs["dummy_task"] = true` and `attrs["manual_dep_edges"] = [source_array]`.
5. **Rewrite consumers.** Covered consumer calls are rebuilt with
   `manual_dep_edges=[barrier_tid]`, leaving all other call attrs unchanged.

For sequential loops, the barrier is inserted inside the loop before the
rewritten body. For parallel loops, a barrier may be inserted before the loop
only when the dependency source is stable for that loop body. `pl.parallel`
does not weaken explicit `manual_scope` dependencies: if the body reads
`deps=[tids]` and also updates `tids[branch]`, the pass keeps direct deps.

## Fallback boundaries

The pass intentionally leaves the existing direct dependency lowering path in
place unless the pattern is clear, safe, and profitable.

Compressed:

- full-array manual-scope fanout with positive estimated edge savings;
- double-buffered phase fences where the body reads one `Array[TASK_ID]` and
  writes a different carrier such as `tids_next`.

Left direct:

- scalar TaskId deps;
- mixed scalar + array deps;
- multiple-array deps;
- partial-slot deps such as `prev = tids[i]; deps=[prev]`;
- current loop iter-arg arrays;
- arrays defined or updated inside the same loop body;
- low-benefit fanout such as `N -> 1` or `2 -> 2`;
- non-manual scopes and non-orchestration functions.

## Output invariants

After the pass:

- every inserted barrier is a `system.task_dummy` call marked with
  `attrs["dummy_task"] = true`;
- the barrier call keeps the original full-array dependency in
  `attrs["manual_dep_edges"]`;
- rewritten consumers depend on the barrier `TaskId`, not the original array;
- fallback shapes retain their original `manual_dep_edges`;
- `arg_directions` remain resolved and are not recomputed by this pass.

## Pass properties

| Field | Value |
| ----- | ----- |
| `required` | `{SSAForm, NoNestedCalls, NormalizedStmtStructure, CallDirectionsResolved}` |
| `produced` | `{SSAForm, NoNestedCalls, NormalizedStmtStructure, CallDirectionsResolved}` |
| `invalidated` | `{}` |

## Reference

- Source: [src/ir/transforms/expand_manual_phase_fence_pass.cpp](../../../../src/ir/transforms/expand_manual_phase_fence_pass.cpp)
- Header: [include/pypto/ir/transforms/passes.h](../../../../include/pypto/ir/transforms/passes.h)
- Attr keys: [include/pypto/ir/expr.h](../../../../include/pypto/ir/expr.h)
- Codegen lowering: [src/codegen/orchestration/orchestration_codegen.cpp](../../../../src/codegen/orchestration/orchestration_codegen.cpp)
- Tests:
  [tests/ut/ir/transforms/test_expand_manual_phase_fence.py](../../../../tests/ut/ir/transforms/test_expand_manual_phase_fence.py),
  [tests/ut/codegen/test_phase_fence_dep_compression.py](../../../../tests/ut/codegen/test_phase_fence_dep_compression.py)
