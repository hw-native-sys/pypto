# Automatic Task Dependency Derivation

## Status

Design proposal for the `auto-deps` branch. The pass described here is not yet
registered in the default pipeline.

## Goal

Derive task-to-task dependencies in the pass layer so users do not need to
write most `pl.submit(..., deps=[...])` edges by hand. The first target is
correctness for `with pl.manual_scope():`; normal auto scope keeps the runtime
TensorMap as a fallback until the analysis is mature.

The lowering target is the existing path:

```text
Call.attrs["manual_dep_edges"] -> orchestration codegen -> Arg::set_dependencies(...)
```

No runtime change is required for P0.

## Current Code Touchpoints

| Area | Current behavior | Auto-deps implication |
| ---- | ---------------- | --------------------- |
| `manual_dep_edges` (`include/pypto/ir/expr.h`) | Var list consumed by codegen as explicit TaskId deps | Reuse the storage format, but keep user and compiler provenance distinguishable |
| Orchestration codegen | Emits one stack `PTO2TaskId[]` and `set_dependencies` per call | Reuse this lowering once compiler deps are attached |
| `DeriveCallDirections` | Resolves `arg_directions`; manual deps are currently parser-owned | Auto-deps should run after directions are stable, not inside direction inference |
| `BufferRootCollector` | Treats `tensor.slice` as a new root for direction/codegen needs | Do not reuse as storage alias analysis |
| `OptimizeOrchTensors` | Makes static out windows explicit and proves some loop disjointness | Reuse the affine/window reasoning later to remove conservative deps |

## Analysis Model

Each task gets an access summary:

```text
TaskAccess {
  task_id_var,
  accesses: [
    { storage_root, region, direction }
  ]
}
```

- `storage_root`: allocation identity for dependency analysis. `tensor.slice`
  and view-like ops inherit the parent storage root.
- `region`: offsets, shape, stride/layout, and symbolic loop expressions
  relative to the storage root.
- `direction`: read, write, or read-write.

Dependency decisions use:

```text
NoAlias                 -> no edge
MustDisjoint            -> no edge
MayOverlap/MustOverlap  -> apply RAW/WAR/WAW hazard rules
```

Hazards:

| Current access | Prior access | Edge? |
| -------------- | ------------ | ----- |
| read | write | yes |
| write | read | yes |
| write | write | yes |
| read | read | no |

## Key Design Constraints

1. `manual_dep_edges` and `arg_directions` are additive unless a later design
   explicitly introduces a pass-owned mode. P0 must not try to cancel runtime
   direction semantics.
2. User-provided deps remain authoritative. Compiler-derived deps supplement
   them and should be tracked separately or tagged before the final merge.
3. Existing `BufferRootCollector` must remain unchanged. Auto-deps needs a new
   `StorageRootAnalysis` because the correct storage semantics differ from the
   direction/codegen root semantics.
4. Dynamic fan-in is only supported when it can be encoded as existing
   `Scalar[TASK_ID]` or fixed-size `Array[N, TASK_ID]` carries. Unsupported
   `manual_scope` cases should produce a targeted diagnostic, not a generic
   internal check.

## P0: Manual-Scope Correctness

Scope:

- Only analyze `with pl.manual_scope():`.
- Only synthesize deps when the representation is statically encodable.
- Preserve user-written `deps=[...]` and add compiler deps after de-duplication.
- Keep normal auto scope unchanged.

Implementation checklist:

1. Add `AutoDeriveTaskDependencies` as a program pass after
   `DeriveCallDirections` and before the final `Simplify`.
2. Add an internal `StorageRootAnalysis` with conservative region tracking for
   assignment, tuple get, yield, loops, `tensor.slice`, `tensor.assemble`, and
   callsite formal-to-actual substitution.
3. Generate or preserve producer TaskId variables for calls that may be used as
   dependency producers.
4. Maintain per-scope prior read/write access sets and emit compiler dependency
   edges for RAW/WAR/WAW hazards.
5. Add tests for overlap, disjoint windows conservatively treated as overlap,
   user deps plus compiler deps, and unsupported dynamic fan-in diagnostics.

## P1: Stable Storage Lineage

P1 expands the analysis without changing the runtime contract:

- Full storage lineage through nested loops, if/yield, tuple returns, and
  callsite formal-to-actual substitution.
- Integration with `MemRef::MayAlias` where MemRefs are present:
  same `base_` plus overlapping byte ranges may alias; symbolic offsets are
  conservative.
- Coverage for Group/Spmd effective directions so access summaries do not read
  raw `param_directions_` incorrectly.

## P2: Remove Conservative Edges

P2 improves parallelism:

- Reuse or factor out the affine out-window disjointness reasoning from
  `OptimizeOrchTensors`.
- Promote more `MayOverlap` cases to `MustDisjoint`.
- Avoid serializing static `pl.parallel` branches that write disjoint windows.

## P3: Static Completeness and Runtime Fallback

P3 closes correctness gaps where a single traced storage root is not expressive
enough, then defines a safe fallback when static dependency derivation cannot
reliably encode the required dependency set.

Implementation target for this phase: finite root-set lineage for `IfStmt`,
loop, and while return variables, plus whole-scope fallback to runtime tracking
when a required dependency cannot be encoded as fixed TaskId deps.

Priority order:

1. Add root-set lineage for `IfStmt` results whose branches yield different
   storage roots. For example:

   ```python
   if cond:
       selected = pl.yield_(a)
   else:
       selected = pl.yield_(b)

   out, _ = pl.submit(self.consume, selected)
   ```

   The result `selected` may alias either `a` or `b`; dependency emission must
   consider prior producers for both roots. If all producer TaskIds are
   statically available, emit deps for the full finite root set.

2. Add loop and while body-yield lineage. A loop return var must not be derived
   only from its `initValue`; the trailing `pl.yield_()` in the loop body can
   change the carried storage root:

   ```python
   selected = a
   for i, selected in pl.range(0, 4, init_values=[selected]):
       selected = pl.yield_(produced_b)

   out, _ = pl.submit(self.consume, selected)
   ```

   If the loop is known to execute at least once and the yield root is
   traceable, the return var can inherit the yield root. If the loop may execute
   zero times, or if init/yield roots differ, the return location should widen
   to a finite root set such as `{a, produced_b}`. Without root-set support this
   case must stay conservative rather than choosing one root.

3. Add whole-scope fallback to the original runtime TensorMap/OverlapMap for
   cases that cannot be recognized or are too error-prone to encode statically.
   Examples include dynamic fan-in with an unbounded number of producer TaskIds,
   dynamic gather/scatter-like aliasing, root-set explosion, missing producer
   TaskIds, or mixed control flow whose required deps are not a fixed list.
   Prefer falling back for the entire `manual_scope` rather than a single call so
   static deps and runtime TensorMap state do not disagree at segment
   boundaries.

## Open Questions

- Should compiler-derived edges use a new attr such as
  `compiler_manual_dep_edges` and merge only at codegen, or reuse
  `manual_dep_edges` with provenance stored elsewhere?
- Where should generated TaskId variables be introduced for non-`pl.submit`
  calls in normal orchestration syntax?
- Which diagnostics should be user-facing errors in `manual_scope`, and which
  should fall back to conservative deps?
