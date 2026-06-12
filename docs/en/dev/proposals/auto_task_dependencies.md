# Automatic Task Dependency Derivation

## Status

Implemented design for the `auto-deps` branch. The pass is registered in the
default pipeline after `DeriveCallDirections`. Current policy keeps
`with pl.manual_scope():` fully user-managed: the pass does not analyze or
demote manual scopes. AUTO-scope analysis remains opt-in through the
compile-time `analyze_auto_scopes_for_deps` switch.

## Goal

Derive task-to-task dependencies in the pass layer for AUTO scopes when callers
explicitly enable it. User-written `with pl.manual_scope():` regions remain a
manual scheduling contract: users must write the required `deps=[...]` edges by
hand, and the compiler does not add or infer more edges there. Normal AUTO scope
keeps runtime TensorMap/OverlapMap tracking unless callers explicitly enable
AUTO-scope analysis.

The lowering target is the existing path:

```text
Call.attrs["manual_dep_edges"] -> orchestration codegen -> Arg::set_dependencies(...)
```

No runtime implementation change is required for the current policy. The
compiler may emit additional `Arg::set_dependencies(...)` calls for analyzed
AUTO scopes, but it does not rewrite user-written MANUAL scopes to AUTO.

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
4. Dynamic fan-in is only supported in analyzed AUTO scopes when it can be
   encoded as existing `Scalar[TASK_ID]` or fixed-size `Array[N, TASK_ID]`
   carries. Unsupported AUTO cases keep AUTO runtime tracking rather than
   leaving partial compiler deps in place.

## P0: AUTO-Scope Opt-In Derivation

Scope:

- Do not analyze `with pl.manual_scope():`; user-written manual scopes are
  honored verbatim.
- Keep AUTO-scope analysis disabled by default; callers must opt in with
  `analyze_auto_scopes_for_deps=True`.
- Only synthesize deps when the representation is statically encodable.
- Preserve user-written `deps=[...]` and add compiler deps after de-duplication.
- If an analyzed AUTO scope cannot be completely encoded as fixed TaskId deps,
  strip partial compiler deps and rely on AUTO TensorMap/OverlapMap tracking.

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
5. Add tests for overlap, proven-disjoint static windows, user deps plus
   compiler deps, and unsupported dynamic fan-in whole-scope fallback.

## Default-Path Effects

- MANUAL scopes in the default pipeline are not analyzed by
  `AutoDeriveTaskDependencies`; user-written `deps=[...]` remain the sole
  dependency source and the scope stays MANUAL.
- AUTO scopes are analyzed only when `analyze_auto_scopes_for_deps=True`. If
  analysis cannot safely encode all required deps, the pass strips any partial
  compiler-derived deps and leaves the scope AUTO so TensorMap/OverlapMap can
  conservatively track it.
- Dead scalar assignment elimination now preserves TaskId tuple-element
  extracts unconditionally. This is a small default-path change that may keep a
  cheap scalar TaskId local which was previously removed, so later dependency
  passes and codegen can still recover producer task ids.

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
loop, and while return variables, plus whole-AUTO-region fallback to runtime
tracking when a required dependency cannot be encoded as fixed TaskId deps.

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
   analyzed AUTO cases that cannot be recognized or are too error-prone to encode statically.
   Examples include dynamic fan-in with an unbounded number of producer TaskIds,
   dynamic gather/scatter-like aliasing, root-set explosion, missing producer
   TaskIds, or mixed control flow whose required deps are not a fixed list.
   Prefer falling back for the entire AUTO analysis region rather than a single
   call so static deps and runtime TensorMap state do not disagree at segment
   boundaries.

## Open Questions

- Should compiler-derived edges use a new attr such as
  `compiler_manual_dep_edges` and merge only at codegen, or reuse
  `manual_dep_edges` with provenance stored elsewhere?
- Where should generated TaskId variables be introduced for non-`pl.submit`
  calls in normal orchestration syntax?
- Which diagnostics should be user-facing errors for opt-in AUTO derivation, and
  which should fall back to conservative runtime tracking?
