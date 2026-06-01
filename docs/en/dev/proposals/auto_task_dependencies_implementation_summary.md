# Auto Task Dependencies Implementation Summary

## Background

`manual_scope` used to depend on users writing `deps=[...]` for every required
task ordering edge. This is precise when users get it right, but it is easy to
miss producer-consumer edges once tensor aliases, slices, or control flow enter
the orchestration code.

The implementation changes `manual_scope` from a fully hand-authored dependency
mode into a compiler-assisted mode: user deps remain authoritative, while the
compiler derives additional fixed TaskId edges when the dependency can be
statically represented. Cases that cannot be represented safely are returned to
runtime TensorMap/OverlapMap tracking.

## Overall Design

### Dependency Ownership

User-authored and compiler-derived dependencies are intentionally stored in
separate attrs:

```text
manual_dep_edges              # deps written by the user
compiler_manual_dep_edges     # deps derived by AutoDeriveTaskDependencies
```

The split keeps IR provenance clear. Codegen merges and deduplicates both lists
immediately before lowering to `Arg::set_dependencies(...)`.

### Storage Access Model

The pass models each task argument as an access tuple:

```text
{ storage_root, region, direction }
```

- `storage_root` is the underlying storage identity used for dependency
  analysis. View-like values such as `tensor.slice` inherit the parent root.
- `region` is the accessed area relative to the root. Constant rectangular
  slices are tracked precisely; symbolic or unknown regions stay conservative.
- `direction` comes from resolved `arg_directions`, so the pass runs after
  `DeriveCallDirections`.

This model is separate from `BufferRootCollector` because dependency roots need
alias semantics, while buffer roots are used for direction/codegen needs.

### Hazard Rules

The pass walks each `RuntimeScopeStmt(manual=true)` in source order and compares
the current task access with prior accesses in the same manual scope.

| Current access | Prior access | Edge |
| -------------- | ------------ | ---- |
| read | write | yes |
| write | read | yes |
| write | write | yes |
| read | read | no |

When a required edge is already present in `manual_dep_edges`, the compiler does
not duplicate it in `compiler_manual_dep_edges`.

### Region and Alias Precision

Static slice regions are used to avoid unnecessary serialization. If two
constant slice windows are proven disjoint, no edge is emitted. If regions
overlap, are unknown, or include symbolic offsets, the pass treats them as
possibly overlapping.

For MemRef-backed tensor types, the pass also consults `MemRef::MayAlias`.
Different tensor variables can therefore still be ordered when their underlying
storage ranges may alias.

### Control-Flow Lineage

The storage lineage analysis keeps tensor roots through common IR constructs:

- assignments and direct aliases;
- tuple get from `pl.submit` returns;
- function output substitution through `Out` and `InOut` parameters;
- `tensor.slice` and `tensor.assemble`;
- `IfStmt` branch `pl.yield_()` values;
- loop and while iter args plus trailing body `pl.yield_()`.

For control flow, a result may have multiple possible roots. The implementation
therefore keeps a finite root set instead of collapsing to a single root. If two
branches or loop paths yield different roots, dependency emission expands across
all alternatives.

### Fallback Policy

Static dependency emission is only safe when both sides of the dependency can
be represented as a bounded set of roots and a fixed list of TaskId variables.
The implementation falls back when it detects any of these cases:

- a required hazard depends on a prior producer whose TaskId is not statically
  bound;
- dynamic fan-in from a producer inside a loop, where one scalar TaskId binding
  would not represent all runtime producer instances;
- dynamic gather/scatter-like access where the touched roots or regions depend
  on runtime indices and cannot be summarized as a bounded static access;
- control-flow joins whose required dependency set is not a fixed list, for
  example mixed branches/loops that combine dynamic producer sets;
- root-set growth beyond the pass cap for static alternatives;
- tensor arguments with dependency-relevant directions whose storage location
  cannot be resolved by the current lineage analysis.

Not every conservative case needs fallback. A symbolic slice with a known root
and a statically available producer TaskId can still be handled by emitting a
conservative edge. Finite `IfStmt` or loop root sets can also stay static when
all possible producers have encodable TaskIds.

The fallback is scope-wide by design. It avoids mixing partial compiler deps
with runtime TensorMap state at a manual/auto boundary.

## Code Changes

### IR Attrs and Codegen

- Added `kAttrCompilerManualDepEdges` in
  `include/pypto/ir/expr.h`.
- Kept `kAttrManualDepEdges` as the user-owned dependency list.
- Updated orchestration codegen to read both attrs, deduplicate them, allocate a
  fixed `PTO2TaskId[]` dependency array, guard invalid TaskIds, and call
  `set_dependencies(...)`.
- Updated the Python printer so compiler-derived deps can be surfaced in IR
  dumps.

### Pass Registration

- Added the `AutoDeriveTaskDependencies` program pass in
  `src/ir/transforms/auto_derive_task_dependencies_pass.cpp`.
- Registered the pass after `DeriveCallDirections` in the default pass manager.
- Added C++ pass declaration, Python binding, and Python type stub coverage.
- Documented the pass in the pass index and per-pass docs.

### StorageRootAnalysis

- Added an internal storage-lineage analysis for orchestration bodies.
- Tracks tensor params as full-root regions.
- Propagates storage locations through assignments, tuple gets, call outputs,
  `tensor.slice`, and `tensor.assemble`.
- Tracks constant slice boxes as root-relative regions.
- Widens unknown or symbolic slice regions to conservative overlap.
- Records MemRef information for root alternatives so aliases can be checked
  across distinct Vars.
- Merges finite root alternatives for `IfStmt`, `ForStmt`, and `WhileStmt`
  return variables.

### Dependency Mutation

- Added a mutator that only acts inside `RuntimeScopeStmt(manual=true)`.
- Maintains prior read/write accesses per manual scope.
- Summarizes each non-builtin call using resolved `arg_directions`.
- Emits compiler deps for RAW/WAR/WAW hazards when storage roots may alias and
  regions may overlap.
- Skips read-read pairs and statically disjoint regions.
- Preserves user deps and avoids duplicate compiler deps.
- Falls back the whole scope to auto mode when a required dependency cannot be
  represented as bounded roots plus fixed TaskId deps.

### TaskId Collection

- Added collection of producer TaskId variables from `pl.submit` tuple returns.
- Preserved submit TaskIds through relevant scalar DCE paths so generated deps
  can still reference the producer.
- Kept support aligned with the existing scalar TaskId and fixed TaskId array
  lowering model.

### Tests and Documentation

- Added unit tests for RAW edges, read-read no-op, auto-scope no-op, user and
  compiler edge separation, static disjoint slices, overlapping slices,
  symbolic slice conservatism, IfStmt root sets, loop yield root sets, MemRef
  may-alias, dynamic gather fallback, loop dynamic fan-in fallback, root-set cap
  fallback, and missing TaskId fallback.
- Updated pass, codegen, IR hierarchy, pass manager, and proposal docs in both
  English and zh-CN.

## Execution Effect

Common `manual_scope` producer-consumer dependencies can now be omitted by the
user and still be emitted as explicit `set_dependencies(...)` edges.

The pass improves parallelism where it can prove static slice disjointness, and
it remains conservative for symbolic regions, MemRef may-alias, and
control-flow-dependent roots.

When static dependency derivation cannot safely encode the required fixed TaskId
set, the implementation falls back to runtime TensorMap/OverlapMap tracking for
the entire scope. The resulting behavior is: derive explicit deps when the
compiler has enough information; otherwise preserve correctness through the
existing runtime dependency mechanism.
