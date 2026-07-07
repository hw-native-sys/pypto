# MaterializeAllocTiles Pass

Promotes the PTO tile **handle** from a codegen-synthesized artifact to an
explicit `alloc_tile` IR op, so PTO codegen becomes a strict 1-to-1 emitter. For
each distinct tile buffer a function uses, this pass inserts exactly one
`alloc_tile(base, byte_offset, shape)` op at a scope that dominates every use of
the buffer (issue #1956).

## Overview

Before this pass, the tile handle (`pto.alloc_tile`) was not an IR node — PTO
codegen synthesized it at each tile variable's definition site. When that site
was inside an `if`/`else` branch (an if/else-yield **phi** producer), the handle
was *declared inside one branch but read from another* — an undeclared-SSA
scoping violation. Under `memory_planner=PYPTO` a shared `addr` masked the
problem; under `memory_planner=PTOAS` (where `MemoryReuse` is skipped) it
miscompiled: the phi read a buffer no branch had written.

Making the handle a first-class op fixes this at the root:

- The handle is placed **once**, at a scope that dominates all uses — so a
  buffer written across branches is declared before the enclosing `if`, never
  inside it.
- Codegen degrades to a straight transcription: it emits `pto.alloc_tile` 1-to-1
  from these ops and resolves every tile variable to its buffer's handle (see
  [PTO codegen](../codegen/00-pto_codegen.md#allocation-generation)). No branch
  is a handle-declaration site any more.

**When to use**: dead last in the `Default` and `DebugTileOptimization`
strategies — after MemRefs and addresses are final (`InitMemRef`,
`MaterializeSemanticAliases`, and under PYPTO `MemoryReuse` + `AllocateMemoryAddr`)
and after the final `Simplify`, so no DCE removes the deliberately-unused handle
variables and no earlier transform has to reason about the `alloc_tile` op. Runs
under **both** memory planners.

**Scope**: only functions that hold `TileType` variables. Orchestration
functions (which submit tasks and never hold tiles) are returned unchanged.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::MaterializeAllocTiles()` | `passes.materialize_alloc_tiles()` | Function-level |

```python
from pypto.pypto_core import passes

materialized = passes.materialize_alloc_tiles()(program)
```

## Behavior

### Which buffers get a handle

`BufferCollector` records one representative per distinct tile buffer, in
first-seen program order, across every construct that codegen must emit a handle
for: `AssignStmt`-defined tiles, `ForStmt` / `WhileStmt` iter-args and
return-vars, and `IfStmt` return-vars. Function params (bound to `%argN`) and
MemRef-less tiles (a cross-core `tpop` result living in the reserved C2V slot, and
views over it) get no handle.

### Handle granularity depends on the memory planner

The grouping key (`BufferKey`, kept in lock-step with codegen's
`BufferHandleKey`) is chosen from `PassContext::GetMemoryPlanner()`:

| Planner | Group key | Rationale |
| ------- | --------- | --------- |
| `PYPTO` (default) | MemRef identity (base + byte_offset + size) **+ `TileBufSignature`** | Handles carry an explicit `addr`, so several typed handles may alias one address. Splitting by signature (memory space, dtype, physical shape, layout, fractal, **pad**) keeps distinct pad / shape / layout on one byte-slot as separate handles — matching the pre-#1956 per-variable model. |
| `PTOAS` | MemRef identity only | ptoas allocates one buffer per handle, so a byte-slot must map to exactly one handle (pad differences collapse — a pre-existing PTOAS trait). |

The `TileBufSignature` key deliberately **excludes** the valid-shape extent: a
`tile.set_validshape` narrowing produces the same physical type, and the extent
rides on the emitted `valid_row` / `valid_col` operands — so such tiles share one
handle. True aliases (loop-carry, if/else-yield phi, in-place op results) share
one signature and therefore one handle under both planners.

### Deps-aware placement

Each handle is placed at the **smallest scope that dominates all uses of its
buffer** and follows the handle's `TileView` operand dependencies, found by
recursing the statement tree:

- A buffer used across several statements — or across `if`/`else` branches, or as
  a loop carry (iter-arg / return-var) — is placed at the current level, before
  its first use. A phi buffer written in both branches is thus declared before the
  enclosing `IfStmt`, dominating both.
- A buffer used **entirely within one nested loop or branch body** descends into
  that scope, so a handle whose `valid_shape` references a loop-body scalar
  (e.g. `valid = i + 1; t = load(..., valid_shapes=[.., valid])`) lands *after*
  that scalar — not hoisted above the loop where the operand is out of scope.

Each statement is rescanned once per enclosing level, so the pass is O(N × depth)
(nesting depth is bounded).

### Dynamic-valid fixup

Deps-aware placement keeps a handle below a *single* loop-body `valid_shape`
operand, but a buffer that memory-reuse shares across **several sibling scopes**
(e.g. one `mem_vec` slot loaded in two consecutive loops, each with its own
loop-local `valid_len`) has no single scope that both dominates every use *and*
sees every operand — it is hoisted to the common ancestor, above all of them.
Emitting the handle's dynamic `valid_col` there would reference an out-of-scope
operand, which ptoas rejects (`'pto.alloc_tile' op valid_col operand is required
because result type v_col is ?`).

A final pass over the placed body repairs this. Walking the tree while tracking
the in-scope variables, for each `alloc_tile` whose `valid_shape` operand is
**not** in scope at the handle's position it:

1. declares the handle with a **static** valid (the physical shape — a
   self-contained, hoistable operand), and
2. re-establishes each use's real valid by injecting a `tile.set_validshape` on
   the handle **immediately before** the dynamic-valid producer, where the
   operand *is* in scope. It must precede the producer (not follow it): a
   `tile.load`'s fill/pad extent follows the destination tile's valid, so the
   buffer must already carry `valid_len` when the load writes — emitting it after
   leaves the buffer at its static valid during the load and corrupts
   partial-valid blocks.

```mlir
%h = pto.alloc_tile addr = %c0 valid_row = %c16 valid_col = %c64 : ...   # static
scf.for ... {                                     # pass 1
  %valid_len = scf.if ...
  pto.set_validshape %h, %c16, %valid_len : ...   # injected — real valid BEFORE the load
  pto.tload ... outs(%h)
  pto.tfillpad ins(%h) ...                        # pads valid_len..64 with min
}
scf.for ... { pto.set_validshape %h, %c16, %valid_len_p2 ... pto.tload ... }   # pass 2
```

This is semantics-preserving (the injected `set_validshape` restores the valid
the descended alloc would otherwise carry) and keeps the strict one-handle
per-buffer model. Because this pass is dead-last, no later DCE drops the
result-unused `set_validshape` side-effect ops.

## Example

```python
# Before — the phi buffer's handle would be synthesized in-branch by codegen
if flag == 0:
    result = partial        # tile
else:
    result = updated        # tile
final = pl.store(result, [0, 0], out)
```

The IR op is `alloc_tile(base, byte_offset, shape)`; PTO codegen emits it as
`pto.alloc_tile`, lowering the tile's `TileView` valid extent into the
`valid_row` / `valid_col` operands shown below (`v_row=?, v_col=?` in the type):

```mlir
; After MaterializeAllocTiles (PTOAS): one handle before the if (it dominates
; both branches), which the branches and the store share — no in-branch decl.
%res__buf1 = pto.alloc_tile valid_row = %c64_index valid_col = %c64_index : !pto.tile_buf<...>
scf.if %flag {
  pto.tmul ... outs(%res__buf1 : !pto.tile_buf<...>)
} else {
  pto.tadd ... outs(%res__buf1 : !pto.tile_buf<...>)
}
pto.tstore ins(%res__buf1 : !pto.tile_buf<...>) outs(...)
```

Under `PYPTO` each `alloc_tile` additionally bakes `addr = <byte_offset>`.

## Verification

**Tests**: `tests/ut/ir/operators/test_alloc_tile_op.py` (op construction + type
deduction), `tests/ut/codegen/test_memory_planner_switch.py` (single-handle
aliasing under both planners), `tests/ut/codegen/test_pto_codegen.py` (fillpad /
if-phi handle granularity), and
`tests/ut/ir/transforms/test_verify_alloc_tile_dominates.py` (the produced
property, below).

This pass produces the `AllocTileDominatesUses` property, checked by a
[verifier](99-verifier.md) that flags any tile use whose buffer lacks a dominating
`alloc_tile` handle.

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `SplitIncoreOrch`, `IncoreTileOps`, `HasMemRefs`, `NormalizedStmtStructure` |
| Produced | `NormalizedStmtStructure`, `AllocTileDominatesUses` |
| Invalidated | — |
