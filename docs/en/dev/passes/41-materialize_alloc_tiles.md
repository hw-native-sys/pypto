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

Each handle is inserted immediately before the first top-level statement whose
subtree uses its buffer. That point (a) dominates all uses — a buffer written
across branches is first seen at the enclosing `IfStmt`, so the handle lands
before it — and (b) follows any body-defined value the handle's `TileView`
references (e.g. a runtime valid length), which a blind hoist to the function head
would precede. The scan is O(N): each top-level subtree is visited once.

## Example

```python
# Before — the phi buffer's handle would be synthesized in-branch by codegen
if flag == 0:
    result = partial        # tile
else:
    result = updated        # tile
final = pl.store(result, [0, 0], out)
```

```mlir
; After MaterializeAllocTiles (PTOAS): one handle at the function head, both
; branches and the store share it — no in-branch declaration.
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
