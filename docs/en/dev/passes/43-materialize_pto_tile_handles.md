# MaterializePTOTileHandles Pass

Materializes explicit PTO tile-buffer handles while preserving logical
`TileType` SSA. It is the verified bridge between value-oriented Tile IR and
destination-passing PTO target IR.

## Overview

The pass runs near the codegen boundary, after all logical Tile optimization,
memory planning, and orchestration passes. For each supported Tile producer it:

1. creates a dominating `pto.alloc_tile` whose result has `PTOTileBufType`;
2. maps logical Tile operands to ordered input handles;
3. maps a logical Tile result to one output handle; and
4. stores that temporary plan on the logical call for the next pass.

The logical call and its `TileType` result remain unchanged at this stage. This
keeps the transition inspectable and lets `PTOHandlesMaterialized` verify the
whole mapping before any logical Tile definition is removed.

## Default pipeline position

This pass is enabled in both `Default` and `DebugTileOptimization`, immediately
after [`ClassifyIterArgCarry`](42-classify_iter_arg_carry.md) and immediately
before [`LowerTileToPTOIR`](44-lower_tile_to_pto_ir.md). It only changes InCore
functions.

If a function contains a Tile family not yet covered by target IR, the pass
marks it `pto.target_lowering_deferred` before making any change. The complete
function then remains on legacy codegen. This explicit whole-function boundary
prevents mixed logical/target IR while target coverage is expanded.

## Allocation operands

The selected memory planner controls only the physical-address operand:

| Planner | `pto.alloc_tile` operands |
| ------- | ------------------------- |
| `PYPTO` | byte offset (`i64`), valid rows (`index`), valid columns (`index`) |
| `PTOAS` | valid rows (`index`), valid columns (`index`) |

`AllocatedMemoryAddr` is not an unconditional pass property because PTOAS
intentionally skips address allocation. In PYPTO mode the pass checks each Tile
has a MemRef with a constant byte offset and reports an error otherwise.

## Supported surface

The implementation supports physical rank-2 Tiles with static dimensions and
static or dynamic valid extents. Covered operations include load/store,
create/full, slice (`pto.subview`), alloc-backed reshape, basic unary/binary and
tile-scalar elementwise operations, move/fillpad, and row sum. It recursively
plans `ForStmt` and `IfStmt`, including Tile carry/phi aliases and required
moves. Under PTOAS, writable region-local producers are retargeted to the
shared phi/carry handle; pass-through values and metadata-only views retain an
explicit move. Tensor transfers may retain their rank-N logical partition
metadata. `pto.subview` records constant valid extents independently per
dimension in its result type while keeping runtime dimensions dynamic.

Tile parameters/returns, dynamic-shape Tensor parameters, `WhileStmt`,
cross-core pipe operations, general reductions, and remaining Tile families
are deferred as whole functions.

## Verification

`PTOHandlesMaterialized` verifies:

- every supported logical Tile producer has exactly one output handle;
- every logical Tile operand has the expected ordered input handle;
- each handle is defined exactly once by a dominating registered handle op
  (`pto.alloc_tile` or `pto.subview`);
- allocation metadata matches the selected planner;
- structured carry/phi aliases reference a defined handle and region-local
  uses satisfy dominance; and
- deferred functions contain no partially inserted target handles.

The property is included in the default verified-property set, so verification
runs at the pass boundary under normal pipeline checking.

## API

| C++ | Python |
| --- | ------ |
| `pass::MaterializePTOTileHandles()` | `passes.materialize_pto_tile_handles()` |

## Pass properties

| - | Properties |
| - | ---------- |
| Required | `SSAForm`, `SplitIncoreOrch`, `IncoreTileOps`, `HasMemRefs`, `TileOps2D`, `TileMemoryInferred`, `NormalizedStmtStructure` |
| Produced | all required properties plus `PTOHandlesMaterialized` |
| Invalidated | — |

## See also

- [LowerTileToPTOIR](44-lower_tile_to_pto_ir.md)
- [Explicit PTO Target IR](../codegen/02-explicit_pto_target_ir.md)
- [PTO codegen](../codegen/00-pto_codegen.md)
