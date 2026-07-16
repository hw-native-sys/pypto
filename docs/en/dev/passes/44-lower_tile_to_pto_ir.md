# LowerTileToPTOIR Pass

Consumes the verified logical-value-to-handle plan and rewrites the supported
logical Tile slice to destination-passing PTO target IR.

## Overview

[`MaterializePTOTileHandles`](43-materialize_pto_tile_handles.md) has already
made allocation and destination choices explicit. This pass performs a
mechanical rewrite:

| Logical operation | Target operation |
| ----------------- | ---------------- |
| `tile.load(tensor, offsets)` | `pto.tload(tensor, offsets, extents, output_handle)` |
| `tile.sqrt(input)` | `pto.tsqrt(input_handle, output_handle)` |
| `tile.add(lhs, rhs)` | `pto.tadd(lhs_handle, rhs_handle, output_handle)` |
| `tile.mul(lhs, rhs)` | `pto.tmul(lhs_handle, rhs_handle, output_handle)` |
| `tile.row_sum(input, tmp)` | `pto.trowsum(input_handle, tmp_handle, output_handle)` |
| `tile.slice(input, ...)` | logical result removed; Step-3 `pto.subview` retained |
| `tile.store(input, offsets, tensor)` | `pto.tstore(input_handle, offsets, extents, tensor)` |

The target calls return no logical value and appear in `EvalStmt`. Their
registered operand schemas describe reads, writes, metadata, and destination
placement. `PTOTileBufType` handles remain SSA values even though target
operations mutate the storage named by those handles.

Logical `tile.alloc` pointer tokens are removed because their only purpose was
to root Tile MemRefs. `tile.store` tensor results become ordinary tensor aliases
after the side-effecting target store is emitted.

## Default pipeline position

This is the last pass in `Default` and `DebugTileOptimization`. It runs after
all logical Tile consumers and produces the IR consumed by `PTOIRPrinter`.
Functions marked `pto.target_lowering_deferred` by the preceding pass are left
unchanged and remain wholly on legacy logical-Tile codegen.

## Resulting invariant

The `PTOBufferized` property requires that an InCore target function:

- contains no logical `TileType` parameter, assignment, return, or call;
- defines every PTO buffer handle once by `pto.alloc_tile`;
- uses each handle only after its dominating definition;
- places allocating and destination-passing ops in the required statement form;
- matches every target op's registered operand and result contract; and
- preserves structured `ForStmt`/`IfStmt` regions while excluding Tile handles
  from `scf` SSA results/carries.

The property is included in the default verified-property set. PTO codegen
detects target IR and delegates the complete program to the mechanical printer.
Synthetic SPMD identity parameters are explicit in this target signature but
remain a private codegen suffix: orchestration call arity continues to use only
the function's public parameters.

## API

| C++ | Python |
| --- | ------ |
| `pass::LowerTileToPTOIR()` | `passes.lower_tile_to_pto_ir()` |

## Pass properties

| - | Properties |
| - | ---------- |
| Required | `SSAForm`, `SplitIncoreOrch`, `IncoreTileOps`, `HasMemRefs`, `TileOps2D`, `TileMemoryInferred`, `NormalizedStmtStructure`, `PTOHandlesMaterialized` |
| Produced | `SSAForm`, `SplitIncoreOrch`, `NormalizedStmtStructure`, `PTOBufferized` |
| Invalidated | `IncoreTileOps`, `HasMemRefs`, `AllocatedMemoryAddr`, `TileOps2D`, `TileMemoryInferred`, `PTOHandlesMaterialized` |

## Limitations

The pass intentionally does not analyze aliasing, choose output buffers, infer
transfer extents, or recover addresses. Those decisions belong in the verified
Step-3 plan. Extending operation coverage therefore requires updating the op
schema, Step 3, Step 4, both property verifiers, printer support, and PTOAS
tests together.

## See also

- [MaterializePTOTileHandles](43-materialize_pto_tile_handles.md)
- [Explicit PTO Target IR](../codegen/02-explicit_pto_target_ir.md)
- [PTO codegen](../codegen/00-pto_codegen.md)
