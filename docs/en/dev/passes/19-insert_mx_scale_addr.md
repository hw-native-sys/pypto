# InsertMxScaleAddr Pass

Inserts compiler-generated `tile.tget_scale_addr` bindings before MX matmul consumers once every operand memory space is concrete.

## Overview

After [`InferTileMemorySpace`](17-infer_tile_memory_space.md) has resolved `Left` / `LeftScale` / `Right` / `RightScale` and inserted any required `tile.move`s, this pass materializes the A5 scale-address bind:

```text
bound_scale = tile.tget_scale_addr(scale, data)
matmul_mx(..., bound_scale, ...)
```

`tile.tget_scale_addr` is intentionally absent from the public `pypto.language` API. Users write only the high-level `matmul_mx` family; the compiler derives the Left/Right side from the matmul operand slots. The low-level `ir.op.tile.tget_scale_addr` remains available for compiler construction and IR parsing. It accepts only the resolved pairs `(LeftScale, Left)` and `(RightScale, Right)`.

**Pipeline position**: Immediately after [`InferTileMemorySpace`](17-infer_tile_memory_space.md), before [`ResolveBackendOpLayouts`](19-resolve_backend_op_layouts.md).

**Requirements**: `SSAForm`, `IncoreTileOps`, `SplitIncoreOrch`, `NormalizedStmtStructure`, `TileMemoryInferred`.

**Produces**: the same property set (property-preserving rewrite).

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::InsertMxScaleAddr()` | `passes.insert_mx_scale_addr()` | Program-level |

```python
from pypto.pypto_core import passes

after = passes.insert_mx_scale_addr()(passes.infer_tile_memory_space()(program))
```

Only `FunctionType::InCore` functions are rewritten.

## Algorithm

Walk each InCore body. For every `tile.matmul_mx` / `tile.matmul_mx_acc` / `tile.matmul_mx_bias` assignment:

1. Pair data/scale operand indices (`(0,1)/(2,3)` for `matmul_mx` / `_bias`; `(1,2)/(3,4)` for `_acc`).
2. Require Var-like operands (`Var` or `IterArg`) whose memory spaces already form a legal LeftScale↔Left or RightScale↔Right pair.
3. Insert `tile.tget_scale_addr(scale, data)` immediately before the matmul and rewrite the matmul to consume the bound scale SSA.

`NormalizedStmtStructure` may unwrap a single-statement `SeqStmts`, leaving a bare `AssignStmt` as an `if` / `for` / `while` (or function) body. The pass wraps that body into a `SeqStmts` when it inserts bindings, matching the bare-body handling in `InsertCommFence`.

### No cross-consumer reuse

`tget_scale_addr` mutates the physical scale buffer address in place (`dst_addr = src_addr >> SHIFT_MX_ADDR`). Ordinary SSA aliases, view aliases, and the bound result may all share that buffer, so SSA identity cannot prove that a cached binding is still live. The pass therefore emits a fresh binding for every MX matmul consumer.

This rule also applies when the scale operand is already the bound result of an earlier `tget_scale_addr`: that result aliases the same mutable buffer and cannot prove that no intervening alias changed its address. Consequently, repeated pass execution conservatively adds another binding layer; the standard lowering pipeline runs the pass once.

## Related

- Op registration / type checks: `src/ir/op/tile_ops/matmul_mx.cpp`
- Memory-space solving for MX operands: [`InferTileMemorySpace`](17-infer_tile_memory_space.md)
- PTOAS may reorder `tget_scale_addr` before Mat→Scale `tmov` (`PTOA5NormalizeTMovPass`)
