# InsertMxScaleAddr Pass

Materializes A5 MX scale address bindings once the operand memory spaces are concrete.

## Overview

After [`InferTileMemorySpace`](22-infer_tile_memory_space.md) has resolved `Left` / `LeftScale` / `Right` / `RightScale` and inserted any required `tile.move`s, this pass binds the destination of an eligible Mat→Scale extract before writing the scale bytes.

```text
dst = tile.bind_scale_addr(data, shape, valid_shape, target_memory=LeftScale)
scale = tile.extract_into(dst, scale_panel, row, col)
matmul_mx(..., scale, ...)
```

`tile.bind_scale_addr` creates a scale handle whose address is derived from the
Left/Right payload. It moves no scale bytes. `tile.extract_into` writes into the
handle and returns an SSA value that aliases it. The PTO emitters map the
two operations to `pto.tget_scale_addr` and `pto.textract`, respectively.
Other producers continue to use `tile.tget_scale_addr(scale, data)`. The
address operations accept the resolved `(LeftScale, Left)` and
`(RightScale, Right)` pairs for FP8E4M3FN, FP8E5M2, or packed FP4 payloads. The new
operations are available through the IR builder for pass dumps and parsing.

**Pipeline position**: Immediately after [`InferTileMemorySpace`](22-infer_tile_memory_space.md), before [`ResolveBackendOpLayouts`](24-resolve_backend_op_layouts.md).

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

Rewrites all InCore-variant functions (`FunctionType::InCore` / `AIC` / `AIV`). Mixed kernels and frontend-written cube/vector functions use AIC/AIV and must receive the same bindings.

## Algorithm

Walk each InCore-variant body. For every `tile.matmul_mx` / `tile.matmul_mx_acc` / `tile.matmul_mx_bias` assignment:

1. Pair data/scale operand indices (`(0,1)/(2,3)` for `matmul_mx` / `_bias`; `(1,2)/(3,4)` for `_acc`).
2. Require Var-like operands (`Var` or `IterArg`) whose memory spaces already form a legal LeftScale↔Left or RightScale↔Right pair.
3. If the scale comes from a Mat→Scale extract in the same straight-line region
   and the payload is already available there, replace the extract in place by
   a bind and an explicit write. An intervening use or definition on the same
   L0 side or a control-flow boundary invalidates the candidate. The
   opposite side is tracked independently.
4. For other producers, reuse the latest valid binding for the same scale/data
   SSA pair or insert `tile.tget_scale_addr(scale, data)` before the matmul.
   The matmul consumes the resulting scale SSA.

`NormalizedStmtStructure` may unwrap a single-statement `SeqStmts`, leaving a bare `AssignStmt` as an `if` / `for` / `while` (or function) body. The pass wraps that body into a `SeqStmts` when it inserts bindings, matching the bare-body handling in `InsertCommFence`.

### Conservative binding reuse

`tget_scale_addr` mutates the physical scale buffer address in place (`dst_addr = src_addr >> SHIFT_MX_ADDR`). Repeating an unchanged binding after a scale consumer makes PTOAS attempt to hoist it across that consumer to the original Mat→Scale fill, which PTOAS 0.70 rejects. The pass therefore retains only the latest scale/data binding for each of the Left and Right sides within straight-line code.

SSA identity alone is insufficient because aliases may share a mutable buffer. A different pair replaces that side's state; any non-MX operation taking or producing a Left/LeftScale or Right/RightScale tile invalidates the corresponding side. Unknown calls and control-flow boundaries clear both sides. Mat loads and Acc stores do not invalidate these bindings. State tracking takes constant space and preserves the pass's linear traversal.

An explicit earlier `tget_scale_addr` also invalidates the state. Consequently, repeated pass execution conservatively adds another binding layer; the standard lowering pipeline runs the pass once.

## Related

- Op registration / type checks: `src/ir/op/tile_ops/matmul_mx.cpp`
- Memory-space solving for MX operands: [`InferTileMemorySpace`](22-infer_tile_memory_space.md)
- PTOAS may reorder `tget_scale_addr` before Mat→Scale `tmov` (`PTOA5NormalizeTMovPass`)
