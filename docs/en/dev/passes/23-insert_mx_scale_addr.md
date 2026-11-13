# InsertMxScaleAddr Pass

Inserts compiler-generated `tile.tget_scale_addr` bindings before MX matmul consumers once every operand memory space is concrete.

## Overview

After [`InferTileMemorySpace`](22-infer_tile_memory_space.md) has resolved `Left` / `LeftScale` / `Right` / `RightScale` and inserted any required `tile.move`s, this pass materializes the A5 scale-address bind:

```text
bound_scale = tile.tget_scale_addr(scale, data)
matmul_mx(..., bound_scale, ...)
```

`tile.tget_scale_addr` is intentionally absent from the public `pypto.language` API. Users write only the high-level `matmul_mx` family; the compiler derives the Left/Right side from the matmul operand slots. The low-level `ir.op.tile.tget_scale_addr` remains available for compiler construction and IR parsing. It accepts only the resolved pairs `(LeftScale, Left)` and `(RightScale, Right)`, with `FP8E4M3FN` or supported native packed-FP4 data tiles.

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
3. Reuse the latest binding on that side only if both SSA operands are unchanged and no invalidating operation intervened. Otherwise insert `tile.tget_scale_addr(scale, data)` immediately before the matmul. Rewrite the matmul to consume the bound scale SSA.

`NormalizedStmtStructure` may unwrap a single-statement `SeqStmts`, leaving a bare `AssignStmt` as an `if` / `for` / `while` (or function) body. The pass wraps that body into a `SeqStmts` when it inserts bindings, matching the bare-body handling in `InsertCommFence`.

### Conservative binding reuse

`tget_scale_addr` mutates the physical scale buffer address in place (`dst_addr = src_addr >> SHIFT_MX_ADDR`). Repeating an unchanged binding after a scale consumer makes PTOAS attempt to hoist it across that consumer to the original Mat→Scale fill, which PTOAS 0.70 rejects. The pass therefore retains only the latest scale/data binding for each of the Left and Right sides within straight-line code.

SSA identity alone is insufficient because aliases may share a mutable buffer. A different pair replaces that side's state; any non-MX operation taking or producing a Left/LeftScale or Right/RightScale tile invalidates the corresponding side. Unknown calls and control-flow boundaries clear both sides. Mat loads and Acc stores do not invalidate these bindings. State tracking takes constant space and preserves the pass's linear traversal.

An explicit earlier `tget_scale_addr` also invalidates the state. Consequently, repeated pass execution conservatively adds another binding layer; the standard lowering pipeline runs the pass once.

### Resident scales across control-flow regions

A scale can be filled outside a panel loop and consumed by MX matmuls inside
that loop. The consumer binding alone is insufficient: `tget_scale_addr`
changes a descriptor address, but does not move the bytes already written at
the scale's provisional allocation. The address must be established before
the filling `tmov`.

The Tile IR scale producer is a functional `tile.move`: its result variable
does not exist before that assignment. Therefore this pass retains the SSA
consumer binding, and PTO codegen establishes the resident producer's binding
after declaring its destination handle and before emitting its fill. A
function-wide index identifies a unique data/scale pairing across regions;
the specific data variable's descriptor and its handle must already be defined
in a dominating active region. Sharing an allocated base buffer is insufficient:
a reshape or other data view must be defined before the scale fill.
This includes previously emitted data views: their own declaration scope and
physical offset are retained. When both `if` branches pass through the same
dominating view, codegen also preserves that view's validity instead of emitting
allocation-style validity restoration on it. Packed-FP4 data descriptors count byte carriers,
so the paired logical scale extent is converted to carrier units before
comparing validity; an ordinary packed resident tile needs no metadata view.
Existing consumer bindings remain in place to preserve subsequent rebinding.

Ambiguous cross-region pairings or data handles defined only inside the
consumer region cannot be repaired by blindly hoisting the binding. Codegen
reports those unsupported cases instead of silently emitting a resident fill
at the wrong address. Same-region fills retain the existing PTOAS
normalization path. Metadata aliases that require an early descriptor view
currently need static producer validity and unpacked physical data storage.
Scales returned from an `if`, or returned by a loop that replaces its carried
scale, currently have no proven unique producer/data pairing. Codegen rejects
these result descriptors with an explicit diagnostic; fill the scale in the
consumer scope instead. Cross-region consumers of zero-copy scale views,
including `tile.slice` and `tile.reinterpret_view`, are also rejected because
the early binding does not yet preserve their view provenance; fill the viewed
scale in the consumer scope. Direct self-yielded invariant loop carries remain
supported.

## Related

- Op registration / type checks: `src/ir/op/tile_ops/matmul_mx.cpp`
- Memory-space solving for MX operands: [`InferTileMemorySpace`](22-infer_tile_memory_space.md)
- PTOAS may reorder `tget_scale_addr` before Mat→Scale `tmov` (`PTOA5NormalizeTMovPass`)
