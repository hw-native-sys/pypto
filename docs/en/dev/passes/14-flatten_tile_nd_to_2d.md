# FlattenTileNdTo2D Pass

Flattens ND tile operations (3D+) to 2D in InCore functions by merging all dimensions except the last.

## Overview

PTO-ISA only accepts 2D tiles. After `ConvertTensorToTileOps`, tiles may have rank > 2 (matching tensor shapes). This pass flattens all >2D tile operations to 2D by merging higher axes into one dimension and keeping the last axis unchanged. For example, a tile `[2, 3, 4]` becomes `[6, 4]`.

For batched matrix multiplication, `ConvertTensorToTileOps` first preserves the
high-level intent as `tile.batch_matmul` (or `tile.batch_matmul_acc` when an
accumulator is involved). `FlattenTileNdTo2D` then becomes the canonical
legalization point that expands them into broadcast-aware per-batch
2D `tile.matmul` / `tile.matmul_acc` operations.

**Requirements**:

- Input IR must be in SSA form
- Input IR must have tile ops (run `ConvertTensorToTileOps` first)
- Every tile's **physical** shape must be static (`ConstInt`); a tile's `valid_shape` may be dynamic
  and is preserved through the flatten (see [Dynamic valid_shape](#dynamic-tile-dimensions-issue-1578))
- All tile reduce ops must reduce along the last axis
- All tile memory must be contiguous

**When to use**: Run after `ConvertTensorToTileOps` and before `ExpandMixedKernel` / `InitMemRef`.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::FlattenTileNdTo2D()` | `passes.flatten_tile_nd_to_2d()` | Function-level |

**Python usage**:

```python
from pypto.pypto_core import passes

flatten_pass = passes.flatten_tile_nd_to_2d()
program_2d = flatten_pass(program)
```

## Algorithm

For each InCore function (InCore, AIC, AIV):

1. **Validate preconditions**: Check static physical shapes, last-axis reduction, no `tile.read`/`tile.write`/`tile.slice` on >2D, and no >2D `tile.assemble` whose written region fails to collapse contiguously
2. **Transform statements**: Walk function body and convert >2D tile ops to 2D, preserving any dynamic `valid_shape` (see [Dynamic valid_shape](#dynamic-tile-dimensions-issue-1578))
3. **Verify postconditions**: The `TileOps2D` property verifier independently checks that the rewritten InCore IR contains only supported tile ranks, 2D `tile.assemble` offsets, and codegen-ready transpose forms

Per-statement handling:

| Tile op | Transformation |
| ------- | -------------- |
| `tile.load` (>2D) | Rebuild the result tile as 2D. For a natural NZ Mat load, also insert a shape-only 2D `tensor.view` on the source tensor, collapse leading offsets/shapes/valid_shape to the 2D source window, and require that window to be row-major contiguous. Vec loads and transposed Mat loads keep the original rank>2 source window and only flatten the result tile |
| `tile.store` (rank>2 tensor) | Inject the original tensor-rank partition `shapes` as an extra 4th operand in the transformed IR so backend codegen can reconstruct the `partition_view`; the DSL source is unchanged. If the tile operand itself is still rank>2 (e.g. a user-written `tile.reshape` to 3D feeding `pl.assemble` into an N-D tensor view), insert a `tile.reshape` to flatten the tile operand to 2D first — the codegen requires a 2D tile while the original tile shape still flows through as the `shapes` partition operand |
| `tile.store` (2D tensor) | Pass through unchanged |
| `tile.create`/`tile.full` (>2D) | Rebuild with flattened 2D shape directly |
| `tile.assemble` (>2D target) | Fold the ND offset into the flattened `(row, col)` space with the same row-major collapse `tile.load` applies to its tensor-rank offsets (`row = ((o0*d1 + o1)*d2 + o2)*… + o[k-2]`, `col = o[k-1]`); the tile operands themselves are flattened by their defining ops. Requires source, target and offset to share one rank, and the written region to collapse to a contiguous row band (`IsRowMajorCollapseContiguous`) — both rejected in the precondition phase otherwise. Without the fold the offset would keep its ND rank on a 2D tile, and codegen (which reads `elements[0]`/`elements[1]` positionally and ignores the rest) would silently place the write at the wrong address |
| `tile.transpose` | Sole owner of `pto.ttrans` scratch materialization. Arrives 3-arg (input, axis1, axis2). **2D**: create one scratch tile (shape = SOURCE page, in the input's memory space) and emit the codegen-ready 4-arg `tile.transpose(in, a1, a2, scratch)`. **>2D** (last-two-axes swap): unroll into per-batch 2D transposes, each a 4-arg form with scratch sliced from a flat `[batch*A, B]` pool, assembled into the merged 2D output. A batch-axis swap is a user error |
| `tile.batch_matmul` | Expand to per-batch 2D `tile.matmul`, honoring batch broadcast. A b_trans/a_trans operand arrives as a zero-copy `tile.transpose_view` over a natural load (no transpose-at-load, no copy); the tile-level op carries no transpose semantic. Each operand is handled identically (see operand handling below). **When the result is itself a batched accumulator** (a downstream `tile.batch_matmul_acc` keeps writing it), the pages are written into ONE column-packed `Acc` tile with `tile.matmul_acc(window, lhs_b, rhs_b, init_cond=True)` instead — see [Batched accumulators pack along columns](#batched-accumulators-pack-along-columns) |
| `tile.batch_matmul_acc` | Expand to per-batch 2D `tile.matmul_acc`, taking one window of the (already-flattened) accumulator per batch index: the **column** window `[0, b*N]` of an `[M, B*N]` tile when the chain is column-packed, the legacy **row** window `[b*M, 0]` of a `[B*M, N]` tile otherwise — see [Batched accumulators pack along columns](#batched-accumulators-pack-along-columns). Memory-space decisions the pass does not already state (Vec/Acc round-trips on a row-packed accumulator, retargetable producer promotion of an upstream `tile.create`, TileView refresh) are deferred to `InferTileMemorySpace` (pass 20) — flatten emits no inline `tile.move` |
| Other tile ops (>2D) | Substitute vars, re-create with 2D types |
| 1D/2D tile ops | Unchanged |

**Unified operand handling — whole-fit slice vs. per-batch load.** Every
batch_matmul operand (lhs or rhs, transposed or not, load- or move-sourced) is
treated identically. The routing is decided **per operand**: keep the whole tile
only when the operands' whole tiles fit Mat (L1) together (`BatchOperandsWholeFit`,
a capacity gate) **and** this operand's whole load collapses contiguously
(`WholeLoadContiguous`); otherwise re-emit it per batch.

- **whole (default):** the operand is brought whole into Mat once and
  per-batch **sliced** — a row slice for a plain (row-batched `[B*rows, cols]`)
  operand, a column slice for a `tile.transpose_view` (column-batched
  `[K, B*N]`) operand. A natural Mat load of a 3D `[B, N, K]` tensor keeps its
  logical ND source semantics here, but this pass inserts the 2D `tensor.view`
  (`[B*N, K]`) before the load so downstream `tile.load` codegen sees the same
  flattened source window as every other consumer. The pass also flattens the
  load's **result tile** to 2D. A broadcast operand reuses its single page.
- **per batch** (the whole tile would overflow L1, **or** the whole load is
  non-contiguous): re-emit the operand from its underlying natural `tile.load`
  one batch at a time (a per-batch `[1, .., X, Y]` window → 2D `[X, Y]`, using the
  load's own window dims so a partial sub-tile re-emits correctly), with a
  per-batch `tile.transpose_view` when transposed. The dead whole load/view is
  then dropped.
  - *Non-contiguous* means a multi-batch load that also partially slices the
    matrix-row (middle) dim — e.g. `[2, K0<K, N]` from `[2, K, N]`. Flattened to
    `[2*K, N]` such a window has gaps between batches, so it cannot be one 2D
    ND2NZ load; per batch each page is `[1, K0, N]` (contiguous) and collapses
    cleanly. This routing keeps the codegen contiguity guard from ever firing on
    a batch_matmul operand.

**Dead-load elimination (per-batch only).** When an operand re-emits per-batch
loads (capacity !fit or non-contiguous), the original whole load/view becomes
dead and the pass drops it. The drop pre-scan applies the **same per-operand
routing** as `LowerBatchMatmul`, so a non-contiguous operand's chain is recognized
as per-batch here too. A chain is drop-eligible when **every** use is a
`tile.batch_matmul[_acc]` operand (the chain `tile.load → tile.transpose_view` is
walked back), and it is dropped only when **every** consuming matmul routes it
per-batch — a chain shared with any whole-kept matmul stays whole. Uses are
counted **recursively** (including nested `If`/`For`/`While`/`Scope` bodies) so a
load also consumed in a nested block is never dropped. The capacity gate is
backend-gated (no backend → reports fit), but the contiguity check is not, so the
non-contiguous routing fires in unit tests too.

> The per-batch V2C move case (a move-sourced operand that does not fit L1) is a
> deferred follow-up; such an operand currently stays on the whole-slice path,
> correct only while the moved tile fits the fixed cross-core ring.

## Batched accumulators pack along columns

A batched `tile.batch_matmul_acc` accumulator is the one value this pass does
**not** flatten with the generic `[prod(leading), last]` collapse. Its pages are
stacked along **columns** instead: one `[M, B*N]` `Acc` tile, page `b` at
`tile.slice(acc, [M, N], [0, b*N])`.

### Why rows are not an option

`Acc` (L0C) is NZ-boxed: for a 4-byte accumulator element, box `(r_b, c_b)` of an
`[M, N]` tile begins at `(c_b * M/16 + r_b) * 1024` bytes. So a **row** window of
a tile with more than one block column is *strided* — its boxes are separated by
the parent's row extent, not by the window's. pto-isa's MAD writes its `[m, n]`
destination compactly from a bare pointer and carries no destination stride
(hw-native-sys/pto-isa#253), so the row-packed `[B*M, N]` shape has no correct
lowering at all: only the first 16 columns of each page would land right. That is
the shape `CanonicalizeTileSlice` (pass 19) rejects; see
[18-canonicalize_tile_slice.md](19-canonicalize_tile_slice.md).

A **column** window spans the parent's full row extent, so the window's own
compact geometry and the parent's coincide and the discarded stride cannot
matter. `GetSliceAccumulatorGeometry` gives exactly this shape its NZ-exact byte
offset (see [33-init_memref.md](34-init_memref.md)).

### What changes on the producer

`LowerBatchMatmul` used to stage a multi-batch result in **Vec** — gathering the
pages in `Acc` would need an L0C→L0C copy, which the ISA lacks. That evacuated the
accumulator from the only space `tile.matmul_acc` accepts. When the result roots
an accumulator chain, the pass now allocates the packed `Acc` tile itself and
writes page `b` with:

```text
acc_0 = tile.create([M, B*N], dtype=FP32, target_memory=Acc)
w_b   = tile.slice(acc_b, [M, N], [0, b*N])
m_b   = tile.matmul_acc(w_b, lhs_b, rhs_b, True)   # init_cond=True
acc_b1 = tile.assemble(acc_b, m_b, [0, b*N])       # self-copy, elided at codegen
```

`init_cond=True` needs no new operator: a literal predicate folds to the
**non-accumulating** arm, so this emits a plain `pto.tmatmul ins(lhs, rhs)
outs(window)`. That is how a matmul reaches a sub-region even though `tile.matmul`
has no destination operand. The `tile.assemble` writeback exists only to thread
SSA; codegen recognises it as a self-copy of the same window and emits nothing —
which is required, not merely an optimisation, since an `Acc` destination has no
legal `tmov`. **The offset tuple is therefore built once and passed to both the
slice and the assemble**: codegen matches the two subviews by their emitted SSA
names, so a rebuilt (equal-valued) offset would emit an illegal L0C→L0C move.

Direct-store fusion is suppressed for such a result: fusing it into the next
`tile.store` would consume a statement the chain still needs.

### What changes on the drain

An `[M, B*N]` tile is not the row-major collapse of a `[B, M, N]` output window,
so a single whole-tile `tile.store` would write garbage with nothing downstream
able to notice. The `tile.store` of a packed accumulator becomes **one store per
page**, straight out of L0C:

```text
d_b  = tile.slice(acc, [M, N], [0, b*N])
out  = tile.store(d_b, [b, 0, 0], out, [1, M, N])
```

No Vec staging and no `tile.move`: a store whose source is an `Acc` tile is
classified CUBE, so it stays on the cube lane and `ExpandMixedKernel`'s AIC→AIV
boundary is not involved. (That boundary — the `Acc`→Vec move — is untouched on
every *non*-accumulator `tile.batch_matmul`, which still stages through Vec.)

### When a chain is packed

The decision is made once for the whole function, before any rewriting
(`acc_packing.cpp`), because a chain routinely spans blocks: the `tile.create`
sits outside the K loop, the `tile.batch_matmul_acc` inside it, the `tile.store`
after it. The rewrite loop's own pre-scan is per-block and cannot see that.

A *chain* is a connected component of the same-buffer alias graph — the
`tile.batch_matmul_acc` in-place edge, plain SSA aliases, loop `iter_arg` /
`yield` / `return_var` carries, and `IfStmt` merges. It is packed only when
**every** member is produced and consumed by a form the pass rewrites page-wise
(a `tile.create` or `tile.batch_matmul` root, `tile.batch_matmul_acc`, a carry, or
a `tile.store` drain) and the geometry is one L0C can address:

| Requirement | Why |
| ----------- | --- |
| `M % 16 == 0` and `N % 16 == 0` | Both parent extents must be whole 16×16 boxes, and the page's column origin `b*N` must be box-aligned, or `GetSliceAccumulatorGeometry` declines and `InitMemRef` silently falls back to row-major arithmetic |
| 4-byte accumulator element (FP32 / INT32) | A 16×16 box is `kAccFractal` (1024) bytes only at 4 bytes per element |
| `B*M*N*4` fits `Acc` | Read from the backend (`GetMemSize(Acc)`), not hard-coded — L0C is 128 KB on Ascend910B and 256 KB on 950 |
| no partial `valid_shape` | Page `b`'s valid region starts at `b*N` but is only `N_valid` wide; the packed parent has no single valid rectangle to describe that |
| `batch_count > 1` | At `B == 1` both packings are the same `[M, N]` tile, so the existing fast paths are byte-identical and stay untouched |

A chain that fails any of these keeps the legacy row-packed lowering — which is
still correct when a page is at most 16 columns wide, since it then fits a single
L0C block column that `CanonicalizeTileSlice` whitelists. Note `M % 16 == 0` is
*stricter* than what row packing needs (`B*M % 16 == 0`), so e.g. `M = 8, B = 2,
N = 16` deliberately falls back.

Three further conditions defeat the row-packed fallback. The first two reject a
chain at **any** page width, 16 included; the third only applies above 16
columns, where the fallback is unavailable anyway:

| Condition | Why row packing cannot rescue it |
| --------- | -------------------------------- |
| more than one allocating definition in the chain (e.g. `if k == 0: acc = matmul(...) else: acc = matmul_acc(acc, ...)`, or a loop body that re-creates the accumulator) | The two buffers meet at a control-flow merge, and coalescing them needs an L0C→L0C copy the ISA does not have. Left alone, the program reaches `MemoryReuse`'s `YieldFixup` twenty passes later and dies with an *internal* error |
| a definition that cannot write `Acc` at all (e.g. `tile.load`) | Batch-independent — the same accumulator fails identically at `B == 1` |
| a `tile.move` drain at `N > 16` | Splitting the move page-wise is easy, but gathering the pages is not: a moved page keeps L0C's `col_major`/1024 block layout and `tile.assemble` cannot write it into the `row_major` vector tile the `[B*M, N]` collapse expects. **Not** an accumulator-specific limit — a plain `batch > 1` `tile.batch_matmul` followed by any vector op fails at the same check (`pto_ops_shared.cpp`, "blayout mismatch between source and result"), which is why nothing is packed for it |

Each of these gets its own diagnostic and its own remedy; the column-packing
rationale is deliberately **not** printed for them, since none is about page
geometry.

Anything else — a wider page that cannot be column-packed, or one whose producer
would have to stage through Vec — is **rejected here** with a diagnostic naming
the DSL workaround, rather than emitted with an address that would silently fall
back to row-major:

```text
tile.batch_matmul_acc: cannot lower a batch-2 accumulator of 16x24 FP32 pages,
because the page column extent N=24 is not a multiple of 16.
The pages of a batched accumulator have to be packed along COLUMNS — one 16x48
Acc (L0C) tile with page b at tile.slice(acc, [16, 24], [0, b * 24]) — because
the hardware MAD writes its destination compactly and has no destination stride
...
Workarounds: write the batch loop out in the kernel and accumulate each page
into its own 2-D tile (pl.matmul / pl.matmul_acc on 2-D operands); or keep the
accumulator at most 16 columns wide, which fits a single L0C block column and
needs no packing.
```

## Example

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(self, x: pl.Tensor[[2, 3, 4], pl.FP32],
                      out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
        x_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
        y_tile: pl.Tile[[2, 3, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
        out_0 = pl.store(y_tile, [0, 0, 0], out_0)
        return out_0
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def main_incore_0(self, x: pl.Tensor[[2, 3, 4], pl.FP32],
                      out_0: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]]) -> pl.Tensor[[2, 3, 4], pl.FP32]:
        x_tile: pl.Tile[[6, 4], pl.FP32] = pl.load(x, [0, 0, 0], [2, 3, 4])
        y_tile: pl.Tile[[6, 4], pl.FP32] = pl.tile.add(x_tile, x_tile)
        out_0 = pl.store(y_tile, [0, 0, 0], out_0)
        return out_0
```

The 3D tile `[2, 3, 4]` is flattened to `[6, 4]`. `tile.load` directly produces a 2D tile —
no `tile.reshape` is inserted. `tile.store` accepts the 2D tile and writes to the original rank>2 tensor. For
rank>2 tensors, the pass injects the original partition `shapes` as an extra 4th operand into the
transformed IR (e.g. `pl.store(y_tile, [0, 0, 0], out_0, (2, 3, 4))`); this operand is only
present in the transformed IR and is not part of the source DSL.

## Dynamic tile dimensions (issue #1578)

Hardware tiles map to fixed-size on-chip buffers, so every **physical** tile dimension must be a
compile-time constant; the runtime extent lives in `TileView.valid_shape`. To process a dynamic
dimension the user **writes the chunk loop themselves**: iterate the dynamic dim with `pl.range` in a
static `CHUNK` step, and load each chunk as a static physical `[1, CHUNK, 512]` tile whose
`valid_shape` carries the runtime tail `min(CHUNK, s - c)`. The chunk size is the user's choice — it
strongly affects performance, so it is not auto-selected by the pass.

```python
# User-written: chunk the dynamic S dim, clamp the tail in valid_shape.
for c, (o,) in pl.range(0, s_dim, CHUNK, init_values=(out,)):
    valid = pl.min(CHUNK, s_dim - c)
    t = pl.load(x, [b, c, 0], [1, CHUNK, 512], valid_shape=[1, valid, 512])
    t = pl.cast(t, target_type=pl.FP32)
    o = pl.store(t, [b, c, 0], o)        # static physical [1, CHUNK, 512], dynamic valid
    pl.yield_(o)
```

Each per-chunk tile is physically `[1, CHUNK, 512]` (static) with a dynamic `valid_shape`
`[1, min(CHUNK, s - c), 512]`. **FlattenTileNdTo2D's only job here is to lower that >2D tile to
`[CHUNK, 512]` while preserving the dynamic `valid_shape`** — `ComputeMergedValidShape` merges the
leading dims of `valid_shape` the same way `ComputeMergedShape` merges the physical shape, but tolerates
dynamic entries, so the runtime tail survives the flatten instead of being reset to the full physical
shape. The loop itself is the user's; the pass does **not** synthesize it.

> The chunk must fit on-chip Vec (UB) memory (`CHUNK * <kept dims> * <live tile bytes> <= UB capacity`),
> otherwise `AllocateMemoryAddr` rejects the kernel with a "Vec buffer usage exceeds platform limit"
> error. Picking the chunk is the user's responsibility.

If a >2D tile reaches the pass with a **dynamic physical shape** (the user did not slice a static
chunk), it cannot be flattened and the pass raises an actionable error pointing to the two fixes:
chunk the dynamic dim with `pl.range`/`pl.parallel`, or reshape to 2D before the InCore (`pl.at`) scope.

## Loop-carry valid-shape repair

Unrolling a `tile.batch_matmul` whose left operand carries a narrowed `valid_shape`
produces 2D matmuls whose results are narrower than the accumulator they flow into. The
loop carry that accumulator travels through is typed from its **init value alone**, so it
keeps advertising the seed's full box height that no `mad` ever wrote:

```text
acc__tile      : Tile[[64, 256], INT32]                         <- pl.create_tensor seed
  iter_arg     : Tile[[64, 256], INT32]                         <- typed from the seed
  yield        : Tile[[64, 256], INT32, Acc, valid=[v, 256], compact]   <- what the body produced
  return_var   : Tile[[64, 256], INT32]                         <- forced back to the iter_arg
```

`mad` lays its product out in L0C at an N-fractal stride of `ceil(v/16)*16`, so a reader
that believes the full height walks the buffer at the physical row pitch and scrambles
every N-fractal above the first (issue #2470). This pass therefore calls
`narrow_loop_carry::NarrowAccCarries` on each function it rewrites, before returning: the
seed is re-declared at the extent the yields prove — `tile.create(compact=True)` plus
`tile.set_validshape`, the same form `AutoTileMatmulL0` builds when it splits K — and the
body's def-use closure is re-typed through the operators' own deducers.

Repairing it here rather than in a later pass is what keeps the pipeline verifiable: the
carry this pass creates would otherwise be rejected by the `TypeCheck` diagnostic and the
`AccCompactValid` property verifier. `ConvertTensorToTileOps` calls the same helper for
the same reason — a 2D seed is narrowed one pass earlier, when `tensor.matmul` becomes
`tile.matmul`.

A carry is left exactly as it is when the two readings of its buffer cannot disagree — a
single-fractal-block `[16, N]` accumulator packs to its physical rows whatever its valid
rows — or when the narrowed extent is only computed inside the loop body, where the
re-declared seed could not name it.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

The implementation is split by responsibility:

| Phase | File | Responsibility |
| ----- | ---- | -------------- |
| Coordination | `src/ir/transforms/flatten_tile_nd_to_2d/pass.cpp` | Select InCore functions, sequence analysis before rewrite, and repair the loop carries the rewrite narrowed |
| Analysis | `src/ir/transforms/flatten_tile_nd_to_2d/analysis.cpp` | Read-only precondition validation |
| Rewrite orchestration | `src/ir/transforms/flatten_tile_nd_to_2d/rewrite.cpp` | Recursive statement traversal and operation dispatch |
| Rewrite utilities | `src/ir/transforms/flatten_tile_nd_to_2d/rewrite_utils.cpp` | Shared shape, index, and capacity helpers |
| Accumulator packing | `src/ir/transforms/flatten_tile_nd_to_2d/acc_packing.cpp` | Whole-function decision of which batched `Acc` accumulator chains pack along columns |
| Batched matmul rewrite | `src/ir/transforms/flatten_tile_nd_to_2d/batch_matmul.cpp` | Batched matmul and matmul-acc page lowering |
| Transpose rewrite | `src/ir/transforms/flatten_tile_nd_to_2d/transpose.cpp` | Standalone N-D transpose lowering |
| Verification | `src/ir/transforms/flatten_tile_nd_to_2d/verification.cpp` | Independent `TileOps2D` postcondition verification |

The phase entry points and rewrite component interface are private to the transform implementation; the public API remains `pass::FlattenTileNdTo2D()`.

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_flatten_tile_nd_to_2d.py`, `tests/ut/ir/transforms/test_narrow_loop_carry_valid_shape.py` (the carry repair), `tests/st/codegen/dsl/test_flatten_dynamic_tile_3d.py` (issue #1578 end-to-end)

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm, IncoreTileOps, NormalizedStmtStructure |
| Produced | SSAForm, TileOps2D, NormalizedStmtStructure |
| Invalidated | — |

## Scope

| Tile rank | Action |
| --------- | ------ |
| 1D | Unchanged |
| 2D | Unchanged |
| 3D+ | Flattened to 2D |

Only InCore-type functions (InCore, AIC, AIV) are processed. Orchestration and Opaque functions are returned unchanged.
