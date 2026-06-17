# SplitVectorKernel Pass

Splits a vector kernel along one tile axis so that two AIV lanes share the
work, halving the per-lane tile shapes and rewriting `tile.load`,
`tile.store`, `tile.tpop_from_aic`, and `tile.reshape` to address each
lane's half. On
Ascend910B, the same pass also handles the **no-split dual-AIV dispatch**
path: when `ExpandMixedKernel` decides a mixed kernel cannot be split, it
tags the AIV function with `dual_aiv_dispatch=True` and this pass wraps
the body in a per-lane `if subblock_idx == 0 ... else` so AIC↔AIV cross-core
handshakes stay balanced even though only lane 0 does real compute.

## Overview

Two distinct rewrites share one pass because both depend on
`subblock_idx` and on cross-core `tpush`/`tpop` accounting:

1. **Split mode** — driven by either `Function::attrs["split"]`
   (`SplitMode::UpDown` or `SplitMode::LeftRight`) or a `split=` kwarg on
   any `tile.tpush_*` / `tile.tpop_*` call inside the function. The AIC
   side only needs the `split=` value synced across its cross-core ops;
   the AIV side gets a real shape rewrite — its tiles are halved on the
   split axis, `tile.load` / `tile.store` offsets are bumped by
   `subblock_idx * half_dim` so each lane addresses its own half, and
   `tile.tpop_from_aic` results are halved on the split axis.

2. **No-split dual-AIV dispatch** — only fires on backends whose
   `BackendHandler::RequiresNoSplitDualAivDispatch()` returns `true`
   (Ascend910B today) and only on AIV functions tagged
   `dual_aiv_dispatch=True` by `ExpandMixedKernel` (see
   [`ExpandMixedKernel`](21-expand_mixed_kernel.md), the "no function
   split mode" paragraph). The pass injects `subblock_idx`, hoists shared
   pipe-setup calls (`reserve_buffer`, `import_peer_buffer`,
   `aic_initialize_pipe`, `aiv_initialize_pipe`) above the lane branch,
   and emits an `IfStmt` whose then-branch is the original body and
   whose else-branch is a "replay" that keeps every cross-core
   `tpush`/`tpop`/`tfree` but forces tile-producing replays to
   `valid_shape=[0, 0]` and drops user-visible `tile.store` writes.

`ResolveSplitMode` decides which mode to use:

- If `attrs["split"]` is set and non-`None`, that wins (cross-core
  `split=` kwargs in the body must agree, otherwise `ValueError`).
- Otherwise the body is scanned by `CrossCoreSplitCollector` and the
  unique non-zero `split=` kwarg becomes the inferred mode.
- Conflicting cross-core `split=` values raise `ValueError`.
- If the function is AIV with `dual_aiv_dispatch=True` *and* the
  resolved split mode is `None`, the no-split dual-dispatch rewrite
  applies instead.

### Split-axis dispatch

| `SplitMode` (int) | Split axis | Halved | Offset adjust on `tile.load` / `tile.store` |
| ----------------- | ---------- | ------ | ------------------------------------------- |
| `None` (0) | — | — | pass is a no-op for that function |
| `UpDown` (1) | dim 0 (height) | rows | `[orig + subblock_idx * H/2, orig]` |
| `LeftRight` (2) | dim 1 (width) | cols | `[orig, orig + subblock_idx * W/2]` |

`subblock_idx` is materialized by `pl.tile.get_subblock_idx()`, injected
as the first stmt of the rewritten AIV body via `InjectSubblockIdx`.
Name collisions with existing params/locals are avoided by
`auto_name::GenerateFreshNameLike`.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::SplitVectorKernel()` | `passes.split_vector_kernel()` | Program-level |

```python
from pypto import passes
result = passes.split_vector_kernel()(program)
```

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `SSAForm`, `MixedKernelExpanded` |
| Produced | `SSAForm`, `VectorKernelSplit`, `NormalizedStmtStructure` |
| Invalidated | — |

`MixedKernelExpanded` is the upstream contract that no `FunctionType::InCore`
function still mixes Cube and Vector ops, and that AIC↔AIV cross-core
ops are already in place. `VectorKernelSplit` advertises that AIV
functions whose `attrs["split"]` is non-`None` have had their tile shapes,
`tile.tpop_from_aic` results, and `tile.load`/`tile.store` offsets
adjusted to per-lane form. Source: `include/pypto/ir/transforms/pass_properties.h`,
`include/pypto/ir/transforms/ir_property.h`.

### Function-attribute invariant on exit

The pass treats `attrs["dual_aiv_dispatch"]` as the single source of truth
for the dual-AIV dispatch decision and maintains the following invariant
on the AIV function it returns:

```text
SplitMode resolved to non-None  ⇒  attrs["dual_aiv_dispatch"] == true
```

`ExpandMixedKernel` is the other writer of this attribute (it sets it on
the no-split mixed-kernel path); `SplitVectorKernel` ensures the split
path also reflects in the attribute. Orchestration codegen
(`RequiresDualAivDispatch` in `src/codegen/orchestration/orchestration_codegen.cpp`)
reads only this attribute and never re-derives from `SplitMode`.

## Algorithm — split-mode

`ProcessFunction` rewrites a single AIC or AIV function whose
`ResolveSplitMode` resolves to `UpDown` or `LeftRight`:

```text
1. Resolve split mode and split dim:
   split_dim = (mode == UpDown) ? 0 : 1
2. Clone params (preserving names and types) into var_replacements so the
   rewritten body still sees the same parameter identity.
3. (AIV only) InjectSubblockIdx prepends:
       subblock_idx = tile.get_subblock_idx()
   to the body, picking a fresh name if 'subblock_idx' is already taken.
4. Walk every statement via ProcessStmt:

   tile.tpush_to_aiv / tile.tpush_to_aic / tile.tpop_from_aiv:
     RebuildCallWithSplit — sync the `split=` kwarg only. AIC keeps the
     full operand tile (cube still consumes the whole matmul output).

   tile.tpop_from_aic (AIV only):
     RebuildTpopWithHalvedShape — halve the result shape on split_dim,
     localize TileView.valid_shape per subblock, and sync `split=`.

   tile.load (AIV only, ≥4-arg form):
     If the result tile's split-axis dim is singleton (e.g. [1, 128]
     under UpDown), keep the load as-is.
     If the tile is rank-1 or rank-0 (rank < 2), keep as-is under EVERY
     split mode — a rank-1 tile carries no 2D split axis (which physical
     axis is "the split axis" only becomes defined once it is reshaped to
     2D), so the consuming tile.reshape introduces and slices the axis
     instead (see below). Halving a rank-1 load directly is unsafe: under
     UpDown it would split a rank-1 column vector along the wrong axis.
     Otherwise: halve result shape, halve the static shape arg, localize
     valid_shape, and add `subblock_idx * H/2` to the split-axis offset.

   tile.store (AIV only, ≥3-arg form):
     If the source tile is tracked in tile_vars (i.e. it was halved
     earlier), bump its split-axis offset by `subblock_idx * H/2`.

   tile.reshape (AIV only):
     A reshape that lifts a full (un-split) source tile onto the split
     axis — typically a rank-1 load that bypassed the load rewrite, e.g. a
     per-channel scale [D] -> [1, D] — must give each lane its own half.
     Because reshape is an offsetless view, halving only its result type
     would leave BOTH lanes reading the first half of the full buffer. So
     when the reshape input is NOT already split and the result's split-axis
     dim is a static non-singleton extent, emit the reshape at full width
     and follow it with a per-subblock tile.slice selecting
     `[..., subblock_idx * half : +half]` on the split axis (the slice
     result, and the original var, are both tracked in tile_vars). If the
     reshaped split-axis dim is singleton (e.g. [D] -> [1, D] under UpDown)
     the singleton rule above keeps it full — both lanes need it. If the
     input is already split, the reshape falls through to result-halving,
     which also halves the explicit target-shape argument on the split axis
     (leaving the literal stale would make memory_reuse size the output from
     the un-split shape and abort against the split-sized slot).

   any other tile.* op producing a TileType (AIV only):
     Halve the result shape on split_dim. For tile.full / tile.create the
     static shape arg is also halved. Reduce ops on the split axis raise
     ValueError via IsReduceOnSplitAxis (partial reduction would be
     incorrect).

   ForStmt:
     Rebuild iter_args whose initValue is a tracked halved tile so they
     carry the halved type. Rebuild return_vars to inherit the halved
     type from their iter_args. Recurse into the body. Loop-carried
     state is repaired by loop_repair::RebuildForStmt.

   IfStmt / SeqStmts:
     Recurse into branches and stmt sequences.

5. After rewriting, transform_utils::Substitute applies var_replacements
   so every reference (param, iter_arg, return_var, tpop result) sees the
   rewritten Var node.
6. DeepClone is applied to detach from any shared IR sub-trees.
7. WithSplitAttrs stamps the resolved SplitMode onto Function::attrs
   (overwriting any prior `split` entry). For AIV functions whose
   resolved mode is non-None it *also* writes `dual_aiv_dispatch=true`
   so orchestration codegen has a single attribute to read instead of
   re-deriving the dual-AIV decision from `SplitMode`.
```

`tile_vars` is the per-pass map that tracks which `Var`s carry halved
tiles (with their `half_dim_size`). It is the mechanism that lets a
`tile.store` issued *outside* the loop still recognize that its source
tile was halved by a `tile.load` *inside* the loop.

## Algorithm — no-split dual-AIV dispatch

`ProcessNoSplitDualAivFunction` only fires when
`RequiresNoSplitDualAivSync(func)` is true — that is, the backend is
Ascend910B (or any backend whose `BackendHandler::RequiresNoSplitDualAivDispatch()`
returns true), the function is AIV, and `attrs["dual_aiv_dispatch"]` is
true. It runs *instead of* `ProcessFunction` (a function never enters
both paths).

```text
1. Clone params into param_replacements (same as split-mode).
2. InjectSubblockIdx — prepend `subblock_idx = tile.get_subblock_idx()`.
3. Strip the leading subblock_idx assign from the body, then split off
   the shared pipe-setup prefix:
     SplitNoSplitSharedPipeSetupPrefix takes the maximal prefix of
     reserve_buffer / import_peer_buffer / aic_initialize_pipe /
     aiv_initialize_pipe stmts (see IsNoSplitSharedPipeSetupCall) so
     they run on both lanes from the original location.
4. Lane 0 body = the original branch stmts (unchanged).
5. Lane 1 body = BuildNoSplitLane1ReplayStmts(branch stmts):
     - tile.store: drop EvalStmt forms entirely; for AssignStmt forms
       passthrough the third arg (the destination tensor) so SSA users
       still see a value, but no write happens.
     - any other call producing a TileType: rewrite via
       RebuildLane1CallWithZeroValidShape — `tile.load` becomes
       `tile.create` whose result type carries `valid_shape=[0, 0]`;
       `tile.slice` and `tile.set_validshape` get their valid_shape args
       zeroed; everything else has its result type's `valid_shape`
       cleared.
     - cross-core tile.tpush_* / tile.tpop_* / system.tfree_* are kept
       so the AIC↔AIV handshake stays balanced.
     - For/While/If recurse with a forked replacements map so SSA
       renames inside a branch do not leak across siblings.
6. Wrap lane 0 and lane 1 in:
       if subblock_idx == 0:
           <lane 0>
       else:
           <lane 1>
7. New body =
     subblock_idx assign
     <hoisted shared pipe-setup>
     <branch IfStmt>
8. Substitute / DeepClone, attrs unchanged (dual_aiv_dispatch=True
   stays — downstream lowering reads it).
```

### Codegen transport: full-column box, preserved rows

The lane-1 replay zeroes its tile `valid_shape` so it produces no visible
writes, but the AIC↔AIV `tpush` it keeps still moves data through the shared
GM FIFO slot the single cube consumer pops in full. On the codegen side
(`EmitSplitTpushTransportValidShape`, `pto_ops_common.cpp`) a no-split
dual-AIV producer that narrowed its `valid_shape` with `set_validshape` must
therefore transport the **full column box**, or the consumer reads stale slot
columns past `valid_col`. Unlike the genuine `UpDown` / `LeftRight` splits —
which widen *both* axes — the no-split path widens **columns only and
preserves the row `valid_shape`**: subblock 0's real push carries the full
column box, while subblock 1's `valid_shape=[0, 0]` replay gets **no transport
at all** (a statically 0-row push moves no data, and emitting a col-widening
`set_validshape` for it perturbs the shared-slot dual-AIV merge — it regressed
the `cross_core_v2c_nosplit` golden), so it stays a true 0-row no-op rather than
racing garbage rows into subblock 0's slot. (Plain `split=0` without
`dual_aiv_dispatch` emits no transport at all either.) The detection lever is
`PTOCodegen::IsDualAivDispatchFunction()`, which reads this pass's
`dual_aiv_dispatch` attribute.

## Constraints

| Constraint | Why |
| ---------- | --- |
| Even split-axis box dim (or dynamic dim with `// 2`) | `ComputeHalfDimSize` raises if a `ConstInt` split-axis box dim is odd; users with odd extents should pad the full box to a multiple of `2 * innerDim` (so the halved subblock box stays innerDim-aligned) and narrow with `set_validshape` — see "Handling odd extents" below. Dynamic dims emit `MakeFloorDiv(dim, 2)` |
| Conflicting function-level vs cross-core split modes | `ResolveSplitMode` raises `ValueError` |
| Conflicting cross-core split kwargs in one body | `CrossCoreSplitCollector` raises `ValueError` |
| Reduce on the split axis is rejected | `IsReduceOnSplitAxis` raises — partial reduction in a single subblock is semantically incorrect |
| Singleton split-axis dim preserved as-is | broadcast tiles like `[1, 128]` under `UpDown` or `[16, 1]` under `LeftRight` still carry the full tile |
| Rank-1 / rank-0 `tile.load` bypasses split rewrites under every mode | a rank-1 tile carries no 2D split axis until reshaped; the consuming `tile.reshape` introduces and slices the axis (halving a rank-1 load directly mis-splits a column vector under `UpDown`) |
| `tile.reshape` of a full input onto the split axis is sliced per lane | reshape is an offsetless view, so the result type is kept full and a per-subblock `tile.slice` is appended; only fires when the input is un-split and the split-axis extent is a static non-singleton |
| AIC keeps full `tile.tpop_from_aiv` shape | cube still consumes the whole matmul operand; only `split=` is synced |
| No-split lane 1 must produce no visible writes | `tile.store` writes are dropped; tile producers are forced to `valid_shape=[0, 0]` so PTO ops run as empty tiles |

## Handling odd extents on the split axis

The box dim that `SplitVectorKernel` halves must be an even `ConstInt`, and
PTOAS additionally requires *both* the full box and the **halved subblock
box** to be multiples of `innerCols` / `innerRows` (16 for fractal=1024 /
Acc, `32 / sizeof(dtype)` for fractal=512 depending on layout). The full
padded box therefore needs to be a multiple of `2 * innerDim`. To ship an
odd extent (e.g. `M = 17` rows or `N = 17` cols), the user pads the tile
box to the next such multiple and records the truthful extent via
`pl.tile.set_validshape`. The pass then halves the padded box on the AIV
side, and `LocalizeValidDimForSplit` clamps the user's odd `valid_shape`
against the halved physical extent so each subblock writes only its share
of the truthful region back to GM. The transport (`tpush_to_aiv` /
`tpop_from_aic`) carries the full padded box so both halves of the
consumer receive complete data — see the `tpush` transport-valid-shape
logic in `src/backend/common/pto_ops_common.cpp`
(`EmitSplitTpushTransportValidShape`).

```python
# Producer: declare a padded box, then narrow with set_validshape.
# For FP32 → Acc (fractal=1024, innerDim=16) the full box must be a multiple
# of 2*innerDim=32 so the halved subblock box (16) is still aligned.
acc: pl.Tile[[32, COLS], pl.FP32] = pl.matmul(a_left, b_right)
narrowed: pl.Tile[
    [32, COLS], pl.FP32, pl.Mem.Acc,
    pl.TileView(valid_shape=[17, COLS]),  # truthful odd extent
] = pl.tile.set_validshape(acc, 17, COLS)
pl.tpush_to_aiv(narrowed, split=1)

# Consumer: same padded box + truthful valid_shape; SplitVectorKernel
# halves the box to [16, COLS] per subblock and localizes the valid extent
# (subblock 0 → 16 valid rows, subblock 1 → 1).
popped: pl.Tile[
    [32, COLS], pl.FP32, pl.Mem.Vec,
    pl.TileView(valid_shape=[17, COLS]),
] = pl.tpop_from_aic(split=1)
```

Auto-padding is intentionally not performed by this pass — slot-buffer
size on `aic_initialize_pipe` / `aiv_initialize_pipe`, reserve-buffer
size on `pto.reserve_buffer`, and the load-side GM access width all
depend on the box choice and must remain in the user's control.

## Examples

### Example 1 — UpDown: tpop halved + store offset adjusted

Distilled from `test_tpop_shape_halved_and_store_offset_adjusted` in
`tests/ut/ir/transforms/test_split_vector_kernel.py`. AIC body keeps
its operand tiles intact and only syncs `split=`; AIV gets the full
halve-and-shift treatment.

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
    def main_aic(self, x: pl.Tensor[[16, 128], pl.BF16], y: pl.Tensor[[128, 128], pl.BF16]):
        x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
        x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
        y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
        y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
        z_tile = pl.matmul(x_left, y_right)
        pl.tpush_to_aiv(z_tile, split=0)        # split=0 is the "None" sentinel

    @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
    def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]):
        z_vec: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView()] = pl.tpop_from_aic(split=0)
        return pl.store(z_vec, [0, 0], out_0)
```

**After**:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
    def main_aic(self, x, y):
        # ... cube ops unchanged ...
        pl.tpush_to_aiv(z_tile, split=1)        # only split kwarg synced

    @pl.function(
        type=pl.FunctionType.AIV,
        attrs={"split": pl.SplitMode.UP_DOWN, "dual_aiv_dispatch": True},
    )
    def main_aiv(self, out_0):
        subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
        z_vec: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=1)
        return pl.store(z_vec, [0 + subblock_idx * 8, 0], out_0)
```

### Example 2 — LeftRight: width halved, dim-1 offset adjusted

Distilled from `test_load_shape_halved_left_right`. AIV does both a
real `tile.load` and a `tpop_from_aic`; both end up on the right half
of the source via `subblock_idx * 64`.

**Before**:

```python
@pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
def main_aiv(self, data: pl.Tensor[[16, 128], pl.FP32],
             out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]):
    prev = pl.load(data, [0, 0], [16, 128], target_memory=pl.Mem.Vec)
    pop_tile: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView()] = pl.tpop_from_aic(split=0)
    result = pl.add(prev, pop_tile)
    return pl.store(result, [0, 0], out_0)
```

**After**:

```python
@pl.function(
    type=pl.FunctionType.AIV,
    attrs={"split": pl.SplitMode.LEFT_RIGHT, "dual_aiv_dispatch": True},
)
def main_aiv(self, data, out_0):
    subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
    prev: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.load(
        data, [0, 0 + subblock_idx * 64], [16, 64], target_memory=pl.Mem.Vec)
    pop_tile: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=2)
    result: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.add(prev, pop_tile)
    return pl.store(result, [0, 0 + subblock_idx * 64], out_0)
```

### Example 3 — LeftRight: rank-1 load reshaped onto the split axis is sliced per lane

Distilled from `test_reshape_of_full_rank1_load_is_sliced_per_subblock`
(the dsv4 `proj_b` per-channel dequant-scale shape). The rank-1 `scale`
load is kept full; the `reshape` onto the split (column) axis is kept full
and a per-subblock `tile.slice` gives each lane its own half. Without the
slice both lanes would read `scale[0:64]`, so lane 1 would apply the wrong
half of the per-channel scale to its (correctly addressed) output columns.

**Before**:

```python
@pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.LEFT_RIGHT})
def main_aiv(self, scale: pl.Tensor[[128], pl.FP32], data: pl.Tensor[[16, 128], pl.FP32],
             out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]):
    scale_row = pl.load(scale, [0], [128], target_memory=pl.Mem.Vec)        # rank-1
    scale_2d: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.reshape(scale_row, [1, 128])
    prev = pl.load(data, [0, 0], [16, 128], target_memory=pl.Mem.Vec)
    result = pl.col_expand_mul(prev, scale_2d)
    return pl.store(result, [0, 0], out_0)
```

**After**:

```python
@pl.function(
    type=pl.FunctionType.AIV,
    attrs={"split": pl.SplitMode.LEFT_RIGHT, "dual_aiv_dispatch": True},
)
def main_aiv(self, scale, data, out_0):
    subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
    scale_row: pl.Tile[[128], pl.FP32, pl.Mem.Vec] = pl.load(             # rank-1 kept full
        scale, [0], [128], target_memory=pl.Mem.Vec)
    scale_2d: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.reshape(        # reshape kept full
        scale_row, [1, 128])
    scale_half: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.slice(         # per-lane slice
        scale_2d, [1, 64], [0, subblock_idx * 64])
    prev: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.load(
        data, [0, 0 + subblock_idx * 64], [16, 64], target_memory=pl.Mem.Vec)
    result: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.col_expand_mul(prev, scale_half)
    return pl.store(result, [0, 0 + subblock_idx * 64], out_0)
```

Under `UpDown` the symmetric case holds: a `[D] -> [1, D]` reshape lands on
a singleton split axis (dim 0) and is kept full (both row-lanes need it),
while a `[D] -> [D, 1]` reshape lands the full extent on the split axis and
is sliced as `[..., subblock_idx * half, 0]`.

### Example 4 — Ascend910B no-split dual-AIV dispatch

Distilled from
`test_no_split_dual_dispatch_producer_replays_compute_and_tpush_on_lane1`.
The AIV function carries `dual_aiv_dispatch=True` (set by
`ExpandMixedKernel` for a no-split mixed kernel) and no `split` attr.
The pass keeps lane 0 doing real work and rebuilds lane 1 as an
empty-tile replay so the `tpush_to_aic` handshake still happens twice.

**Before**:

```python
@pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
def main_aiv(self, a, b, out):
    slot_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
    pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
    a_tile = pl.load(a, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
    b_tile = pl.load(b, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
    summed = pl.add(a_tile, b_tile)
    pl.tpush_to_aic(summed, split=0)
    return out
```

**After** (shape-preserving — lane 1 carries empty tiles via
`valid_shape=[0, 0]`):

```python
@pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
def main_aiv(self, a, b, out):
    subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
    slot_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
    pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=slot_buf)
    if subblock_idx == 0:
        a_tile = pl.load(a, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        b_tile = pl.load(b, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
        summed = pl.add(a_tile, b_tile)
        pl.tpush_to_aic(summed, split=0)
    else:
        # tile.load -> tile.create with valid_shape=[0, 0]
        a_tile: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = \
            pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        b_tile: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = \
            pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        summed: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = \
            pl.add(a_tile, b_tile)
        pl.tpush_to_aic(summed, split=0)        # handshake still fires
    return out
```

`reserve_buffer` and `aiv_initialize_pipe` are hoisted above the
`if`/`else` so both lanes share the same buffer state.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass SplitVectorKernel();
```

**Implementation**: `src/ir/transforms/split_vector_kernel_pass.cpp`

- `ResolveSplitMode` — picks function-level vs body-derived split mode.
- `ProcessFunction` / `ProcessStmt` / `ProcessStmts` — split-mode rewrite.
- `RebuildCallWithSplit` / `RebuildTpopWithHalvedShape` — cross-core
  call rewriters.
- `HalveTileShape` / `ApplyTrackedTileShape` /
  `LocalizeValidDimForSplit` — tile type rewriters.
- `AdjustOffsets` — split-axis offset shift on `tile.load`/`tile.store`.
- `IsReduceOnSplitAxis` — guard for partial-reduction errors.
- `RequiresNoSplitDualAivSync` / `ProcessNoSplitDualAivFunction` /
  `BuildNoSplitLane1ReplayStmts` / `RebuildLane1CallWithZeroValidShape` /
  `IsNoSplitSharedPipeSetupCall` — Ascend910B no-split path.

**Properties**: `include/pypto/ir/transforms/pass_properties.h`

```cpp
inline const PassProperties kSplitVectorKernelProperties{
    .required = {IRProperty::SSAForm, IRProperty::MixedKernelExpanded},
    .produced = {IRProperty::SSAForm, IRProperty::VectorKernelSplit,
                 IRProperty::NormalizedStmtStructure}};
```

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("split_vector_kernel", &pass::SplitVectorKernel,
           "Create a pass that splits vector kernels based on SplitMode "
           "(adjusts tpush/tpop split, halves tpop shapes, adjusts store offsets)");
```

**Type stub**: `python/pypto/pypto_core/passes.pyi`

```python
def split_vector_kernel() -> Pass:
    """Create a pass that splits vector kernels based on SplitMode."""
```

**Tests**: `tests/ut/ir/transforms/test_split_vector_kernel.py`
(`TestSplitVectorKernelUpDown`, `TestSplitVectorKernelLeftRight`, and
`TestSplitVectorKernelNoSplitA2A3`).

## Related

- [`ExpandMixedKernel`](21-expand_mixed_kernel.md) — upstream producer of
  AIC/AIV functions and of the `dual_aiv_dispatch` marker.
- [`InjectGMPipeBuffer`](22-inject_gm_pipe_buffer.md) — runs immediately
  before; backend-gated GM pipe buffer wiring this pass relies on.
- [`NormalizeReturnOrder`](24-normalize_return_order.md) — runs immediately
  after; sees the per-lane tile shapes produced here.
