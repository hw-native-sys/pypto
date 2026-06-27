# SplitVectorKernel Pass

After the staged convergence refactor, `SplitVectorKernel` has two narrow jobs;
it **no longer halves any function body**:

1. **`split_aiv` attribute stamping** — the SOLE split path through this pass.
   A `split_aiv` kernel (hand-authored, or produced upstream by
   [`LowerAutoVectorSplit`](21-lower_auto_vector_split.md)) has already lowered
   its explicit `tile.aiv_shard` / `tile.aic_gather` into split-stamped
   `tpush`/`tpop` (folded by `ExpandMixedKernel`) and carries already-halved
   compute tiles plus its own `tile.get_subblock_idx()`. This pass leaves the
   body untouched and only stamps `split` (+ `dual_aiv_dispatch` for AIV
   functions) on the function attrs.

2. **No-split dual-AIV dispatch** — on Ascend910B (any backend whose
   `BackendHandler::RequiresNoSplitDualAivDispatch()` returns `true`), when
   `ExpandMixedKernel` decides a mixed kernel cannot be split it tags the AIV
   function `dual_aiv_dispatch=True`. This pass wraps the body in a per-lane
   `if subblock_idx == 0 ... else` replay so AIC↔AIV cross-core handshakes stay
   balanced even though only lane 0 does real compute.

> **Historical note.** This pass used to drive per-op AIV halving
> (`ProcessFunction` / `ResolveSplitMode` / `CrossCoreSplitCollector`). That
> driver was deleted once `LowerAutoVectorSplit` became the live auto-split
> lowering path: after it runs, every split function reaches this pass already
> `split_aiv`-marked, so re-halving here would double-halve the already-half
> body. The halving machinery itself (shape halving, offset localization,
> `tile.slice` arg-halving, reduce-on-split-axis rejection, loop tracking) now
> lives in `split_axis_utils`, called by `LowerAutoVectorSplit` — see that
> pass's doc for the per-op rewrite rules.

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
function still mixes Cube and Vector ops, and that AIC↔AIV cross-core ops are
already in place. `VectorKernelSplit` advertises that split AIV functions are in
per-lane form (achieved upstream by `LowerAutoVectorSplit` + `ExpandMixedKernel`;
this pass certifies it via the attr-stamping arm). Source:
`include/pypto/ir/transforms/pass_properties.h`,
`include/pypto/ir/transforms/ir_property.h`.

### Function-attribute invariant on exit

The pass treats `attrs["dual_aiv_dispatch"]` as the single source of truth for
the dual-AIV dispatch decision and maintains:

```text
split_aiv function with non-None split mode  ⇒  attrs["dual_aiv_dispatch"] == true (AIV)
```

`split_aiv` functions without a function-level split mode are an
`INTERNAL_CHECK` failure (`OutlineIncoreScopes` / `LowerAutoVectorSplit` must
co-propagate `split` with `split_aiv`). Orchestration codegen
(`RequiresDualAivDispatch` in `src/codegen/orchestration/orchestration_codegen.cpp`)
reads only this attribute and never re-derives from `SplitMode`.

## Dispatch

```text
for each function:
  if (AIV or AIC) and attrs["split_aiv"]:
      # SOLE split path — stamp attrs, pass body through unchanged.
      assert function-level split mode is set and non-None  (INTERNAL_CHECK)
      attrs = WithSplitAttrs(func, mode, is_aiv)            # split (+ dual_aiv_dispatch for AIV)
  elif RequiresNoSplitDualAivSync(func):
      # Ascend910B no-split dual-AIV dispatch (orthogonal path).
      ProcessNoSplitDualAivFunction(func)
  else:
      pass through unchanged
```

## Algorithm — no-split dual-AIV dispatch

`ProcessNoSplitDualAivFunction` only fires when `RequiresNoSplitDualAivSync(func)`
is true — the backend is Ascend910B (or any backend whose
`BackendHandler::RequiresNoSplitDualAivDispatch()` returns true), the function is
AIV, and `attrs["dual_aiv_dispatch"]` is true.

```text
1. Clone params into param_replacements.
2. InjectSubblockIdx — prepend `subblock_idx = tile.get_subblock_idx()`.
3. Strip the leading subblock_idx assign, then split off the shared pipe-setup
   prefix: SplitNoSplitSharedPipeSetupPrefix takes the maximal prefix of
   reserve_buffer / import_peer_buffer / aic_initialize_pipe /
   aiv_initialize_pipe stmts so they run on both lanes from the original
   location.
4. Lane 0 body = the original branch stmts (unchanged).
5. Lane 1 body = BuildNoSplitLane1ReplayStmts(branch stmts):
     - tile.store: drop EvalStmt forms; for AssignStmt forms passthrough the
       destination tensor so SSA users still see a value, but no write happens.
     - any other call producing a TileType: rewrite via
       RebuildLane1CallWithZeroValidShape — tile.load becomes tile.create with
       valid_shape=[0, 0]; tile.slice / tile.set_validshape get their
       valid_shape zeroed; everything else has its result valid_shape cleared.
     - cross-core tile.tpush_* / tile.tpop_* / system.tfree_* are KEPT so the
       AIC↔AIV handshake stays balanced.
     - For/While/If recurse with a forked replacements map.
6. Wrap lane 0 / lane 1 in `if subblock_idx == 0: <lane 0> else: <lane 1>`.
7. New body = subblock_idx assign; hoisted shared pipe-setup; branch IfStmt.
8. Substitute / DeepClone; attrs unchanged (dual_aiv_dispatch=True stays).
```

### Codegen transport: full-column box, preserved rows

The lane-1 replay zeroes its tile `valid_shape` so it produces no visible
writes, but the AIC↔AIV `tpush` it keeps still moves data through the shared GM
FIFO slot the single cube consumer pops in full. On the codegen side
(`EmitSplitTpushTransportValidShape`, `pto_ops_common.cpp`) a no-split dual-AIV
producer that narrowed its `valid_shape` with `set_validshape` must transport
the **full column box**, or the consumer reads stale slot columns past
`valid_col`. The no-split path widens **columns only and preserves the row
`valid_shape`**: subblock 0's real push carries the full column box, while
subblock 1's `valid_shape=[0, 0]` replay gets **no transport at all** (a
statically 0-row push moves no data), so it stays a true 0-row no-op rather than
racing garbage rows into subblock 0's slot. The detection lever is
`PTOCodegen::IsDualAivDispatchFunction()`, which reads this pass's
`dual_aiv_dispatch` attribute.

## Examples

### Example 1 — split_aiv attribute stamping (the sole split path)

A `split_aiv` AIC/AIV pair arrives already in explicit per-lane form (halved
`[8, 128]` compute tiles, one hand/`LowerAutoVectorSplit`-written
`get_subblock_idx`). The pass stamps attrs and leaves the body untouched.

```python
@pl.function(type=pl.FunctionType.AIV,
             attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
def main_aiv(self, out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]):
    subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
    z_vec: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=1)
    return pl.store(z_vec, [0 + subblock_idx * 8, 0], out_0)
```

After the pass, `attrs` gains `dual_aiv_dispatch=True`; `z_vec` stays `[8, 128]`
(NOT re-halved) and there is exactly one `get_subblock_idx`.

### Example 2 — Ascend910B no-split dual-AIV dispatch

Distilled from
`test_no_split_dual_dispatch_producer_replays_compute_and_tpush_on_lane1`. The
AIV function carries `dual_aiv_dispatch=True` (set by `ExpandMixedKernel` for a
no-split mixed kernel) and no `split` attr.

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

**After** (lane 1 carries empty tiles via `valid_shape=[0, 0]`):

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
        a_tile: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = \
            pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        b_tile: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = \
            pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        summed: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = \
            pl.add(a_tile, b_tile)
        pl.tpush_to_aic(summed, split=0)        # handshake still fires
    return out
```

`reserve_buffer` and `aiv_initialize_pipe` are hoisted above the `if`/`else` so
both lanes share the same buffer state.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass SplitVectorKernel();
```

**Implementation**: `src/ir/transforms/split_vector_kernel_pass.cpp`

- `WithSplitAttrs` — stamps `split` (+ `dual_aiv_dispatch` for AIV) on the
  split_aiv arm.
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
passes.def("split_vector_kernel", &pass::SplitVectorKernel, ...);
```

**Tests**: `tests/ut/ir/transforms/test_split_vector_kernel.py`
(`TestSplitVectorKernelExplicitSplitAivBypass`, `TestSplitVectorKernelNoSplitA2A3`,
`TestSplitVectorKernelNoSplitPassthrough`). The per-op halving tests moved to
`tests/ut/ir/transforms/test_lower_auto_vector_split.py`.

## Related

- [`LowerAutoVectorSplit`](21-lower_auto_vector_split.md) — the live auto-split
  lowering path; produces the `split_aiv` functions this pass stamps. The per-op
  vector halving rules live there + in `split_axis_utils`.
- [`ExpandMixedKernel`](22-expand_mixed_kernel.md) — upstream producer of
  AIC/AIV functions and of the `dual_aiv_dispatch` marker.
- [`InjectGMPipeBuffer`](23-inject_gm_pipe_buffer.md) — runs immediately before;
  backend-gated GM pipe buffer wiring this pass relies on.
- [`NormalizeReturnOrder`](26-normalize_return_order.md) — runs immediately after.
