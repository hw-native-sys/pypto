# LowerAutoVectorSplit Pass

Converts an AUTO `pl.split` mixed `InCore` function into the **explicit
`split_aiv` form** *before* `ExpandMixedKernel`. It inserts `tile.aiv_shard`
at cube→vector boundaries and `tile.aic_gather` at vector→cube boundaries,
halves only the **vector sub-region** along the split axis, injects
`tile.get_subblock_idx()`, and stamps `split` + `split_aiv` on the function.

This is the **live auto-split lowering path**: it always runs, immediately
before `ExpandMixedKernel`. After it runs, every split function reaches
[`SplitVectorKernel`](21-split_vector_kernel.md) already `split_aiv`-marked,
so that pass only stamps attributes (its split_aiv arm) — its former per-op
halving driver was deleted, and the halving machinery now lives solely in
`split_axis_utils`, shared by this pass.

This pass is also the **sole consumer of the first-class `SplitAivScopeStmt`
region node** (`for aiv_id in pl.split_aiv(...)`). The region survives parse →
SSA → `ResolveBackendOpLayouts` as a structural node; here each region is
lowered in place and the scope wrapper is **erased**, so no `SplitAivScopeStmt`
reaches `ExpandMixedKernel` (pass 22) or codegen.

## Why this pass exists

A mixed `InCore` function written with `pl.split` describes cube and vector
work in one body, with the split intent expressed only by the function-level
`split` mode. Two ways to realize that split were possible:

1. **Late, per-op halving in `SplitVectorKernel`** — after `ExpandMixedKernel`
   has already separated the body into AIC + AIV functions with cross-core
   `tpush`/`tpop`, halve the AIV body op-by-op. This duplicated the boundary
   semantics already encoded by `tile.aiv_shard` / `tile.aic_gather`.
2. **Early, explicit lowering (this pass)** — rewrite the AUTO `pl.split`
   body into the same explicit `split_aiv` shape a hand-authored kernel uses,
   *before* `ExpandMixedKernel`. Then the single op-driven boundary arm in
   `ExpandMixedKernel` folds `tile.aiv_shard` / `tile.aic_gather` into
   split-stamped `tpush`/`tpop` uniformly — auto and hand-written kernels take
   the identical downstream path.

Approach 2 is the live path. It is byte-identical to the old per-op halving
(proved during the staged convergence) because both call the same
`split_axis::ProcessStmts` machinery; only the entry point and the boundary
handling differ.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::LowerAutoVectorSplit()` | `passes.lower_auto_vector_split()` | Program-level |

```python
from pypto import passes
result = passes.lower_auto_vector_split()(program)
```

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `SSAForm` |
| Produced | `SSAForm` |
| Invalidated | — |

Source: `include/pypto/ir/transforms/pass_properties.h`
(`kLowerAutoVectorSplitProperties`).

## Scope

A function is rewritten iff **all** of:

- `func_type_ == FunctionType::InCore`, and
- it carries a function-level split mode (`UpDown` / `LeftRight`,
  `mode != None`), and
- it is **not already** `split_aiv` (hand-authored explicit kernels are left
  untouched — they already carry the explicit shard/gather form), and
- it is **genuinely mixed** (cube↔vector): its rolled-up affinity is `MIXED`,
  the same `ClassifyCallAffinity` / `CombineAffinity` decision `ExpandMixedKernel`
  uses for `is_mixed`.

Everything else is passed through unchanged. The last condition matters: a
**pure-vector** `pl.split` function (e.g. an elementwise op split across the two
AIV lanes, with no cube and no C↔V boundary) has nothing to converge. It is left
untouched, so `ExpandMixedKernel` converts it to a plain AIV function and strips
its `split` attr exactly as before — preserving its prior (un-split) behavior.
Were it lowered here, it would carry `split_aiv` without a `split` mode after
that strip, and `SplitVectorKernel` would reject it.

## Explicit `SplitAivScopeStmt` region path

In addition to the AUTO whole-function path above, an `InCore` function whose
body still carries one or more `SplitAivScopeStmt` regions takes a separate
**region path** (`LowerExplicitRegionFunction`), checked **before** the AUTO
path. Each region carries its own `split_` mode, so this handles the multi-mode
case the single function-level mode cannot. Region-local `tile_vars` /
`var_replacements` maps keep a halved var from leaking into a sibling region or
an out-of-region full-width op. Statements **outside** any region are emitted
full-width. After all regions are lowered, the scope wrappers are dropped and the
function is stamped `split_aiv` + `split_aiv_region_validated` (the latter signals
[`ExpandMixedKernel`](19-expand_mixed_kernel.md) to skip its single-func-mode
transpose check — pass 21 validates each region's transpose hazard with the
correct per-region split axis instead).

A function-level AUTO split (`optimizations=[pl.split(mode)]`) and explicit
`pl.split_aiv` regions are **mutually exclusive** — a scope carrying both is
rejected. This is enforced earlier, at
[`OutlineIncoreScopes`](08-outline_incore_scopes.md), where the scope's own
`split_` (the user's `pl.split`) and its regions are both still visible; the
combination is rejected there because this region path would otherwise lower per
region and silently drop the function-level split. (Post-outline the two merge
indistinguishably: a *single* `pl.split_aiv` region legitimately derives a
function-level representative `split` mode, so the conflict cannot be detected
here.)

Three region body shapes are handled, selected by the region's `split_` mode:

- **Data-parallel, full-width body** (`UpDown` / `LeftRight`, no explicit boundary
  op): the region body holds full-width vector compute. The region path injects a
  per-region `subblock_idx`, routes the vector ops through the shared
  `split_axis::ProcessStmts` halving machinery (region-scoped), and validates the
  per-region transpose hazard. This is the paradigm the auto-converged form
  produces.
- **Data-parallel, explicit boundary body** (`UpDown` / `LeftRight` with
  `tile.aiv_shard` / `tile.aic_gather` already present): the user manually sharded
  the cube tile and wrote the vector compute on the per-lane half, so the body is
  **already** in half-width form. The region path detects this
  (`RegionBodyHasExplicitBoundary`) and **splices the body through unchanged** — no
  re-halving, no duplicate `subblock_idx`. Re-running the halving here would
  double-shard (a downstream Acc→Vec move would be misread as a fresh cube→vector
  boundary and rewritten to a second `aiv_shard`), orphaning a halved Acc memref
  with no allocation and crashing PTO codegen. `ExpandMixedKernel` folds the
  explicit boundary into `tpush`/`tpop` exactly as for a hand-authored split_aiv
  kernel.
- **Task-parallel body** (`None`): there is **no split axis** — both AIV lanes run
  the **full** body for disjoint work the author dispatches via the region's
  `aiv_id` lane index (e.g. an `aiv_id`-strided loop). The region path **splices
  the body through unchanged** (no halving, no offset localization, no injected
  `subblock_idx`; the author's `aiv_id = get_subblock_idx()` binding already
  carries the lane). A `tile.aiv_shard` / `tile.aic_gather` inside a `None` region
  is rejected (nothing to shard without a split axis) — both by the `AivSplitValid`
  verifier and by an always-on guard here. The function is still stamped
  `split_aiv`, so downstream [`ExpandMixedKernel`](19-expand_mixed_kernel.md) /
  `SplitVectorKernel` dispatch it to **both** AIV lanes (via `dual_aiv_dispatch`)
  and **not** the lane-0-only no-split replay (which is only for non-`split_aiv`
  kernels) — so both lanes run the full body. Use this when the region's tiles
  cannot be halved (unit dims) or a reduction must stay full-width.

Because the region is built via the generic `BeginScope`/`EndScope` and is
non-outlined, it can be **nested** inside a `pl.range` / `pl.pipeline` loop or an
`if`; the region path recurses into compound statements to find and lower every
region while preserving the surrounding control flow.

## Split-axis dispatch

| `SplitMode` (int) | Split axis | Vector sub-region halved on |
| ----------------- | ---------- | --------------------------- |
| `None` (0) | — (no split axis) | nothing — task-parallel; tiles stay FULL, `aiv_id` dispatches both lanes |
| `UpDown` (1) | dim 0 (height) | rows |
| `LeftRight` (2) | dim 1 (width) | cols |

`SplitDimension(mode)` returns `0` for `UpDown`, `1` for `LeftRight`
(`split_axis_utils`); it is **not** called for `None` (the region path branches on
`None` first — there is no axis to derive).

## Algorithm

`LowerFunction` rewrites one mixed `InCore` function:

```text
1. split_dim = SplitDimension(mode); split_int = int(mode).
2. InjectSubblockIdx(func, is_aiv=true) prepends
       subblock_idx = tile.get_subblock_idx()
   to the body (fresh name if 'subblock_idx' is taken).
3. LowerStmts walks the flat body:

   Boundary tile.move (ClassifyMoveDirection):
     CUBE_TO_VECTOR — replace the move with
         tile.aiv_shard(full_cube_tile, split=int(mode))   -> HALF
       The deduced HALF type already carries the consuming-lane memory
       (Vec): the split deducer leaves memory_space null and
       OpRegistry::Create fills it from tile.aiv_shard's set_output_memory
       declaration, so this path and the explicit pl.aiv_shard form read
       one declaration. Seed the result var into tile_vars (its half
       extent) and record the old->new var rebind. The cube source (the
       matmul / Acc result) stays FULL.
     VECTOR_TO_CUBE — insert
         tile.aic_gather(half_vector_tile, split=int(mode))  -> FULL
       resolving the source to its halved var so the gather doubles
       HALF -> FULL, then keep the original cube-placement move on the
       gathered FULL tile (named "<dest>_mat" so ExpandMixedKernel's V->C
       boundary names its synthesized tpop after it).

   Affinity gate (ClassifyCallAffinity):
     VECTOR-affine leaf — route the single statement through
       split_axis::ProcessStmts({stmt}, ..., is_aiv=true): the SAME machinery
       the deleted SplitVectorKernel driver used. Halves tile.load /
       tile.store / tile.slice / tile.reshape / compute results on split_dim,
       localizes offsets per subblock, tracks halved vars in tile_vars.
     CUBE-affine leaf — passed through FULL, never halved.

   ForStmt / IfStmt — recurse into the body for vector content.

4. CheckNoCubeTileHalved re-walks the rebuilt body and asserts no CUBE-affine
   op consumes or produces a tile in tile_vars (the affinity gate must never
   leak a halved tile into a cube operand) — INTERNAL_CHECK on failure.
5. transform_utils::Substitute applies var_replacements; DeepClone detaches
   shared sub-trees.
6. WithSplitAivAttrs stamps split + split_aiv (dropping any prior split /
   split_aiv / dual_aiv_dispatch entries).
```

The per-op vector halving (shape halved on the split axis, offset localized by
`subblock_idx * half`, `tile.slice` static-shape-arg halving in lockstep with
the result type, rank-1-load reshape sliced per lane, reduce-on-split-axis
rejected, singleton split-dim preserved, loop `iter_arg`/`return_var`
tracking) is all produced by `split_axis::ProcessStmts` / `ProcessStmt` —
documented in detail in the shared machinery; the same facts are exercised by
`tests/ut/ir/transforms/test_lower_auto_vector_split.py`.

## The affinity gate

Only **vector** work is halved; cube work stays full. Affinity is decided by
`core_affinity::ClassifyCallAffinity` (memory-space driven): an op producing or
consuming a `Vec` tile is `VECTOR`; matmul operands and the Acc/Mat cube result
are `CUBE`. The C→V boundary `tile.aiv_shard` is the seam: the FULL cube tile is
its input, the HALF vector tile is its output. `CheckNoCubeTileHalved` is the
backstop — if a cube operand were ever shrunk, it fires.

## Example — cube→vector boundary, vector region halved (UpDown)

A mixed kernel: a cube tile (`Mat`) crosses to `Vec`, a vector `add` runs on
it, the result is stored.

**Before** (post-InferTileMemorySpace mixed `InCore`):

```python
@pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
def split_auto(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat],
               out_0: pl.Out[pl.Tensor[[128, 128], pl.FP32]]):
    popped: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec] = pl.tile.move(qk, target_memory=pl.Mem.Vec)
    y: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec] = pl.add(popped, popped)
    return pl.store(y, [0, 0], out_0)
```

**After**:

```python
@pl.function(type=pl.FunctionType.InCore,
             attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True})
def split_auto(qk, out_0):
    subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
    popped: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.tile.aiv_shard(qk, split=1)  # C->V, HALF
    y: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec] = pl.add(popped, popped)
    return pl.store(y, [0 + subblock_idx * 64, 0], out_0)
```

The cube operand `qk` stays `[128, 128]`; the vector sub-region is halved to
`[64, 128]` and the store offset is localized per subblock.

## Example — vector→cube boundary stays full (UpDown)

A V→C `tile.move` becomes `tile.aic_gather`; the cube placement move on the
gathered tile keeps the FULL `[128, 128]` `Mat` shape — the cube side never
sees a halved tile:

```python
gathered_mat: pl.Tile[[..], pl.FP32, pl.Mem.Mat]  = pl.tile.aic_gather(vec, split=1)
gathered:     pl.Tile[[128, 128], pl.FP32, pl.Mem.Mat] = pl.tile.move(gathered_mat,
                                                                      target_memory=pl.Mem.Mat)
```

The gather result is `Mat`, not `Vec`: the declared type of a boundary op names
the **consuming** lane's space, and AIC pops a V→C transfer into L1. (`Vec` would
name the *producing* lane, contradicting the mirror op `tile.aiv_shard`, which
declares the vector-side `Vec` for its cube-produced operand.) The cube placement
move that follows is what puts the tile in its final operand space — `Mat → Left`
for a matmul operand; the `Mat → Mat` shown here is a no-op that survives only
because the pass preserves the author's original move.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass LowerAutoVectorSplit();
```

**Implementation**: `src/ir/transforms/lower_auto_vector_split_pass.cpp`

- `LowerFunction` / `LowerStmts` — boundary rewrite + affinity-gated halving.
- `MakeReshapeOpCall` — builds `tile.aiv_shard` / `tile.aic_gather` calls.
- `CheckNoCubeTileHalved` — cube-operand integrity backstop.
- `WithSplitAivAttrs` — stamps `split` + `split_aiv`.

**Shared machinery**: `src/ir/transforms/utils/split_axis_utils.cpp`
(`ProcessStmts`, `InjectSubblockIdx`, `SplitDimension`, `IsReduceOnSplitAxis`)
— the per-op vector halving, shared with `SplitVectorKernel`'s standalone-split
arm (`ProcessStandaloneSplitFunction`) and the `AivSplitValid` verifier
(`SplitDimension` / `IsReduceOnSplitAxis`).

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("lower_auto_vector_split", &pass::LowerAutoVectorSplit, ...);
```

**Tests**: `tests/ut/ir/transforms/test_lower_auto_vector_split.py` and the
end-to-end `pl.split` golden scenarios in
`tests/st/codegen/torch/test_torch_codegen_cross_core.py`
(`test_lower_auto_vector_split_golden`).

## Related

- [`ResolveBackendOpLayouts`](17-resolve_backend_op_layouts.md) — runs
  immediately before.
- [`ExpandMixedKernel`](19-expand_mixed_kernel.md) — runs immediately after;
  folds `tile.aiv_shard` / `tile.aic_gather` into split-stamped `tpush`/`tpop`.
- [`SplitVectorKernel`](21-split_vector_kernel.md) — downstream; only stamps
  attrs for the `split_aiv` functions this pass produces, plus the no-split
  dual-AIV path.
