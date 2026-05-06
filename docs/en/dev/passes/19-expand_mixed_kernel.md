# ExpandMixedKernel Pass

Expands mixed InCore functions into separate AIC (Cube) + AIV (Vector) kernels wrapped in a Group function. Non-mixed InCore functions get their FunctionType converted to AIC or AIV.

## Overview

After `OutlineIncoreScopes` and `ConvertTensorToTileOps`, InCore functions may contain both Cube ops (`tile.matmul`, `tile.gemv`, etc.) and Vector ops (`tile.add`, `tile.exp`, etc.). Some ops like `tile.load`, `tile.store`, `tile.move`, and `tile.reshape` are classified as Cube or Vector based on the MemorySpace of their tile operands. Functions containing ops from both sides are **mixed InCore functions**. Hardware requires Cube and Vector operations to run on separate core types, so this pass splits them into:

- **AIC function** (`FunctionType::AIC`) — contains only Cube + shared ops
- **AIV function** (`FunctionType::AIV`) — contains only Vector + shared ops
- **Group function** (`FunctionType::Group`) — calls AIC then AIV, replaces the original

When an existing Group function already calls the InCore function (e.g. from `OutlineClusterScopes`), the pass **rewrites that Group in-place** to call AIC + AIV directly, avoiding a redundant Group wrapper. When a standalone `Spmd` wrapper calls the InCore function, the pass preserves the `Spmd` wrapper and creates a `Group` callee under it so launch semantics stay on `FunctionType::Spmd`.

For **non-mixed InCore functions** (pure Cube or pure Vector), the pass converts `FunctionType::InCore` to the corresponding type without splitting:

- Pure Cube → `FunctionType::AIC`
- Pure Vector or shared-only → `FunctionType::AIV`

After this pass, no `FunctionType::InCore` functions remain in the program.

Cross-core data transfer at CV boundaries is handled by splitting explicit `tile.move` ops into `tpush`/`tpop` pairs:

| Direction | AIC side | AIV side |
| --------- | -------- | -------- |
| Cube→Vector (e.g. Acc→Vec) | `tpush_to_aiv(source_tile)` | `dest_var = tpop_from_aic()` |
| Vector→Cube (e.g. Vec→Mat/Left/Right) | `dest_var = tpop_from_aiv()` | `tile.move` to adapt fractal layout, then `tpush_to_aic(adapted_tile)` |

**Fractal TileView layout**: Cross-core transfer tile views are computed by `BuildCrossCoreTransferView` based on the destination memory space. The mapping differs between backends:

Ascend950 (a5) — hardware cross-core pipe carries data in fractal layout:

| Direction | Push/Pop TileView (blayout, slayout) | Name |
| --------- | ------------------------------------ | ---- |
| Vec→Left | col_major, row_major | NZ |
| Vec→Right | row_major, col_major | ZN |
| Vec→Mat | must be explicitly set in move | — |
| Mat/Acc→Vec | must be explicitly set in move | — |

Ascend910B (a2a3) — cross-core transfer goes through GM → Mat, and Mat only supports the NZ layout. Both Left and Right destinations use NZ at the transfer boundary; the final Left/Right layout is resolved by the subsequent `Mat → Left/Right` `tile.move` (MTE1):

| Direction | Push/Pop TileView (blayout, slayout) | Name |
| --------- | ------------------------------------ | ---- |
| Vec→Left | col_major, row_major | NZ |
| Vec→Right | col_major, row_major | NZ |
| Vec→Mat | preserve original | — |
| Mat/Acc→Vec | preserve original | — |

On both backends, the AIV push side (V→C) inserts a `tile.move` before `tpush_to_aic` to convert the source tile into the required fractal layout. The `tile.move` helper (`CreateMove`) propagates `blayout`/`slayout` kwargs when the result type carries a TileView.

When split kernels contain cross-core `tpush`/`tpop`, the pass also prepends the required frontend pipe setup automatically:

- `system.reserve_buffer(...)` on the consumer side
- `system.import_peer_buffer(...)` on the producer side
- `system.aic_initialize_pipe(...)` / `system.aiv_initialize_pipe(...)` on both sides

In addition, the pass inserts `system.tfree_to_aic(...)` / `system.tfree_to_aiv(...)`
after every consumer-side `tpop` chain.

Setup is derived from the split bodies:

- `dir_mask`: `C2V=1`, `V2C=2`, bidirectional=`3`
- `id`: omitted for automatic setup, so PTOAS uses the default frontend pipe id `0`
- `slot_size`: max tile byte size across all directions (`shape * dtype bits / 8`)
- `slot_num`: `8` for unidirectional, `4` per direction for bidirectional
- `buffer_size`: `slot_num * slot_size`
- buffer names: `<func>_c2v_slot_buffer` / `<func>_v2c_slot_buffer`
- reserve-buffer base: `AUTO` on insertion, then resolved to an explicit address by `AllocateMemoryAddr`

When cross-core directions use different tile sizes, the pass picks `max(all observed tile byte sizes)` as the common `slot_size` for `initialize_pipe`. Smaller tiles leave unused bytes in each slot but hardware correctness is preserved. Explicit user-authored programs can still create multiple independent pipes by supplying different `id` values to `initialize_pipe` and matching `tpush` / `tpop` / `tfree` ops.

For consumer-side cross-core tiles, the pass also ensures each `tile.tpop_*` has a matching
`system.tfree_*`. When an existing free is obviously too early, the pass delays it to a later statement in the same
block without reordering independent `tpop` chains. When AIC must post-process a `tile.tpop_from_aiv` result with a
same-side `tile.move` (for example Mat -> Left/Right/Bias), the generated `system.tfree_to_aiv(...)` is rewritten to
free the canonical popped tile value and can be delayed past that carrier chain.

For Ascend910B (a2a3), mixed kernels with **no function split mode** (`split` unset or `SplitMode.None`) are also
supported. In that case the pass keeps a single AIV kernel body, marks it for **dual AIV dispatch**, and later lowering
emits a runtime `subblock_idx` branch: AIV lane 0 executes the original body, while AIV lane 1 replays the cross-core
handshakes with tile-producing replay work forced to `valid_shape=[0, 0]`, and visible `tile.store` writes suppressed.
This keeps the AIC/AIV handshakes balanced for `pl.at(level=CORE_GROUP, optimizations=[pl.auto_chunk])` no-split mixed
kernels while the secondary sync lane avoids real DMA/compute work.

**Requirements**:

- Input IR must have tile ops (run `ConvertTensorToTileOps` first)
- Input IR must have InCore scopes outlined (run `OutlineIncoreScopes` first)
- Tile ops must be flattened to 2D (run `FlattenTileNdTo2D` first)
- Tile memory space must be inferred (run `InferTileMemorySpace` first)
- Cross-core fractal TileView assignment is supported on Ascend950 and Ascend910B backends

**When to use**: Run after `InferTileMemorySpace` when InCore functions may contain both Cube and Vector tile operations.

> **Note**: This pass is part of the default tile optimization pipeline. You can also invoke it explicitly via
> `passes.expand_mixed_kernel()(program)` when debugging or building a custom pass pipeline.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::ExpandMixedKernel()` | `passes.expand_mixed_kernel()` | Program-level |

**Python usage**:

```python
from pypto.pypto_core import passes

expand_pass = passes.expand_mixed_kernel()
program_expanded = expand_pass(program)
```

## Algorithm

```text
Phase 1 — Pre-scan:
  Identify InCore functions that have existing Group callers and
  callers that still need the original function name to remain callable.

Phase 2 — Expand each InCore function F:
  1. Recursively classify affinity of all statements (including inside loops/conditionals)
  2. Detect CV boundary moves: tile.move ops crossing cube↔vector memory spaces
     (recorded in a separate boundary_moves map, not as a distinct affinity)
  3. If not mixed (no CUBE ops or no VECTOR ops, and no boundary moves):
     convert FunctionType to AIC (pure Cube) or AIV (pure Vector / shared-only)
  4. Build AIC body: keep CUBE + SHARED stmts, prune VECTOR, recurse into MIXED loops
     - For boundary move (Cube→Vector): emit tpush_to_aiv(source_tile)
     - For boundary move (Vector→Cube): emit dest_var = tpop_from_aiv() with fractal TileView
  5. Build AIV body: symmetric (keep VECTOR + SHARED, prune CUBE)
     - For boundary move (Cube→Vector): emit dest_var = tpop_from_aic() with fractal TileView
     - For boundary move (Vector→Cube): emit tile.move to adapt fractal layout, then tpush_to_aic(adapted_tile)
  5a. Assign fractal TileView to all boundary tpop result types and pre-tpush tile.move ops
      via BuildCrossCoreTransferView:
      - Ascend950: Left→NZ, Right→ZN, Mat/Vec→preserve
      - Ascend910B: Left→NZ, Right→NZ (Mat only supports NZ), Mat/Vec→preserve
  6. Repair loop-carried state on both bodies
     - Strip dead iter_args whose carried values are unused on this side
     - Pull back missing init-value definitions for surviving iter_args
     - Rewrite dangling yields to identity yields when a branch-local value was stripped
     - Remap dangling tile.store result vars (SSA versions stripped by AIC-side splitting) to the corresponding output parameter
  7. Run dead code elimination on both bodies (recursive into loops)
  8. Normalize loop-carried state again, because DCE may remove a SHARED-only
     post-loop use that temporarily kept an iter_arg alive
  9. Run dead code elimination again to clean up init-value chains exposed
     by the second strip
 10. Ensure each consumer-side `tpop` chain has a matching
     `system.tfree_to_aic` / `system.tfree_to_aiv`, delaying obviously early frees within the same block when needed
 11. If the split bodies use cross-core tile ops and do not already contain setup,
      derive reserve/import/initialize_pipe prologues and prepend them
 12. Create AIC function (no return) and AIV function (original return)
     - On Ascend910B no-split mixed kernels, tag the generated AIV with dual-dispatch metadata
       so later lowering launches the same AIV kernel on both vector lanes and rewrites the secondary lane's tile
       replay path to `valid_shape=[0, 0]`
 13. If a non-Group caller still needs the original function name
     (for example a standalone Spmd wrapper): also create a Group
     function (calls AIC then AIV)
 14. If there are only existing Group callers: skip the extra Group wrapper

Phase 3 — Rewrite Group callers:
  For each Group function that calls a split InCore, replace the InCore call
  with an AIC call + AIV call sequence (EvalStmt for AIC, AssignStmt for AIV).
```

**Affinity classification**:

| Affinity | Ops | Classification Rule |
| -------- | --- | ------------------- |
| CUBE | `tile.matmul`, `tile.matmul_acc`, `tile.matmul_bias`, `tile.gemv`, `tile.gemv_acc`, `tile.gemv_bias`, `tile.batch_matmul`, `tile.batch_matmul_acc` | Always CUBE (op name) |
| CUBE or VECTOR | `tile.load` | By `target_memory` kwarg: cube memory (Mat, Left, Right, Acc, Bias) → CUBE; Vec → VECTOR |
| CUBE or VECTOR | `tile.store`, `tile.reshape` | By source tile's `memory_space`: cube memory → CUBE; Vec → VECTOR |
| MIXED | `tile.move` crossing cube↔vector memory | Leaf cross-side move — also recorded in the `boundary_moves` map (see below) |
| CUBE or VECTOR | `tile.move` (same-side) | By source tile's `memory_space` |
| VECTOR | All other `tile.*` ops (`tile.add`, `tile.exp`, `tile.sub`, etc.) | Always VECTOR (op name) |
| SHARED | Non-tile ops, function calls, control flow, scalar ops | — |
| MIXED | Compound statements containing both CUBE and VECTOR children | — |

**CV boundary detection**: A `tile.move` is a CV boundary when its source tile memory and target memory are on different core sides. Cube-side memory: Mat, Left, Right, Acc, Bias. Vector-side memory: Vec. Same-side moves (e.g. Mat→Left) are classified by their source memory as usual. Boundary leaf moves are tagged `MIXED` for affinity purposes and are also recorded in a separate `boundary_moves` map; the cross-core direction (Cube→Vector vs Vector→Cube) is recovered via `ClassifyMoveDirection` at the call sites that need it (`CollectCVBoundaryMoves`, `BuildCoreBody`).

**Nested structure handling**: ForStmt, IfStmt, and WhileStmt containing mixed ops are duplicated into both AIC and AIV bodies with recursively pruned contents.

**Loop-state repair after splitting**: mixed-loop control flow is intentionally preserved during body construction, which can leave one side with extra iter_args, missing init-value definitions, or yields that reference stripped branch-local values. The pass repairs those cases before DCE, then normalizes loop-carried state once more after DCE because dead shared aliases can disappear and make an iter_arg removable only at that stage. A final DCE pass cleans up any init-value chains that become dead after the second strip.

## Example 1: InCore without existing Group caller

When Orchestration calls InCore directly, a new Group wrapper is created.

**Before** (after `InferTileMemorySpace`):

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def compute_incore_0(self, x: pl.Tensor[[16, 128], pl.BF16],
                         y: pl.Tensor[[128, 128], pl.BF16],
                         out_0: pl.Out[pl.Tensor[[16, 128], pl.FP32]]
                         ) -> pl.Tensor[[16, 128], pl.FP32]:
        x_mat: pl.Tile[[16, 128], pl.BF16] = pl.load(x, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
        y_mat: pl.Tile[[128, 128], pl.BF16] = pl.load(y, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        x_left: pl.Tile[[16, 128], pl.BF16] = pl.move(x_mat, target_memory=pl.Mem.Left)
        y_right: pl.Tile[[128, 128], pl.BF16] = pl.move(y_mat, target_memory=pl.Mem.Right)
        z_tile: pl.Tile[[16, 128], pl.FP32] = pl.matmul(x_left, y_right)
        z_vec: pl.Tile[[16, 128], pl.FP32] = pl.move(z_tile, target_memory=pl.Mem.Vec)
        out_0 = pl.store(z_vec, [0, 0], out_0)
        return out_0

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_incore_0(x, y, out_0)
```

**After** (conceptual — actual IR includes type annotations on all variables; auto-generated pipe setup is omitted here for brevity):

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.AIC)
    def compute_incore_0_aic(self, x, y, out_0):
        x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.Mem.Mat)  # CUBE: load to Mat
        y_mat = pl.load(y, [0, 0], [128, 128], target_memory=pl.Mem.Mat) # CUBE: load to Mat
        x_left = pl.move(x_mat, target_memory=pl.Mem.Left)   # CUBE: Mat→Left (same-side)
        y_right = pl.move(y_mat, target_memory=pl.Mem.Right)  # CUBE: Mat→Right (same-side)
        z_tile = pl.matmul(x_left, y_right)              # CUBE op
        pl.tile.tpush_to_aiv(z_tile, split=0)        # boundary move: push Acc tile to AIV

    @pl.function(type=pl.FunctionType.AIV)
    def compute_incore_0_aiv(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        # tpop result carries fractal TileView (no layout for Vec destination → preserved)
        z_vec: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec, pl.TileView()] = pl.tile.tpop_from_aic(split=0)
        out_0 = pl.store(z_vec, [0, 0], out_0)           # VECTOR op
        pl.system.tfree_to_aic(z_vec)                    # release the consumed cross-core slot
        return out_0

    @pl.function(type=pl.FunctionType.Group)
    def compute_incore_0(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        self.compute_incore_0_aic(x, y, out_0)
        result = self.compute_incore_0_aiv(x, y, out_0)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_incore_0(x, y, out_0)  # calls Group (same name)
```

## Example 2: InCore with existing Group caller

When `OutlineClusterScopes` has already created a Group function calling the InCore, the pass rewrites the existing Group instead of creating a new wrapper. When `OutlineClusterScopes` has created a standalone `Spmd` wrapper, the pass keeps that `Spmd` function and retargets it to a new `Group` callee with the original InCore name.

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def compute_incore_0(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        # ... mixed Cube + Vector ops ...

    @pl.function(type=pl.FunctionType.Group)
    def compute_group(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        result = self.compute_incore_0(x, y, out_0)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_group(x, y, out_0)
```

**After** — existing Group is rewritten, no `compute_incore_0` Group wrapper:

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.AIC)
    def compute_incore_0_aic(self, x, y, out_0):
        # ... Cube ops + tpush ...

    @pl.function(type=pl.FunctionType.AIV)
    def compute_incore_0_aiv(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        # ... tpop + Vector ops ...

    @pl.function(type=pl.FunctionType.Group)
    def compute_group(self, x, y, out_0) -> pl.Tensor[[16, 128], pl.FP32]:
        self.compute_incore_0_aic(x, y, out_0)       # rewritten: AIC call
        result = self.compute_incore_0_aiv(x, y, out_0)  # rewritten: AIV call
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x, y) -> pl.Tensor[[16, 128], pl.FP32]:
        out_0 = pl.create_tensor([16, 128], dtype=pl.FP32)
        return self.compute_group(x, y, out_0)  # unchanged
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Implementation**: `src/ir/transforms/expand_mixed_kernel_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_expand_mixed_kernel.py`

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SSAForm, IncoreTileOps, SplitIncoreOrch, TileOps2D, TileMemoryInferred |
| Produced | SSAForm, MixedKernelExpanded |
| Invalidated | — |

## Property Verifier

`MixedKernelExpandedPropertyVerifier` checks:

- no remaining `FunctionType::InCore` function contains both Cube and Vector tile ops
- `tile.tpop_from_aiv` in AIC lands in `MemorySpace::Mat`
- `tile.tpop_from_aic` in AIV lands in `MemorySpace::Vec`
- AIC/AIV functions with cross-core `tpush`/`tpop` also contain the required pipe setup
- every AIC/AIV `tile.tpop_*` has a matching `system.tfree_*`
- cross-core tile ops have statically known tile shapes (required for auto pipe setup)

This moves common failures (missing `initialize_pipe`, missing `reserve_buffer` / `import_peer_buffer`, missing `tfree`,
non-static tile sizes, mismatched cross-core `tpop`/`tfree` pairs) from PTO codegen / `ptoas` time to immediately after
`ExpandMixedKernel`. The verifier checks pairing by tile value/op; it does not prove that a `tfree` is placed after the
true last use of that tile.

## Design Decisions

| Decision | Rationale |
| -------- | --------- |
| Move-based CV boundary detection | Explicit `tile.move` ops mark boundaries — no fragile variable data-flow analysis needed |
| `boundary_moves` map (not a separate enum value) | Boundary status is derivable from `ClassifyMoveDirection` at the call sites that need it; storing it as a side map keeps `CoreAffinity` to the four execution-side categories (CUBE / VECTOR / SHARED / MIXED) |
| MemorySpace-based classification for data-movement ops | `tile.load`/`tile.store`/`tile.move`/`tile.reshape` serve Cube or Vector depending on which memory they touch; `InferTileMemorySpace` sets this before the pass runs |
| Fractal TileView on tpop results | tpop result types carry the fractal TileView (NZ/ZN) directly rather than stripping layout — downstream passes and codegen see the correct layout without extra inference |
| Pre-tpush `tile.move` on AIV side | V→C transfers require data in fractal layout; inserting an explicit `tile.move` before `tpush_to_aic` makes the layout conversion visible in IR |
| `CreateMove` propagates layout kwargs | When the result type has a TileView, `blayout`/`slayout` are forwarded as kwargs so the generated `tile.move` call is self-describing |
| Group keeps original function name | When no existing Group caller: Orchestration call sites work unchanged — no call-site rewriting needed |
| Rewrite existing Group callers | When a Group already calls the InCore (e.g. from `OutlineClusterScopes`): rewrite it in-place to call AIC + AIV, avoiding redundant Group→Group nesting |
| Preserve standalone Spmd wrappers | When a standalone `Spmd` calls the InCore: keep `FunctionType::Spmd`, create a `Group` callee underneath, and keep `core_num` / `sync_start` on the Spmd wrapper |
| Parameters copied to all three functions | Simplifies wiring; DCE removes unused params in downstream passes |
| Recursive compound-stmt handling | Correctly splits mixed ops inside `ForStmt`, `IfStmt`, `WhileStmt` |
| Two-stage post-split loop-state repair | First makes loop-carried state valid, then re-strips iter_args after DCE removes dead shared aliases, with a final DCE to clean up exposed init-value chains |
| Auto-generated pipe setup | Tensor-level mixed kernels do not need handwritten `reserve_buffer` / `import_peer_buffer` / `initialize_pipe`; the pass derives them from cross-core tile ops |
| Auto-generated tfree chains | Consumer-side split kernels insert missing `tfree` calls, rewrite them to free the canonical popped tile value, and delay obviously early frees within the same block without reordering independent `tpop` chains |
| Max-slot-size policy | Uses `max(all tile byte sizes)` as the single `initialize_pipe.slot_size`, matching the backend assumption of one automatic reserve/import buffer per direction while preserving legacy bidirectional `dir_mask=3` behavior |
