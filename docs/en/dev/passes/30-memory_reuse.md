# MemoryReuse Pass

Uses dependency analysis to identify memory reuse opportunities and removes redundant alloc operations.

## Overview

This pass analyzes variable lifetimes and dependencies to enable memory sharing. Variables with non-overlapping lifetimes in the same memory space can share MemRef objects, reducing memory footprint.

After applying MemRef sharing, the pass also **removes redundant `tile.alloc` statements** for MemRefs that are no longer referenced by any TileType variable.

**Key insights**:

- Variables that don't overlap in lifetime can reuse memory
- Only variables in the same memory space can share MemRef
- Lifetime is determined by def-use analysis
- After sharing, MemRefs that become unreferenced are cleaned up along with their alloc statements

**When to use**: Run after InitMemRef and before AllocateMemoryAddr. Reduces memory allocation overhead.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::MemoryReuse()` | `passes.memory_reuse()` | Function-level |

**Factory function**:

```cpp
Pass MemoryReuse();
```

**Python usage**:

```python
from pypto.pypto_core import passes

reuse_pass = passes.memory_reuse()
program_optimized = reuse_pass(program)
```

## Algorithm

1. **Lifetime Analysis**: Walk the full IR tree (including nested control flow bodies) to compute variable lifetimes via def-use analysis. Variables defined outside a loop but used inside have their lifetime extended to the end of the loop (loop-aware extension)
2. **Interference Check**: Identify variables with overlapping lifetimes
3. **MemRef Sharing**: Assign same MemRef pointer to non-interfering variables in the same memory space
4. **Loop-carry re-alignment** (`AlignLoopCarriesToInitMutator`): Sharing (step 3) only retypes `AssignStmt`-defined vars (producers/init); loop-carried `iter_arg`/`return_var` nodes are excluded from the lifetime/sharing maps and keep their original MemRef. This step walks `ForStmt`s **top-down** and retypes each loop's `iter_arg`/`return_var` to its (now-reused) `initValue` MemRef, seeding `var_remap_` before recursing so a nested loop observes the corrected outer `iter_arg` as its init. Without it, a reused **nested pipelined `matmul_acc`** accumulator splits across two Acc buffers and step 5 emits invalid `acc→acc tile.move` ops that ptoas rejects on Ascend 910B ([#1352](https://github.com/hw-native-sys/pypto/issues/1352))
5. **Yield fixup**: Fix MemRef mismatches in control flow return variables:
   - **ForStmt**: Ensure all 4 loop-carry variables (initValue, iter_arg, yield value, return_var) share the same MemRef. Inserts `tile.move` before yield if MemRefs differ
   - **IfStmt**: Patch return_vars to match yield value's MemRef
6. **Remove redundant allocs**: Collect all MemRefs still referenced by TileType variables, then remove `tile.alloc` statements whose MemRef is no longer in use

**Reuse conditions**:

- Non-overlapping lifetimes (no interference). Two variables do NOT overlap when `prev.last_use <= curr.def` (i.e., the source's last use can be at the same statement as the target's definition, since inputs are read before outputs are written within a single statement).
- Same memory space
- Compatible sizes (reuse target must be large enough)
- **L0 cube-input exception (Left/Right)**: buffers in `Mem.Left` / `Mem.Right` hold sub-tiles produced by view ops (`tile.extract` / `tile.slice` / `tile.reshape`), which PTO codegen materialises per tile var at the buffer base. Two such buffers in the same L0 space, with non-overlapping lifetimes and sufficient **byte** size, may therefore share one slot even when their **shapes differ** — the `AreTileTypesCompatible` shape/dtype/view check below is skipped for them (gated on both producers being view ops, PTO view ops so the shared MemRef stays PTO-materialisable). This lets fused-attention reuse the QK `Right` buffer (`[k, SEQ]`) for the PV `Right` buffer (`[k', HEAD]`), halving peak L0B (issue #1595). All other spaces (Vec/Acc/Mat) keep the strict match:
- TileType compatibility — checked by `AreTileTypesCompatible`:
  - Same shape (all dimensions must match exactly)
  - Same dtype (e.g., FP32 vs BF16 prevents reuse, handling `tile.cast` automatically)
  - Same TileView storage attributes when present: `stride`, `start_offset`, `blayout`, `slayout`, `fractal`, `pad` must all match structurally (e.g., `tile.fillpad` changes `pad`, so its output cannot reuse its input — `pad` divergence alone blocks reuse)
  - View **presence** may differ when the present view is storage-trivial: a tile with no TileView has default physical storage (contiguous, zero offset, row-major/none-box, default fractal, no pad), so it is compatible with a tile whose view sets only `valid_shape` (all storage fields at their defaults). This lets reuse span structurally-cloned tiles with asymmetric views, e.g. the two mutually-exclusive arms of a dual-AIV dispatch `if` produced by `SplitVectorKernel`, where one arm's tiles carry a trivial `valid_shape` view and the other's carry none. A view whose storage fields diverge from the defaults is still incompatible with a no-view tile.
  - `valid_shape` is **not** required to match for 2D tiles: each reused tile keeps its own `valid_shape` in its TileType, and PTO codegen emits per-variable `alloc_tile` declarations that alias the shared buffer with each member's own static valid extent. This lets sibling-branch tiles produced by `PartialUnrollTileLoops` (differing only in boundary-guard `valid_shape`) share one backing allocation. For N-D tiles, `valid_shape` divergence still blocks reuse.

**Alloc cleanup**:

After MemRef sharing, some MemRef objects become unreferenced (their variables now point to a different shared MemRef). The pass traverses the surrounding `SeqStmts` and removes any `tile.alloc` `AssignStmt` whose LHS MemRef pointer is not in the set of still-used MemRefs.

## Ascend910B load + tpop_from_aic hazard

On Ascend910B AIV functions with a non-`None` `SplitMode`, a writer that consumes **both** a `tile.load` result (or a legal-view descendant of one) **and** a `tile.tpop_from_aic` value must not place its output in the same physical buffer as that load result. Allowing the writer's output to in-place-reuse the load buffer produces silently wrong results on this hardware.

MemoryReuse owns every buffer-coalescing decision, so it prevents the hazardous sharing from ever forming rather than relying on a later split. When the guard is active, the reuse decision is blocked exactly when:

- the writer's defining op consumes a `tile.tpop_from_aic` value, **and**
- the buffer member it would reuse in place (whose last use is the writer's def statement) is load-derived.

The guard is gated by `BackendHandler::RequiresSplitLoadTpopWorkaround()` (true only for Ascend910B) and the function being split-AIV; on every other backend / function kind the inputs are empty and reuse behaviour is unchanged. The writer is still free to reuse any **non**-load buffer — only the load + tpop in-place combination is rejected. (This guard previously lived in a dedicated `LegalizePTOBufferReuse` pass that split the buffer after the fact; it now folds into MemoryReuse.)

## Example

### MemRef Sharing with Alloc Cleanup

**Before** (after InitMemRef):

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
mem_vec_2: MemRefType = tile.alloc(Vec, -1, 16384, 2)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
# tile_a last use ↑
tile_c: Tile[[64, 64], FP32, memref=mem_vec_2] = tile.load(...)
# ]
```

**After** (tile_c reuses mem_vec_0 from tile_a, alloc for mem_vec_2 removed):

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
# mem_vec_2 alloc removed — no longer referenced
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
tile_c: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
# tile_c now shares mem_vec_0 with tile_a
# ]
```

### Producer-Consumer Reuse

When a variable's last use is at the same statement that defines a new variable (producer-consumer relationship), the new variable can reuse the old variable's memory because inputs are read before outputs are written:

```python
# Before:
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.create(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.muls(tile_a, 0.0)
# tile_a.last_use == tile_b.def → reuse allowed

# After:
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.create(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.muls(tile_a, 0.0)
# tile_b reuses mem_vec_0
```

### Overlapping Lifetimes (No Reuse)

When a variable is still alive **after** another variable's definition (last_use > def), their lifetimes truly overlap and they cannot share memory:

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.load(...)
# tile_a.last_use > tile_b.def → tile_a still live when tile_b is defined
# ]
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass MemoryReuse();
```

**Implementation**: `src/ir/transforms/memory_reuse_pass.cpp`

- `LifetimeAnalyzer` walks the full IR tree to compute variable lifetimes (including nested control flow)
- `ComputeLifetimes` builds MemRef sharing groups and lifetime intervals
- `IdentifyReuseOpportunities` finds reuse candidates
- `ApplyMemRefSharing` updates MemRef pointers via `MemRefSharingMutator`
- `YieldFixupMutator` fixes ForStmt/IfStmt yield/return_var MemRef mismatches after reuse (inserts `tile.move` when needed)
- `UsedMemRefCollector` gathers still-referenced MemRef pointers after sharing
- `RemoveUnusedAllocStatements` filters out redundant `tile.alloc` statements from `SeqStmts`

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("memory_reuse", &pass::MemoryReuse, "Memory reuse optimization");
```

**Tests**: `tests/ut/ir/transforms/test_memory_reuse.py`

- Tests non-overlapping lifetime reuse with MemRef sharing
- Tests producer-consumer reuse (last_use == def at same statement)
- Tests overlapping lifetime no-reuse
- Tests memory space separation
- Tests size and shape compatibility
- Tests dtype compatibility (cross-dtype reuse blocked, same-dtype reuse allowed)
- Tests view operation MemRef sharing preservation
- Tests redundant alloc statement removal
- Tests control flow lifetime analysis (nested IfStmt in ForStmt, branch variable sharing)
