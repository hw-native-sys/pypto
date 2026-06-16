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
- Compatible **byte** sizes (reuse target must be large enough)
- **No-alias guard** (op-semantic): the op that defines the reusing variable may forbid its output from sharing a buffer with one or more of its input operands. Two registry declarations feed a single check:
  - `not_inplace_safe()` — the op cannot run with `src == dst`, so its output must not alias **any** input operand (e.g. `tile.recip`, `tile.mrgsort_format1`).
  - `forbid_output_alias(i)` — the op is in-place-safe w.r.t. its value operands but reads a **specific** operand while writing its output, so the output must not alias that one operand's buffer (e.g. `tile.sel`'s predicate mask (arg 0) and tmp scratch (arg 3), which the TSEL intrinsic reads while writing `dst`).

  Both reduce to the same per-output set of forbidden input vars (collected by `ForbidAliasCollector`); MemoryReuse refuses to place the output on the root owner — or any variable already sharing the buffer — of a forbidden operand.

**No shape / dtype / TileView compatibility gate**: tiles that share a physical MemRef may carry **different** shapes, dtypes, or `TileView` attributes. PTO codegen binds a per-variable `alloc_tile` to each tile, so each alias declares the shared base with its own static shape / dtype / layout / `valid_shape`. This permits, for example:

- cross-dtype reuse — a BF16 tile reusing a dead FP32 tile's buffer (e.g. across `tile.cast`);
- `tile.fillpad` output reusing its input, and two fillpad outputs with different `pad` sharing one buffer;
- N-D tiles with divergent `valid_shape` sharing a buffer (each keeps its own `valid_shape` on its own `alloc_tile`);
- L0 cube-input `Left` / `Right` sub-tiles of differing shape sharing one slot (e.g. fused-attention QK `Right` `[k, SEQ]` reused by PV `Right` `[k', HEAD]`, halving peak L0B — issue #1595).

  Earlier revisions gated reuse on an `AreTileTypesCompatible` shape / dtype / view match (with a narrow L0 byte-reuse exception); that gate has been removed. Correctness for read-while-write ops is now handled precisely by the no-alias guard above rather than by a coarse whole-tile match.

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
- Tests byte-size compatibility
- Tests cross-dtype / cross-`TileView` reuse (now permitted: BF16↔FP32, fillpad output↔input, divergent `valid_shape`)
- Tests the no-alias guard (`not_inplace_safe` ops and `tile.sel`'s `forbid_output_alias` mask/tmp operands)
- Tests view operation MemRef sharing preservation
- Tests redundant alloc statement removal
- Tests control flow lifetime analysis (nested IfStmt in ForStmt, branch variable sharing)
