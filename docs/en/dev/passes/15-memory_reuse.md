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
4. **Yield fixup**: Fix MemRef mismatches in control flow return variables:
   - **ForStmt**: Ensure all 4 loop-carry variables (initValue, iter_arg, yield value, return_var) share the same MemRef. Inserts `tile.move` before yield if MemRefs differ
   - **IfStmt**: Patch return_vars to match yield value's MemRef
5. **Remove redundant allocs**: Collect all MemRefs still referenced by TileType variables, then remove `tile.alloc` statements whose MemRef is no longer in use

**Reuse conditions**:

- Non-overlapping lifetimes (no interference). Two variables do NOT overlap when `prev.last_use <= curr.def` (i.e., the source's last use can be at the same statement as the target's definition, since inputs are read before outputs are written within a single statement).
- Same memory space
- Compatible sizes (reuse target must be large enough)
- Full TileType compatibility — checked by `AreTileTypesCompatible`:
  - Same shape (all dimensions must match exactly)
  - Same dtype (e.g., FP32 vs BF16 prevents reuse, handling `tile.cast` automatically)
  - Same TileView attributes when present: all fields (`valid_shape`, `stride`, `start_offset`, `blayout`, `slayout`, `fractal`, `pad`) are compared via `TileView::operator==` (e.g., `tile.fillpad` changes `valid_shape` and `pad`, so its output cannot reuse its input)

**Alloc cleanup**:

After MemRef sharing, some MemRef objects become unreferenced (their variables now point to a different shared MemRef). The pass traverses the surrounding `SeqStmts` and removes any `tile.alloc` `AssignStmt` whose LHS MemRef pointer is not in the set of still-used MemRefs.

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
- `YieldFixupMutator` fixes ForStmt/IfStmt yield/return_var MemRef mismatches after reuse
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
