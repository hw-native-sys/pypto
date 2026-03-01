# BasicMemoryReuse Pass

Uses dependency analysis to identify memory reuse opportunities and removes redundant alloc operations.

## Overview

This pass analyzes variable lifetimes and dependencies to enable memory sharing. Variables with non-overlapping lifetimes in the same memory space can share MemRef objects, reducing memory footprint.

After applying MemRef sharing, the pass also **removes redundant `block.alloc` statements** for MemRefs that are no longer referenced by any TileType variable.

**Key insights**:

- Variables that don't overlap in lifetime can reuse memory
- Only variables in the same memory space can share MemRef
- Lifetime is determined by def-use analysis
- After sharing, MemRefs that become unreferenced are cleaned up along with their alloc statements

**When to use**: Run after InitMemRef and before AllocateMemoryAddr. Reduces memory allocation overhead.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::BasicMemoryReuse()` | `passes.basic_memory_reuse()` | Function-level |

**Factory function**:

```cpp
Pass BasicMemoryReuse();
```

**Python usage**:

```python
from pypto.pypto_core import passes

reuse_pass = passes.basic_memory_reuse()
program_optimized = reuse_pass(program)
```

## Algorithm

1. **Dependency Graph**: Build dependency graph from data flow using `DependencyAnalyzer`
2. **Lifetime Analysis**: Compute def-use chains and live ranges for each variable
3. **Interference Check**: Identify variables with overlapping lifetimes
4. **MemRef Sharing**: Assign same MemRef pointer to non-interfering variables in the same memory space
5. **Remove redundant allocs**: Collect all MemRefs still referenced by TileType variables, then remove `block.alloc` statements whose MemRef is no longer in use

**Reuse conditions**:

- Non-overlapping lifetimes (no interference)
- Same memory space
- Compatible sizes (reuse target must be large enough)

**Alloc cleanup**:

After MemRef sharing, some MemRef objects become unreferenced (their variables now point to a different shared MemRef). The pass traverses OpStmts blocks and removes any `block.alloc` `AssignStmt` whose LHS MemRef pointer is not in the set of still-used MemRefs. Empty OpStmts blocks are removed entirely.

## Example

### MemRef Sharing with Alloc Cleanup

**Before** (after InitMemRef):

```python
# OpStmts [
mem_vec_0: MemRefType = block.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = block.alloc(Vec, -1, 16384, 1)
mem_vec_2: MemRefType = block.alloc(Vec, -1, 16384, 2)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = block.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = block.add(tile_a, ...)
# tile_a last use ↑
tile_c: Tile[[64, 64], FP32, memref=mem_vec_2] = block.load(...)
# ]
```

**After** (tile_c reuses mem_vec_0 from tile_a, alloc for mem_vec_2 removed):

```python
# OpStmts [
mem_vec_0: MemRefType = block.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = block.alloc(Vec, -1, 16384, 1)
# mem_vec_2 alloc removed — no longer referenced
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = block.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = block.add(tile_a, ...)
tile_c: Tile[[64, 64], FP32, memref=mem_vec_0] = block.load(...)
# tile_c now shares mem_vec_0 with tile_a
# ]
```

### Overlapping Lifetimes (No Reuse)

**Before/After** (no change — alloc statements preserved):

```python
# OpStmts [
mem_vec_0: MemRefType = block.alloc(Vec, -1, 16384, 0)
mem_vec_1: MemRefType = block.alloc(Vec, -1, 16384, 1)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = block.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = block.load(...)
tile_c: Tile[[64, 64], FP32, memref=...] = block.add(tile_a, tile_b)
# tile_a and tile_b are both live here → cannot reuse
# ]
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass BasicMemoryReuse();
```

**Implementation**: `src/ir/transforms/basic_memory_reuse_pass.cpp`

- `DependencyAnalyzer` builds the dependency graph
- `ComputeLifetimesFromDependencies` calculates live ranges
- `IdentifyReuseOpportunities` finds reuse candidates
- `ApplyMemRefSharing` updates MemRef pointers via `MemRefSharingMutator`
- `UsedMemRefCollector` gathers still-referenced MemRef pointers after sharing
- `RemoveUnusedAllocStatements` filters out redundant `block.alloc` statements from OpStmts

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("basic_memory_reuse", &pass::BasicMemoryReuse, "Memory reuse optimization");
```

**Tests**: `tests/ut/ir/transforms/test_basic_memory_reuse.py`

- Tests non-overlapping lifetime reuse with MemRef sharing
- Tests overlapping lifetime no-reuse
- Tests memory space separation
- Tests size compatibility
- Tests view operation (reshape) MemRef sharing preservation
- Tests redundant alloc statement removal
