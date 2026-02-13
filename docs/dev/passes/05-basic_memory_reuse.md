# BasicMemoryReuse Pass

Uses dependency analysis to identify memory reuse opportunities.

## Overview

This pass analyzes variable lifetimes and dependencies to enable memory sharing. Variables with non-overlapping lifetimes in the same memory space can share MemRef objects, reducing memory footprint.

**Key insights**:
- Variables that don't overlap in lifetime can reuse memory
- Only variables in the same memory space can share MemRef
- Lifetime is determined by def-use analysis

**When to use**: Run after InitMemRef and before AddAlloc. Reduces memory allocation overhead.

## API

| C++ | Python | Level |
|-----|--------|-------|
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

1. **Lifetime Analysis**: Compute def-use chains and live ranges for each variable
2. **Dependency Graph**: Build dependency graph from data flow
3. **Interference Check**: Identify variables with overlapping lifetimes
4. **Memory Space Grouping**: Group variables by memory space (UB vs DDR)
5. **MemRef Sharing**: Assign same MemRef to non-interfering variables in same space
6. **Size Compatibility**: Ensure shared variables have compatible sizes

**Reuse conditions**:
- Non-overlapping lifetimes (no interference)
- Same memory space (UB or DDR)
- Compatible sizes (exact match or can fit)

## Example

### Non-overlapping Lifetimes

**Before**:
```python
tile_a: Tile[[64, 64], FP32, MemRef(id=0, space=UB)] = block.load(...)
tile_b: Tile[[64, 64], FP32, MemRef(id=0, space=UB)] = block.add(tile_a, ...)
# tile_a last use here

tile_c: Tile[[64, 64], FP32, MemRef(id=1, space=UB)] = block.load(...)
# tile_c first use here (after tile_a's last use)
```

**After**:
```python
tile_a: Tile[[64, 64], FP32, MemRef(id=0, space=UB)] = block.load(...)
tile_b: Tile[[64, 64], FP32, MemRef(id=2, space=UB)] = block.add(tile_a, ...)
# tile_a last use here

tile_c: Tile[[64, 64], FP32, MemRef(id=0, space=UB)] = block.load(...)
# tile_c reuses MemRef(id=0) from tile_a (non-overlapping lifetimes)
```

### Overlapping Lifetimes (No Reuse)

**Before/After** (no change):
```python
tile_a: Tile[[64, 64], FP32, MemRef(id=0, space=UB)] = block.load(...)
tile_b: Tile[[64, 64], FP32, MemRef(id=1, space=UB)] = block.load(...)
tile_c: Tile[[64, 64], FP32, MemRef(id=2, space=UB)] = block.add(tile_a, tile_b)
# tile_a and tile_b are both live here → cannot reuse
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`
```cpp
Pass BasicMemoryReuse();
```

**Implementation**: `src/ir/transforms/basic_memory_reuse.cpp`
- Performs liveness analysis
- Builds interference graph
- Uses graph coloring for MemRef assignment
- Respects memory space boundaries

**Python binding**: `python/bindings/modules/passes.cpp`
```cpp
passes.def("basic_memory_reuse", &pass::BasicMemoryReuse, "Memory reuse optimization");
```

**Tests**: `tests/ut/ir/transforms/test_basic_memory_reuse.py`
- Tests non-overlapping lifetime reuse
- Tests overlapping lifetime no-reuse
- Tests memory space separation
- Tests size compatibility
- Tests loop-carried dependencies
