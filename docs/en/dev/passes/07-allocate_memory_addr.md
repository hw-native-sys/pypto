# AllocateMemoryAddr Pass

Assigns real memory addresses to existing alloc operations.

## Overview

This pass allocates concrete memory addresses for non-DDR MemRefs and updates the existing `block.alloc` statements in place. Unlike creating new alloc operations, this pass only modifies the address field of alloc statements that were created by InitMemRef (with `addr=-1`).

**Key responsibilities**:

- Collect unique MemRef objects from TileType variables
- Allocate sequential, 32-byte aligned addresses within each memory space
- Update MemRef addresses in all variable types
- Update `block.alloc` statement arguments with the allocated addresses

**When to use**: Run after BasicMemoryReuse (to respect shared MemRefs) and before code generation. Final pass in memory management pipeline.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::AllocateMemoryAddr()` | `passes.allocate_memory_addr()` | Function-level |

**Factory function**:

```cpp
Pass AllocateMemoryAddr();
```

**Python usage**:

```python
from pypto.pypto_core import passes

alloc_pass = passes.allocate_memory_addr()
program_with_addrs = alloc_pass(program)
```

## Algorithm

1. **Collect MemRefs**: Traverse function body to find all unique MemRef objects from TileType variables
2. **Group by memory space**: Organize MemRefs by memory space (Vec, Mat, Left, Right, Acc)
3. **Allocate addresses**: For each memory space, sort MemRefs by ID and assign sequential 32-byte aligned addresses starting from 0
4. **Update in place**: Use `MemRefUpdateMutator` to:
   - Replace old MemRef references in variable types (TileType/TensorType) with new MemRefs containing real addresses
   - Update existing `block.alloc` `AssignStmt`s: replace LHS MemRef and update addr argument in the Call expression

**Address allocation**:

- Each memory space has its own address space starting from 0
- Addresses are 32-byte aligned: `next_addr = align32(current_addr + size)`
- MemRefs are sorted by ID for deterministic allocation order
- DDR MemRefs are skipped (addresses managed externally)

## Example

### Before (after InitMemRef + BasicMemoryReuse)

```python
# OpStmts [
mem_vec_0: MemRefType = block.alloc(Vec, -1, 16384, 0)   # addr=-1 (unallocated)
mem_vec_1: MemRefType = block.alloc(Vec, -1, 16384, 1)   # addr=-1 (unallocated)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = block.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = block.add(tile_a, ...)
# ]
```

### After (addresses assigned)

```python
# OpStmts [
mem_vec_0: MemRefType = block.alloc(Vec, 0, 16384, 0)      # addr=0
mem_vec_1: MemRefType = block.alloc(Vec, 16384, 16384, 1)   # addr=16384 (aligned)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = block.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = block.add(tile_a, ...)
# ]
```

### Multiple Memory Spaces

```python
# Before:
mem_vec_0: MemRefType = block.alloc(Vec, -1, 2048, 0)
mem_left_1: MemRefType = block.alloc(Left, -1, 2048, 1)
mem_right_2: MemRefType = block.alloc(Right, -1, 2048, 2)
mem_acc_3: MemRefType = block.alloc(Acc, -1, 2048, 3)

# After (each space starts from addr=0):
mem_vec_0: MemRefType = block.alloc(Vec, 0, 2048, 0)
mem_left_1: MemRefType = block.alloc(Left, 0, 2048, 1)
mem_right_2: MemRefType = block.alloc(Right, 0, 2048, 2)
mem_acc_3: MemRefType = block.alloc(Acc, 0, 2048, 3)
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass AllocateMemoryAddr();
```

**Implementation**: `src/ir/transforms/allocate_memory_addr_pass.cpp`

- `MemRefCollectorVisitor` collects unique MemRefs from TileType variables
- `AllocateMemoryAddresses` assigns sequential aligned addresses per memory space
- `MemRefUpdateMutator` updates both variable types and `block.alloc` statement arguments in a single traversal
- DDR MemRefs are skipped (no address allocation needed)

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("allocate_memory_addr", &pass::AllocateMemoryAddr,
           "Allocates real memory addresses for existing alloc operations.");
```

**Tests**: `tests/ut/ir/transforms/test_allocate_memory_addr_pass.py`

- Tests address allocation with 32-byte alignment
- Tests multiple MemRef allocations
- Tests empty function (no tiles)
- Tests alloc statements are inside first OpStmts
- Tests raw pointer uniqueness for MemRef deduplication
