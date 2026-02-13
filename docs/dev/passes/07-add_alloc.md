# AddAlloc Pass

Creates alloc operations for MemRefs and assigns memory addresses.

## Overview

This pass traverses all TileType variables in each function, collects unique MemRef objects, and creates alloc operations for each. The alloc operations are prepended to the function body.

**Key responsibilities**:
- Identify all TileType variables requiring allocation
- Collect unique MemRef objects (accounting for memory reuse)
- Create alloc operations for each unique MemRef
- Assign base addresses to MemRefs
- Insert alloc operations at function beginning

**When to use**: Run after BasicMemoryReuse (to respect shared MemRefs) and before code generation. Final pass in memory management pipeline.

## API

| C++ | Python | Level |
|-----|--------|-------|
| `pass::AddAlloc()` | `passes.add_alloc()` | Function-level |

**Factory function**:
```cpp
Pass AddAlloc();
```

**Python usage**:
```python
from pypto.pypto_core import passes

alloc_pass = passes.add_alloc()
program_with_allocs = alloc_pass(program)
```

## Algorithm

1. **Collect TileType Variables**: Traverse function body to find all TileType variables
2. **Extract MemRefs**: Get MemRef from each TileType variable's type
3. **Deduplicate**: Collect unique MemRefs (multiple variables may share same MemRef due to BasicMemoryReuse)
4. **Calculate Sizes**: Compute buffer size for each MemRef based on shape and dtype
5. **Assign Addresses**: Assign base addresses (offset within memory space)
6. **Create Alloc Ops**: Create alloc operation for each unique MemRef
7. **Prepend to Body**: Insert alloc operations at beginning of function body

**MemRef tracking**:
- Each alloc binds to a specific MemRef pointer
- MemRef contains: memory space (UB/DDR), size, base address
- Multiple variables with same MemRef → single alloc

## Example

### Single MemRef

**Before**:
```python
def compute(...):
    tile_a: Tile[[64, 64], FP32, MemRef(id=0, space=UB)] = block.load(...)
    tile_b: Tile[[64, 64], FP32, MemRef(id=0, space=UB)] = block.add(tile_a, ...)
    # No alloc operation
```

**After**:
```python
def compute(...):
    alloc(memref=MemRef(id=0, space=UB, size=16384, addr=0))  # 64*64*4 bytes
    tile_a: Tile[[64, 64], FP32, MemRef(id=0, space=UB, addr=0)] = block.load(...)
    tile_b: Tile[[64, 64], FP32, MemRef(id=0, space=UB, addr=0)] = block.add(tile_a, ...)
```

### Multiple MemRefs (Memory Reuse)

**Before** (after BasicMemoryReuse):
```python
def compute(...):
    tile_a: Tile[[64, 64], FP32, MemRef(id=0, space=UB)] = block.load(...)
    tile_b: Tile[[64, 64], FP32, MemRef(id=1, space=UB)] = block.add(tile_a, ...)
    # tile_a last use

    tile_c: Tile[[64, 64], FP32, MemRef(id=0, space=UB)] = block.load(...)
    # Reuses MemRef(id=0)
```

**After**:
```python
def compute(...):
    alloc(memref=MemRef(id=0, space=UB, size=16384, addr=0))
    alloc(memref=MemRef(id=1, space=UB, size=16384, addr=16384))
    # Only 2 allocs for 3 variables (tile_a and tile_c share MemRef 0)

    tile_a: Tile[[64, 64], FP32, MemRef(id=0, space=UB, addr=0)] = block.load(...)
    tile_b: Tile[[64, 64], FP32, MemRef(id=1, space=UB, addr=16384)] = block.add(tile_a, ...)
    tile_c: Tile[[64, 64], FP32, MemRef(id=0, space=UB, addr=0)] = block.load(...)
```

### Different Memory Spaces

**Before**:
```python
def compute(...):
    tensor: Tensor[[128, 128], FP32, MemRef(id=0, space=DDR)] = ...
    tile: Tile[[64, 64], FP32, MemRef(id=1, space=UB)] = block.load(tensor, ...)
```

**After**:
```python
def compute(...):
    alloc(memref=MemRef(id=0, space=DDR, size=65536, addr=0))  # DDR allocation
    alloc(memref=MemRef(id=1, space=UB, size=16384, addr=0))   # UB allocation
    # Addresses are per-memory-space (both can have addr=0)

    tensor: Tensor[[128, 128], FP32, MemRef(id=0, space=DDR, addr=0)] = ...
    tile: Tile[[64, 64], FP32, MemRef(id=1, space=UB, addr=0)] = block.load(tensor, ...)
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`
```cpp
Pass AddAlloc();
```

**Implementation**: `src/ir/transforms/add_alloc.cpp`
- Uses IRVisitor to collect TileType variables
- Tracks unique MemRefs using pointer comparison
- Calculates sizes from TileType shapes and dtypes
- Assigns addresses sequentially within each memory space
- Creates alloc operations and prepends to function body

**Python binding**: `python/bindings/modules/passes.cpp`
```cpp
passes.def("add_alloc", &pass::AddAlloc, "Add allocation operations");
```

**Tests**: `tests/ut/ir/transforms/test_add_alloc.py`
- Tests single MemRef allocation
- Tests multiple MemRef allocations
- Tests memory reuse (shared MemRefs)
- Tests address assignment
- Tests different memory spaces
