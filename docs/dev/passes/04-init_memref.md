# InitMemRef Pass

Initializes MemRef for all variables in functions based on their usage.

## Overview

This pass analyzes variable usage and initializes MemRef for TileType and TensorType variables with appropriate memory spaces:

- **TileType variables**: Memory space = UB (Unified Buffer) by default
- **TensorType variables**: Memory space = DDR by default
- **Special cases**: block.load/block.store operands get DDR memory space

**When to use**: Run this pass after SSA conversion and before memory optimization passes. Required before BasicMemoryReuse, InsertSync, and AddAlloc.

## API

| C++ | Python | Level |
|-----|--------|-------|
| `pass::InitMemRef()` | `passes.init_mem_ref()` | Function-level |

**Factory function**:
```cpp
Pass InitMemRef();
```

**Python usage**:
```python
from pypto.pypto_core import passes

init_pass = passes.init_mem_ref()
program_with_memrefs = init_pass(program)
```

## Algorithm

1. **Traverse Variables**: Iterate through all variables in function
2. **Check Type**: Determine if variable is TileType or TensorType
3. **Determine Memory Space**:
   - TileType → UB (Unified Buffer)
   - TensorType used in block.load/block.store → DDR
   - Other TensorType → DDR
4. **Create MemRef**: Allocate MemRef with appropriate memory space
5. **Attach to Type**: Update variable's type to include MemRef

**Memory space rules**:
```
TileType → MemRef(space=UB)
TensorType (block.load/store operand) → MemRef(space=DDR)
TensorType (other) → MemRef(space=DDR)
```

## Example

### TileType Variables

**Before**:
```python
tile_a: Tile[[64, 64], FP32] = block.load(tensor_a, [0, 0], [64, 64])
# tile_a has no MemRef
```

**After**:
```python
tile_a: Tile[[64, 64], FP32, MemRef(space=UB)] = block.load(tensor_a, [0, 0], [64, 64])
# tile_a has MemRef with UB space
```

### TensorType Variables

**Before**:
```python
def compute(input: Tensor[[128, 128], FP32]) -> Tensor[[128, 128], FP32]:
    # input has no MemRef
    tile = block.load(input, [0, 0], [64, 64])
    ...
```

**After**:
```python
def compute(input: Tensor[[128, 128], FP32, MemRef(space=DDR)]) -> Tensor[[128, 128], FP32]:
    # input has MemRef with DDR space (used in block.load)
    tile = block.load(input, [0, 0], [64, 64])
    ...
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`
```cpp
Pass InitMemRef();
```

**Implementation**: `src/ir/transforms/init_memref.cpp`
- Uses IRVisitor to analyze usage patterns
- Creates MemRef objects with memory spaces
- Updates variable types with MemRef

**Python binding**: `python/bindings/modules/passes.cpp`
```cpp
passes.def("init_mem_ref", &pass::InitMemRef, "Initialize MemRef for variables");
```

**Tests**: `tests/ut/ir/transforms/test_init_memref.py`
- Tests TileType variables get UB
- Tests TensorType variables get DDR
- Tests block.load/store operands
- Tests function parameters and returns
