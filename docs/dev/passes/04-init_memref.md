# InitMemRef Pass

Initializes MemRef for all variables and creates alloc operations with unallocated addresses.

## Overview

This pass performs three tasks:

1. **Normalizes statement structure** (calls NormalizeStmtStructure internally)
2. **Initializes MemRef** for TileType and TensorType variables with appropriate memory spaces
3. **Creates `block.alloc` operations** for each non-DDR MemRef with `addr=-1` (unallocated)

Memory space assignment rules:

- **Function parameters** → DDR
- **block.load/block.move** → Extract from `target_memory` kwarg (default Vec)
- **block.store** → DDR (shares MemRef with output tensor)
- **block.matmul/block.matmul_acc** → Acc
- **Other block operations** → Vec
- **Other variables** → DDR (default)

**Requires**: TypeChecked, SSAForm, SplitIncoreOrch, IncoreBlockOps.

**Produces**: HasMemRefs, NormalizedStmtStructure.

**Invalidates**: SSAForm (new MemRef variables are introduced).

**When to use**: Run after SSA conversion, outlining, and block-op conversion. Required before BasicMemoryReuse, InsertSync, and AllocateMemoryAddr.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
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

1. **Normalize structure**: Call `NormalizeStmtStructure` to ensure SeqStmts/OpStmts structure
2. **Analyze usage**: Traverse function body to determine memory space for each variable
3. **Initialize MemRef**: Create MemRef objects (addr=-1) and attach to variable types
4. **Collect non-DDR MemRefs**: Gather unique MemRef objects from TileType variables that are not in DDR
5. **Create alloc statements**: For each non-DDR MemRef, create `block.alloc(memspace, -1, size, id)`
6. **Insert into first OpStmts**: Prepend alloc statements to the first OpStmts in the function body

## Example

**Before** (after SSA/block-op conversion):

```python
def main(input_a: Tensor[[64, 64], FP32], output: Tensor[[64, 64], FP32]):
    tile_a: Tile[[64, 64], FP32] = block.load(input_a, [0, 0], [64, 64])
    tile_b: Tile[[64, 64], FP32] = block.add(tile_a, tile_a)
    result: Tensor[[64, 64], FP32] = block.store(tile_b, [0, 0], [64, 64], output)
    return result
```

**After**:

```python
def main(
    input_a: Tensor[[64, 64], FP32, MemRef(space=DDR, addr=-1, id=0)],
    output: Tensor[[64, 64], FP32, MemRef(space=DDR, addr=-1, id=1)],
):
    # SeqStmts [
    #   OpStmts [
    mem_vec_2: MemRefType = block.alloc(Vec, -1, 16384, 2)
    mem_vec_3: MemRefType = block.alloc(Vec, -1, 16384, 3)
    tile_a: Tile[[64, 64], FP32, memref=mem_vec_2] = block.load(input_a, [0, 0], [64, 64])
    tile_b: Tile[[64, 64], FP32, memref=mem_vec_3] = block.add(tile_a, tile_a)
    result: Tensor[[64, 64], FP32, memref=mem_ddr_1] = block.store(tile_b, [0, 0], [64, 64], output)
    #   ]
    #   ReturnStmt [result]
    # ]
```

Key observations:

- `addr=-1` indicates addresses are not yet assigned (done later by AllocateMemoryAddr)
- DDR MemRefs (params) do not get `block.alloc` statements
- `block.store` result shares MemRef with the output tensor parameter
- Alloc statements are placed at the beginning of the first OpStmts

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass InitMemRef();
```

**Implementation**: `src/ir/transforms/init_memref.cpp`

- `NormalizeStmtStructure` is called internally before MemRef initialization
- `MemRefUsageVisitor` analyzes memory space for each variable
- `InitMemRefMutator` creates MemRef objects and attaches to types
- `NonDDRMemRefCollector` collects unique non-DDR MemRefs
- `CreateAllocStatement` / `InsertAllocsIntoBody` create and insert alloc ops

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("init_mem_ref", &pass::InitMemRef, "Initialize MemRef for variables");
```

**Tests**: `tests/ut/ir/transforms/test_init_memref.py`

- Tests memory space assignment (Vec, Mat, Left, Right, Acc, DDR)
- Tests addr=-1 for all MemRefs
- Tests block.alloc statements are created for non-DDR MemRefs
- Tests normalized SeqStmts/OpStmts structure
- Tests block.store result shares MemRef with output param
