# InitMemRef Pass

Initializes MemRef for all variables and creates alloc operations with unallocated addresses.

## Overview

This pass performs three tasks:

1. **Normalizes statement structure** (calls NormalizeStmtStructure internally)
2. **Initializes MemRef** for TileType and TensorType variables with appropriate memory spaces
3. **Creates `tile.alloc` operations** for each non-DDR MemRef with `addr=-1` (unallocated)

Memory space assignment rules:

- **Function parameters** → DDR
- **tile.store return values** → DDR (special-cased, returns TensorType)
- **Other tile ops** → Resolved via OpRegistry memory specs (see `OpMemorySpaceSpec`)
- **Non-tile variables** → DDR (default)

**Requires**: TypeChecked, SSAForm, SplitIncoreOrch, IncoreTileOps.

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

1. **Normalize structure**: Call `NormalizeStmtStructure` to ensure flat `SeqStmts` structure
2. **Analyze usage**: Traverse function body to determine memory space for each variable
3. **Initialize MemRef**: Create MemRef objects (addr=-1) and attach to variable types
4. **Collect non-DDR MemRefs**: Gather unique MemRef objects from TileType variables that are not in DDR
5. **Create alloc statements**: For each non-DDR MemRef, create `tile.alloc(memspace, -1, size, id)`
6. **Prepend allocs**: Insert alloc statements at the beginning of the function body's top-level `SeqStmts`

## Example

**Before** (after SSA/block-op conversion):

```python
def main(input_a: Tensor[[64, 64], FP32], output: Tensor[[64, 64], FP32]):
    tile_a: Tile[[64, 64], FP32] = tile.load(input_a, [0, 0], [64, 64])
    tile_b: Tile[[64, 64], FP32] = tile.add(tile_a, tile_a)
    result: Tensor[[64, 64], FP32] = tile.store(tile_b, [0, 0], output)
    return result
```

**After**:

```python
def main(
    input_a: Tensor[[64, 64], FP32, MemRef(space=DDR, addr=-1, id=0)],
    output: Tensor[[64, 64], FP32, MemRef(space=DDR, addr=-1, id=1)],
):
    # SeqStmts [
    mem_vec_2: MemRefType = tile.alloc(Vec, -1, 16384, 2)
    mem_vec_3: MemRefType = tile.alloc(Vec, -1, 16384, 3)
    tile_a: Tile[[64, 64], FP32, memref=mem_vec_2] = tile.load(input_a, [0, 0], [64, 64])
    tile_b: Tile[[64, 64], FP32, memref=mem_vec_3] = tile.add(tile_a, tile_a)
    result: Tensor[[64, 64], FP32, memref=mem_ddr_1] = tile.store(tile_b, [0, 0], output)
    #   ReturnStmt [result]
    # ]
```

Key observations:

- `addr=-1` indicates addresses are not yet assigned (done later by AllocateMemoryAddr)
- DDR MemRefs (params) do not get `tile.alloc` statements
- `tile.store` result shares MemRef with the output tensor parameter
- Alloc statements are placed at the beginning of the function body's top-level `SeqStmts`

## ForStmt Loop-Carry Variables

ForStmt has four loop-carry related variables with specific MemRef sharing rules:

| Role | Description | MemRef Source |
| ---- | ----------- | ------------- |
| initValue | Initial value before first iteration | From producing operation |
| iter_arg | Loop body variable | Inherited from initValue |
| yield value | Produced at end of each iteration | From producing operation (independent) |
| return_var | Captures final yield value after loop | Inherited from yield value |

**Sharing groups**:

- Group A: initValue + iter_arg (same MemRef)
- Group B: yield value + return_var (same MemRef)

Group A and B may have different MemRefs. The yield-to-iter_arg mismatch is resolved later by BasicMemoryReuse (which inserts `tile.move` if needed).

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
- Tests tile.alloc statements are created for non-DDR MemRefs
- Tests normalized `SeqStmts` structure
- Tests tile.store result shares MemRef with output param
- Tests ForStmt loop-carry MemRef relationships (initValue/iter_arg sharing, yield/return_var sharing)
