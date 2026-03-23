# InitMemRef Pass

Initializes MemRef for all variables and creates alloc operations with unallocated addresses.

## Overview

This pass performs three tasks:

1. **Normalizes statement structure** (calls NormalizeStmtStructure internally)
2. **Initializes MemRef** for TileType and TensorType variables
3. **Creates `tile.alloc` operations** for each non-DDR MemRef with `addr=-1` (unallocated)

Memory space is read from `TileType::memory_space_` (set by InferTileMemorySpace). Variables without `memory_space` default to DDR.

**Requires**: SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred.

**Produces**: HasMemRefs, NormalizedStmtStructure.

**Invalidates**: SSAForm (new MemRef variables are introduced).

**When to use**: Run after SSA conversion, outlining, and block-op conversion. Required before MemoryReuse, InsertSync, and AllocateMemoryAddr.

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
2. **Initialize MemRef**: Read `memory_space` from `TileType` (set by InferTileMemorySpace), create MemRef objects (addr=-1) and attach to variable types
   - **tile.store**: result shares MemRef with the output tensor argument
   - **View ops** (e.g. `tile.reshape`): output shares MemRef with the input tile
   - **Accumulate ops** (e.g. `tile.matmul_acc`, `tile.gemv_acc`): output shares MemRef with the accumulator input (specified by `output_reuses_input_arg` registry attribute)
   - **ForStmt/IfStmt return_vars**: patched to share MemRef with corresponding yield values
3. **Collect non-DDR MemRefs**: Gather unique MemRef objects from TileType variables that are not in DDR
4. **Create alloc statements**: For each non-DDR MemRef, create `tile.alloc(memspace, -1, size, id)`
5. **Prepend allocs**: Insert alloc statements at the beginning of the function body's top-level `SeqStmts`

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
- Accumulate ops (`matmul_acc`, `gemv_acc`) share MemRef with their accumulator input, preventing redundant Acc allocs
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

Group A and B may have different MemRefs. The yield-to-iter_arg mismatch is resolved later by MemoryReuse (which inserts `tile.move` if needed).

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass InitMemRef();
```

**Implementation**: `src/ir/transforms/init_memref.cpp`

- `NormalizeStmtStructure` is called internally before MemRef initialization
- `InitMemRefMutator` reads `memory_space` from `TileType` and creates MemRef objects
  - Handles MemRef sharing for `tile.store`, view ops, accumulate ops, and ForStmt/IfStmt yield values
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
- Tests accumulate op (matmul_acc) MemRef sharing with accumulator input
- Tests ForStmt loop-carry MemRef relationships (initValue/iter_arg sharing, yield/return_var sharing)
