# InitMemRef Pass

Materializes compiler-owned PTO level3 scratch, initializes MemRef for all variables, and creates alloc operations with unallocated addresses.

## Overview

This pass performs four tasks:

1. **Normalizes statement structure** (calls NormalizeStmtStructure internally)
2. **Materializes compiler-owned level3 scratch** for the A2/A3 `tile.ci`, narrowing `tile.cast`, and required `tile.sort32` forms that need an explicit tmp under level3
3. **Initializes MemRef** for TileType and TensorType variables
4. **Creates `tile.alloc` operations** for each non-DDR MemRef with `addr=-1` (unallocated)

Memory space is read from `TileType::memory_space_` (set by InferTileMemorySpace). Variables without `memory_space` default to DDR.

**Requires**: SSAForm, SplitIncoreOrch, IncoreTileOps, TileOps2D, TileMemoryInferred.

**Produces**: HasMemRefs, NormalizedStmtStructure.

**Invalidates**: SSAForm (new MemRef variables are introduced).

**When to use**: Run after SSA conversion, outlining, and block-op conversion. Required before MemoryReuse and AllocateMemoryAddr.

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
2. **Materialize level3 scratch**: Under the PyPTO or DSA-RP planner on A2/A3, insert ordinary Vec `tile.create` values for missing compiler-owned `tile.ci`, narrowing `tile.cast`, and required `tile.sort32` scratch. Explicit caller tmp on `tile.sel` / `tile.sels` / `tile.prelu` is preserved. The PTOAS planner and A5 are unchanged.
3. **Resolve declared allocations**: Collect every one-argument `pl.MemRef(...)` declaration and derive each one's size and memory space from the tiles bound to it (see [Declared allocations](#declared-allocations))
4. **Initialize MemRef**: Read `memory_space` from `TileType` (set by InferTileMemorySpace), create MemRef objects (addr=-1) and attach to variable types
   - **tile.store**: result shares MemRef with the output tensor argument (specified by `output_reuses_input_arg` registry attribute)
   - **View ops** (e.g. `tile.reshape`): output shares MemRef with the input tile
   - **Reuse-input ops** (e.g. `tile.matmul_acc`, `tile.gemv_acc`): output shares MemRef with the specified input (via `output_reuses_input_arg` registry attribute)
   - **ForStmt/IfStmt return_vars**: patched to share MemRef with corresponding yield values
   - **Declared allocations**: the tile keeps the allocation the author declared instead of getting a fresh one
5. **Collect non-DDR MemRefs**: Gather unique MemRef objects from TileType variables that are not in DDR
6. **Create alloc statements**: For each non-DDR MemRef, create `tile.alloc(memspace, -1, size, id)` — with `pinned=True` when the base was declared by the author
7. **Prepend allocs**: Insert alloc statements at the beginning of the function body's top-level `SeqStmts`

## Declared allocations

`pl.Tile[[...], dtype, <alloc>, pl.Mem.Vec]` binds a tile to an allocation the kernel
author declared, where `<alloc>` is a `pl.MemRef("name")` referenced by variable (or the
same one-argument form inline, which is what the printer emits). Tiles referencing the
same allocation share it; `MemoryReuse` never packs anything else into it. This is manual reuse control — see
[MemoryReuse](34-memory_reuse.md#declared-allocations) for why an author would want it.

**How the declaration reaches this pass.** The parser resolves a one-argument
`pl.MemRef` to a `MemRef` whose `base_` Ptr is interned by name (so two annotations
naming one allocation share one base), with `byte_offset = 0`, no size, and
**`is_pinned_` set**. That flag is what identifies a declaration — re-parsing a
post-allocation dump also puts MemRefs on `TileType`s, and those are the compiler's.
Recording it explicitly rather than inferring it (from a sentinel size, or from where in
the pipeline the pass sits) keeps the classification a property of the data, and lets
the printer emit the one-argument form so a dump round-trips without inventing a size or
address.

This pass **consumes** the declaration: the MemRef it produces is an ordinary one
carrying the derived size, with the flag cleared. From there on the allocation's
`pinned=True` kwarg is what marks it as the author's.

Slot geometry is the exception — `slot_count_` and `slot_index_` survive on the resolved
MemRef. Resolving the index into `byte_offset_` answers *where* the slot lands; it does
not stop the MemRef from being slot *k* of an *N*-slot allocation, and PTO codegen reads
exactly that to emit one ptoas `pto.alloc_multi_tile` region with a `pto.multi_tile_get`
per use, instead of *N* unrelated allocs. The two are related, not independent:
`byte_offset_` is derived from the index, and `AllocateMemoryAddr` may rebase it onto a
physical address, so the index is the author's *selection* and the offset is its resolved
*location*. Both print, so a dump round-trips as
`pl.MemRef(base, offset, size, slots=N)[k]`.

Under `memory_planner=PTOAS` a **single-slot** declaration is rejected: its isolation is
enforced by `MemoryReuse`, which ptoas replaces wholesale, and no ptoas concept carries it
instead. A multi-slot declaration is accepted — its slots become a ptoas region whose
segments ptoas is forbidden to merge — see
[Python syntax](../language/00-python_syntax.md#under-the-ptoas-memory-planner).

The declaration lives on the assigned `Var`, not on the RHS `Call` — `ConvertToSSA` merges
it into the Var's type and op type deduction never produces a MemRef — so any pass that
rebuilds a type from the Call must carry it over explicitly. `ConvertToSSA` does the
merge; `FlattenTileNdTo2D` carries it through all four of its rebuilds (ND flatten,
≤2D `tile.load`, rank>2 `tile.create`/`tile.full`, generic tile op); `InferTileMemorySpace` keeps it when syncing the Var
type to a rebuilt Call. Passes that clone rather than rebuild (including the per-stage
bodies `LowerPipelineLoops` emits) preserve it through `MemRef`'s clone path.

**What the pass derives.** The author writes neither a size nor an address:

| Property | Derived from |
| -------- | ------------ |
| Size | The largest tile bound to the allocation |
| Memory space | The space the bound tiles share (they must agree) |
| Address | Left to `AllocateMemoryAddr`, exactly as for compiler allocations |

**Rejected bindings** (all `pypto::ValueError`, all with the offending tile's span):

- A bound tile with a dynamic shape — a declared allocation must be sized at compile time.
- Tiles on one allocation disagreeing on memory space.
- Binding the output of a view / in-place op (`tile.reshape`, `tile.matmul_acc`, …).
  Such a result lands in its source's allocation, so it cannot be placed elsewhere;
  bind the source instead.

A fourth rule — tiles bound to **one slot** must not be live at the same time — needs
lifetime information and is therefore checked in
[MemoryReuse](34-memory_reuse.md#declared-allocations). Tiles on *different* slots are
meant to be live together; that is what a multi-slot declaration is for.

```python
ping, pong = pl.MemRef(), pl.MemRef()

t0: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.load(a, [0, 0], [64, 64])
t1: pl.Tile[[64, 64], pl.FP32, pong, pl.Mem.Vec] = pl.exp(t0)
t2: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.exp(t1)  # shares t0's allocation
```

becomes two pinned allocations, with `t0` and `t2` on `ping`:

```python
ping: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384, pinned=True)
pong: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384, pinned=True)
```

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
- `tile.store` result shares MemRef with the output tensor parameter (via `output_reuses_input_arg` registry attribute)
- Reuse-input ops (`tile.store`, `matmul_acc`, `gemv_acc`) share MemRef with their designated input, preventing redundant allocs
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
- `MaterializePtoLevel3ScratchMutator` inserts only missing compiler-owned optional scratch (`tile.ci`, narrowing `tile.cast`, required `tile.sort32`) after all tile shape/cast rewrites and before MemRef collection; explicit caller tmp on `tile.sel` / `tile.sels` / `tile.prelu` is preserved
- `InitMemRefMutator` reads `memory_space` from `TileType` and creates MemRef objects
  - Handles MemRef sharing for view ops, reuse-input ops (`tile.store`, `matmul_acc`, `gemv_acc`), tile aliases (`a = b`), and ForStmt/IfStmt yield values
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
- Tests planner/target-gated `tile.ci`, narrowing `tile.cast`, and `tile.sort32` scratch materialization
- Tests tile.store result shares MemRef with output param
- Tests accumulate op (matmul_acc) MemRef sharing with accumulator input
- Tests ForStmt loop-carry MemRef relationships (initValue/iter_arg sharing, yield/return_var sharing)
