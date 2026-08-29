# InitMemRef Pass

Materializes compiler-owned PTO level3 scratch, initializes MemRef for all variables, and creates alloc operations with unallocated addresses.

## Overview

This pass performs four tasks:

1. **Normalizes statement structure** (calls NormalizeStmtStructure internally)
2. **Materializes compiler-owned level3 scratch** for the A2/A3 `tile.ci` and narrowing `tile.cast` ABIs, plus required `tile.sort32` forms on A2/A3 and A5
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
2. **Materialize level3 scratch**: Under the PyPTO or DSA-RP planner, insert ordinary Vec `tile.create` values for missing compiler-owned scratch. `tile.ci` and narrowing `tile.cast` remain A2/A3-only, while required `tile.sort32` scratch is materialized on both A2/A3 and A5. Explicit caller tmp on `tile.sel` / `tile.sels` / `tile.prelu` is preserved. The PTOAS planner remains unchanged because its level-2 `PlanMemory` owns implicit scratch.
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

## Slice views follow the parent's layout

A `tile.slice` / `tensor.slice` result shares its parent's `base_` Ptr and adds a byte
offset plus a narrowed view size. Where the pass can describe the parent's real storage
layout exactly, both numbers are computed **in that layout** rather than in a fixed
row-major model:

| Parent layout | Offset of logical `(r, c)` | View span |
| ------------- | -------------------------- | --------- |
| `none_box` row-major dense (Vec / DDR / every `tensor.slice`) | `(r * cols + c) * elem_bytes` | Envelope from the first element to the last, in parent strides |
| Accumulator (L0C) — `blayout=col_major`, `slayout=row_major`, `fractal=1024` — **full-row-extent window only** | `c * rows * elem_bytes` | Exactly the `cols * rows * elem_bytes` its box columns occupy, contiguous |

L0C is not row-major dense: it is a grid of 16x16 boxes stored **column of boxes first**,
so box `(r_b, c_b)` of an `[M, N]` accumulator begins at `(c_b * M/16 + r_b) * 1024` bytes.
`tile_view_semantics::GetAccumulatorTileGeometry` recognises that dual from the parent's
effective `TileView` and hands back the two element strides — one logical row step is one
box width (16 elements), one logical column step is a whole physical column (`M` elements).

Why the distinction matters even though a `tile.slice` normally lowers to `pto.subview`
(which carries the *logical* window indices and re-derives the address itself): a view op
that does **not** lower to a subview — `tile.reshape` stacked on the slice — inherits this
byte offset and turns it into its own `pto.alloc_tile addr`. A row-major number there is
not merely misaligned; it addresses the wrong data.

### Why only a full-row-extent window

That inherited offset becomes a **standalone** address, and the `alloc_tile` around it
carries the *slice's* own `rows`. PTO derives the box-column stride from that `rows`, so
the NZ form describes the parent's boxes only when the window keeps the parent's full row
extent — then `rows` is unchanged and every box lands where the parent put it.

`GetSliceAccumulatorGeometry` (`include/pypto/ir/transforms/utils/memref_utils.h`) is the
single predicate both the offset and the span go through. It admits a window only when:

- the parent is a 2-D, statically shaped, whole-box NZ accumulator with a 4-byte element;
- the window keeps the parent's full row extent and starts at row 0;
- the column origin is a multiple of 16, **or** is a run-time value — PTO lowers the very
  same linear form at run time, so the two agree exactly, and neither side can check box
  alignment statically.

This predicate is what a batched accumulator is shaped around.
[`FlattenTileNdTo2D`](13-flatten_tile_nd_to_2d.md#batched-accumulators-pack-along-columns)
packs the `B` pages of a `tile.batch_matmul_acc` accumulator into one `[M, B*N]` Acc tile
with page `b` at `[0, b*N]` — a full-row-extent window with a box-aligned column origin,
i.e. exactly the admitted shape — and rejects the batch geometries that would not satisfy
it (`M % 16 != 0`, `N % 16 != 0`, a non-4-byte element) rather than emitting a window this
predicate would silently decline. So the list above is not just a description of what gets
the better offset: it is the contract that pass compiles against.

Consequence worth knowing: an admitted column window is contiguous in NZ, so its span is
exactly the bytes it occupies. Two disjoint column windows therefore read as disjoint to
lifetime/overlap analysis, where the row-major envelopes used to overlap and made
MemoryReuse reject two windows carried through one loop. That is also what lets the `B`
pages of one packed accumulator coexist: they are `B` disjoint column windows of a single
allocation.

### What keeps the row-major arithmetic, and what is still approximate

Everything the predicate declines falls back, bit-for-bit, to the arithmetic these slices
have always used. Two of those cases are genuinely row-major dense; the rest are known
gaps, listed here so the table above is not mistaken for a claim of correctness:

- **An Acc window that narrows the row extent** (or starts at a non-zero row). This is a
  known gap, not a supported case: such a window has *no* correct standalone base address
  at all, because its box columns are strided by the parent's `rows` while its own
  descriptor's are strided by its own. The pass therefore leaves the pre-existing
  row-major number in place rather than substituting a differently wrong one. The same
  applies to a slice chained under such a window: its ancestor is already unaddressable,
  and no leaf-local arithmetic can repair that.
- **An Acc window whose static column origin sits inside a 16-wide box.**
  `CanonicalizeTileSlice` explicitly whitelists exactly that window as a legal MAD
  destination (it lies entirely inside one box column), and on that path the slice lowers
  to `pto.subview` and this byte offset is dead — so it must not be rejected here.
- **Fractal-512 Mat / Left / Right tiles.** These are boxed too, *not* row-major dense:
  PTO gives them `rowStride = innerCols`, `colStride = rows`, the same shape of box
  structure as the accumulator. Their slice offsets carry the same latent mismatch and are
  deliberately left unchanged here; modelling them is separate work.

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
