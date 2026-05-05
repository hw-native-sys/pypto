# LegalizePTOBufferReuse Pass

Splits MemRefs left over by `MemoryReuse` whose shared writers cannot be expressed as PTO-compatible `alloc_tile` / view combinations.

## Overview

Generic `MemoryReuse` decides reuse on lifetime, memory space, dtype, shape, and (for 2D tiles) `valid_shape` differences. PTO codegen is stricter: multiple tile SSA values may lower to distinct `pto.alloc_tile` ops that alias the same MemRef address / byte offset, but sharing is only legal when the non-view writers have **identical `TileBufSignature`s** or any differences are materializable by existing PTO view ops. If a shared MemRef would require incompatible writer signatures that PTO cannot materialize via views, it must be split into distinct allocations before address assignment.

This pass detects illegal cross-type sharing and rebinds the offending writer (and its transitive view chain) onto a fresh MemRef.

**"Legal" cross-type sharing** is defined by `TileBufSignature::IsPTOMaterializable` (`include/pypto/codegen/pto/tile_buf_signature.h`) — differences that an existing PTO view op can materialise:

- `tile.reshape` — same `memory_space` / `dtype` / layout / fractal, equal element count
- `tile.extract`, `tile.slice`, `tensor.slice` — view-only consumers (no new storage)
- `tile.fillpad`, `tile.fillpad_inplace` — `pad`-only differences
- Load-with-padding / dynamic `valid_shape` — `valid_shape`-only differences
- `[1, N]` row-major ↔ `[N, 1]` col-major (zero-cost reshape between physically identical layouts)

All other differences are illegal and trigger a MemRef split.

**When to use**: between `MemoryReuse` and `AllocateMemoryAddr`. Idempotent on legal IR.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::LegalizePTOBufferReuse()` | `passes.legalize_pto_buffer_reuse()` | Function-level |

**Factory function**:

```cpp
Pass LegalizePTOBufferReuse();
```

**Python usage**:

```python
from pypto.pypto_core import passes

legalize_pass = passes.legalize_pto_buffer_reuse()
program_legal = legalize_pass(program)
```

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | `SplitIncoreOrch`, `IncoreTileOps`, `HasMemRefs`, `TileOps2D` |
| Produced | — |
| Invalidated | — |

The pass only rebinds variables and inserts new `tile.alloc` statements; it does not change control-flow shape, SSA form, or normalised structure.

## Algorithm

The transform runs in four phases over each `Function`:

1. **Collect (`MemRefUsageCollector`)** — visit every `AssignStmt` that defines a tile-typed variable. For each MemRef base pointer, record:
   - **Writers**: non-view producers (e.g. `tile.load`, `tile.add`, `tile.tpop_from_aic`) plus the `TileBufSignature` extracted from the LHS `TileType` (`TileBufSignature::FromTileType`); the input `Var`s of the RHS call are collected separately and used downstream (e.g. for the Ascend910B `load + tpop_from_aic` hazard detection), not as part of the signature.
   - **View users**: assignments whose RHS is a legal-view op and whose source argument lives in the same MemRef. The pass also records the source→user edge in `view_edges` so transitive views can be redirected later.
   - **`tile.tpop_from_aic` set**: tracked separately for the Ascend910B hazard described below.

2. **Plan (`PlanMemRefSplits`)** — for each MemRef with more than one writer:

   1. Take writer 0's signature as the reference.
   2. Group remaining writers by `IsPTOMaterializable` against an existing group representative; group 0 keeps the original MemRef, group `g ≥ 1` gets a fresh MemRef.
   3. **Force-split** any writer hitting the Ascend910B split-AIV `load + tpop_from_aic` hazard (see below) into its own group.
   4. For every non-zero group, allocate a fresh base `Var` named via `BuildBasePtrName(memory_space, next_id++)`, build a new `MemRef` with the maximum observed allocation size, and call `PropagateSplitToViewUsers` to walk `view_edges` and redirect every transitive view user onto the new MemRef.

   `next_id` is seeded by `MaxMemRefIdCollector`, which extracts the highest existing `mem_<space>_<n>` counter so generated names never collide.

3. **Mutate (`MemRefSplitMutator`)** — clone every affected `Var` / `IterArg` with a new `TileType` whose `MemRef` is the split target; all references to the old `Var` are remapped through `var_remap_` so SSA users follow the rebinding.

4. **Insert allocs (`InsertNewAllocStatements`)** — for each unique new base pointer, build a `tile.alloc` `AssignStmt` via `CreateAllocStatement(memref, memory_space)`. When the function body is already a non-empty `SeqStmts`, the pass prepends the new allocs to that sequence so they appear before any user of the new MemRef; otherwise the body is returned unchanged. In the `Default` pipeline this precondition holds because upstream passes establish the `NormalizedStmtStructure` property required by `MemoryReuse`.

### Ascend910B split-AIV `load + tpop_from_aic` hazard

On Ascend910B AIV functions whose `SplitMode` is non-`None`, sharing a MemRef between (a) the output of `tile.load` (or any of its view descendants) and (b) the input of an op that also consumes a `tile.tpop_from_aic` value produces a hardware hazard. `BackendHandler::RequiresSplitLoadTpopWorkaround()` returns true for this backend; when active, `CollectForcedSplitWriterIndices` flags every offending writer for force-split into its own group, regardless of signature compatibility.

This is the only place in the pass where backend dispatch leaks in, and it goes through `PassContext::Current()->GetBackendHandler()` per `.claude/rules/pass-context-config.md`.

## Examples

### Different shapes → split

Two writers sharing `mem_vec_0` with different physical shapes (`[128, 128]` vs `[64, 64]`) are not PTO-materializable from each other, so the second writer gets a fresh `mem_vec_1`.

**Before** (after `MemoryReuse`):

```python
# SeqStmts [
mem_vec_0: pl.Ptr = tile.alloc(Vec, 65536)
t1: Tile[[128, 128], FP32, memref=mem_vec_0, Mem.Vec] = tile.load(a, ...)
t2: Tile[[64, 64],  FP32, memref=mem_vec_0, Mem.Vec] = tile.load(a, ...)
result: Tensor[[128, 128], FP32] = tile.store(t2, [0, 0], b)
# ]
```

**After**:

```python
# SeqStmts [
mem_vec_0: pl.Ptr = tile.alloc(Vec, 65536)
mem_vec_1: pl.Ptr = tile.alloc(Vec, 65536)  # fresh MemRef for t2
t1: Tile[[128, 128], FP32, memref=mem_vec_0, Mem.Vec] = tile.load(a, ...)
t2: Tile[[64, 64],  FP32, memref=mem_vec_1, Mem.Vec] = tile.load(a, ...)
result: Tensor[[128, 128], FP32] = tile.store(t2, [0, 0], b)
# ]
```

See `tests/ut/ir/transforms/test_legalize_pto_buffer_reuse.py::TestIllegalSharingSplit::test_different_shape_same_memref_splits`.

### View chain follows the split

A `tile.fillpad` view of a split writer must rebind to the new MemRef so the view's storage matches its source.

**Before**:

```python
t1: Tile[[128, 128], FP32, memref=mem_vec_0, Mem.Vec] = tile.load(a, ...)
t2: Tile[[64, 64],  FP32, memref=mem_vec_0, Mem.Vec] = tile.load(a, ...)
t3: Tile[[64, 64],  FP32, memref=mem_vec_0, Mem.Vec, view(pad=max)]
   = tile.fillpad(t2, pad_value=max)
```

**After** (`t2` and its view `t3` both move to `mem_vec_1`):

```python
mem_vec_1: pl.Ptr = tile.alloc(Vec, 65536)
t1: Tile[[128, 128], FP32, memref=mem_vec_0, Mem.Vec] = tile.load(a, ...)
t2: Tile[[64, 64],  FP32, memref=mem_vec_1, Mem.Vec] = tile.load(a, ...)
t3: Tile[[64, 64],  FP32, memref=mem_vec_1, Mem.Vec, view(pad=max)]
   = tile.fillpad(t2, pad_value=max)
```

See `TestIllegalSharingSplit::test_split_propagates_through_view_chain`.

### Legal sharing preserved

Two writers with the **same** `TileBufSignature` (or differences materializable by `tile.fillpad` / `tile.reshape` / `valid_shape`) keep their shared MemRef untouched. See `TestLegalSharingPreserved`.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass LegalizePTOBufferReuse();
```

**Implementation**: `src/ir/transforms/legalize_pto_buffer_reuse_pass.cpp`

- `IsLegalViewOp` — string allowlist of view op names (`tile.reshape`, `tile.extract`, `tile.slice`, `tile.fillpad`, `tile.fillpad_inplace`, `tensor.slice`)
- `MemRefUsageCollector` — Phase 1: per-MemRef writer / view-user / `tile.tpop_from_aic` index
- `CollectLoadFamilyVars` / `CollectForcedSplitWriterIndices` — Ascend910B split-AIV hazard detection
- `PlanMemRefSplits` — Phase 2: signature grouping and fresh MemRef allocation
- `PropagateSplitToViewUsers` — Phase 2 helper: BFS over `view_edges` to redirect transitive views
- `MemRefSplitMutator` — Phase 3: rewrite `Var` / `IterArg` types with the new MemRef
- `InsertNewAllocStatements` — Phase 4: prepend `tile.alloc` for each fresh MemRef
- `MaxMemRefIdCollector` — seeds fresh-id counter from existing names

**Backend dispatch**: `BackendHandler::RequiresSplitLoadTpopWorkaround()` accessed via `PassContext::Current()->GetBackendHandler()` (per `.claude/rules/pass-context-config.md`).

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("legalize_pto_buffer_reuse", &pass::LegalizePTOBufferReuse,
           "Create a PTO buffer reuse legalisation pass\n\n"
           "After generic MemoryReuse, detects illegal cross-type MemRef sharing\n"
           "that PTO codegen cannot express and splits such MemRefs.");
```

**Type stub**: `python/pypto/pypto_core/passes.pyi`

```python
def legalize_pto_buffer_reuse() -> Pass:
    """Create a PTO buffer reuse legalisation pass."""
```

**Tests**: `tests/ut/ir/transforms/test_legalize_pto_buffer_reuse.py`

- `TestLegalSharingPreserved` — same-signature and `tile.fillpad`-view sharing kept intact
- `TestAscend910BSplitLoadTpopHazard` — split-AIV hazard force-split on 910B; no force-split on Ascend950
- `TestIllegalSharingSplit` — different-shape split and view-chain propagation
- `TestLegalizeWithCodegen` — end-to-end alloc count / address checks via PTO codegen
