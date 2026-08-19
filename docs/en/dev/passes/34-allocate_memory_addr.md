# AllocateMemoryAddr Pass

Assigns real memory addresses to MemRefs backed by existing allocations.

## Overview

This pass allocates concrete memory addresses for non-DDR MemRefs. The
`tile.alloc` statements declare allocation roots and sizes, so their pointer
results and Call arguments remain unchanged; addressed MemRefs live on tile and
tensor types. The pass also resolves
`system.reserve_buffer(base=AUTO)` before PTO code generation. Address
placement is selected by `MemoryPlanner`:

- `PYPTO` uses the deterministic sequential allocator after `MemoryReuse`.
- `DSA_RP` constructs and solves capacity-constrained dynamic storage
  allocation with reuse penalties in process.
- `PTOAS` skips this pass and leaves address assignment to ptoas.

**Key responsibilities**:

- Collect unique MemRef objects from TileType variables
- Resolve `system.reserve_buffer` bases to explicit addresses per function
- Allocate aligned addresses within each independent memory space
- Under `DSA_RP`, preserve correctness constraints and minimize recognized
  costly reuse among capacity-fitting placements
- Update MemRef addresses in all variable types
- Preserve `tile.alloc` pointer declarations while updating every MemRef use

**When to use**: Final memory-management pass before code generation. Under
`PYPTO` it runs after `MemoryReuse`; under `DSA_RP`, `MemoryReuse` is skipped
and this pass consumes the allocation identities from
`MaterializeSemanticAliases`.

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

Select the opt-in DSA-RP mode at compilation:

```python
from pypto.ir import compile
from pypto.pypto_core import passes

compile(program, memory_planner=passes.MemoryPlanner.DSA_RP)
```

## Algorithm

1. **Collect MemRefs**: Traverse the function and collect unique allocation
   identities and their views.
2. **Resolve reserve buffers**: Assign explicit bases to AUTO reservations and
   exclude their ranges from ordinary placement.
3. **Select placement**:
   - `PYPTO`: sort MemRefs by name and allocate sequential aligned addresses.
   - `DSA_RP`: build the in-memory problem described below and run canonical
     greedy.
4. **Update in place**: Use `MemRefUpdateMutator` to:
   - Replace old MemRef references in variable types (TileType/TensorType) with new MemRefs containing real addresses
   - Rewrite `system.reserve_buffer` kwargs with the resolved explicit `base`

### DSA-RP policy

Each on-chip memory space is an independent fixed-capacity arena. One
post-alias allocation identity becomes one buffer with byte size, alignment,
and a conservative half-open lifetime. The problem has:

- **hard constraints** for lifetime interference, reserved ranges, semantic
  no-alias rules, target hazards, and requested pipeline-stage separation.
  Author-declared `pl.MemRef` allocations
  are also hard-separated from every other allocation in their memory space.
  A multi-slot declaration is placed as one buffer covering its full declared
  extent, while each member retains its constant or runtime-selected slot offset;
- **soft unit-weight pairs** for lifetime-compatible physical reuse that the
  built-in recognizer identifies as a cross-pipe WAR or WAW handoff; and
- a hard arena-capacity bound. Capacity and correctness are never traded for a
  lower reuse cost.

Recognition is conservative. It requires complete access information,
full-allocation handoff endpoints, and a verified initial write. Same-pipe,
partial-view, or uncertain cases receive no penalty. The active backend maps
each supported executable call to a hardware pipe from its operation, resolved
source/destination memory spaces, and the selected SoC's direct memory graph;
an op-specific backend hook handles routes that are not uniquely inferable.
Unsupported or ambiguous routes are skipped. The recognizer consumes that backend metadata
and does not duplicate an architecture route table, invoke ptoas, or simulate
its synchronization pass.

The explicit pair model is output-sensitive. With `B` reusable buffers, a
kernel can contain `Theta(B^2)` lifetime conflicts or candidate penalty pairs,
so recognition and solver graph construction are quadratic in the worst case.
This documented complexity exception is confined to the opt-in `DSA_RP`
planner; the default planner is unchanged.

Canonical greedy tries offset zero, reserved-range ends, and aligned tops of
already placed hard or soft neighbors. For each buffer it chooses the candidate
with the lowest incremental penalty, then the lowest address. It evaluates
several deterministic orders and retains a feasible penalty-blind first-fit
placement as an incumbent. Every selected placement is checked by an
independent validator before writeback.

Pipeline intent uses a strict-then-soft policy:

1. Run the bounded canonical-greedy search with every requested cross-stage
   pipeline separation hard.
2. If that search finds no fitting placement—this is not a proof that the
   strict mathematical problem is infeasible—relax only pairs whose sole hard
   reason is pipeline intent, add unit reuse penalties for them, and search
   again.
3. If the selected placement overlaps a relaxed pair, emit the
   `PH-DSA-001` performance diagnostic. All semantic and target-hazard
   separations remain hard. If the relaxed bounded search also finds no fit,
   report a compile-time OOM/no-fit error; this remains a search failure, not
   an infeasibility certificate.

> **Toolchain requirement:** `DSA_RP` relies on ptoas InsertSync recognizing
> physical range overlap across distinct allocation roots. Use a modern ptoas
> containing the tile-native memory planner and its cross-root local-overlap
> analysis (PTOAS PR #913 and follow-up fixes). Older releases that compare
> allocation-root identity without comparing planned physical ranges are not
> compatible with DSA-RP placements.

The model, recognizer, solver, validation, and writeback are all in-process.
`DSA_RP` exposes no problem export, placement replay, reference-placement, or
profiling interface.

### Sequential `PYPTO` policy

- Each memory space has its own address space starting from 0 unless `system.reserve_buffer` already reserved a leading window in that space
- Addresses are 32-byte aligned: `next_addr = align32(current_addr + size)`
- MemRefs are sorted by name for deterministic allocation order
- DDR MemRefs are skipped (addresses managed externally)

**View MemRefs (slices) share one slot**:

MemRefs that share the same `base_` Ptr (a root allocation plus its `tile.slice` views) are co-located in a single slot sized by the largest member, since every view physically aliases its parent. Each member keeps its own relative offset within the slot: `new_addr = slot_base + member.byte_offset` (the relative offset InitMemRef computed). The root sits at `slot_base`; a view at row `k` sits at `slot_base + k * row_stride`. This matters for chains where a view's offset is not re-derived at codegen — e.g. a `tile.reshape` of a `tile.slice` does not emit `pto.subview`, so its `pto.alloc_tile addr` is read directly from this MemRef offset.

Backends can override these defaults by supplying a custom `MemoryAllocatorPolicy` via `Backend::CreateMemoryAllocatorPolicy()`. See [Allocation Policy](#allocation-policy) below.

## Example

### Before (after the selected reuse analysis)

```python
# SeqStmts [
mem_vec_0: Ptr = tile.alloc(Vec, 16384)
mem_vec_1: Ptr = tile.alloc(Vec, 16384)
tile_a: Tile[[64, 64], FP32, MemRef(mem_vec_0, -1, 16384)] = tile.load(...)
tile_b: Tile[[64, 64], FP32, MemRef(mem_vec_1, -1, 16384)] = tile.add(tile_a, ...)
# ]
```

### After (addresses assigned)

```python
# SeqStmts [
mem_vec_0: Ptr = tile.alloc(Vec, 16384)  # unchanged declaration
mem_vec_1: Ptr = tile.alloc(Vec, 16384)  # unchanged declaration
tile_a: Tile[[64, 64], FP32, MemRef(mem_vec_0, 0, 16384)] = tile.load(...)
tile_b: Tile[[64, 64], FP32, MemRef(mem_vec_1, 16384, 16384)] = tile.add(tile_a, ...)
# ]
```

### Multiple Memory Spaces

```python
# Before:
mem_vec_0: Ptr = tile.alloc(Vec, 2048)
mem_left_1: Ptr = tile.alloc(Left, 2048)
mem_right_2: Ptr = tile.alloc(Right, 2048)
mem_acc_3: Ptr = tile.alloc(Acc, 2048)

# After (each space starts from addr=0):
tile_vec: Tile[..., MemRef(mem_vec_0, 0, 2048)] = ...
tile_left: Tile[..., MemRef(mem_left_1, 0, 2048)] = ...
tile_right: Tile[..., MemRef(mem_right_2, 0, 2048)] = ...
tile_acc: Tile[..., MemRef(mem_acc_3, 0, 2048)] = ...
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass AllocateMemoryAddr();
```

**Implementation**: `src/ir/transforms/allocate_memory_addr_pass.cpp`

- `MemRefCollectorVisitor` collects unique MemRefs from TileType variables
- `AllocateMemoryAddresses` assigns sequential aligned addresses per memory space using a `MemoryAllocatorPolicy`
- `dsa_adapter::BuildDsaAllocationPlan` derives conservative lifetimes and
  mandatory separations in `src/ir/transforms/dsa/allocation_plan.cpp`
- `dsa_adapter::BuildProblem` derives the narrow in-memory DSA-RP model
- `dsa::CanonicalGreedySolver` searches capacity-fitting placements and
  `dsa::ValidateSolution` independently checks the result
- `MemRefUpdateMutator` updates MemRefs in variable and expression types and
  rewrites resolved `system.reserve_buffer` bases in one traversal; `tile.alloc`
  remains a pointer-and-size declaration

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("allocate_memory_addr", &pass::AllocateMemoryAddr,
           "Allocates real memory addresses for existing alloc operations.");
```

**Tests**:
`tests/ut/ir/transforms/test_allocate_memory_addr_pass.py`,
`tests/ut/ir/transforms/test_dsa_reuse_penalty_recognizer.py`, and
`tests/ut/cpp/dsa_reuse_penalty_solver_test.cpp`

- Tests address allocation with 32-byte alignment
- Tests multiple MemRef allocations
- Tests empty function (no tiles)
- Tests alloc statements are prepended to the function body's top-level `SeqStmts`
- Tests raw pointer uniqueness for MemRef deduplication
- Tests default policy behavior without a backend configured
- Tests the capacity diagnostic attributes reserved cross-core pipe bytes (see below)
- Tests DSA-RP geometry, capacity, hard constraints, penalty activation,
  deterministic canonical-greedy placement, and independent validation
- Tests exact pre-solver recognized-edge sets as well as their final placement geometry
- Characterizes that canonical-greedy `kNoFit` is a bounded-search result, not an infeasibility proof

## Allocation Policy

The pass delegates placement decisions to a `MemoryAllocatorPolicy` interface (`include/pypto/ir/memory_allocator_policy.h`), making the allocation strategy extensible without modifying the pass itself.

### Interface

```cpp
class MemoryAllocatorPolicy {
 public:
  virtual ~MemoryAllocatorPolicy() = default;
  virtual bool ShouldAllocate(MemorySpace space) const = 0;
  virtual uint64_t AlignAddress(uint64_t addr, MemorySpace space) const = 0;
  virtual void OrderMemRefs(std::vector<MemRefPtr>& refs) const = 0;
};
```

| Method | Purpose | Default behavior |
| ------ | ------- | ---------------- |
| `ShouldAllocate` | Filter which memory spaces receive addresses | Skip DDR; allocate all on-chip spaces |
| `AlignAddress` | Align a raw address for a given space | 32-byte alignment |
| `OrderMemRefs` | Sort MemRefs within a space before allocation | Ascending by `MemRef::name_hint_` |

### Default policy

`DefaultMemoryAllocatorPolicy` preserves the original hard-coded behavior (skip DDR, 32-byte alignment, sort by name).

### Backend override

When a backend is configured (`BackendConfig::IsConfigured()`), the pass calls `Backend::CreateMemoryAllocatorPolicy()` to obtain the policy. The default `Backend` implementation returns `DefaultMemoryAllocatorPolicy`. Custom backends can override this virtual method to provide different alignment rules, ordering, or space filtering:

```cpp
class MyBackend : public Backend {
 public:
  MemoryAllocatorPolicyPtr CreateMemoryAllocatorPolicy() const override {
    return std::make_unique<MyCustomPolicy>();
  }
};
```

When no backend is configured (e.g., in unit tests), the pass falls back to `DefaultMemoryAllocatorPolicy` automatically.

## Capacity verification

Capacity is checked in two places: `AllocateMemoryAddresses`' own in-pass `CHECK`, which owns the only exact footprint (it counts a declared allocation's unbound slots and does not need every tile address to be constant), and the `AllocatedMemoryAddr` property verifier, which tracks the high-water mark (`addr + size`) per memory space. Both compare against `Backend::GetMemSize(space)`, and both emit the note below — which of them a compile hits depends on configuration, so the wording is shared (`ReservedBytesNote`) rather than living in one of them.

Because `system.reserve_buffer` reserves a leading window that every tile is then allocated *above*, its bytes are counted in the high-water mark but are **not** a MemRef — they are invisible in the per-tile accounting an author can inspect. When the overflowing space is the one that pays for a reserve buffer, the diagnostic therefore names that window explicitly. It is stated as the allocation **floor** (`reserved_end_by_space`, the aligned max-END tiles are placed above) rather than as "bytes the buffers occupy": an explicitly based buffer or an alignment gap makes the floor exceed the summed buffer sizes, and the floor is what the overflow was charged.

```text
Function 'qk_pv_aic': Mat buffer usage (1064960 bytes) exceeds platform limit (524288 bytes).
The first 1048576 bytes of that space are reserved by system.reserve_buffer, so tiles
are allocated above them — this is the cross-core pipe ring. Lower its depth with
optimizations=[pl.cross_core_slot(slot_num=N)] on the enclosing pl.at(...), or shrink the
tile that crosses the cube/vector boundary
```

The ring is `slot_size x slot_num` bytes, built by `BuildAutomaticPipeSetup` (`src/ir/transforms/utils/cross_core_pipe.cpp`) — `slot_size` is the FULL tile the consuming core pops, and `slot_num` defaults to `kDefaultAutoPipeSlotNum` (2, in every direction). That policy number is deliberately kept out of the message so it cannot go stale; the byte count it reports is read from the same `ResolveReserveBufferBases` result the allocator used as its floor. The ring lives in the **consuming** core's memory — Mat/L1 for V2C (`pl.aic_gather`), Vec/UB for C2V (`pl.aiv_shard`) — so the note is emitted only for the space that `GetReserveBufferMemorySpace` maps the function to. Two things are scoped deliberately: a Vec overflow in a function whose reserve buffer is in Mat gets the bare message, and the `pl.cross_core_slot` remediation is appended only when a buffer is actually one of the pipe rings. That is an **exact** name match, not a suffix test: `BuildPipeBufferName` is re-applied to this function's own kernel name (`<kernel>_aic` / `<kernel>_aiv` -> `<kernel>_v2c_slot_buffer`), because `pl.reserve_buffer` takes an arbitrary name and a hand-authored `scratch_v2c_slot_buffer` must not be pointed at a knob that cannot resize it. A hand-authored `pl.reserve_buffer` still gets its bytes attributed, but is not pointed at a knob that cannot shrink it.

The ring depth is deliberately **not** auto-capped to fit: depth is the cross-core pipelining depth, so silently shrinking it would turn a loud compile error into a quiet throughput regression.
