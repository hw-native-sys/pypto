# AllocateMemoryAddr Pass

Assigns real memory addresses to existing alloc operations.

## Overview

This pass is the physical-address boundary for non-DDR MemRefs. It resolves
`system.reserve_buffer(base=AUTO)`, chooses placements, and updates the existing
`tile.alloc` statements before PTO code generation. It never creates allocation
operations: InitMemRef has already created them with unallocated addresses.

The default `MemoryPlanner.PYPTO` path keeps the existing aligned bump placement
after MemoryReuse. The optional `MemoryPlanner.DSA` path receives the unmerged
allocation identities, exports the same structured problem used by the benchmark
framework, invokes the standalone solver, independently validates its result, and
writes the validated offsets back.

**Key responsibilities**:

- Collect unique MemRef objects from TileType variables
- Resolve `system.reserve_buffer` bases to explicit addresses per function
- Allocate sequential, 32-byte aligned addresses within each memory space
- Or, in DSA mode, jointly choose lifetime reuse and offsets with the standalone solver
- Update MemRef addresses in all variable types
- Update `tile.alloc` statement arguments with the allocated addresses

**When to use**: Run before code generation as the final memory-management pass.
The default pipeline runs it after MemoryReuse. The DSA pipeline deliberately
skips MemoryReuse, but still runs MaterializeSemanticAliases first so views,
loop-carried values, and in-place operations retain their mandatory identities.

## Planner modes

| Mode | Input to this pass | Placement | Failure behavior |
| ---- | ------------------ | --------- | ---------------- |
| `MemoryPlanner.PYPTO` | Opportunistically merged MemRefs from MemoryReuse | Backend-policy aligned bump allocation | Existing verifier reports invalid or over-capacity addresses |
| `MemoryPlanner.DSA` | Unmerged MemRefs after MaterializeSemanticAliases | Standalone first-fit DSA solver over schema-v1 `pypto_hard_v1` or explicitly experimental `pypto_research_v1` | Invalid export, capability mismatch, infeasibility, or validator failure stops compilation; no silent fallback |
| `MemoryPlanner.PTOAS` | None | This pass is skipped; ptoas `PlanMemory` owns placement | Deferred to ptoas |

DSA support is an optional CMake dependency. Build and consume an installed
`dsa-solver` 0.8 package as follows:

```bash
cmake -S /path/to/dsa-solver -B /path/to/dsa-solver/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/dsa-install
cmake --build /path/to/dsa-solver/build --parallel 2
cmake --install /path/to/dsa-solver/build

cmake -B build -DPYPTO_ENABLE_DSA_SOLVER=ON \
  -DCMAKE_PREFIX_PATH=/path/to/dsa-install
cmake --build build --parallel 2
```

The default build keeps `PYPTO_ENABLE_DSA_SOLVER=OFF`. It still exposes the
planner enum so configuration can be serialized consistently, but selecting DSA
at execution time produces an actionable error explaining how to reconfigure.
`passes.is_dsa_solver_available()` reports whether the active build includes the
adapter, allowing optional tests and applications to gate DSA selection.

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

Select the standalone path and optionally emit deterministic corpus documents
through PassContext-owned configuration:

```python
from pypto.pypto_core import passes

with passes.PassContext(
    [],
    memory_planner=passes.MemoryPlanner.DSA,
    dsa_export_dir="build/dsa-corpus",
):
    program_with_addrs = passes.allocate_memory_addr()(program)
```

Full compilation accepts the same selection as
`ir.compile(..., memory_planner=passes.MemoryPlanner.DSA,
dsa_export_dir="build/dsa-corpus")`.
`RunConfig` exposes the same `memory_planner` and `dsa_export_dir` fields. The
system-test harness additionally accepts `--memory-planner=dsa` and
`--dsa-export-dir=...` for suite-wide device validation and corpus capture.

The default export is `pypto_hard_v1`: standard DSA geometry with fixed memory
spaces, one conservative allocation-lifetime hull, capacities/reservations,
alignment, and typed separations. Lifetime-disjoint buffers may partially reuse
freed regions, including the subdivision required by #1908. If the adapter
emits the uncalibrated pipeline-serialization proxy, it upgrades that document
to `pypto_research_v1`; the proxy is not a production objective.
Legacy `pypto_structured` documents remain readable in the standalone tools but
are no longer emitted.

## Algorithm

1. **Collect MemRefs**: Traverse function body to find all unique MemRef objects from TileType variables
2. **Group by memory space**: Organize MemRefs by memory space (Vec, Mat, Left, Right, Acc)
3. **Resolve reserve buffers**: For each function, scan `system.reserve_buffer` calls, assign explicit bases to AUTO buffers, and compute the reserved end address per memory space
4. **Allocate addresses**: For each memory space, delegate to a `MemoryAllocatorPolicy` to filter spaces, order MemRefs, and align addresses. The default policy sorts by ID, uses 32-byte alignment, and starts from the reserved end (or `0`)
5. **Update in place**: Use `MemRefUpdateMutator` to:
   - Replace old MemRef references in variable types (TileType/TensorType) with new MemRefs containing real addresses
   - Update existing `tile.alloc` `AssignStmt`s: replace LHS MemRef and update addr argument in the Call expression
   - Rewrite `system.reserve_buffer` kwargs with the resolved explicit `base`

### Standalone DSA path

When `MemoryPlanner.DSA` is active, step 4 is replaced by this guarded path:

1. Reuse the phi/loop-aware lifetime analysis from MemoryReuse without running
   its opportunistic coalescer.
2. Export one buffer per mandatory `MemRef.base_` identity. The buffer size is
   the largest member size, so differently sized values may occupy that identity
   over its lifetime. The exported lifetime is the conservative allocation hull
   from the earliest member definition through the latest member use. Individual
   SSA-member gaps are not treated as physical dead time: loop carries, views,
   and in-place aliases can preserve a value across such a gap. Multi-interval
   reuse requires a separate proof that the physical value is dead in each hole.
3. Convert PyPTO statement points into half-open read/write events. A definition
   starts at `2 * def + 1`; the final read ends at `2 * last_use + 1`. A value
   with no later read receives one write event. Consequently, an input's final
   read may share an address with the result written by that statement.
4. Export fixed memory pools, backend capacities, a leading reserved range, and
   hard separation pairs for pipeline clones, backend hazards, and op-specific
   no-alias rules. Pipeline residue counts come from a dry run of MemoryReuse's
   exact whole-space packer, so depth shedding accounts for alignment, reserved
   memory, co-resident tiles, and other pipeline groups rather than using only
   `capacity / largest_stage`. Each separation retains its typed source.
5. Retain normalized alias-class members and pipeline group/stage/residue data.
   Capacity-folded same-residue stages emit sparse chronological cross-pipe
   reuse penalties; the generic constraints and cost model remain authoritative.
6. Validate the schema/profile, match solver capabilities, solve, and validate
   every placement independently against sizes, alignment, lifetimes, pools,
   capacities, reserved ranges, and separations.
7. Write each placement back while preserving every view's relative byte offset.

The version-1 adapter intentionally keeps pool assignment fixed and uses the
portable peak objective. Its exported reuse costs are therefore benchmark data
for cost-aware solvers and do not change the currently selected first-fit result.
Branch exclusivity that is not visible in the exported intervals is still
conservative rather than unsound. The buffers remain fixed-size allocations;
subdivision comes from jointly assigning offsets after an earlier region expires,
not from resizing a buffer during its lifetime. Cost-aware objectives and
PyPTO-structured search moves remain research extensions behind capability
matching.

Scheduling itself is also fixed before this pass, even though a different legal
schedule would produce different lifetimes. See
[Joint Scheduling and Local-Memory Planning](../proposals/joint_schedule_memory_cooptimization.md)
for the PyPTO-owned, PTOAS-owned, and cross-layer co-optimization options.

If `dsa_export_dir` is set, each InCore function is written as
`pypto_<escaped-function-name>.dsa.json`. Serialization is deterministic and
contains no IR pointers or machine-specific paths, so the document can be copied
directly into the standalone real-instance corpus.

**Address allocation (default policy)**:

- Each memory space has its own address space starting from 0 unless `system.reserve_buffer` already reserved a leading window in that space
- Addresses are 32-byte aligned: `next_addr = align32(current_addr + size)`
- MemRefs are sorted by ID for deterministic allocation order
- DDR MemRefs are skipped (addresses managed externally)

**View MemRefs (slices) share one slot**:

MemRefs that share the same `base_` Ptr (a root allocation plus its `tile.slice` views) are co-located in a single slot sized by the largest member, since every view physically aliases its parent. Each member keeps its own relative offset within the slot: `new_addr = slot_base + member.byte_offset` (the relative offset InitMemRef computed). The root sits at `slot_base`; a view at row `k` sits at `slot_base + k * row_stride`. This matters for chains where a view's offset is not re-derived at codegen — e.g. a `tile.reshape` of a `tile.slice` does not emit `pto.subview`, so its `pto.alloc_tile addr` is read directly from this MemRef offset.

Backends can override these defaults by supplying a custom `MemoryAllocatorPolicy` via `Backend::CreateMemoryAllocatorPolicy()`. See [Allocation Policy](#allocation-policy) below.

## Example

### Before (after InitMemRef + MemoryReuse)

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 16384, 0)   # addr=-1 (unallocated)
mem_vec_1: MemRefType = tile.alloc(Vec, -1, 16384, 1)   # addr=-1 (unallocated)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
# ]
```

### After (addresses assigned)

```python
# SeqStmts [
mem_vec_0: MemRefType = tile.alloc(Vec, 0, 16384, 0)      # addr=0
mem_vec_1: MemRefType = tile.alloc(Vec, 16384, 16384, 1)   # addr=16384 (aligned)
tile_a: Tile[[64, 64], FP32, memref=mem_vec_0] = tile.load(...)
tile_b: Tile[[64, 64], FP32, memref=mem_vec_1] = tile.add(tile_a, ...)
# ]
```

### Multiple Memory Spaces

```python
# Before:
mem_vec_0: MemRefType = tile.alloc(Vec, -1, 2048, 0)
mem_left_1: MemRefType = tile.alloc(Left, -1, 2048, 1)
mem_right_2: MemRefType = tile.alloc(Right, -1, 2048, 2)
mem_acc_3: MemRefType = tile.alloc(Acc, -1, 2048, 3)

# After (each space starts from addr=0):
mem_vec_0: MemRefType = tile.alloc(Vec, 0, 2048, 0)
mem_left_1: MemRefType = tile.alloc(Left, 0, 2048, 1)
mem_right_2: MemRefType = tile.alloc(Right, 0, 2048, 2)
mem_acc_3: MemRefType = tile.alloc(Acc, 0, 2048, 3)
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

```cpp
Pass AllocateMemoryAddr();
```

**Implementation**: `src/ir/transforms/allocate_memory_addr_pass.cpp`

- `memref_collectors::CollectMemRefsWithSpace` collects unique MemRefs and their memory spaces
- `AllocateMemoryAddresses` assigns sequential aligned addresses per memory space using a `MemoryAllocatorPolicy`
- `dsa_adapter::BuildStructuredProblem` exports the IR-free schema-v1 problem
- `dsa_adapter::SolveWithFirstFit` capability-matches, solves, and independently validates
- `dsa_adapter::BuildMemRefReplacements` performs view-aware writeback
- `MemRefUpdateMutator` updates both variable types and `tile.alloc` statement arguments in a single traversal

**Python binding**: `python/bindings/modules/passes.cpp`

```cpp
passes.def("allocate_memory_addr", &pass::AllocateMemoryAddr,
           "Allocates real memory addresses for existing alloc operations.");
```

**Tests**: `tests/ut/ir/transforms/test_allocate_memory_addr_pass.py`

- Tests address allocation with 32-byte alignment
- Tests multiple MemRef allocations
- Tests empty function (no tiles)
- Tests alloc statements are prepended to the function body's top-level `SeqStmts`
- Tests raw pointer uniqueness for MemRef deduplication
- Tests default policy behavior without a backend configured
- Tests DSA read-before-write reuse, reserved ranges, view-offset writeback, and deterministic export
- Tests alias-class, typed separation, pipeline-group/residue, and sparse reuse-cost export
- Replays the #1908 fragmentation shape through exporter, standalone solver, validator, and writeback

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
| `OrderMemRefs` | Sort MemRefs within a space before allocation | Ascending by `MemRef::id_` |

### Default policy

`DefaultMemoryAllocatorPolicy` preserves the original hard-coded behavior (skip DDR, 32-byte alignment, sort by ID).

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
