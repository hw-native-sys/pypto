# AllocateMemoryAddr Pass

Assigns real memory addresses to existing alloc operations.

## Overview

This pass allocates concrete memory addresses for non-DDR MemRefs and updates the existing `tile.alloc` statements in place. It also resolves `system.reserve_buffer(base=AUTO)` to explicit base addresses before PTO code generation. Unlike creating new alloc operations, this pass only modifies the address field of alloc statements that were created by InitMemRef (with `addr=-1`).

**Key responsibilities**:

- Collect unique MemRef objects from TileType variables
- Resolve `system.reserve_buffer` bases to explicit addresses per function
- Allocate sequential, 32-byte aligned addresses within each memory space
- Update MemRef addresses in all variable types
- Update `tile.alloc` statement arguments with the allocated addresses

**When to use**: Run after MemoryReuse (to respect shared MemRefs) and before code generation. Final pass in memory management pipeline.

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

## Algorithm

1. **Collect MemRefs**: Traverse function body to find all unique MemRef objects from TileType variables
2. **Group by memory space**: Organize MemRefs by memory space (Vec, Mat, Left, Right, Acc)
3. **Resolve reserve buffers**: For each function, scan `system.reserve_buffer` calls, assign explicit bases to AUTO buffers, and compute the reserved end address per memory space
4. **Allocate addresses**: For each memory space, delegate to a `MemoryAllocatorPolicy` to filter spaces, order MemRefs, and align addresses. The default policy sorts by ID, uses 32-byte alignment, and starts from the reserved end (or `0`)
5. **Update in place**: Use `MemRefUpdateMutator` to:
   - Replace old MemRef references in variable types (TileType/TensorType) with new MemRefs containing real addresses
   - Update existing `tile.alloc` `AssignStmt`s: replace LHS MemRef and update addr argument in the Call expression
   - Rewrite `system.reserve_buffer` kwargs with the resolved explicit `base`

**Address allocation (default policy)**:

- Each memory space has its own address space starting from 0 unless `system.reserve_buffer` already reserved a leading window in that space
- Addresses are 32-byte aligned: `next_addr = align32(current_addr + size)`
- MemRefs are sorted by ID for deterministic allocation order
- DDR MemRefs are skipped (addresses managed externally)

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

- `MemRefCollectorVisitor` collects unique MemRefs from TileType variables
- `AllocateMemoryAddresses` assigns sequential aligned addresses per memory space using a `MemoryAllocatorPolicy`
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
