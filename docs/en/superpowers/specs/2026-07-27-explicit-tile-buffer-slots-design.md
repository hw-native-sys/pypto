# Explicit Tile Buffer Slots Design

**Issue:** #2131

## Summary

PyPTO will expose first-class, tuple-like sets of on-chip tile buffers. A user
can allocate a fixed number of homogeneous physical slots, select a slot with a
runtime integer expression, and bind supported tile producers directly to that
slot. L1, L0B, and L0C sets are independent, so nested loops can rotate them at
different granularities without `MemoryReuse` coalescing distinct slots.

The same IR supports both memory planners. The PyPTO planner assigns one base
address to the complete set, while the PTOAS planner leaves the address absent
for `PlanMemory`. Both emit `pto.alloc_multi_tile` and select a runtime slot
with `pto.multi_tile_get`.

## Goals

- Give users stable buffer-set and slot identity across loop iterations.
- Permit runtime slot selection such as `iteration % 2`.
- Support independent L1, L0B, and L0C rotations in nested pipelines.
- Bind load, extract, move, matmul, and matmul-accumulate results directly to a
  selected destination slot.
- Preserve correct RAW, WAR, and WAW dependencies when a slot is reused.
- Prevent `MemoryReuse` from coalescing explicit slots or unrelated buffers
  into an explicit buffer set.
- Work with both `MemoryPlanner.PYPTO` and `MemoryPlanner.PTOAS`.
- Keep the common Python syntax tuple-like without representing the set as an
  IR `TupleType`.

## Non-goals

- User-specified physical byte addresses. The selected memory planner owns
  address placement.
- Arbitrary heterogeneous buffer collections.
- Applying `out=` to every tile-producing operator in the first version.
- Replacing the existing automatic `pl.pipeline` buffering strategy.
- Adding a standalone example file. The feature is documented and tested in
  the existing documentation and test hierarchy.

## User API

### Creating and selecting buffers

```python
l1_buffers = pl.create_tile_buffers(
    2, [128, 512], pl.BF16, pl.Mem.Mat
)
l0b_buffers = pl.create_tile_buffers(
    2, [128, 128], pl.BF16, pl.Mem.Right
)
l0c_buffers = pl.create_tile_buffers(
    2, [16, 128], pl.FP32, pl.Mem.Acc
)

for stack in pl.range(STACKS):
    l1_slot = l1_buffers[stack % 2]
    b_l1 = pl.load(
        b,
        [stack * K, 0],
        [K, STACK_N],
        target_memory=pl.Mem.Mat,
        out=l1_slot,
    )

    for col in pl.range(0, STACK_N, L0_N):
        sub = col // L0_N
        l0_index = sub % 2
        l0b_slot = l0b_buffers[l0_index]
        l0c_slot = l0c_buffers[l0_index]

        b_l0 = pl.tile.extract(
            b_l1,
            0,
            col,
            [K, L0_N],
            target_memory=pl.Mem.Right,
            out=l0b_slot,
        )
        acc = pl.tile.matmul(q_l0, b_l0, out=l0c_slot)
        out = pl.store(acc, [stack * M, col], out)

        pl.tile.release(l0b_slot)
        pl.tile.release(l0c_slot)

    pl.tile.release(l1_slot)
```

`create_tile_buffers` returns a Python `TileBufferSet` wrapper. The wrapper
supports `len(buffers)` and `buffers[index]`, but its underlying expression has
`TileBufferSetType`, not `TupleType`. Indexing returns a normal `Tile` wrapper
whose expression is a `tile.buffer_slot` call.

### Supported destination forms

The initial public `out=` surface is:

- `pl.load(..., out=slot)`
- `pl.tile.extract(..., out=slot)`
- `pl.tile.move(..., out=slot)`
- `pl.tile.matmul(..., out=slot)`
- `pl.tile.matmul_acc(..., out=slot)`

Existing calls without `out=` retain their current signatures and behavior.
The Python layer lowers `out=` calls to distinct internal destination-form ops:

```text
tile.load_into(..., destination)
tile.extract_into(..., destination)
tile.move_into(..., destination)
tile.matmul_into(..., destination)
tile.matmul_acc_into(..., destination)
```

Separate IR ops avoid changing the meaning of existing ops and make destination
binding explicit to every pass and code generator.

## Type Model

### `TileBufferSetType`

`TileBufferSetType` is a first-class IR type with these fields:

- per-slot static shape;
- element dtype;
- tile view/layout;
- on-chip memory space;
- compile-time slot count;
- optional group `MemRef`, populated by `InitMemRef` for the PyPTO planner.

It represents storage, not a tile value. Ordinary tile operators reject it.
Only buffer-set operations such as `tile.buffer_slot` accept it.

`TupleType` is not reused because tuple elements are independent SSA values,
may be heterogeneous, have independent allocation identities, and support
structural rather than runtime selection. A buffer set instead represents one
homogeneous allocation group with runtime indexing.

### Selected slot type

`tile.buffer_slot(set, index)` returns the set's per-slot `TileType`. After
`InitMemRef`, the selected tile carries a slot `MemRef` that shares the set's
allocation base and has a symbolic relative offset:

```text
index * aligned_slot_size
```

The group `MemRef` has offset zero and size `count * aligned_slot_size`. The
allocator places only the group root. Slot-relative offsets remain symbolic and
are consumed by `pto.multi_tile_get`, rather than becoming independent
allocations.

Values produced by a destination-form op reuse the destination slot `MemRef`.
This preserves allocation lineage through subsequent uses while codegen binds
the producer's `outs` operand to the selected slot handle.

## IR Operations

```text
tile.create_buffer_set(shape)
    attrs: dtype, target_memory, count, layout options
    result: TileBufferSetType

tile.buffer_slot(buffer_set, index)
    result: per-slot TileType

tile.release(slot)
    result: none; lifetime marker consumed before PTO codegen
```

The creation op maps to `pto.alloc_multi_tile`; slot selection maps to
`pto.multi_tile_get`. The release op is frontend lifetime metadata and has no
PTO instruction.

## Lifetime and Dependency Semantics

Indexing a set acquires a slot lease. The lease ends at the earliest of:

1. an explicit `tile.release(slot)` marker; or
2. the last SSA use when no release marker is present.

All reads of the slot and of destination-form results that alias it must finish
before the lease ends. A use after explicit release is invalid. Acquiring the
same physical slot again creates a new lease; dependency analysis retains the
required WAR or WAW edge from the prior lease.

Explicit release is optional. It lets users shorten a conservatively inferred
lifetime, but it never removes a dependency from an operation that still uses
the slot.

The dependency key for a selected slot is:

```text
(buffer-set allocation identity, normalized slot index)
```

Two accesses are independent when the compiler proves their normalized slot
indices differ. If it cannot prove that, it conservatively treats them as
possibly aliasing.

## Pipeline Lowering and Scheduling

Explicit buffer operations participate in the existing pipeline lowering.
They do not replace `pl.pipeline`.

For a two-stage inner pipeline:

```python
for col in pl.pipeline(0, 512, 128, stage=2):
    index = (col // 128) % 2
    b = l0b_buffers[index]
    c = l0c_buffers[index]
    b_value = pl.tile.extract(..., out=b)
    acc = pl.tile.matmul(q, b_value, out=c)
    pl.store(acc, ...)
```

`LowerPipelineLoops` clones the body for the two stages and preserves the
buffer-set identity. Scalar simplification normalizes the cloned indices to
different slots. `CanonicalizeIOOrder` may then schedule:

```text
TEXTRACT(..., out=B0)
TEXTRACT(..., out=B1)
TMATMUL(..., out=C0)
TMATMUL(..., out=C1)
TSTORE(C0)
TSTORE(C1)
```

At the next reuse of `B0` or `C0`, the real loop-carried dependency remains.
Nested L1 and L0 pipelines use separate buffer-set identities, so their modulo
expressions and cadences are independent.

## Memory Planner Behavior

### PyPTO planner

`InitMemRef` creates one allocation root for each `TileBufferSetType`, sized as
the aligned per-slot footprint multiplied by `count`. `MemoryReuse` treats that
root as explicit and non-coalescible. `AllocateMemoryAddr` assigns one base
address to the group.

PTO codegen emits:

```text
%buffers = pto.alloc_multi_tile addr = %group_addr ...
%slot = pto.multi_tile_get %buffers[%index] ...
```

It does not emit per-slot `pto.alloc_tile` operations.

### PTOAS planner

The same IR reaches codegen without a PyPTO-assigned address:

```text
%buffers = pto.alloc_multi_tile ...
%slot = pto.multi_tile_get %buffers[%index] ...
```

PTOAS `PlanMemory` owns group placement. Destination and lifetime semantics are
identical between planners.

## Validation and Errors

- `count` must be in `[2, 16]`, matching `pto.multi_tile_buf`.
- Per-slot shape must be non-empty, static, and positive.
- Memory space must be one of `Vec`, `Mat`, `Left`, `Right`, `Acc`, or `Bias`.
- A constant index outside `[0, count)` is rejected at compile time.
- A dynamic index must have integer dtype. Runtime range safety is the user's
  responsibility; `% count` is the normal form.
- Destination and slot shape, dtype, memory space, valid shape, and layout must
  be compatible for the chosen producer.
- A buffer set cannot be passed directly to an ordinary tile op.
- A lease cannot be used after explicit release.
- A lease cannot be the destination of two overlapping producers unless normal
  dependency analysis orders them.
- Explicit buffer-set capacity participates in existing per-memory-space
  capacity verification.

Errors caused by source programs use user-facing checks with the source span
and report the received and expected values. Internal pass invariants use
`INTERNAL_CHECK_SPAN`.

## Pass and Component Changes

- IR type, reflection, structural equality/hash, serialization, bindings, and
  stubs gain `TileBufferSetType`.
- Python typing and parser support gain the tuple-like `TileBufferSet` wrapper.
- Tile memory ops gain create/select/release and destination forms.
- Type inference validates destination compatibility.
- `LowerPipelineLoops` and scalar simplification preserve and normalize slot
  selection across cloned stages.
- Statement dependency analysis understands buffer-set plus slot-index identity.
- `InitMemRef` materializes group allocation and slot lineage.
- `MemoryReuse` excludes explicit group roots from coalescing.
- `AllocateMemoryAddr` places group roots for the PyPTO planner.
- PTO codegen emits address-bearing or address-free multi-buffer operations.
- A lifetime verifier rejects release and aliasing misuse before release markers
  are erased.

## Testing Strategy

### Type and frontend tests

- Construct and inspect `TileBufferSetType`.
- Verify structural equality and structural hash behavior.
- Verify printer/parser and serialization round trips.
- Verify `len(buffers)` and dynamic `buffers[index]` lowering.
- Reject direct use of a buffer set as a tile value.

### Validation tests

- Reject counts 1 and 17.
- Reject empty, dynamic, and non-positive slot shapes.
- Reject constant out-of-range indices and non-integer dynamic indices.
- Reject mismatched destination shape, dtype, memory space, and layout.
- Reject use after release.

### Memory pass tests

- Two `[16, 128]` FP32 Acc slots form one 16 KiB allocation group.
- Distinct slots and groups are not coalesced by `MemoryReuse`.
- Independent L1, L0B, and L0C groups retain separate allocation identities.
- Existing capacity diagnostics include explicit groups.

### Pipeline regression tests

- Reproduce the issue's two-level 512-token L1 and 128-token L0 cadence.
- Verify two Right slots and two Acc slots survive lowering.
- Verify the two-extract, two-matmul, two-store schedule.
- Verify the dependency before the next reuse of slot zero remains.

### Codegen tests

- PyPTO planner emits `pto.alloc_multi_tile` with `addr`.
- PTOAS planner emits the same op without `addr`.
- Both emit dynamic `pto.multi_tile_get`.
- No redundant per-slot `pto.alloc_tile`, Acc-to-Acc `TMOV`, or disconnected
  destination allocation is emitted.

Documentation updates cover the English and Chinese DSL and affected pass
documentation. Code examples remain identical between languages.

## Alternatives Considered

### Add `count` to `TileType`

This reduces the number of IR type classes, but every existing
`As<TileType>()` check would also accept a buffer collection. Missing a new
`count == 1` guard could silently pass a whole set to a compute op or propagate
the count through type inference. A separate type makes such omissions fail
immediately.

### Reuse `TupleType`

Tuple elements have independent value and allocation identity and are normally
selected structurally. They do not represent one homogeneous allocation group
or runtime selection. Only the Python tuple-like syntax is reused.

### Annotate `pl.pipeline`

A pipeline-only buffer-count annotation would be smaller, but it would not let
users select slots with independent nested indices or bind individual
destinations. It does not satisfy issue #2131's general buffer identity contract.

### Pin static addresses with markers

Static pinning can protect separate buffers under the PyPTO planner, but it
cannot express runtime slot selection and makes users own platform-specific
addresses. Address placement remains a planner responsibility in this design.
