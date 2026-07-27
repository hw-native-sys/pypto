# Explicit Tile Buffer Slots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class, dynamically indexed tile buffer sets that let users independently rotate L1, L0B, and L0C destinations with both PyPTO and PTOAS memory planners.

**Architecture:** Introduce `TileBufferSetType` as a storage type and expose it through a tuple-like Python `TileBufferSet`. `tile.create_buffer_set` owns one homogeneous allocation group, `tile.buffer_slot` selects a runtime slot as an ordinary `TileType`, and destination-form producer ops bind writes directly to that slot. Memory and dependency passes preserve the allocation-set identity and use `(set identity, normalized slot index)` for conservative aliasing.

**Tech Stack:** C++17 IR and passes, nanobind Python bindings, Python 3.10+ DSL, pytest, PTO MLIR code generation.

## Global Constraints

- `count` is a compile-time integer in `[2, 16]`.
- Slot shapes are non-empty, static, and positive.
- Supported memory spaces are `Vec`, `Mat`, `Left`, `Right`, `Acc`, and `Bias`.
- Existing producer calls without `out=` retain their current signatures and behavior.
- English documentation is authoritative and Chinese documentation must remain synchronized.
- Every production behavior is implemented only after its focused test fails for the expected reason.

---

### Task 1: First-class `TileBufferSetType`

**Files:**

- Modify: `include/pypto/ir/core.h`
- Modify: `include/pypto/ir/kind_traits.h`
- Modify: `include/pypto/ir/type.h`
- Modify: `src/ir/type.cpp`
- Modify: `src/ir/transforms/structural_equal.cpp`
- Modify: `src/ir/transforms/structural_hash.cpp`
- Modify: `src/ir/serialization/serializer.cpp`
- Modify: `src/ir/serialization/deserializer.cpp`
- Modify: `src/ir/transforms/python_printer.cpp`
- Modify: `python/bindings/modules/ir.cpp`
- Modify: `python/pypto/pypto_core/ir.pyi`
- Create: `tests/ut/ir/core/test_tile_buffer_set_type.py`
- Modify: `tests/ut/ir/transforms/test_serialization.py`
- Modify: `tests/ut/ir/transforms/test_equality.py`

**Interfaces:**

- Produces: `TileBufferSetType(shape, dtype, count, memref=None, tile_view=None, memory_space=None)`.
- Produces fields: `shape_`, `dtype_`, `count_`, `memref_`, `tile_view_`, and `memory_space_`.
- Consumes: existing `ShapedType`, `TileView`, `MemRef`, and reflection infrastructure.

- [ ] **Step 1: Write failing type behavior tests**

```python
def test_tile_buffer_set_type_exposes_homogeneous_slot_contract():
    ty = ir.TileBufferSetType([16, 128], DataType.FP32, 2, None, None, ir.MemorySpace.Acc)
    assert ty.count == 2
    assert [dim.value for dim in ty.shape] == [16, 128]
    assert ty.dtype == DataType.FP32
    assert ty.memory_space == ir.MemorySpace.Acc

@pytest.mark.parametrize("count", [1, 17])
def test_tile_buffer_set_type_rejects_invalid_count(count):
    with pytest.raises(ValueError, match=r"count.*\[2, 16\]"):
        ir.TileBufferSetType([16, 128], DataType.FP32, count)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/ut/ir/core/test_tile_buffer_set_type.py -v`

Expected: collection or attribute failure because `ir.TileBufferSetType` does not exist.

- [ ] **Step 3: Add the C++ type, object kind, bindings, and stub**

Implement a final `ShapedType` subclass whose constructor validates count, static positive dimensions, and supported on-chip memory space. Add its reflection fields, nanobind constructors/properties, and `.pyi` declaration.

- [ ] **Step 4: Add failing structural and serialization round-trip tests**

```python
def test_tile_buffer_set_type_count_participates_in_structural_equal():
    left = ir.TileBufferSetType([16, 128], DataType.FP32, 2, memory_space=ir.MemorySpace.Acc)
    right = ir.TileBufferSetType([16, 128], DataType.FP32, 3, memory_space=ir.MemorySpace.Acc)
    assert not ir.structural_equal(left, right)
```

Add a program round trip containing a variable of this type and assert structural equality after serialization.

- [ ] **Step 5: Run the new tests and implement printer/equality/hash/serialization support**

Run: `python -m pytest tests/ut/ir/core/test_tile_buffer_set_type.py tests/ut/ir/transforms/test_equality.py tests/ut/ir/transforms/test_serialization.py -v`

Expected before implementation: FAIL on unhandled `TileBufferSetType`; expected after implementation: PASS.

- [ ] **Step 6: Commit the type slice**

```bash
git add include/pypto/ir/core.h include/pypto/ir/kind_traits.h include/pypto/ir/type.h src/ir/type.cpp src/ir/transforms/structural_equal.cpp src/ir/transforms/structural_hash.cpp src/ir/serialization/serializer.cpp src/ir/serialization/deserializer.cpp src/ir/transforms/python_printer.cpp python/bindings/modules/ir.cpp python/pypto/pypto_core/ir.pyi tests/ut/ir/core/test_tile_buffer_set_type.py tests/ut/ir/transforms/test_equality.py tests/ut/ir/transforms/test_serialization.py
git commit -m "feat(ir): add tile buffer set type"
```

### Task 2: Buffer-set IR operations and validation

**Files:**

- Modify: `src/ir/op/tile_ops/memory.cpp`
- Modify: `python/pypto/ir/op/tile_ops.py`
- Modify: `python/pypto/ir/op/__init__.py`
- Create: `tests/ut/ir/operators/test_tile_buffer_set_ops.py`

**Interfaces:**

- Consumes: `TileBufferSetType` from Task 1.
- Produces: `tile.create_buffer_set(shape, dtype, target_memory, count) -> TileBufferSetType`.
- Produces: `tile.buffer_slot(buffer_set, index) -> TileType`.
- Produces: `tile.release(slot) -> ScalarType(BOOL)` as an erasable lifetime marker.

- [ ] **Step 1: Write failing operator tests**

```python
def test_buffer_slot_accepts_dynamic_integer_index():
    buffers = tile_ops.create_buffer_set([16, 128], DataType.FP32, ir.MemorySpace.Acc, 2)
    index = ir.Var("index", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    slot = tile_ops.buffer_slot(buffers, index)
    assert isinstance(slot.type, ir.TileType)
    assert slot.type.memory_space == ir.MemorySpace.Acc

def test_buffer_slot_rejects_constant_out_of_range_index():
    buffers = tile_ops.create_buffer_set([16, 128], DataType.FP32, ir.MemorySpace.Acc, 2)
    with pytest.raises(ValueError, match="index.*out of range"):
        tile_ops.buffer_slot(buffers, 2)
```

Also test a non-integer dynamic index and passing an ordinary tile as the first argument.

- [ ] **Step 2: Run the operator tests and verify RED**

Run: `python -m pytest tests/ut/ir/operators/test_tile_buffer_set_ops.py -v`

Expected: FAIL because the three operators are not registered.

- [ ] **Step 3: Register and wrap the three operations**

`tile.create_buffer_set` deduces `TileBufferSetType`; `tile.buffer_slot` returns a matching `TileType` without an independent allocation; `tile.release` validates a selected-slot operand and carries no PTO-side behavior.

- [ ] **Step 4: Run focused operator tests and existing tile-op tests**

Run: `python -m pytest tests/ut/ir/operators/test_tile_buffer_set_ops.py tests/ut/ir/operators/test_tile_ops.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the operator slice**

```bash
git add src/ir/op/tile_ops/memory.cpp python/pypto/ir/op/tile_ops.py python/pypto/ir/op/__init__.py tests/ut/ir/operators/test_tile_buffer_set_ops.py
git commit -m "feat(ir): add tile buffer set operations"
```

### Task 3: Tuple-like Python DSL API

**Files:**

- Create: `python/pypto/language/typing/tile_buffer_set.py`
- Modify: `python/pypto/language/typing/__init__.py`
- Modify: `python/pypto/language/op/unified_ops.py`
- Modify: `python/pypto/language/op/tile_ops.py`
- Modify: `python/pypto/language/op/__init__.py`
- Modify: `python/pypto/language/__init__.py`
- Modify: `python/pypto/language/parser/_dsl_invoker.py`
- Modify: `python/pypto/language/parser/ast_parser.py`
- Modify: `python/pypto/language/parser/type_resolver.py`
- Create: `tests/ut/language/parser/test_tile_buffer_set.py`

**Interfaces:**

- Consumes: low-level operations from Task 2.
- Produces: `create_tile_buffers(count, shape, dtype, memory_space) -> TileBufferSet`.
- Produces: `TileBufferSet.__len__() -> int` and `TileBufferSet.__getitem__(index: int | Scalar) -> Tile`.
- Produces: `pl.tile.release(slot: Tile) -> None`.
- Produces annotation form: `pl.TileBufferSet[[16, 128], pl.FP32, 2, pl.MemorySpace.Acc]` for printer round trips.

- [ ] **Step 1: Write failing DSL parse tests**

```python
@pl.program
class DynamicSlotProgram:
    @pl.function
    def main(self, x: pl.Tensor[[16, 128], pl.FP32], out: pl.Out[pl.Tensor[[16, 128], pl.FP32]]):
        buffers = pl.create_tile_buffers(2, [16, 128], pl.FP32, pl.MemorySpace.Vec)
        for i in pl.range(2):
            slot = buffers[i % 2]
            value = pl.load(x, [0, 0], [16, 128], out=slot)
            pl.tile.release(value)
```

Assert that the parsed IR contains one `tile.create_buffer_set`, a dynamic `tile.buffer_slot`, and one `tile.release`.

- [ ] **Step 2: Run the parser test and verify RED**

Run: `python -m pytest tests/ut/language/parser/test_tile_buffer_set.py -v`

Expected: FAIL because `create_tile_buffers` and `TileBufferSet` are missing.

- [ ] **Step 3: Implement the wrapper, exports, and parser invocation support**

The wrapper stores the underlying expression and compile-time count. `__getitem__` unwraps `Scalar` indices and calls `tile.buffer_slot`; it does not reuse `TupleGetItemExpr`. The type resolver reconstructs printed `pl.TileBufferSet[shape, dtype, count, memory_space]` annotations as `TileBufferSetType`.

- [ ] **Step 4: Run parser and typing regressions**

Run: `python -m pytest tests/ut/language/parser/test_tile_buffer_set.py tests/ut/language/parser/test_subscript_syntax.py tests/ut/language/parser/test_tuple_syntax.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the DSL slice**

```bash
git add python/pypto/language/typing/tile_buffer_set.py python/pypto/language/typing/__init__.py python/pypto/language/op/unified_ops.py python/pypto/language/op/tile_ops.py python/pypto/language/op/__init__.py python/pypto/language/__init__.py python/pypto/language/parser/_dsl_invoker.py python/pypto/language/parser/ast_parser.py python/pypto/language/parser/type_resolver.py tests/ut/language/parser/test_tile_buffer_set.py
git commit -m "feat(language): expose explicit tile buffer sets"
```

### Task 4: Grouped MemRef planning

**Files:**

- Modify: `include/pypto/ir/transforms/utils/memref_utils.h`
- Modify: `src/ir/transforms/init_memref.cpp`
- Modify: `src/ir/transforms/memory_reuse_pass.cpp`
- Modify: `src/ir/transforms/allocate_memory_addr_pass.cpp`
- Modify: `src/ir/transforms/mutator.cpp`
- Modify: `tests/ut/ir/transforms/test_init_memref.py`
- Modify: `tests/ut/ir/transforms/test_memory_reuse.py`
- Modify: `tests/ut/ir/transforms/test_allocate_memory_addr_pass.py`

**Interfaces:**

- Consumes: `TileBufferSetType` and buffer-set ops.
- Produces: one group `MemRef` of `count * aligned_slot_size` on the set.
- Produces: selected-slot `TileType` values whose `MemRef.base_` is the group base and whose relative offset is represented by the slot index rather than a separately allocated root.

- [ ] **Step 1: Write failing `InitMemRef` group-size and lineage tests**

Use two FP32 `[16, 128]` Acc slots and assert exactly one allocation root of 16384 bytes, with both selected values sharing that root.

- [ ] **Step 2: Run `InitMemRef` tests and verify RED**

Run: `python -m pytest tests/ut/ir/transforms/test_init_memref.py -k tile_buffer_set -v`

Expected: FAIL because the pass does not recognize the storage type.

- [ ] **Step 3: Implement group creation and slot lineage cloning**

Extend type-remapping utilities to retain `count`, view, and memory space. Hoist one `tile.alloc`-style group root for PyPTO planning while keeping the slot offset symbolic for codegen.

- [ ] **Step 4: Write failing reuse and address tests**

Assert that two explicit groups remain distinct after `MemoryReuse`, unrelated automatic tiles cannot reuse either group, and `AllocateMemoryAddr` assigns one aligned address range per group.

- [ ] **Step 5: Implement explicit-group exclusion and address placement**

Identify explicit roots by the defining `tile.create_buffer_set` lineage, exclude them from reuse candidates, and account for their total group size in capacity diagnostics.

- [ ] **Step 6: Run memory pass tests**

Run: `python -m pytest tests/ut/ir/transforms/test_init_memref.py tests/ut/ir/transforms/test_memory_reuse.py tests/ut/ir/transforms/test_allocate_memory_addr_pass.py -v`

Expected: PASS.

- [ ] **Step 7: Commit the memory slice**

```bash
git add include/pypto/ir/transforms/utils/memref_utils.h src/ir/transforms/init_memref.cpp src/ir/transforms/memory_reuse_pass.cpp src/ir/transforms/allocate_memory_addr_pass.cpp src/ir/transforms/mutator.cpp tests/ut/ir/transforms/test_init_memref.py tests/ut/ir/transforms/test_memory_reuse.py tests/ut/ir/transforms/test_allocate_memory_addr_pass.py
git commit -m "feat(passes): plan explicit tile buffer groups"
```

### Task 5: Destination-form producers and PTO codegen

**Files:**

- Modify: `src/ir/op/tile_ops/memory.cpp`
- Modify: `src/ir/op/tile_ops/transform.cpp`
- Modify: `src/ir/op/tile_ops/matmul.cpp`
- Modify: `python/pypto/ir/op/tile_ops.py`
- Modify: `python/pypto/language/op/unified_ops.py`
- Modify: `python/pypto/language/op/tile_ops.py`
- Modify: `src/backend/common/pto_ops_memory.cpp`
- Modify: `src/backend/common/pto_ops_datamove.cpp`
- Modify: `src/backend/common/pto_ops_elementwise.cpp`
- Modify: `src/backend/common/pto_ops_shared.cpp`
- Modify: `include/pypto/codegen/pto/pto_codegen.h`
- Modify: `src/codegen/pto/pto_codegen.cpp`
- Create: `tests/ut/ir/operators/test_tile_destination_ops.py`
- Create: `tests/ut/codegen/test_explicit_tile_buffers.py`

**Interfaces:**

- Produces internal ops: `tile.load_into`, `tile.extract_into`, `tile.move_into`, `tile.matmul_into`, and `tile.matmul_acc_into`.
- Each internal op takes the selected destination as its final argument and returns a `TileType` that aliases the destination `MemRef`.
- Existing public functions gain keyword-only `out: Tile | None = None` and lower to the original op when `out is None`.

- [ ] **Step 1: Write failing destination validation tests**

Cover a successful Acc `matmul_into`, and separate failures for shape, dtype, memory-space, valid-shape, and layout mismatch. Assert that legacy calls still create the original op names.

- [ ] **Step 2: Run operator tests and verify RED**

Run: `python -m pytest tests/ut/ir/operators/test_tile_destination_ops.py -v`

Expected: FAIL because destination-form ops and `out=` parameters are absent.

- [ ] **Step 3: Register destination ops and add Python lowering**

Share existing type-deduction logic, then compare its inferred result against the destination slot contract. Return a type cloned from the destination so allocation lineage is explicit.

- [ ] **Step 4: Write failing dual-planner codegen tests**

For the same program, assert PyPTO emits `pto.alloc_multi_tile addr = ...`, PTOAS emits `pto.alloc_multi_tile` without `addr`, both emit `pto.multi_tile_get`, and the producer `outs(...)` names the selected slot. Assert no per-slot `pto.alloc_tile` or Acc-to-Acc `pto.tmov` is emitted.

- [ ] **Step 5: Implement allocation, selection, release erasure, and destination codegen**

Register buffer-set operations in the common backend. Add codegen state mapping a set variable to its `multi_tile_buf` SSA value and bind selected-slot variables to `pto.multi_tile_get` results. Destination-form handlers reuse existing emitters with the selected output handle.

- [ ] **Step 6: Run focused codegen tests**

Run: `python -m pytest tests/ut/ir/operators/test_tile_destination_ops.py tests/ut/codegen/test_explicit_tile_buffers.py tests/ut/codegen/test_memory_planner_switch.py -v`

Expected: PASS.

- [ ] **Step 7: Commit the codegen slice**

```bash
git add src/ir/op/tile_ops/memory.cpp src/ir/op/tile_ops/transform.cpp src/ir/op/tile_ops/matmul.cpp python/pypto/ir/op/tile_ops.py python/pypto/language/op/unified_ops.py python/pypto/language/op/tile_ops.py src/backend/common/pto_ops_memory.cpp src/backend/common/pto_ops_datamove.cpp src/backend/common/pto_ops_elementwise.cpp src/backend/common/pto_ops_shared.cpp include/pypto/codegen/pto/pto_codegen.h src/codegen/pto/pto_codegen.cpp tests/ut/ir/operators/test_tile_destination_ops.py tests/ut/codegen/test_explicit_tile_buffers.py
git commit -m "feat(codegen): bind producers to explicit tile slots"
```

### Task 6: Slot lifetime, dependencies, and pipeline scheduling

**Files:**

- Modify: `include/pypto/ir/transforms/utils/stmt_dependency_analysis.h`
- Modify: `src/ir/transforms/utils/stmt_dependency_analysis.cpp`
- Modify: `src/ir/transforms/lower_pipeline_loops_pass.cpp`
- Modify: `src/ir/transforms/canonicalize_io_order_pass.cpp`
- Modify: `src/ir/transforms/simplify_pass.cpp`
- Modify: `include/pypto/ir/verifier/verifier.h`
- Create: `src/ir/verifier/verify_tile_buffer_lifetime.cpp`
- Modify: `src/ir/verifier/property_verifier_registry.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/ut/ir/transforms/test_lower_pipeline_loops.py`
- Modify: `tests/ut/ir/transforms/test_canonicalize_io_order.py`
- Create: `tests/ut/ir/verifier/test_tile_buffer_lifetime.py`

**Interfaces:**

- Consumes: selected-slot lineage and destination ops.
- Produces dependency key `(buffer-set allocation identity, normalized slot index)`.
- Produces verifier entry `VerifyTileBufferLifetime(const ProgramPtr&)`.

- [ ] **Step 1: Write failing lifetime verifier tests**

Test explicit release followed by use, release of a non-slot tile, and two overlapping destination writes without an ordering edge. Each test must assert a source-facing diagnostic.

- [ ] **Step 2: Run verifier tests and verify RED**

Run: `python -m pytest tests/ut/ir/verifier/test_tile_buffer_lifetime.py -v`

Expected: FAIL because the verifier is missing.

- [ ] **Step 3: Implement lease collection and release validation**

Track each `tile.buffer_slot` result and aliases returned by destination-form ops. End a lease at explicit release or last SSA use, reject use after release, and erase release markers before PTO codegen.

- [ ] **Step 4: Write failing nested-pipeline dependency tests**

Build the issue cadence with an L1 set indexed by `stack % 2`, a Right set indexed by `col % 2`, and an Acc set indexed by `col % 2`. Assert two extracts precede two matmuls, two matmuls precede two stores, and the next slot-zero reuse retains its WAW/WAR dependency.

- [ ] **Step 5: Implement normalized slot dependency analysis**

Preserve set identity across loop cloning. Simplify cloned modulo expressions; prove unequal constant residues independent, and conservatively alias whenever residue equality cannot be decided.

- [ ] **Step 6: Run lifetime and pipeline tests**

Run: `python -m pytest tests/ut/ir/verifier/test_tile_buffer_lifetime.py tests/ut/ir/transforms/test_lower_pipeline_loops.py tests/ut/ir/transforms/test_canonicalize_io_order.py -v`

Expected: PASS.

- [ ] **Step 7: Commit the scheduling slice**

```bash
git add include/pypto/ir/transforms/utils/stmt_dependency_analysis.h src/ir/transforms/utils/stmt_dependency_analysis.cpp src/ir/transforms/lower_pipeline_loops_pass.cpp src/ir/transforms/canonicalize_io_order_pass.cpp src/ir/transforms/simplify_pass.cpp include/pypto/ir/verifier/verifier.h src/ir/verifier/verify_tile_buffer_lifetime.cpp src/ir/verifier/property_verifier_registry.cpp CMakeLists.txt tests/ut/ir/transforms/test_lower_pipeline_loops.py tests/ut/ir/transforms/test_canonicalize_io_order.py tests/ut/ir/verifier/test_tile_buffer_lifetime.py
git commit -m "feat(passes): schedule explicit tile slot leases"
```

### Task 7: Documentation and final regression

**Files:**

- Modify: `docs/en/dev/ir/02-types.md`
- Modify: `docs/zh-cn/dev/ir/02-types.md`
- Modify: `docs/en/dev/ir/05-operators.md`
- Modify: `docs/zh-cn/dev/ir/05-operators.md`
- Modify: `docs/en/dev/language/00-python_syntax.md`
- Modify: `docs/zh-cn/dev/language/00-python_syntax.md`
- Modify: `docs/en/dev/passes/25-lower_pipeline_loops.md`
- Modify: `docs/zh-cn/dev/passes/25-lower_pipeline_loops.md`
- Modify: `docs/en/dev/passes/28-init_memref.md`
- Modify: `docs/zh-cn/dev/passes/28-init_memref.md`
- Modify: `docs/en/dev/passes/30-memory_reuse.md`
- Modify: `docs/zh-cn/dev/passes/30-memory_reuse.md`
- Create: `tests/ut/codegen/test_issue_2131_explicit_buffers.py`

**Interfaces:**

- Documents the final public API and its validation/lifetime contract.
- Verifies issue #2131 end-to-end with both planners.

- [ ] **Step 1: Add the end-to-end regression test and verify RED if a requirement remains uncovered**

The test compiles a two-level L1/L0 pipeline and checks exact counts for `pto.alloc_multi_tile`, dynamic `pto.multi_tile_get`, extracts, matmuls, stores, and absence of redundant allocations/moves.

- [ ] **Step 2: Complete synchronized English and Chinese documentation**

Document creation, dynamic indexing, `out=`, optional `tile.release`, validation boundaries, planner differences, and the nested pipeline example from the design.

- [ ] **Step 3: Run focused feature tests**

Run: `python -m pytest tests/ut/ir/core/test_tile_buffer_set_type.py tests/ut/ir/operators/test_tile_buffer_set_ops.py tests/ut/ir/operators/test_tile_destination_ops.py tests/ut/language/parser/test_tile_buffer_set.py tests/ut/ir/verifier/test_tile_buffer_lifetime.py tests/ut/codegen/test_explicit_tile_buffers.py tests/ut/codegen/test_issue_2131_explicit_buffers.py -v`

Expected: PASS.

- [ ] **Step 4: Build and run affected suites**

```bash
cmake --build build --parallel
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python -m pytest tests/ut/ir/core/ tests/ut/ir/operators/ tests/ut/ir/transforms/test_init_memref.py tests/ut/ir/transforms/test_memory_reuse.py tests/ut/ir/transforms/test_allocate_memory_addr_pass.py tests/ut/ir/transforms/test_lower_pipeline_loops.py tests/ut/ir/transforms/test_canonicalize_io_order.py tests/ut/codegen/test_explicit_tile_buffers.py tests/ut/codegen/test_issue_2131_explicit_buffers.py -n auto --maxprocesses 8 -v
```

Expected: build exit 0 and selected suites report zero failures.

- [ ] **Step 5: Run static checks**

```bash
ruff check .
ruff format --check .
pyright
pre-commit run --all-files
```

Expected: each command exits 0. Baseline-wide unrelated failures previously acknowledged by the user are reported separately and do not replace focused feature verification.

- [ ] **Step 6: Review the complete diff and commit**

```bash
git diff --check
git status --short
git add docs/en/dev docs/zh-cn/dev tests/ut/codegen/test_issue_2131_explicit_buffers.py
git commit -m "docs(ir): document explicit tile buffer slots"
```

- [ ] **Step 7: Run final verification from committed HEAD**

Repeat the focused feature-test command, `git diff --check HEAD^..HEAD`, and `git status --short`. Record exact pass/fail counts before claiming completion.
