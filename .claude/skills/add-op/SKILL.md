---
name: add-op
description: >-
  Add new operator definitions to PyPTO across all layers (C++, Python IR, Python
  DSL, tests, codegen, docs). Covers tile ops, tensor ops, tensor-to-tile
  conversion, and codegen registration. Use when the user asks to add a new op,
  define a new operator, implement a new tile/tensor operation, or extend the
  operator system.
---

# Add New Operator to PyPTO

## Overview

Adding a new op follows a layered workflow with three phases:

- **Phase A** (required): Tile op definition + tests + docs
- **Phase B** (optional): Tensor op + tensor-to-tile conversion + tests + docs
- **Phase C** (optional): Codegen (orchestration, PTO) + system tests

Ask the user which phases are needed before starting.

This file covers the sequence and the decisions at each step. [reference.md](reference.md)
holds what to type: code templates per layer (§1–§7), naming conventions (§9), build
and test commands (§10), and the full file-location table for every layer (§11).

## Task Tracking

Copy and track progress:

```text
A (required): [ ] A1 C++ tile op    [ ] A2 IR wrapper   [ ] A3 DSL wrapper
              [ ] A4 unit tests     [ ] A5 docs
B (optional): [ ] B1 C++ tensor op  [ ] B2 IR wrapper   [ ] B3 DSL wrapper
              [ ] B4 conversion     [ ] B5 unit tests   [ ] B6 docs
C (optional): [ ] C1 orch codegen   [ ] C2 PTO codegen  [ ] C3 system tests
```

## Phase A: Tile Op

### A1: C++ Tile Op Registration

**File**: `src/ir/op/tile_ops/<category>.cpp` (pick the matching semantic group)

Categories: `elementwise.cpp`, `unary.cpp`, `reduction.cpp`, `matmul.cpp`,
`memory.cpp`, `transform.cpp`, `broadcast.cpp`, `cross_core.cpp`

If no existing file fits, create a new `.cpp` and add it to `CMakeLists.txt`
(around line 98–106 where `tile_ops/*.cpp` are listed).

Use the `REGISTER_OP` fluent API — template in [reference.md §1](reference.md).

**Key rules**:

- **All TileOps must set memory spaces** — `ValidateTileOps()` checks at load time
- Memory spaces: `Vec`, `Left`, `Right`, `Acc`, `Unknown`
- Use shared deduction helpers when possible (e.g. `DeduceTileOpElementwiseBinaryType`)

#### Execution-memory access and pipe metadata

Memory-space constraints describe where values live; they do not say whether an
operator actually reads or writes those allocations when the kernel executes.
Every new TileOp must classify that separately:

- `.functional_execution_memory_access()` — the emitted PTO operation physically
  reads its tile operands and/or writes its tile results.
- `.no_execution_memory_access()` — declarations and metadata-only views that
  emit no physical access.
- Default `Unknown` — only for an intentionally unmodeled effect such as a
  partial/subrange or destination-passing access. Explain the omission in a
  comment and add a test proving the conservative consumer behavior.

The DSA reuse-hazard recognizer also needs the exact execution pipe. If a new
operator's source/destination memory route does not determine a unique pipe in
`Backend::TryInferPipe`, register backend-specific `.f_infer_pipe(...)` metadata
together with PTO codegen. Do not duplicate architecture route tables inside an
IR transform. Add a direct recognizer test when the new op should create or
suppress a reuse-penalty edge.

### A2: Python IR Wrapper

**File**: `python/pypto/ir/op/tile_ops.py`

Thin wrapper returning `_ir_core.create_op_call("tile.<op_name>", ...)`, with a Google-style
docstring and a `span: Span | None = None` parameter. Ops with kwargs pass a dict as the
third argument. Template: [reference.md §2](reference.md).

### A3: Python DSL Wrapper

**File**: `python/pypto/language/op/tile_ops.py`

Unwrap the `Tile` args → call the IR function → wrap the result in `Tile`. Also add
`"<op_name>"` to `__all__` if the file has one. Template: [reference.md §3](reference.md).

### A4: Unit Tests

**File**: `tests/ut/ir/operators/test_tile_ops.py`

Build a `@pl.program` InCore function using `pl.load` → `pl.tile.<op_name>` → `pl.store`,
then assert `"tile.<op_name>"` appears in `str(Program)`. Test edge cases: shape mismatches,
dtype combinations, dynamic dims. Template: [reference.md §7](reference.md).

### A5: Documentation

Add the op to the appropriate table in `docs/en/dev/ir/05-operators.md`, and keep
`docs/zh/dev/ir/05-operators.md` in sync.

## Phase B: Tensor Op + Conversion

### B1: C++ Tensor Op Registration

**File**: `src/ir/op/tensor_ops/<category>.cpp`

Same `REGISTER_OP` pattern as A1, but:

- Use `.set_op_category("TensorOp")`
- **No memory spaces** (tensors live in DDR)
- Type deduction returns `TensorType` with broadcasting support

### B2–B3: Python IR + DSL Wrappers

Same pattern as A2/A3, in `python/pypto/ir/op/tensor_ops.py` and
`python/pypto/language/op/tensor_ops.py`. Use `Tensor` instead of `Tile`,
`TensorType` instead of `TileType`. Scalar variants (e.g. `tensor.adds`) dispatch
on the RHS type — see [reference.md §2](reference.md).

### B4: Tensor-to-Tile Conversion

**File**: `src/ir/transforms/op_conversion_registry.cpp` — register in the
`OpConversionRegistry` constructor (around line 150+).

| Case | Use |
| ---- | --- |
| 1:1 name mapping (most common) | `RegisterSimple("tensor.<op>", "tile.<op>")` |
| Broadcast handling, prologue statements, extra logic | `RegisterCustom(...)` returning a `ConversionResult` |

A `ConversionResult` may carry a `prologue` — statements inserted before the
converted op. Patterns for both forms: [reference.md §4](reference.md).

### B5: Unit Tests

- **Tensor op** — `tests/ut/ir/operators/test_tensor_ops.py`: assert the `Call`
  op name and that `call.type` is a `TensorType`.
- **Conversion** — `tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py`:
  Before/Expected `@pl.program` pair compared with `ir.assert_structural_equal`,
  where Expected spells out the `tile.load` + `tile.<op>` + `tile.store` pattern.

Templates: [reference.md §7](reference.md).

### B6: Documentation

Add the tensor op entry to the same `05-operators.md` tables (en + zh).

## Phase C: Codegen

### C1: Orchestration Codegen (Tensor Op on Host)

**File**: `src/codegen/tensor_op_codegen.cpp`

Only needed if the tensor op can appear in orchestration (host-side) code.
Register with `REGISTER_ORCHESTRATION_OP` — see [reference.md §6](reference.md).

### C2: PTO Codegen (Tile Op on Device)

**File**: `src/backend/common/pto_ops_common.cpp`

| Case | Use |
| ---- | --- |
| Simple N-ary op mapping to one PTO instruction | Add a row to the `kSimpleOps` table |
| Anything else | `backend.RegisterOp("tile.<op>").f_codegen(...)` |

Also check whether the 910B backend needs special handling in
`src/backend/910B/backend_910b_ops.cpp`. Templates: [reference.md §5](reference.md).

### C3: System Tests

- **Codegen UT** (no hardware) — `tests/ut/codegen/test_pto_codegen_ops.py`: build
  IR with `tile.<op_name>`, run `PTOCodegen`, assert the expected PTO instruction.
- **ST** (needs hardware) — `tests/st/codegen/`, following
  `tests/st/codegen/dsl/test_add_mul_orch_codegen.py`.

## Checklist Before Commit

- [ ] C++ op registered with `REGISTER_OP` and correct category/memory spaces
- [ ] Execution-memory access is marked functional/no-access (or Unknown is justified and tested)
- [ ] Exact backend pipe inference exists when the source/destination route is not uniquely inferable
- [ ] New `.cpp` added to `CMakeLists.txt` if created
- [ ] Python IR wrapper in `ir/op/{tile,tensor}_ops.py`
- [ ] Python DSL wrapper in `language/op/{tile,tensor}_ops.py`
- [ ] Conversion registered in `op_conversion_registry.cpp` (if Phase B)
- [ ] Codegen registered in backend (if Phase C)
- [ ] Unit tests pass: `python3 -m pytest tests/ut/ir/operators/ -v -k <op_name>`
- [ ] Conversion tests pass (if Phase B)
- [ ] Codegen tests pass (if Phase C)
- [ ] `docs/en/dev/ir/05-operators.md` updated
- [ ] `docs/zh/dev/ir/05-operators.md` updated in sync
- [ ] `pre-commit run --all-files` passes
