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
- **Phase C** (optional): Codegen (orchestration, PTO, CCE) + system tests

## Task Tracking

Copy and track progress:

```text
Phase A — Tile Op (Required):
- [ ] A1: C++ tile op registration
- [ ] A2: Python IR wrapper
- [ ] A3: Python DSL wrapper
- [ ] A4: Unit tests (tile op)
- [ ] A5: Documentation update

Phase B — Tensor Op (Optional):
- [ ] B1: C++ tensor op registration
- [ ] B2: Python IR wrapper
- [ ] B3: Python DSL wrapper
- [ ] B4: Tensor-to-tile conversion registration
- [ ] B5: Unit tests (tensor op + conversion)
- [ ] B6: Documentation update

Phase C — Codegen (Optional):
- [ ] C1: Orchestration codegen (if tensor op exists)
- [ ] C2: PTO codegen registration
- [ ] C3: System tests
```

Ask the user which phases are needed before starting.

---

## Phase A: Tile Op

### A1: C++ Tile Op Registration

**File**: `src/ir/op/tile_ops/<category>.cpp` (pick the matching semantic group)

Categories: `elementwise.cpp`, `unary.cpp`, `reduction.cpp`, `matmul.cpp`,
`memory.cpp`, `transform.cpp`, `broadcast.cpp`, `cross_core.cpp`

If no existing file fits, create a new `.cpp` and add it to `CMakeLists.txt`
(around line 98–106 where `tile_ops/*.cpp` are listed).

**Pattern** — use `REGISTER_OP` fluent API:

```cpp
#include "pypto/ir/op_registry.h"

REGISTER_OP("tile.<op_name>")
    .set_op_category("TileOp")
    .set_description("<human-readable description>")
    .add_argument("<arg1>", "<arg1 description>")
    .add_argument("<arg2>", "<arg2 description>")
    // kwargs if needed:
    .set_attr<bool>("some_flag")
    // memory spaces (required for all TileOps):
    .set_input_memory(0, MemorySpace::Vec)
    .set_input_memory(1, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    // type deduction:
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      // Validate args, compute output shape/dtype, return TypePtr
      return std::make_shared<TileType>(output_shape, output_dtype);
    });
```

**Key rules**:

- **All TileOps must set memory spaces** — `ValidateTileOps()` checks at load time
- Memory spaces: `Vec`, `Left`, `Right`, `Acc`, `Unknown`
- Use shared deduction helpers when possible (e.g. `DeduceTileOpElementwiseBinaryType`)

### A2: Python IR Wrapper

**File**: `python/pypto/ir/op/tile_ops.py`

```python
def <op_name>(arg1: Expr, arg2: Expr, span: Span | None = None) -> Call:
    """<Description>.

    Args:
        arg1: <description>
        arg2: <description>
        span: Optional source span for debugging (auto-captured if not provided)

    Returns:
        Call expression for <op_name>
    """
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("tile.<op_name>", [arg1, arg2], {}, actual_span)
```

For ops with kwargs, pass a dict as the third argument to `create_op_call`.

### A3: Python DSL Wrapper

**File**: `python/pypto/language/op/tile_ops.py`

```python
def <op_name>(arg1: Tile, arg2: Tile) -> Tile:
    """<Description>.

    Args:
        arg1: <description>
        arg2: <description>

    Returns:
        Tile wrapping the <op_name> operation
    """
    call_expr = _ir_ops.<op_name>(arg1.unwrap(), arg2.unwrap())
    return Tile(expr=call_expr)
```

Also add `"<op_name>"` to the `__all__` list if one exists in the file.

### A4: Unit Tests

**Tile op test**: `tests/ut/ir/operators/test_tile_ops.py`

```python
def test_tile_<op_name>(self):
    """Test tile.<op_name> operator."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a: pl.Tensor[[M, N], pl.FP32],
            b: pl.Tensor[[M, N], pl.FP32],
            output: pl.Tensor[[M, N], pl.FP32],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            tile_a: pl.Tile[[m, n], pl.FP32] = pl.load(a, [0, 0], [m, n])
            tile_b: pl.Tile[[m, n], pl.FP32] = pl.load(b, [0, 0], [m, n])
            tile_c: pl.Tile[[m, n], pl.FP32] = pl.tile.<op_name>(tile_a, tile_b)
            result: pl.Tensor[[M, N], pl.FP32] = pl.store(tile_c, [0, 0], output)
            return result

    ir_str = str(Program)
    assert "tile.<op_name>" in ir_str
```

Test edge cases: shape mismatches, dtype combinations, dynamic dims.

### A5: Documentation

Update `docs/en/dev/ir/05-operators.md` — add the new op to the appropriate table.
Keep `docs/zh-cn/dev/ir/05-operators.md` in sync.

For detailed file paths and code templates, see [reference.md](reference.md).

---

## Phase B: Tensor Op + Conversion

### B1: C++ Tensor Op Registration

**File**: `src/ir/op/tensor_ops/<category>.cpp`

Same `REGISTER_OP` pattern as tile ops, but:

- Use `.set_op_category("TensorOp")`
- **No memory spaces** (tensors live in DDR)
- Type deduction returns `TensorType` with broadcasting support

### B2–B3: Python IR + DSL Wrappers

Same pattern as A2/A3 but in:

- **IR**: `python/pypto/ir/op/tensor_ops.py`
- **DSL**: `python/pypto/language/op/tensor_ops.py`

Use `Tensor` instead of `Tile`, `TensorType` instead of `TileType`.

### B4: Tensor-to-Tile Conversion

**File**: `src/ir/transforms/op_conversion_registry.cpp`

Register in the `OpConversionRegistry` constructor (around line 150+):

**Simple 1:1 mapping** (most common):

```cpp
RegisterSimple("tensor.<op_name>", "tile.<op_name>");
```

**Custom conversion** (when extra logic is needed):

```cpp
RegisterCustom("tensor.<op_name>",
    [](const std::vector<ExprPtr>& args,
       const std::vector<std::pair<std::string, std::any>>& kwargs,
       const Span& span) -> ConversionResult {
  auto& reg = OpRegistry::GetInstance();
  // Custom logic: broadcast detection, prologue stmts, etc.
  return ConversionResult{reg.Create("tile.<op_name>", args, span)};
});
```

**ConversionResult** can include a `prologue` (vector of statements inserted before).

### B5: Unit Tests

**Tensor op test**: `tests/ut/ir/operators/test_tensor_ops.py`

```python
def test_tensor_<op_name>():
    call = ir.op.tensor.<op_name>(args...)
    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.<op_name>"
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
```

**Conversion test**: `tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py`

Uses Before/Expected pattern with `ir.assert_structural_equal`:

```python
def test_<op_name>_conversion(self):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(self, x: pl.Tensor[...]) -> pl.Tensor[...]:
            y = pl.<op_name>(x, ...)
            return y
        @pl.function
        def main(self, x: pl.Tensor[...]) -> pl.Tensor[...]:
            y = self.main_incore_0(x)
            return y

    @pl.program
    class Expected:
        # tile.load + tile.<op_name> + tile.store pattern
        ...

    After = passes.convert_tensor_to_tile_ops()(Before)
    ir.assert_structural_equal(After, Expected)
```

### B6: Documentation

Update `docs/en/dev/ir/05-operators.md` with the tensor op entry.
Update `docs/zh-cn/dev/ir/05-operators.md` in sync.

---

## Phase C: Codegen

### C1: Orchestration Codegen (Tensor Op on Host)

**File**: `src/codegen/tensor_op_codegen.cpp`

Only needed if the tensor op can appear in orchestration (host-side) code.

```cpp
REGISTER_ORCHESTRATION_OP(tensor_<op_name>, ("tensor.<op_name>")) {
  std::string result_var = codegen.GetCurrentResultTarget();
  std::ostringstream oss;
  // Generate host-side C++ code
  oss << "Tensor " << result_var << " = ...;";
  return oss.str();
}
```

### C2: PTO Codegen (Tile Op on Device)

**File**: `src/backend/common/pto_ops_common.cpp`

**Simple N-ary op** — add to `kSimpleOps` table:

```cpp
{"tile.<op_name>", "pto.<pto_instruction>", <arity>},
```

**Custom codegen** — register with `backend.RegisterOp`:

```cpp
backend.RegisterOp("tile.<op_name>").f_codegen(
    [](const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  // Custom PTO MLIR generation
  codegen.Emit("pto.<instruction> " + GenerateInsOutsClause(op, codegen));
  return "";
});
```

Also check if 910B backend needs special handling:
`src/backend/910B_PTO/backend_910b_pto_ops.cpp`

For CCE backend: `src/backend/910B_CCE/backend_910b_cce_ops.cpp`

### C3: System Tests

**File**: `tests/st/codegen/test_<op_name>_codegen.py` (or add to existing file)

System tests require hardware/environment. Follow patterns in:

- `tests/st/codegen/test_add_mul_orch_cce_codegen.py`

**Codegen unit tests** (no hardware needed):
`tests/ut/codegen/test_pto_codegen_ops.py`

```python
def test_<op_name>_pto_codegen(self):
    # Build IR program with tile.<op_name>
    # Run PTOCodegen
    # Assert output contains expected PTO instruction
```

---

## Quick Reference — File Locations

| Layer | Tile Op | Tensor Op |
| ----- | ------- | --------- |
| C++ registration | `src/ir/op/tile_ops/*.cpp` | `src/ir/op/tensor_ops/*.cpp` |
| Python IR | `python/pypto/ir/op/tile_ops.py` | `python/pypto/ir/op/tensor_ops.py` |
| Python DSL | `python/pypto/language/op/tile_ops.py` | `python/pypto/language/op/tensor_ops.py` |
| Conversion | — | `src/ir/transforms/op_conversion_registry.cpp` |
| PTO codegen | `src/backend/common/pto_ops_common.cpp` | — |
| Orchestration codegen | — | `src/codegen/tensor_op_codegen.cpp` |
| Tile op UT | `tests/ut/ir/operators/test_tile_ops.py` | — |
| Tensor op UT | — | `tests/ut/ir/operators/test_tensor_ops.py` |
| Conversion UT | — | `tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py` |
| Codegen UT | `tests/ut/codegen/test_pto_codegen_ops.py` | `tests/ut/codegen/test_orchestration_codegen.py` |
| ST | `tests/st/codegen/` | `tests/st/codegen/` |
| Docs (en) | `docs/en/dev/ir/05-operators.md` | `docs/en/dev/ir/05-operators.md` |
| Docs (zh-cn) | `docs/zh-cn/dev/ir/05-operators.md` | `docs/zh-cn/dev/ir/05-operators.md` |
| Codegen docs | `docs/en/dev/codegen/00-pto_codegen.md` | `docs/en/dev/codegen/02-orchestration_codegen.md` |
| CMake | `CMakeLists.txt` (line ~98–116) | `CMakeLists.txt` (line ~109–116) |

For complete code templates and detailed examples, see [reference.md](reference.md).

---

## Checklist Before Commit

- [ ] C++ op registered with `REGISTER_OP` and correct category/memory spaces
- [ ] New `.cpp` added to `CMakeLists.txt` if created
- [ ] Python IR wrapper in `ir/op/{tile,tensor}_ops.py`
- [ ] Python DSL wrapper in `language/op/{tile,tensor}_ops.py`
- [ ] Conversion registered in `op_conversion_registry.cpp` (if Phase B)
- [ ] Codegen registered in backend (if Phase C)
- [ ] Unit tests pass: `python3 -m pytest tests/ut/ir/operators/ -v -k <op_name>`
- [ ] Conversion tests pass (if Phase B)
- [ ] Codegen tests pass (if Phase C)
- [ ] `docs/en/dev/ir/05-operators.md` updated
- [ ] `docs/zh-cn/dev/ir/05-operators.md` updated in sync
- [ ] `pre-commit run --all-files` passes
