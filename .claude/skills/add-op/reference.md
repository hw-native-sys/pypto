# Add Op — Detailed Reference

Complete file paths, code templates, and conventions for adding operators.

## 1. C++ Op Registration

### Common Includes

```cpp
#include "pypto/ir/op_registry.h"     // REGISTER_OP, OpRegistry
#include "pypto/ir/type.h"            // TileType, TensorType, ScalarType, TileView
#include "pypto/core/common.h"        // DataType, MemorySpace, kDynamicDim
#include "pypto/ir/expr.h"            // ExprPtr, CallPtr
```

### Tile Op — Memory Space Reference

| MemorySpace | Usage |
| ----------- | ----- |
| `Vec` | General-purpose vector buffer (elementwise, unary) |
| `Left` | Left operand of matmul |
| `Right` | Right operand of matmul |
| `Acc` | Accumulator (matmul output) |
| `Unknown` | Memory space resolved by later pass |

### Shared Type Deduction Helpers

Located in the same `tile_ops/*.cpp` or `tensor_ops/*.cpp` files:

- `DeduceTileOpElementwiseBinaryType(args, kwargs, op_name)` — binary same-shape
- `DeduceTileOpElementwiseUnaryType(args, kwargs, op_name)` — unary
- `DeduceTensorOpElementwiseBinaryType(args, kwargs, op_name)` — tensor binary with broadcasting
- `DeduceTileMatMulType(args, kwargs, op_name)` — matmul shape rules

### CMakeLists.txt — Adding New Source Files

Location: root `CMakeLists.txt`

Tile ops are listed around line 98–106:

```cmake
src/ir/op/tile_ops/elementwise.cpp
src/ir/op/tile_ops/matmul.cpp
# ... add your new file here
src/ir/op/tile_ops/<new_category>.cpp
```

Tensor ops around line 109–116:

```cmake
src/ir/op/tensor_ops/elementwise.cpp
src/ir/op/tensor_ops/matmul.cpp
# ... add your new file here
src/ir/op/tensor_ops/<new_category>.cpp
```

---

## 2. Python IR Layer

### File: `python/pypto/ir/op/tile_ops.py`

Key imports at the top:

```python
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, Expr, Span
```

Helper used for span capture:

```python
actual_span = _get_span_or_capture(span)
```

Creating an op call:

```python
_ir_core.create_op_call("tile.<op_name>", [arg1, arg2], {}, actual_span)
# With kwargs:
_ir_core.create_op_call("tile.<op_name>", [arg1], {"flag": True}, actual_span)
```

### File: `python/pypto/ir/op/tensor_ops.py`

Same pattern, but uses `"tensor.<op_name>"`.

For scalar variants (e.g. `tensor.adds` for tensor+scalar):

```python
def add(lhs: Expr, rhs: int | float | Expr, span: Span | None = None) -> Call:
    actual_span = _get_span_or_capture(span)
    rhs_expr = (
        _normalize_expr(rhs, actual_span, int_dtype=DataType.FP32, float_dtype=DataType.FP32)
        if not isinstance(rhs, Expr)
        else rhs
    )
    rhs_type = rhs_expr.type
    if isinstance(rhs_type, ScalarType):
        return _ir_core.create_op_call("tensor.adds", [lhs, rhs_expr], {}, actual_span)
    else:
        return _ir_core.create_op_call("tensor.add", [lhs, rhs_expr], {}, actual_span)
```

---

## 3. Python DSL Layer

### File: `python/pypto/language/op/tile_ops.py`

Key imports:

```python
from pypto.ir.op import tile_ops as _ir_ops
from pypto.language.typing.tile import Tile
```

Pattern: unwrap `Tile` → call IR function → wrap result:

```python
def <op_name>(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_ops.<op_name>(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)
```

### File: `python/pypto/language/op/tensor_ops.py`

Key imports:

```python
from pypto.ir.op import tensor_ops as _ir_ops
from pypto.language.typing.tensor import Tensor
from pypto.language.typing.scalar import Scalar
```

Helper for mixed tensor/scalar RHS:

```python
def _unwrap_rhs(rhs: int | float | Tensor | Scalar) -> Expr:
    ...
```

---

## 4. Tensor-to-Tile Conversion

### File: `src/ir/transforms/op_conversion_registry.cpp`

### Header: `include/pypto/ir/transforms/op_conversion_registry.h`

### Registration Location

All conversions are registered in the `OpConversionRegistry` constructor, around
line 150–300 of `op_conversion_registry.cpp`.

### ConversionResult Structure

```cpp
struct ConversionResult {
  CallPtr result;                              // The converted call
  std::vector<StmtPtr> prologue = {};          // Statements to insert before
};
```

### Pattern: Simple 1:1

```cpp
RegisterSimple("tensor.neg", "tile.neg");
RegisterSimple("tensor.adds", "tile.adds");
```

### Pattern: Broadcast-Aware Binary

```cpp
auto MakeBroadcastBinaryConv = [](const std::string& tile_op,
                                  const std::string& row_expand_op) -> ConversionFunc {
  return [tile_op, row_expand_op](...) -> ConversionResult {
    auto [wider, narrower] = DetectRowBroadcast(args);
    if (wider >= 0) {
      return ConversionResult{op_reg.Create(row_expand_op, {args[wider], args[narrower]}, span)};
    }
    return ConversionResult{op_reg.Create(tile_op, args, span)};
  };
};
RegisterCustom("tensor.add", MakeBroadcastBinaryConv("tile.add", "tile.row_expand_add"));
```

### Pattern: Complex with Prologue

```cpp
RegisterCustom("tensor.matmul", [](const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const Span& span) -> ConversionResult {
  // Create prologue statements (e.g. transpose, cast)
  std::vector<StmtPtr> prologue;
  // ...
  auto result = reg.Create("tile.matmul", new_args, span);
  return ConversionResult{result, prologue};
});
```

---

## 5. PTO Codegen

### File: `src/backend/common/pto_ops_common.cpp`

### Simple Ops Table (`kSimpleOps`)

```cpp
static const SimpleOpEntry kSimpleOps[] = {
    {"tile.add",    "pto.tadd",    2},
    {"tile.sub",    "pto.tsub",    2},
    {"tile.mul",    "pto.tmul",    2},
    {"tile.neg",    "pto.tneg",    1},
    // Add new op here:
    {"tile.<op_name>", "pto.<pto_instr>", <arity>},
};
```

All simple ops are batch-registered by `RegisterPTOOps()`:

```cpp
void RegisterPTOOps(Backend& backend, const std::unordered_set<std::string>& exclude_ops) {
  for (const auto& entry : kSimpleOps) {
    if (exclude_ops.count(entry.op_name) > 0) continue;
    auto reg_entry = backend.RegisterOp(entry.op_name);
    reg_entry.f_codegen([pto_op, arity](const CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeNaryCodegenPTO(pto_op, arity, op, codegen);
    });
  }
  // Custom registrations follow...
}
```

### Custom PTO Codegen

For ops that need more than simple ins/outs:

```cpp
backend.RegisterOp("tile.<op_name>").f_codegen(
    [](const CallPtr& op, codegen::CodegenBase& codegen_base) -> std::string {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  // Use codegen helpers:
  // codegen.GetExprAsCode(expr) — get SSA name for an expression
  // codegen.NewTemp() — generate a new SSA temp name
  // codegen.Emit(line) — emit a line of PTO MLIR
  // GenerateInsOutsClause(op, codegen) — standard ins(...)/outs(...) clause
  codegen.Emit("pto.<instruction> " + GenerateInsOutsClause(op, codegen));
  return "";
});
```

### Backend-Specific Files

| Backend | Ops Registration | Codegen |
| ------- | ---------------- | ------- |
| 910B PTO | `src/backend/910B_PTO/backend_910b_pto_ops.cpp` | `src/codegen/pto/pto_codegen.cpp` |
| 910B CCE | `src/backend/910B_CCE/backend_910b_cce_ops.cpp` | `src/codegen/cce/cce_codegen.cpp` |

---

## 6. Orchestration Codegen

### File: `src/codegen/tensor_op_codegen.cpp`

### Header: `include/pypto/codegen/orchestration_op_registry.h`

```cpp
REGISTER_ORCHESTRATION_OP(tensor_<op_name>, ("tensor.<op_name>")) {
  auto result_type = As<TensorType>(op->GetType());
  std::string result_var = codegen.GetCurrentResultTarget();

  std::ostringstream oss;
  // codegen.GenerateExprString(expr) — expression to C++ string
  // codegen.GetRuntimeDataTypeString(dtype) — DataType::FP32 → "DataType::FP32"
  oss << "Tensor " << result_var << " = runtime_<op_name>(...);";
  return oss.str();
}
```

---

## 7. Unit Test Patterns

### Tile Op Test (`tests/ut/ir/operators/test_tile_ops.py`)

Uses `@pl.program` + IR string assertion:

```python
class TestTile<Category>Ops:
    def test_tile_<op_name>(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, a: pl.Tensor[[128, 128], pl.FP32], ...) -> ...:
                tile_a = pl.load(a, [0, 0], [32, 32])
                result = pl.tile.<op_name>(tile_a, ...)
                ...
        ir_str = str(Program)
        assert "tile.<op_name>" in ir_str
```

### Tensor Op Test (`tests/ut/ir/operators/test_tensor_ops.py`)

Uses direct IR API + type assertions:

```python
def test_tensor_<op_name>():
    call = ir.op.tensor.<op_name>(arg1, arg2)
    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.<op_name>"
    assert isinstance(call.type, ir.TensorType)
```

### Conversion Test (`tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py`)

Uses Before/Expected structural equality:

```python
def test_<op_name>_conversion(self):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y = pl.<op_name>(x, x)
            return y
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return self.main_incore_0(x)

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(self, x_tile: pl.Tile[[64], pl.FP32]) -> pl.Tile[[64], pl.FP32]:
            result = pl.tile.<op_name>(x_tile, x_tile)
            return result
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return self.main_incore_0(x)

    After = passes.convert_tensor_to_tile_ops()(Before)
    ir.assert_structural_equal(After, Expected)
```

### Codegen Unit Test (`tests/ut/codegen/test_pto_codegen_ops.py`)

Tests PTO output string:

```python
def test_<op_name>_pto_codegen(self):
    # Build IR with tile.<op_name>
    # Generate PTO code
    # Assert "pto.<instruction>" in output
```

---

## 8. Documentation

### Operator Docs

- English: `docs/en/dev/ir/05-operators.md`
- Chinese: `docs/zh-cn/dev/ir/05-operators.md`

Add to the appropriate category table (TensorOp or TileOp section).

### Codegen Docs

- PTO: `docs/en/dev/codegen/00-pto_codegen.md`
- CCE: `docs/en/dev/codegen/01-cce_codegen.md`
- Orchestration: `docs/en/dev/codegen/02-orchestration_codegen.md`

### Pass Docs (if conversion changes are significant)

- Pass manager: `docs/en/dev/passes/00-pass_manager.md`

### Translation Rules

- File names stay in English
- Code examples are NOT translated
- Technical terms: dual notation on first mention (e.g. "type deduction" in both languages)
- English is authoritative; Chinese must match

---

## 9. Naming Conventions

| Item | Convention | Example |
| ---- | ---------- | ------- |
| IR op name | `{category}.{snake_case}` | `tile.row_expand_add` |
| Python IR function | `snake_case` | `def row_expand_add(...)` |
| Python DSL function | `snake_case` | `def row_expand_add(...)` |
| C++ deduction helper | `PascalCase` | `DeduceTileRowExpandAddType` |
| Test function | `test_{category}_{op}` | `test_tile_row_expand_add` |
| PTO instruction | `pto.t{op}` | `pto.tadd` |
| CMake source | `src/ir/op/{category}_ops/{group}.cpp` | `src/ir/op/tile_ops/elementwise.cpp` |

---

## 10. Build & Test Commands

```bash
# Build after C++ changes
cmake --build build --parallel

# Run specific op tests
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python3 -m pytest tests/ut/ir/operators/test_tile_ops.py -v -k "<op_name>"
python3 -m pytest tests/ut/ir/operators/test_tensor_ops.py -v -k "<op_name>"
python3 -m pytest tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py -v -k "<op_name>"
python3 -m pytest tests/ut/codegen/test_pto_codegen_ops.py -v -k "<op_name>"

# Full test suite
python3 -m pytest tests/ut/ -n auto --maxprocesses 8 -v

# Pre-commit checks
pre-commit run --all-files
```
