# IR Builder

The IR Builder provides a convenient API for constructing PyPTO IR incrementally using context managers (Python) or Begin/End patterns (C++). It manages context stacks, validates construction, and tracks source locations.

## Overview

### Key Features

- **Context Management**: Stack-based tracking ensures proper nesting
- **Automatic Span Tracking** (Python): Uses `inspect` module
- **Explicit Span Parameters** (C++): All methods accept spans
- **Validation**: Checks proper context usage and structure
- **Nested Constructs**: Supports loops in functions, if in loops, etc.

## Python API

Uses context managers (`with` statements) for clean interface.

### Basic Function

```python
from pypto import ir, DataType
from pypto.ir import IRBuilder

ib = IRBuilder()

with ib.function("add") as f:
    x = f.param("x", ir.ScalarType(DataType.INT64))
    y = f.param("y", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))

    result = ib.var("result", ir.ScalarType(DataType.INT64))
    add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
    ib.assign(result, add_expr)

func = f.get_result()

# With function type
with ib.function("orchestrator", type=ir.FunctionType.Orchestration) as f:
    n = f.param("n", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))
    # ... function body

func_orch = f.get_result()
```

### For Loops with Iteration Arguments

```python
with ib.function("sum_to_n") as f:
    n = f.param("n", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))

    i = ib.var("i", ir.ScalarType(DataType.INT64))
    start = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
    step = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
    init_val = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())

    with ib.for_loop(i, start, n, step) as loop:
        sum_iter = loop.iter_arg("sum", init_val)
        sum_final = loop.return_var("sum_final")

        add_expr = ir.Add(sum_iter, i, DataType.INT64, ir.Span.unknown())
        ib.emit(ir.YieldStmt([add_expr], ir.Span.unknown()))

    result = loop.output()  # Get first return variable

func = f.get_result()
```

### If Statements

```python
with ib.function("max") as f:
    x = f.param("x", ir.ScalarType(DataType.INT64))
    y = f.param("y", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))

    condition = ir.Gt(x, y, DataType.INT64, ir.Span.unknown())

    with ib.if_stmt(condition) as if_builder:
        if_builder.return_var("phi_result", ir.ScalarType(DataType.INT64))
        ib.emit(ir.YieldStmt([x], ir.Span.unknown()))

        if_builder.else_()
        ib.emit(ir.YieldStmt([y], ir.Span.unknown()))

    result = if_builder.output()

func = f.get_result()
```

### Accessing Return Variables

Both `ForLoopBuilder` and `IfStmtBuilder` provide convenient access methods:

| Method | Description | Example |
|--------|-------------|---------|
| `output(index=0)` | Get single return variable by index | `result = loop.output()` |
| `outputs()` | Get all return variables as list | `sum_result, prod_result = loop.outputs()` |

### Return Statements

```python
with ib.function("add_and_return") as f:
    x = f.param("x", ir.ScalarType(DataType.INT64))
    y = f.param("y", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))

    add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
    ib.return_stmt(add_expr)  # Single value
    # ib.return_stmt([x, y])  # Multiple values
    # ib.return_stmt()        # Empty return

func = f.get_result()
```

## C++ API

Uses Begin/End methods with explicit span parameters.

### Example: Function with For Loop

```cpp
#include "pypto/ir/builder.h"
using namespace pypto::ir;

IRBuilder ib;
auto here = [](int line) { return Span(__FILE__, line, 0); };

// Begin function (with optional type parameter)
ib.BeginFunction("sum_to_n", here(__LINE__), FunctionType::Opaque);
auto n = ib.FuncArg("n", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__));
ib.ReturnType(std::make_shared<ScalarType>(DataType::INT64));

// Begin for loop
auto i = ib.Var("i", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__));
auto start = std::make_shared<ConstInt>(0, DataType::INT64, here(__LINE__));
auto step = std::make_shared<ConstInt>(1, DataType::INT64, here(__LINE__));
ib.BeginForLoop(i, start, n, step, here(__LINE__));

// Add iter_arg and return_var
auto init_val = std::make_shared<ConstInt>(0, DataType::INT64, here(__LINE__));
auto sum_iter = std::make_shared<IterArg>("sum", std::make_shared<ScalarType>(DataType::INT64),
                                          init_val, here(__LINE__));
ib.AddIterArg(sum_iter);
ib.AddReturnVar(ib.Var("sum_final", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__)));

// Loop body
auto add_expr = std::make_shared<Add>(sum_iter, i, DataType::INT64, here(__LINE__));
ib.Emit(std::make_shared<YieldStmt>(std::vector<ExprPtr>{add_expr}, here(__LINE__)));

// End loop and function
ib.EndForLoop(here(__LINE__));
ib.Return(std::vector<ExprPtr>{...}, here(__LINE__));
auto func = ib.EndFunction(here(__LINE__));
```

**Key differences from Python API:**
- Use `BeginFunction`/`EndFunction` instead of `with` statement
- Use `BeginForLoop`/`EndForLoop` for loops
- Use `BeginIf`/`EndIf` for if statements
- All methods require explicit `Span` parameter

## Context Stack and Validation

### Validation Rules

| Rule | Description |
|------|-------------|
| **No nested functions** | Cannot call `BeginFunction` inside another function |
| **Context matching** | Must end contexts with correct End method |
| **Iter args match return vars** | For loops must have equal numbers |
| **Proper nesting** | Loops/if must be inside function or valid context |

### Error Messages

```python
with ib.function("outer") as f:
    with ib.function("inner") as f2:  # Error!
        pass
# RuntimeError: Cannot begin function 'inner': already inside function 'outer' at file.py:10
```

### Context State Queries

```python
ib.in_function()  # True if inside function context
ib.in_loop()      # True if inside for loop context
ib.in_if()        # True if inside if statement context
```

```cpp
ib.InFunction()  // true if inside function
ib.InLoop()      // true if inside loop
ib.InIf()        // true if inside if
```

## Type Creation Helpers (Python)

Convenient methods for creating types with memory references and tile views.

### MemRef

```python
# Create memory reference
memref = ib.memref(
    memory_space=ir.MemorySpace.DDR,
    addr=0x1000,  # Can be int or Expr
    size=1024
)

# With symbolic address
base_addr = ib.var("base_addr", ir.ScalarType(DataType.INT64))
memref = ib.memref(ir.MemorySpace.UB, base_addr, 2048)
```

### TileView

```python
# Integer dimensions
tile_view = ib.tile_view(
    valid_shape=[16, 16],
    stride=[1, 16],
    start_offset=0
)

# Symbolic dimensions
n = ib.var("n", ir.ScalarType(DataType.INT64))
tile_view = ib.tile_view([n, n], [1, n], 0)
```

### TensorType and TileType

```python
# Simple types
tensor_t = ib.tensor_type([64, 128], DataType.FP32)
tile_t = ib.tile_type([16, 16], DataType.FP16)

# With memory reference
memref = ib.memref(ir.MemorySpace.DDR, 0x1000, 8192)
tensor_t = ib.tensor_type([64, 128], DataType.FP32, memref=memref)

# Complete tile with memref and tile_view
memref = ib.memref(ir.MemorySpace.L0A, 0, 512)
tile_view = ib.tile_view([16, 16], [1, 16], 0)
tile_t = ib.tile_type([16, 16], DataType.FP16, memref=memref, tile_view=tile_view)
```

### Complete Example

```python
ib = IRBuilder()

with ib.function("matmul_tile") as f:
    # Create tile types with memory references
    memref_a = ib.memref(ir.MemorySpace.L0A, 0, 512)
    tile_t_a = ib.tile_type([16, 16], DataType.FP16, memref=memref_a)

    memref_b = ib.memref(ir.MemorySpace.L0B, 0, 512)
    tile_t_b = ib.tile_type([16, 16], DataType.FP16, memref=memref_b)

    a = f.param("a", tile_t_a)
    b = f.param("b", tile_t_b)

    memref_c = ib.memref(ir.MemorySpace.L0C, 0, 512)
    tile_view_c = ib.tile_view([16, 16], [1, 16], 0)
    tile_t_c = ib.tile_type([16, 16], DataType.FP32, memref=memref_c, tile_view=tile_view_c)
    f.return_type(tile_t_c)

func = f.get_result()
```

## Design Principles

1. **Explicit Spans**: All IR nodes require source location. Python captures automatically; C++ requires explicit parameters.
2. **Immutable IR**: Builder creates immutable IR nodes.
3. **Progressive Construction**: Build IR incrementally, statement by statement.
4. **Context Safety**: Builder validates proper nesting and closure.
5. **SSA Style**: For loops use iteration arguments for SSA-style loop-carried values.

## Testing

See `tests/ut/ir/test_builder.py` and `tests/ut/ir/test_flash_attention_builder.py` for comprehensive examples.

## Implementation Details

### Files

- `include/pypto/ir/builder.h` - C++ header
- `src/ir/builder.cpp` - C++ implementation
- `python/pypto/ir/builder.py` - Python wrapper
- `python/bindings/modules/ir_builder.cpp` - Python bindings

### Key Classes

| Class | Purpose |
|-------|---------|
| **IRBuilder** | Main builder with context stack |
| **BuildContext** | Base class for contexts |
| **FunctionContext** | Function building context |
| **ForLoopContext** | For loop building context |
| **IfStmtContext** | If statement building context |
| **FunctionBuilder** | Python function helper |
| **ForLoopBuilder** | Python loop helper |
| **IfStmtBuilder** | Python if helper |
