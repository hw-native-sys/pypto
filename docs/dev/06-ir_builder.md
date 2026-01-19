# IR Builder

The IR Builder provides a convenient API for constructing PyPTO IR incrementally using context managers (Python) or Begin/End patterns (C++). It automatically manages context stacks, validates construction, and tracks source location information.

## Overview

The IR Builder supports building:
- **Functions** with parameters and return types
- **For loops** with iteration arguments (SSA-style loop-carried values)
- **If statements** with then/else branches
- **Return statements** to return values from functions
- **Statements** and **Expressions**

### Key Features

- **Context Management**: Stack-based context tracking ensures proper nesting
- **Automatic Span Tracking** (Python): Uses `inspect` module to capture source locations
- **Explicit Span Parameters** (C++): All methods accept explicit span parameters
- **Validation**: Checks for proper context usage and structure
- **Nested Constructs**: Supports loops in functions, if statements in loops, etc.

## Python API

The Python API uses context managers (`with` statements) for a clean, Pythonic interface.

### Basic Example

```python
from pypto import ir, DataType
from pypto.ir import IRBuilder

ib = IRBuilder()

# Build a simple function
with ib.function("add") as f:
    x = f.param("x", ir.ScalarType(DataType.INT64))
    y = f.param("y", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))

    result = ib.var("result", ir.ScalarType(DataType.INT64))
    add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
    ib.assign(result, add_expr)

func = f.get_result()
print(ir.python_print(func))
```

### For Loops with Iteration Arguments

```python
ib = IRBuilder()

with ib.function("sum_to_n") as f:
    n = f.param("n", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))

    # Loop variable
    i = ib.var("i", ir.ScalarType(DataType.INT64))
    start = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())
    step = ir.ConstInt(1, DataType.INT64, ir.Span.unknown())
    init_val = ir.ConstInt(0, DataType.INT64, ir.Span.unknown())

    # For loop with iter_arg
    with ib.for_loop(i, start, n, step) as loop:
        # Iteration argument: value carried across iterations
        sum_iter = loop.iter_arg("sum", init_val)
        # Return variable: captures final value after loop
        sum_final = loop.return_var("sum_final")

        # Loop body: sum = sum + i
        add_expr = ir.Add(sum_iter, i, DataType.INT64, ir.Span.unknown())
        yield_stmt = ir.YieldStmt([add_expr], ir.Span.unknown())
        ib.emit(yield_stmt)

func = f.get_result()
```

### If Statements

```python
ib = IRBuilder()

with ib.function("max") as f:
    x = f.param("x", ir.ScalarType(DataType.INT64))
    y = f.param("y", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))

    result = ib.var("result", ir.ScalarType(DataType.INT64))
    condition = ir.Gt(x, y, DataType.INT64, ir.Span.unknown())

    with ib.if_stmt(condition) as if_builder:
        # Declare return variable (type required)
        if_builder.return_var("phi_result", ir.ScalarType(DataType.INT64))

        # Then branch
        ib.emit(ir.YieldStmt([x], ir.Span.unknown()))

        # Else branch
        if_builder.else_()
        ib.emit(ir.YieldStmt([y], ir.Span.unknown()))

    # Access the return variable after the if statement
    result = if_builder.output()  # Get the first return variable

    # Or for multiple return variables:
    # result1 = if_builder.output(0)
    # result2 = if_builder.output(1)
    # Or get all at once:
    # results = if_builder.outputs()

func = f.get_result()
```

### Return Statements

```python
ib = IRBuilder()

with ib.function("add_and_return") as f:
    x = f.param("x", ir.ScalarType(DataType.INT64))
    y = f.param("y", ir.ScalarType(DataType.INT64))
    f.return_type(ir.ScalarType(DataType.INT64))

    # Return a single value
    add_expr = ir.Add(x, y, DataType.INT64, ir.Span.unknown())
    ib.return_stmt(add_expr)

func = f.get_result()
```

Return statements can return:
- A single value: `ib.return_stmt(x)`
- Multiple values: `ib.return_stmt([x, y])`
- No values (empty return): `ib.return_stmt()`

### Span Handling

By default, the Python IR Builder automatically captures spans from the call site:

```python
# Automatic span capture
with ib.function("my_func") as f:  # Span captured from this line
    x = f.param("x", type)  # Span captured from this line
```

You can also provide explicit spans:

```python
my_span = ir.Span("my_file.py", 42, 1)

with ib.function("my_func", span=my_span) as f:
    x = f.param("x", type, span=my_span)
```

## C++ API

The C++ API uses Begin/End methods with explicit span parameters.

### Basic Example

```cpp
#include "pypto/ir/builder.h"

using namespace pypto::ir;

IRBuilder ib;

// Helper to create span at current location
auto here = [](int line) { return Span(__FILE__, line, 0); };

// Build a function
ib.BeginFunction("add", here(__LINE__));
auto x = ib.FuncArg("x", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__));
auto y = ib.FuncArg("y", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__));
ib.ReturnType(std::make_shared<ScalarType>(DataType::INT64));

// Body
auto result = ib.Var("result", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__));
auto add_expr = std::make_shared<Add>(x, y, DataType::INT64, here(__LINE__));
ib.Assign(result, add_expr, here(__LINE__));

auto func = ib.EndFunction(here(__LINE__));
```

### For Loops

```cpp
IRBuilder ib;
auto here = [](int line) { return Span(__FILE__, line, 0); };

ib.BeginFunction("sum_to_n", here(__LINE__));
auto n = ib.FuncArg("n", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__));
ib.ReturnType(std::make_shared<ScalarType>(DataType::INT64));

// Loop setup
auto i = ib.Var("i", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__));
auto start = std::make_shared<ConstInt>(0, DataType::INT64, here(__LINE__));
auto stop = n;
auto step = std::make_shared<ConstInt>(1, DataType::INT64, here(__LINE__));

// Begin loop
ib.BeginForLoop(i, start, stop, step, here(__LINE__));

// Iteration argument
auto init_val = std::make_shared<ConstInt>(0, DataType::INT64, here(__LINE__));
auto sum_iter = std::make_shared<IterArg>("sum", std::make_shared<ScalarType>(DataType::INT64),
                                          init_val, here(__LINE__));
ib.AddIterArg(sum_iter);

// Return variable
auto sum_final = ib.Var("sum_final", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__));
ib.AddReturnVar(sum_final);

// Loop body
auto add_expr = std::make_shared<Add>(sum_iter, i, DataType::INT64, here(__LINE__));
auto yield_stmt = std::make_shared<YieldStmt>(std::vector<ExprPtr>{add_expr}, here(__LINE__));
ib.Emit(yield_stmt);

// End loop
auto for_stmt = ib.EndForLoop(here(__LINE__));

auto func = ib.EndFunction(here(__LINE__));
```

### Return Statements

```cpp
IRBuilder ib;
auto here = [](int line) { return Span(__FILE__, line, 0); };

ib.BeginFunction("add_and_return", here(__LINE__));
auto x = ib.FuncArg("x", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__));
auto y = ib.FuncArg("y", std::make_shared<ScalarType>(DataType::INT64), here(__LINE__));
ib.ReturnType(std::make_shared<ScalarType>(DataType::INT64));

// Return statement with expression
auto add_expr = std::make_shared<Add>(x, y, DataType::INT64, here(__LINE__));
ib.Return(std::vector<ExprPtr>{add_expr}, here(__LINE__));

auto func = ib.EndFunction(here(__LINE__));
```

Return statements can return:
- Multiple values: `ib.Return(std::vector<ExprPtr>{x, y}, span)`
- No values (empty return): `ib.Return(span)`

## Context Stack and Validation

The IR Builder maintains a context stack to track nested scopes:

- **Function Context**: Created by `BeginFunction`, ended by `EndFunction`
- **For Loop Context**: Created by `BeginForLoop`, ended by `EndForLoop`
- **If Statement Context**: Created by `BeginIf`, ended by `EndIf`

### Validation Rules

1. **No nested functions**: Cannot call `BeginFunction` inside another function
2. **Context matching**: Must end contexts with the correct End method
3. **Iter args match return vars**: For loops must have equal numbers of iteration arguments and return variables
4. **Proper nesting**: Loops and if statements must be inside a function or another valid context

### Error Messages

The builder provides clear error messages when validation fails:

```python
with ib.function("outer") as f:
    with ib.function("inner") as f2:  # Error!
        pass
# RuntimeError: Cannot begin function 'inner': already inside function 'outer' at file.py:10
```

## Context State Queries

You can query the current builder state:

```python
ib.in_function()  # True if inside a function context
ib.in_loop()      # True if inside a for loop context
ib.in_if()        # True if inside an if statement context
```

```cpp
ib.InFunction()  // true if inside a function context
ib.InLoop()      // true if inside a for loop context
ib.InIf()        // true if inside an if statement context
```

## Type Creation Helpers (Python)

The Python IRBuilder provides convenient helper methods for creating types with memory references and tile views.

### Creating MemRef

```python
from pypto import ir, DataType
from pypto.ir import IRBuilder

ib = IRBuilder()

# Create a memory reference
memref = ib.memref(
    memory_space=ir.MemorySpace.DDR,
    addr=0x1000,  # Can be int or Expr
    size=1024
)

# With symbolic address
base_addr = ib.var("base_addr", ir.ScalarType(DataType.INT64))
memref = ib.memref(ir.MemorySpace.UB, base_addr, 2048)
```

### Creating TileView

```python
ib = IRBuilder()

# Create a tile view with integer dimensions
tile_view = ib.tile_view(
    valid_shape=[16, 16],
    stride=[1, 16],
    start_offset=0
)

# With symbolic dimensions
n = ib.var("n", ir.ScalarType(DataType.INT64))
tile_view = ib.tile_view(
    valid_shape=[n, n],
    stride=[1, n],
    start_offset=0
)
```

### Creating TensorType with MemRef

```python
ib = IRBuilder()

# Simple tensor type (no memref)
tensor_t = ib.tensor_type([64, 128], DataType.FP32)

# Tensor type with memory reference
memref = ib.memref(ir.MemorySpace.DDR, 0x1000, 8192)
tensor_t = ib.tensor_type([64, 128], DataType.FP32, memref=memref)
```

### Creating TileType with MemRef and TileView

```python
ib = IRBuilder()

# Simple tile type (no memref or tile_view)
tile_t = ib.tile_type([16, 16], DataType.FP16)

# Tile type with memory reference
memref = ib.memref(ir.MemorySpace.L0A, 0, 512)
tile_t = ib.tile_type([16, 16], DataType.FP16, memref=memref)

# Complete tile type with memref and tile_view
memref = ib.memref(ir.MemorySpace.L0A, 0, 512)
tile_view = ib.tile_view([16, 16], [1, 16], 0)
tile_t = ib.tile_type([16, 16], DataType.FP16, memref=memref, tile_view=tile_view)
```

### Complete Example with Function

```python
from pypto import ir, DataType
from pypto.ir import IRBuilder

ib = IRBuilder()

with ib.function("matmul_tile") as f:
    # Create tile type parameters with memory references
    memref_a = ib.memref(ir.MemorySpace.L0A, 0, 512)
    tile_t_a = ib.tile_type([16, 16], DataType.FP16, memref=memref_a)

    memref_b = ib.memref(ir.MemorySpace.L0B, 0, 512)
    tile_t_b = ib.tile_type([16, 16], DataType.FP16, memref=memref_b)

    # Function parameters
    a = f.param("a", tile_t_a)
    b = f.param("b", tile_t_b)

    # Result tile type with view
    memref_c = ib.memref(ir.MemorySpace.L0C, 0, 512)
    tile_view_c = ib.tile_view([16, 16], [1, 16], 0)
    tile_t_c = ib.tile_type([16, 16], DataType.FP32, memref=memref_c, tile_view=tile_view_c)

    f.return_type(tile_t_c)

    # Function body would go here...

func = f.get_result()
```

## Design Principles

1. **Explicit Spans**: All IR nodes require source location information. Python automatically captures it; C++ requires explicit parameters.

2. **Immutable IR**: The builder creates immutable IR nodes. Once built, IR cannot be modified.

3. **Progressive Construction**: Build IR incrementally, statement by statement, rather than constructing large trees manually.

4. **Context Safety**: The builder validates that contexts are properly nested and closed.

5. **SSA Style**: For loops use iteration arguments for SSA-style loop-carried values, ensuring proper dataflow representation.

## Testing

See `tests/ut/ir/test_builder.py` for comprehensive unit tests covering:
- Function construction
- For loops with iteration arguments
- If/else statements
- Context validation
- Serialization

See `tests/ut/ir/test_flash_attention_builder.py` for complex nested IR construction examples.

## Implementation Details

### Files

- `include/pypto/ir/builder.h` - C++ header with IRBuilder class
- `src/ir/builder.cpp` - C++ implementation
- `python/pypto/ir/builder.py` - Python wrapper with context managers
- `python/bindings/modules/ir_builder.cpp` - Python bindings

### Key Classes

- **IRBuilder**: Main builder class with context stack
- **BuildContext**: Base class for build contexts
- **FunctionContext**: Context for building functions
- **ForLoopContext**: Context for building for loops
- **IfStmtContext**: Context for building if statements
- **FunctionBuilder**: Python helper for function construction
- **ForLoopBuilder**: Python helper for loop construction
- **IfStmtBuilder**: Python helper for if construction
