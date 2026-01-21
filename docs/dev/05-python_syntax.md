# Python IR Syntax Specification

## Overview

This document specifies the Python-style syntax for PyPTO's IR (Intermediate Representation). The syntax is designed to be:

1. **Complete**: Includes all information needed to reconstruct the IR
2. **Parseable**: Can be parsed back into the IR (parser implemented - see [IR Parser](07-ir_parser.md))
3. **Pythonic**: Follows Python programming style and passes most Python linters
4. **SSA-style**: Uses SSA (Static Single Assignment) style for control flow with `pl.yield()` and `pl.range()`

## Module Structure

Every IR module starts with a program header and import statement. The default uses `pl` as the module alias (recommended):

```python
# pypto.program: program_name
import pypto.language as pl
```

For unnamed programs:

```python
# pypto.program
import pypto.language as pl
```

**Note:** The module prefix is configurable. You can use `pi` for legacy code or any custom prefix.

## Type System

### Scalar Types

Scalar types use the module prefix (default `pl`) followed by the type name:

```python
x: pl.INT64
y: pl.FP32
z: pl.BOOL
```

Available scalar types:
- **Integers**: `pl.INT4`, `pl.INT8`, `pl.INT16`, `pl.INT32`, `pl.INT64`
- **Unsigned integers**: `pl.UINT4`, `pl.UINT8`, `pl.UINT16`, `pl.UINT32`, `pl.UINT64`
- **Floating point**: `pl.FP4`, `pl.FP8`, `pl.FP16`, `pl.FP32`
- **Brain float**: `pl.BF16`
- **Hisilicon float**: `pl.HF4`, `pl.HF8`
- **Boolean**: `pl.BOOL`

### Tensor Types

Tensor types use subscript notation (recommended):

```python
a: pl.Tensor[[4, 8], pl.FP32]      # Fixed shape (4, 8)
b: pl.Tensor[[n, m], pl.INT64]     # Symbolic shape (n, m)
```

### Tile Types

Tile types are 2D tensors (at most 2 dimensions), also using subscript notation:

```python
t: pl.Tile[[16, 16], pl.FP16]      # 2D tile (16, 16)
```

### Memory References (MemRef)

`MemRef` describes memory allocation information for tensors and tiles. It can be created using constructor syntax:

```python
# Create a MemRef with constructor
addr_expr = pl.ConstInt(0x1000, pl.INT64, span)
memref = pl.MemRef(pl.MemorySpace.DDR, addr_expr, 1024)

# Available memory spaces:
# - pl.MemorySpace.DDR   (off-chip main memory)
# - pl.MemorySpace.UB    (on-chip Unified Buffer)
# - pl.MemorySpace.L1    (L1 cache)
# - pl.MemorySpace.L0A   (L0A buffer for matrix A)
# - pl.MemorySpace.L0B   (L0B buffer for matrix B)
# - pl.MemorySpace.L0C   (L0C buffer for matrix C/result)
```

Tensors and tiles can include optional `memref` parameter:

```python
# Tensor with memory reference
tensor: pl.Tensor[[64, 128], pl.FP32], memref=pl.MemRef(pl.MemorySpace.DDR, addr], 8192))

# Tile with memory reference
tile: pl.Tile[[16, 16], pl.FP16], memref=pl.MemRef(pl.MemorySpace.L0A, addr], 512))
```

### Tile Views (TileView)

`TileView` describes the layout and access pattern for a tile, including valid shape, stride, and start offset:

```python
# Create a TileView with constructor
valid_shape = [pl.ConstInt(16, pl.INT64, span), pl.ConstInt(16, pl.INT64, span)]
stride = [pl.ConstInt(1, pl.INT64, span), pl.ConstInt(16, pl.INT64, span)]
start_offset = pl.ConstInt(0, pl.INT64, span)

tile_view = pl.TileView(valid_shape=valid_shape, stride=stride, start_offset=start_offset)
```

Tile types can include both `memref` and `tile_view`:

```python
# Complete tile type with memory reference and view
tile: pl.Tile(
    (16, 16),
    pl.FP16,
    memref=pl.MemRef(pl.MemorySpace.L0A, addr, 512),
    tile_view=pl.TileView(
        valid_shape=[pl.ConstInt(16, pl.INT64, span), pl.ConstInt(16, pl.INT64, span)],
        stride=[pl.ConstInt(1, pl.INT64, span), pl.ConstInt(16, pl.INT64, span)],
        start_offset=pl.ConstInt(0, pl.INT64, span)
    )
)
```

## Expressions

### Variables

Variables are referenced by name:

```python
x
tensor_a
```

### Constants

Integer and floating-point literals:

```python
42
-5
3.14
```

### Binary Operations

Use Python operators naturally:

```python
a + b           # Add
a - b           # Sub
a * b           # Mul
a // b          # FloorDiv
a % b           # FloorMod
a / b           # FloatDiv
a ** b          # Pow
a == b          # Eq
a != b          # Ne
a < b           # Lt
a <= b          # Le
a > b           # Gt
a >= b          # Ge
a and b         # And
a or b          # Or
a ^ b           # Xor (logical)
a & b           # BitAnd
a | b           # BitOr
a << b          # BitShiftLeft
a >> b          # BitShiftRight
```

### Unary Operations

```python
-x              # Neg
~x              # BitNot
not x           # Not
abs(x)          # Abs
```

### Min/Max

```python
min(a, b)       # Min
max(a, b)       # Max
```

### Function/Op Calls

```python
# Regular Op call
op_name(arg1, arg2)

# Op call with attributes (keyword arguments)
tensor_add(a, b, broadcast=True, axis=0)

# Function call (GlobalVar) - implicit from program context
my_function(x, y)
```

## Statements

### Assignment

Assignments include type annotations:

```python
x: pl.INT64 = expr
y: pl.Tensor[[4], pl.FP32] = tensor_op(a)
```

### If Statement (SSA-style)

If statements with return variables use `pl.yield()` to return values from each branch:

```python
# If with both branches returning values
if condition:
    y1 = pl.yield(value1)
else:
    y1 = pl.yield(value2)

# If without else
if condition:
    y1 = pl.yield(value)

# Multiple return values (no inline type annotations - not valid Python)
if condition:
    y1, y2 = pl.yield(value1, value2)
else:
    y1, y2 = pl.yield(value3, value4)
```

**Key points:**
- `pl.yield()` in each branch assigns to SSA phi nodes
- Variables defined in yield become accessible after the if statement
- Both branches must yield the same variables for SSA consistency
- Type annotations cannot be used inline with tuple unpacking (Python limitation)

### For Loop (SSA-style with iter_args)

For loops with loop-carried values (iter_args) use `pl.range()` with tuple unpacking:

```python
# Simple loop without iter_args (no type annotation in loop header)
for i in range(start, stop, step):
    body_statements

# Loop with iter_args (loop-carried values)
# No inline type annotations - not valid Python syntax
j_init: pl.INT64 = 0
for i, (j,) in pl.range(0, n, 1, init_values=[j_init]):
    j = pl.yield(j + 1)
j_final = j

# Multiple iter_args
sum_init: pl.INT64 = 0
prod_init: pl.INT64 = 1
for i, (sum, prod) in pl.range(0, 10, 1, init_values=[sum_init, prod_init]):
    sum, prod = pl.yield(sum + i, prod * i)
sum_final, prod_final = sum, prod   # function return values
```

**Key points:**
- Loop-carried values (iter_args) use `pl.range()` with `init_values`
- Tuple unpacking `(j,)` declares the iter_args
- `pl.yield()` updates values for the next iteration
- After the loop, iter_args contain final values and can be assigned to output variables (e.g., `j_final = j`)
- Type annotations cannot be used in for loop headers or tuple unpacking (Python limitation)

### Yield Statement

```python
yield            # YieldStmt with no values
yield x          # YieldStmt with single value
yield x, y       # YieldStmt with multiple values
```

### Statement Sequences

Statements are naturally sequenced in Python (maps to `SeqStmts`):

```python
stmt1
stmt2
stmt3
```

Multiple assignments in a row create `OpStmts`:

```python
x: pl.INT64 = expr1
y: pl.INT64 = expr2
z: pl.INT64 = expr3
```

## Functions

Functions use Python's `def` syntax with type annotations:

```python
def function_name(param1: pl.INT64, param2: pl.FP32) -> pl.INT64:
    x: pl.INT64 = param1 + 1
    return x
```

Multiple return types use `tuple`:

```python
def function_name(x: pl.INT64) -> tuple[pl.INT64, pl.INT64]:
    y: pl.INT64 = x + 1
    z: pl.INT64 = x * 2
    return y, z
```

No return types:

```python
def function_name(x: pl.INT64):
    y: pl.INT64 = x + 1
```

## Complete Examples

### Example 1: Simple Function

```python
def add_one(x: pl.INT64) -> pl.INT64:
    y: pl.INT64 = x + 1
    return y
```

### Example 2: Tensor Operation

```python
def tensor_add_wrapper(
    a: pl.Tensor[[4, 8], pl.FP32],
    b: pl.Tensor[[4, 8], pl.FP32]
) -> pl.Tensor[[4, 8], pl.FP32]:
    c: pl.Tensor[[4, 8], pl.FP32] = tensor_add(a, b)
    return c
```

### Example 3: Control Flow with If

```python
def conditional(x: pl.INT64) -> pl.INT64:
    if x > 0:
        y1: pl.INT64 = pl.yield(x * 2)
    else:
        y1: pl.INT64 = pl.yield(x * 3)
    return y1
```

### Example 4: Loop with iter_args

```python
def loop_sum(n: pl.INT64) -> pl.INT64:
    sum_init: pl.INT64 = 0
    for i, (sum,) in pl.range(0, n, 1, init_values=[sum_init]):
        sum = pl.yield(sum + i)
    sum_final: pl.INT64 = sum
    return sum_final
```

### Example 5: Full Program

```python
# pypto.program: my_program
import pypto.language as pl

def add(x: pl.INT64, y: pl.INT64) -> pl.INT64:
    z: pl.INT64 = x + y
    return z

def multiply(x: pl.INT64, y: pl.INT64) -> pl.INT64:
    z: pl.INT64 = x * y
    return z

def compute(a: pl.INT64, b: pl.INT64) -> pl.INT64:
    temp1: pl.INT64 = add(a, b)
    temp2: pl.INT64 = multiply(temp1, 2)
    return temp2
```

### Example 6: Complex Nested Control Flow

```python
def flash_attention_kernel(
    q: pl.Tensor((64, 128), pl.FP16),
    k: pl.Tensor((1024, 128), pl.FP16),
    v: pl.Tensor((1024, 128), pl.FP16)
) -> pl.Tensor((64, 128), pl.FP32):
    attention_init: pl.Tensor((64, 128), pl.FP32) = tensor_create(shape=[64, 128], dtype=pl.FP32)
    oi_init: pl.Tensor((64, 128), pl.FP32) = tensor_create(shape=[64, 128], dtype=pl.FP32)
    li_init: pl.Tensor((64, 1), pl.FP32) = tensor_create(shape=[64, 1], dtype=pl.FP32)
    mi_init: pl.Tensor((64, 1), pl.FP32) = tensor_create(shape=[64, 1], dtype=pl.FP32)

    for loop_idx, (
        mi,
        li,
        attention,
        oi
    ) in pl.range(0, 16, 1, init_values=[mi_init, li_init, attention_init, oi_init]):
        kj: pl.Tensor((64, 128), pl.FP16) = tensor_view(k, shape=[64, 128], offset=[loop_idx * 64, 0])
        vj: pl.Tensor((64, 128), pl.FP16) = tensor_view(v, shape=[64, 128], offset=[loop_idx * 64, 0])
        sij: pl.Tensor((64, 128), pl.FP16) = tensor_matmul(q, kj, aTrans=False, bTrans=True)

        # ... more computation ...

        mi, li, attention, oi = pl.yield(
            mi_updated, li_updated, attention_updated, oi_updated
        )

    attention_final: pl.Tensor((64, 128), pl.FP32) = attention
    return attention_final
```

## SSA-Style Control Flow Semantics

### If Statements

The `pl.yield()` in if statements creates SSA phi nodes at the merge point:

```python
# Before if: x is defined
if condition:
    y1 = pl.yield(x + 1)
else:
    y1 = pl.yield(x + 2)
# After if: y1 is defined (phi node merging the two branches)
```

This is equivalent to SSA IR:
```
y1 = phi(x + 1, x + 2)  // based on condition
```

### For Loops

The `pl.yield()` in for loops updates loop-carried values (iter_args):

```python
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(0, 10, 1, init_values=[sum_init]):
    sum = pl.yield(sum + i)
sum_final: pl.INT64 = sum
```

This is equivalent to SSA IR with phi nodes:
```
sum_init = 0
loop (i = 0 to 10, sum_phi):
    sum_phi = phi(sum_init, sum_next)  // first iteration uses sum_init, then sum_next
    sum_next = sum_phi + i
    yield sum_next
sum_final = sum_next  // capture final value in return variable
```

**Key semantics:**
- `sum_init`: Initial value before loop
- `sum`: IterArg variable, scoped to the loop
- `sum_final`: Return variable that captures the final value after loop completion
- Assignment after the loop (`sum_final = sum`) captures the final yielded value

## Configurable Module Prefix

The printer supports configurable module prefixes to match your import style:

### Default: `pi` (Recommended)

```python
# Recommended: short and clear
import pypto.ir as pi

x: pl.INT64 = 42
tensor: pl.Tensor[[64, 128], pl.FP32] = ...
```

### Legacy: `ir`

```python
# Legacy style for backward compatibility
import pypto.ir as pi

x: pl.INT64 = 42
tensor: pl.Tensor[[64, 128], pl.FP32] = ...
```

### Custom Prefix

```python
# Any custom prefix you prefer
import pypto.ir as myir

x: mypl.INT64 = 42
tensor: mypl.Tensor[[64, 128], mypl.FP32] = ...
```

## Usage with Python Printer

The IR can be printed to Python syntax using:

```python
import pypto.language as pl

# Print with default "pl" prefix (recommended)
expr = ir.Add(a, b, dtype, span)
print(ir.python_print(expr))  # "a + b"

stmt = ir.AssignStmt(x, expr, span)
print(ir.python_print(stmt))  # "x: pl.INT64 = a + b"

# Print with custom prefix
print(ir.python_print(stmt, "ir"))     # "x: ir.INT64 = a + b"
print(ir.python_print(stmt, "mypl"))   # "x: mypl.INT64 = a + b"

# Print programs
program = ir.Program([func], "my_program", span)
print(ir.python_print(program))          # Uses "import pypto.language as pl" (default)
print(ir.python_print(program, "pi"))    # Uses "import pypto.ir as pi" (legacy)

# str() uses Python printer with default "pl" prefix
print(str(program))

# as_python() method also accepts custom prefix
print(program.as_python("custom"))
```

## Migration from Old Printer

The old `IRPrinter` is deprecated. Code using the old printer should migrate to `IRPythonPrinter`:

**Old code:**
```python
import pypto.ir as pi
printer = ir.IRPrinter()  # Deprecated
output = printer.Print(expr)
```

**New code:**
```python
import pypto.ir as pi
output = ir.python_print(expr)  # Recommended
# Or use str()
output = str(expr)  # Also uses Python printer
```

## Future Work

1. ~~**Parser Implementation**: A Python parser to read this syntax and construct IR is planned~~ ✅ **Completed** - See [IR Parser](07-ir_parser.md)
2. **Span Support**: Optional span information (source location) can be added via comments or function calls
3. **Type Inference**: Allow omitting type annotations where they can be inferred
4. **Pretty Printing Options**: Configurable formatting (compact vs. verbose, indentation style, etc.)

## References

- [IR Definition](00-ir_definition.md) - Core IR structures
- [Structural Comparison](01-structural_comparison.md) - IR equality and hashing
- [Operator Registration](03-operator_registration.md) - Op system and type inference
