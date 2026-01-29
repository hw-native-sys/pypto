# Python IR Syntax Specification

## Overview

Python-style syntax for PyPTO IR:
- **Complete**: All information needed to reconstruct IR
- **Parseable**: Can be parsed back into IR (see [IR Parser](09-ir_parser.md))
- **Pythonic**: Follows Python style, passes most linters
- **SSA-style**: Uses SSA with `pl.yield()` and `pl.range()`

## Module Structure

```python
# pypto.program: program_name
import pypto.language as pl
```

For unnamed programs: `# pypto.program`

**Note:** Module prefix is configurable (default `pl`, legacy `ir`, custom allowed).

## Type System

### Scalar Types

```python
x: pl.INT64
y: pl.FP32
z: pl.BOOL
```

Available types:

| Category | Types |
|----------|-------|
| **Integers** | `INT4`, `INT8`, `INT16`, `INT32`, `INT64` |
| **Unsigned** | `UINT4`, `UINT8`, `UINT16`, `UINT32`, `UINT64` |
| **Float** | `FP4`, `FP8`, `FP16`, `FP32` |
| **Brain Float** | `BF16` |
| **Hisilicon** | `HF4`, `HF8` |
| **Boolean** | `BOOL` |

### Tensor and Tile Types

```python
# Tensor (subscript notation)
a: pl.Tensor[[4, 8], pl.FP32]      # Fixed shape
b: pl.Tensor[[n, m], pl.INT64]     # Symbolic shape

# Tile (2D tensors, at most 2 dimensions)
t: pl.Tile[[16, 16], pl.FP16]
```

### Memory References (MemRef)

```python
# Create MemRef
addr_expr = pl.ConstInt(0x1000, pl.INT64, span)
memref = pl.MemRef(pl.MemorySpace.DDR, addr_expr, 1024)

# Memory spaces: DDR, UB, L1, L0A, L0B, L0C

# Tensor with memref
tensor: pl.Tensor[[64, 128], pl.FP32], memref=pl.MemRef(pl.MemorySpace.DDR, addr, 8192))
```

### Tile Views (TileView)

```python
# Create TileView
valid_shape = [pl.ConstInt(16, pl.INT64, span)] * 2
stride = [pl.ConstInt(1, pl.INT64, span), pl.ConstInt(16, pl.INT64, span)]
start_offset = pl.ConstInt(0, pl.INT64, span)
tile_view = pl.TileView(valid_shape=valid_shape, stride=stride, start_offset=start_offset)

# Tile with memref and tile_view
tile: pl.Tile(
    (16, 16), pl.FP16,
    memref=pl.MemRef(pl.MemorySpace.L0A, addr, 512),
    tile_view=pl.TileView(valid_shape=..., stride=..., start_offset=...)
)
```

## Expressions

### Variables and Constants

```python
x              # Variable reference
tensor_a       # Tensor variable
42             # Integer literal
3.14           # Float literal
```

### Binary Operations

| Python Operator | PyPTO IR | Category |
|----------------|----------|----------|
| `+` | Add | Arithmetic |
| `-` | Sub | Arithmetic |
| `*` | Mul | Arithmetic |
| `//` | FloorDiv | Arithmetic |
| `%` | FloorMod | Arithmetic |
| `/` | FloatDiv | Arithmetic |
| `**` | Pow | Arithmetic |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | Eq, Ne, Lt, Le, Gt, Ge | Comparison |
| `and`, `or` | And, Or | Logical |
| `^` | Xor | Logical |
| `&`, `|` | BitAnd, BitOr | Bitwise |
| `<<`, `>>` | BitShiftLeft, BitShiftRight | Bitwise |

### Unary Operations and Functions

```python
-x              # Neg
~x              # BitNot
not x           # Not
abs(x)          # Abs
min(a, b)       # Min
max(a, b)       # Max
```

### Function/Op Calls

```python
op_name(arg1, arg2)                      # Op call
tensor_add(a, b, broadcast=True, axis=0) # Op with kwargs
my_function(x, y)                        # Function call
```

## Statements

### Assignment

```python
x: pl.INT64 = expr
y: pl.Tensor[[4], pl.FP32] = tensor_op(a)
```

### If Statement (SSA-style)

```python
# If with both branches
if condition:
    y1 = pl.yield(value1)
else:
    y1 = pl.yield(value2)

# Multiple return values (no inline type annotations)
if condition:
    y1, y2 = pl.yield(value1, value2)
else:
    y1, y2 = pl.yield(value3, value4)
```

**Key points:**
- `pl.yield()` assigns to SSA phi nodes
- Variables defined in yield become accessible after if
- Both branches must yield the same variables
- Type annotations cannot be used inline with tuple unpacking

### For Loop (SSA-style with iter_args)

```python
# Simple loop without iter_args
for i in range(start, stop, step):
    body_statements

# Loop with iter_args (loop-carried values)
j_init: pl.INT64 = 0
for i, (j,) in pl.range(0, n, 1, init_values=[j_init]):
    j = pl.yield(j + 1)
j_final = j

# Multiple iter_args
sum_init: pl.INT64 = 0
prod_init: pl.INT64 = 1
for i, (sum, prod) in pl.range(0, 10, 1, init_values=[sum_init, prod_init]):
    sum, prod = pl.yield(sum + i, prod * i)
sum_final, prod_final = sum, prod
```

**Key points:**
- Loop-carried values use `pl.range()` with `init_values`
- Tuple unpacking `(j,)` declares iter_args
- `pl.yield()` updates values for next iteration
- After loop, iter_args contain final values

### Yield Statement

```python
yield            # No values
yield x          # Single value
yield x, y       # Multiple values
```

### Statement Sequences

```python
stmt1            # Natural Python sequencing
stmt2
stmt3
```

## Functions

```python
# Single return type
def function_name(param1: pl.INT64, param2: pl.FP32) -> pl.INT64:
    x: pl.INT64 = param1 + 1
    return x

# Multiple return types
def function_name(x: pl.INT64) -> tuple[pl.INT64, pl.INT64]:
    y: pl.INT64 = x + 1
    z: pl.INT64 = x * 2
    return y, z

# No return types
def function_name(x: pl.INT64):
    y: pl.INT64 = x + 1
```

## Complete Example

```python
# pypto.program: my_program
import pypto.language as pl

def loop_sum(n: pl.INT64) -> pl.INT64:
    sum_init: pl.INT64 = 0
    for i, (sum,) in pl.range(0, n, 1, init_values=[sum_init]):
        sum = pl.yield(sum + i)
    return sum
```

## SSA-Style Control Flow Semantics

### If Statements

`pl.yield()` creates SSA phi nodes at merge point:

```python
if condition:
    y1 = pl.yield(x + 1)
else:
    y1 = pl.yield(x + 2)
# y1 is phi node: y1 = phi(x + 1, x + 2)
```

### For Loops

`pl.yield()` updates loop-carried values (iter_args):

```python
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(0, 10, 1, init_values=[sum_init]):
    sum = pl.yield(sum + i)
sum_final: pl.INT64 = sum
```

**Semantics:** `sum_init` is initial value, `sum` is IterArg (scoped to loop), `sum_final` captures final value.

## Configurable Module Prefix

The printer supports configurable module prefixes:

| Prefix | Import Statement | Usage |
|--------|------------------|-------|
| **pl** (Recommended) | `import pypto.language as pl` | `x: pl.INT64` |
| **pi** (Alternative) | `import pypto.ir as pi` | `x: pi.INT64` |
| **ir** (Legacy) | `import pypto.ir as ir` | `x: ir.INT64` |
| **custom** | `import pypto.language as mypl` | `x: mypl.INT64` |

### Usage with Printer

```python
# Print with default "pl" prefix
print(ir.python_print(expr))  # "a + b"
print(ir.python_print(stmt))  # "x: pl.INT64 = a + b"

# Print with custom prefix
print(ir.python_print(stmt, "ir"))    # "x: ir.INT64 = a + b"
print(ir.python_print(program, "pi")) # Uses "import pypto.ir as pi"

# str() uses default "pl" prefix
print(str(program))
```

## References

- [IR Overview](00-ir_overview.md) - Core IR structures
- [IR Parser](09-ir_parser.md) - Parsing Python syntax back to IR
- [Operator Registration](05-operator_registration.md) - Op system and type inference
