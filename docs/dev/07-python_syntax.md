# Python IR Syntax Specification

## Overview

Python-style syntax for PyPTO IR:
- **Complete**: All information needed to reconstruct IR
- **Parseable**: Can be parsed back into IR (see [IR Parser](09-ir_parser.md))
- **Pythonic**: Follows Python style, passes most linters
- **SSA-style**: Uses SSA with `pl.yield_()` and `pl.range()`

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

# Tile (block in unified buffer)
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
# Explicit namespace
pl.op.tensor.add(a, b)                  # Tensor addition
pl.op.block.load(t, 0, 0, 64, 64)      # Block load

# Unified dispatch (auto-selects tensor/block based on input type)
pl.op.add(a, b)                          # Tensor or Tile — dispatched automatically
pl.op.mul(tile, 2.0)                     # Tile + scalar → block.muls
pl.op.exp(tile)                          # Tile → block.exp

# Promoted ops (single-module ops accessible at pl.op.*)
pl.op.load(t, 0, 0, 64, 64)            # Promoted from block
pl.op.create([64], dtype=pl.FP32)       # Promoted from tensor
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
    y1 = pl.yield_(value1)
else:
    y1 = pl.yield_(value2)

# Multiple return values (no inline type annotations)
if condition:
    y1, y2 = pl.yield_(value1, value2)
else:
    y1, y2 = pl.yield_(value3, value4)
```

**Key points:**
- `pl.yield_()` assigns to SSA phi nodes
- Variables defined in yield become accessible after if
- Both branches must yield the same variables
- Type annotations cannot be used inline with tuple unpacking

### For Loop (SSA-style with iter_args)

```python
# Simple loop
for i in pl.range(start, stop, step):
    body_statements

# Loop with iter_args (loop-carried values)
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(0, n, 1, init_values=[sum_init]):
    sum = pl.yield_(sum + i)
sum_final = sum
```

**Key points:** Loop-carried values use `pl.range()` with `init_values`, tuple unpacking `(sum,)` declares iter_args, `pl.yield_()` updates values for next iteration, after loop iter_args contain final values.

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

# With function type
@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(n: pl.INT64) -> pl.INT64:
    return n + 1

@pl.function(type=pl.FunctionType.InCore)
def aicore_kernel(x: pl.INT64) -> pl.INT64:
    return x * 2
```

### Function Types

| Type | Usage | Description |
|------|-------|-------------|
| `pl.FunctionType.Opaque` | Default | Unspecified function type |
| `pl.FunctionType.Orchestration` | Host/AICPU | Control flow and dependency analysis |
| `pl.FunctionType.InCore` | AICore | Sub-graph on specific AICore |

When no type is specified, functions default to `Opaque`.

## Complete Example

### Tensor Operations (Loop with iter_args)

```python
# pypto.program: my_program
import pypto.language as pl

def loop_sum(n: pl.INT64) -> pl.INT64:
    sum_init: pl.INT64 = 0
    for i, (sum,) in pl.range(0, n, 1, init_values=[sum_init]):
        sum = pl.yield_(sum + i)
    return sum
```

### Block Operations (Tile-based computation)

```python
import pypto.language as pl

@pl.program
class BlockExample:
    @pl.function
    def tile_add(
        self,
        input_a: pl.Tensor[[64, 64], pl.FP32],
        input_b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(input_a, 0, 0, 64, 64)
        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.op.block.load(input_b, 0, 0, 64, 64)
        tile_c: pl.Tile[[64, 64], pl.FP32] = pl.op.block.add(tile_a, tile_b)
        result: pl.Tensor[[64, 64], pl.FP32] = pl.op.block.store(tile_c, 0, 0, 64, 64, output)
        return result
```

## SSA-Style Control Flow

`pl.yield_()` creates SSA phi nodes for if/for statements:

```python
# If: phi node at merge point
if condition:
    y1 = pl.yield_(x + 1)
else:
    y1 = pl.yield_(x + 2)
# y1 = phi(x + 1, x + 2)

# For: loop-carried values via iter_args
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(0, 10, 1, init_values=[sum_init]):
    sum = pl.yield_(sum + i)
sum_final: pl.INT64 = sum  # captures final value
```

## Configurable Module Prefix

The printer supports configurable module prefixes (`pl`, `pi`, `ir`, or custom):

```python
print(ir.python_print(stmt))          # "x: pl.INT64 = a + b" (default)
print(ir.python_print(stmt, "ir"))    # "x: ir.INT64 = a + b"
```

## References

- [IR Overview](00-ir_overview.md) - Core IR structures
- [IR Parser](09-ir_parser.md) - Parsing Python syntax back to IR
- [Operator Registration](05-operator_registration.md) - Op system and type inference
