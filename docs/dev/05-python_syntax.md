# Python IR Syntax Specification

## Overview

This document specifies the Python-style syntax for PyPTO's IR (Intermediate Representation). The syntax is designed to be:

1. **Complete**: Includes all information needed to reconstruct the IR
2. **Parseable**: Can be parsed back into the IR (parser to be implemented)
3. **Pythonic**: Follows Python programming style and passes most Python linters
4. **SSA-style**: Uses SSA (Static Single Assignment) style for control flow with `pi.yield()` and `pi.range()`

## Module Structure

Every IR module starts with a program header and import statement. The default uses `pi` as the module alias (recommended):

```python
# pypto.program: program_name
import pypto.ir as pi
```

For unnamed programs:

```python
# pypto.program
import pypto.ir as pi
```

**Note:** The module prefix is configurable. You can use `ir` for legacy code or any custom prefix.

## Type System

### Scalar Types

Scalar types use the module prefix (default `pi`) followed by the type name:

```python
x: pi.Int64
y: pi.FP32
z: pi.Bool
```

Available scalar types:
- **Integers**: `pi.Int4`, `pi.Int8`, `pi.Int16`, `pi.Int32`, `pi.Int64`
- **Unsigned integers**: `pi.UInt4`, `pi.UInt8`, `pi.UInt16`, `pi.UInt32`, `pi.UInt64`
- **Floating point**: `pi.FP4`, `pi.FP8`, `pi.FP16`, `pi.FP32`
- **Brain float**: `pi.BFloat16`
- **Hisilicon float**: `pi.HF4`, `pi.HF8`
- **Boolean**: `pi.Bool`

### Tensor Types

Tensor types use PyTorch-style syntax with shape as a tuple:

```python
a: pi.Tensor((4, 8), pi.FP32)      # Fixed shape (4, 8)
b: pi.Tensor((n, m), pi.Int64)     # Symbolic shape (n, m)
```

### Tile Types

Tile types are 2D tensors (at most 2 dimensions), also using PyTorch-style syntax:

```python
t: pi.Tile((16, 16), pi.FP16)      # 2D tile (16, 16)
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
x: pi.Int64 = expr
y: pi.Tensor((4,), pi.FP32) = tensor_op(a)
```

### If Statement (SSA-style)

If statements with return variables use `pi.yield()` to return values from each branch:

```python
# If with both branches returning values
if condition:
    y1 = pi.yield(value1)
else:
    y1 = pi.yield(value2)

# If without else
if condition:
    y1 = pi.yield(value)

# Multiple return values (no inline type annotations - not valid Python)
if condition:
    y1, y2 = pi.yield(value1, value2)
else:
    y1, y2 = pi.yield(value3, value4)
```

**Key points:**
- `pi.yield()` in each branch assigns to SSA phi nodes
- Variables defined in yield become accessible after the if statement
- Both branches must yield the same variables for SSA consistency
- Type annotations cannot be used inline with tuple unpacking (Python limitation)

### For Loop (SSA-style with iter_args)

For loops with loop-carried values (iter_args) use `pi.range()` with tuple unpacking:

```python
# Simple loop without iter_args (no type annotation in loop header)
for i in range(start, stop, step):
    body_statements

# Loop with iter_args (loop-carried values)
# No inline type annotations - not valid Python syntax
j_init: pi.Int64 = 0
for i, (j,) in pi.range(0, n, 1, init_values=[j_init]):
    j = pi.yield(j + 1)
j_final = j

# Multiple iter_args
sum_init: pi.Int64 = 0
prod_init: pi.Int64 = 1
for i, (sum, prod) in pi.range(0, 10, 1, init_values=[sum_init, prod_init]):
    sum, prod = pi.yield(sum + i, prod * i)
sum_final, prod_final = sum, prod   # function return values
```

**Key points:**
- Loop-carried values (iter_args) use `pi.range()` with `init_values`
- Tuple unpacking `(j,)` declares the iter_args
- `pi.yield()` updates values for the next iteration
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
x: pi.Int64 = expr1
y: pi.Int64 = expr2
z: pi.Int64 = expr3
```

## Functions

Functions use Python's `def` syntax with type annotations:

```python
def function_name(param1: pi.Int64, param2: pi.FP32) -> pi.Int64:
    x: pi.Int64 = param1 + 1
    return x
```

Multiple return types use `tuple`:

```python
def function_name(x: pi.Int64) -> tuple[pi.Int64, pi.Int64]:
    y: pi.Int64 = x + 1
    z: pi.Int64 = x * 2
    return y, z
```

No return types:

```python
def function_name(x: pi.Int64):
    y: pi.Int64 = x + 1
```

## Complete Examples

### Example 1: Simple Function

```python
def add_one(x: pi.Int64) -> pi.Int64:
    y: pi.Int64 = x + 1
    return y
```

### Example 2: Tensor Operation

```python
def tensor_add_wrapper(
    a: pi.Tensor((4, 8), pi.FP32),
    b: pi.Tensor((4, 8), pi.FP32)
) -> pi.Tensor((4, 8), pi.FP32):
    c: pi.Tensor((4, 8), pi.FP32) = tensor_add(a, b)
    return c
```

### Example 3: Control Flow with If

```python
def conditional(x: pi.Int64) -> pi.Int64:
    if x > 0:
        y1: pi.Int64 = pi.yield(x * 2)
    else:
        y1: pi.Int64 = pi.yield(x * 3)
    return y1
```

### Example 4: Loop with iter_args

```python
def loop_sum(n: pi.Int64) -> pi.Int64:
    sum_init: pi.Int64 = 0
    for i, (sum,) in pi.range(0, n, 1, init_values=[sum_init]):
        sum = pi.yield(sum + i)
    sum_final: pi.Int64 = sum
    return sum_final
```

### Example 5: Full Program

```python
# pypto.program: my_program
import pypto.ir as pi

def add(x: pi.Int64, y: pi.Int64) -> pi.Int64:
    z: pi.Int64 = x + y
    return z

def multiply(x: pi.Int64, y: pi.Int64) -> pi.Int64:
    z: pi.Int64 = x * y
    return z

def compute(a: pi.Int64, b: pi.Int64) -> pi.Int64:
    temp1: pi.Int64 = add(a, b)
    temp2: pi.Int64 = multiply(temp1, 2)
    return temp2
```

### Example 6: Complex Nested Control Flow

```python
def flash_attention_kernel(
    q: pi.Tensor((64, 128), pi.FP16),
    k: pi.Tensor((1024, 128), pi.FP16),
    v: pi.Tensor((1024, 128), pi.FP16)
) -> pi.Tensor((64, 128), pi.FP32):
    attention_init: pi.Tensor((64, 128), pi.FP32) = tensor_create(shape=[64, 128], dtype=pi.FP32)
    oi_init: pi.Tensor((64, 128), pi.FP32) = tensor_create(shape=[64, 128], dtype=pi.FP32)
    li_init: pi.Tensor((64, 1), pi.FP32) = tensor_create(shape=[64, 1], dtype=pi.FP32)
    mi_init: pi.Tensor((64, 1), pi.FP32) = tensor_create(shape=[64, 1], dtype=pi.FP32)

    for loop_idx, (
        mi,
        li,
        attention,
        oi
    ) in pi.range(0, 16, 1, init_values=[mi_init, li_init, attention_init, oi_init]):
        kj: pi.Tensor((64, 128), pi.FP16) = tensor_view(k, shape=[64, 128], offset=[loop_idx * 64, 0])
        vj: pi.Tensor((64, 128), pi.FP16) = tensor_view(v, shape=[64, 128], offset=[loop_idx * 64, 0])
        sij: pi.Tensor((64, 128), pi.FP16) = tensor_matmul(q, kj, aTrans=False, bTrans=True)

        # ... more computation ...

        mi, li, attention, oi = pi.yield(
            mi_updated, li_updated, attention_updated, oi_updated
        )

    attention_final: pi.Tensor((64, 128), pi.FP32) = attention
    return attention_final
```

## SSA-Style Control Flow Semantics

### If Statements

The `pi.yield()` in if statements creates SSA phi nodes at the merge point:

```python
# Before if: x is defined
if condition:
    y1 = pi.yield(x + 1)
else:
    y1 = pi.yield(x + 2)
# After if: y1 is defined (phi node merging the two branches)
```

This is equivalent to SSA IR:
```
y1 = phi(x + 1, x + 2)  // based on condition
```

### For Loops

The `pi.yield()` in for loops updates loop-carried values (iter_args):

```python
sum_init: pi.Int64 = 0
for i, (sum,) in pi.range(0, 10, 1, init_values=[sum_init]):
    sum = pi.yield(sum + i)
sum_final: pi.Int64 = sum
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

x: pi.Int64 = 42
tensor: pi.Tensor((64, 128), pi.FP32) = ...
```

### Legacy: `ir`

```python
# Legacy style for backward compatibility
import pypto.ir as pi

x: pi.Int64 = 42
tensor: pi.Tensor((64, 128), pi.FP32) = ...
```

### Custom Prefix

```python
# Any custom prefix you prefer
import pypto.ir as myir

x: mypi.Int64 = 42
tensor: mypi.Tensor[mypi.FP32, 64, 128] = ...
```

## Usage with Python Printer

The IR can be printed to Python syntax using:

```python
import pypto.ir as pi

# Print with default "pi" prefix (recommended)
expr = ir.Add(a, b, dtype, span)
print(ir.python_print(expr))  # "a + b"

stmt = ir.AssignStmt(x, expr, span)
print(ir.python_print(stmt))  # "x: pi.Int64 = a + b"

# Print with custom prefix
print(ir.python_print(stmt, "ir"))     # "x: pi.Int64 = a + b"
print(ir.python_print(stmt, "myir"))   # "x: mypi.Int64 = a + b"

# Print programs
program = ir.Program([func], "my_program", span)
print(ir.python_print(program))          # Uses "import pypto.ir as pi"
print(ir.python_print(program, "ir"))    # Uses "import pypto.ir as pi"

# str() uses Python printer with default "pi" prefix
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

1. **Parser Implementation**: A Python parser to read this syntax and construct IR is planned
2. **Span Support**: Optional span information (source location) can be added via comments or function calls
3. **Type Inference**: Allow omitting type annotations where they can be inferred
4. **Pretty Printing Options**: Configurable formatting (compact vs. verbose, indentation style, etc.)

## References

- [IR Definition](00-ir_definition.md) - Core IR structures
- [Structural Comparison](01-structural_comparison.md) - IR equality and hashing
- [Operator Registration](03-operator_registration.md) - Op system and type inference
