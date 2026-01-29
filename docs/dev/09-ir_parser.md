# IR Parser for DSL Functions

## Overview

The IR parser converts Python DSL code to PyPTO IR using decorators (`@pl.function`, `@pl.program`). It enforces SSA properties, tracks source locations, and supports nested control flow.

**Key components**: Decorator → AST Parser → IR Builder → Scope Manager (SSA) → ir.Function

See [IR Builder](08-ir_builder.md) for manual IR construction and [Python IR Syntax](07-python_syntax.md) for full syntax.

## Usage

### Basic Function

```python
import pypto
import pypto.language as pl

@pl.function
def simple_add(
    x: pl.Tensor[[64, 128], pl.FP16],
    y: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    result: pl.Tensor[[64, 128], pl.FP16] = pl.op.tensor.add(x, y)
    return result

# simple_add is now an ir.Function object
assert isinstance(simple_add, pypto.ir.Function)
```

### Type Annotations

All parameters and local variables require type annotations:

```python
x: pl.Tensor[[64, 128], pl.FP16]  # Recommended subscript syntax
x: pl.Tensor((64, 128), pl.FP16)  # Legacy call syntax (also accepted)
```

Both syntaxes are equivalent; printer always outputs subscript notation.

### For Loops with Iteration Arguments

Use `pl.range()` with tuple unpacking for loop-carried values (iter_args):

```python
for i, (sum_val,) in pl.range(10, init_values=[sum_init]):
    new_sum: pl.Tensor[[1], pl.INT32] = pl.op.tensor.add(sum_val, i)
    sum_out = pl.yeild(new_sum)  # Use pl.yeild (not yield)
```

**Syntax**: `loop_var, (iter_arg1, ...)` - number of iter_args must match init_values.

### Yielding and If Statements

Use `pl.yeild()` to return values from nested scopes:

```python
# Single/multiple value yield
result = pl.yeild(expr)
v1, v2, v3 = pl.yeild(expr1, expr2, expr3)

# If statements create phi nodes
if x > 0:
    positive: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(x, 2.0)
    result = pl.yeild(positive)
else:
    negative: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(x, -1.0)
    result = pl.yeild(negative)
```

**Note**: Spelled `yeild` (not `yield`) to avoid Python keyword conflict.

## Text-Based Parsing

Parse DSL code from strings or files for dynamic code generation:

| Function | Purpose | Example |
|----------|---------|---------|
| `pl.parse(code)` | Parse from string | `func = pl.parse("@pl.function\ndef f(x): ...")` |
| `pl.load(path)` | Load from file | `func = pl.load('kernel.py')` |
| `pl.parse_program(code)` | Parse program from string | `prog = pl.parse_program("@pl.program\nclass P: ...")` |
| `pl.load_program(path)` | Load program from file | `prog = pl.load_program('program.py')` |

**Features**:
- `import pypto.language as pl` automatically injected if missing
- Single function/program per parse (raises `ValueError` otherwise)
- Produces identical `ir.Function`/`ir.Program` objects as decorators
- See `examples/ir_parser/parse_from_text.py` for examples

## SSA Properties

The parser enforces Static Single Assignment:

**Single Assignment**: Each variable assigned once per scope
```python
# ✓ Valid
y: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)

# ✗ Invalid - SSA violation
y: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
y = pl.op.tensor.mul(x, 2.0)  # Error: y already defined
```

**Scope Isolation**: Variables from inner scopes must be yielded
```python
# ✗ Invalid - temp not yielded
for i, (sum_val,) in pl.range(10, init_values=[x]):
    temp: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(sum_val, i)
return temp  # Error: temp not in outer scope

# ✓ Valid - explicit yield
for i, (sum_val,) in pl.range(10, init_values=[x]):
    temp: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(sum_val, i)
    result = pl.yeild(temp)
return result  # OK
```

**Iteration Arguments**: Create new SSA values each iteration via phi nodes.

## Span Tracking and Operations

**Span Tracking**: Preserves source locations for better error messages
- Each IR node includes `Span` with filename, line/column ranges
- Enables debugging, error reporting, and source-to-IR mapping

**Supported Operations**:

| Category | Examples |
|----------|----------|
| **Tensor Ops** | `pl.op.tensor.{add, mul, sub, div, matmul, cast, view, ...}` |
| **Binary Expr** | `a + b`, `a - b`, `a * b`, `a / b`, `i == 0`, `x < 10` |
| **Literals** | `42` → `ConstInt`, `3.14` → `ConstFloat` |

See [Python IR Syntax](07-python_syntax.md) for complete operation list.

## Complete Example

Nested control flow example:

```python
@pl.function
def flash_attn_simplified(
    q: pl.Tensor[[64, 128], pl.FP16],
    k: pl.Tensor[[1024, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP32]:
    attn_init: pl.Tensor[[64, 128], pl.FP32] = pl.op.tensor.create([64, 128], dtype=pl.FP32)

    for i, (attn,) in pl.range(16, init_values=[attn_init]):
        k_block: pl.Tensor[[64, 128], pl.FP16] = pl.op.tensor.view(k, [64, 128], [i * 64, 0])
        scores: pl.Tensor[[64, 128], pl.FP16] = pl.op.tensor.matmul(q, k_block, b_trans=True)

        if i == 0:
            new_attn: pl.Tensor[[64, 128], pl.FP32] = pl.op.tensor.cast(scores, target_type=pl.FP32)
            result = pl.yeild(new_attn)
        else:
            updated: pl.Tensor[[64, 128], pl.FP32] = pl.op.tensor.add(attn, scores)
            result = pl.yeild(updated)

        final = pl.yeild(result)

    return final
```

## Multi-Function Programs with @pl.program

Define programs containing multiple functions that can call each other:

```python
@pl.program
class MathOps:
    @pl.function
    def square(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        result: pl.Tensor[[1], pl.INT32] = pl.op.tensor.mul(x, x)
        return result

    @pl.function
    def sum_of_squares(self, a: pl.Tensor[[1], pl.INT32], b: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        a_squared: pl.Tensor[[1], pl.INT32] = self.square(a)  # Cross-function call
        b_squared: pl.Tensor[[1], pl.INT32] = self.square(b)
        result: pl.Tensor[[1], pl.INT32] = pl.op.tensor.add(a_squared, b_squared)
        return result
```

**Key Rules**:
- Class-based syntax with `@pl.program`
- Methods require `self` parameter (automatically stripped from IR)
- Cross-function calls use `self.method_name()` → resolved to `GlobalVar` references
- Two-pass parsing: collect `GlobalVar`s, then parse bodies (supports forward references)
- Access functions: `program.get_function("name")`
- Text parsing: `pl.parse_program(code)`, `pl.load_program(path)`
- Printing: `pypto.ir.python_print(program)` generates valid `@pl.program` class

**Examples**: See `examples/ir_parser/program_example.py` and `examples/ir_builder/program_builder_example.py`

## Limitations and Testing

**Current Limitations**:
- Scalar comparisons only in if conditions (not tensors)
- No nested function definitions inside `@pl.function`
- Limited Python subset (no classes, decorators within functions)
- Explicit yields required for all scope outputs
- Type annotations required for all variables

**Testing**: Run `pytest tests/ut/language/parser/` for comprehensive parser tests.

## See Also

- [Python IR Syntax](07-python_syntax.md) - Full syntax specification
- [IR Builder](08-ir_builder.md) - Manual IR construction API
- [IR Overview](00-ir_overview.md) - Core IR concepts
