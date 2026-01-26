# IR Parser for DSL Functions

## Overview

The IR parser provides a decorator-based system for converting high-level Python DSL code into PyPTO IR structures. It allows developers to write IR-generating functions in a more natural Python syntax and have them automatically converted to IR.

## Architecture

The parser consists of several cooperating components:

```
@pl.function decorator
       ↓
   AST Parser ──→ Span Tracker
       ↓              ↓
  IR Builder ←── Type Resolver
       ↓
 Scope Manager (SSA Verification)
       ↓
   ir.Function
```

### Components

1. **Decorator (`decorator.py`)**: Entry point that extracts AST and orchestrates parsing
2. **AST Parser (`ast_parser.py`)**: Converts Python AST nodes to IR builder calls
3. **Span Tracker (`span_tracker.py`)**: Preserves source location information
4. **Scope Manager (`scope_manager.py`)**: Enforces SSA properties and scope isolation
5. **Type Resolver (`type_resolver.py`)**: Converts type annotations to IR types
6. **DSL APIs (`dsl_api.py`)**: Helper functions (`range`, `yeild`, `Tensor`)

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

All function parameters and local variables must have type annotations:

```python
@pl.function
def typed_example(
    scalar: pl.Tensor[[1], pl.INT32],
    tensor: pl.Tensor[[64, 128], pl.FP32],
) -> pl.Tensor[[64, 128], pl.FP32]:
    # Local variables need type annotations
    result: pl.Tensor[[64, 128], pl.FP32] = pl.op.tensor.mul(tensor, 2.0)
    return result
```

**Type syntax**: Two syntaxes are supported:

**Recommended (subscript notation):**
```python
x: pl.Tensor[[64, 128], pl.FP16]
```

**Legacy (call notation):**
```python
x: pl.Tensor((64, 128), pl.FP16)  # Also supported but not recommended
```

Both syntaxes work identically. The parser accepts both, but the printer always outputs the recommended subscript notation.

**Type components:**
- Shape: List `[64, 128]` or tuple `(64, 128)`
- DType: `pl.FP16`, `pl.FP32`, `pl.INT32`, etc.

### For Loops with Iteration Arguments

For loops use `pl.range()` with tuple unpacking to support loop-carried values (iter_args):

```python
@pl.function
def sum_loop(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
    sum_init: pl.Tensor[[1], pl.INT32] = pl.op.tensor.create([1], dtype=pl.INT32)

    # Loop variable and iter_args are unpacked
    for i, (sum_val,) in pl.range(10, init_values=[sum_init]):
        new_sum: pl.Tensor[[1], pl.INT32] = pl.op.tensor.add(sum_val, i)
        # Yield the value for next iteration
        sum_out = pl.yeild(new_sum)

    return sum_out
```

**Syntax**: `for loop_var, (iter_arg1, iter_arg2, ...) in pl.range(stop, init_values=[...])`
- `loop_var`: Loop index variable (e.g., `i`)
- `(iter_arg1, ...)`: Tuple of iteration arguments (loop-carried values)
- `init_values`: List of initial values for iter_args

**Important**: The number of iter_args must match the number of init_values.

### Yielding Values from Scopes

Use `pl.yeild()` to explicitly return values from nested scopes (for loops, if statements):

```python
# Single value yield
result = pl.yeild(expr)

# Multiple value yield (tuple unpacking)
v1, v2, v3 = pl.yeild(expr1, expr2, expr3)

# Annotated yield
output: pl.Tensor[[64, 128], pl.FP32] = pl.yeild(value)
```

**Note**: The spelling is `yeild` (not `yield`) to avoid conflicts with Python's keyword.

### If Statements with Phi Nodes

If statements create SSA phi nodes for values that differ between branches:

```python
@pl.function
def conditional(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    if x > 0:
        positive: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(x, 2.0)
        result = pl.yeild(positive)
    else:
        negative: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(x, -1.0)
        result = pl.yeild(negative)

    return result
```

The `result` variable captures the phi node output from the if statement.

## Text-Based Parsing

In addition to the `@pl.function` decorator, PyPTO provides functions to parse DSL code from strings or files. This is useful for:
- Dynamic code generation
- Loading kernels from configuration files or databases
- Programmatic IR construction
- Building domain-specific code generators

### pl.parse() - Parse from String

Parse a DSL function from a string containing Python code:

```python
import pypto.language as pl

code = """
@pl.function
def vector_add(
    x: pl.Tensor[[128], pl.FP32],
    y: pl.Tensor[[128], pl.FP32],
) -> pl.Tensor[[128], pl.FP32]:
    result: pl.Tensor[[128], pl.FP32] = pl.op.tensor.add(x, y)
    return result
"""

func = pl.parse(code)
assert isinstance(func, pypto.ir.Function)
```

The `import pypto.language as pl` statement is automatically injected if not present:

```python
# This works too - import is automatically added
code_without_import = """
@pl.function
def vector_mul(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.op.tensor.mul(x, 2.0)
    return result
"""

func = pl.parse(code_without_import)
```

### pl.load() - Load from File

Load a DSL function from a Python file:

```python
import pypto.language as pl

# Load from file
func = pl.load('my_kernel.py')
```

The file should contain a single `@pl.function` decorated function. Multiple functions or no functions will raise a `ValueError`.

### Equivalence with Decorator

Both approaches produce identical `ir.Function` objects:

```python
# Using decorator
@pl.function
def my_func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x

# Using parse
func_parsed = pl.parse("""
@pl.function
def my_func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
""")

# Both are ir.Function objects with same structure
assert isinstance(my_func, pypto.ir.Function)
assert isinstance(func_parsed, pypto.ir.Function)
```

### Error Handling

The parser provides clear error messages for common issues:

```python
# No function defined
pl.parse("x = 42")
# ValueError: No @pl.function decorated functions found

# Multiple functions
code = """
@pl.function
def f1(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x

@pl.function
def f2(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return x
"""
pl.parse(code)
# ValueError: Multiple functions found: ['f1', 'f2']

# Syntax errors
pl.parse("@pl.function\ndef bad(x): return x +")
# SyntaxError: Failed to compile code
```

### Use Cases

**Dynamic Kernel Generation:**

```python
def generate_elementwise_kernel(op_name: str, op_func: str) -> pypto.ir.Function:
    """Generate an elementwise operation kernel from a template."""
    code = f"""
@pl.function
def elementwise_{op_name}(
    x: pl.Tensor[[1024], pl.FP32],
    y: pl.Tensor[[1024], pl.FP32],
) -> pl.Tensor[[1024], pl.FP32]:
    result: pl.Tensor[[1024], pl.FP32] = pl.op.tensor.{op_func}(x, y)
    return result
"""
    return pl.parse(code)

# Generate different kernels
add_kernel = generate_elementwise_kernel("add", "add")
mul_kernel = generate_elementwise_kernel("multiply", "mul")
sub_kernel = generate_elementwise_kernel("subtract", "sub")
```

**Loading from Configuration:**

```python
import json

# Load kernel specifications from config
with open('kernels.json') as f:
    kernel_specs = json.load(f)

# Parse each kernel
kernels = {}
for name, code in kernel_specs.items():
    kernels[name] = pl.parse(code)
```

**Serialization Workflow:**

```python
kernel_code = """
@pl.function
def my_kernel(x: pl.Tensor[[1], pl.FP32]) -> pl.Tensor[[1], pl.FP32]:
    return x
"""

# Parse function from text
func = pl.parse(kernel_code)

# Serialize to msgpack
data = pypto.ir.serialize(func)

# Save to file
with open('kernel.msgpack', 'wb') as f:
    f.write(data)

# Later: deserialize
with open('kernel.msgpack', 'rb') as f:
    restored_func = pypto.ir.deserialize(f.read())
```

See `examples/ir_parser/parse_from_text.py` for more examples.

## SSA (Static Single Assignment) Properties

The parser enforces SSA properties to ensure valid IR:

### Single Assignment Rule

Each variable can only be assigned once within a scope:

```python
# ✓ Valid - single assignment per scope
@pl.function
def valid_ssa(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    y: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
    return y

# ✗ Invalid - double assignment
@pl.function
def invalid_ssa(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    y: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)
    y = pl.op.tensor.mul(x, 2.0)  # Error: SSA violation
    return y
```

### Scope Isolation

Variables defined in inner scopes are not accessible in outer scopes unless explicitly yielded:

```python
# ✗ Invalid - variable leaks from scope
@pl.function
def invalid_scope(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for i, (sum_val,) in pl.range(10, init_values=[x]):
        temp: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(sum_val, i)
        # temp is not yielded

    return temp  # Error: temp not defined in outer scope

# ✓ Valid - explicit yield
@pl.function
def valid_scope(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for i, (sum_val,) in pl.range(10, init_values=[x]):
        temp: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(sum_val, i)
        result = pl.yeild(temp)  # Explicitly yield

    return result  # OK: result is output of loop
```

### Iteration Arguments

Loop-carried values (iter_args) create new SSA values on each iteration:

```python
@pl.function
def iter_args_example(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
    init: pl.Tensor[[1], pl.INT32] = pl.op.tensor.create([1], dtype=pl.INT32)

    for i, (accumulator,) in pl.range(10, init_values=[init]):
        # accumulator is a phi node - different value each iteration
        # On iteration 0: accumulator = init
        # On iteration 1+: accumulator = previous yield value

        next_val: pl.Tensor[[1], pl.INT32] = pl.op.tensor.add(accumulator, i)
        result = pl.yeild(next_val)  # Becomes accumulator in next iteration

    return result
```

## Span Tracking

The parser preserves source location information from the original Python code:

```python
@pl.function
def example(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    # Each statement gets a span pointing to the source line
    y: pl.Tensor[[64], pl.FP32] = pl.op.tensor.add(x, 1.0)  # Span: line 3
    return y  # Span: line 4
```

Every IR node created by the parser includes a `Span` object with:
- `filename`: Source file path
- `begin_line`: Starting line number
- `begin_column`: Starting column offset
- `end_line`: Ending line number (for multi-line constructs)
- `end_column`: Ending column offset

This enables:
- Better error messages pointing to source code
- Debugging and code visualization tools
- Source-to-IR mapping for analysis

## Supported Operations

### Tensor Operations

All `pl.op.tensor.*` operations are supported:

```python
# Creation
t = pl.op.tensor.create([64, 128], dtype=pl.FP32)

# Binary operations
result = pl.op.tensor.add(a, b)
result = pl.op.tensor.mul(a, b)
result = pl.op.tensor.sub(a, b)
result = pl.op.tensor.div(a, b)

# Matrix operations
result = pl.op.tensor.matmul(a, b, out_dtype=pl.FP32, b_trans=True)

# Reductions
max_val = pl.op.tensor.row_max(tensor, axis=-1, keep_dim=1)
sum_val = pl.op.tensor.row_sum(tensor, axis=-1, keep_dim=1)

# Transforms
casted = pl.op.tensor.cast(tensor, target_type=pl.FP32, mode="round")
viewed = pl.op.tensor.view(tensor, [32, 256], [offset_h, offset_w])

# Element-wise
result = pl.op.tensor.exp(tensor)
result = pl.op.tensor.maximum(a, b)
```

### Binary Expressions

Python operators are automatically converted to IR expressions:

```python
# Arithmetic
result = a + b  # ir.add(a, b, span)
result = a - b  # ir.sub(a, b, span)
result = a * b  # ir.mul(a, b, span)
result = a / b  # ir.truediv(a, b, span)

# Comparisons
cond = i == 0    # ir.eq(i, 0, span)
cond = x < 10    # ir.lt(x, 10, span)
cond = y >= 5    # ir.ge(y, 5, span)
```

### Literals

Python literals are automatically converted:

```python
# Integer literals → ConstInt
i = 42  # ir.ConstInt(42, DataType.INT64, span)

# Float literals → ConstFloat
f = 3.14  # ir.ConstFloat(3.14, DataType.FP32, span)

# List literals → Python list (for operation arguments)
shape = [64, 128]
tensor = pl.op.tensor.create(shape, dtype=pl.FP32)
```

## Complex Example

Here's a complete example showing nested control flow:

```python
@pl.function
def flash_attn_simplified(
    q: pl.Tensor[[64, 128], pl.FP16],
    k: pl.Tensor[[1024, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP32]:
    # Initialize accumulator
    attn_init: pl.Tensor[[64, 128], pl.FP32] = pl.op.tensor.create(
        [64, 128], dtype=pl.FP32
    )

    # Loop over blocks
    for i, (attn,) in pl.range(16, init_values=[attn_init]):
        # Get block
        k_block: pl.Tensor[[64, 128], pl.FP16] = pl.op.tensor.view(
            k, [64, 128], [i * 64, 0]
        )

        # Compute attention for this block
        scores: pl.Tensor[[64, 128], pl.FP16] = pl.op.tensor.matmul(
            q, k_block, b_trans=True
        )

        # Conditional update
        if i == 0:
            new_attn: pl.Tensor[[64, 128], pl.FP32] = pl.op.tensor.cast(
                scores, target_type=pl.FP32
            )
            result = pl.yeild(new_attn)
        else:
            updated: pl.Tensor[[64, 128], pl.FP32] = pl.op.tensor.add(attn, scores)
            result = pl.yeild(updated)

        final = pl.yeild(result)

    return final
```

## Implementation Details

### Parser Pipeline

1. **Decorator invocation**: `@pl.function` captures the function
2. **AST extraction**: `inspect.getsource()` gets source code, `ast.parse()` creates AST
3. **Function parsing**:
   - Parse function signature (parameters, return type)
   - Create IRBuilder and function context
   - Parse function body statements
4. **Statement parsing**:
   - Annotated assignments → `ib.let()`
   - For loops → `ib.for_loop()` with iter_args
   - If statements → `ib.if_stmt()` with return_vars
   - Returns → `ib.return_stmt()`
5. **Expression parsing**: Convert Python expressions to IR expressions
6. **Type resolution**: Convert type annotations to IR types
7. **Scope management**: Track variables and enforce SSA
8. **Span tracking**: Preserve source locations
9. **IR generation**: Build and return `ir.Function`

### Error Handling

The parser provides detailed error messages:

```python
# SSA violation
ValueError: SSA violation: Variable 'x' is already defined in current scope.
Each variable can only be assigned once per scope.

# Undefined variable
ValueError: Undefined variable: result

# Type mismatch
ValueError: Type mismatch in let statement for variable 'x':
  Inferred type: TensorType([64, 128], FP32)
  Provided type: TensorType([64, 64], FP32)

# Missing type annotation
ValueError: Parameter 'x' missing type annotation
```

## Limitations

Current limitations of the parser:

1. **Scalar comparisons only**: If conditions must use scalar values, not tensors
2. **No nested functions**: Cannot define functions inside `@pl.function`
3. **Limited Python features**: Only supports subset of Python (no classes, decorators within functions, etc.)
4. **Explicit yields required**: All scope outputs must be explicitly yielded
5. **Type annotations required**: All variables must have type annotations

## Testing

Comprehensive tests are available in `tests/ut/ir/parser/`:

```bash
# Run parser tests
pytest tests/ut/ir/parser/

# Run with coverage
pytest tests/ut/ir/parser/ --cov=pypto.ir.parser
```

## Multi-Function Programs with @pl.program

The `@pl.program` decorator enables defining programs containing multiple related functions that can call each other. This is useful for organizing kernel libraries, implementing complex algorithms that require helper functions, and building reusable IR components.

### Basic Usage

```python
import pypto.language as pl

@pl.program
class VectorOps:
    """Program with vector operation functions."""

    @pl.function
    def vector_add(
        self,
        x: pl.Tensor[[128], pl.FP32],
        y: pl.Tensor[[128], pl.FP32],
    ) -> pl.Tensor[[128], pl.FP32]:
        """Add two vectors element-wise."""
        result: pl.Tensor[[128], pl.FP32] = pl.op.tensor.add(x, y)
        return result

    @pl.function
    def vector_mul(
        self,
        x: pl.Tensor[[128], pl.FP32],
        scalar: pl.Tensor[[1], pl.FP32],
    ) -> pl.Tensor[[128], pl.FP32]:
        """Multiply vector by scalar."""
        result: pl.Tensor[[128], pl.FP32] = pl.op.tensor.mul(x, scalar)
        return result

# VectorOps is now an ir.Program object
assert isinstance(VectorOps, pypto.ir.Program)
assert len(VectorOps.functions) == 2
```

### Key Rules for @pl.program

1. **Class-based syntax**: Use a class decorated with `@pl.program`
2. **Methods have `self`**: All `@pl.function` methods must have `self` as first parameter (standard Python)
3. **`self` is transparent**: The `self` parameter is automatically stripped from the IR - it won't appear in the generated functions
4. **Access functions**: Use `program.get_function("name")` to retrieve individual functions
5. **Sorted storage**: Functions are automatically sorted alphabetically by name in the program

### Cross-Function Calls

Functions within a program can call each other using `self.method_name()` syntax. The parser automatically resolves these to `GlobalVar` references:

```python
@pl.program
class MathOps:
    @pl.function
    def square(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        """Square a number."""
        result: pl.Tensor[[1], pl.INT32] = pl.op.tensor.mul(x, x)
        return result

    @pl.function
    def sum_of_squares(
        self,
        a: pl.Tensor[[1], pl.INT32],
        b: pl.Tensor[[1], pl.INT32],
    ) -> pl.Tensor[[1], pl.INT32]:
        """Compute a^2 + b^2 by calling square()."""
        # Call square method using self.square()
        a_squared: pl.Tensor[[1], pl.INT32] = self.square(a)
        b_squared: pl.Tensor[[1], pl.INT32] = self.square(b)
        result: pl.Tensor[[1], pl.INT32] = pl.op.tensor.add(a_squared, b_squared)
        return result
```

**How it works:**
- Parser performs **two-pass parsing**:
  1. **Pass 1**: Scan all `@pl.function` methods to create `GlobalVar` references
  2. **Pass 2**: Parse function bodies, resolving `self.method()` calls to `GlobalVar` references
- In the IR, `self.square(a)` becomes `ir.Call(square_globalvar, [a])`
- Forward references work: functions can call others defined later in the class

### Text-Based Program Parsing

Like functions, programs can be parsed from strings or files:

```python
import pypto.language as pl

# Parse from string
code = """
@pl.program
class MyProgram:
    @pl.function
    def increment(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        one: pl.Tensor[[1], pl.INT32] = pl.op.tensor.create([1], dtype=pl.INT32)
        result: pl.Tensor[[1], pl.INT32] = pl.op.tensor.add(x, one)
        return result

    @pl.function
    def double_increment(self, x: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[1], pl.INT32]:
        once: pl.Tensor[[1], pl.INT32] = self.increment(x)
        result: pl.Tensor[[1], pl.INT32] = self.increment(once)
        return result
"""

program = pl.parse_program(code)
assert isinstance(program, pypto.ir.Program)
assert program.name == "MyProgram"

# Load from file
program = pl.load_program("my_program.py")
```

**Note**: The import statement `import pypto.language as pl` is automatically injected if not present.

### Manual Program Construction with IRBuilder

For programmatic IR generation, use the IRBuilder API:

```python
from pypto import DataType
from pypto.ir import IRBuilder
from pypto.pypto_core import ir

ib = IRBuilder()

with ib.program("MathLib") as prog:
    # Declare functions up front to enable cross-function calls
    square_gvar = prog.declare_function("square")
    cube_gvar = prog.declare_function("cube")

    # Build square function
    with ib.function("square") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))
        result = ib.let("result", x * x)
        ib.return_stmt(result)

    prog.add_function(f.get_result())

    # Build cube function that calls square
    with ib.function("cube") as f:
        x = f.param("x", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        # Call square using its GlobalVar
        x_squared = ib.let("x_squared", ir.Call(square_gvar, [x], ir.Span.unknown()))
        result = ib.let("result", x * x_squared)
        ib.return_stmt(result)

    prog.add_function(f.get_result())

program = prog.get_result()
```

### Printing Programs

Programs can be printed back to Python code using `pypto.ir.python_print()`:

```python
program = ...  # Your program

# Print as @pl.program class
code = pypto.ir.python_print(program)
print(code)
```

**Output format:**
```python
@pl.program
class ProgramName:
    @pl.function
    def method1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        # method body

    @pl.function
    def method2(self, y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        # method body
```

**Key features:**
- Methods automatically get `self` parameter added
- Cross-function calls are printed as `self.method_name()`
- Output is valid Python that can be parsed back
- Round-trip consistency: parse → print → parse produces equivalent IR

### Examples

Complete working examples are available:
- [`examples/ir_parser/program_example.py`](../../examples/ir_parser/program_example.py) - Using `@pl.program` decorator with cross-function calls
- [`examples/ir_builder/program_builder_example.py`](../../examples/ir_builder/program_builder_example.py) - Manual program construction with IRBuilder

## See Also

- [Python IR Syntax](05-python_syntax.md) - Full syntax specification
- [IR Builder](06-ir_builder.md) - Manual IR construction API
- [IR Definition](00-ir_definition.md) - Core IR concepts
