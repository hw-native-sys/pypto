# Functions and Program Structure

Function declaration forms, parameter directions, cross-module reuse, and how to
print IR back to Python syntax.

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
| ---- | ----- | ----------- |
| `pl.FunctionType.Opaque` | Default | Unspecified function type |
| `pl.FunctionType.Orchestration` | Host/AICPU | Control flow and dependency analysis |
| `pl.FunctionType.InCore` | AICore | Sub-graph on specific AICore (unspecialized) |
| `pl.FunctionType.AIC` | Cube core | Cube core kernel (specialized InCore) |
| `pl.FunctionType.AIV` | Vector core | Vector core kernel (specialized InCore) |
| `pl.FunctionType.Group` | Multi-core | Co-scheduled group of AIC + AIV kernels |
| `pl.FunctionType.Graph` | Host/AICPU | Recordable orchestration fragment, replayed by the `host_build_graph` runtime (see below) |

When no type is specified, functions default to `Opaque`.

### Graph Fragments

`pl.FunctionType.Graph` marks a function as a recordable orchestration fragment.
Under the `host_build_graph` runtime each call site becomes one task launch that
the runtime records on the first call and replays afterwards, so N calls cost one
graph build instead of N:

```python
@pl.program
class Decoder:
    @pl.function(type=pl.FunctionType.Graph)
    def layer(self, cur, normed, next_hidden, wq, layer_base: pl.Scalar[pl.INDEX]):
        ...

    @pl.function
    def decode(self, cur, normed, next_hidden, wq):
        for i in pl.range(40):
            self.layer(cur, normed, next_hidden, wq, i * 5120)
```

One Graph function is one recorded topology: the runtime identifies the recording
by the address of the emitted C++ function, so there is no cache key to name or
keep unique.

### Parameter Directions

Parameters can have `In` (default), `Out`, or `InOut` directions using wrapper types:

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(
    qi: pl.Tensor[[16, 128], pl.BF16],                   # In (default)
    output: pl.InOut[pl.Tensor[[16, 128], pl.FP32]],      # InOut
    result: pl.Out[pl.Tensor[[16, 128], pl.FP32]],        # Out
    scale: pl.Scalar[pl.FP32],                             # In (default)
) -> pl.Tensor[[16, 128], pl.FP32]:
    ...
```

| Direction | Wrapper | Description |
| --------- | ------- | ----------- |
| `In` | None (default) | Read-only input parameter |
| `Out` | `pl.Out[type]` | Write-only output parameter |
| `InOut` | `pl.InOut[type]` | Read-write input/output parameter |

**Constraint:** `Scalar` parameters cannot have `InOut` direction (raises `ParserTypeError`).

#### Writing an `Out` / `InOut` parameter

**A bare assignment does not write the parameter.** It rebinds the Python name:
the parameter Var is re-pointed at a freshly computed tensor and the caller's
buffer is left untouched. The program still compiles and runs. What the caller
gets back depends on the direction: an `Out` buffer is freshly allocated and
never initialised, so it reads as garbage; an `InOut` buffer still holds the
input the caller passed in, so the result is silently stale.

```python
out = pl.add(a, b)          # ❌ writes nothing
out[:] = pl.add(a, b)       # ✅ writes the whole tensor
```

Use the subscript form to write through the parameter:

| Spelling | Writes | Use when |
| -------- | ------ | -------- |
| `out[:] = <expr>` | the whole tensor | the result is the entire output |
| `out[<slices>] = <expr>` | that sub-window | writing part of the output |
| `out = pl.assemble(out, <expr>, <offset>)` | a window at `<offset>` | the explicit form the subscript sugar builds |
| `out = <expr>` | **nothing** | never — see the warning below |

Only the first and third are equivalent, and only when the slice covers the full
extent and `<offset>` is all zeros.

##### The `OutParamWriteDropped` warning

The compiler reports a bare assignment that drops the write:

```text
[warning] [OutParamWriteDropped] (pipeline_input) Assigning to Out parameter 'out'
in function 'main' rebinds the name only — the caller's buffer is never written.
Use 'out[:] = <expr>' to write the whole tensor, or 'out[<slices>] = <expr>' for
a sub-window. at repro.py:12:9
```

The check is data-flow based, not syntactic. A value can reach the parameter
without naming it — through a loop carry, for instance — and that is a genuine
write-through, so it stays silent:

```python
for col, (d,) in pl.range(0, n, chunk, init_values=(data,)):
    d = pl.store(local, [0, col], d)
    staged = pl.yield_(d)
data = pld.tensor.allreduce(staged, signal, ...)   # `staged` *is* `data`; no warning
```

The check is deliberately conservative: a value that merely *reads* the
parameter, such as `out = pl.add(out, b)`, is not reported even though it also
drops the write. Telling "reads the parameter" from "writes through the
parameter" needs per-op write semantics the operator registry does not record, and
a false warning on correct code costs more than a missed one. Write
`out[:] = pl.add(out, b)` when the whole tensor is meant.

Disable the check with `disabled_diagnostics` if it is not useful for your
program:

```python
disabled = passes.DiagnosticCheckSet()
disabled.insert(passes.DiagnosticCheck.OutParamWriteDropped)
ir.compile(program, disabled_diagnostics=disabled)
```

## How a `@pl.program` Class Is Located

`@pl.program` parses the class *body from source*, so it first has to find the
`class` statement that produced the decorated object. The class name alone does
not identify it: one function may define the same name in several branches, and
every one of them carries the same `__qualname__`.

The decorator resolves this from the line numbers of the methods in the class
body, so each branch is parsed from its own source:

```python
def make(case):
    if case == "add":
        @pl.program
        class Prog:                                # parsed from *this* body
            @pl.function
            def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                return pl.add(x, 1.0)
        return Prog

    @pl.program
    class Prog:                                    # and this one from *this* body
        @pl.function
        def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
            return pl.mul(x, 3.0)
    return Prog
```

When the definitions genuinely cannot be told apart, the decorator raises a
`ParserSyntaxError` naming every candidate line rather than picking one — a
wrong pick would compile a body you never wrote. Give the classes distinct
names, or define the class once and vary it through a closure variable:

```python
def make(scale):
    @pl.program
    class Prog:                                    # one definition, parameterised
        @pl.function
        def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
            return pl.mul(x, scale)
    return Prog
```

## Cross-Module Function Reuse

Functions defined outside a `@pl.program` class can be reused via two mechanisms.

### External `@pl.function` Calls

An externally-defined `@pl.function` can be called by name inside `@pl.program`. The function is automatically added to the Program and an `ir.Call(GlobalVar, args)` is emitted.

```python
@pl.function
def softmax(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...

@pl.program
class MyModel:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = softmax(x)   # ir.Call(GlobalVar("softmax"), [x])
        return y
```

**Rules:**

- Uses the function's `.name` as GlobalVar (aliases are transparent)
- External and internal function names must not conflict
- Two different externals with the same `.name` is an error
- Same external called from multiple methods is added once

### `@pl.inline` Decorator

`@pl.inline` captures a function for statement-level inlining. No function is added to the Program — the body is expanded at each call site.

```python
@pl.inline
def normalize(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    return result

@pl.program
class MyModel:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = normalize(x)  # statements inlined in-place
        return y
```

**Rules:**

- Argument count must match parameter list exactly
- Closure variables from the inline definition site are available
- Inline functions can be called multiple times (each expansion is independent)
- Nested inline calls are supported

## Complete Example

### Tensor Operations (Loop with iter_args)

```python
# pypto.program: my_program
import pypto.language as pl

def loop_sum(n: pl.INT64) -> pl.INT64:
    sum_init: pl.INT64 = 0
    for i, (sum,) in pl.range(n, init_values=(sum_init,)):
        sum = pl.yield_(sum + i)
    return sum
```

### Tile Operations (Tile-based computation)

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
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_b, [0, 0], [64, 64])
        tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
        result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
        return result
```

## Printing IR Nodes

Use `as_python()` on any IR node to get its Python representation:

```python
print(stmt.as_python())          # "x: pl.Scalar[pl.INT64] = a + b" (default "pl" prefix)
print(stmt.as_python("ir"))      # "x: ir.Scalar[ir.INT64] = a + b" (custom prefix)
```

### Concise Mode

Pass `concise=True` to omit intermediate type annotations. Function signature types (parameters and return) are always preserved:

```python
print(func.as_python())                  # verbose (default): type on every assignment
print(func.as_python(concise=True))      # concise: omits intermediate type annotations
```

Verbose output:

```python
def main(self, x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP16]:
    y: pl.Tensor[[64, 128], pl.FP32] = pl.some_op(x)
    result: pl.Tensor[[64, 128], pl.FP16] = pl.cast(y, pl.FP16)
    return result
```

Concise output:

```python
def main(self, x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP16]:
    y = pl.some_op(x)
    result = pl.cast(y, pl.FP16)
    return result
```

The free function `ir.python_print(node)` is also available and supports the same parameters.

## See Also

- [Python IR Syntax Specification](00-python_syntax.md) — types and expressions
- [Statements and Control Flow](01-statements.md) — statement forms inside a function body
- [Integrating Hand-Written C++ Kernels](04-external-kernels.md) — calling external kernels
