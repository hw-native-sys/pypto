# Python IR Syntax Specification

## Overview

Python-style syntax for PyPTO IR:

- **Complete**: All information needed to reconstruct IR
- **Parseable**: Can be parsed back into IR (see [IR Parser](../ir/07-parser.md))
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
| -------- | ----- |
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
memref = pl.MemRef(addr_expr, 1024, 0)

# Memory spaces: DDR, Vec, Mat, Left, Right, Acc
# Note: pl.Mem is a short alias for pl.MemorySpace

# Tensor with memref
tensor: pl.Tensor[[64, 128], pl.FP32, pl.MemRef(addr_expr, 8192, 0)]

# Tiles keep memory space on the tile annotation, not inside MemRef
tile: pl.Tile[[16, 16], pl.FP16, pl.MemRef(addr_expr, 512, 0), pl.Mem.Left]
```

### Tile Views (TileView)

```python
# Create TileView
valid_shape = [pl.ConstInt(16, pl.INT64, span)] * 2
stride = [pl.ConstInt(1, pl.INT64, span), pl.ConstInt(16, pl.INT64, span)]
start_offset = pl.ConstInt(0, pl.INT64, span)
tile_view = pl.TileView(valid_shape=valid_shape, stride=stride, start_offset=start_offset)

# Tile with memref and tile_view
tile: pl.Tile[
    [16, 16], pl.FP16,
    pl.MemRef(addr_expr, 512, 0), pl.Mem.Left,
    pl.TileView(valid_shape=..., stride=..., start_offset=...)
]
```

**Notes:**

- Omitting `pl.TileView(...)` does **not** mean "no TileView semantics". The DSL infers an implicit
  TileView from the tile shape and, when present, the tile memory space.
- In that implicit form, `valid_shape` defaults to the tile shape. Layout/fractal defaults are also
  inferred from the shape / memory-space combination.
- An explicit `pl.TileView()` (or one that only repeats those implicit defaults) is treated as
  semantically equivalent to the omitted form. Parser / printer roundtrips may canonicalize both
  forms to the same printed syntax.

## Expressions

### Variables and Constants

```python
x              # Variable reference
tensor_a       # Tensor variable
42             # Integer literal
3.14           # Float literal
```

**Closure variables:** Names not found in the DSL scope are resolved from the enclosing Python scope. Supported types: `int`, `float`, `bool`, `list`, `tuple`, and IR expressions.

```python
OFFSET = [0, 0]
TILE_SHAPE = [64, 64]

@pl.function
def func(t: pl.Tensor[[128, 128], pl.FP32], out: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[128, 128], pl.FP32]:
    a: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(t, OFFSET, TILE_SHAPE)  # closure vars as positional args
    ...
```

### Binary Operations

| Python Operator | PyPTO IR | Category |
| --------------- | -------- | -------- |
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
| `&` | BitAnd | Bitwise |
| `\|` | BitOr | Bitwise |
| `<<`, `>>` | BitShiftLeft, BitShiftRight | Bitwise |

**Note:** `and`/`or` are parsed from Python's `ast.BoolOp` syntax. Chained expressions like `a and b and c` are folded left-to-right into `And(And(a, b), c)`. Unlike Python, IR `And`/`Or` nodes evaluate both operands (no short-circuit semantics). The corresponding IR factory functions are `ir.and_(lhs, rhs)` and `ir.or_(lhs, rhs)`.

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
pl.tensor.add(a, b)                  # Tensor addition
pl.tile.load(t, [0, 0], [64, 64])      # Tile load

# Unified dispatch (auto-selects tensor/tile based on input type)
pl.add(a, b)                          # Tensor or Tile — dispatched automatically
pl.mul(tile, 2.0)                     # Tile + scalar → tile.muls
pl.exp(tile)                          # Tile → tile.exp

# Promoted ops (single-module ops accessible at pl.*)
pl.load(t, [0, 0], [64, 64])            # Promoted from block
pl.create_tensor([64], dtype=pl.FP32)       # Promoted from tensor

# System operations (synchronization primitives)
pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
pl.system.bar_v()                        # Vector barrier
pl.system.bar_m()                        # Matrix barrier
pl.system.bar_all()                      # Global barrier

# Cross-core operations (TPUSH/TPOP protocol)
pl.tpush_to_aic(tile0, split=0, id=0)        # Vector → Cube push on pipe 0
pl.tpush_to_aic(tile1, split=0, id=1)        # Vector → Cube push on pipe 1
tile0 = pl.tpop_from_aiv(split=0, id=0)      # Cube pops from Vector pipe 0
tile1 = pl.tpop_from_aiv(split=0, id=1)      # Cube pops from Vector pipe 1
pl.tfree_to_aiv(tile0, id=0)                 # Release slot to Vector pipe 0
pl.tfree_to_aiv(tile1, id=1)                 # Release slot to Vector pipe 1

# Cross-core pipe initialization and buffer management
buf = pl.reserve_buffer(name="slot_buf", size=4096, base=pl.AUTO)
peer = pl.import_peer_buffer(name="slot_buf", peer_func="other_func")
pl.aic_initialize_pipe(pl.const(0, pl.INT32), buf, dir_mask=2, slot_size=512, id=0)
pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size=512, id=0)
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
# Simple loop (1-3 positional args, like Python's range())
for i in pl.range(stop):                    # start=0, step=1
for i in pl.range(start, stop):             # step=1
for i in pl.range(start, stop, step):       # explicit

# Loop with iter_args (loop-carried values)
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(n, init_values=(sum_init,)):
    sum = pl.yield_(sum + i)
sum_final = sum

# Parallel for loop (same 1-3 arg forms)
for i in pl.parallel(stop):
for i in pl.parallel(start, stop, step):
    body_statements
```

**Key points:** Loop-carried values use `pl.range()` or `pl.parallel()` with `init_values`, tuple unpacking `(sum,)` declares iter_args, `pl.yield_()` updates values for next iteration, after loop iter_args contain final values. `pl.parallel()` produces a `ForKind.Parallel` loop while `pl.range()` produces `ForKind.Sequential` (default).

#### Chunked Loops

```python
# Split loop into chunks of C iterations (nested outer/inner loops)
for i in pl.range(10, chunk=5):
    body_statements

for i in pl.parallel(8, chunk=4):
    body_statements

for i in pl.unroll(12, chunk=4):
    body_statements
```

**Key points:** `chunk=C` splits the loop into an outer sequential loop and an inner loop of `C` iterations. The inner loop preserves the original kind (Sequential/Parallel/Unroll). `init_values` is supported with chunked loops (iter_args thread through the generated outer/inner/remainder loops). `chunk=` loops are only valid inside a `with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):` — outside that scope the parser rejects them with an error. See [SplitChunkedLoops Pass](../passes/07-split_chunked_loops.md).

### While Loop (SSA-style with iter_args)

```python
# Natural while: condition is the while-header expression
i: pl.Scalar[pl.INT64] = 0
while i < n:
    i = i + 1

# SSA form with init_values: header tuple = iter_args, first stmt is pl.cond().
# yield-LHS supplies the post-loop binding name (mirrors pl.range).
x_init: pl.Scalar[pl.INT64] = 0
for (x,) in pl.while_(init_values=(x_init,)):
    pl.cond(x < n)
    x_next = pl.yield_(x + 1)
# `x_next` is bound here (from the yield-LHS); `x` is loop-scoped only.

# Pre-SSA: no pl.yield_ at all; ConvertToSSA synthesizes it later.
for (x,) in pl.while_(init_values=(x_init,)):
    pl.cond(x < n)
    x = x + 1

# ❌ Bare pl.yield_(...) with non-empty init_values is rejected at parse time:
#    for (x,) in pl.while_(init_values=(x_init,)):
#        pl.cond(x < n)
#        pl.yield_(x + 1)             # ParserSyntaxError: requires assignment-form pl.yield_
```

**Key points:** `pl.while_(init_values=(...,))` reuses the `for ... in` header for SSA-style loops; the first body statement must be `pl.cond(<bool>)`. The post-loop binding name comes from the **yield-LHS** (`x_next` above), not the header tuple — header-tuple names are scoped to the loop body only. This convention is **uniform with `pl.range`**: assignment-form yield is required whenever `init_values` is non-empty AND the body contains a `pl.yield_(...)` call. Pre-SSA loops with no yield at all are still valid (last form above).

### Scope Context Managers

| Form | Scope Kind | Notes |
| ---- | ---------- | ----- |
| `pl.at(level=pl.Level.CORE_GROUP)` | `InCore` | Fixed-boundary outline at CORE_GROUP |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(MODE)])` | `InCore` | InCore + cross-core split hint |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk])` | `AutoInCore` | Compiler-driven chunked loop split |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk, pl.split(MODE)])` | `AutoInCore` | AutoInCore + split hint (independent entries) |
| `pl.at(level=pl.Level.HOST)` *(or any non-`CORE_GROUP` level)* | `Hierarchy` | Distributed hierarchy scope |
| `pl.cluster()` | `Cluster` | Co-scheduled AIC+AIV group |
| `pl.manual_scope()` | `Runtime(manual=true)` | Orchestrator region where the user manages task ordering — see [Manual dependency primitives](#manual-dependency-primitives) |
| `pl.incore()` *(deprecated)* | `InCore` | Use `pl.at(level=pl.Level.CORE_GROUP)` instead |
| `pl.auto_incore(split=...)` *(deprecated)* | `AutoInCore` | Use `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk, pl.split(...)])` |
| `pl.at(..., optimization=pl.chunked_loop_optimizer[(split=...)])` *(deprecated)* | `AutoInCore` | Use `pl.at(..., optimizations=[pl.auto_chunk, pl.split(...)])` |
| `pl.at(..., split=...)` *(deprecated)* | `InCore` | Use `pl.at(..., optimizations=[pl.split(...)])` |

See [Language Guide](../../user/01-language_guide.md#incore-scopes) for examples.

### Manual dependency primitives

By default the runtime auto-derives task→task dependencies from buffer
read/write overlap (the `OverlapMap`). Two complementary primitives let the
user opt out and manage ordering explicitly:

| Surface | Granularity | Effect |
| ------- | ----------- | ------ |
| `pl.no_dep(arg)` | per-call argument | At a kernel call site, the wrapped argument's `ArgDirection` becomes `NoDep`. Auto-tracking ignores that slot for this submission only. **Auto scope only** — has no semantic effect inside `pl.manual_scope`. |
| `with pl.manual_scope():` | per-region | Lowers to `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`. Inside, the runtime never auto-tracks; the user must declare **every** required ordering edge explicitly via `deps=[...]`. |
| `kernel(..., deps=[var, ...])` | per-call (manual_scope only) | Declares explicit task-id edges. Each entry must be a tensor `Var` produced by a prior `self.kernel(...)` in the same `manual_scope`, an iter_arg/return-var carrying such a producer through a loop, or a function parameter passed in as an initial seed. |

Manual scope is **declare-only**: omitting a `deps=[scratch]` on `stage2` will not be backfilled by the pass, even if `stage2` reads what `stage1` wrote. Auto-derivation from data flow was removed because it produced false edges whenever a buffer was reused in-place across unrelated kernels.

```python
@pl.function(type=pl.FunctionType.Orchestration)
def main(self, x: pl.Tensor[[64], pl.FP32],
         scratch: pl.Out[pl.Tensor[[64], pl.FP32]],
         out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
    with pl.manual_scope():
        scratch = self.stage1(x, scratch)
        out     = self.stage2(scratch, out, deps=[scratch])   # required: stage2 reads scratch
    return out
```

Inside `with pl.manual_scope():`, the
[DeriveManualScopeDeps](../passes/33-derive_manual_scope_deps.md)
pass (the public name for the underlying `LowerManualDepsToTaskId` lowering)
resolves each `deps=[var, ...]` entry to a TaskId companion of the producer,
attaches it to the call's `manual_dep_edges` attr, and threads matching TaskId
iter_args / return-vars / yields through every enclosing `pl.range` /
`pl.parallel`. There is no cap on the number of edges per call — codegen
sizes the emitted `ArgWithDeps<N>` to the exact dep count.

`pl.no_dep(arg)` is an auto-scope primitive; inside `pl.manual_scope` it has
no effect since the pass never inspects per-arg directions for dependency
resolution.

#### `pl.parallel` under manual scope: array-carry fence

When a manual-dep edge is carried through a `pl.parallel` loop (i.e. the
loop's iter_arg holds the TaskId being depended on), the orchestration codegen
treats the corresponding TaskId iter_arg as **an array of size equal to the
parallel loop's trip count**. Each parallel iteration writes its own slot,
and downstream consumers depend on **every** slot (not just the
last-dispatched task). This guarantees the user-declared fence semantics
even when iters finish out of dispatch order.

Requirements for the array-carry path:

- The `pl.parallel` trip count must be a Python literal (statically known).
  A dynamic trip count under `pl.parallel` carrying a manual dep is rejected
  at codegen with a "statically-known trip count" message.

```python
with pl.manual_scope():
    for phase in pl.range(N_PHASES):
        for branch in pl.parallel(N_BRANCHES):           # const trip count
            row = (phase * N_BRANCHES + branch) * TILE_M
            out = self.kernel_stripe(data, row, 1.0, out, deps=[out])
```

Each task in phase `N+1` waits for all `N_BRANCHES` tasks of phase `N`,
not just the last one.

### Yield Statement

```python
yield            # No values
yield x          # Single value
yield x, y       # Multiple values
```

### Break and Continue

```python
break              # Exit innermost loop
continue           # Skip to next iteration
```

**Restrictions:** Only valid when the **innermost** enclosing loop is sequential (`pl.range`) or `while`. Not supported when the innermost loop is `pl.parallel()` or `pl.unroll()`. A `break` in an inner `pl.range` loop nested inside an outer `pl.parallel` loop is valid. **Note:** Codegen backend support for `break`/`continue` is tracked in [#448](https://github.com/hw-native-sys/pypto/issues/448).

### Compile-Time Debugging

`pl.static_print()` and `pl.static_assert()` are parse-time-only constructs for inspecting IR state and asserting conditions during parsing. They produce **no IR**.

```python
@pl.function
def func(x: pl.Tensor[[128, 64], pl.FP16]) -> pl.Tensor[[128, 64], pl.FP16]:
    pl.static_print("input:", x)          # → static_print [file:line]: input: x: pl.Tensor[[128, 64], pl.FP16]
    pl.static_print(f"input: {x}")        # → static_print [file:line]: input: x: pl.Tensor[[128, 64], pl.FP16]
    pl.static_assert(True)                # passes silently
    pl.static_assert(N > 32, "N too small")  # checks closure variable N at parse time
    return x
```

| Function | Purpose | On failure |
| -------- | ------- | ---------- |
| `pl.static_print(*args)` | Print variable types/values to stdout | Requires ≥1 argument |
| `pl.static_assert(cond, msg="")` | Assert compile-time condition | Raises `ParserError` |

**Key points:**

- Both are statement-only (cannot be used in expressions)
- `static_print` accepts variables, constants, string labels (printed as-is), and f-strings with plain `{expr}` placeholders (formatted as IR). Conversions (`!r`, `!s`, `!a`) and format specs (`:...`) are not supported.
- `static_assert` supports closure variable expressions (e.g. `N > 32`) and IR constants; message must be a string literal
- Output appears even if parsing fails later — useful for debugging parse errors

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
| ---- | ----- | ----------- |
| `pl.FunctionType.Opaque` | Default | Unspecified function type |
| `pl.FunctionType.Orchestration` | Host/AICPU | Control flow and dependency analysis |
| `pl.FunctionType.InCore` | AICore | Sub-graph on specific AICore (unspecialized) |
| `pl.FunctionType.AIC` | Cube core | Cube core kernel (specialized InCore) |
| `pl.FunctionType.AIV` | Vector core | Vector core kernel (specialized InCore) |
| `pl.FunctionType.Group` | Multi-core | Co-scheduled group of AIC + AIV kernels |

When no type is specified, functions default to `Opaque`.

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
for i, (sum,) in pl.range(10, init_values=(sum_init,)):
    sum = pl.yield_(sum + i)
sum_final: pl.INT64 = sum  # captures final value
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

## References

- [IR Overview](../ir/00-overview.md) - Core IR structures
- [IR Parser](../ir/07-parser.md) - Parsing Python syntax back to IR
- [Operator Registration](../ir/05-operators.md) - Op system and type inference
