# PyPTO Language Guide

Complete reference for the `pypto.language` (`pl`) module.

## Type System

### Data Types

| Constant | Bits | Description |
| -------- | ---- | ----------- |
| `pl.BOOL` | 1 | Boolean |
| `pl.INT4` / `pl.UINT4` | 4 | Signed / unsigned 4-bit integer |
| `pl.INT8` / `pl.UINT8` | 8 | Signed / unsigned 8-bit integer |
| `pl.INT16` / `pl.UINT16` | 16 | Signed / unsigned 16-bit integer |
| `pl.INT32` / `pl.UINT32` | 32 | Signed / unsigned 32-bit integer |
| `pl.INT64` / `pl.UINT64` | 64 | Signed / unsigned 64-bit integer |
| `pl.FP16` | 16 | IEEE half-precision float |
| `pl.BF16` | 16 | Brain float 16 |
| `pl.FP32` | 32 | IEEE single-precision float |
| `pl.FP4` | 4 | 4-bit float |
| `pl.FP8E4M3FN` | 8 | 8-bit float (e4m3fn) |
| `pl.FP8E5M2` | 8 | 8-bit float (e5m2) |
| `pl.HF4` / `pl.HF8` | 4/8 | Hisilicon float formats |
| `pl.INDEX` | 64 | Index type (alias for INT64) — loop vars, dimensions |

### Container Types

**`pl.Tensor[[shape], dtype]`** — DDR memory array (off-chip global memory).

```python
x: pl.Tensor[[64, 128], pl.FP32]        # 2D, 64×128, float32
y: pl.Tensor[[256], pl.FP16]            # 1D, 256 elements, float16
z: pl.Tensor[[64, 128], pl.FP16, pl.NZ] # With NZ layout
```

**`pl.Tile[[shape], dtype]`** — on-chip memory buffer (unified buffer by default).

```python
t: pl.Tile[[64, 64], pl.FP32]           # 2D tile, 64×64
```

**`pl.Scalar[dtype]`** — single scalar value.

```python
s: pl.Scalar[pl.FP32]                   # float32 scalar
idx: pl.Scalar[pl.INDEX]                # index scalar
```

### Tensor Layouts

Layouts control the physical memory arrangement of Tensors:

| Layout | Description |
| ------ | ----------- |
| `pl.ND` | N-Dimensional (default, row-major) |
| `pl.DN` | DN layout |
| `pl.NZ` | NZ fractal format (hardware-specific tiling) |

```python
# Specify layout as third type parameter
a: pl.Tensor[[64, 128], pl.FP16, pl.NZ]
```

### Dynamic Shapes

Use `pl.dynamic()` for dimensions determined at runtime:

```python
M = pl.dynamic("M")
N = pl.dynamic("N")

@pl.function
def dynamic_kernel(
    a: pl.Tensor[[M, N], pl.FP32],
) -> pl.Tensor[[M, N], pl.FP32]:
    ...
```

### Parameter Directions

By default, parameters are read-only inputs. Use wrappers for output parameters:

| Direction | Syntax | Description |
| --------- | ------ | ----------- |
| Input (default) | `a: pl.Tensor[...]` | Read-only |
| Output | `a: pl.Out[pl.Tensor[...]]` | Write-only output |
| In/Out | `a: pl.InOut[pl.Tensor[...]]` | Read-write |

```python
@pl.function
def kernel(
    input_a: pl.Tensor[[64], pl.FP32],                    # In
    output_b: pl.Out[pl.Tensor[[64], pl.FP32]],            # Out
    accum_c: pl.InOut[pl.Tensor[[64], pl.FP32]],           # InOut
) -> pl.Tensor[[64], pl.FP32]:
    ...
```

## Operations

### Dispatch Model

PyPTO operations exist at three levels:

| Namespace | Level | Description |
| --------- | ----- | ----------- |
| `pl.*` | Unified | Auto-dispatches based on input type (Tensor or Tile) |
| `pl.tensor.*` | Tensor | DDR-level operations on `Tensor` objects |
| `pl.block.*` | Tile | On-chip operations on `Tile` objects |

**Recommended:** Use `pl.*` (unified) when possible. The dispatcher picks the right implementation.

```python
# Unified — works with both Tensor and Tile
result = pl.add(a, b)       # dispatches to tensor.add or block.add
result = pl.mul(a, scalar)   # dispatches to tensor.mul_scalar or block.muls

# Explicit tile-level (when you need tile-specific ops)
tile = pl.block.load(tensor, [0, 0], [64, 64])
tile = pl.block.adds(tile, 1.0)
```

### Python Operators

Standard Python operators map to IR operations:

| Python | IR operation | Example |
| ------ | ------------ | ------- |
| `a + b` | `add` | `c = a + b` |
| `a - b` | `sub` | `c = a - b` |
| `a * b` | `mul` | `c = a * b` |
| `a / b` | `div` | `c = a / b` |
| `a == b` | `eq` (compare) | `if a == 0:` |
| `a != b` | `ne` (compare) | `if a != 0:` |
| `a < b` | `lt` (compare) | `if a < n:` |
| `a > b` | `gt` (compare) | `if a > 0:` |

### Unified Operations

Common `pl.*` operations — see [Operation Reference](02-operation_reference.md) for the complete list:

```python
c = pl.add(a, b)            # arithmetic (also sub, mul, div)
c = pl.add(a, 1.0)          # scalar rhs auto-detected
c = pl.cast(a, pl.FP16)     # type cast
c = pl.reshape(a, [16, 8])  # shape operations (also transpose, view)
c = pl.matmul(a, b)         # linear algebra
c = pl.row_sum(a)            # reductions (also row_max)
```

Use `pl.block.*` for tile-specific operations (memory transfers, broadcast, bitwise, etc.).

## Variable Assignment and SSA

PyPTO's IR supports both **SSA** (Static Single Assignment) and **non-SSA** forms. In SSA form, every variable is assigned exactly once; in non-SSA form, you can reassign the same variable name multiple times.

### Writing Style

**Non-SSA (default)** — reassign variables freely, like normal Python:

```python
@pl.function
def example(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    result: pl.Tensor[[64], pl.FP32] = pl.add(result, 1.0)  # reassignment OK
    return result
```

**SSA style** — each variable assigned once, using unique names:

```python
@pl.function
def example(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result_0: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    result_1: pl.Tensor[[64], pl.FP32] = pl.add(result_0, 1.0)
    return result_1
```

Both produce valid IR. Use whichever style you prefer.

### Automatic SSA Conversion

Most optimization passes require SSA form. The compilation pipeline automatically runs `ConvertToSSA` early in the pipeline, so you don't need to worry about it — write non-SSA code and the compiler handles the conversion.

### Strict SSA Mode

Pass `strict_ssa=True` to enforce SSA at parse time. The parser will raise an error if you reassign a variable:

```python
@pl.function(strict_ssa=True)
def example(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    result: pl.Tensor[[64], pl.FP32] = pl.add(result, 1.0)  # ERROR: SSAViolationError
    return result
```

This is useful for catching unintended variable shadowing, but is entirely optional.

### Why `yield_` Exists

In SSA form, control flow (loops, if/else) cannot simply reassign a variable — each assignment must be unique. `pl.yield_()` is the mechanism that carries values out of a control flow scope:

- **Loops**: `pl.yield_()` passes the updated accumulator to the next iteration
- **If/else**: `pl.yield_()` in both branches creates a merge point (phi node), producing a single result variable

This is why loops with accumulators require `init_values` + `yield_`, and why if/else branches that produce values must both `yield_`.

## Control Flow

### For Loops — `pl.range()`

**Simple loop:**

```python
for i in pl.range(10):
    # i = 0, 1, 2, ..., 9
    ...

for i in pl.range(2, 10):
    # i = 2, 3, ..., 9
    ...

for i in pl.range(0, 100, 4):
    # i = 0, 4, 8, ..., 96
    ...
```

**Loop with accumulators (`init_values`):**

Accumulators carry values across iterations. Each iteration receives the previous values and must `yield_` new ones:

```python
@pl.function
def sum_16_elements(data: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[1], pl.FP32]:
    init_sum: pl.Tensor[[1], pl.FP32] = pl.create_tensor([1], dtype=pl.FP32)

    for i, (running_sum,) in pl.range(16, init_values=(init_sum,)):
        chunk: pl.Tensor[[1], pl.FP32] = pl.view(data, [1], [i])
        new_sum: pl.Tensor[[1], pl.FP32] = pl.add(running_sum, chunk)
        sum_out: pl.Tensor[[1], pl.FP32] = pl.yield_(new_sum)

    # sum_out holds the final accumulated value after the loop
    return sum_out
```

**Multiple accumulators:**

```python
@pl.function
def find_max_and_sum(
    data: pl.Tensor[[4, 64], pl.FP32],
) -> pl.Tensor[[1, 64], pl.FP32]:
    init_max: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)
    init_sum: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)

    for i, (acc_max, acc_sum) in pl.range(4, init_values=(init_max, init_sum)):
        row: pl.Tensor[[1, 64], pl.FP32] = pl.view(data, [1, 64], [i, 0])
        new_max: pl.Tensor[[1, 64], pl.FP32] = pl.maximum(acc_max, row)
        new_sum: pl.Tensor[[1, 64], pl.FP32] = pl.add(acc_sum, row)
        out_max, out_sum = pl.yield_(new_max, new_sum)

    return out_sum
```

### Parallel Loops — `pl.parallel()`

Same syntax as `pl.range()`, but iterations may execute in parallel:

```python
for i in pl.parallel(0, num_blocks):
    # iterations are independent, can run in parallel
    ...
```

### While Loops — `pl.while_()`

Always requires `init_values`. The condition is set with `pl.cond()` as the **first statement** in the loop body:

```python
for (x,) in pl.while_(init_values=(0,)):
    pl.cond(x < 10)          # continue while x < 10
    new_x = x + 1
    x_out = pl.yield_(new_x)
```

### If/Else with `pl.yield_()`

Branches that produce values must `yield_` them. This creates SSA phi nodes — both branches must yield the same number and type of values:

```python
@pl.function
def conditional_update(
    a: pl.Tensor[[64], pl.FP32],
    delta: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[64], pl.FP32]:
    init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)

    for i, (prev,) in pl.range(4, init_values=(init,)):
        if i == 0:
            result: pl.Tensor[[64], pl.FP32] = pl.yield_(a)
        else:
            updated: pl.Tensor[[64], pl.FP32] = pl.add(prev, delta)
            result: pl.Tensor[[64], pl.FP32] = pl.yield_(updated)
        # result holds whichever branch executed
        out: pl.Tensor[[64], pl.FP32] = pl.yield_(result)

    return out
```

**Rule:** If one branch yields, the other must too. Both yield the same number of values.

## Programs and Functions

### `@pl.function`

Parses a Python function into IR:

```python
@pl.function
def my_func(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...
```

With function type:

```python
@pl.function(type=pl.FunctionType.InCore)
def compute_kernel(...):
    ...

@pl.function(type=pl.FunctionType.Orchestration)
def task_graph(...):
    ...
```

| Function Type | Description | Typical Use |
| ------------- | ----------- | ----------- |
| `Opaque` | No specified context (default) | Standalone functions |
| `InCore` | AICore compute kernel | Load/compute/store patterns |
| `Orchestration` | Host-side coordinator | Create tensors, dispatch InCore tasks |

### `@pl.program`

Groups multiple functions into a program that can be compiled:

```python
@pl.program
class MyProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, ...):
        ...

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, ...):
        result = self.kernel(...)   # cross-function call
        return result
```

**Rules:**

- Every method must have `self` as first parameter (stripped from IR)
- Cross-function calls use `self.method_name(...)`
- The decorated class becomes an `ir.Program`, not a Python class

### `@pl.inline`

Defines a function whose body is expanded at each call site (no separate function in the program):

```python
@pl.inline
def normalize(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    return result

@pl.program
class MyProgram:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = normalize(x)  # body inlined here
        return y
```

### External Function Calls

A standalone `@pl.function` can be called from within a `@pl.program`. It is added to the program as a separate function:

```python
@pl.function
def softmax(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...

@pl.program
class Model:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = softmax(x)  # call to external function
        return y
```

### InCore Scopes

Mark a code region as InCore execution without making a separate function:

```python
with pl.incore():
    y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
```

## Memory and Data Movement

### Memory Hierarchy

```text
DDR (off-chip, global memory)
 │
 ├── Vec (unified buffer, on-chip)     ← pl.load() / pl.store()
 │    └── Compute (vector operations)
 │
 ├── Mat (L1 buffer)                   ← pl.load(..., target_memory=pl.MemorySpace.Mat)
 │    ├── Left (L0A)                   ← pl.move(..., target_memory=pl.MemorySpace.Left)
 │    └── Right (L0B)                  ← pl.move(..., target_memory=pl.MemorySpace.Right)
 │         └── Acc (L0C)              ← pl.matmul() result
 │              └── DDR               ← pl.store()
```

### Memory Spaces

| Space | Enum | Description |
| ----- | ---- | ----------- |
| DDR | `MemorySpace.DDR` | Off-chip global memory (Tensor parameters) |
| Vec | `MemorySpace.Vec` | Unified vector buffer (default for `pl.load`) |
| Mat | `MemorySpace.Mat` | L1 matrix buffer |
| Left | `MemorySpace.Left` | L0A — left matmul operand |
| Right | `MemorySpace.Right` | L0B — right matmul operand |
| Acc | `MemorySpace.Acc` | L0C — matmul accumulator |

### Data Movement Operations

```python
tile = pl.load(tensor, [0, 0], [64, 64])                                  # DDR → Vec
tile_l1 = pl.load(tensor, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)  # DDR → Mat
tile_l0a = pl.move(tile_l1, target_memory=pl.MemorySpace.Left)            # Mat → Left
out = pl.store(tile, [0, 0], [64, 64], output)                            # Tile → DDR
```

### Pattern: Matrix Multiply (DDR → Mat → Left/Right → Acc → DDR)

```python
a_l1 = pl.load(a, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
b_l1 = pl.load(b, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
a_l0a = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
b_l0b = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
c_acc = pl.matmul(a_l0a, b_l0b)                     # result → Acc
out = pl.store(c_acc, [0, 0], [32, 32], output)      # Acc → DDR
```

## Compilation

### `ir.compile()`

```python
from pypto import ir
from pypto.backend import BackendType

output_dir = ir.compile(
    program,
    output_dir=None,                           # auto-generated if None
    strategy=ir.OptimizationStrategy.Default,  # Default or PTOAS
    dump_passes=True,                          # print IR after each pass
    backend_type=BackendType.PTO,              # PTO or CCE
)
```

| Parameter | Options | Description |
| --------- | ------- | ----------- |
| `strategy` | `Default`, `PTOAS` | `Default` = full pipeline with sync insertion. `PTOAS` = PTO assembly (no scheduling) |
| `backend_type` | `PTO`, `CCE` | Code generator backend |
| `dump_passes` | `True`/`False` | Print IR before/after each optimization pass |
| `skip_ptoas` | `True`/`False` | Skip PTOAS step, emit raw MLIR files (default `False`) |
| `output_dir` | path or `None` | Output directory (auto-created if `None`) |
| `verification_level` | `NONE`, `BASIC` | IR verification level (`BASIC` is default) |

### Optimization Pipeline

The `Default` strategy runs these passes in order:

1. **ConvertToSSA** — convert to static single assignment form
2. **FlattenCallExpr** — flatten nested function calls
3. **RunVerifier** — verify IR structural integrity
4. **InitMemRef** — assign memory spaces, insert buffer allocations
5. **MemoryReuse** — share buffers with non-overlapping lifetimes
6. **InsertSync** — insert synchronization barriers between pipeline stages
7. **AllocateMemoryAddr** — assign concrete memory addresses

### Debugging

Use `ir.python_print()` to inspect IR for functions or programs. Compile with `dump_passes=True` to see IR at each optimization stage.
