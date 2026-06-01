# Getting Started with PyPTO

## What is PyPTO?

PyPTO is a Python-based kernel programming framework for Ascend NPUs. You write compute kernels in Python using the `pypto.language` module, and PyPTO compiles them into optimized device code.

```python
import pypto.language as pl
from pypto import ir
```

All kernel code uses the `pl` namespace. The `ir` module provides compilation and IR utilities.

## Hello World: Vector Add (Tensor Level)

The simplest kernel operates on **Tensors** — high-level arrays in DDR memory. PyPTO automatically handles data movement and memory allocation.

```python
import pypto.language as pl
from pypto import ir

@pl.function
def vector_add(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
    return result
```

**Line by line:**

| Line | What it does |
| ---- | ------------ |
| `@pl.function` | Parses the Python function body into PyPTO IR |
| `a: pl.Tensor[[64], pl.FP32]` | Input: 1D tensor, 64 elements, 32-bit float |
| `pl.add(a, b)` | Element-wise addition (dispatches to tensor add) |
| `return result` | The function returns a tensor |

After decoration, `vector_add` is an `ir.Function` object — not a Python callable. Print the IR:

```python
print(vector_add.as_python())
```

## Tile Kernel: Load-Compute-Store

For hardware-level control, use **Tiles** — on-chip memory buffers. You explicitly load data from DDR, compute on-chip, and store results back.

```python
@pl.function
def vector_add_tile(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
    output: pl.Out[pl.Tensor[[64], pl.FP32]],
) -> pl.Tensor[[64], pl.FP32]:
    # Load from DDR → on-chip (Vec memory)
    a_tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
    b_tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])

    # Compute on-chip
    result: pl.Tile[[64], pl.FP32] = pl.add(a_tile, b_tile)

    # Store back to DDR
    out: pl.Tensor[[64], pl.FP32] = pl.store(result, [0], output)
    return out
```

**Key differences from the Tensor version:**

| Concept | Tensor level | Tile level |
| ------- | ------------ | ---------- |
| Data location | DDR (automatic) | Explicit load/store |
| Type | `pl.Tensor` | `pl.Tile` (on-chip) |
| Output parameter | Return value | `pl.Out[pl.Tensor[...]]` |
| Memory control | Compiler decides | You decide |

**`pl.load(tensor, offsets, shapes)`** copies a region from a DDR Tensor into an on-chip Tile.

**`pl.store(tile, offsets, output_tensor)`** copies a Tile back to DDR.

## Loops and Accumulation

Use `pl.range()` for loops. With `init_values`, you get loop-carried values (accumulators):

```python
@pl.function
def sum_elements(
    a: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[1], pl.FP32]:
    zero: pl.Tensor[[1], pl.FP32] = pl.create_tensor([1], dtype=pl.FP32)

    for i, (acc,) in pl.range(64, init_values=(zero,)):
        elem: pl.Tensor[[1], pl.FP32] = pl.slice(a, [1], [i])
        new_acc: pl.Tensor[[1], pl.FP32] = pl.add(acc, elem)
        acc_out: pl.Tensor[[1], pl.FP32] = pl.yield_(new_acc)

    return acc_out
```

**How `init_values` works:**

1. `init_values=(zero,)` — initial value for the accumulator
2. `for i, (acc,)` — `i` is the loop variable, `acc` is the current accumulator
3. `pl.yield_(new_acc)` — passes `new_acc` as the accumulator to the next iteration
4. After the loop, `acc_out` holds the final value

Simple loops without accumulators:

```python
for i in pl.range(10):
    # i goes from 0 to 9
    ...

for i in pl.range(0, 100, 2):
    # i goes from 0 to 98, step 2
    ...
```

## Multi-Function Programs

Use `@pl.program` to group multiple functions that call each other:

```python
@pl.program
class VectorAddProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        output: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        a_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
        b_tile: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
        result: pl.Tile[[128, 128], pl.FP32] = pl.add(a_tile, b_tile)
        out: pl.Tensor[[128, 128], pl.FP32] = pl.store(
            result, [0, 0], output
        )
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor(
            [128, 128], dtype=pl.FP32
        )
        c = self.kernel_add(a, b, c)
        return c
```

**Key concepts:**

| Concept | Description |
| ------- | ----------- |
| `@pl.program` | Decorates a class → becomes an `ir.Program` |
| `self` | Required first parameter; stripped from IR |
| `self.kernel_add(...)` | Cross-function call within the program |
| `FunctionType.InCore` | Runs on AICore (compute kernel) |
| `FunctionType.Orchestration` | Runs on host (task graph coordinator) |

**Function types:**

- **`Opaque`** (default) — no specific execution context
- **`InCore`** — AICore compute kernel; uses load/store for data movement
- **`Orchestration`** — host-side function that creates tensors and dispatches InCore tasks

## Compiling

Compile a program to generate device code:

```python
from pypto.backend import BackendType

output_dir = ir.compile(
    VectorAddProgram,
    strategy=ir.OptimizationStrategy.Default,
    dump_passes=True,
    backend_type=BackendType.Ascend910B,
)
print(f"Generated code in: {output_dir}")
```

**`ir.compile()` parameters:**

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `program` | (required) | The `ir.Program` to compile |
| `output_dir` | `None` → `<base>/<name>_<timestamp>` | Directory for codegen, reports, and (when dumping) pass IR. `<base>` is the `PYPTO_PROG_BUILD_DIR` env var, or `build_output` if unset |
| `strategy` | `OptimizationStrategy.Default` | Pass pipeline preset (`Default` or `DebugTileOptimization`) |
| `dump_passes` | `True` | When `True`, write IR snapshots under `output_dir/passes_dump/` after each pass |
| `backend_type` | `BackendType.Ascend910B` | Target hardware for passes and codegen (`Ascend910B` or `Ascend950`) |
| `skip_ptoas` | `False` | If `True`, skip the ptoas step and emit raw `.pto` (MLIR) instead of compiled C++ wrappers |
| `verification_level` | `None` | Optional `ir.VerificationLevel` override; `None` uses defaults (or `PYPTO_VERIFY_LEVEL`) |

`DebugTileOptimization` is a debug-only shortcut for inspecting the PTO tile
pipeline. Prefer `Default` unless you are explicitly debugging strategy
selection or pass ordering.

**Inspect IR without compiling:**

```python
# Print a single function
print(vector_add.as_python())

# Print an entire program
print(VectorAddProgram.as_python())

# Print without intermediate type annotations (concise mode)
print(vector_add.as_python(concise=True))
```

## Reusing weights on the worker (DeviceTensor)

When the same large tensor is consumed by many kernel invocations — e.g. a
weight matrix used across batches of a forward pass — uploading it on every
call wastes bandwidth. `ChipWorker.alloc_tensor` allocates persistent device
memory and returns a `DeviceTensor` handle that `CompiledProgram` accepts in
place of a `torch.Tensor`. The runtime treats the buffer as already resident
and skips both H2D and D2H copies for that argument.

```python
import torch
from pypto import ir
from pypto.runtime import ChipWorker, RunConfig

compiled = ir.compile(MyKernel)

with ChipWorker(config=RunConfig(platform="a2a3sim")) as w:
    weight = w.alloc_tensor((1024, 4096), torch.float16, init=host_weight)
    for batch in batches:
        out = torch.empty(batch.shape[0], 4096, dtype=torch.float16)
        compiled(batch, weight, out)
    w.free_tensor(weight)
```

### Caveats

- A `DeviceTensor` is never copied back to the host. If a kernel writes to
  one, call `w.copy_from(host_ptr, t.data_ptr, t.nbytes)` on the same
  ChipWorker instance to read the result.
- Free the handle with `w.free_tensor(t)` before the ChipWorker is closed,
  otherwise the memory leaks for the lifetime of the ChipWorker.
- Only the ChipWorker instance that allocated the buffer can use it.

### Distributed (L3+) programs

L3+ distributed programs returned by `ir.compile` (a `DistributedCompiledProgram`)
accept `DeviceTensor` arguments the same way as `CompiledProgram`: pass a
worker-resident buffer in place of a `torch.Tensor` and the runtime skips H2D/D2H
for that argument. This is the recommended way to keep large static weights
resident across the many dispatches of a generate loop.

```python
import torch
from pypto.runtime import DeviceTensor

compiled = ir.compile(MyDistributedProgram)   # returns DistributedCompiledProgram
weight = DeviceTensor(dev_ptr, (1024, 4096), torch.float16)   # caller-managed buffer
compiled(x, weight, out)                       # weight: no H2D/D2H copy
```

#### Reusing setup across dispatches (`prepare()`)

`compiled(*args)` runs the full distributed setup (per-chip assembly, simpler
Worker construction + fork) on every call. For a resident service that
dispatches the same program many times (e.g. a generate loop), call
`compiled.prepare()` once to get a `DistributedWorker` handle that runs setup
once and dispatches many times on the same worker.

Per-call IO buffers (inputs **and** outputs) are **shared-memory host tensors
allocated before `prepare()`** and reused in place — the forked chip worker
reads/writes them through the inherited mapping, so you read the output straight
back from the tensor. Large static weights are uploaded once to a worker-resident
`DeviceTensor` via `rt.alloc_tensor` (its `init` source must also be a pre-`prepare`
shared tensor) and mixed in. A non-shared host tensor (or one allocated after
`prepare()`) is rejected — the chip worker would not see it.

```python
compiled = ir.compile(MyDistributedProgram)

# shared-memory host buffers — allocated BEFORE prepare()
host_x = torch.zeros((seq, 4096), dtype=torch.float16).share_memory_()
host_out = torch.zeros((seq, 4096), dtype=torch.float16).share_memory_()
host_weight = load_weight().share_memory_()

with compiled.prepare() as rt:                  # setup runs once
    weight = rt.alloc_tensor(host_weight.shape, host_weight.dtype, init=host_weight)
    for step in generate_steps:
        host_x.copy_(next_input(step))          # refresh input in place
        rt(host_x, weight, host_out)            # host shm IO + resident weight
        consume(host_out)                       # read output directly
    rt.free_tensor(weight)
# rt.close() runs on exit
```

## What's Next

- **[Language Guide](01-language_guide.md)** — complete reference for types, operations, control flow, memory, and compilation
- **[Operation Reference](02-operation_reference.md)** — lookup tables for every `pl.*` operation
