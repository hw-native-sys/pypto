# Quickstart

Write, inspect, and compile your first PyPTO kernels — from a one-line tensor add to a multi-function program.

> **Prerequisites:** PyPTO installed and importable — see [Installation](01-installation.md).
> Nothing on this page dispatches to a device, so no NPU is needed. The kernel-authoring
> examples need only the install; the [compiling](#compiling) example additionally needs
> **ptoas**, which `pip` does not install — pass `skip_ptoas=True` if you do not have it.

## Concept

A PyPTO kernel is Python source that is *parsed*, not executed. `@pl.function`
reads the decorated body and builds PyPTO IR from it; the resulting object is an
`ir.Function`, not a callable. Nothing runs until you compile the IR and dispatch it.

That is why the examples below use type annotations everywhere. The annotations are
not documentation — they are the shape and dtype contract the parser reads to build
the IR. A missing annotation is a missing piece of the program.

Two abstraction levels appear on this page. At **tensor level** you name whole arrays
in DDR and let the compiler place data and insert movement. At **tile level** you name
on-chip buffers and move data yourself. Tensor level is where you start; tile level is
where you go when you need control over what sits on chip and when.

```python
import pypto.language as pl
from pypto import ir
```

`pl` is the language surface — types, operators, control flow. `ir` is compilation and
IR utilities. Every example below assumes these two imports.

## Quickstart: vector add at tensor level

The smallest complete kernel. It names two input tensors, adds them, and returns the
result; where the data lives and how it moves is the compiler's problem.

```python
import pypto.language as pl

@pl.function
def vector_add(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
    return result

print(vector_add.as_python())
```

| Line | What it does |
| ---- | ------------ |
| `@pl.function` | Parses the body into PyPTO IR. `vector_add` is now an `ir.Function` |
| `a: pl.Tensor[[64], pl.FP32]` | Input: 1-D tensor, 64 elements, 32-bit float. `In` is the default direction |
| `result: pl.Tensor[...] = pl.add(a, b)` | Element-wise add. The annotation on the assignment target is required — it types the IR binding |
| `return result` | The function's return value, and the source of its return type |

Expected output — note that `pl.add` has resolved to its tensor-namespace form:

```python
@pl.function
def vector_add(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.tensor.add(a, b)
    return result
```

`as_python()` re-prints the IR as DSL source. It is the fastest way to check what the
parser actually built, and it is worth reading whenever a kernel behaves unexpectedly —
what you get back is the program the compiler sees, not the one you think you wrote.

## Mechanics

### Tile level: load, compute, store

At tile level you allocate on-chip buffers explicitly and move data in and out yourself.

```python
@pl.function
def vector_add_tile(
    a: pl.Tensor[[64], pl.FP32],
    b: pl.Tensor[[64], pl.FP32],
    output: pl.Out[pl.Tensor[[64], pl.FP32]],
) -> pl.Tensor[[64], pl.FP32]:
    # DDR -> on-chip
    a_tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
    b_tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])

    # compute on-chip
    result: pl.Tile[[64], pl.FP32] = pl.add(a_tile, b_tile)

    # on-chip -> DDR
    out: pl.Tensor[[64], pl.FP32] = pl.store(result, [0], output)
    return out
```

| Concept | Tensor level | Tile level |
| ------- | ------------ | ---------- |
| Where data lives | DDR; compiler places it | You name the on-chip buffer |
| Type | `pl.Tensor` | `pl.Tile` |
| Data movement | Compiler inserts it | Explicit `pl.load` / `pl.store` |
| Result delivery | Return value | `pl.Out[...]` parameter, written through `pl.store` |

- **`pl.load(tensor, offsets, shapes)`** copies a region out of a DDR tensor into a new
  on-chip tile. `offsets` is where the region starts, `shapes` is how big it is — both
  per dimension.
- **`pl.store(tile, offsets, output_tensor)`** copies a tile back into a DDR tensor at
  `offsets`, and returns the written tensor.

The `pl.Out[...]` wrapper is a **direction**, and it is load-bearing rather than
decorative: it tells the compiler this parameter is written, not read, which decides
whether the runtime uploads the buffer before the call and downloads it after. Every
tensor parameter carries a direction — `In` by default, or an explicit `pl.Out[...]` /
`pl.InOut[...]`.

### Loops and loop-carried values

`pl.range()` builds a loop in the IR. With `init_values`, the loop carries values from
one iteration to the next — the PyPTO spelling of an accumulator.

```python
@pl.function
def sum_elements(a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[1], pl.FP32]:
    zero: pl.Tensor[[1], pl.FP32] = pl.create_tensor([1], dtype=pl.FP32)

    for i, (acc,) in pl.range(64, init_values=(zero,)):
        elem: pl.Tensor[[1], pl.FP32] = pl.slice(a, [1], [i])
        new_acc: pl.Tensor[[1], pl.FP32] = pl.add(acc, elem)
        acc_out: pl.Tensor[[1], pl.FP32] = pl.yield_(new_acc)

    return acc_out
```

1. `init_values=(zero,)` — the carried value going into iteration 0.
2. `for i, (acc,)` — `i` is the loop index; `acc` is the carried value for this iteration.
3. `pl.yield_(new_acc)` — hands `new_acc` to the next iteration as `acc`.
4. `acc_out` — after the loop, holds the value yielded by the last iteration.

`pl.yield_` is what makes this SSA-clean: `acc` is never mutated, each iteration binds a
new value and yields it. Reading `acc_out` outside the loop is how the final value
escapes.

Loops without carried values need neither `init_values` nor `pl.yield_`:

```python
for i in pl.range(10):        # 0 .. 9
    ...

for i in pl.range(0, 100, 2): # 0, 2, 4, ... 98
    ...
```

### Multi-function programs

`@pl.program` groups functions that call each other into one compilation unit.

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
        out: pl.Tensor[[128, 128], pl.FP32] = pl.store(result, [0, 0], output)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        c: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        c = self.kernel_add(a, b, c)
        return c
```

| Element | Meaning |
| ------- | ------- |
| `@pl.program` | Decorates a class into an `ir.Program` |
| `self` | Required first parameter of every method; stripped from the IR |
| `self.kernel_add(...)` | A cross-function call inside the program |
| `type=pl.FunctionType.InCore` | Compute kernel; runs on an AICore |
| `type=pl.FunctionType.Orchestration` | Host-side coordinator; creates tensors and dispatches kernels |

This is the **control plane / execution plane** split, and it is the single most
important structural idea in PyPTO:

```text
main            (Orchestration)  — host: allocates c, dispatches kernel_add
  └── kernel_add (InCore)        — device: loads tiles, computes, stores
```

The orchestration function never touches tile memory; the InCore function never
allocates tensors or dispatches work. Mixing the two is the most common structural
mistake in a first program.

`FunctionType` values you will meet early: `Opaque` (the default — no specific execution
context), `InCore`, `Orchestration`, and `Inline`. The rest (`AIC`, `AIV`, `Group`,
`Spmd`) are produced by the compiler or belong to later chapters.

### Compiling

```python
from pypto import ir
from pypto.backend import BackendType

compiled = ir.compile(
    VectorAddProgram,
    strategy=ir.OptimizationStrategy.Default,
    backend_type=BackendType.Ascend910B,
    skip_ptoas=True,   # drop this once ptoas is on the machine
)
print(f"Generated code in: {compiled.output_dir}")
```

`skip_ptoas=True` stops after emitting `.pto` (MLIR), which is what makes this example
runnable on a plain `pip install`. Drop it to get compiled C++ kernel wrappers — that
step invokes **ptoas**, which is distributed separately from the Python package.

`ir.compile()` returns a **`CompiledProgram`**, not a path — the directory is
`compiled.output_dir`. The `CompiledProgram` is also callable, which is how you dispatch
it to a device once you have a worker.

The most useful parameters when starting out (`ir.compile` takes 15 in total):

| Parameter | Default | What it does |
| --------- | ------- | ------------ |
| `program` | (required) | The `ir.Program` to compile |
| `output_dir` | `None` → `<base>/<name>_<timestamp>` | Where codegen, reports, and pass dumps land. `<base>` is `$PYPTO_PROG_BUILD_DIR`, or `build_output` when unset |
| `strategy` | `OptimizationStrategy.Default` | Pass pipeline preset. `DebugTileOptimization` exists but is a debugging shortcut — prefer `Default` |
| `dump_passes` | `True` | `bool`, or a `PassDumpLevel` (`NONE` / `CONCISE` / `EXPLICIT`) for finer control. Writes IR snapshots under `output_dir/passes_dump/` |
| `backend_type` | `BackendType.Ascend910B` | Target architecture — `Ascend910B` or `Ascend950` |
| `skip_ptoas` | `False` | Stop after emitting `.pto` (MLIR) instead of invoking ptoas. Useful when the ptoas toolchain is unavailable |

The remaining nine parameters (`verification_level`, `diagnostic_phase`,
`disabled_diagnostics`, `memory_planner`, `enable_pypto_l0c_double_buffer`, `profiling`,
`platform`, `distributed_config`, `analyze_auto_scopes_for_deps`) control verification,
diagnostics, memory planning, and distributed compilation; they are covered where those
topics are.

What lands in `output_dir`:

```text
kernels/       generated device kernels, one per InCore function
orchestration/ generated host-side C++
report/        compile-time reports, including perf hints
debug/         a runnable `run.py` harness
passes_dump/   per-pass IR snapshots (only when dump_passes is on)
```

### Inspecting IR without compiling

```python
print(vector_add.as_python())                 # one function
print(VectorAddProgram.as_python())           # a whole program
print(vector_add.as_python(concise=True))     # drop intermediate type annotations
```

## Edge Cases

> **Fatal pitfall:** `@pl.function` **parses** the body — it does not run it. Calling
> `vector_add(a, b)` from ordinary Python does not compute anything, and a `print()` or
> `assert` inside the body never executes at runtime. Debug by reading `as_python()`, not
> by adding print statements.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`AttributeError` on the decorated function** | Treating `ir.Function` as a Python callable | Compile the program and dispatch the `CompiledProgram`, or use `@pl.jit` |
| **Parse error pointing at an assignment** | Assignment target has no type annotation | Annotate every binding: `x: pl.Tile[[64], pl.FP32] = ...` |
| **Output tensor comes back unchanged** | Result written to a parameter not declared `pl.Out[...]` | Wrap the parameter direction, and write through `pl.store` |
| **`compiled.output_dir` is `None` / path errors** | Reading the return of `ir.compile` as a string | `ir.compile` returns a `CompiledProgram`; read `.output_dir` |
| **ptoas failures on a machine without the toolchain** | Codegen ran the assembler | Pass `skip_ptoas=True` to stop at `.pto` |

`PYPTO_PROG_BUILD_DIR` is a **runtime environment variable** — `PYPTO_PROG_BUILD_DIR=/tmp/out python kernel.py`
relocates every compile output. Distinguish it from `SIMPLER_HOST_STRACE` and
`SIMPLER_DFX`, which are **compile-time macros** of the runtime (`-DXXX=1` at build
time); setting those in the shell has no effect.

## See Also

- [Installation](01-installation.md) — getting to the point where these examples import.
- [Language Guide](01-language_guide.md) — the full type system, control flow, memory model, and scopes.
- [Operation Reference](02-operation_reference.md) — the operator surface across `pl.*`, `pl.tensor.*`, and `pl.tile.*`.
- [Running on Device](00-getting_started.md) — resident device tensors, explicit dispatch, benchmarking, and distributed execution.
- [Python IR Syntax Specification](../dev/language/00-python_syntax.md) — the exact syntax the parser accepts.
